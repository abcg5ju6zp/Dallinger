"""Isolated lifecycle management for experiment replays.

Every replay -- a Notebook ``restore_state_from_replay`` session or a
server-side :class:`~dallinger.experiment_server.replay.ReplayBackend` run --
is represented by a :class:`ReplayGeneration`. A generation owns every
resource that used to leak across replays:

* the temporary ``*-import-*`` database, its engine and import session;
* the process-wide ``dallinger_experiment`` module alias;
* the in-process configuration override;
* the cursor of events that have already been confirmed.

Resources are acquired in a fixed order -- validate the export package,
register the database and import session, apply configuration overrides and
only then publish ``replay_start`` -- and are released by
:meth:`ReplayGeneration.converge`. ``converge`` is idempotent and
best-effort, so a failed import, a failing event hook, a Notebook cell being
re-executed or a process restart all end at the same clean state. Teardown
never replays events; confirmed events stay confirmed.
"""

import datetime
import hashlib
import logging
import os
import sys
import threading
import uuid
from collections import namedtuple
from zipfile import BadZipFile, ZipFile

from sqlalchemy import create_engine, func, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import scoped_session, sessionmaker

from dallinger import db
from dallinger.config import get_config
from dallinger.data import find_experiment_export, ingest_zip, table_names
from dallinger.db import db_url, init_db
from dallinger.models import Info

logger = logging.getLogger(__name__)

#: Sentinel meaning ``sys.modules`` had no ``dallinger_experiment`` entry.
_MISSING = object()

#: PostgreSQL identifier limit (``NAMEDATALEN - 1``).
_MAX_DB_NAME_LENGTH = 63

#: Process-wide registry of live generations, keyed by generation id.
_registry = {}

#: Guards the registry and the handover of the global module alias.
_registry_lock = threading.RLock()


#: Position of a single event inside a replay, used for the resumable cursor
#: and for failure diagnostics.
ReplayPosition = namedtuple(
    "ReplayPosition", ["index", "event_id", "event_type", "creation_time"]
)


def _position(event, index):
    """Build a :class:`ReplayPosition` from an arbitrary event-like object."""
    if event is None:
        return ReplayPosition(index, None, None, None)
    return ReplayPosition(
        index,
        getattr(event, "id", None),
        getattr(event, "type", None),
        getattr(event, "creation_time", None),
    )


class ReplayError(Exception):
    """Raised when a replay generation cannot be opened at all.

    Event and user hook failures are not wrapped: the original exception is
    re-raised after the main session has been rolled back, while the
    diagnostic position is recorded on the generation.
    """

    def __init__(self, generation, message, stage=None, position=None):
        self.generation = generation
        self.stage = stage
        self.position = position
        super().__init__(message)


def validate_export_package(zip_path):
    """Validate that ``zip_path`` is a readable Dallinger export archive.

    The archive must contain a ``.csv`` member for every table
    :func:`~dallinger.data.ingest_zip` imports. This runs before any database
    or global state is touched, so a corrupt or incomplete package cannot
    leak unpublished resources.
    """
    try:
        archive = ZipFile(zip_path)
    except (OSError, BadZipFile) as exc:
        raise ReplayError(
            None,
            'Experiment export "{}" is not a readable zip archive: {}'.format(
                zip_path, exc
            ),
            stage="validate",
        ) from exc
    with archive:
        names = archive.namelist()
        missing = [
            name
            for name in table_names
            if not any(name in member for member in names)
        ]
    if missing:
        raise ReplayError(
            None,
            'Experiment export "{}" is missing data for table(s): {}'.format(
                zip_path, ", ".join(sorted(missing))
            ),
            stage="validate",
        )


def active_generations():
    """Return all currently registered replay generations."""
    with _registry_lock:
        return tuple(_registry.values())


def get_generation(generation_id):
    """Look up a live generation by id."""
    return _registry.get(generation_id)


def converge_generations(app_id=None):
    """Converge (tear down) live generations.

    With no ``app_id`` every live generation is converged. Otherwise only
    generations for that ``app_id`` are. Used when a replay is restarted and
    only teardown -- never replaying confirmed events -- should continue.
    """
    with _registry_lock:
        generations = tuple(
            gen
            for gen in _registry.values()
            if app_id is None or gen.app_id == app_id
        )
    for generation in generations:
        generation.converge()


class ReplayGeneration:
    """A single, isolated replay generation.

    Resource-owning generations (Notebook restores) are used as context
    managers. Lightweight generations (server-side :class:`ReplayBackend`
    runs) only track progress and diagnostics; they create no databases and
    never touch global state.
    """

    def __init__(
        self,
        experiment,
        app_id=None,
        *,
        zip_path=None,
        configuration_options=None,
        import_resources=True,
    ):
        self.experiment = experiment
        self.app_id = app_id
        self.generation_id = uuid.uuid4().hex[:12]
        self.zip_path = zip_path
        self.configuration_options = dict(configuration_options or {})
        self.import_resources = import_resources

        # Lifecycle flags.
        self.validated = False
        self.registered = False
        self.published = False
        self.finished = False
        self.closed = False
        self.failed = False

        # Diagnostics.
        self.failure_stage = None
        self.failure_position = None
        self.failure_error = None

        # Resumable progress.
        self.confirmed_count = 0
        self.last_position = None

        # Owned resources.
        self.config = None
        self._override = None
        self._override_active = False
        self.database_name = None
        self.import_engine = None
        self.import_session = None
        self._previous_module = _MISSING
        self._module_owned = False
        self._our_module = None

    # -- construction helpers ------------------------------------------------

    @classmethod
    def lightweight(cls, experiment):
        """Create and register a generation that owns no resources.

        Used by the server-side replay backend, which replays against the
        already restored main database.
        """
        generation = cls(
            experiment,
            getattr(experiment, "app_id", None),
            import_resources=False,
        )
        with _registry_lock:
            _registry[generation.generation_id] = generation
        generation.registered = True
        return generation

    # -- context manager protocol --------------------------------------------

    def __enter__(self):
        if not self.import_resources:
            raise ReplayError(
                self,
                "Lightweight replay generations cannot manage import resources.",
                stage="register",
            )
        try:
            self.validate()
            self.register()
            self.publish()
        except BaseException:
            # Import/registration/publish failed: delete unpublished
            # resources and leave global state as we found it.
            self.converge()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closed:
            # This generation was converged early (e.g. superseded by a
            # Notebook re-run); it must not finish or touch a successor's
            # session.
            return False
        if exc_val is not None and not self.failed:
            # The replay body (an event hook or user code) raised without
            # recording the failure itself.
            self.record_failure("replay", exc=exc_val)
        if exc_val is None and self.published and not self.failed:
            # replay_finish is committed only when every event and hook
            # succeeded. A failing finish hook is reported to the caller
            # after resources have converged.
            try:
                self.experiment.replay_finish()
            except Exception as exc:
                self.record_failure("finish", exc=exc)
                self.converge()
                raise
            self.finished = True
        self.converge()
        # Never suppress the original exception.
        return False

    # -- lifecycle stages -----------------------------------------------------

    def validate(self):
        """Locate and validate the export package before resources exist."""
        if self.validated:
            return
        if self.zip_path is None:
            self.zip_path = find_experiment_export(self.app_id)
        if self.zip_path is None:
            msg = 'Dataset export for app id "{}" could not be found.'
            raise IOError(msg.format(self.app_id))
        validate_export_package(self.zip_path)
        self.validated = True

    def register(self):
        """Register the temp database, import session and global state."""
        if self.registered:
            return
        with _registry_lock:
            _registry[self.generation_id] = self
            self.registered = True

            # Notebook cell re-runs and overlapping replays converge the
            # older generation for the same app_id before taking over.
            for previous in tuple(_registry.values()):
                if (
                    previous is not self
                    and previous.import_resources
                    and not previous.closed
                    and previous.app_id == self.app_id
                ):
                    logger.warning(
                        "Replacing active replay generation %s for app %s "
                        "with new generation %s",
                        previous.generation_id,
                        self.app_id,
                        self.generation_id,
                    )
                    previous.converge()

            # Fake ``dallinger_experiment`` to point at the current
            # experiment. init_db() imports that module to register model
            # metadata, so the alias must be in place first.
            self._our_module = sys.modules[type(self.experiment).__module__]
            self._previous_module = sys.modules.get(
                "dallinger_experiment", _MISSING
            )
            if self._previous_module is self._our_module:
                # Already pointing at our module (e.g. inside a running
                # experiment server); we do not own the alias.
                self._module_owned = False
            else:
                if self._previous_module is not _MISSING:
                    logger.warning(
                        "dallinger_experiment is already set, updating"
                    )
                sys.modules["dallinger_experiment"] = self._our_module
                self._module_owned = True

        # Load the configuration and publish the generation's experiment
        # state.
        self.config = get_config(load=True)
        self.experiment.app_id = self.app_id
        self.experiment.original_app_id = self.app_id
        self.experiment.exp_config = self.config
        # The replay index is initialised to 1970 as that is guaranteed to be
        # before any experiment Info object.
        self.experiment._replay_time_index = datetime.datetime(
            1970, 1, 1, 1, 1, 1
        )
        self.experiment._replay_generation = self

        # Create the isolated temporary database and import session.
        base_url = make_url(db_url)
        self.database_name = self._database_name(
            base_url.database, self.app_id, self.generation_id
        )
        self._create_database(self.database_name)
        self.import_engine = create_engine(
            str(base_url.set(database=self.database_name))
        )
        # Clear the temporary storage and import it.
        init_db(drop_all=True, bind=self.import_engine)
        self.import_session = scoped_session(
            sessionmaker(
                autocommit=False, autoflush=True, bind=self.import_engine
            )
        )
        self.experiment.import_session = self.import_session

        print(
            "Ingesting dataset from {}...".format(
                os.path.basename(self.zip_path)
            )
        )
        ingest_zip(self.zip_path, engine=self.import_engine)
        self.experiment._replay_range = tuple(
            self.import_session.query(
                func.min(Info.creation_time),
                func.max(Info.creation_time),
            )
        )[0]

    def publish(self):
        """Apply config overrides and publish ``replay_start``."""
        if self.published:
            return
        self._override = self.config.override(
            self.configuration_options, strict=True
        )
        self._override.__enter__()
        self._override_active = True
        self.experiment.replay_start()
        self.published = True

    # -- progress and failure handling ---------------------------------------

    def confirm(self, event):
        """Mark an event (and the hooks around it) as successfully replayed."""
        index = self.confirmed_count
        self.confirmed_count += 1
        self.last_position = _position(event, index)

    def record_failure(self, stage, event=None, index=None, exc=None):
        """Record a failure, preserve its position and roll back the main DB.

        The main session is rolled back so partially replayed state never
        leaks into the next replay. The diagnostic position (event index, id,
        type and creation time) is kept on the generation for inspection.
        """
        if index is None:
            index = self.confirmed_count
        self.failed = True
        self.failure_stage = stage
        self.failure_position = _position(event, index)
        self.failure_error = exc
        logger.error(
            "Replay generation %s failed during %r at position %s; rolling "
            "back the main session.",
            self.generation_id,
            stage,
            self.failure_position,
            exc_info=exc,
        )
        try:
            db.session.rollback()
        except Exception:
            logger.exception(
                "Could not roll back the main session after replay failure"
            )

    @property
    def progress(self):
        """A snapshot of resumable progress and failure diagnostics."""
        return {
            "generation_id": self.generation_id,
            "app_id": self.app_id,
            "published": self.published,
            "finished": self.finished,
            "failed": self.failed,
            "failure_stage": self.failure_stage,
            "failure_position": self.failure_position,
            "confirmed_count": self.confirmed_count,
            "last_position": self.last_position,
        }

    # -- teardown --------------------------------------------------------------

    def converge(self):
        """Idempotently release every resource owned by this generation.

        Safe to call more than once and safe to call after an import, event
        or hook failure. Sessions are closed, global module/config state is
        restored and temporary databases are dropped; no events are replayed
        and ``replay_finish`` is never called from here.
        """
        if self.closed:
            return
        self.closed = True
        with _registry_lock:
            _registry.pop(self.generation_id, None)

        if not self.import_resources:
            # Backend runs replay against the live main database, which is
            # managed elsewhere; they own nothing to close here.
            return

        # 1. Leave the configuration override.
        if self._override_active:
            try:
                self._override.__exit__(None, None, None)
            except Exception:
                logger.exception(
                    "Replay generation %s: could not restore configuration",
                    self.generation_id,
                )
            self._override_active = False

        # 2. Roll back and close the import session.
        if self.import_session is not None:
            try:
                self.import_session.rollback()
                self.import_session.close()
            except Exception:
                logger.exception(
                    "Replay generation %s: could not close import session",
                    self.generation_id,
                )
            self.import_session = None

        # 3. Dispose of the import engine so its connections do not block
        #    DROP DATABASE.
        if self.import_engine is not None:
            try:
                self.import_engine.dispose()
            except Exception:
                logger.exception(
                    "Replay generation %s: could not dispose import engine",
                    self.generation_id,
                )
            self.import_engine = None

        # 4. Roll back and close the main session.
        try:
            db.session.rollback()
            db.session.close()
        except Exception:
            logger.exception(
                "Replay generation %s: could not close the main session",
                self.generation_id,
            )

        # 5. Restore the global ``dallinger_experiment`` alias, but only if we
        #    are still its owner.
        if self._module_owned:
            current = sys.modules.get("dallinger_experiment", _MISSING)
            if current is self._our_module:
                if self._previous_module is _MISSING:
                    sys.modules.pop("dallinger_experiment", None)
                else:
                    sys.modules["dallinger_experiment"] = self._previous_module
            self._module_owned = False

        # 6. Drop the temporary database.
        if self.database_name is not None:
            self._drop_database(self.database_name)
            self.database_name = None

        # 7. Reset configuration, as the historical teardown did.
        if self.config is not None:
            try:
                self.config._reset(register_defaults=True)
            except Exception:
                logger.exception(
                    "Replay generation %s: could not reset configuration",
                    self.generation_id,
                )

        # 8. Detach from the experiment.
        if getattr(self.experiment, "_replay_generation", None) is self:
            try:
                del self.experiment._replay_generation
            except AttributeError:
                pass

    # -- database helpers ------------------------------------------------------

    @staticmethod
    def _admin_engine():
        """An AUTOCOMMIT engine on the server, for CREATE/DROP DATABASE."""
        return create_engine(db_url, isolation_level="AUTOCOMMIT")

    def _create_database(self, name):
        engine = self._admin_engine()
        try:
            with engine.connect() as conn:
                exists = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": name},
                ).scalar()
                if not exists:
                    quoted = conn.dialect.identifier_preparer.quote(name)
                    conn.execute(text("CREATE DATABASE {}".format(quoted)))
        finally:
            engine.dispose()

    def _drop_database(self, name):
        engine = self._admin_engine()
        try:
            with engine.connect() as conn:
                quoted = conn.dialect.identifier_preparer.quote(name)
                # Terminate any lingering connections (e.g. an import session
                # that failed to close) before dropping.
                conn.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :name AND pid <> pg_backend_pid()"
                    ),
                    {"name": name},
                )
                conn.execute(
                    text("DROP DATABASE IF EXISTS {}".format(quoted))
                )
        except Exception:
            logger.exception(
                "Replay generation %s: failed to drop temporary database %s",
                self.generation_id,
                name,
            )
        finally:
            engine.dispose()

    @staticmethod
    def _database_name(base_name, app_id, generation_id):
        """Compose a unique, Postgres-safe temporary database name."""
        name = "{}-import-{}-{}".format(base_name, app_id, generation_id)
        if len(name) > _MAX_DB_NAME_LENGTH:
            # Keep the name unique and readable while respecting the 63 byte
            # identifier limit for long app ids.
            digest = hashlib.sha1(app_id.encode("utf-8")).hexdigest()[:16]
            name = "{}-import-{}-{}".format(base_name, digest, generation_id)
        return name

    def __repr__(self):
        return "<ReplayGeneration {} app_id={!r} failed={} finished={}>".format(
            self.generation_id, self.app_id, self.failed, self.finished
        )
