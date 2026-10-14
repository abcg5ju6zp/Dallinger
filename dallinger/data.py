"""Data-handling tools."""

import csv
import fcntl
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from zipfile import ZIP_DEFLATED, ZipFile

import boto3
import botocore
import psycopg2

from dallinger import db, models
from dallinger.heroku.tools import HerokuApp
from dallinger.postgres_copy import copy_from
from dallinger.utils import open_for_csv

from .config import get_config

logger = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    try:
        import tablib
    except ImportError:
        logger.debug("Failed to import tablib.")


class S3BucketUnavailable(Exception):
    """No Amazon S3 bucket could be found based on the user's
    configuration.
    """


table_names = [
    "info",
    "network",
    "node",
    "notification",
    "participant",
    "question",
    "transformation",
    "transmission",
    "vector",
]


def find_experiment_export(app_id):
    """Attempt to find a zipped export of an experiment with the ID provided
    and return its path. Returns None if not found.

    Search order:
        1. local "data" subdirectory
        2. user S3 bucket
        3. Dallinger S3 bucket
    """

    # Check locally first
    cwd = os.getcwd()
    data_filename = "{}-data.zip".format(app_id)
    path_to_data = os.path.join(cwd, "data", data_filename)
    if os.path.exists(path_to_data):
        try:
            Data(path_to_data)
        except IOError:
            from dallinger import logger

            logger.exception(
                "Error reading local data file {}, checking remote.".format(
                    path_to_data
                )
            )
        else:
            return path_to_data

    # Get remote file instead
    path_to_data = os.path.join(tempfile.mkdtemp(), data_filename)

    config = get_config(load=True)

    if not aws_access_keys_present(config):
        print("✘ AWS credentials not present, skipping download from S3.")
        return

    buckets = [user_s3_bucket(), dallinger_s3_bucket()]

    for bucket in buckets:
        if bucket is None:
            continue
        try:
            bucket.download_file(data_filename, path_to_data)
        except botocore.exceptions.ClientError:
            pass
        else:
            return path_to_data


def load(app_id):
    """Load the data from wherever it is found."""
    path_to_data = find_experiment_export(app_id)
    if path_to_data is None:
        raise IOError("Dataset {} could not be found.".format(app_id))

    return Data(path_to_data)


def dump_database(id):
    """Dump the database to a temporary directory."""

    tmp_dir = tempfile.mkdtemp()
    current_dir = os.getcwd()
    os.chdir(tmp_dir)

    FNULL = open(os.devnull, "w")
    heroku_app = HerokuApp(dallinger_uid=id, output=FNULL)
    heroku_app.backup_capture()
    heroku_app.backup_download()

    for filename in os.listdir(tmp_dir):
        if filename.startswith("latest.dump"):
            os.rename(filename, "database.dump")

    os.chdir(current_dir)

    return os.path.join(tmp_dir, "database.dump")


def backup(id):
    """Backup the database to S3."""
    filename = dump_database(id)
    key = "{}.dump".format(id)

    bucket = user_s3_bucket()
    bucket.upload_file(filename, key)
    return _generate_s3_url(bucket, key)


def registration_key(id):
    return "{}.reg".format(id)


def register(id, url=None, generation=None):
    """Register a UUID key in the global S3 bucket.

    When ``generation`` is supplied the write is fenced: the registration is
    only (re)written if no newer export generation currently owns the key.
    Late results from an older, superseded export attempt therefore can never
    replace a newer downloadable URL. The registered body stays a plain URL
    (or ``"missing"``) for backwards compatibility; the generation travels as
    object metadata.
    """
    bucket = registration_s3_bucket()
    if bucket is None:
        return
    key = registration_key(id)
    obj = bucket.Object(key)
    if generation is not None and not _is_newer_than_remote(obj, generation):
        # A newer generation already registered its URL; leave it in place.
        return None
    kwargs = {"Body": url or "missing"}
    if generation is not None:
        kwargs["Metadata"] = {"generation": generation}
    obj.put(**kwargs)
    return _generate_s3_url(bucket, key)


def _remote_generation(obj):
    """Best-effort read of the generation recorded on an existing S3 object.

    Returns ``None`` when the object does not exist, its generation metadata
    is missing, or it cannot be read (e.g. a write-only bucket).
    """
    try:
        obj.load()
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response["Error"].get("Code", 0))
        if error_code in (404, 403):
            return None
        raise
    metadata = getattr(obj, "metadata", None) or {}
    return metadata.get("generation")


def _is_newer_than_remote(obj, generation):
    """True when ``generation`` is allowed to replace the remote object."""
    remote = _remote_generation(obj)
    return remote is None or _is_newer_generation(generation, remote)


def is_registered(id):
    """Check if a UUID is already registered"""
    bucket = registration_s3_bucket()
    if bucket is None:
        return False
    key = registration_key(id)
    found_keys = set(obj.key for obj in bucket.objects.filter(Prefix=key))
    return key in found_keys


def copy_heroku_to_local(id):
    """Copy a Heroku database locally."""
    heroku_app = HerokuApp(dallinger_uid=id)
    try:
        subprocess.call(["dropdb", heroku_app.name])
    except Exception:
        pass

    heroku_app.pg_pull()


def copy_db_to_csv(dsn, path, scrub_pii=False):
    """Copy a database to a set of CSV files.

    Every table is copied through a single read-only, deferrable
    ``SERIALIZABLE`` transaction, so all CSV files reflect the same database
    point in time even when participants are still writing. The pinned
    snapshot metadata (PostgreSQL snapshot id and transaction id) is returned
    so an export generation can record exactly what it contains.

    On failure the transaction is rolled back, the connection is closed and
    the original exception propagates; callers are responsible for discarding
    any partially written directory.
    """
    if path:
        os.makedirs(path, exist_ok=True)
    if "postgresql://" in dsn or "postgres://" in dsn:
        conn = psycopg2.connect(dsn=dsn)
    else:
        conn = psycopg2.connect(database=dsn, user="dallinger")
    # Pin one read point for the whole export: all COPY statements below run
    # against the same MVCC snapshot.
    conn.set_session(
        readonly=True,
        deferrable=True,
        isolation_level=psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE,
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_export_snapshot(), txid_current()")
        pg_snapshot, txid = cur.fetchone()
        for table in table_names:
            csv_path = os.path.join(path, "{}.csv".format(table))
            with open(csv_path, "w") as f:
                sql = "COPY {} TO STDOUT WITH CSV HEADER".format(table)
                cur.copy_expert(sql, f)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    if scrub_pii:
        _scrub_participant_table(path)
    return {"pg_snapshot": pg_snapshot, "txid": txid}


# Backwards compatibility for imports
copy_local_to_csv = copy_db_to_csv


def _scrub_participant_table(path_to_data):
    """Scrub PII from the given participant table."""
    path = os.path.join(path_to_data, "participant.csv")
    with open_for_csv(path, "r") as input, open("{}.0".format(path), "w") as output:
        reader = csv.reader(input)
        writer = csv.writer(output)
        headers = next(reader)
        writer.writerow(headers)
        for i, row in enumerate(reader):
            row[headers.index("worker_id")] = row[headers.index("id")]
            row[headers.index("unique_id")] = "{}:{}".format(
                row[headers.index("id")], row[headers.index("assignment_id")]
            )
            if "client_ip_address" in headers:
                row[headers.index("client_ip_address")] = ""
            writer.writerow(row)

        os.rename("{}.0".format(path), path)


def export(id, local=False, scrub_pii=False):
    """Export data from an experiment.

    Every invocation is a single, isolated *export generation*: one pinned
    database read point, one experiment identity file, one staged code
    snapshot and one anonymization result are validated together and only
    published once the final archive has been built and verified. See
    :class:`_ExportGeneration` for the failure, retry and recovery rules.
    """

    print("Preparing to export the data...")

    if local:
        db_uri = db.db_url
    else:
        db_uri = HerokuApp(id).db_uri
    return export_db_uri(id, db_uri=db_uri, local=local, scrub_pii=scrub_pii)


# --- export generations -----------------------------------------------------
#
# An export generation owns everything produced by one export attempt for one
# experiment, kept under ``data/.exports/<experiment>/<generation>/``:
#
#     stage/data/<table>.csv   table copies from one pinned DB snapshot
#     stage/<id>-code.zip      the code snapshot that was available at export
#     stage/experiment_id.md   the experiment identity, written once
#     package/<id>-data.zip    verified archive before it is published
#     state.json               stage journal
#     FAILED                   marker + traceback for diagnosable failures
#
# The publishable artifact (``data/<id>-data.zip``), the S3 object and the
# registered URL are only created/updated after all earlier stages validate.

DATA_DIR = "data"
EXPORTS_DIRNAME = ".exports"
STATE_FILENAME = "state.json"
FAILED_FILENAME = "FAILED"
CURRENT_FILENAME = "current.json"
LOCK_FILENAME = "lock"
STAGE_DIRNAME = "stage"
PACKAGE_DIRNAME = "package"
CSV_DIRNAME = "data"

SNAPSHOT_STAGE = "snapshot"
ANONYMIZE_STAGE = "anonymize"
CODE_STAGE = "code"
IDENTITY_STAGE = "identity"
PACKAGE_STAGE = "package"
PUBLISH_STAGE = "publish"
UPLOAD_STAGE = "upload"
REGISTER_STAGE = "register"

STATUS_IN_PROGRESS = "in_progress"
STATUS_FAILED = "failed"
STATUS_SUPERSEDED = "superseded"
STATUS_COMPLETE = "complete"

#: Attempts made to register an already-uploaded object. Registration points
#: at the same object on every try; the archive is never re-exported.
REGISTER_ATTEMPTS = 5
REGISTER_RETRY_DELAY = 0.5


#: Stages whose failure is caused by the outside world (disk publish, object
#: storage, URL registration). Such failures keep the generation resumable so a
#: later attempt continues with the same, already validated artifacts instead
#: of producing a half-finished or mixed archive.
RETRYABLE_STAGES = frozenset((PUBLISH_STAGE, UPLOAD_STAGE, REGISTER_STAGE))


class ExportError(Exception):
    """Base class for export pipeline errors."""


class ExportStageError(ExportError):
    """A stage of the export pipeline failed.

    ``generation`` identifies the attempt, ``stage`` the failed stage. When
    ``retain`` is true the isolated staging directory is preserved for
    diagnosis; otherwise (e.g. a failed table copy) it was discarded and no
    publishable file remains.
    """

    def __init__(self, message, generation=None, stage=None, retain=True,
                 retryable=False):
        super().__init__(message)
        self.generation = generation
        self.stage = stage
        self.retain = retain
        self.retryable = retryable


class ExportSuperseded(ExportError):
    """The attempt was overtaken by a newer export generation.

    The attempt's archive was neither published nor registered, so a late
    result from an old attempt can never replace a newer export.
    """


def new_generation_id():
    """A fixed-width, chronologically sortable generation identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    return "{}-{}".format(timestamp, uuid.uuid4().hex[:8])


def _is_newer_generation(candidate, existing):
    """Generation ids embed a fixed-width UTC timestamp, so lexical order is
    chronological order."""
    return candidate > existing


def _exports_root(experiment_id):
    return os.path.join(DATA_DIR, EXPORTS_DIRNAME, experiment_id)


def _generation_dir(experiment_id, generation):
    return os.path.join(_exports_root(experiment_id), generation)


def _lock_path(experiment_id):
    return os.path.join(_exports_root(experiment_id), LOCK_FILENAME)


def _current_path(experiment_id):
    return os.path.join(_exports_root(experiment_id), CURRENT_FILENAME)


def _archive_filename(experiment_id):
    return "{}-data.zip".format(experiment_id)


def _canonical_archive_path(experiment_id):
    return os.path.join(DATA_DIR, _archive_filename(experiment_id))


@contextmanager
def _export_lock(experiment_id):
    """Serialize all export attempts for one experiment (cross-process).

    The lock is released automatically by the OS if a process dies while
    holding it, so a restarted exporter never blocks forever.
    """
    os.makedirs(_exports_root(experiment_id), exist_ok=True)
    lock_file = open(_lock_path(experiment_id), "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    tmp = "{}.tmp-{}".format(path, uuid.uuid4().hex[:8])
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_current(experiment_id):
    path = _current_path(experiment_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _remove_orphan_temp_files(experiment_id):
    """Remove temp archives left behind by a crashed attempt."""
    tmp_prefix = ".{}.tmp-".format(_archive_filename(experiment_id))
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            if name.startswith(tmp_prefix):
                with suppress(OSError):
                    os.remove(os.path.join(DATA_DIR, name))


def _clean_package_temp_files(generation_root, archive_name):
    package_dir = os.path.join(generation_root, PACKAGE_DIRNAME)
    if not os.path.isdir(package_dir):
        return
    tmp_prefix = ".{}.tmp-".format(archive_name)
    for name in os.listdir(package_dir):
        if name.startswith(tmp_prefix):
            with suppress(OSError):
                os.remove(os.path.join(package_dir, name))


def _newer_generation_states(experiment_id, generation):
    """State summaries of every attempt newer than ``generation``."""
    newer = []
    root = _exports_root(experiment_id)
    if os.path.isdir(root):
        for name in os.listdir(root):
            state_file = os.path.join(root, name, STATE_FILENAME)
            if not os.path.isfile(state_file):
                continue
            try:
                with open(state_file) as f:
                    state = json.load(f)
            except (OSError, ValueError):
                continue
            other = state.get("generation")
            if other and _is_newer_generation(other, generation):
                newer.append(state)
    current = _read_current(experiment_id)
    if current and _is_newer_generation(current.get("generation", ""), generation):
        newer.append(current)
    return newer


class _ExportGeneration:
    """One isolated, journaled export attempt.

    The stage journal makes the attempt restart-safe: stages are executed in
    order, journaled only after their outputs validate, and resumed stages
    re-verify their inputs before use. Nothing is published or registered
    until the package stage has produced a verified archive.
    """

    def __init__(self, experiment_id, state):
        self.experiment_id = experiment_id
        self.state = state
        self.generation = state["generation"]
        self.local = state["local"]
        self.scrub_pii = state["scrub_pii"]
        self.root = _generation_dir(experiment_id, self.generation)
        self.stage_path = os.path.join(self.root, STAGE_DIRNAME)
        self.csv_path = os.path.join(self.stage_path, CSV_DIRNAME)
        self.package_path = os.path.join(self.root, PACKAGE_DIRNAME)
        self.code_name = "{}-code.zip".format(experiment_id)
        self.archive_name = _archive_filename(experiment_id)
        self.canonical_path = _canonical_archive_path(experiment_id)

    @classmethod
    def create(cls, experiment_id, local, scrub_pii):
        generation = new_generation_id()
        state = {
            "experiment_id": experiment_id,
            "generation": generation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "local": local,
            "scrub_pii": scrub_pii,
            "status": STATUS_IN_PROGRESS,
            "stages": {},
            "error": None,
        }
        instance = cls(experiment_id, state)
        os.makedirs(instance.csv_path, exist_ok=False)
        os.makedirs(instance.package_path, exist_ok=False)
        instance.save()
        return instance

    @classmethod
    def from_dir(cls, experiment_id, generation):
        with open(
            os.path.join(_generation_dir(experiment_id, generation), STATE_FILENAME)
        ) as f:
            return cls(experiment_id, json.load(f))

    def save(self):
        _write_json_atomic(os.path.join(self.root, STATE_FILENAME), self.state)

    @property
    def stages(self):
        return self.state["stages"]

    def _mark_stage(self, name, detail=None):
        self.stages[name] = detail or {"status": "complete"}
        self.state["error"] = None
        self.save()

    def _record_failure(self, stage, exc, terminal=True):
        """Journal the failure. Terminal failures drop a FAILED marker and
        keep the isolated directory for diagnosis; retryable failures stay
        ``in_progress`` so a restart or duplicate request can continue from
        the already validated stages.
        """
        self.state["error"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if terminal:
            self.state["status"] = STATUS_FAILED
        with suppress(OSError):
            self.save()
        if terminal:
            with suppress(OSError):
                with open(os.path.join(self.root, FAILED_FILENAME), "w") as f:
                    f.write(
                        "Export generation {} failed at stage {}.\n".format(
                            self.generation, stage
                        )
                    )
                    f.write("{}: {}\n".format(type(exc).__name__, exc))

    def _fail_if_superseded(self):
        for state in _newer_generation_states(self.experiment_id, self.generation):
            stages = state.get("stages") or {}
            if stages.get(PACKAGE_STAGE) or state.get("archive"):
                raise ExportSuperseded(
                    "Export generation {} was superseded by {}.".format(
                        self.generation, state.get("generation")
                    )
                )

    def _run_stage(self, name, func):
        if name in self.stages:
            # Already journaled: resumed attempt; stage helpers re-verify
            # their artifacts before relying on them.
            return self.stages[name]
        self._fail_if_superseded()
        try:
            detail = func()
        except ExportSuperseded:
            raise
        except Exception as exc:
            if name == SNAPSHOT_STAGE:
                # A broken table copy must not leave anything publishable.
                shutil.rmtree(self.root, ignore_errors=True)
                raise ExportStageError(
                    str(exc),
                    generation=self.generation,
                    stage=name,
                    retain=False,
                ) from exc
            retryable = name in RETRYABLE_STAGES
            # Retryable stages stay resumable; other stage failures are
            # terminal and must never result in a registered half-archive.
            self._record_failure(name, exc, terminal=not retryable)
            raise ExportStageError(
                str(exc),
                generation=self.generation,
                stage=name,
                retain=True,
                retryable=retryable,
            ) from exc
        self._mark_stage(name, detail)
        return detail

    # -- stages -------------------------------------------------------------

    def _stage_snapshot(self, db_uri):
        # scrubbing is deliberately a separate (later) stage
        info = copy_db_to_csv(db_uri, self.csv_path, scrub_pii=False)
        files = self._validate_csv_files()
        return {"snapshot": info, "files": files}

    def _verify_snapshot(self):
        detail = self.stages.get(SNAPSHOT_STAGE)
        if not detail:
            return False
        return self._checksums_match(detail.get("files", {}))

    def _validate_csv_files(self):
        files = {}
        for table in table_names:
            rel = os.path.join(CSV_DIRNAME, "{}.csv".format(table))
            path = os.path.join(self.stage_path, rel)
            if not os.path.isfile(path):
                raise ValueError("missing {} in export snapshot".format(rel))
            with open_for_csv(path, "r") as f:
                try:
                    header = next(csv.reader(f))
                except StopIteration:
                    header = None
            if not header:
                raise ValueError("{} has no CSV header".format(rel))
            files[rel.replace(os.sep, "/")] = _sha256(path)
        return files

    def _checksums_match(self, expected):
        for rel, checksum in expected.items():
            path = os.path.join(self.stage_path, *rel.split("/"))
            if not os.path.isfile(path) or _sha256(path) != checksum:
                return False
        return True

    def _stage_anonymize(self):
        _scrub_participant_table(self.csv_path)
        rel = os.path.join(CSV_DIRNAME, "participant.csv").replace(os.sep, "/")
        # The scrubbed file must still be a valid participant table.
        with open_for_csv(os.path.join(self.stage_path, *rel.split("/")), "r") as f:
            if not next(csv.reader(f), None):
                raise ValueError("participant.csv is unreadable after anonymization")
        return {"files": {rel: _sha256(os.path.join(self.stage_path, *rel.split("/")))}}

    def _stage_code(self):
        source = os.path.join("snapshots", self.code_name)
        if not os.path.exists(source):
            # No code snapshot was available for this experiment; this stays
            # compatible with archives that contain no code.
            return {"status": "unavailable"}
        with ZipFile(source) as zf:
            bad = zf.testzip()
        if bad is not None:
            raise ValueError("corrupt code snapshot entry: {}".format(bad))
        destination = os.path.join(self.stage_path, self.code_name)
        shutil.copyfile(source, destination)
        with ZipFile(destination) as zf:
            bad = zf.testzip()
        if bad is not None:
            raise ValueError("copied code snapshot is corrupt: {}".format(bad))
        return {
            "status": "complete",
            "file": self.code_name,
            "sha256": _sha256(destination),
        }

    def _verify_code(self):
        detail = self.stages.get(CODE_STAGE)
        if not detail or detail.get("status") != "complete":
            return True
        path = os.path.join(self.stage_path, detail["file"])
        return (
            os.path.isfile(path)
            and _sha256(path) == detail["sha256"]
            and ZipFile(path).testzip() is None
        )

    def _stage_identity(self):
        path = os.path.join(self.stage_path, "experiment_id.md")
        if os.path.exists(path):
            with open(path) as f:
                if f.read() != self.experiment_id:
                    raise ValueError("staged experiment id does not match generation")
            return {"status": "complete"}
        with open(path, "w") as f:
            f.write(self.experiment_id)
        return {"status": "complete"}

    def _stage_files(self):
        """All files that belong in the archive, relative to the stage dir."""
        files = []
        for table in table_names:
            files.append(os.path.join(CSV_DIRNAME, "{}.csv".format(table)))
        code = self.stages.get(CODE_STAGE)
        if code and code.get("status") == "complete":
            files.append(self.code_name)
        files.append("experiment_id.md")
        return files

    def _stage_package(self):
        expected = dict(self.stages[SNAPSHOT_STAGE]["files"])
        anonymized = self.stages.get(ANONYMIZE_STAGE)
        if anonymized:
            expected.update(anonymized.get("files", {}))
        code = self.stages.get(CODE_STAGE)
        if code and code.get("status") == "complete":
            expected[code["file"]] = code["sha256"]
        expected["experiment_id.md"] = _sha256(
            os.path.join(self.stage_path, "experiment_id.md")
        )
        if not self._checksums_match(expected):
            raise ValueError("staged files fail checksum validation")
        tmp = os.path.join(
            self.package_path,
            ".{}.tmp-{}".format(self.archive_name, uuid.uuid4().hex[:8]),
        )
        package = os.path.join(self.package_path, self.archive_name)
        print("Zipping up the package...")
        with ZipFile(tmp, "w", ZIP_DEFLATED, allowZip64=True) as zf:
            for rel in self._stage_files():
                zf.write(os.path.join(self.stage_path, rel), rel)
        with ZipFile(tmp) as zf:
            bad = zf.testzip()
        if bad is not None:
            with suppress(OSError):
                os.remove(tmp)
            raise ValueError("corrupt entry in export archive: {}".format(bad))
        os.replace(tmp, package)
        return {"file": self.archive_name, "sha256": _sha256(package)}

    def _verify_package(self):
        detail = self.stages.get(PACKAGE_STAGE)
        if not detail:
            return False
        package = os.path.join(self.package_path, detail["file"])
        return (
            os.path.isfile(package)
            and _sha256(package) == detail["sha256"]
            and ZipFile(package).testzip() is None
        )

    def _stage_publish(self):
        package = os.path.join(
            self.package_path, self.stages[PACKAGE_STAGE]["file"]
        )
        checksum = self.stages[PACKAGE_STAGE]["sha256"]
        tmp = os.path.join(
            DATA_DIR, ".{}.tmp-{}".format(self.archive_name, self.generation)
        )
        shutil.copyfile(package, tmp)
        if _sha256(tmp) != checksum:
            with suppress(OSError):
                os.remove(tmp)
            raise ValueError("published archive does not match verified package")
        os.replace(tmp, self.canonical_path)
        print("Done. Local export available in {}".format(self.canonical_path))
        return {"file": self.archive_name, "sha256": checksum}

    def _stage_upload(self):
        config = get_config(load=True)
        if not aws_access_keys_present(config):
            print("✘ AWS credentials not present, skipping export to S3.")
            return {"status": "skipped"}
        bucket = user_s3_bucket()
        key = self.archive_name
        obj = bucket.Object(key)
        remote = _remote_generation(obj)
        if remote and remote != self.generation and not _is_newer_generation(
            self.generation, remote
        ):
            raise ExportSuperseded(
                "S3 already holds a newer export generation: {}".format(remote)
            )
        if remote != self.generation:
            # publish object storage content only now, from the verified
            # canonical archive
            bucket.upload_file(
                self.canonical_path,
                key,
                ExtraArgs={"Metadata": {"generation": self.generation}},
            )
        registration_url = _generate_s3_url(bucket, key)
        s3_console_url = (
            "https://s3.console.aws.amazon.com/s3/object/{}"
            "?region={}&prefix={}".format(bucket.name, config.aws_region, key)
        )
        print(
            "A copy of your export was saved also to Amazon S3:\n"
            " - bucket name: {}\n"
            " - S3 console URL: {}".format(bucket.name, s3_console_url)
        )
        return {
            "status": "complete",
            "bucket": bucket.name,
            "key": key,
            "url": registration_url,
        }

    def _stage_register(self):
        if registration_s3_bucket() is None:
            return {"status": "skipped"}
        upload = self.stages.get(UPLOAD_STAGE) or {}
        url = upload.get("url")
        if not url:
            raise ValueError("cannot register without an uploaded export object")
        last_exc = None
        for attempt in range(REGISTER_ATTEMPTS):
            try:
                result = register(
                    self.experiment_id, url, generation=self.generation
                )
            except botocore.exceptions.ClientError as exc:
                # Object is already uploaded: retry the registration of the
                # same object instead of re-exporting.
                last_exc = exc
                time.sleep(REGISTER_RETRY_DELAY * (attempt + 1))
                continue
            if result is None:
                raise ExportSuperseded(
                    "a newer export generation already registered this experiment"
                )
            return {"status": "complete", "url": result}
        raise last_exc

    # -- driver -------------------------------------------------------------

    def run(self, db_uri):
        self._fail_if_superseded()
        try:
            self._run_stage(SNAPSHOT_STAGE, lambda: self._stage_snapshot(db_uri))
            if self.scrub_pii:
                self._run_stage(ANONYMIZE_STAGE, self._stage_anonymize)
            self._run_stage(CODE_STAGE, self._stage_code)
            self._run_stage(IDENTITY_STAGE, self._stage_identity)
            self._run_stage(PACKAGE_STAGE, self._stage_package)
            self._run_stage(PUBLISH_STAGE, self._stage_publish)
            if not self.local:
                upload = self._run_stage(UPLOAD_STAGE, self._stage_upload)
                if upload.get("status") == "skipped":
                    self._mark_stage(REGISTER_STAGE, {"status": "skipped"})
                else:
                    self._run_stage(REGISTER_STAGE, self._stage_register)
        except ExportSuperseded:
            # A newer generation has already produced an artifact: retire
            # this attempt quietly; nothing of it is published or registered.
            self.state["status"] = STATUS_SUPERSEDED
            with suppress(OSError):
                self.save()
            shutil.rmtree(self.root, ignore_errors=True)
            raise
        self.state["status"] = STATUS_COMPLETE
        self.save()
        summary = {
            "experiment_id": self.experiment_id,
            "generation": self.generation,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "local": self.local,
            "archive": self.archive_name,
            "sha256": self.stages[PUBLISH_STAGE]["sha256"],
            "stages": {
                name: (detail or {}).get("status", "complete")
                for name, detail in self.stages.items()
            },
        }
        previous = _read_current(self.experiment_id)
        if not (
            previous
            and _is_newer_generation(
                previous.get("generation", ""), self.generation
            )
        ):
            _write_json_atomic(_current_path(self.experiment_id), summary)
        shutil.rmtree(self.root, ignore_errors=True)
        return self.canonical_path

    def is_resumable(self):
        """True when the attempt died (or was restarted) after taking a
        complete, verifiable snapshot and can continue without re-exporting."""
        _clean_package_temp_files(self.root, self.archive_name)
        if not self._verify_snapshot():
            return False
        if self.scrub_pii and ANONYMIZE_STAGE in self.stages:
            anonymized = self.stages[ANONYMIZE_STAGE]
            if not self._checksums_match(anonymized.get("files", {})):
                return False
        if not self._verify_code():
            return False
        if PACKAGE_STAGE in self.stages and not self._verify_package():
            return False
        publish = self.stages.get(PUBLISH_STAGE)
        if publish and (
            not os.path.isfile(self.canonical_path)
            or _sha256(self.canonical_path) != publish["sha256"]
        ):
            return False
        return True


def _reconcile_generations(experiment_id):
    """Reconcile export state on entry and return a resumable generation.

    Orphaned temp directories and unverifiable interrupted attempts are
    removed. Failed generations stay on disk (with a FAILED marker) for
    diagnosis. The newest generation whose completed stages still verify is
    returned so the attempt resumes exactly where it left off.
    """
    os.makedirs(_exports_root(experiment_id), exist_ok=True)
    _remove_orphan_temp_files(experiment_id)
    resumable = []
    root = _exports_root(experiment_id)
    for name in os.listdir(root):
        if name in (LOCK_FILENAME, CURRENT_FILENAME):
            continue
        gdir = os.path.join(root, name)
        if not os.path.isdir(gdir):
            with suppress(OSError):
                os.remove(gdir)
            continue
        state_file = os.path.join(gdir, STATE_FILENAME)
        if not os.path.isfile(state_file):
            # No owner/journal: orphan from a crashed attempt.
            shutil.rmtree(gdir, ignore_errors=True)
            continue
        try:
            generation = _ExportGeneration.from_dir(experiment_id, name)
        except (OSError, ValueError, KeyError):
            shutil.rmtree(gdir, ignore_errors=True)
            continue
        if generation.state.get("status") == STATUS_COMPLETE:
            # Leftover directory from an attempt that finished but could not
            # clean up; its result is already recorded in current.json.
            shutil.rmtree(gdir, ignore_errors=True)
            continue
        if generation.state.get("status") == STATUS_SUPERSEDED:
            # Terminal attempt overtaken by a newer generation.
            shutil.rmtree(gdir, ignore_errors=True)
            continue
        if generation.state.get("status") == STATUS_FAILED:
            _clean_package_temp_files(gdir, generation.archive_name)
            continue
        if generation.is_resumable():
            resumable.append(generation)
        else:
            shutil.rmtree(gdir, ignore_errors=True)
    resumable.sort(key=lambda g: g.generation)
    return resumable[-1] if resumable else None


def export_db_uri(id, db_uri, local, scrub_pii):
    """Run an export generation against ``db_uri`` and return the path to the
    published local archive.

    A per-experiment lock serializes attempts; on entry, orphaned temporary
    directories are cleaned up and a still-complete interrupted generation is
    resumed instead of starting a new export.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with _export_lock(id):
        generation = _reconcile_generations(id)
        if generation is None:
            generation = _ExportGeneration.create(
                id, local=local, scrub_pii=scrub_pii
            )
        return generation.run(db_uri=db_uri)


def aws_access_keys_present(config):
    """Verifies that AWS access keys are set"""
    if (
        config.get("aws_access_key_id") == "YourAccessKeyId"
        or config.get("aws_secret_access_key") == "YourSecretAccessKey"
        or not config.get("aws_access_key_id")
        or not config.get("aws_secret_access_key")
    ):
        return False

    return True


def bootstrap_db_from_zip(zip_path, engine):
    """Given a path to a zip archive created with `export()`, first empty the
    database, then recreate it based on the data stored in the included .csv
    files.
    """
    db.init_db(drop_all=True, bind=engine)
    ingest_zip(zip_path, engine=engine)


def ingest_zip(path, engine=None):
    """Given a path to a zip file created with `export()`, recreate the
    database with the data stored in the included .csv files.
    """
    import_order = [
        "network",
        "participant",
        "node",
        "info",
        "notification",
        "question",
        "transformation",
        "vector",
        "transmission",
    ]
    with ZipFile(path, "r") as archive:
        filenames = archive.namelist()
        for name in import_order:
            filename = [f for f in filenames if name in f][0]
            model_name = name.capitalize()
            model = getattr(models, model_name)
            with archive.open(filename) as binary_file:
                file = io.TextIOWrapper(binary_file, encoding="utf-8", newline="")
                ingest_to_model(file, model, engine)


def fix_autoincrement(engine, table_name):
    """Auto-increment pointers are not updated when IDs are set explicitly,
    so we manually update the pointer so subsequent inserts work correctly.
    """
    engine.execute("select setval('{0}_id_seq', max(id)) from {0}".format(table_name))


def ingest_to_model(file, model, engine=None):
    """Load data from a CSV file handle into storage for a
    SQLAlchemy model class.
    """
    if engine is None:
        engine = db.engine
    reader = csv.reader(file)
    columns = tuple('"{}"'.format(n) for n in next(reader))
    with engine.begin() as conn:  # <-- BEGIN; will COMMIT or ROLLBACK
        copy_from(
            file,
            model,
            conn,
            columns=columns,
            format="csv",
            HEADER=False,
        )
        # run helper DDL in the same transaction
        fix_autoincrement(conn, model.__table__.name)


def archive_data(id, src, dst):
    print("Zipping up the package...")
    with ZipFile(dst, "w", ZIP_DEFLATED, allowZip64=True) as zf:
        for root, dirs, files in os.walk(src):
            for file in files:
                filename = os.path.join(root, file)
                arcname = filename.replace(src, "").lstrip("/")
                zf.write(filename, arcname)
    shutil.rmtree(src)
    print(f"Done. Local export available in {dst}")


def _get_canonical_aws_user_id(s3):
    return s3.meta.client.list_buckets()["Owner"]["ID"]


def _get_or_create_s3_bucket(s3, name):
    """Get an S3 bucket resource after making sure it exists"""
    exists = True
    try:
        s3.meta.client.head_bucket(Bucket=name)
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            exists = False
        else:
            raise

    if not exists:
        s3.create_bucket(Bucket=name)

    return s3.Bucket(name)


def _generate_s3_url(bucket, key):
    return "https://{}.s3.amazonaws.com/{}".format(bucket.name, key)


def user_s3_bucket(canonical_user_id=None):
    """Get the user's S3 bucket."""
    s3 = _s3_resource()
    if not canonical_user_id:
        canonical_user_id = _get_canonical_aws_user_id(s3)

    s3_bucket_name = "dallinger-{}".format(
        hashlib.sha256(canonical_user_id.encode("utf8")).hexdigest()[0:8]
    )

    return _get_or_create_s3_bucket(s3, s3_bucket_name)


def dallinger_s3_bucket():
    """The public `dallinger` S3 bucket."""
    s3 = _s3_resource(dallinger_region=True)
    return s3.Bucket("dallinger")


def registration_s3_bucket():
    """The public write-only `dallinger-registration` S3 bucket."""
    config = get_config(load=True)
    if config.get("enable_global_experiment_registry", False):
        s3 = _s3_resource(dallinger_region=True)
        return s3.Bucket("dallinger-registrations")


def _s3_resource(dallinger_region=False):
    """A boto3 S3 resource using the AWS keys in the config."""
    config = get_config(load=True)
    region = "us-east-1" if dallinger_region else config.get("aws_region")
    return boto3.resource(
        "s3",
        region_name=region,
        aws_access_key_id=config.get("aws_access_key_id"),
        aws_secret_access_key=config.get("aws_secret_access_key"),
    )


class Data:
    """Dallinger data object."""

    def __init__(self, URL):
        self.source = URL

        if self.source.endswith(".zip"):
            input_zip = ZipFile(URL)
            tmp_dir = tempfile.mkdtemp()
            input_zip.extractall(tmp_dir)

            for tab in table_names:
                setattr(
                    self,
                    "{}s".format(tab),
                    Table(os.path.join(tmp_dir, "data", "{}.csv".format(tab))),
                )


class Table:
    """Dallinger data-table object."""

    def __init__(self, path):
        with open(path) as f:
            self.tablib_dataset = tablib.Dataset().load(f.read(), "csv")

    @property
    def csv(self):
        """Comma-separated values."""
        return self.tablib_dataset.csv

    @property
    def dict(self):
        """A Python dictionary."""
        return self.tablib_dataset.dict[0]

    @property
    def df(self):
        """A pandas DataFrame."""
        return self.tablib_dataset.df

    @property
    def html(self):
        """An HTML table."""
        return self.tablib_dataset.html

    @property
    def latex(self):
        """A LaTeX table."""
        return self.tablib_dataset.latex

    @property
    def ods(self):
        """An OpenDocument Spreadsheet."""
        return self.tablib_dataset.ods

    @property
    def tsv(self):
        """Tab-separated values."""
        return self.tablib_dataset.tsv

    @property
    def xls(self):
        """Legacy Excel spreadsheet format."""
        return self.tablib_dataset.xls

    @property
    def xlsx(self):
        """Modern Excel spreadsheet format."""
        return self.tablib_dataset.xlsx

    @property
    def yaml(self):
        """YAML."""
        return self.tablib_dataset.yaml
