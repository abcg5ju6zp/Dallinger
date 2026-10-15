"""Tests for the data module."""

import os
import shutil
import sys
import zipfile
from datetime import datetime
from unittest import mock

import pytest
from sqlalchemy import create_engine, text

from dallinger.db import db_url
from dallinger.replay import ReplayError, active_generations

_MISSING = object()


@pytest.fixture
def zip_path():
    return os.path.join("tests", "datasets", "test_export.zip")


def database_exists(name):
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            return bool(
                conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": name},
                ).scalar()
            )
    finally:
        engine.dispose()


@pytest.mark.slow
class TestReplayState:
    @pytest.fixture
    def cleanup(self):
        yield
        shutil.rmtree("data", ignore_errors=True)

    @pytest.fixture
    def experiment(self):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        yield Bartlett1932(no_configure=True)

    bartlett_export = os.path.join("tests", "datasets", "bartlett_bots.zip")
    target = datetime(2017, 6, 23, 12, 0, 29, 941148)

    @pytest.fixture
    def scrubber(self, experiment, db_session):
        with experiment.restore_state_from_replay(
            "bartlett-test", session=db_session, zip_path=self.bartlett_export
        ) as scrubber:
            yield scrubber

    def test_scrub_forwards(self, scrubber):
        target = datetime(2017, 6, 23, 12, 0, 29, 941148)
        with mock.patch(
            "dlgr.demos.bartlett1932.experiment.Bartlett1932.replay_event"
        ) as replay_event:
            replay_event.assert_not_called()
            scrubber(target)
            assert replay_event.call_count == 4
            scrubber(datetime.now())
            assert replay_event.call_count == 5

    def test_cannot_scrub_backwards(self, scrubber):
        target = datetime(2017, 6, 23, 12, 0, 29, 941148)
        with mock.patch(
            "dlgr.demos.bartlett1932.experiment.Bartlett1932.replay_event"
        ) as replay_event:
            replay_event.assert_not_called()
            scrubber(datetime.now())
            assert replay_event.call_count == 5
            with pytest.raises(NotImplementedError):
                scrubber(target)

    def test_successful_replay_converges_everything(self, experiment):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        module_before = sys.modules.get("dallinger_experiment", _MISSING)
        generation = None
        with experiment.restore_state_from_replay(
            "bartlett-clean", zip_path=self.bartlett_export
        ) as scrubber:
            generation = experiment._replay_generation
            temp_db = generation.database_name
            assert database_exists(temp_db)
            assert generation.published is True
            assert generation.failed is False

            with mock.patch.object(
                Bartlett1932, "replay_event", autospec=True
            ) as replay_event:
                scrubber(self.target)
                assert replay_event.call_count == 4

            # Export queries keep using the per-scrub app id suffix.
            assert experiment.app_id == "{}_{}".format(
                experiment.original_app_id, self.target.isoformat()
            )

        assert generation.finished is True
        assert generation.closed is True
        assert not database_exists(temp_db)
        assert sys.modules.get("dallinger_experiment", _MISSING) is module_before
        assert active_generations() == ()

    def test_missing_export_leaves_no_resources(self, experiment):
        module_before = sys.modules.get("dallinger_experiment", _MISSING)
        with pytest.raises(IOError):
            with experiment.restore_state_from_replay(
                "bartlett-missing", zip_path=None
            ):
                pass
        assert sys.modules.get("dallinger_experiment", _MISSING) is module_before
        assert active_generations() == ()
        assert not any(
            "bartlett-missing" in gen
            for gen in self._import_database_names()
        )

    def test_corrupt_export_leaves_no_resources(self, experiment, tmp_path):
        corrupt = tmp_path / "corrupt.zip"
        corrupt.write_bytes(b"not a real zip")
        module_before = sys.modules.get("dallinger_experiment", _MISSING)
        with pytest.raises(ReplayError):
            with experiment.restore_state_from_replay(
                "bartlett-corrupt", zip_path=str(corrupt)
            ):
                pass
        assert sys.modules.get("dallinger_experiment", _MISSING) is module_before
        assert active_generations() == ()
        assert not any(
            "bartlett-corrupt" in gen
            for gen in self._import_database_names()
        )

    def test_replay_event_failure_rolls_back_and_converges(self, experiment):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        from dallinger import db

        with pytest.raises(ValueError, match="boom"):
            with experiment.restore_state_from_replay(
                "bartlett-hookfail", zip_path=self.bartlett_export
            ) as scrubber:
                generation = experiment._replay_generation
                temp_db = generation.database_name

                calls = {"n": 0}

                def failing_event(self, event):
                    calls["n"] += 1
                    if calls["n"] == 3:
                        raise ValueError("boom")

                with mock.patch.object(
                    Bartlett1932, "replay_event", autospec=True
                ) as replay_event, mock.patch.object(
                    db.session, "rollback", autospec=True
                ) as rollback, mock.patch.object(
                    Bartlett1932, "replay_finish", autospec=True
                ) as replay_finish:
                    replay_event.side_effect = failing_event
                    scrubber(self.target)

        # The failure position and confirmed progress are preserved.
        assert generation.failed is True
        assert generation.failure_stage == "event"
        assert generation.failure_position.index == 2
        assert generation.confirmed_count == 2
        # The main session was rolled back and replay_finish was not called.
        assert rollback.called
        assert replay_finish.call_count == 0
        assert generation.finished is False
        # Resources still converged despite the failure.
        assert generation.closed is True
        assert not database_exists(temp_db)
        assert active_generations() == ()

    def test_replay_start_hook_failure_converges(self, experiment):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        with mock.patch.object(
            Bartlett1932, "replay_start", autospec=True
        ) as replay_start:
            replay_start.side_effect = RuntimeError("start failed")
            with pytest.raises(RuntimeError, match="start failed"):
                with experiment.restore_state_from_replay(
                    "bartlett-startfail", zip_path=self.bartlett_export
                ):
                    pass

        assert active_generations() == ()
        assert not any(
            "bartlett-startfail" in gen
            for gen in self._import_database_names()
        )

    def test_replay_finish_hook_failure_converges(self, experiment):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        with mock.patch.object(
            Bartlett1932, "replay_finish", autospec=True
        ) as replay_finish:
            replay_finish.side_effect = RuntimeError("finish failed")
            with pytest.raises(RuntimeError, match="finish failed"):
                with experiment.restore_state_from_replay(
                    "bartlett-finishfail", zip_path=self.bartlett_export
                ):
                    generation = experiment._replay_generation
                    temp_db = generation.database_name

        assert generation.failed is True
        assert generation.failure_stage == "finish"
        assert generation.finished is False
        assert not database_exists(temp_db)
        assert active_generations() == ()

    def test_converge_is_idempotent(self, experiment):
        with experiment.restore_state_from_replay(
            "bartlett-idempotent", zip_path=self.bartlett_export
        ):
            generation = experiment._replay_generation
            temp_db = generation.database_name
        assert not database_exists(temp_db)
        # A restart/re-entered cleanup only converges resources again.
        generation.converge()
        assert generation.closed is True
        assert not database_exists(temp_db)

    def test_double_cleanup_as_notebook_rerun_is_safe(self, experiment):
        replay = experiment.restore_state_from_replay(
            "bartlett-double-exit", zip_path=self.bartlett_export
        )
        scrubber = replay.__enter__()
        temp_db = experiment._replay_generation.database_name
        assert database_exists(temp_db)
        replay.__exit__(None, None, None)
        assert not database_exists(temp_db)
        # Re-running the cell invokes the stored cleanup a second time.
        replay.__exit__(None, None, None)
        assert active_generations() == ()
        assert scrubber.generation.closed is True

    def test_concurrent_replays_get_isolated_databases(self):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        first = Bartlett1932(no_configure=True)
        second = Bartlett1932(no_configure=True)
        with first.restore_state_from_replay(
            "bartlett-conc-a", zip_path=self.bartlett_export
        ):
            gen_a = first._replay_generation
            with second.restore_state_from_replay(
                "bartlett-conc-b", zip_path=self.bartlett_export
            ):
                gen_b = second._replay_generation
                assert gen_a.generation_id != gen_b.generation_id
                assert gen_a.database_name != gen_b.database_name
                assert database_exists(gen_a.database_name)
                assert database_exists(gen_b.database_name)
                assert set(active_generations()) == {gen_a, gen_b}
            assert not database_exists(gen_b.database_name)
        assert not database_exists(gen_a.database_name)
        assert active_generations() == ()

    def test_rerun_same_app_id_converges_previous_generation(self):
        from dlgr.demos.bartlett1932.experiment import Bartlett1932

        first = Bartlett1932(no_configure=True)
        second = Bartlett1932(no_configure=True)
        with first.restore_state_from_replay(
            "bartlett-rerun", zip_path=self.bartlett_export
        ):
            gen_a = first._replay_generation
            name_a = gen_a.database_name
            with second.restore_state_from_replay(
                "bartlett-rerun", zip_path=self.bartlett_export
            ):
                gen_b = second._replay_generation
                name_b = gen_b.database_name
                # The older generation is converged and its temp db dropped.
                assert gen_a.closed is True
                assert not database_exists(name_a)
                assert name_a != name_b
                assert database_exists(name_b)
                assert gen_b.finished is False
            assert not database_exists(name_b)
        # Unwinding the superseded context is a no-op: no finish, no error.
        assert gen_a.finished is False
        assert gen_b.finished is True
        assert active_generations() == ()

    def test_incomplete_export_leaves_no_resources(self, experiment, tmp_path):
        incomplete = tmp_path / "incomplete.zip"
        with zipfile.ZipFile(incomplete, "w") as archive:
            archive.writestr("data/network.csv", "id\n1\n")
        with pytest.raises(ReplayError):
            with experiment.restore_state_from_replay(
                "bartlett-incomplete", zip_path=str(incomplete)
            ):
                pass
        assert active_generations() == ()

    @staticmethod
    def _import_database_names():
        engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT datname FROM pg_database "
                        "WHERE datname LIKE '%-import-%'"
                    )
                ).fetchall()
        finally:
            engine.dispose()
        return [row[0] for row in rows]
