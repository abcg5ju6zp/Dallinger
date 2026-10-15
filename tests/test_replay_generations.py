"""Unit tests for isolated replay generation primitives.

These tests do not need a database: export validation, database naming and
the lightweight generation registry are exercised directly.
"""

import os
from unittest import mock

import pytest

from dallinger.replay import (
    ReplayGeneration,
    ReplayError,
    active_generations,
    validate_export_package,
)

DATASETS = os.path.join("tests", "datasets")
BARTLETT_EXPORT = os.path.join(DATASETS, "bartlett_bots.zip")
TEST_EXPORT = os.path.join(DATASETS, "test_export.zip")
_MISSING = object()


class DummyExperiment:
    app_id = "dummy-app"


class TestExportValidation:
    def test_accepts_complete_exports(self):
        validate_export_package(BARTLETT_EXPORT)
        validate_export_package(TEST_EXPORT)

    def test_missing_path_is_replay_error(self, tmp_path):
        missing = tmp_path / "does-not-exist.zip"
        with pytest.raises(ReplayError) as exc_info:
            validate_export_package(str(missing))
        assert exc_info.value.stage == "validate"

    def test_corrupt_zip_is_replay_error(self, tmp_path):
        corrupt = tmp_path / "corrupt.zip"
        corrupt.write_bytes(b"this is not a zip file")
        with pytest.raises(ReplayError) as exc_info:
            validate_export_package(str(corrupt))
        assert exc_info.value.stage == "validate"

    def test_archive_without_tables_is_rejected(self, tmp_path):
        import zipfile

        incomplete = tmp_path / "incomplete.zip"
        with zipfile.ZipFile(incomplete, "w") as archive:
            archive.writestr("data/network.csv", "id\n1\n")
        with pytest.raises(ReplayError) as exc_info:
            validate_export_package(str(incomplete))
        assert exc_info.value.stage == "validate"
        assert "info" in str(exc_info.value)


class TestDatabaseNaming:
    def test_short_app_id_gets_readable_name(self):
        name = ReplayGeneration._database_name(
            "dallinger", "my-app", "abc123"
        )
        assert name == "dallinger-import-my-app-abc123"

    def test_name_is_unique_per_generation(self):
        names = {
            ReplayGeneration._database_name("dallinger", "app", gid)
            for gid in ("aaa111", "bbb222", "ccc333")
        }
        assert len(names) == 3

    def test_long_app_id_is_hashed_and_within_postgres_limit(self):
        long_app_id = "x" * 100
        name = ReplayGeneration._database_name(
            "dallinger", long_app_id, "abc123"
        )
        assert len(name) <= 63
        assert name.startswith("dallinger-import-")
        assert "abc123" in name


class TestLightweightGeneration:
    def test_generations_are_registered_with_unique_ids(self):
        experiment = DummyExperiment()
        first = ReplayGeneration.lightweight(experiment)
        second = ReplayGeneration.lightweight(experiment)
        try:
            assert first.generation_id != second.generation_id
            assert first in active_generations()
            assert second in active_generations()
        finally:
            first.converge()
            second.converge()

    def test_converge_is_idempotent(self):
        generation = ReplayGeneration.lightweight(DummyExperiment())
        generation.converge()
        generation.converge()
        assert generation.closed is True
        assert generation not in active_generations()

    def test_progress_tracks_confirmed_events(self):
        class Event:
            def __init__(self, event_id):
                self.id = event_id
                self.type = "info"
                self.creation_time = object()

        generation = ReplayGeneration.lightweight(DummyExperiment())
        try:
            generation.confirm(Event(10))
            generation.confirm(Event(11))
            assert generation.confirmed_count == 2
            assert generation.last_position.index == 1
            assert generation.last_position.event_id == 11
            progress = generation.progress
            assert progress["confirmed_count"] == 2
            assert progress["failed"] is False
        finally:
            generation.converge()

    def test_record_failure_preserves_position(self):
        class Event:
            id = 7
            type = "info"
            creation_time = "2017-01-01"

        error = RuntimeError("boom")
        generation = ReplayGeneration.lightweight(DummyExperiment())
        try:
            with mock.patch("dallinger.replay.db") as db:
                generation.record_failure("event", event=Event(), exc=error)
            db.session.rollback.assert_called_once_with()
            assert generation.failed is True
            assert generation.failure_stage == "event"
            assert generation.failure_position.event_id == 7
            assert generation.failure_position.event_type == "info"
            assert generation.progress["failed"] is True
        finally:
            generation.converge()

    def test_lightweight_generation_cannot_be_used_as_context_manager(self):
        generation = ReplayGeneration.lightweight(DummyExperiment())
        try:
            with pytest.raises(ReplayError):
                with generation:
                    pass
        finally:
            generation.converge()
