"""Tests for the data module."""

import csv
import io
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from unittest import mock
from zipfile import ZipFile

import botocore.exceptions
import pandas as pd
import psycopg2
import pytest

import dallinger
from dallinger.utils import generate_random_id, open_for_csv

LOCAL_DB_URI = "postgresql://dallinger:dallinger@localhost/dallinger"


def _client_error(code):
    return botocore.exceptions.ClientError(
        {"Error": {"Code": str(code), "Message": "boom"}}, "Op"
    )


@pytest.fixture
def zip_path():
    return os.path.join("tests", "datasets", "test_export.zip")


@pytest.mark.s3buckets
@pytest.mark.usefixtures("check_s3buckets")
class TestDataS3BucketCreation:
    """Tests that actually create Buckets on S3."""

    def test_user_s3_bucket_first_time(self):
        bucket = dallinger.data.user_s3_bucket(canonical_user_id=generate_random_id())
        assert bucket
        bucket.delete()

    def test_user_s3_bucket_thrice(self):
        id = generate_random_id()
        for i in range(3):
            bucket = dallinger.data.user_s3_bucket(canonical_user_id=id)
            assert bucket
        bucket.delete()

    def test_user_s3_bucket_no_id_provided(self):
        bucket = dallinger.data.user_s3_bucket()
        assert bucket


@pytest.mark.slow
class TestDataS3Integration:
    """Tests that interact with the network and S3, but do not create
    S3 buckets.
    """

    def test_connection_to_s3(self):
        s3 = dallinger.data._s3_resource()
        assert s3

    @pytest.mark.skip(
        reason="Temporarily skipped because of error preventing new release. OSError: Dataset 3b9c2aeb-0eb7-4432-803e-bc437e17b3bb could not be found."
    )
    def test_data_loading(self):
        data = dallinger.data.load("3b9c2aeb-0eb7-4432-803e-bc437e17b3bb")
        assert data
        assert data.networks.csv

    def test_register_id_disabled(self, active_config):
        new_uuid = "12345-12345-12345-12345"
        url = dallinger.data.register(new_uuid, "http://original-url.com/value")
        assert url is None

    def test_register_id(self, active_config):
        new_uuid = "12345-12345-12345-12345"
        active_config.set("enable_global_experiment_registry", True)
        with mock.patch("dallinger.data.boto3") as boto:
            s3 = boto.resource = mock.Mock()
            s3.return_value = s3
            s3_bucket = s3.Bucket = mock.Mock()
            s3_bucket.return_value = s3_bucket
            s3_bucket.name = "my-fake-registrations"
            s3_object = s3_bucket.Object = mock.Mock()
            s3_object.return_value = s3_object
            s3_objects = s3_bucket.objects = mock.Mock()
            s3_filter = s3_objects.filter = mock.Mock()
            s3_filter.return_value = []
            url = dallinger.data.register(new_uuid, "http://original-url.com/value")

            s3.assert_called_once()
            s3_bucket.assert_called_once_with("dallinger-registrations")
            s3_object.assert_called_once_with(new_uuid + ".reg")
            s3_object.put.assert_called_once_with(Body="http://original-url.com/value")

            # The registration creates a new file in the dallinger-registrations bucket
            assert url.startswith("https://my-fake-registrations.")
            assert new_uuid in url

            # We should be able to check that the UUID is registered
            assert dallinger.data.is_registered(new_uuid) is False
            s3_filter.assert_called_once_with(Prefix=new_uuid + ".reg")


class TestDataLocally:
    """Tests that interact with local data only, and are relatively fast to
    execute.
    """

    @pytest.fixture
    def cleanup(self):
        yield
        if os.path.exists("data"):
            shutil.rmtree("data")

    @pytest.fixture
    def export(self, cleanup):
        path = dallinger.data.export("12345-12345-12345-12345", local=True)
        return path

    data_path = os.path.join(
        "tests", "datasets", "12eee6c6-f37f-4963-b684-da585acd77f1-data.zip"
    )

    bartlett_export = os.path.join("tests", "datasets", "bartlett_bots.zip")

    def test_dataset_creation(self):
        """Load a dataset."""
        dallinger.data.Data(self.data_path)

    def test_conversions(self):
        data = dallinger.data.Data(self.data_path)
        assert data.networks.csv
        assert data.networks.dict
        assert data.networks.html
        assert data.networks.latex
        assert data.networks.ods
        assert data.networks.tsv
        assert data.networks.xls
        assert data.networks.xlsx
        assert data.networks.yaml
        assert type(data.networks.tablib_dataset.df) is pd.DataFrame

    def test_csv_conversion(self):
        data = dallinger.data.Data(self.data_path)
        assert data.networks.csv[0:3] == "id,"

    def test_tsv_conversion(self):
        data = dallinger.data.Data(self.data_path)
        assert data.networks.tsv[0:3] == "id\t"

    def test_dict_conversion(self):
        data = dallinger.data.Data(self.data_path)
        assert type(data.networks.dict) is dict

    def test_df_conversion(self):
        data = dallinger.data.Data(self.data_path)
        assert data.networks.tablib_dataset.df.shape == (1, 13)

    def test_local_data_loading(self):
        local_data_id = "77777-77777-77777-77777"
        dallinger.data.export(local_data_id, local=True)
        data = dallinger.data.load(local_data_id)
        assert data
        assert data.networks.csv

    def test_export_of_nonexistent_database(self):
        nonexistent_local_db = str(uuid.uuid4())
        with pytest.raises(psycopg2.OperationalError):
            dallinger.data.copy_db_to_csv(nonexistent_local_db, "")

    def test_export_of_dallinger_database(self):
        export_dir = tempfile.mkdtemp()
        dallinger.data.copy_db_to_csv(
            "postgresql://dallinger:dallinger@localhost/dallinger", export_dir
        )
        assert os.path.isfile(os.path.join(export_dir, "network.csv"))

    def test_exported_database_includes_headers(self):
        export_dir = tempfile.mkdtemp()
        dallinger.data.copy_db_to_csv(
            "postgresql://dallinger:dallinger@localhost/dallinger", export_dir
        )
        network_table_path = os.path.join(export_dir, "network.csv")
        assert os.path.isfile(network_table_path)
        with open_for_csv(network_table_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader)
            assert "creation_time" in header

    def test_export(self, export):
        assert os.path.isfile("data/12345-12345-12345-12345-data.zip")

    def test_export_directory_format(self, export):
        archive = ZipFile(export)
        assert "data/info.csv" in archive.namelist()

    def test_export_compatible_with_data(self, export):
        assert dallinger.data.Data(export)

    def test_scrub_pii(self):
        path_to_data = os.path.join("tests", "datasets", "pii")
        dallinger.data._scrub_participant_table(path_to_data)
        with open_for_csv(os.path.join(path_to_data, "participant.csv"), "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)  # Skip the header
            for row in reader:
                assert "PII" not in row

    def test_scrub_pii_preserves_participants(self, db_session, zip_path, cleanup):
        dallinger.data.ingest_zip(zip_path)
        assert len(dallinger.models.Participant.query.all()) == 4
        path = dallinger.data.export("test_export", local=True, scrub_pii=True)
        with ZipFile(path) as zf:
            with zf.open("data/participant.csv") as p_file:
                p_file = io.TextIOWrapper(p_file, encoding="utf8", newline="")
                assert len(p_file.readlines()) == 5  # 4 Participants + header row

    def test_copy_db_to_csv_includes_participant_data(self, db_session):
        dallinger.data.ingest_zip(self.bartlett_export)
        export_dir = tempfile.mkdtemp()
        dallinger.data.copy_db_to_csv(
            "postgresql://dallinger:dallinger@localhost/dallinger",
            export_dir,
            scrub_pii=False,
        )
        participant_table_path = os.path.join(export_dir, "participant.csv")
        assert os.path.isfile(participant_table_path)
        with open_for_csv(participant_table_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader)
            row1 = next(reader)
            assert row1[header.index("worker_id")] == "SM6DMD"

    def test_copy_db_to_csv_includes_scrubbed_participant_data(self, db_session):
        dallinger.data.ingest_zip(self.bartlett_export)
        export_dir = tempfile.mkdtemp()
        dallinger.data.copy_db_to_csv(
            "postgresql://dallinger:dallinger@localhost/dallinger",
            export_dir,
            scrub_pii=True,
        )
        participant_table_path = os.path.join(export_dir, "participant.csv")
        assert os.path.isfile(participant_table_path)
        with open_for_csv(participant_table_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader)
            row1 = next(reader)
            assert row1[header.index("worker_id")] == "1"


class TestImport:
    @pytest.fixture
    def network_file(self):
        data = """id,creation_time,property1,property2,property3,property4,property5,failed,time_of_death,type,max_size,full,role
1,2001-01-01 09:46:40.133536,,,,,,f,,fully-connected,4,f,experiment"""
        f = io.StringIO(initial_value=data)
        return f

    @pytest.fixture
    def missing_column_required(self):
        """Test participant table without worker_id column"""
        data = """id,creation_time,property1,property2,property3,property4,property5,failed,time_of_death,type,worker_id,\
assignment_id,unique_id,hit_id,mode,end_time,base_pay,bonus,status
1,2001-01-01 09:46:40.133536,,,,,,f,,participant,,8,8:36V4Q8R5ZLTJWMX0SFF0G6R67PCQMI,\
3EHVO81VN5E60KEEQ146ZGFI3FH1H6,live,2017-03-30 20:06:44.618385,,,returned"""
        f = io.StringIO(initial_value=data)
        return f

    @pytest.fixture
    def missing_column_not_required(self):
        """Test participant table without fingerprint_hash column"""
        data = """id,creation_time,property1,property2,property3,property4,property5,failed,time_of_death,type,worker_id,\
assignment_id,unique_id,hit_id,mode,end_time,base_pay,bonus,status
1,2001-01-01 09:46:40.133536,,,,,,f,,participant,8,36V4Q8R5ZLTJWMX0SFF0G6R67PCQMI,8:36V4Q8R5ZLTJWM\
X0SFF0G6R67PCQMI,3EHVO81VN5E60KEEQ146ZGFI3FH1H6,live,2017-03-30 20:06:44.618385,,,returned"""
        f = io.StringIO(initial_value=data)
        return f

    def test_ingest_to_model(self, db_session, network_file):
        dallinger.data.ingest_to_model(network_file, dallinger.models.Network)

        assert db_session.query(dallinger.models.Network).count() == 1
        network = db_session.query(dallinger.models.Network).first()
        assert network.type == "fully-connected"
        assert network.creation_time == datetime(2001, 1, 1, 9, 46, 40, 133536)
        assert network.role == "experiment"

    def test_ingest_to_model_allows_subsequent_insert(self, db_session, network_file):
        dallinger.data.ingest_to_model(network_file, dallinger.models.Network)

        db_session.add(dallinger.models.Network())
        db_session.flush()
        db_session.commit()

        networks = db_session.query(dallinger.models.Network).all()
        assert networks[1].id == 2

    def test_missing_column_required(self, db_session, missing_column_required):
        with pytest.raises(psycopg2.IntegrityError):
            dallinger.data.ingest_to_model(
                missing_column_required, dallinger.models.Participant
            )

    def test_missing_column_not_required(self, db_session, missing_column_not_required):
        dallinger.data.ingest_to_model(
            missing_column_not_required, dallinger.models.Participant
        )

        assert db_session.query(dallinger.models.Participant).count() == 1
        participant = db_session.query(dallinger.models.Participant).first()
        assert participant.creation_time == datetime(2001, 1, 1, 9, 46, 40, 133536)

    def test_ingest_zip_recreates_network(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)

        assert db_session.query(dallinger.models.Network).count() == 1
        network = db_session.query(dallinger.models.Network).first()
        assert network.type == "chain"

    def test_ingest_zip_recreates_participants(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)

        assert db_session.query(dallinger.models.Participant).count() == 4
        participants = db_session.query(dallinger.models.Participant).all()
        for p in participants:
            assert p.status == "approved"

    def test_ingest_zip_recreates_nodes(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)
        assert db_session.query(dallinger.models.Node).count() == 5

    def test_ingest_zip_recreates_infos(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)

        assert db_session.query(dallinger.models.Info).count() == 5
        infos = db_session.query(dallinger.models.Info).all()
        for info in infos:
            assert info.contents.startswith("One night two young men")

    def test_ingest_zip_recreates_notifications(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)
        assert db_session.query(dallinger.models.Notification).count() == 8

    def test_ingest_zip_recreates_questions(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)

        model = dallinger.models.Question
        p1_questions = db_session.query(model).filter_by(participant_id=1).all()
        for q in p1_questions:
            if q.response:
                assert q.response == "5"

    def test_ingest_zip_recreates_vectors(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)
        assert db_session.query(dallinger.models.Vector).count() == 4

    def test_ingest_zip_recreates_transmissions(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)
        assert db_session.query(dallinger.models.Transmission).count() == 4


class TestExportGenerations:
    """Consistency, staging, failure, retry and recovery semantics for
    export generations."""

    @pytest.fixture
    def cleanup(self):
        yield
        if os.path.exists("data"):
            shutil.rmtree("data")
        if os.path.exists("snapshots"):
            shutil.rmtree("snapshots")

    @pytest.fixture
    def db_with_data(self, db_session, zip_path):
        dallinger.data.ingest_zip(zip_path)

    @pytest.fixture
    def bartlett_data(self, db_session):
        dallinger.data.ingest_zip(
            os.path.join("tests", "datasets", "bartlett_bots.zip")
        )

    @staticmethod
    def _write_code_snapshot(experiment_id, payload=b"print('experiment')\n"):
        os.makedirs("snapshots", exist_ok=True)
        path = os.path.join("snapshots", "{}-code.zip".format(experiment_id))
        with ZipFile(path, "w") as zf:
            zf.writestr("experiment.py", payload)
        return path

    @staticmethod
    def _generation_dirs(experiment_id):
        root = os.path.join("data", ".exports", experiment_id)
        if not os.path.isdir(root):
            return []
        return [
            name
            for name in os.listdir(root)
            if name not in ("lock", "current.json")
        ]

    @staticmethod
    def _current(experiment_id):
        path = os.path.join("data", ".exports", experiment_id, "current.json")
        with open(path) as f:
            return json.load(f)

    # -- consistent database read point ------------------------------------

    def test_copy_pins_one_serializable_read_only_snapshot(self, cleanup):
        export_dir = tempfile.mkdtemp()
        with mock.patch("dallinger.data.psycopg2") as psy:
            conn = psy.connect.return_value
            cur = conn.cursor.return_value
            cur.fetchone.return_value = ("00001A2B-1", 42)
            info = dallinger.data.copy_db_to_csv(LOCAL_DB_URI, export_dir)

        conn.set_session.assert_called_once_with(
            readonly=True,
            deferrable=True,
            isolation_level=psy.extensions.ISOLATION_LEVEL_SERIALIZABLE,
        )
        cur.execute.assert_called_once_with(
            "SELECT pg_export_snapshot(), txid_current()"
        )
        # All nine table copies happen inside the pinned transaction, and the
        # transaction is committed only after the last COPY.
        assert cur.copy_expert.call_count == len(dallinger.data.table_names)
        conn.commit.assert_called_once_with()
        conn.rollback.assert_not_called()
        assert info == {"pg_snapshot": "00001A2B-1", "txid": 42}

    def test_copy_rolls_back_and_propagates_on_failure(self, cleanup):
        export_dir = tempfile.mkdtemp()
        with mock.patch("dallinger.data.psycopg2") as psy:
            conn = psy.connect.return_value
            cur = conn.cursor.return_value
            cur.fetchone.return_value = ("00001A2B-1", 42)
            cur.copy_expert.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError):
                dallinger.data.copy_db_to_csv(LOCAL_DB_URI, export_dir)
        conn.rollback.assert_called_once_with()
        conn.close.assert_called_once_with()

    # -- happy path and data-format compatibility --------------------------

    def test_local_export_publishes_consistent_archive(
        self, cleanup, db_with_data
    ):
        experiment_id = "gen-local-1"
        self._write_code_snapshot(experiment_id)
        path = dallinger.data.export(experiment_id, local=True, scrub_pii=True)

        assert path == os.path.join("data", "{}-data.zip".format(experiment_id))
        with ZipFile(path) as zf:
            names = zf.namelist()
            for table in dallinger.data.table_names:
                assert "data/{}.csv".format(table) in names
            assert "experiment_id.md" in names
            assert "{}-code.zip".format(experiment_id) in names
            assert zf.read("experiment_id.md") == experiment_id.encode()
            assert zf.testzip() is None
        # Existing data format stays readable.
        assert dallinger.data.Data(path)

    def test_successful_export_removes_staging(self, cleanup, db_with_data):
        experiment_id = "gen-local-2"
        dallinger.data.export(experiment_id, local=True)
        assert self._generation_dirs(experiment_id) == []
        current = self._current(experiment_id)
        assert current["generation"]
        assert current["archive"] == "{}-data.zip".format(experiment_id)
        assert current["stages"]["package"] == "complete"

    def test_export_without_code_snapshot_still_compatible(
        self, cleanup, db_with_data
    ):
        experiment_id = "gen-local-3"
        path = dallinger.data.export(experiment_id, local=True)
        with ZipFile(path) as zf:
            assert not any(n.endswith("-code.zip") for n in zf.namelist())
            assert "experiment_id.md" in zf.namelist()
        assert dallinger.data.Data(path)

    def test_repeated_export_creates_newer_generation(
        self, cleanup, db_with_data
    ):
        experiment_id = "gen-local-4"
        dallinger.data.export(experiment_id, local=True)
        first = self._current(experiment_id)["generation"]
        dallinger.data.export(experiment_id, local=True)
        second = self._current(experiment_id)["generation"]
        assert second > first
        assert self._generation_dirs(experiment_id) == []

    # -- failure semantics ---------------------------------------------------

    def test_snapshot_failure_leaves_no_publishable_file(
        self, cleanup, db_with_data
    ):
        experiment_id = "gen-fail-snapshot"
        with mock.patch(
            "dallinger.data.copy_db_to_csv", side_effect=RuntimeError("db down")
        ):
            with pytest.raises(dallinger.data.ExportStageError) as exc_info:
                dallinger.data.export_db_uri(
                    experiment_id, LOCAL_DB_URI, True, False
                )
        assert exc_info.value.stage == "snapshot"
        assert exc_info.value.retain is False
        assert not os.path.exists(
            os.path.join("data", "{}-data.zip".format(experiment_id))
        )
        assert self._generation_dirs(experiment_id) == []

    def test_anonymization_failure_keeps_diagnosable_stage(
        self, cleanup, bartlett_data
    ):
        experiment_id = "gen-fail-anon"
        with mock.patch(
            "dallinger.data._scrub_participant_table",
            side_effect=ValueError("bad pii"),
        ):
            with pytest.raises(dallinger.data.ExportStageError) as exc_info:
                dallinger.data.export(experiment_id, local=True, scrub_pii=True)
        assert exc_info.value.stage == "anonymize"
        assert exc_info.value.retain is True
        generation = self._generation_dirs(experiment_id)
        assert len(generation) == 1
        gdir = os.path.join(
            "data", ".exports", experiment_id, generation[0]
        )
        assert os.path.isfile(os.path.join(gdir, "FAILED"))
        # Snapshot data is preserved for diagnosis...
        assert os.path.isfile(os.path.join(gdir, "stage", "data", "participant.csv"))
        # ...but no archive is published or registered.
        assert not os.path.exists(
            os.path.join("data", "{}-data.zip".format(experiment_id))
        )

    def test_corrupt_code_snapshot_prevents_publish(self, cleanup, db_with_data):
        experiment_id = "gen-fail-code"
        os.makedirs("snapshots", exist_ok=True)
        with open(
            os.path.join("snapshots", "{}-code.zip".format(experiment_id)), "wb"
        ) as f:
            f.write(b"not a zip")
        with pytest.raises(dallinger.data.ExportStageError) as exc_info:
            dallinger.data.export(experiment_id, local=True)
        assert exc_info.value.stage == "code"
        assert self._generation_dirs(experiment_id)
        assert not os.path.exists(
            os.path.join("data", "{}-data.zip".format(experiment_id))
        )

    def test_terminal_failure_not_resumed_by_next_export(
        self, cleanup, bartlett_data
    ):
        experiment_id = "gen-fail-recover"
        with mock.patch(
            "dallinger.data._scrub_participant_table",
            side_effect=ValueError("bad pii"),
        ):
            with pytest.raises(dallinger.data.ExportStageError):
                dallinger.data.export(experiment_id, local=True, scrub_pii=True)
        failed = self._generation_dirs(experiment_id)
        # A later (non-anonymized) export starts a fresh generation but the
        # failed stage is retained for diagnosis.
        path = dallinger.data.export(experiment_id, local=True, scrub_pii=False)
        assert os.path.isfile(path)
        gens = self._generation_dirs(experiment_id)
        assert failed[0] in gens
        assert len(gens) == 1

    # -- restart, resume and orphan cleanup ---------------------------------

    def test_reconcile_removes_orphan_temp_dirs(self, cleanup, db_with_data):
        experiment_id = "gen-orphan"
        root = os.path.join("data", ".exports", experiment_id)
        os.makedirs(os.path.join(root, "ownerless", "stage", "data"))
        os.makedirs("data", exist_ok=True)
        with open(
            os.path.join("data", ".{}-data.zip.tmp-stale".format(experiment_id)),
            "w",
        ) as f:
            f.write("partial")
        dallinger.data.export(experiment_id, local=True)
        assert not os.path.exists(os.path.join(root, "ownerless"))
        assert not os.path.exists(
            os.path.join("data", ".{}-data.zip.tmp-stale".format(experiment_id))
        )

    def test_unverifiable_interrupted_attempt_is_discarded(
        self, cleanup, db_with_data
    ):
        experiment_id = "gen-interrupted"
        real_connect = psycopg2.connect
        real_publish = dallinger.data._ExportGeneration._stage_publish
        attempts = {"n": 0}

        def flaky_publish(self):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("disk full")
            return real_publish(self)

        with mock.patch(
            "dallinger.data.psycopg2.connect", wraps=real_connect
        ) as connect_spy, mock.patch.object(
            # Simulate a crash while publishing the local archive; a later
            # attempt publishes normally.
            dallinger.data._ExportGeneration,
            "_stage_publish",
            flaky_publish,
        ):
            with pytest.raises(dallinger.data.ExportStageError) as exc_info:
                dallinger.data.export(experiment_id, local=True)
            assert exc_info.value.retryable is True
            generation = self._generation_dirs(experiment_id)[0]
            # Tamper with a staged artifact after the crash: the attempt can
            # no longer be verified and must not be resumed.
            with open(
                os.path.join(
                    "data", ".exports", experiment_id, generation,
                    "stage", "data", "info.csv",
                ),
                "a",
            ) as f:
                f.write("corrupt,row\n")
            path = dallinger.data.export(experiment_id, local=True)
        # A fresh snapshot was taken: the unverifiable generation was dropped.
        assert connect_spy.call_count == 2
        assert os.path.isfile(path)
        assert self._generation_dirs(experiment_id) == []

    def test_register_failure_retries_same_object_without_reexport(
        self, cleanup, db_with_data, active_config
    ):
        experiment_id = "gen-retry-register"
        data_obj = mock.Mock()
        data_obj.metadata = {}
        bucket = mock.Mock()
        bucket.name = "user-bucket"
        bucket.Object.return_value = data_obj
        reg_obj = mock.Mock()
        reg_obj.load.side_effect = _client_error(404)
        reg_bucket = mock.Mock()
        reg_bucket.Object.return_value = reg_obj
        real_connect = psycopg2.connect

        with mock.patch("dallinger.data.aws_access_keys_present", lambda c: True), \
            mock.patch("dallinger.data.user_s3_bucket", return_value=bucket), \
            mock.patch(
                "dallinger.data.registration_s3_bucket", return_value=reg_bucket
            ), \
            mock.patch(
                "dallinger.data.psycopg2.connect", wraps=real_connect
            ) as connect_spy, \
            mock.patch.object(dallinger.data, "REGISTER_RETRY_DELAY", 0):
            reg_obj.put.side_effect = _client_error(503)
            with pytest.raises(dallinger.data.ExportStageError) as exc_info:
                dallinger.data.export_db_uri(
                    experiment_id, LOCAL_DB_URI, False, False
                )
            assert exc_info.value.stage == "register"
            assert exc_info.value.retryable is True
            generation = self._generation_dirs(experiment_id)[0]
            state_path = os.path.join(
                "data", ".exports", experiment_id, generation, "state.json"
            )
            with open(state_path) as f:
                state = json.load(f)
            assert state["status"] == "in_progress"
            assert not os.path.exists(
                os.path.join(
                    "data", ".exports", experiment_id, generation, "FAILED"
                )
            )
            uploaded_url = state["stages"]["upload"]["url"]
            connections_after_first = connect_spy.call_count
            uploads_after_first = bucket.upload_file.call_count

            # Restart / duplicate request: registration now succeeds.
            reg_obj.put.side_effect = None
            path = dallinger.data.export_db_uri(
                experiment_id, LOCAL_DB_URI, False, False
            )

        # Same object is registered; the data is neither copied nor uploaded
        # again.
        assert connect_spy.call_count == connections_after_first
        assert bucket.upload_file.call_count == uploads_after_first
        _, kwargs = reg_obj.put.call_args
        assert kwargs["Body"] == uploaded_url
        assert kwargs["Metadata"] == {"generation": generation}
        assert os.path.isfile(path)
        assert self._generation_dirs(experiment_id) == []

    # -- generation fencing --------------------------------------------------

    def test_old_generation_does_not_overwrite_newer_registration(
        self, cleanup, active_config
    ):
        older = "20200101T000000.000000-aaaaaaaa"
        newer = "20260101T000000.000000-bbbbbbbb"
        reg_obj = mock.Mock()
        reg_obj.metadata = {"generation": newer}
        reg_bucket = mock.Mock()
        reg_bucket.Object.return_value = reg_obj
        with mock.patch(
            "dallinger.data.registration_s3_bucket", return_value=reg_bucket
        ):
            result = dallinger.data.register(
                "exp-fence", "http://old-url", generation=older
            )
        assert result is None
        reg_obj.put.assert_not_called()

    def test_newer_generation_overwrites_older_registration(
        self, cleanup, active_config
    ):
        older = "20200101T000000.000000-aaaaaaaa"
        newer = "20260101T000000.000000-bbbbbbbb"
        reg_obj = mock.Mock()
        reg_obj.metadata = {"generation": older}
        reg_bucket = mock.Mock()
        reg_bucket.Object.return_value = reg_obj
        reg_bucket.name = "dallinger-registrations"
        with mock.patch(
            "dallinger.data.registration_s3_bucket", return_value=reg_bucket
        ):
            result = dallinger.data.register(
                "exp-fence", "http://new-url", generation=newer
            )
        assert result
        _, kwargs = reg_obj.put.call_args
        assert kwargs["Body"] == "http://new-url"
        assert kwargs["Metadata"] == {"generation": newer}

    def test_remote_newer_generation_blocks_upload(
        self, cleanup, db_with_data, active_config
    ):
        experiment_id = "gen-fence-upload"
        data_obj = mock.Mock()
        data_obj.metadata = {"generation": "29990101T000000.000000-ffffffff"}
        bucket = mock.Mock()
        bucket.name = "user-bucket"
        bucket.Object.return_value = data_obj
        with mock.patch("dallinger.data.aws_access_keys_present", lambda c: True), \
            mock.patch("dallinger.data.user_s3_bucket", return_value=bucket), \
            mock.patch(
                "dallinger.data.registration_s3_bucket", return_value=None
            ):
            with pytest.raises(dallinger.data.ExportSuperseded):
                dallinger.data.export_db_uri(
                    experiment_id, LOCAL_DB_URI, False, False
                )
        bucket.upload_file.assert_not_called()
        # The superseded attempt is retired; no in-progress generation lingers.
        assert self._generation_dirs(experiment_id) == []

