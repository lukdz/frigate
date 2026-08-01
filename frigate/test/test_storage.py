import datetime
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from peewee import DoesNotExist
from peewee_migrate import Router
from playhouse.sqlite_ext import SqliteExtDatabase
from playhouse.sqliteq import SqliteQueueDatabase
from pydantic import ValidationError

from frigate.config import FrigateConfig
from frigate.models import Event, Recordings
from frigate.storage import StorageMaintainer
from frigate.test.const import TEST_DB, TEST_DB_CLEANUPS


class TestHttp(unittest.TestCase):
    def setUp(self):
        # setup clean database for each test run
        migrate_db = SqliteExtDatabase("test.db")
        del logging.getLogger("peewee_migrate").handlers[:]
        router = Router(migrate_db)
        router.run()
        migrate_db.close()
        self.db = SqliteQueueDatabase(TEST_DB)
        models = [Event, Recordings]
        self.db.bind(models)
        self.test_dir = tempfile.mkdtemp()

        self.minimal_config = {
            "mqtt": {"host": "mqtt"},
            "detectors": {"test": {"type": "zmq"}},
            "model": {"labelmap_path": "labelmap.txt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {
                        "height": 1080,
                        "width": 1920,
                        "fps": 5,
                    },
                }
            },
        }
        self.double_cam_config = {
            "mqtt": {"host": "mqtt"},
            "detectors": {"test": {"type": "zmq"}},
            "model": {"labelmap_path": "labelmap.txt"},
            "cameras": {
                "front_door": {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {
                        "height": 1080,
                        "width": 1920,
                        "fps": 5,
                    },
                },
                "back_door": {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.2:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {
                        "height": 1080,
                        "width": 1920,
                        "fps": 5,
                    },
                },
            },
        }

    def tearDown(self):
        if not self.db.is_closed():
            self.db.close()

        try:
            for file in TEST_DB_CLEANUPS:
                os.remove(file)
        except OSError:
            pass

    def test_segment_calculations(self):
        """Test that the segment calculations are correct."""
        config = FrigateConfig(**self.double_cam_config)
        storage = StorageMaintainer(config, MagicMock())

        time_keep = datetime.datetime.now().timestamp()
        rec_fd_id = "1234567.frontdoor"
        rec_bd_id = "1234568.backdoor"
        _insert_mock_recording(
            rec_fd_id,
            os.path.join(self.test_dir, f"{rec_fd_id}.tmp"),
            time_keep,
            time_keep + 10,
            camera="front_door",
            seg_size=4,
            seg_dur=10,
        )
        _insert_mock_recording(
            rec_bd_id,
            os.path.join(self.test_dir, f"{rec_bd_id}.tmp"),
            time_keep + 10,
            time_keep + 20,
            camera="back_door",
            seg_size=8,
            seg_dur=20,
        )
        storage.calculate_camera_bandwidth()
        assert storage.camera_storage_stats == {
            "front_door": {"bandwidth": 1440, "needs_refresh": True},
            "back_door": {"bandwidth": 2880, "needs_refresh": True},
        }

    def test_segment_calculations_with_zero_segments(self):
        """Ensure segment calculation does not fail when migrating from previous version."""
        config = FrigateConfig(**self.minimal_config)
        storage = StorageMaintainer(config, MagicMock())

        time_keep = datetime.datetime.now().timestamp()
        rec_fd_id = "1234567.frontdoor"
        _insert_mock_recording(
            rec_fd_id,
            os.path.join(self.test_dir, f"{rec_fd_id}.tmp"),
            time_keep,
            time_keep + 10,
            camera="front_door",
            seg_size=0,
            seg_dur=10,
        )
        storage.calculate_camera_bandwidth()
        assert storage.camera_storage_stats == {
            "front_door": {"bandwidth": 0, "needs_refresh": True},
        }

    def test_storage_cleanup_at_max_usage_percent(self):
        """Ensure cleanup starts when disk usage reaches the configured limit."""
        config = self.minimal_config | {
            "record": {"storage_limit": {"max_usage_percent": 90}}
        }
        frigate_config = FrigateConfig(**config)
        storage = StorageMaintainer(frigate_config, MagicMock())

        assert not hasattr(frigate_config.cameras["front_door"].record, "storage_limit")

        with patch("frigate.storage.shutil.disk_usage") as disk_usage:
            disk_usage.return_value = MagicMock(
                total=1000 * pow(2, 20),
                used=899 * pow(2, 20),
                free=101 * pow(2, 20),
            )
            assert not storage.check_storage_needs_cleanup()

            disk_usage.return_value = MagicMock(
                total=1000 * pow(2, 20),
                used=900 * pow(2, 20),
                free=100 * pow(2, 20),
            )
            assert storage.check_storage_needs_cleanup()

    def test_storage_cleanup_reclaims_full_percent_deficit_in_one_pass(self):
        """Ensure a single cleanup pass reclaims the full percent-based deficit, not just the hourly amount."""
        config = self.minimal_config | {
            "record": {"storage_limit": {"max_usage_percent": 80}}
        }
        frigate_config = FrigateConfig(**config)
        storage = StorageMaintainer(frigate_config, MagicMock())
        storage.camera_storage_stats = {
            "front_door": {"bandwidth": 20, "needs_refresh": False}
        }

        time_delete = datetime.datetime.now().timestamp() - 7200
        files = []
        for i in range(20):
            id = f"{123456 + i}.delete"
            file = os.path.join(self.test_dir, f"{id}.tmp")
            _insert_mock_recording(
                id,
                file,
                time_delete + i * 10,
                time_delete + i * 10 + 10,
                seg_size=10,
            )
            files.append(file)

        with patch("frigate.storage.shutil.disk_usage") as disk_usage:
            disk_usage.return_value = MagicMock(
                total=1000 * pow(2, 20),
                used=950 * pow(2, 20),
                free=50 * pow(2, 20),
            )
            # deficit (150 MB) is far larger than the hourly bandwidth (20 MB)
            assert storage.get_cleanup_target_size() == 150

            storage.reduce_storage_consumption()

        # 16 of the 20 recordings (160 MB) must be removed on disk to clear the
        # deficit, not just the 2-3 the old hourly-only target would remove.
        # Checked on disk rather than via a DB re-select, since the recordings
        # iterator used by reduce_storage_consumption breaks early once the
        # target is met, leaving its cursor open and the read connection's
        # view stale until it is closed elsewhere.
        removed = [file for file in files if not os.path.exists(file)]
        assert len(removed) == 16

    def test_storage_cleanup_with_less_than_one_hour_free(self):
        """Ensure the existing low available storage check remains active."""
        config = FrigateConfig(**self.minimal_config)
        storage = StorageMaintainer(config, MagicMock())
        storage.camera_storage_stats = {
            "front_door": {"bandwidth": 101, "needs_refresh": False}
        }

        with patch("frigate.storage.shutil.disk_usage") as disk_usage:
            disk_usage.return_value = MagicMock(
                total=1000 * pow(2, 20),
                used=900 * pow(2, 20),
                free=100 * pow(2, 20),
            )
            assert storage.check_storage_needs_cleanup()

    def test_storage_limit_cannot_be_configured_per_camera(self):
        """Ensure the recording storage limit is global only."""
        self.minimal_config["cameras"]["front_door"]["record"] = {
            "storage_limit": {"max_usage_percent": 90}
        }

        with self.assertRaises(ValidationError):
            FrigateConfig(**self.minimal_config)

    def test_storage_cleanup(self):
        """Ensure that all recordings are cleaned up when necessary."""
        config = FrigateConfig(**self.minimal_config)
        storage = StorageMaintainer(config, MagicMock())

        id = "123456.keep"
        time_keep = datetime.datetime.now().timestamp()
        _insert_mock_event(
            id,
            time_keep,
            time_keep + 30,
            True,
        )
        rec_k_id = "1234567.keep"
        rec_k2_id = "1234568.keep"
        rec_k3_id = "1234569.keep"
        _insert_mock_recording(
            rec_k_id,
            os.path.join(self.test_dir, f"{rec_k_id}.tmp"),
            time_keep,
            time_keep + 10,
        )
        _insert_mock_recording(
            rec_k2_id,
            os.path.join(self.test_dir, f"{rec_k2_id}.tmp"),
            time_keep + 10,
            time_keep + 20,
        )
        _insert_mock_recording(
            rec_k3_id,
            os.path.join(self.test_dir, f"{rec_k3_id}.tmp"),
            time_keep + 20,
            time_keep + 30,
        )

        id2 = "7890.delete"
        time_delete = datetime.datetime.now().timestamp() - 360
        _insert_mock_event(id2, time_delete, time_delete + 30, False)
        rec_d_id = "78901.delete"
        rec_d2_id = "78902.delete"
        rec_d3_id = "78903.delete"
        _insert_mock_recording(
            rec_d_id,
            os.path.join(self.test_dir, f"{rec_d_id}.tmp"),
            time_delete,
            time_delete + 10,
        )
        _insert_mock_recording(
            rec_d2_id,
            os.path.join(self.test_dir, f"{rec_d2_id}.tmp"),
            time_delete + 10,
            time_delete + 20,
        )
        _insert_mock_recording(
            rec_d3_id,
            os.path.join(self.test_dir, f"{rec_d3_id}.tmp"),
            time_delete + 20,
            time_delete + 30,
        )

        storage.calculate_camera_bandwidth()
        storage.reduce_storage_consumption()
        with self.assertRaises(DoesNotExist):
            assert Recordings.get(Recordings.id == rec_k_id)
            assert Recordings.get(Recordings.id == rec_k2_id)
            assert Recordings.get(Recordings.id == rec_k3_id)
            Recordings.get(Recordings.id == rec_d_id)
            Recordings.get(Recordings.id == rec_d2_id)
            Recordings.get(Recordings.id == rec_d3_id)

    def test_storage_cleanup_keeps_retained(self):
        """Ensure that all recordings are cleaned up when necessary."""
        config = FrigateConfig(**self.minimal_config)
        storage = StorageMaintainer(config, MagicMock())

        id = "123456.keep"
        time_keep = datetime.datetime.now().timestamp()
        _insert_mock_event(
            id,
            time_keep,
            time_keep + 30,
            True,
        )
        rec_k_id = "1234567.keep"
        rec_k2_id = "1234568.keep"
        rec_k3_id = "1234569.keep"
        _insert_mock_recording(
            rec_k_id,
            os.path.join(self.test_dir, f"{rec_k_id}.tmp"),
            time_keep,
            time_keep + 10,
        )
        _insert_mock_recording(
            rec_k2_id,
            os.path.join(self.test_dir, f"{rec_k2_id}.tmp"),
            time_keep + 10,
            time_keep + 20,
        )
        _insert_mock_recording(
            rec_k3_id,
            os.path.join(self.test_dir, f"{rec_k3_id}.tmp"),
            time_keep + 20,
            time_keep + 30,
        )

        time_delete = datetime.datetime.now().timestamp() - 7200
        for i in range(0, 59):
            id = f"{123456 + i}.delete"
            _insert_mock_recording(
                id,
                os.path.join(self.test_dir, f"{id}.tmp"),
                time_delete,
                time_delete + 600,
            )

        storage.calculate_camera_bandwidth()
        storage.reduce_storage_consumption()
        assert Recordings.get(Recordings.id == rec_k_id)
        assert Recordings.get(Recordings.id == rec_k2_id)
        assert Recordings.get(Recordings.id == rec_k3_id)


def _insert_mock_event(
    id: str,
    start: int,
    end: int,
    retain: bool,
    camera: str = "front_door",
    label: str = "Mock",
) -> Event:
    """Inserts a basic event model with a given id."""
    return Event.insert(
        id=id,
        label=label,
        camera=camera,
        start_time=start,
        end_time=end,
        top_score=100,
        false_positive=False,
        zones=list(),
        thumbnail="",
        region=[],
        box=[],
        area=0,
        has_clip=True,
        has_snapshot=True,
        retain_indefinitely=retain,
    ).execute()


def _insert_mock_recording(
    id: str,
    file: str,
    start: int,
    end: int,
    camera="front_door",
    seg_size=8,
    seg_dur=10,
) -> Event:
    """Inserts a basic recording model with a given id."""
    # we must open the file so storage maintainer will delete it
    with open(file, "w"):
        pass

    return Recordings.insert(
        id=id,
        camera=camera,
        path=file,
        start_time=start,
        end_time=end,
        duration=seg_dur,
        motion=True,
        objects=True,
        segment_size=seg_size,
    ).execute()
