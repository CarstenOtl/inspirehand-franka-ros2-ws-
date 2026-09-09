from datetime import datetime, timezone
import json

import cv2
import numpy as np

from camera_calibration.calibration_math import make_transform
from camera_calibration.run_logging import (
    create_run_directory,
    transform_record,
    write_json,
    write_sample_snapshot,
)


def test_create_run_directory_is_timestamped_and_collision_safe(tmp_path):
    moment = datetime(2026, 9, 8, 20, 15, 30, 123456, tzinfo=timezone.utc)
    first = create_run_directory(tmp_path, moment)
    second = create_run_directory(tmp_path, moment)
    assert first.name == "20260908T201530_123456Z"
    assert second.name == "20260908T201530_123456Z_01"


def test_transform_record_and_json_are_machine_readable(tmp_path):
    transform = make_transform(np.eye(3), [0.1, -0.2, 0.3])
    payload = {"world_to_hand": transform_record(transform)}
    destination = tmp_path / "sample_001.json"
    write_json(destination, payload)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["world_to_hand"]["translation"] == {
        "x": 0.1,
        "y": -0.2,
        "z": 0.3,
    }


def test_sample_snapshot_contains_annotations(tmp_path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    corners = np.asarray([[120, 80], [200, 80], [200, 160], [120, 160]], dtype=float)
    camera_to_tag = make_transform(np.eye(3), [0.0, 0.0, 0.5])
    camera_matrix = np.asarray([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    destination = tmp_path / "sample_001.png"
    write_sample_snapshot(
        destination,
        image,
        corners,
        camera_to_tag,
        camera_matrix,
        np.zeros(5),
        0.04,
        1,
        0.25,
    )
    saved = cv2.imread(str(destination))
    assert saved is not None
    assert np.count_nonzero(saved) > 0
