"""Tests for the calibrated-camera passive MuJoCo scene builder."""

import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest


SCRIPT = Path(__file__).with_name("visualize_calibrated_camera.py")
SPEC = importlib.util.spec_from_file_location("camera_pose_visualizer_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def calibration_result():
    optical = np.eye(4)
    optical[:3, 3] = [0.7, -0.2, 0.8]
    return {
        "schema_version": 1,
        "parent_frame": "world",
        "child_frame": "camera_link",
        "camera_optical_frame": "camera_color_optical_frame",
        "camera_intrinsics": {
            "matrix_3x3": [
                [1380.0, 0.0, 960.0],
                [0.0, 1382.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion": [0.0] * 5,
            "width": 1920,
            "height": 1080,
        },
        "transform": {"matrix_4x4": np.eye(4).tolist()},
        "world_to_camera_optical": {"matrix_4x4": optical.tolist()},
    }


def test_reads_pure_json_and_last_result_from_ros_log(tmp_path):
    first = calibration_result()
    second = calibration_result()
    second["parent_frame"] = "base"
    pure = tmp_path / "calibration.json"
    pure.write_text(json.dumps(first), encoding="utf-8")
    assert MODULE.load_calibration_result(pure)["parent_frame"] == "world"

    log = tmp_path / "calibration.log"
    log.write_text(
        "[INFO] CALIBRATION_RESULT " + json.dumps(first) + "\n"
        "later text\n[INFO] CALIBRATION_RESULT " + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    assert MODULE.load_calibration_result(log)["parent_frame"] == "base"


def test_requires_exact_optical_transform_and_a_coincident_root_frame():
    legacy = calibration_result()
    legacy.pop("world_to_camera_optical")
    with pytest.raises(ValueError, match="legacy calibration result"):
        MODULE.calibration_poses(legacy)

    wrong_root = calibration_result()
    wrong_root["parent_frame"] = "map"
    with pytest.raises(ValueError, match="not coincident"):
        MODULE.calibration_poses(wrong_root)


def test_decorated_scene_contains_noncolliding_housing_camera_and_frustum():
    result = calibration_result()
    mount, optical, matrix, size = MODULE.calibration_poses(result)
    xml = MODULE.decorated_scene_xml(
        MODULE.DEFAULT_MODEL, mount, optical, matrix, size, 0.35
    )
    root = ET.fromstring(xml)

    camera = root.find(f".//camera[@name='{MODULE.CAMERA_NAME}']")
    assert camera is not None
    assert float(camera.get("fovy")) == pytest.approx(
        np.degrees(2.0 * np.arctan(1080.0 / (2.0 * 1382.0)))
    )
    assert root.find(".//geom[@name='calibrated_camera_housing']") is not None
    optical_body = root.find(".//body[@name='calibrated_camera_optical']")
    assert optical_body is not None
    capsules = optical_body.findall("geom[@type='capsule']")
    assert len(capsules) == 7  # three optical axes plus four frustum rays
    assert all(geom.get("contype") == "0" for geom in capsules)
    assert all(geom.get("group") == "5" for geom in capsules)


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ],
)
def test_matrix_to_quaternion_handles_all_trace_branches(rotation):
    quaternion = MODULE._matrix_to_quaternion_wxyz(rotation)
    w, x, y, z = quaternion
    recovered = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(recovered, rotation, atol=1.0e-12)
