"""Keep the calibration target transform synchronized with the MuJoCo asset."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from camera_calibration.calibration_math import (
    make_transform,
    quaternion_xyzw_to_matrix,
)
from camera_calibration.model_geometry import (
    DEFAULT_HAND_TO_TAG_QUATERNION_XYZW,
    DEFAULT_HAND_TO_TAG_XYZ,
    TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW,
    TAG_BODY_TO_PRINTED_TAG_XYZ,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_SCENE = REPO_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand.xml"


def _numbers(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(component) for component in value.split())


def _body_transform(body: ET.Element) -> np.ndarray:
    xyz = _numbers(body.get("pos"), (0.0, 0.0, 0.0))
    w, x, y, z = _numbers(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    return make_transform(quaternion_xyzw_to_matrix((x, y, z, w)), xyz)


def test_default_flange_to_printed_tag_matches_calibration_mujoco_model():
    root = ET.parse(ROBOT_SCENE).getroot()
    flange = root.find(".//body[@name='fr3_link8']")
    assert flange is not None
    hand = flange.find("body[@name='hand_base_link']")
    assert hand is not None
    tag = hand.find("body[@name='apriltag_0']")
    assert tag is not None

    model_flange_to_printed_tag = (
        _body_transform(hand)
        @ _body_transform(tag)
        @ make_transform(
            quaternion_xyzw_to_matrix(TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW),
            TAG_BODY_TO_PRINTED_TAG_XYZ,
        )
    )
    calibration_flange_to_printed_tag = make_transform(
        quaternion_xyzw_to_matrix(DEFAULT_HAND_TO_TAG_QUATERNION_XYZW),
        DEFAULT_HAND_TO_TAG_XYZ,
    )

    np.testing.assert_allclose(
        calibration_flange_to_printed_tag,
        model_flange_to_printed_tag,
        atol=1.0e-12,
    )


def test_calibration_model_has_physical_flange_clocking_and_adapter():
    root = ET.parse(ROBOT_SCENE).getroot()
    flange = root.find(".//body[@name='fr3_link8']")
    assert flange is not None
    hand = flange.find("body[@name='hand_base_link']")
    assert hand is not None

    assert _numbers(hand.get("pos"), ()) == (0.0, 0.0, 0.010)
    # MuJoCo quaternion order is wxyz; (0, 0, 0, 1) is yaw=pi.
    assert _numbers(hand.get("quat"), ()) == (0.0, 0.0, 0.0, 1.0)


def test_calibration_target_is_flat_2p3_mm_plate_at_measured_corner():
    root = ET.parse(ROBOT_SCENE).getroot()
    tag = root.find(".//body[@name='apriltag_0']")
    assert tag is not None

    # The first black pixel is the upper-left corner of the 40 mm black square
    # in the plate body's UV orientation.
    hand_to_first_black = _body_transform(tag) @ np.asarray(
        (-0.020, -0.020, 0.0023, 1.0)
    )
    np.testing.assert_allclose(hand_to_first_black[1], -0.020545646, atol=1.0e-12)
    np.testing.assert_allclose(hand_to_first_black[2], 0.080, atol=1.0e-12)

    plate_path = ROBOT_SCENE.parent / "hand" / "apriltag_36h11_id0_dorsal.obj"
    vertices = np.asarray(
        [
            tuple(float(value) for value in line.split()[1:])
            for line in plate_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]
    )
    assert vertices.shape == (8, 3)
    np.testing.assert_allclose(vertices.min(axis=0), (-0.025, -0.025, 0.0))
    np.testing.assert_allclose(vertices.max(axis=0), (0.025, 0.025, 0.0023))
