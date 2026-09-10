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
    rz1 = flange.find("body[@name='apriltag_rz1']")
    assert rz1 is not None
    tag = rz1.find("body[@name='apriltag_0']")
    assert tag is not None

    model_flange_to_printed_tag = (
        _body_transform(rz1)
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


def test_calibration_target_is_flat_2p3_mm_plate_at_measured_holder_pose():
    root = ET.parse(ROBOT_SCENE).getroot()
    flange = root.find(".//body[@name='fr3_link8']")
    assert flange is not None
    rz1 = flange.find("body[@name='apriltag_rz1']")
    assert rz1 is not None
    tag = rz1.find("body[@name='apriltag_0']")
    assert tag is not None

    flange_to_printed_tag = (
        _body_transform(rz1)
        @ _body_transform(tag)
        @ make_transform(
            quaternion_xyzw_to_matrix(TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW),
            TAG_BODY_TO_PRINTED_TAG_XYZ,
        )
    )
    expected_flange_to_printed_tag = make_transform(
        quaternion_xyzw_to_matrix(
            (
                -0.270598050073099,
                0.653281482438188,
                -0.653281482438188,
                0.270598050073099,
            )
        ),
        (0.0417193000900063, -0.0417193000900063, 0.035),
    )
    np.testing.assert_allclose(
        flange_to_printed_tag,
        expected_flange_to_printed_tag,
        atol=1.0e-12,
    )

    # Rz1 is a pure -45-degree rotation about the parent flange Z. The centre
    # translation is expressed in Rz1, not directly in fr3_link8.
    assert _numbers(rz1.get("pos"), (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert _numbers(rz1.get("quat"), ()) == (
        0.923879532511287,
        0.0,
        0.0,
        -0.382683432365090,
    )
    assert _numbers(tag.get("pos"), ()) == (0.059, 0.0, 0.035)
    assert rz1 in list(flange)
    assert tag in list(rz1)

    # Rz1 rotates the horizontal tag axes and normal by -45 degrees without
    # changing tag +Y, which stays parallel to flange -Z.
    np.testing.assert_allclose(
        flange_to_printed_tag[:3, 2],
        (np.sqrt(0.5), -np.sqrt(0.5), 0.0),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        flange_to_printed_tag[:3, 0],
        (-np.sqrt(0.5), -np.sqrt(0.5), 0.0),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        flange_to_printed_tag[:3, 1], (0.0, 0.0, -1.0), atol=1.0e-12
    )

    geom = tag.find("geom[@name='apriltag_36h11_id0']")
    assert geom is not None
    assert _numbers(geom.get("pos"), ()) == (0.0, 0.0, -0.0023)

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
