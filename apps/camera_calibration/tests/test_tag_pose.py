"""Corner order, averaging, and the tag pose itself.

The corner-order test is the point of this file. The native AprilTag detector
and OpenCV's IPPE_SQUARE solver number a square's corners in opposite
directions, and getting it wrong costs nothing visible: the reprojection error
stays near zero because a mirrored planar point set is just the same tag rotated
180 deg about its own x axis. The only symptom is a reported tag frame that is
upside down, so it needs a test rather than an inspection.
"""

import cv2
import numpy as np
import pytest

from camera_calibration.calibration_math import invert_transform, make_transform, rotation_angle_deg
from camera_calibration.tag_pose import (
    TagPoseError,
    average_corners,
    estimate_tag_pose,
    object_corners,
    project_tag_corners,
    tag_view_angle_deg,
    to_ippe_order,
)


CAMERA_MATRIX = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
DISTORTION = np.zeros(5)
TAG_SIZE = 0.040


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def true_pose():
    return make_transform(axis_angle([0.3, 1.0, 0.2], 0.35), [0.05, -0.03, 0.60])


def detector_corners(camera_to_tag):
    """What the native detector would report: IPPE order, reversed."""
    return project_tag_corners(camera_to_tag, CAMERA_MATRIX, DISTORTION, TAG_SIZE)[::-1]


def test_detector_corners_round_trip_to_the_true_pose():
    expected = true_pose()
    corners = to_ippe_order(detector_corners(expected))
    recovered, error = estimate_tag_pose(corners, CAMERA_MATRIX, DISTORTION, TAG_SIZE)
    assert error < 1e-6
    np.testing.assert_allclose(recovered, expected, atol=1e-9)


def test_skipping_the_reorder_flips_the_tag_by_180_degrees_about_x():
    expected = true_pose()
    wrong, error = estimate_tag_pose(
        detector_corners(expected), CAMERA_MATRIX, DISTORTION, TAG_SIZE
    )
    # It fits the image just as well, which is exactly why it went unnoticed.
    assert error < 1e-6
    difference = invert_transform(expected) @ wrong
    assert rotation_angle_deg(difference[:3, :3]) == pytest.approx(180.0, abs=1e-6)
    np.testing.assert_allclose(np.abs(difference[:3, :3] @ [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0],
                               atol=1e-9)


def test_object_corners_match_the_order_opencv_documents():
    half = TAG_SIZE * 0.5
    np.testing.assert_allclose(
        object_corners(TAG_SIZE),
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
    )
    with pytest.raises(TagPoseError):
        object_corners(0.0)


def test_averaging_a_burst_reduces_noise_and_reports_the_scatter():
    expected = true_pose()
    clean = to_ippe_order(detector_corners(expected))
    random = np.random.default_rng(7)
    burst = [clean + random.normal(scale=0.2, size=(4, 2)) for _ in range(40)]
    mean, scatter = average_corners(burst)
    assert scatter == pytest.approx(0.2, rel=0.4)
    assert np.abs(mean - clean).max() < np.abs(burst[0] - clean).max()
    recovered, _ = estimate_tag_pose(mean, CAMERA_MATRIX, DISTORTION, TAG_SIZE)
    single, _ = estimate_tag_pose(burst[0], CAMERA_MATRIX, DISTORTION, TAG_SIZE)
    averaged_error = np.linalg.norm(recovered[:3, 3] - expected[:3, 3])
    single_error = np.linalg.norm(single[:3, 3] - expected[:3, 3])
    assert averaged_error < single_error
    with pytest.raises(TagPoseError):
        average_corners([])


def test_view_angle_is_zero_for_a_tag_facing_the_camera():
    facing = make_transform(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.5])
    assert tag_view_angle_deg(facing) == pytest.approx(0.0, abs=1e-9)
    tilted = make_transform(np.diag([1.0, -1.0, -1.0]) @ axis_angle([1, 0, 0], np.radians(30.0)),
                            [0.0, 0.0, 0.5])
    assert tag_view_angle_deg(tilted) == pytest.approx(30.0, abs=1e-6)
