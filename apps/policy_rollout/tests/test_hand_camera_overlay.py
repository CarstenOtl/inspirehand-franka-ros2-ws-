"""The fingertip overlay's projection and its reading of the driver's hand state."""

import numpy as np
import pytest

from utils.hand_camera_overlay import measured_depth, policy_hand_joints, project


CAMERA = (600.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)


def test_a_point_on_the_optical_axis_lands_on_the_principal_point():
    uv, depth = project([[1.0, 2.0, 3.5]], np.eye(3), (1.0, 2.0, 3.0), CAMERA)
    assert uv[0] == pytest.approx((320.0, 240.0))
    assert depth[0] == pytest.approx(0.5)


def test_projection_goes_through_the_camera_rotation():
    # A camera whose optical axis is the base x axis: x_cam = -y_base, y_cam = -z_base.
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    uv, depth = project([[2.0, -0.1, 0.0]], rotation, (0.0, 0.0, 0.0), CAMERA)
    assert depth[0] == pytest.approx(2.0)
    assert uv[0] == pytest.approx((320.0 + 600.0 * 0.05, 240.0))


def test_measured_depth_ignores_holes():
    depth = np.zeros((20, 20))
    depth[9:12, 9:12] = 0.5
    assert measured_depth(depth, np.array([10.0, 10.0])) == pytest.approx(0.5)
    assert np.isnan(measured_depth(np.zeros((20, 20)), np.array([10.0, 10.0])))
    assert np.isnan(measured_depth(depth, np.array([-5.0, 10.0])))


def test_policy_hand_removes_the_drivers_thumb_overlay():
    # The registers that put the real thumb on the exp19 reset (1.185, 0.0506).
    reported = {
        "pinky_proximal_joint": 1.33,
        "ring_proximal_joint": 1.33,
        "middle_proximal_joint": 1.33,
        "index_proximal_joint": 0.2147,
        "thumb_proximal_pitch_joint": 0.1104,
        "thumb_proximal_yaw_joint": 0.9151,
    }
    with_overlay = policy_hand_joints(reported, thumb_overlay=True)
    assert with_overlay["thumb_proximal_yaw_joint"] == pytest.approx(1.185, abs=1.0e-3)
    assert with_overlay["thumb_proximal_pitch_joint"] == pytest.approx(0.0506, abs=1.0e-3)
    assert with_overlay["index_proximal_joint"] == pytest.approx(0.2147)
    assert with_overlay["index_intermediate_joint"] > 0.0

    without = policy_hand_joints(reported, thumb_overlay=False)
    assert without["thumb_proximal_yaw_joint"] == pytest.approx(0.9151)
