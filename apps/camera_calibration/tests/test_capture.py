"""Whether a burst of frames is accepted as a standstill.

Every threshold here exists because the hand-guided run had no such check: it
took one frame per sample while the arm was moving, and nothing in the data said
so. The corner scatter and the joint spread are that missing evidence.
"""

import numpy as np
import pytest

from camera_calibration.capture import CaptureError, summarize_burst


CORNERS = np.array([[300.0, 200.0], [340.0, 200.0], [340.0, 240.0], [300.0, 240.0]])
JOINTS = np.tile(np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8]), (10, 1))


def burst(count=10, scatter=0.05, error=0.3, seed=1):
    random = np.random.default_rng(seed)
    return [
        (100.0 + index / 30.0, CORNERS + random.normal(scale=scatter, size=(4, 2)), error)
        for index in range(count)
    ]


def test_a_still_burst_is_averaged():
    summary = summarize_burst(burst(), JOINTS)
    assert summary.frames_used == 10
    assert summary.frames_seen == 10
    np.testing.assert_allclose(summary.corners, CORNERS, atol=0.1)
    np.testing.assert_allclose(summary.joint_positions, JOINTS[0], atol=1e-12)
    assert summary.corner_std_px < 0.1
    assert summary.joint_spread_rad == 0.0
    assert summary.reprojection_error_px == pytest.approx(0.3)
    assert summary.first_stamp_s < summary.last_stamp_s


def test_a_burst_taken_while_the_tag_still_moves_is_refused():
    with pytest.raises(CaptureError, match="corners moved"):
        summarize_burst(burst(scatter=1.5), JOINTS)


def test_a_burst_taken_while_the_joints_still_creep_is_refused():
    creeping = JOINTS + np.linspace(0.0, 0.01, len(JOINTS))[:, None]
    with pytest.raises(CaptureError, match="joints moved"):
        summarize_burst(burst(), creeping)


def test_frames_with_a_poor_reprojection_are_dropped():
    frames = burst(count=10)
    for index in (2, 5):
        stamp, corners, _ = frames[index]
        frames[index] = (stamp, corners, 4.0)
    summary = summarize_burst(frames, JOINTS, minimum_frames=6)
    assert summary.frames_used == 8
    assert summary.frames_seen == 10


def test_too_few_clean_frames_is_refused():
    frames = [(stamp, corners, 9.0) for stamp, corners, _ in burst()]
    with pytest.raises(CaptureError, match="clean tag detection"):
        summarize_burst(frames, JOINTS)


def test_a_burst_with_no_joint_states_is_refused():
    with pytest.raises(CaptureError, match="no joint states"):
        summarize_burst(burst(), np.zeros((0, 7)))
