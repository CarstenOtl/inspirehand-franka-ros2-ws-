"""The policy observes the driver's fitted hand model unchanged."""

import numpy as np
import pytest

from policy_rollout.hardware import HAND_JOINTS, physical_hand_state_to_policy


def test_the_policy_joints_are_read_straight_from_the_driver():
    positions = np.array([0.3, 0.4, 0.5, 0.2147, 0.0506, 1.185])
    velocities = np.array([0.0, 0.0, 0.0, 0.3, 0.2, 0.1])
    q, dq = physical_hand_state_to_policy(positions, velocities)
    assert q == pytest.approx([1.185, 0.0506, 0.2147])
    assert dq == pytest.approx([0.1, 0.2, 0.3])


def test_a_wrong_length_or_non_finite_state_is_refused():
    with pytest.raises(ValueError):
        physical_hand_state_to_policy(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError):
        physical_hand_state_to_policy(np.full(len(HAND_JOINTS), np.nan), np.zeros(len(HAND_JOINTS)))
