import numpy as np
import pytest

pytest.importorskip("inspire_hand_driver")
from inspire_hand_driver import command_overlays as co  # noqa: E402
from inspire_hand_driver import kinematics as kin  # noqa: E402

from policy_rollout.hardware import (  # noqa: E402
    HAND_JOINTS,
    SUPPORT_FINGER_INDICES,
    home_hand_in_stages,
    physical_hand_state_to_policy,
)

THUMB = kin.dof_index(co.THUMB_ABDUCTION_JOINT)


def test_homing_curls_the_support_fingers_before_the_index_and_thumb():
    calls = []

    class Client:
        def command_hand(self, target, indices=None):
            calls.append(("command", indices))

        def wait_for_hand(self, target, timeout, tolerance, indices=None):
            calls.append(("wait", indices))

    home_hand_in_stages(Client(), np.zeros(len(HAND_JOINTS)), 5.0, 0.01)

    assert calls == [
        ("command", SUPPORT_FINGER_INDICES),
        ("wait", SUPPORT_FINGER_INDICES),
        ("command", None),
        ("wait", None),
    ]
    assert sorted(HAND_JOINTS[i] for i in SUPPORT_FINGER_INDICES) == [
        "middle_proximal_joint",
        "pinky_proximal_joint",
        "ring_proximal_joint",
    ]


def _state(physical_ratio):
    q = np.zeros(len(HAND_JOINTS))
    q[HAND_JOINTS.index(co.THUMB_ABDUCTION_JOINT)] = kin.open_ratio_to_rad(THUMB, physical_ratio)
    return q, np.zeros(len(HAND_JOINTS))


def test_a_thumb_a_few_counts_past_its_stop_reads_as_the_stop():
    # 2026-10-26: commanded onto the 1.246 rad limit, the feedback settled just
    # below the overlay's zero (physical open ratio 0.25).
    q_policy, _ = physical_hand_state_to_policy(*_state(co.THUMB_ABDUCTION_ZERO_OPEN_RATIO - 0.003))
    assert q_policy[0] == pytest.approx(kin.open_ratio_to_rad(THUMB, 0.0))


def test_a_thumb_far_outside_the_overlaid_range_is_still_a_fault():
    with pytest.raises(ValueError, match="outside the overlaid policy range"):
        physical_hand_state_to_policy(*_state(0.1))


def test_the_exp19_reset_registers_read_as_the_exp19_reset():
    q = np.zeros(len(HAND_JOINTS))
    q[HAND_JOINTS.index(co.THUMB_ABDUCTION_JOINT)] = 1.185 - 0.27
    q[HAND_JOINTS.index(co.THUMB_FLEXION_JOINT)] = (0.0506 + 0.03) / 0.73
    q_policy, _ = physical_hand_state_to_policy(q, np.zeros(len(HAND_JOINTS)))
    assert q_policy[0] == pytest.approx(1.185)
    assert q_policy[1] == pytest.approx(0.0506)


def test_a_flexion_register_more_open_than_the_model_reads_as_the_models_zero():
    # The hand as it powers up, before the driver has written a calibrated command.
    q_policy, _ = physical_hand_state_to_policy(np.zeros(len(HAND_JOINTS)), np.zeros(len(HAND_JOINTS)))
    assert q_policy[1] == pytest.approx(0.0)
    assert q_policy[0] == pytest.approx(0.27)


def test_only_the_flexion_rate_is_scaled():
    rates = np.zeros(len(HAND_JOINTS))
    rates[HAND_JOINTS.index(co.THUMB_ABDUCTION_JOINT)] = 0.4
    rates[HAND_JOINTS.index(co.THUMB_FLEXION_JOINT)] = 0.2
    _, dq_policy = physical_hand_state_to_policy(np.zeros(len(HAND_JOINTS)), rates)
    assert dq_policy[0] == pytest.approx(0.4)
    assert dq_policy[1] == pytest.approx(0.2 * 0.73)
