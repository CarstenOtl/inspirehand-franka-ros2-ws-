import numpy as np
import pytest

pytest.importorskip("inspire_hand_driver")
from inspire_hand_driver import command_overlays as co  # noqa: E402
from inspire_hand_driver import kinematics as kin  # noqa: E402

from policy_rollout.hardware import HAND_JOINTS, physical_hand_state_to_policy  # noqa: E402

THUMB = kin.dof_index(co.THUMB_ABDUCTION_JOINT)


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
