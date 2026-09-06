"""Tests for the six-channel <-> twelve-joint mapping.

The two failure modes worth guarding against are both silent and both bad:
commanding the wrong finger (a channel/joint index slip), and driving a joint
the wrong way (an open/closed sign flip that closes the hand on a command to
open it).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.protocol import CHANNEL_IDS, DOF_ORDER  # noqa: E402


def test_dofs_follow_the_register_order():
    # The register blocks are positional: index i of every 6-register block is
    # DOFS[i]. A reordering here would drive the wrong fingers, silently.
    assert tuple(d.channel for d in kin.DOFS) == CHANNEL_IDS
    assert tuple(d.name for d in kin.DOFS) == DOF_ORDER


def test_joint_names_are_unique_and_complete():
    assert len(kin.DRIVEN_JOINTS) == 6
    assert len(kin.PASSIVE_JOINTS) == 6
    assert len(set(kin.ALL_JOINTS)) == 12


def test_open_ratio_one_puts_every_joint_at_its_open_limit():
    for i, dof in enumerate(kin.DOFS):
        assert kin.open_ratio_to_rad(i, 1.0) == pytest.approx(dof.lower)


def test_open_ratio_zero_puts_every_joint_at_its_closed_limit():
    for i, dof in enumerate(kin.DOFS):
        assert kin.open_ratio_to_rad(i, 0.0) == pytest.approx(dof.upper)


def test_closing_increases_every_driven_joint_angle():
    # Every driven joint has its open pose at the lower limit, so closing must
    # increase the angle. A DOF that ran the other way would need its own sign.
    for i in range(len(kin.DOFS)):
        assert kin.open_ratio_to_rad(i, 0.0) > kin.open_ratio_to_rad(i, 1.0)


def test_ratio_and_radian_conversions_round_trip():
    for i in range(len(kin.DOFS)):
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert kin.rad_to_open_ratio(i, kin.open_ratio_to_rad(i, ratio)) == pytest.approx(
                ratio
            )


def test_open_ratio_is_clamped_outside_the_unit_interval():
    assert kin.open_ratio_to_rad(0, 1.5) == pytest.approx(kin.DOFS[0].lower)
    assert kin.open_ratio_to_rad(0, -0.5) == pytest.approx(kin.DOFS[0].upper)


# The authoritative coupling values and follower limits, transcribed from the
# <mimic> and <limit> tags of the URDF that inspire_hand_description vendors.
# inspire_hand_description has its own test asserting the shipped URDF still
# says this; here they pin the driver's copy of the same numbers.
URDF_MIMICS = {
    "pinky_intermediate_joint": (1.06399, -0.04545),
    "ring_intermediate_joint": (1.06399, -0.04545),
    "middle_intermediate_joint": (1.06399, -0.04545),
    "index_intermediate_joint": (1.06399, -0.04545),
    "thumb_intermediate_joint": (1.334, 0.0),
    "thumb_distal_joint": (0.667, 0.0),
}

URDF_LIMITS = {
    "pinky_intermediate_joint": (-0.04545, 1.56),
    "ring_intermediate_joint": (-0.04545, 1.56),
    "middle_intermediate_joint": (-0.04545, 1.56),
    "index_intermediate_joint": (-0.04545, 1.56),
    "thumb_intermediate_joint": (0.0, 0.8),
    "thumb_distal_joint": (0.0, 0.4),
}


def test_couplings_match_the_urdf_mimic_tags():
    couplings = {c.joint: c for d in kin.DOFS for c in d.couplings}
    assert set(couplings) == set(URDF_MIMICS)
    for joint, (multiplier, offset) in URDF_MIMICS.items():
        assert couplings[joint].multiplier == pytest.approx(multiplier)
        assert couplings[joint].offset == pytest.approx(offset)


def test_coupling_limits_match_the_urdf_limits():
    couplings = {c.joint: c for d in kin.DOFS for c in d.couplings}
    for joint, (lower, upper) in URDF_LIMITS.items():
        assert couplings[joint].lower == pytest.approx(lower)
        assert couplings[joint].upper == pytest.approx(upper)


def test_followers_sit_at_their_open_limit_when_the_hand_is_open():
    fully_open = dict(zip(kin.ALL_JOINTS, kin.joint_positions([1.0] * 6)))
    for joint, (lower, _) in URDF_LIMITS.items():
        assert fully_open[joint] == pytest.approx(lower, abs=1e-6)


def test_the_thumb_followers_are_clamped_at_the_closed_pose():
    # Upstream rounds the thumb multipliers (1.334 and 0.667 rather than 4/3
    # and 2/3), which overshoots both limits by 4e-4 rad. Clamping is what keeps
    # TF and MuJoCo agreeing about where the thumb is.
    fully_closed = dict(zip(kin.ALL_JOINTS, kin.joint_positions([0.0] * 6)))
    assert fully_closed["thumb_intermediate_joint"] == pytest.approx(0.8, abs=1e-12)
    assert fully_closed["thumb_distal_joint"] == pytest.approx(0.4, abs=1e-12)


def test_the_finger_followers_stop_short_of_their_limit():
    # 1.06399 * 1.47 - 0.04545 = 1.51862, about 2.4 deg inside the 1.56 limit.
    # Not a bug to "fix" by widening the multiplier: the limit is loose, and the
    # multiplier is what the linkage actually does.
    fully_closed = dict(zip(kin.ALL_JOINTS, kin.joint_positions([0.0] * 6)))
    assert fully_closed["index_intermediate_joint"] == pytest.approx(1.51862, abs=1e-5)


def test_joint_positions_stay_within_limits_across_the_whole_range():
    limits = dict(URDF_LIMITS)
    limits.update({d.joint: (d.lower, d.upper) for d in kin.DOFS})
    for step in range(0, 21):
        ratios = [step / 20.0] * 6
        for joint, value in zip(kin.ALL_JOINTS, kin.joint_positions(ratios)):
            lower, upper = limits[joint]
            assert lower <= value <= upper, f"{joint} out of range at ratio {ratios[0]}"


def test_dof_index_accepts_channel_ids_and_driven_joint_names():
    for i, dof in enumerate(kin.DOFS):
        assert kin.dof_index(dof.channel) == i
        assert kin.dof_index(dof.joint) == i


def test_passive_joints_are_not_addressable():
    # Commanding a follower independently of its driver is not something the
    # hardware can do, so resolving one must fail rather than silently move
    # the joint that drives it.
    for joint in kin.PASSIVE_JOINTS:
        with pytest.raises(KeyError):
            kin.dof_index(joint)


def test_joint_positions_rejects_the_wrong_number_of_ratios():
    with pytest.raises(ValueError):
        kin.joint_positions([1.0] * 5)
