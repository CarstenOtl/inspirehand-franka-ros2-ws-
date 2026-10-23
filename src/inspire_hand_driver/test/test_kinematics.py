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
    "pinky_intermediate_joint": (1.1169, 0.0),
    "ring_intermediate_joint": (1.1169, 0.0),
    "middle_intermediate_joint": (1.1169, 0.0),
    "index_intermediate_joint": (1.1169, 0.0),
    "thumb_intermediate_joint": (1.1425, 0.0),
    "thumb_distal_joint": (0.7508, 0.0),
}

URDF_LIMITS = {
    "pinky_intermediate_joint": (0.0, 1.527),
    "ring_intermediate_joint": (0.0, 1.527),
    "middle_intermediate_joint": (0.0, 1.527),
    "index_intermediate_joint": (0.0, 1.527),
    "thumb_intermediate_joint": (0.0, 0.3578),
    "thumb_distal_joint": (0.0, 0.2775),
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


def test_the_thumb_distal_follows_the_intermediate_not_the_pitch():
    couplings = {c.joint: c for d in kin.DOFS for c in d.couplings}
    assert couplings["thumb_distal_joint"].source == "thumb_intermediate_joint"
    assert couplings["thumb_intermediate_joint"].source == ""


def test_the_thumb_intermediate_clamps_and_the_distal_stops_with_it():
    # 1.1425 * 0.48 = 0.548 would overshoot the intermediate's 0.3578 limit, so
    # it clamps from about two thirds closed. The distal follows the clamped
    # intermediate (0.7508 * 0.3578 = 0.26864), short of its own 0.2775 limit.
    fully_closed = dict(zip(kin.ALL_JOINTS, kin.joint_positions([0.0] * 6)))
    assert fully_closed["thumb_intermediate_joint"] == pytest.approx(0.3578, abs=1e-12)
    assert fully_closed["thumb_distal_joint"] == pytest.approx(0.7508 * 0.3578, abs=1e-12)


def test_the_thumb_followers_stop_moving_once_the_intermediate_clamps():
    thumb = kin.dof_index("thumb_proximal_pitch_joint")
    ratios = [1.0] * 6
    ratios[thumb] = 0.1  # pitch 0.432 rad, past the clamp at 0.313
    rates = kin.joint_velocities(ratios, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    by_name = dict(zip(kin.ALL_JOINTS, rates))
    assert by_name["thumb_intermediate_joint"] == 0.0
    assert by_name["thumb_distal_joint"] == 0.0
    ratios[thumb] = 0.9  # pitch 0.048 rad, well before it
    by_name = dict(zip(kin.ALL_JOINTS, kin.joint_velocities(ratios, [0, 0, 0, 0, 1.0, 0])))
    assert by_name["thumb_intermediate_joint"] == pytest.approx(1.1425)
    assert by_name["thumb_distal_joint"] == pytest.approx(1.1425 * 0.7508)


def test_the_finger_followers_stop_short_of_their_limit():
    # 1.1169 * 1.333 = 1.48883, about 2.2 deg inside the 1.527 limit.
    # Not a bug to "fix" by widening the multiplier: the limit is loose, and the
    # multiplier is what the linkage actually does.
    fully_closed = dict(zip(kin.ALL_JOINTS, kin.joint_positions([0.0] * 6)))
    assert fully_closed["index_intermediate_joint"] == pytest.approx(1.48883, abs=1e-5)


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
