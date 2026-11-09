"""Tests for final per-DOF command calibration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver import command_overlays  # noqa: E402
from inspire_hand_driver import kinematics  # noqa: E402

YAW = command_overlays.THUMB_ABDUCTION_DOF
FLEXION = command_overlays.THUMB_FLEXION_DOF


def register_for(dof, model_rad):
    ratio = command_overlays.apply_open_ratio_overlay(
        dof, kinematics.rad_to_open_ratio(dof, model_rad)
    )
    return kinematics.open_ratio_to_rad(dof, ratio)


@pytest.mark.parametrize("model_rad", (0.3, 0.6, 0.9, 1.185))
def test_thumb_yaw_is_a_static_shift_from_its_register(model_rad):
    assert register_for(YAW, model_rad) == pytest.approx(model_rad - 0.27)


def test_thumb_yaw_below_the_offset_rests_on_the_open_stop():
    assert register_for(YAW, 0.1) == pytest.approx(0.0)
    assert command_overlays.apply_open_ratio_overlay(YAW, 1.0) == pytest.approx(1.0)


def test_thumb_yaw_at_the_closed_limit_stops_short_of_the_register_limit():
    upper = kinematics.DOFS[YAW].upper
    assert register_for(YAW, upper) == pytest.approx(upper - 0.27)
    assert command_overlays.apply_open_ratio_overlay(YAW, 0.0) == pytest.approx(
        command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    )


def test_thumb_flexion_reaches_the_exp19_reset_at_the_fitted_register():
    assert register_for(FLEXION, 0.0506) == pytest.approx((0.0506 + 0.03) / 0.73)
    assert register_for(FLEXION, 0.0506) == pytest.approx(0.1104, abs=1.0e-3)


def test_thumb_flexion_past_the_real_range_stops_at_the_register_limit():
    assert register_for(FLEXION, 0.48) == pytest.approx(kinematics.DOFS[FLEXION].upper)


@pytest.mark.parametrize(
    ("dof", "ratio"),
    ((YAW, 0.0), (YAW, 0.25), (YAW, 0.5), (YAW, 0.75), (FLEXION, 0.4), (FLEXION, 0.5), (FLEXION, 1.0)),
)
def test_the_calibration_inverts_back_to_the_original_command(dof, ratio):
    register = command_overlays.apply_open_ratio_overlay(dof, ratio)
    assert command_overlays.invert_open_ratio_overlay(dof, register) == pytest.approx(ratio)


def test_registers_outside_the_models_reach_invert_outside_the_range():
    assert command_overlays.invert_open_ratio_overlay(FLEXION, 1.0) > 1.0
    assert command_overlays.invert_open_ratio_overlay(YAW, 0.0) < 0.0


@pytest.mark.parametrize("dof", (YAW, FLEXION))
def test_commands_stay_monotonic(dof):
    registers = [
        command_overlays.apply_open_ratio_overlay(dof, index / 20.0) for index in range(21)
    ]
    assert all(b >= a for a, b in zip(registers, registers[1:]))


def test_rate_gain_follows_each_fit():
    assert command_overlays.model_rate_gain(YAW) == 1.0
    assert command_overlays.model_rate_gain(FLEXION) == pytest.approx(0.73)
    assert command_overlays.model_rate_gain(FLEXION, enabled=False) == 1.0


def test_every_other_dof_is_unchanged():
    for index in range(len(kinematics.DOFS)):
        if index in (YAW, FLEXION):
            continue
        for ratio in (0.0, 0.1, 0.25, 0.5, 1.0):
            assert command_overlays.apply_open_ratio_overlay(index, ratio) == ratio
            assert command_overlays.invert_open_ratio_overlay(index, ratio) == ratio
        assert command_overlays.model_rate_gain(index) == 1.0


def test_thumb_identities_are_resolved_from_the_kinematics_table():
    assert YAW == kinematics.dof_index("thumb_proximal_yaw_joint")
    assert kinematics.DOFS[YAW].channel == "6"
    assert FLEXION == kinematics.dof_index("thumb_proximal_pitch_joint")
    assert kinematics.DOFS[FLEXION].channel == "5"


@pytest.mark.parametrize("dof", (YAW, FLEXION))
def test_the_calibration_can_be_switched_off_for_a_simulated_hand(dof):
    for ratio in (0.0, 0.3, 1.0):
        assert command_overlays.apply_open_ratio_overlay(dof, ratio, enabled=False) == ratio
        assert command_overlays.invert_open_ratio_overlay(dof, ratio, enabled=False) == ratio
