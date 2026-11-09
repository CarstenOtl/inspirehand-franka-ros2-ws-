"""Tests for the thumb yaw register scale."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver import command_overlays  # noqa: E402
from inspire_hand_driver import kinematics  # noqa: E402

YAW = command_overlays.THUMB_ABDUCTION_DOF


def test_thumb_yaw_command_zero_is_register_250_and_one_is_register_1000():
    assert command_overlays.apply_open_ratio_overlay(YAW, 0.0) * 1000 == pytest.approx(250.0)
    assert command_overlays.apply_open_ratio_overlay(YAW, 1.0) * 1000 == pytest.approx(1000.0)


@pytest.mark.parametrize("ratio", (0.1, 0.25, 0.5, 0.75, 0.9))
def test_thumb_yaw_commands_between_are_scaled_linearly(ratio):
    assert command_overlays.apply_open_ratio_overlay(YAW, ratio) == pytest.approx(
        0.25 + ratio * 0.75
    )


@pytest.mark.parametrize("ratio", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_the_scale_inverts_back_to_the_original_command(ratio):
    register = command_overlays.apply_open_ratio_overlay(YAW, ratio)
    assert command_overlays.invert_open_ratio_overlay(YAW, register) == pytest.approx(ratio)


def test_registers_outside_the_command_range_invert_outside_it():
    assert command_overlays.invert_open_ratio_overlay(YAW, 0.0) < 0.0


def test_rate_gain():
    assert command_overlays.model_rate_gain(YAW) == pytest.approx(1.0 / 0.75)
    assert command_overlays.model_rate_gain(YAW, enabled=False) == 1.0


def test_every_other_dof_is_unchanged():
    for index in range(len(kinematics.DOFS)):
        if index == YAW:
            continue
        for ratio in (0.0, 0.1, 0.25, 0.5, 1.0):
            assert command_overlays.apply_open_ratio_overlay(index, ratio) == ratio
            assert command_overlays.invert_open_ratio_overlay(index, ratio) == ratio
        assert command_overlays.model_rate_gain(index) == 1.0


def test_disabled_scale_is_the_identity():
    assert command_overlays.apply_open_ratio_overlay(YAW, 0.0, enabled=False) == 0.0
    assert command_overlays.invert_open_ratio_overlay(YAW, 0.25, enabled=False) == 0.25


def test_thumb_yaw_is_resolved_from_the_kinematics_table():
    assert YAW == kinematics.dof_index("thumb_proximal_yaw_joint")
    assert kinematics.DOFS[YAW].channel == "6"
