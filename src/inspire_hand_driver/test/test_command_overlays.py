"""Tests for final per-DOF command calibration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver import command_overlays  # noqa: E402
from inspire_hand_driver import kinematics  # noqa: E402


@pytest.mark.parametrize(
    ("commanded", "expected"),
    (
        (0.0, 0.25),
        (0.1, 0.325),
        (0.25, 0.4375),
        (0.5, 0.625),
        (0.75, 0.8125),
        (1.0, 1.0),
    ),
)
def test_thumb_abduction_is_rescaled_onto_point_25_through_one(commanded, expected):
    actual = command_overlays.apply_open_ratio_overlay(
        command_overlays.THUMB_ABDUCTION_DOF, commanded
    )
    assert actual == pytest.approx(expected)


def test_thumb_abduction_rescale_keeps_every_command_distinct():
    """A floor collapsed the bottom of the range onto one pose; this must not."""
    commanded = [index / 20.0 for index in range(21)]
    physical = [
        command_overlays.apply_open_ratio_overlay(
            command_overlays.THUMB_ABDUCTION_DOF, ratio
        )
        for ratio in commanded
    ]

    assert len(set(physical)) == len(commanded)
    assert all(b > a for a, b in zip(physical, physical[1:]))
    assert min(physical) == pytest.approx(
        command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    )
    assert max(physical) == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", (0.0, 0.1, 0.25, 0.5, 0.75, 1.0))
def test_thumb_abduction_overlay_inverts_back_to_the_original_command(ratio):
    physical = command_overlays.apply_open_ratio_overlay(
        command_overlays.THUMB_ABDUCTION_DOF, ratio
    )
    assert command_overlays.invert_open_ratio_overlay(
        command_overlays.THUMB_ABDUCTION_DOF, physical
    ) == pytest.approx(ratio)


def test_every_other_dof_is_unchanged():
    for index in range(len(kinematics.DOFS)):
        if index == command_overlays.THUMB_ABDUCTION_DOF:
            continue
        for ratio in (0.0, 0.1, 0.25, 0.5, 1.0):
            assert command_overlays.apply_open_ratio_overlay(index, ratio) == ratio
            assert command_overlays.invert_open_ratio_overlay(index, ratio) == ratio


def test_thumb_abduction_identity_is_resolved_from_the_kinematics_table():
    assert command_overlays.THUMB_ABDUCTION_DOF == kinematics.dof_index(
        "thumb_proximal_yaw_joint"
    )
    assert kinematics.DOFS[command_overlays.THUMB_ABDUCTION_DOF].channel == "6"
