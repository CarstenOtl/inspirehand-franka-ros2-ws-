"""Final command overlays applied after the hand's normal command scaling.

Keep task-specific calibration here rather than changing the generic
open-ratio/radian kinematics. Every driver command path passes through this
module immediately before conversion to the hand's integer registers.
"""

from __future__ import annotations

from . import kinematics


THUMB_ABDUCTION_JOINT = "thumb_proximal_yaw_joint"
THUMB_ABDUCTION_DOF = kinematics.dof_index(THUMB_ABDUCTION_JOINT)

# Physically, thumb abduction at open ratio 0 carries the thumb past the palm
# plane, so the bottom of the commanded range is unusable. This is the open
# ratio that a command of 0.0 is remapped onto: the thumb's new zero.
THUMB_ABDUCTION_ZERO_OPEN_RATIO = 0.25


def apply_open_ratio_overlay(index: int, open_ratio: float) -> float:
    """Apply final per-DOF calibration while preserving the command convention."""
    ratio = float(open_ratio)
    if index == THUMB_ABDUCTION_DOF:
        zero = THUMB_ABDUCTION_ZERO_OPEN_RATIO
        return zero + (1.0 - zero) * ratio
    return ratio


def invert_open_ratio_overlay(index: int, open_ratio: float) -> float:
    """Recover the command that :func:`apply_open_ratio_overlay` would map here.

    The hand reports its true physical pose, so anything comparing feedback
    against a pre-overlay command needs one side moved onto the other. Prefer
    applying the overlay forward to the command; this exists for the cases that
    only hold the physical value.
    """
    ratio = float(open_ratio)
    if index == THUMB_ABDUCTION_DOF:
        zero = THUMB_ABDUCTION_ZERO_OPEN_RATIO
        return (ratio - zero) / (1.0 - zero)
    return ratio


__all__ = [
    "THUMB_ABDUCTION_DOF",
    "THUMB_ABDUCTION_JOINT",
    "THUMB_ABDUCTION_ZERO_OPEN_RATIO",
    "apply_open_ratio_overlay",
    "invert_open_ratio_overlay",
]
