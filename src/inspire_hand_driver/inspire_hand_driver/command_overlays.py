"""Thumb yaw's register scale, applied after the hand's normal command scaling.

The hand model in :mod:`inspire_hand_driver.kinematics` is fitted to the real
hand, so a joint angle means the real joint angle and most channels map an open
ratio straight onto a register. Thumb yaw is the exception: its command range
stops short of the register's closed end, because past register 250 the thumb
reaches its motor's end stop, where it has been seen to shift. Command 0.0 is
therefore register 250, 1.0 is register 1000, and the commands between are
scaled linearly.
"""

from __future__ import annotations

from . import kinematics

THUMB_ABDUCTION_JOINT = "thumb_proximal_yaw_joint"
THUMB_ABDUCTION_DOF = kinematics.dof_index(THUMB_ABDUCTION_JOINT)

#: The register open ratio that a yaw command of 0.0 (fully closed) lands on.
THUMB_ABDUCTION_ZERO_OPEN_RATIO = 0.25

#: Names this mapping in a recording's metadata.
HAND_CALIBRATION_ID = "hardware_fit_20261109"


def apply_open_ratio_overlay(index: int, open_ratio: float, enabled: bool = True) -> float:
    """The register open ratio for a commanded open ratio.

    ``enabled=False`` turns the scale off (the driver's
    ``thumb_abduction_overlay`` parameter), which a simulated hand needs, since
    its registers are the model.
    """
    ratio = float(open_ratio)
    if not enabled or index != THUMB_ABDUCTION_DOF:
        return ratio
    zero = THUMB_ABDUCTION_ZERO_OPEN_RATIO
    return max(0.0, min(1.0, zero + ratio * (1.0 - zero)))


def invert_open_ratio_overlay(index: int, open_ratio: float, enabled: bool = True) -> float:
    """The commanded open ratio that a register open ratio corresponds to.

    A register outside the commanded range gives a ratio outside ``[0, 1]``
    rather than a clamped one, so the caller can see that.
    """
    ratio = float(open_ratio)
    if not enabled or index != THUMB_ABDUCTION_DOF:
        return ratio
    zero = THUMB_ABDUCTION_ZERO_OPEN_RATIO
    return (ratio - zero) / (1.0 - zero)


def model_rate_gain(index: int, enabled: bool = True) -> float:
    """Commanded open-ratio rate per unit of register open-ratio rate."""
    if not enabled or index != THUMB_ABDUCTION_DOF:
        return 1.0
    return 1.0 / (1.0 - THUMB_ABDUCTION_ZERO_OPEN_RATIO)


__all__ = [
    "HAND_CALIBRATION_ID",
    "THUMB_ABDUCTION_DOF",
    "THUMB_ABDUCTION_JOINT",
    "THUMB_ABDUCTION_ZERO_OPEN_RATIO",
    "apply_open_ratio_overlay",
    "invert_open_ratio_overlay",
    "model_rate_gain",
]
