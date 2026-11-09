"""Final command calibration applied after the hand's normal command scaling.

Keep task-specific calibration here rather than changing the generic
open-ratio/radian kinematics. Every driver command path passes through this
module immediately before conversion to the hand's integer registers.

Commands arrive in the URDF model's coordinates. The thumb's registers do not
read in those coordinates: its real pose, fitted from camera depth against the
model on 2026-11-09, differs from the angle the register implies. So commands
are moved from the model's angle onto the register that puts the thumb there,
and anything holding a register reading comes back through the inverse.
"""

from __future__ import annotations

from typing import Dict, NamedTuple

from . import kinematics


class ThumbFit(NamedTuple):
    """``real_rad = gain * register_rad + offset_rad`` for one thumb joint."""

    gain: float
    offset_rad: float


THUMB_ABDUCTION_JOINT = "thumb_proximal_yaw_joint"
THUMB_ABDUCTION_DOF = kinematics.dof_index(THUMB_ABDUCTION_JOINT)
THUMB_FLEXION_JOINT = "thumb_proximal_pitch_joint"
THUMB_FLEXION_DOF = kinematics.dof_index(THUMB_FLEXION_JOINT)

#: The real thumb sits this far past its register's yaw at every stop of the
#: sweep (+0.26 to +0.31 rad against the reported angle, no visible hysteresis).
THUMB_ABDUCTION_OFFSET_RAD = 0.27

#: Register-to-model fits. Yaw is a pure shift. Flexion comes from the two
#: sweeps run on the calibrated driver (registers 0.106 to 0.48 rad, yaw held at
#: the confirmed offset); the model's followers tie the distal links to it.
THUMB_CALIBRATION: Dict[int, ThumbFit] = {
    THUMB_ABDUCTION_DOF: ThumbFit(gain=1.0, offset_rad=THUMB_ABDUCTION_OFFSET_RAD),
    THUMB_FLEXION_DOF: ThumbFit(gain=0.73, offset_rad=-0.03),
}

#: Names this calibration in a recording's metadata, so an analysis can tell
#: which mapping turned its logical joints into registers.
HAND_CALIBRATION_ID = "thumb_camera_fit_20261109"

#: The register open ratio that a command of 0.0 lands on. The real thumb
#: reaches the model's closed limit there, so the registers below it stay unused.
_YAW = kinematics.DOFS[THUMB_ABDUCTION_DOF]
THUMB_ABDUCTION_ZERO_OPEN_RATIO = THUMB_ABDUCTION_OFFSET_RAD / (_YAW.upper - _YAW.lower)


def apply_open_ratio_overlay(index: int, open_ratio: float, enabled: bool = True) -> float:
    """The register open ratio that puts a joint at a commanded model angle.

    The result is limited to the register's range. A yaw command under the
    offset is not reachable (the thumb's open stop is already past it), so it
    lands on the open stop, and the feedback reports where the thumb really is.

    ``enabled=False`` turns the calibration off (the driver's
    ``thumb_abduction_overlay`` parameter): the command passes through
    unchanged, which is what a simulated hand needs, since its registers are
    the model.
    """
    ratio = float(open_ratio)
    fit = THUMB_CALIBRATION.get(index)
    if not enabled or fit is None:
        return ratio
    dof = kinematics.DOFS[index]
    span = dof.upper - dof.lower
    model_rad = dof.lower + (1.0 - ratio) * span
    register_rad = (model_rad - fit.offset_rad) / fit.gain
    return max(0.0, min(1.0, 1.0 - (register_rad - dof.lower) / span))


def invert_open_ratio_overlay(index: int, open_ratio: float, enabled: bool = True) -> float:
    """The model's open ratio for a register open ratio.

    The hand reports its registers, so anything comparing feedback against a
    model-space command needs one side moved onto the other. Prefer applying
    the calibration forward to the command; this exists for the cases that only
    hold the register value. A register outside the model's reach gives a ratio
    outside ``[0, 1]`` rather than a clamped one, so the caller can see that.
    With ``enabled=False`` it is the identity.
    """
    ratio = float(open_ratio)
    fit = THUMB_CALIBRATION.get(index)
    if not enabled or fit is None:
        return ratio
    dof = kinematics.DOFS[index]
    span = dof.upper - dof.lower
    register_rad = dof.lower + (1.0 - ratio) * span
    model_rad = fit.gain * register_rad + fit.offset_rad
    return 1.0 - (model_rad - dof.lower) / span


def model_rate_gain(index: int, enabled: bool = True) -> float:
    """Model joint rate per unit of register joint rate."""
    fit = THUMB_CALIBRATION.get(index)
    if not enabled or fit is None:
        return 1.0
    return fit.gain


__all__ = [
    "HAND_CALIBRATION_ID",
    "THUMB_ABDUCTION_DOF",
    "THUMB_ABDUCTION_JOINT",
    "THUMB_ABDUCTION_OFFSET_RAD",
    "THUMB_ABDUCTION_ZERO_OPEN_RATIO",
    "THUMB_CALIBRATION",
    "THUMB_FLEXION_DOF",
    "THUMB_FLEXION_JOINT",
    "ThumbFit",
    "apply_open_ratio_overlay",
    "invert_open_ratio_overlay",
    "model_rate_gain",
]
