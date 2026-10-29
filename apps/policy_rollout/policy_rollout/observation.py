"""Framework-independent ForgeUltra student observation contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch


# ForgeUltra's q and dq cover the seven arm joints followed by the three driven
# hand joints.  A checkpoint trained with ``hand_velocity_observation:
# omitted`` was shown only the arm half of dq, which is the whole of why its
# proprioception is 26 values wide rather than 29.
POLICY_DOF = 10
ARM_DOF = 7
PREVIOUS_ACTION_DIM = 9
HAND_VELOCITY_OBSERVATION_MODES = ("included", "omitted")
NATIVE_OSC_PROPRIO_DIM = 29

# A checkpoint trained with ``force_observation: included`` sees nine more
# values after the previous action: the thumb and index pad forces in N
# divided by 20 (ForgeUltra ``build_student_force_observation`` over
# ``distillation_fingertip_force_n``) and the seven FR3 joint torques divided
# by the effort limits (``distillation_arm_joint_torque_observation``), in the
# teacher actor's order. The checkpoint records no layout; the training side
# confirmed it (student.py ``build_native_osc_student_proprio``), and the first
# proprio layer agrees: the previous-action column fingerprint sits right
# before the forces.
FORCE_OBSERVATION_MODES = ("omitted", "included")
FINGERTIP_FORCE_DIM = 2
FINGERTIP_FORCE_SCALE_N = 20.0
ARM_TORQUE_DIM = 7
FR3_ARM_EFFORT_LIMITS_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def hand_velocity_width(hand_velocity_observation: str) -> int:
    """Return how many of the ten dq values the given convention keeps."""

    if hand_velocity_observation not in HAND_VELOCITY_OBSERVATION_MODES:
        raise ValueError(
            "hand_velocity_observation must be one of "
            + ", ".join(repr(mode) for mode in HAND_VELOCITY_OBSERVATION_MODES)
        )
    return POLICY_DOF if hand_velocity_observation == "included" else ARM_DOF


def force_observation_width(force_observation: str) -> int:
    """Return how many force/torque values the given convention appends."""

    if force_observation not in FORCE_OBSERVATION_MODES:
        raise ValueError(
            "force_observation must be one of "
            + ", ".join(repr(mode) for mode in FORCE_OBSERVATION_MODES)
        )
    return FINGERTIP_FORCE_DIM + ARM_TORQUE_DIM if force_observation == "included" else 0


def native_osc_proprio_dim(
    hand_velocity_observation: str = "included", force_observation: str = "omitted"
) -> int:
    """Return the proprioception width those conventions produce."""

    return (
        POLICY_DOF
        + hand_velocity_width(hand_velocity_observation)
        + PREVIOUS_ACTION_DIM
        + force_observation_width(force_observation)
    )


def normalize_fr3_joint_torque(arm_joint_torque_nm) -> torch.Tensor:
    """ForgeUltra ``normalize_fr3_joint_torque``: divide by the limits, clip to 1."""

    torque = _finite_vector(arm_joint_torque_nm, ARM_TORQUE_DIM, "arm_joint_torque_nm")
    return (torque / torch.tensor(FR3_ARM_EFFORT_LIMITS_NM)).clamp(-1.0, 1.0)


FORGE_POLICY_JOINT_NAMES = (
    *(f"fr3_joint{index}" for index in range(1, 8)),
    "thumb_joint_0",
    "thumb_joint_1",
    "index_joint_0",
)

# The last three Forge coordinates correspond to these driven joints in this
# workspace's physical RH56 driver.  This name mapping does not claim that the
# two hand models have identical kinematics; hardware rollout remains gated on
# a dedicated validation of that boundary.
FR3_POLICY_JOINT_NAMES = (
    *(f"fr3_joint{index}" for index in range(1, 8)),
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
)


def _finite_vector(value, width: int, label: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32).flatten()
    if result.shape != (width,):
        raise ValueError(f"{label} must contain {width} values")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{label} contains non-finite values")
    return result


def build_native_osc_proprio(
    joint_position,
    joint_velocity,
    previous_filtered_native_action,
    *,
    hand_velocity_observation: str = "included",
    force_observation: str = "omitted",
    fingertip_force_n=None,
    arm_joint_torque_nm=None,
) -> torch.Tensor:
    """Return ForgeUltra's exact ``q10 + dq + previous action [+ forces]`` vector.

    ``joint_velocity`` is always the full ten-joint measurement.  A checkpoint
    that omits the hand velocities keeps only its arm prefix, so a caller feeds
    and records what it measured without knowing which layout it is serving.
    A force-observing checkpoint also needs ``[thumb, index]`` pad compression
    in N and the seven arm joint torques in N m; the rest ignore them.
    """

    width = hand_velocity_width(hand_velocity_observation)
    force_width = force_observation_width(force_observation)
    q = _finite_vector(joint_position, POLICY_DOF, "joint_position")
    dq = _finite_vector(joint_velocity, POLICY_DOF, "joint_velocity")
    previous = _finite_vector(
        previous_filtered_native_action,
        PREVIOUS_ACTION_DIM,
        "previous_filtered_native_action",
    )
    if not force_width:
        return torch.cat((q, dq[:width], previous))
    if fingertip_force_n is None or arm_joint_torque_nm is None:
        raise ValueError(
            "a force-observing checkpoint needs fingertip_force_n and arm_joint_torque_nm"
        )
    # Unilateral, like the hardware sensor and ForgeUltra's projection.
    force = (
        _finite_vector(fingertip_force_n, FINGERTIP_FORCE_DIM, "fingertip_force_n").clamp(min=0.0)
        / FINGERTIP_FORCE_SCALE_N
    )
    return torch.cat(
        (q, dq[:width], previous, force, normalize_fr3_joint_torque(arm_joint_torque_nm))
    )


def one_hot_process_phase(phase: str, feature_names: Sequence[str]) -> torch.Tensor:
    """Encode the cyclic phase using the schema stored in the checkpoint."""

    names = tuple(str(name) for name in feature_names)
    aliases = {
        "policy": "policy",
        "follow": "follow_waypoints",
        "follow_waypoints": "follow_waypoints",
        "return": "return_to_reset",
        "return_to_reset": "return_to_reset",
    }
    canonical = aliases.get(str(phase).strip().lower(), str(phase).strip())
    if canonical not in names:
        raise ValueError(f"phase {phase!r} is not in checkpoint schema {names}")
    result = torch.zeros(len(names), dtype=torch.float32)
    result[names.index(canonical)] = 1.0
    return result


__all__ = [
    "ARM_TORQUE_DIM",
    "FINGERTIP_FORCE_DIM",
    "FINGERTIP_FORCE_SCALE_N",
    "FORCE_OBSERVATION_MODES",
    "FR3_ARM_EFFORT_LIMITS_NM",
    "FR3_POLICY_JOINT_NAMES",
    "FORGE_POLICY_JOINT_NAMES",
    "HAND_VELOCITY_OBSERVATION_MODES",
    "NATIVE_OSC_PROPRIO_DIM",
    "build_native_osc_proprio",
    "force_observation_width",
    "hand_velocity_width",
    "native_osc_proprio_dim",
    "normalize_fr3_joint_torque",
    "one_hot_process_phase",
]
