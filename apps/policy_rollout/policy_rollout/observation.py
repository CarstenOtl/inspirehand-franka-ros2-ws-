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


def hand_velocity_width(hand_velocity_observation: str) -> int:
    """Return how many of the ten dq values the given convention keeps."""

    if hand_velocity_observation not in HAND_VELOCITY_OBSERVATION_MODES:
        raise ValueError(
            "hand_velocity_observation must be one of "
            + ", ".join(repr(mode) for mode in HAND_VELOCITY_OBSERVATION_MODES)
        )
    return POLICY_DOF if hand_velocity_observation == "included" else ARM_DOF


def native_osc_proprio_dim(hand_velocity_observation: str = "included") -> int:
    """Return the proprioception width that dq convention produces."""

    return (
        POLICY_DOF
        + hand_velocity_width(hand_velocity_observation)
        + PREVIOUS_ACTION_DIM
    )


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
) -> torch.Tensor:
    """Return ForgeUltra's exact ``q10 + dq + previous action`` vector.

    ``joint_velocity`` is always the full ten-joint measurement.  A checkpoint
    that omits the hand velocities keeps only its arm prefix, so a caller feeds
    and records what it measured without knowing which layout it is serving.
    """

    width = hand_velocity_width(hand_velocity_observation)
    q = _finite_vector(joint_position, POLICY_DOF, "joint_position")
    dq = _finite_vector(joint_velocity, POLICY_DOF, "joint_velocity")
    previous = _finite_vector(
        previous_filtered_native_action,
        PREVIOUS_ACTION_DIM,
        "previous_filtered_native_action",
    )
    return torch.cat((q, dq[:width], previous))


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
    "FR3_POLICY_JOINT_NAMES",
    "FORGE_POLICY_JOINT_NAMES",
    "HAND_VELOCITY_OBSERVATION_MODES",
    "NATIVE_OSC_PROPRIO_DIM",
    "build_native_osc_proprio",
    "hand_velocity_width",
    "native_osc_proprio_dim",
    "one_hot_process_phase",
]
