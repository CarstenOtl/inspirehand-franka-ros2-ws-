"""Framework-independent ForgeUltra student observation contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch


NATIVE_OSC_PROPRIO_DIM = 29
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
) -> torch.Tensor:
    """Return ForgeUltra's exact ``q10 + dq10 + previous action`` vector."""

    q = _finite_vector(joint_position, 10, "joint_position")
    dq = _finite_vector(joint_velocity, 10, "joint_velocity")
    previous = _finite_vector(
        previous_filtered_native_action,
        9,
        "previous_filtered_native_action",
    )
    return torch.cat((q, dq, previous))


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
    "NATIVE_OSC_PROPRIO_DIM",
    "build_native_osc_proprio",
    "one_hot_process_phase",
]
