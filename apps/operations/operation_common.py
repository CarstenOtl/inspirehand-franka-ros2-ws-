"""Small, ROS-independent helpers shared by the robot operations scripts."""

from __future__ import annotations

import math


HAND_CHANNELS = ("1", "2", "3", "4", "5", "6")
HAND_ZERO_RATIOS = (1.0,) * 6

# Franka's documented start configuration. Literal zero is not a valid FR3
# configuration: joint 4 must be negative and joint 6 must be positive.
ARM_HOME = (
    0.0,
    -math.pi / 4.0,
    0.0,
    -3.0 * math.pi / 4.0,
    0.0,
    math.pi / 2.0,
    math.pi / 4.0,
)

# Limits shipped by franka_description/robots/fr3/joint_limits.yaml.
FR3_LIMITS = (
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.0770, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
)


def namespaced(namespace: str, *parts: str) -> str:
    """Join a possibly empty ROS namespace and one or more graph names."""
    namespace = str(namespace or "").strip("/")
    pieces = [str(part).strip("/") for part in parts if part]
    return "/" + "/".join(([namespace] if namespace else []) + pieces)


def arm_joint_names(robot_type: str = "fr3", arm_prefix: str = "") -> tuple[str, ...]:
    prefix = f"{arm_prefix}_" if arm_prefix else ""
    return tuple(f"{prefix}{robot_type}_joint{index}" for index in range(1, 8))


def validate_arm_target(target) -> tuple[float, ...]:
    values = tuple(float(value) for value in target)
    if len(values) != 7:
        raise ValueError(f"an FR3 target needs 7 joint values, got {len(values)}")
    invalid = [
        f"joint {index}: {value:g} outside [{lower:g}, {upper:g}]"
        for index, (value, (lower, upper)) in enumerate(zip(values, FR3_LIMITS), start=1)
        if not lower <= value <= upper
    ]
    if invalid:
        raise ValueError("FR3 target exceeds joint limits: " + "; ".join(invalid))
    return values


def positions_by_name(names, positions) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, positions)}


def maximum_error(actual: dict[str, float], names, target) -> float | None:
    """Return the maximum named position error, or None when feedback is incomplete."""
    if not all(name in actual for name in names):
        return None
    return max(abs(actual[name] - expected) for name, expected in zip(names, target))


def conflicting_controllers(controllers, joint_names, excluded=()):
    """Find active, non-broadcaster controllers claiming one of ``joint_names``."""
    joints = set(joint_names)
    excluded = set(excluded)
    conflicts = []
    for controller in controllers:
        if (
            controller.name in excluded
            or controller.state != "active"
            or "broadcaster" in controller.type.lower()
        ):
            continue
        claimed = {interface.split("/", 1)[0] for interface in controller.claimed_interfaces}
        if claimed & joints:
            conflicts.append(controller.name)
    return conflicts
