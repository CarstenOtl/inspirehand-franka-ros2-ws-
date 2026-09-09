from types import SimpleNamespace

import pytest

from operation_common import (
    ARM_HOME,
    arm_joint_names,
    conflicting_controllers,
    maximum_error,
    namespaced,
    validate_arm_target,
)


def test_default_home_is_inside_fr3_limits():
    assert validate_arm_target(ARM_HOME) == ARM_HOME


def test_literal_zero_pose_is_rejected():
    with pytest.raises(ValueError, match="joint 4"):
        validate_arm_target([0.0] * 7)


def test_namespace_and_prefix_names():
    assert namespaced("cell_1", "controller_manager") == "/cell_1/controller_manager"
    assert arm_joint_names("fr3", "left") == tuple(
        f"left_fr3_joint{index}" for index in range(1, 8)
    )


def test_feedback_error_requires_every_joint():
    names = arm_joint_names()
    assert maximum_error({names[0]: 0.0}, names, ARM_HOME) is None
    actual = dict(zip(names, ARM_HOME))
    actual[names[3]] += 0.02
    assert maximum_error(actual, names, ARM_HOME) == pytest.approx(0.02)


def test_only_active_joint_claimers_conflict():
    controllers = [
        SimpleNamespace(
            name="gravity", state="active", type="example/Controller",
            claimed_interfaces=["fr3_joint1/effort"],
        ),
        SimpleNamespace(
            name="states", state="active", type="joint_state_broadcaster/Broadcaster",
            claimed_interfaces=["fr3_joint1/position"],
        ),
        SimpleNamespace(
            name="inactive", state="inactive", type="example/Controller",
            claimed_interfaces=["fr3_joint2/effort"],
        ),
    ]
    assert conflicting_controllers(controllers, arm_joint_names()) == ["gravity"]
