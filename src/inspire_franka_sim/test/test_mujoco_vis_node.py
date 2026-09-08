"""Unit tests for the passive viewer's mapping and interpolation core."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_vis_node.py"
SPEC = importlib.util.spec_from_file_location("mujoco_vis_node_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mapping_entries_are_explicit_and_unambiguous():
    assert MODULE.parse_mapping_entries(
        ["panda_joint1 = fr3_joint1", "arm/elbow=fr3_joint4"]
    ) == {
        "panda_joint1": "fr3_joint1",
        "arm/elbow": "fr3_joint4",
    }
    with pytest.raises(ValueError, match="expected ROS_NAME=MUJOCO_NAME"):
        MODULE.parse_mapping_entries(["fr3_joint1"])
    with pytest.raises(ValueError, match="more than one"):
        MODULE.parse_mapping_entries(["j=a", "j=b"])


def test_linear_trajectory_interpolates_position_and_velocity():
    plan = MODULE.TrajectoryPlan(
        names=("joint",),
        times=np.array([2.0]),
        positions=np.array([[4.0]]),
        velocities=np.array([[np.nan]]),
        started_at=10.0,
        start_positions=np.array([0.0]),
        start_velocities=np.array([np.nan]),
    )
    command, complete = plan.sample(11.0)
    assert not complete
    assert command.positions["joint"] == pytest.approx(2.0)
    assert command.velocities["joint"] == pytest.approx(2.0)


def test_cubic_trajectory_uses_endpoint_velocities_and_holds_at_end():
    plan = MODULE.TrajectoryPlan(
        names=("joint",),
        times=np.array([1.0]),
        positions=np.array([[1.0]]),
        velocities=np.array([[0.0]]),
        started_at=5.0,
        start_positions=np.array([0.0]),
        start_velocities=np.array([0.0]),
    )
    halfway, complete = plan.sample(5.5)
    assert not complete
    assert halfway.positions["joint"] == pytest.approx(0.5)
    assert halfway.velocities["joint"] == pytest.approx(1.5)
    final, complete = plan.sample(6.0)
    assert complete
    assert final.positions["joint"] == pytest.approx(1.0)
    assert final.velocities["joint"] == 0.0


class FakeMujoco:
    mjtObj = SimpleNamespace(mjOBJ_JOINT=0)
    mjtEq = SimpleNamespace(mjEQ_JOINT=1)

    def __init__(self, names):
        self.names = names

    def mj_id2name(self, _model, _object_type, object_id):
        return self.names[object_id]


def test_joint_index_uses_qpos_dof_addresses_and_projects_mimic_joint():
    # A free joint occupies 7 qpos / 6 qvel entries before the two hinges.  A
    # broken implementation that uses joint_id as an array index fails here.
    model = SimpleNamespace(
        njnt=3,
        nq=9,
        nv=8,
        jnt_qposadr=np.array([0, 7, 8]),
        jnt_dofadr=np.array([0, 6, 7]),
        neq=1,
        eq_type=np.array([1]),
        eq_obj1id=np.array([2]),
        eq_obj2id=np.array([1]),
        eq_data=np.array([[0.25, 1.5, 0.0, 0.0, 0.0]]),
    )
    index = MODULE.JointIndex(
        FakeMujoco(["floating_base", "driver", "follower"]),
        model,
        {"ros_driver": "driver"},
    )
    assert index.resolve("floating_base") is None
    assert index.resolve("ros_driver").qpos_address == 7
    assert index.resolve("ros_driver").qvel_address == 6

    data = SimpleNamespace(qpos=np.zeros(9), qvel=np.zeros(8))
    data.qpos[7] = 2.0
    data.qvel[6] = 3.0
    index.apply_couplings(data, {"driver"})
    assert data.qpos[8] == pytest.approx(3.25)
    assert data.qvel[7] == pytest.approx(4.5)


def test_node_source_contains_no_ros_publisher():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "create_publisher" not in source
    # A typed empty string-array parameter is NOT_SET in rclpy/Jazzy.  Reading
    # the Parameter returned by declare_parameter is intentional; a later
    # get_parameter("joint_map") raises ParameterUninitializedException.
    assert 'get_parameter("joint_map")' not in source
    # `_subscriptions` belongs to rclpy.Node; shadowing it corrupts shutdown.
    assert "self._subscriptions" not in source
