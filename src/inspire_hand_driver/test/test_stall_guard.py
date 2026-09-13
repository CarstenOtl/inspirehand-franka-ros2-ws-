"""The force threshold and the stall guard, against the mock's stall model.

A finger that cannot reach its target pushes until the firmware latches an
error and the DOF goes dead until CLEAR_ERROR or a power cycle. The node's
two answers -- a force threshold that is always applied, and a guard that
backs a stalled DOF off and clears the error -- are exercised here without
jamming a real hand.
"""

import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rclpy = pytest.importorskip("rclpy")

from diagnostic_msgs.msg import DiagnosticStatus  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from inspire_hand_msgs.srv import SetForce  # noqa: E402

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.driver_node import InspireHandNode  # noqa: E402
from inspire_hand_driver.protocol import (  # noqa: E402
    ERROR_LOCKED_ROTOR,
    STATUS_AT_FORCE,
    STATUS_AT_TARGET,
    STATUS_LOCKED_ROTOR,
    HandCommunicationError,
)

INDEX = kin.dof_index("index_proximal_joint")
OBSTACLE = 400  # register counts; where the mock's object stops the finger


@pytest.fixture
def make_node():
    instances = []

    def factory(*params):
        args = ["--ros-args", "-p", "mock:=true"]
        for param in params:
            args += ["-p", param]
        rclpy.init(args=args)
        node = InspireHandNode()
        assert node._mock
        mock = node._transport
        mock._slew = 1e6  # arrive within one tick
        mock.stall_after_sec = 0.01
        instances.append(node)
        return node

    yield factory
    for node in instances:
        node.destroy_node()
    rclpy.shutdown()


def tick(node, pause=0.02):
    time.sleep(pause)
    node._on_timer()


def close_index_onto_obstacle(node):
    node._transport.obstacles[INDEX] = OBSTACLE
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    tick(node)  # arrives at the obstacle


# -- the force threshold ------------------------------------------------------

def test_force_threshold_is_applied_at_startup_by_default(make_node):
    node = make_node()
    assert node._transport.force_thresholds == [500] * 6


def test_zero_leaves_the_hands_own_threshold_alone(make_node):
    node = make_node("startup_force:=0")
    assert node._transport.force_thresholds == [0] * 6
    assert node._force_cache is None


def test_with_a_threshold_a_blocked_finger_stops_cleanly(make_node):
    node = make_node()
    mock = node._transport
    close_index_onto_obstacle(node)
    tick(node)
    tick(node)
    assert mock.read_angles()[INDEX] == OBSTACLE
    assert mock.status_codes()[INDEX] == STATUS_AT_FORCE
    assert mock.latched_errors() == [0] * 6
    assert mock.clear_error_writes == 0, "nothing to clear"
    assert node._last_command[INDEX] == 0, "the target was not touched"


def test_threshold_is_reapplied_when_the_hand_comes_back_after_silence(make_node):
    node = make_node("max_read_failures:=2")
    mock = node._transport
    mock.force_thresholds = [0] * 6  # the hand rebooted to its defaults ...
    real = mock.read_registers
    mock.read_registers = lambda addr, count: (_ for _ in ()).throw(
        HandCommunicationError("silence")
    )
    tick(node)
    tick(node)  # ... and was silent long enough to count as lost
    mock.read_registers = real
    tick(node)
    assert mock.force_thresholds == [500] * 6


def test_threshold_is_reapplied_when_readback_disagrees(make_node):
    node = make_node()
    mock = node._transport
    mock.force_thresholds[INDEX] = 0  # a reboot too brief to miss a read
    node._last_limits_check = -math.inf
    tick(node)
    assert mock.force_thresholds == [500] * 6


def test_set_force_overrides_the_default_and_readback_respects_it(make_node):
    node = make_node()
    mock = node._transport
    response = node._on_set_force(
        SetForce.Request(name=["index_proximal_joint"], force=[200]), SetForce.Response()
    )
    assert response.accepted, response.message
    assert mock.force_thresholds[INDEX] == 200
    assert node._force_cache[INDEX] == 200
    node._last_limits_check = -math.inf
    tick(node)
    assert mock.force_thresholds[INDEX] == 200, "a deliberate override is not 'lost'"


# -- the stall guard ----------------------------------------------------------

def test_stalled_finger_is_backed_off_cleared_and_moves_again(make_node):
    node = make_node("startup_force:=0", "stall_holdoff_sec:=5.0")
    mock = node._transport
    close_index_onto_obstacle(node)
    tick(node)  # the mock's protection latches, and the guard sees it
    assert mock.clear_error_writes == 1
    assert node._last_command[INDEX] == OBSTACLE + 30, "backed off before clearing"
    assert mock.latched_errors() == [0] * 6
    tick(node)
    assert mock.read_angles()[INDEX] == OBSTACLE + 30, "alive again, at the backed-off pose"
    assert mock.status_codes()[INDEX] == STATUS_AT_TARGET
    assert mock.flash_saves == 0, "must never commit to flash"
    assert INDEX not in node._stalled


def test_commands_into_a_fresh_stall_are_held_at_the_backoff(make_node):
    node = make_node("startup_force:=0", "stall_holdoff_sec:=5.0")
    close_index_onto_obstacle(node)
    tick(node)
    # A stream re-sending the unreachable target does not re-stall the finger.
    accepted, message = node._apply(["index_proximal_joint"], [0.0])
    assert accepted, message
    assert node._last_command[INDEX] == OBSTACLE + 30
    # Opening is always allowed.
    node._apply(["index_proximal_joint"], [1.0])
    assert node._last_command[INDEX] == 1000
    # Once the hold-off has passed, the target goes through again.
    node._stall_floor[INDEX] = (OBSTACLE + 30, time.monotonic() - 1.0)
    node._apply(["index_proximal_joint"], [0.0])
    assert node._last_command[INDEX] == 0
    assert node._stall_floor[INDEX] is None


def test_other_dof_are_not_moved_by_the_backoff(make_node):
    node = make_node("startup_force:=0")
    node._on_command(JointState(name=["pinky_proximal_joint"], position=[0.3]))
    close_index_onto_obstacle(node)
    tick(node)
    assert node._last_command[INDEX] == OBSTACLE + 30
    assert node._last_command[kin.dof_index("pinky_proximal_joint")] == 300


def test_stall_guard_off_only_reports(make_node):
    node = make_node("startup_force:=0", "stall_guard:=false")
    mock = node._transport
    close_index_onto_obstacle(node)
    tick(node)
    tick(node)
    assert mock.clear_error_writes == 0
    assert node._last_command[INDEX] == 0
    assert mock.status_codes()[INDEX] == STATUS_LOCKED_ROTOR
    assert mock.latched_errors()[INDEX] == ERROR_LOCKED_ROTOR
    assert INDEX in node._stalled


def test_clear_errors_service_writes_clear_error(make_node):
    node = make_node()
    response = node._on_clear_errors(Trigger.Request(), Trigger.Response())
    assert response.success, response.message
    assert node._transport.clear_error_writes == 1
    assert node._transport.flash_saves == 0


def test_diagnostics_flag_the_stalled_dof(make_node):
    node = make_node("startup_force:=0", "stall_holdoff_sec:=5.0")
    published = []
    node._diag_pub.publish = published.append
    close_index_onto_obstacle(node)
    tick(node)
    stalled = published[-1]
    by_name = {s.name: s for s in stalled.status}
    assert by_name[f"{node.get_name()}: index"].level == DiagnosticStatus.ERROR
    assert by_name[f"{node.get_name()}: hand"].level == DiagnosticStatus.ERROR
    assert "index" in by_name[f"{node.get_name()}: hand"].message
    tick(node)
    tick(node)
    recovered = {s.name: s for s in published[-1].status}
    assert recovered[f"{node.get_name()}: index"].level == DiagnosticStatus.OK
    assert recovered[f"{node.get_name()}: hand"].message == "ok"
