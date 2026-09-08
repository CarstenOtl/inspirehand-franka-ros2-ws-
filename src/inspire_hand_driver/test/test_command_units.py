"""The commanding unit, and the range check that used to be a silent clamp.

Both command paths take **open ratios**: 1.0 fully open, 0.0 fully closed,
matching ``~/state`` and the registers. Names address DOF and nothing more.

That replaced inferring the unit from the naming, which put two conventions on
one ``position`` field: ``1.5`` as a channel id clamped to a fully open hand,
the same ``1.5`` as a joint name clamped to a fully closed one. One number,
opposite ends of travel, nothing logged. Out-of-range now rejects instead.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rclpy = pytest.importorskip("rclpy")

from inspire_hand_msgs.srv import SetAngles  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.driver_node import InspireHandNode  # noqa: E402
from inspire_hand_driver.protocol import ANGLE_MAX  # noqa: E402

INDEX = kin.dof_index("index_proximal_joint")
INDEX_CHANNEL = kin.DOFS[INDEX].channel
PINKY = kin.dof_index("pinky_proximal_joint")


@pytest.fixture
def node():
    # The transport is built in __init__, so mock mode has to be a parameter
    # override at init time; setting it afterwards is too late.
    rclpy.init(args=["--ros-args", "-p", "mock:=true"])
    instance = InspireHandNode()
    assert instance._mock, "test must not touch a real serial port"
    try:
        yield instance
    finally:
        instance.destroy_node()
        rclpy.shutdown()


def commanded_ratio(node, index):
    """The open ratio the node last wrote for one DOF."""
    return node._last_command[index] / ANGLE_MAX


# -- the unit is the same however the DOF is addressed ---------------------

@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
def test_topic_and_service_agree_across_both_namings(node, ratio):
    written = []
    for name in ("index_proximal_joint", INDEX_CHANNEL):
        node._on_command(JointState(name=[name], position=[ratio]))
        written.append(commanded_ratio(node, INDEX))
        response = node._on_set_angles(
            SetAngles.Request(name=[name], open_ratio=[ratio]), SetAngles.Response()
        )
        assert response.accepted, response.message
        written.append(commanded_ratio(node, INDEX))
    assert written == [pytest.approx(ratio)] * 4


def test_one_is_fully_open_and_zero_fully_closed(node):
    node._on_command(JointState(name=["index_proximal_joint"], position=[1.0]))
    assert node._last_command[INDEX] == ANGLE_MAX
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    assert node._last_command[INDEX] == 0


def test_naming_may_be_mixed_now_that_it_does_not_select_a_unit(node):
    accepted, message = node._apply(["pinky_proximal_joint", INDEX_CHANNEL], [0.0, 1.0])
    assert accepted, message
    assert commanded_ratio(node, PINKY) == pytest.approx(0.0)
    assert commanded_ratio(node, INDEX) == pytest.approx(1.0)


# -- the range check ------------------------------------------------------

@pytest.mark.parametrize("value", [1.5, 1.0001, -0.5, -3.0, float("nan")])
def test_targets_outside_the_range_are_rejected_not_clamped(node, value):
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.5]))
    before = node._last_command[INDEX]
    accepted, message = node._apply(["index_proximal_joint"], [value])
    assert not accepted
    assert "out of the commandable range" in message
    assert node._last_command[INDEX] == before, "a rejected command must not move the DOF"


def test_a_rejected_entry_rejects_the_whole_message(node):
    node._on_command(JointState(name=[INDEX_CHANNEL], position=[0.5]))
    before = list(node._last_command)
    accepted, _ = node._apply(["pinky_proximal_joint", "index_proximal_joint"], [0.2, 1.5])
    assert not accepted
    assert node._last_command == before, "the in-range entry must not be applied either"


def test_the_rejection_message_names_the_offending_entries(node):
    _, message = node._apply(
        ["pinky_proximal_joint", "index_proximal_joint"], [0.2, 1.5]
    )
    assert "index_proximal_joint=1.5" in message
    assert "pinky_proximal_joint" not in message


def test_set_angles_rejects_out_of_range_too(node):
    response = node._on_set_angles(
        SetAngles.Request(name=[INDEX_CHANNEL], open_ratio=[1.5]), SetAngles.Response()
    )
    assert not response.accepted
    assert "out of the commandable range" in response.message


def test_the_range_bounds_themselves_are_accepted(node):
    for value in (0.0, 1.0):
        accepted, message = node._apply(["index_proximal_joint"], [value])
        assert accepted, message


def test_mismatched_name_and_value_counts_are_rejected(node):
    accepted, message = node._apply(["index_proximal_joint"], [0.1, 0.2])
    assert not accepted
    assert "entries" in message


# -- unchanged guarantees -------------------------------------------------

def test_passive_joints_are_not_commandable(node):
    accepted, message = node._apply(["index_intermediate_joint"], [0.0])
    assert not accepted
    assert "no recognised channel" in message


def test_unaddressed_dof_hold_their_previous_target(node):
    node._on_command(JointState(name=[INDEX_CHANNEL], position=[0.25]))
    before = node._last_command[PINKY]
    node._on_command(JointState(name=["thumb_proximal_yaw_joint"], position=[0.3]))
    assert node._last_command[PINKY] == before
    assert commanded_ratio(node, INDEX) == pytest.approx(0.25)


def test_state_extras_divisor_holds_current_and_force_between_reads():
    """RS485 is half-duplex, so a read the replay never uses is a command lost."""
    from inspire_hand_driver.protocol import REG_ANGLE_ACT, REG_CURRENT, REG_FORCE_ACT

    rclpy.init(args=[
        "--ros-args", "-p", "mock:=true", "-p", "state_extras_divisor:=3",
    ])
    instance = InspireHandNode()
    reads = []
    original = instance._transport.read_registers
    instance._transport.read_registers = lambda addr, count: (
        reads.append(addr) or original(addr, count)
    )
    try:
        for _ in range(6):
            instance._on_timer()
    finally:
        instance.destroy_node()
        rclpy.shutdown()

    assert reads.count(REG_ANGLE_ACT) == 6, "angles are joint_states, every cycle"
    assert reads.count(REG_CURRENT) == 2, "current rides the divisor"
    assert reads.count(REG_FORCE_ACT) == 2, "and so does force"
