"""Compliant mode in the driver: where the yield is applied, and where it is not.

The law itself is tested against exact time steps in ``test_compliance.py``.
Here the rates are turned up so that a push settles within one tick, because
what is being asserted is the plumbing -- the yield is an offset on top of the
commanded pose, the commanded pose is what every other path still sees, and
leaving the mode puts the hand back exactly where it was told to be.

The mock grows one hook for this: ``external_force``, standing in for a thumb
on a fingertip. Nothing else can produce force without also jamming the finger.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rclpy = pytest.importorskip("rclpy")

from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import SetBool, Trigger  # noqa: E402

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.driver_node import InspireHandNode  # noqa: E402
from inspire_hand_driver.protocol import (  # noqa: E402
    REG_FORCE_ACT,
    HandCommunicationError,
)

INDEX = kin.dof_index("index_proximal_joint")
PINKY = kin.dof_index("pinky_proximal_joint")
THUMB_ROTATION = kin.dof_index("thumb_proximal_yaw_joint")

#: A push worth 160 counts of yield at :data:`GAIN`: 400 g at the tip, 80 of
#: which the deadband eats.
PUSH = 400
#: The gain these tests do their arithmetic in, pinned rather than inherited.
#: The default is a tuning choice that follows the hand it was measured on;
#: the plumbing these tests cover is not supposed to follow it. The default
#: itself is pinned in test_compliance.py.
GAIN = 0.5


@pytest.fixture
def make_node():
    instances = []

    def factory(*params):
        # The rates are turned up so a push settles within one tick: what these
        # tests are about is where the yield is applied, not how fast.
        args = [
            "--ros-args",
            "-p",
            "mock:=true",
            "-p",
            "compliance_yield_rate:=1000000.0",
            "-p",
            f"compliance_counts_per_gram:={GAIN}",
        ]
        for param in params:
            args += ["-p", param]
        rclpy.init(args=args)
        node = InspireHandNode()
        assert node._mock
        node._transport._slew = 1e6
        instances.append(node)
        return node

    yield factory
    for node in instances:
        node.destroy_node()
        rclpy.shutdown()


def tick(node, pause=0.02):
    time.sleep(pause)
    node._on_timer()


def written(node):
    """What the hand was actually told to go to, yield included."""
    return [int(round(t)) for t in node._transport._targets]


def push(node, index=INDEX, grams=PUSH):
    node._transport.external_force[index] = grams


def compliance(node, enable):
    if enable:
        # Engaging takes a tare, and a tare needs a force reading to take. A
        # running driver always has one by the time anyone calls the service;
        # a freshly built test node does not, so give it the cycle it would
        # have had. Without this the tare is deferred onto the next tick and
        # captures whatever the test pushes with, zeroing it away.
        tick(node)
    response = node._on_set_compliance(SetBool.Request(data=enable), SetBool.Response())
    assert response.success, response.message
    return response


def test_compliance_is_off_unless_asked_for(make_node):
    node = make_node()
    assert not node._spring.engaged
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    push(node)
    tick(node)
    tick(node)
    assert written(node)[INDEX] == 0, "a push on a stiff hand moves nothing"


def test_the_launch_parameter_turns_it_on(make_node):
    node = make_node("compliance:=true")
    assert node._spring.engaged


def test_a_pushed_fingertip_opens_that_finger(make_node):
    node = make_node()
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    compliance(node, True)
    push(node)
    tick(node)
    assert written(node)[INDEX] == 160
    assert node._last_command[INDEX] == 0, "the commanded pose is the rest position"


def test_releasing_the_fingertip_returns_exactly_to_the_commanded_grasp(make_node):
    node = make_node("compliance_return_rate:=1000000.0")
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.2]))
    compliance(node, True)
    push(node)
    tick(node)
    assert written(node)[INDEX] == 200 + 160
    node._transport.external_force[INDEX] = 0
    tick(node)
    assert written(node)[INDEX] == 200
    assert node._yield == [0] * 6


def test_a_command_during_a_push_lands_where_it_was_asked_to(make_node):
    # The merge base is the commanded pose, not the pushed-open one; otherwise
    # every partial command sent while someone is holding a fingertip would
    # bake that push into the grasp permanently.
    node = make_node("compliance_return_rate:=2000.0")
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    compliance(node, True)
    push(node)
    tick(node)
    node._on_command(JointState(name=["pinky_proximal_joint"], position=[0.5]))
    assert node._last_command[INDEX] == 0
    assert written(node)[INDEX] == 160, "still giving"
    assert written(node)[PINKY] == 500, "and unaffected"
    node._transport.external_force[INDEX] = 0
    for _ in range(10):
        tick(node)
    assert written(node)[INDEX] == 0


def test_leaving_compliant_mode_ramps_out_rather_than_snapping_shut(make_node):
    node = make_node("compliance_return_rate:=2000.0")
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    compliance(node, True)
    push(node)
    tick(node)
    assert written(node)[INDEX] == 160
    compliance(node, False)
    tick(node)
    assert 0 < written(node)[INDEX] < 160, "on its way back, not there yet"
    assert node._spring.active, "still being driven"
    for _ in range(10):
        tick(node)
    assert written(node)[INDEX] == 0
    assert not node._spring.active


def test_a_yield_cannot_push_a_finger_past_fully_open(make_node):
    node = make_node()
    node._on_command(JointState(name=["index_proximal_joint"], position=[1.0]))
    compliance(node, True)
    push(node, grams=1000)
    tick(node)
    assert written(node)[INDEX] == 1000


def test_force_is_read_every_cycle_while_compliant(make_node):
    # Force normally rides the extras divisor. It is the spring's only input,
    # so sampling it at a fifth of the loop rate would not do.
    node = make_node("state_extras_divisor:=5")
    mock = node._transport
    reads = []
    real = mock.read_registers
    mock.read_registers = lambda addr, count: (reads.append(addr), real(addr, count))[1]

    for _ in range(5):
        tick(node)
    assert reads.count(REG_FORCE_ACT) == 1

    compliance(node, True)
    reads.clear()
    for _ in range(5):
        tick(node)
    assert reads.count(REG_FORCE_ACT) == 5


def test_gains_retune_live_and_bad_ones_are_refused(make_node):
    node = make_node()
    ok = node.set_parameters([Parameter("compliance_counts_per_gram", value=1.0)])
    assert ok[0].successful
    assert node._spring.gains.counts_per_gram == 1.0

    refused = node.set_parameters([Parameter("compliance_counts_per_gram", value=-1.0)])
    assert not refused[0].successful
    assert "negative" in refused[0].reason
    assert node._spring.gains.counts_per_gram == 1.0, "the working tune survives"


def test_the_parameter_toggles_the_mode_too(make_node):
    node = make_node()
    assert node.set_parameters([Parameter("compliance", value=True)])[0].successful
    assert node._spring.engaged
    assert node.set_parameters([Parameter("compliance", value=False)])[0].successful
    assert not node._spring.engaged


def test_thumb_rotation_is_left_out_by_default_and_can_be_put_back(make_node):
    node = make_node()
    assert node._spring.channels[THUMB_ROTATION] is False
    assert node.set_parameters(
        [Parameter("compliance_channels", value=["5", "6"])]
    )[0].successful
    assert node._spring.channels[THUMB_ROTATION] is True
    assert node._spring.channels[INDEX] is False


def test_a_hand_that_went_away_comes_back_without_a_stale_yield(make_node):
    node = make_node("max_read_failures:=2")
    mock = node._transport
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    compliance(node, True)
    push(node)
    tick(node)
    assert node._yield[INDEX] == 160

    real = mock.read_registers
    mock.read_registers = lambda addr, count: (_ for _ in ()).throw(
        HandCommunicationError("silence")
    )
    tick(node)
    tick(node)
    mock.read_registers = real
    node._transport.external_force[INDEX] = 0
    tick(node)
    assert node._yield == [0] * 6
    assert node._spring.engaged, "the mode is not silently abandoned"


def test_diagnostics_carry_the_yield(make_node):
    node = make_node()
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    compliance(node, True)
    push(node)
    tick(node)
    published = []
    node._diag_pub.publish = published.append
    tick(node)
    array = published[-1]
    values = {kv.key: kv.value for kv in array.status[INDEX + 1].values}
    assert values["compliance_yield"] == "160"
    assert "giving" in array.status[INDEX + 1].message
    summary = {kv.key: kv.value for kv in array.status[0].values}
    assert summary["compliance"] == "on"


# -- the fingertip zero, which moves ------------------------------------------

def test_engaging_tares_against_whatever_the_fingertips_read(make_node):
    # The sensors' zero shifts after a heavy push and stays shifted, so the
    # reading at the moment compliance is switched on is what "untouched"
    # has to mean for that episode.
    node = make_node()
    push(node, grams=219)
    tick(node)
    compliance(node, True)
    assert node._spring.zeros[INDEX] == 219.0


def test_a_finger_with_a_shifted_zero_does_not_sit_permanently_open(make_node):
    # The bench failure this came from: the index read 219 g fully open and
    # untouched, so an untared law held it open and reported "gave, did not
    # come back".
    node = make_node("compliance_deadband:=30.0", "compliance_counts_per_gram:=0.18")
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.3]))
    push(node, grams=219)
    tick(node)
    compliance(node, True)
    for _ in range(5):
        tick(node)
    assert node._yield[INDEX] == 0, "the shifted zero is not a push"
    assert written(node)[INDEX] == 300, "and the finger holds the commanded pose"


def test_a_real_push_on_top_of_a_shifted_zero_still_gives(make_node):
    node = make_node("compliance_deadband:=30.0", "compliance_counts_per_gram:=0.18")
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.3]))
    push(node, grams=219)
    tick(node)
    compliance(node, True)
    tick(node)
    assert node._yield[INDEX] == 0
    push(node, grams=219 + 530)
    tick(node)
    assert node._yield[INDEX] == 90, "(530 - 30) g at 0.18 counts/g"


def test_the_tare_service_rezeroes_a_zero_that_moved_mid_session(make_node):
    node = make_node()
    compliance(node, True)
    push(node, grams=219)
    tick(node)
    assert node._yield[INDEX] > 0, "before taring it reads as a push"
    response = node._on_tare_force(Trigger.Request(), Trigger.Response())
    assert response.success and "219" in response.message
    for _ in range(50):
        tick(node, 0.0)
    assert node._spring.zeros[INDEX] == 219.0


def test_a_tare_before_any_force_read_waits_for_one(make_node):
    # compliance:=true at launch runs before a single register has been read,
    # and taring against the placeholder zeros would make every channel's real
    # resting offset look like a push.
    node = make_node("compliance:=true")
    assert node._tare_pending, "deferred, not taken against nothing"
    assert node._spring.zeros == [0.0] * 6
    push(node, grams=219)
    tick(node)
    assert not node._tare_pending
    assert node._spring.zeros[INDEX] == 219.0


def test_diagnostics_show_the_zero_each_channel_is_using(make_node):
    node = make_node()
    push(node, grams=219)
    tick(node)
    compliance(node, True)
    published = []
    node._diag_pub.publish = published.append
    tick(node)
    values = {kv.key: kv.value for kv in published[-1].status[INDEX + 1].values}
    assert values["force_zero"] == "219"
