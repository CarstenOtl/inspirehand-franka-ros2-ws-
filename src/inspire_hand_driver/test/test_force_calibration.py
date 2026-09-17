"""The hand's own force-sensor calibration, and the six seconds it costs.

``~/tare_force`` is arithmetic in the driver; this is the hand rewriting what
its sensors report, and for the duration the hand -- not this node -- is the
thing commanding the fingers. So most of what is worth asserting here is about
restraint: that every write path stands down, that the stall guard does not
treat a self-driving hand as six faults, and that the node picks the pose back
up afterwards rather than leaving the fingers where the routine dropped them.

The six seconds are never actually waited out. The deadline is monotonic and
the tests move it into the past, which is the same thing the clock would do,
only sooner.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rclpy = pytest.importorskip("rclpy")

from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import SetBool, Trigger  # noqa: E402

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.driver_node import InspireHandNode  # noqa: E402
from inspire_hand_driver.protocol import (  # noqa: E402
    ANGLE_MAX,
    CHANNEL_IDS,
    ERROR_LOCKED_ROTOR,
    HandCommunicationError,
)

INDEX = kin.dof_index("index_proximal_joint")
THUMB_BEND = kin.dof_index("thumb_proximal_pitch_joint")
from inspire_hand_driver.driver_node import FINGER_DOF, THUMB_DOF  # noqa: E402


@pytest.fixture
def make_node():
    instances = []

    def factory(*params):
        # Calibration is off by default on a real node -- the routine jammed
        # the hand it was developed against. These tests are about what the
        # mode does once it is asked for, so unless one names its own mode,
        # they get the one that stops before the thumb.
        args = ["--ros-args", "-p", "mock:=true"]
        if not any(p.startswith("calibration_mode:=") for p in params):
            args += ["-p", "calibration_mode:=fingers"]
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
    return [int(round(t)) for t in node._transport._targets]


def close(node):
    """Command every channel shut and let the hand get there.

    Not asserted as ``[0] * 6``: thumb rotation carries a command overlay, so
    a commanded 0.0 reaches the register as 250. That is the pose that puts
    the thumb across the palm, which is the whole reason staging exists.
    """
    node._on_command(JointState(name=list(CHANNEL_IDS), position=[0.0] * 6))
    tick(node)
    assert written(node)[INDEX] == 0


def calibrate(node, stage=True):
    """Request a calibration, and by default let the clearance staging finish.

    Staging is a real move on a real hand, so most tests want it over with;
    the ones that are about staging itself pass ``stage=False`` and drive the
    cycles themselves.
    """
    response = node._on_calibrate_force(Trigger.Request(), Trigger.Response())
    if stage and response.success:
        tick(node)  # the hand reaches the clearance pose and the register goes out
    return response


def expire(node):
    """Put the calibration deadline in the past, then run the cycle that sees it."""
    node._calibrating_until = time.monotonic() - 0.001
    tick(node)


def summary(node):
    """The driver's own diagnostics summary, as a plain dict."""
    tick(node)
    array = node._diagnostics(
        node.get_clock().now().to_msg(),
        node._transport.read_angles(),
        [0] * 6,
        node._forces,
        node._transport.read_health(),
    )
    return {kv.key: kv.value for kv in array.status[0].values}


def test_the_service_triggers_the_register(make_node):
    node = make_node()
    tick(node)
    response = calibrate(node)
    assert response.success, response.message
    assert node._transport.force_calibrations == 1
    assert "3.0s" in response.message, "the caller is told how long the hand will move"


def test_the_hand_is_opened_clear_before_the_register_is_written(make_node):
    """The routine never commands thumb rotation, so it has to be posed first.

    Bending the thumb out of a grasp pose swings it through the fingers. This
    is the only moment anything can move that axis.
    """
    node = make_node()
    close(node)

    response = calibrate(node, stage=False)
    assert response.success, response.message
    assert node._transport.force_calibrations == 0, "not until the fingers are there"
    assert [written(node)[i] for i in FINGER_DOF] == [ANGLE_MAX] * 4

    tick(node)
    assert node._transport.force_calibrations == 1


def test_staging_does_not_move_the_thumb_it_is_keeping_out(make_node):
    """Otherwise this node is the only reason the thumb moves at all."""
    node = make_node()
    close(node)
    before = written(node)
    calibrate(node, stage=False)
    after = written(node)
    assert [after[i] for i in THUMB_DOF] == [before[i] for i in THUMB_DOF]
    assert [after[i] for i in FINGER_DOF] == [ANGLE_MAX] * 4


def test_staging_does_swing_the_thumb_clear_when_it_is_taking_part(make_node):
    """Then thumb rotation has to come out of the fingers' sweep first."""
    node = make_node("calibration_mode:=full")
    close(node)
    calibrate(node, stage=False)
    assert written(node) == [ANGLE_MAX] * 6, "including channel 6, fully abducted"


def test_it_waits_for_the_hand_to_actually_get_there(make_node):
    """Commanding the pose is not reaching it; the thumb has travel to cross."""
    node = make_node()
    close(node)
    node._transport._slew = 200.0  # counts/s: a full sweep takes five seconds

    calibrate(node, stage=False)
    for _ in range(3):
        tick(node)
        assert node._transport.force_calibrations == 0, "still on its way"
    assert node._staging_until is not None


def test_a_dof_that_will_not_open_abandons_the_calibration(make_node):
    """Running the routine with one finger out of place is the collision itself."""
    node = make_node("calibration_clearance_sec:=0.05")
    close(node)
    # One finger that never gets there, however long it is given.
    node._transport.read_angles = lambda: [1000, 1000, 1000, 400, 1000, 1000]

    calibrate(node, stage=False)
    time.sleep(0.06)
    tick(node)
    assert node._transport.force_calibrations == 0
    assert node._staging_until is None, "abandoned, not left half-started"
    assert node._calibrating_until is None
    accepted, _ = node._apply(["index_proximal_joint"], [0.0])
    assert accepted, "and the node is writing again"


def test_staging_can_be_turned_off_for_a_hand_posed_by_hand(make_node):
    node = make_node("calibration_clearance:=false")
    close(node)
    before = written(node)
    calibrate(node, stage=False)
    assert node._transport.force_calibrations == 1
    assert written(node) == before, "nothing was moved on the caller's behalf"


def test_commands_are_refused_while_the_hand_is_being_posed(make_node):
    node = make_node()
    close(node)
    node._transport._slew = 200.0
    calibrate(node, stage=False)

    accepted, message = node._apply(["index_proximal_joint"], [0.0])
    assert not accepted
    assert "posed for a force calibration" in message


def test_it_says_the_hand_is_still_moving_when_it_returns(make_node):
    """The register is a trigger; nothing reports completion, so say so."""
    node = make_node("calibration_clearance:=false")
    tick(node)
    assert "returns before the routine does" in calibrate(node, stage=False).message


def test_it_says_the_staging_has_not_happened_yet_either(make_node):
    node = make_node()
    tick(node)
    assert "returns before any of it happens" in calibrate(node, stage=False).message


def test_commands_are_refused_while_the_hand_calibrates(make_node):
    node = make_node()
    node._on_command(JointState(name=["index_proximal_joint"], position=[1.0]))
    tick(node)
    calibrate(node)

    accepted, message = node._apply(["index_proximal_joint"], [0.0])
    assert not accepted
    assert "force calibration in progress" in message
    tick(node)
    assert written(node)[INDEX] == 1000, "nothing this node sent moved the finger"


def test_a_second_calibration_is_refused_while_the_first_runs(make_node):
    node = make_node()
    tick(node)
    calibrate(node)
    again = calibrate(node)
    assert not again.success
    assert "already running" in again.message
    assert node._transport.force_calibrations == 1


def test_the_stall_guard_stands_down_for_the_duration(make_node):
    """The routine drives fingers into their limits. That is not a fault."""
    node = make_node("state_extras_divisor:=1")
    tick(node)
    calibrate(node)
    node._transport._errors[INDEX] |= ERROR_LOCKED_ROTOR
    tick(node)
    assert node._stalls == 0
    assert node._transport.clear_error_writes == 0

    # And picks it straight back up once the hand is ours again.
    expire(node)
    tick(node)
    assert node._stalls == 1


def test_it_re_zeroes_against_the_new_sensors_when_it_finishes(make_node):
    """The old zero described sensors that have just been replaced."""
    node = make_node()
    tick(node)
    calibrate(node)
    node._transport.external_force[INDEX] = 300
    expire(node)
    assert node._spring.zeros[INDEX] == 300, "whatever it reads now is the new nothing"


def test_it_restores_the_pose_from_before_staging_opened_the_hand(make_node):
    """Not the clearance pose: staging is a command, and it moved the anchor."""
    node = make_node()
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.2]))
    tick(node)
    calibrate(node)
    assert node._last_command[INDEX] == ANGLE_MAX, "staging is what is commanded now"

    # The hand, driving itself, has left the finger somewhere else entirely.
    node._transport._targets[INDEX] = 1000.0
    expire(node)
    assert written(node)[INDEX] == 200
    assert node._last_command[INDEX] == 200


def test_it_is_refused_while_compliant_mode_is_working(make_node):
    node = make_node()
    tick(node)
    node._on_set_compliance(SetBool.Request(data=True), SetBool.Response())
    response = calibrate(node)
    assert not response.success
    assert "compliant mode" in response.message
    assert node._transport.force_calibrations == 0


def test_it_is_refused_while_the_yield_is_still_ramping_out(make_node):
    """Released is not the same as finished: the offset is still on the wire."""
    node = make_node("compliance_yield_rate:=1000000.0", "compliance_return_rate:=1.0")
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))
    tick(node)
    node._on_set_compliance(SetBool.Request(data=True), SetBool.Response())
    node._transport.external_force[INDEX] = 400
    tick(node)
    node._on_set_compliance(SetBool.Request(data=False), SetBool.Response())

    assert not node._spring.engaged and node._spring.active
    response = calibrate(node)
    assert not response.success
    assert node._transport.force_calibrations == 0


def test_compliant_mode_cannot_be_entered_mid_calibration(make_node):
    node = make_node()
    tick(node)
    calibrate(node)
    response = node._on_set_compliance(SetBool.Request(data=True), SetBool.Response())
    assert not response.success, "a refusal reported as success is how a hand lies"
    assert not node._spring.engaged


def refuse_calibration(node):
    def refuse():
        raise HandCommunicationError("no reply")

    node._transport.calibrate_force_sensors = refuse


def test_a_transport_failure_leaves_the_node_writing(make_node):
    """A calibration that never started must not take the hand offline with it."""
    node = make_node("calibration_clearance:=false")
    tick(node)
    refuse_calibration(node)
    response = calibrate(node, stage=False)
    assert not response.success
    assert node._calibrating_until is None
    accepted, _ = node._apply(["index_proximal_joint"], [0.0])
    assert accepted


def test_a_transport_failure_after_staging_also_lets_go(make_node):
    """By then the service has already answered, so only the node can clean up."""
    node = make_node()
    close(node)
    refuse_calibration(node)
    assert calibrate(node, stage=False).success, "staging started fine"
    tick(node)
    assert node._calibrating_until is None
    assert node._staging_until is None
    accepted, _ = node._apply(["index_proximal_joint"], [0.0])
    assert accepted


def test_diagnostics_say_whether_a_calibration_is_running(make_node):
    node = make_node()
    tick(node)
    assert summary(node)["force_calibration"] == "idle"
    calibrate(node)
    assert summary(node)["force_calibration"] == "running"
    expire(node)
    values = summary(node)
    assert values["force_calibration"] == "idle"
    assert values["force_calibrations"] == "1"


def test_the_service_is_refused_unless_a_mode_was_chosen(make_node):
    """It jams this hand. An open service to a self-driving hand needs consent."""
    node = make_node("calibration_mode:=none")
    tick(node)
    response = calibrate(node, stage=False)
    assert not response.success
    assert "calibration_mode is 'none'" in response.message
    assert "~/tare_force" in response.message, "and what to do instead"
    assert node._transport.force_calibrations == 0


def test_an_unusable_mode_falls_back_to_refusing(make_node):
    node = make_node("calibration_mode:=fingres")  # a plausible typo
    tick(node)
    assert node._calibration_mode == "none"
    assert not calibrate(node, stage=False).success


def test_the_thumb_is_left_out_of_the_routine_in_finger_mode(make_node):
    """One register, one fixed sequence: the only lever is when to stop it."""
    node = make_node()
    close(node)
    calibrate(node)
    remaining = node._calibrating_until - time.monotonic()
    assert 2.5 < remaining <= 3.0, "cut short, not the full six seconds"

    expire(node)
    assert node._transport.force_calibration_stops == 1, "it asks the routine to stop"


def test_the_whole_routine_can_be_asked_for(make_node):
    node = make_node("calibration_mode:=full")
    close(node)
    calibrate(node)
    remaining = node._calibrating_until - time.monotonic()
    assert 5.5 < remaining <= 6.0

    expire(node)
    assert node._transport.force_calibration_stops == 0, "let it run to the end"
    assert node._thumb_watch is None, "nothing to check: the thumb was meant to move"


def logged(node, level):
    """Collect one level of log output. The driver says things worth asserting."""
    lines = []
    setattr(node.get_logger(), level, lines.append)
    return lines


def test_it_says_so_when_the_firmware_ignores_the_stop(make_node):
    """Writing 0 to the register is undocumented, so it is checked, not assumed."""
    node = make_node()
    close(node)
    calibrate(node)
    node.THUMB_SETTLE_SEC = 0.0
    expire(node)

    # The thumb is still somewhere the routine put it, not where we asked.
    thumb = node._last_command[THUMB_BEND]
    node._transport.read_angles = lambda: [1000, 1000, 1000, 1000, thumb - 400, 1000]
    errors = logged(node, "error")
    tick(node)
    assert node._thumb_watch is None
    assert errors and "did NOT stop its calibration routine" in errors[0]
    assert "calibration_mode:=full" in errors[0], "and what to do about it"


def test_it_says_so_when_the_firmware_does_let_go(make_node):
    node = make_node()
    close(node)
    calibrate(node)
    node.THUMB_SETTLE_SEC = 0.0
    expire(node)

    infos = logged(node, "info")
    tick(node)
    assert node._thumb_watch is None
    assert any("released the thumb" in line for line in infos)
