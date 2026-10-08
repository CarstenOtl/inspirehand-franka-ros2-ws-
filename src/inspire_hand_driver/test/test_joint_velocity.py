"""Joint velocities differenced from the hand's ANGLE readings.

The RH56 has no speed sensor. Its register map exposes POS_ACT, ANGLE_ACT,
FORCE_ACT, CURRENT, ERROR, STATUS and TEMP, and SPEED_SET is a commanded
limit, not a measurement. So the only way to put a rate in ``JointState`` is
to difference the position the hand does report, which is what these tests
pin: the arithmetic, the clamp that keeps a dropped or jumped sample from
surfacing as a non-physical spike, and the opt-out.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rclpy = pytest.importorskip("rclpy")

from sensor_msgs.msg import JointState  # noqa: E402

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.driver_node import InspireHandNode  # noqa: E402
from inspire_hand_driver.protocol import ANGLE_INVALID  # noqa: E402

INDEX = kin.dof_index("index_proximal_joint")


@pytest.fixture
def make_node():
    instances = []

    def factory(*params):
        # A 20 Hz period leaves the sane-gap window wide enough that an
        # ordinary scheduling hiccup in the test does not look like a clock
        # jump to the driver.
        args = ["--ros-args", "-p", "mock:=true", "-p", "publish_rate_hz:=20.0"]
        for param in params:
            args += ["-p", param]
        rclpy.init(args=args)
        node = InspireHandNode()
        assert node._mock
        node.published = []
        node._joint_state_pub.publish = node.published.append
        instances.append(node)
        return node

    yield factory
    for node in instances:
        node.destroy_node()
    rclpy.shutdown()


def tick(node, pause=0.05):
    time.sleep(pause)
    node._on_timer()
    return node.published[-1]


def close_index(node):
    # The command topic speaks open ratios: 1.0 fully open, 0.0 fully closed.
    node._on_command(JointState(name=["index_proximal_joint"], position=[0.0]))


# -- the arithmetic -----------------------------------------------------------

def test_velocity_is_published_for_every_joint(make_node):
    node = make_node()
    message = tick(node)
    assert len(message.velocity) == len(kin.ALL_JOINTS)
    assert len(message.velocity) == len(message.position)


def test_the_first_sample_has_nothing_to_difference(make_node):
    node = make_node()
    assert list(tick(node).velocity) == [0.0] * len(kin.ALL_JOINTS)


def test_a_still_hand_reports_zero(make_node):
    node = make_node()
    tick(node)
    tick(node)
    assert tick(node).velocity[INDEX] == pytest.approx(0.0, abs=1e-9)


def test_a_moving_finger_reports_its_own_position_slope(make_node):
    node = make_node("velocity_filter_hz:=0.0")
    tick(node)
    close_index(node)
    first = tick(node)
    second = tick(node)
    dt = (second.header.stamp.sec + second.header.stamp.nanosec * 1e-9) - (
        first.header.stamp.sec + first.header.stamp.nanosec * 1e-9
    )
    expected = (second.position[INDEX] - first.position[INDEX]) / dt
    assert second.velocity[INDEX] == pytest.approx(expected, rel=1e-6)
    assert second.velocity[INDEX] > 0.0, "closing raises the joint angle"


def test_followers_turn_at_their_coupling_ratio(make_node):
    node = make_node()
    tick(node)
    close_index(node)
    tick(node)
    message = tick(node)
    coupling = kin.DOFS[INDEX].couplings[0]
    follower = len(kin.DOFS) + sum(
        len(d.couplings) for d in kin.DOFS[:INDEX]
    )
    assert message.name[follower] == coupling.joint
    assert message.velocity[follower] == pytest.approx(
        coupling.multiplier * message.velocity[INDEX], rel=1e-9
    )


# -- the guards ---------------------------------------------------------------

def test_the_commanded_speed_caps_the_reported_rate(make_node):
    """A rate above what the hand was told it may travel is not physical."""
    node = make_node("startup_speed:=100", "velocity_filter_hz:=0.0")
    ceiling = kin.speed_counts_to_rad_per_s(INDEX, 100)
    tick(node)
    close_index(node)
    for _ in range(4):
        message = tick(node)
        assert abs(message.velocity[INDEX]) <= ceiling + 1e-9
    # The mock slews far faster than a speed of 100 allows, so the clamp is
    # doing work here rather than passing an already-small number through.
    assert message.velocity[INDEX] == pytest.approx(ceiling, rel=1e-6)


def test_an_invalid_reading_is_rebased_rather_than_differenced(make_node):
    node = make_node()
    tick(node)
    close_index(node)
    tick(node)
    node._transport.read_angles = lambda: [ANGLE_INVALID] * 6
    assert tick(node).velocity[INDEX] == 0.0, "no measurement, no rate"
    del node._transport.read_angles
    assert tick(node).velocity[INDEX] == 0.0, "first sample after the gap"
    assert tick(node).velocity[INDEX] != 0.0, "differencing resumes"


def test_a_stalled_clock_is_not_an_infinite_rate(make_node):
    node = make_node()
    tick(node)
    close_index(node)
    tick(node)
    node._on_timer()  # same instant, so dt is far below one period
    assert node.published[-1].velocity[INDEX] == 0.0


# -- the opt-out and the speed readback ---------------------------------------

def test_publishing_velocity_can_be_turned_off(make_node):
    node = make_node("publish_velocity:=false")
    tick(node)
    assert list(tick(node).velocity) == []


def test_the_startup_readback_records_what_the_hand_is_running_at(make_node):
    """An unset SPEED_SET reads back as zeros, so the flash default is the answer."""
    node = make_node()
    assert node._transport.read_speeds() == [0] * 6
    assert node._transport.read_default_speeds() == [1000] * 6
    assert node._speed_observed == [1000] * 6


def test_a_speed_register_of_zero_never_clamps_motion_away(make_node):
    node = make_node()
    node._speed_cache = None
    node._speed_observed = [0] * 6
    assert node._speed_ceiling_counts() == [1000] * len(kin.DOFS)


def test_a_commanded_speed_outranks_the_hands_own(make_node):
    node = make_node("startup_speed:=250")
    assert node._speed_ceiling_counts() == [250] * 6


def test_an_unreadable_speed_register_falls_back_to_the_loosest_clamp(make_node):
    node = make_node("startup_speed:=0")
    node._speed_observed = None
    assert node._speed_ceiling_counts() == [1000] * len(kin.DOFS)


def test_the_filter_is_on_by_default_and_lags_a_step(make_node):
    """One pole means the first sample of a new rate is only part of the way."""
    node = make_node()
    assert node._velocity_filter_hz == pytest.approx(10.0)
    before = tick(node)
    close_index(node)
    first = tick(node)
    dt = (first.header.stamp.sec + first.header.stamp.nanosec * 1e-9) - (
        before.header.stamp.sec + before.header.stamp.nanosec * 1e-9
    )
    unfiltered = (first.position[INDEX] - before.position[INDEX]) / dt
    # A single pole admits only part of a step, so the reported rate is on the
    # way to the raw slope rather than at it, and still points the right way.
    assert 0.0 < first.velocity[INDEX] < unfiltered


def test_the_filter_can_be_disabled(make_node):
    node = make_node("velocity_filter_hz:=0.0")
    assert node._velocity_filter_hz == 0.0


# -- retuning on a running hand -----------------------------------------------
def test_the_filter_can_be_retuned_without_relaunching(make_node):
    """Judging a filter means the same move with and without it.

    A relaunch between the two is not a neutral act -- it loses the speed
    registers and the force tare -- so a parameter that silently kept its
    startup value would make the comparison meaningless.
    """
    from rclpy.parameter import Parameter

    node = make_node()
    assert node._velocity_filter_hz == pytest.approx(10.0)
    results = node.set_parameters([Parameter("velocity_filter_hz", value=0.0)])
    assert all(r.successful for r in results)
    assert node._velocity_filter_hz == 0.0

    # And the published rate follows: with no pole left, a move reports the
    # whole slope of the positions rather than part of the way to it.
    tick(node)
    close_index(node)
    first = tick(node)
    second = tick(node)
    dt = (second.header.stamp.sec + second.header.stamp.nanosec * 1e-9) - (
        first.header.stamp.sec + first.header.stamp.nanosec * 1e-9
    )
    expected = (second.position[INDEX] - first.position[INDEX]) / dt
    assert second.velocity[INDEX] == pytest.approx(expected, rel=1e-6)


def test_publishing_can_be_turned_off_on_a_running_hand(make_node):
    from rclpy.parameter import Parameter

    node = make_node()
    tick(node)
    assert tick(node).velocity
    assert all(
        r.successful
        for r in node.set_parameters([Parameter("publish_velocity", value=False)])
    )
    assert list(tick(node).velocity) == []


def test_a_negative_filter_is_refused_rather_than_clamped(make_node):
    """A refusal is visible; a clamp is indistinguishable from having worked."""
    from rclpy.parameter import Parameter

    node = make_node()
    results = node.set_parameters([Parameter("velocity_filter_hz", value=-1.0)])
    assert not results[0].successful
    assert "velocity_filter_hz" in results[0].reason
    assert node._velocity_filter_hz == pytest.approx(10.0)


def test_the_clamp_the_velocities_use_is_published(make_node):
    """A clamp nobody can see is indistinguishable from a slow finger.

    The startup log is not enough: ``~/set_speed`` moves the ceiling, so
    anything judging the velocities has to be able to read the current one.
    """
    node = make_node("startup_speed:=400")
    tick(node)
    published = []
    node._diag_pub.publish = published.append
    tick(node)
    values = {
        status.hardware_id.rsplit("/", 1)[-1]: {v.key: v.value for v in status.values}
        for status in published[-1].status
    }
    index_channel = kin.DOFS[INDEX].channel
    assert values[index_channel]["speed_ceiling"] == "400"
    assert float(values[index_channel]["speed_ceiling_rad_s"]) == pytest.approx(
        kin.speed_counts_to_rad_per_s(INDEX, 400), rel=1e-6
    )


def test_the_published_clamp_follows_a_speed_written_at_runtime(make_node):
    node = make_node("startup_speed:=400")
    tick(node)
    accepted, _ = node._write_limits(
        ["4"], [100], node._transport.write_speeds, "speed"
    )
    assert accepted
    published = []
    node._diag_pub.publish = published.append
    tick(node)
    values = {
        status.hardware_id.rsplit("/", 1)[-1]: {v.key: v.value for v in status.values}
        for status in published[-1].status
    }
    assert values["4"]["speed_ceiling"] == "100"
