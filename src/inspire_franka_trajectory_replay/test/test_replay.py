import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from inspire_franka_trajectory_replay.replay import _stream_hand
from inspire_franka_trajectory_replay.joint_trajectory_client import JointTrajectoryClient


def test_hardware_replay_uses_stock_position_controller_and_franka_timing():
    config_path = Path(__file__).parents[1] / "config" / "controllers_internal_impedance.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["/**"]
    manager = config["controller_manager"]["ros__parameters"]
    controller = config["trajectory_replay_controller"]["ros__parameters"]

    assert manager["trajectory_replay_controller"]["type"] == (
        "joint_trajectory_controller/JointTrajectoryController"
    )
    assert manager["thread_priority"] == 97
    assert manager["overruns"] == {"manage": False, "print_warnings": False}
    assert controller["command_interfaces"] == ["position"]
    assert controller["interpolate_from_desired_state"] is True
    assert controller["set_last_command_interface_value_as_state_on_activation"] is True


class RecordingNode:
    def __init__(self):
        self.commands = []

    def command_hand(self, positions):
        self.commands.append(np.asarray(positions))

    def get_clock(self):
        return self

    def now(self):
        return self

    @property
    def nanoseconds(self):
        return time.monotonic_ns()


def test_goto_endpoint_requests_rest_to_rest_quintic_spline():
    target = np.array([-0.3, 0.1, 0.2, -1.8, 1.0, 2.0, -0.2])
    point = JointTrajectoryClient._point(
        target, np.zeros(7), 5.0, accelerations=np.zeros(7)
    )

    assert point.positions == pytest.approx(target)
    assert point.velocities == pytest.approx(np.zeros(7))
    assert point.accelerations == pytest.approx(np.zeros(7))
    assert point.time_from_start.sec == 5
    assert point.time_from_start.nanosec == 0


def test_hand_stream_starts_only_after_arm_acceptance():
    node = RecordingNode()
    started = threading.Event()
    stopped = threading.Event()
    errors = []
    thread = threading.Thread(
        target=_stream_hand,
        args=(node, started, stopped, np.array([0.0]), np.zeros((1, 6)), errors),
    )
    thread.start()
    assert not node.commands

    started.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(node.commands) == 1
    assert not errors


def test_hand_stream_stops_while_waiting_for_arm_acceptance():
    node = RecordingNode()
    started = threading.Event()
    stopped = threading.Event()
    stopped.set()
    errors = []

    _stream_hand(node, started, stopped, np.array([0.0]), np.zeros((1, 6)), errors)

    assert not node.commands
    assert not errors


def test_command_hand_converts_radians_to_open_ratios():
    """The one place the runner's radians meet the driver's ratios.

    Everything above ``command_hand`` is in radians: the Forge trajectories,
    the homing YAMLs (which also hold the FR3's seven joints, so they cannot be
    ratios), and the tracking comparison against ``joint_states``. The driver
    commands in open ratios, running the opposite way. Getting this backwards
    would command a fully open hand where the recording wanted a closed one, so
    it is pinned at both ends of travel and in between.
    """
    from inspire_franka_trajectory_replay.replay import (
        HAND_LOWER,
        HAND_UPPER,
        CoordinatedReplayClient,
    )
    from inspire_franka_trajectory_replay.trajectory import HAND_JOINTS

    published = []

    class FakePublisher:
        def publish(self, message):
            published.append(message)

    from builtin_interfaces.msg import Time

    class FakeClock:
        def now(self):
            return self

        def to_msg(self):
            return Time()

    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    node._hand_publisher = FakePublisher()
    node.get_clock = FakeClock

    # The open pose is 0 rad and ratio 1.0; the closed pose is each joint's
    # upper limit and ratio 0.0. Opposite directions, which is the point.
    node.command_hand(HAND_LOWER)
    assert list(published[-1].position) == [1.0] * len(HAND_JOINTS)

    node.command_hand(HAND_UPPER)
    assert list(published[-1].position) == [0.0] * len(HAND_JOINTS)

    node.command_hand(0.5 * HAND_UPPER)
    assert list(published[-1].position) == pytest.approx([0.5] * len(HAND_JOINTS))

    assert list(published[-1].name) == list(HAND_JOINTS)


def test_command_hand_output_is_inside_the_drivers_accepted_range():
    """Every sample of the checked-in recording must survive the range check.

    The driver rejects a ratio outside [0, 1] instead of clamping it, so a
    conversion that overshot even slightly would now abort a replay mid-stream
    rather than quietly saturating.
    """
    from inspire_franka_trajectory_replay.replay import HAND_DOF
    from inspire_hand_driver import kinematics as kin

    for index in HAND_DOF:
        dof = kin.DOFS[index]
        for radians in np.linspace(dof.lower, dof.upper, 50):
            ratio = kin.rad_to_open_ratio(index, radians)
            assert 0.0 <= ratio <= 1.0


def test_float32_undershoot_at_a_limit_is_snapped_not_rejected():
    """Forge writes float32, so a joint driven onto its stop lands just past it."""
    from inspire_franka_trajectory_replay.replay import HAND_LOWER, _validate_hand

    values = np.tile(HAND_LOWER, (3, 1))
    values[1, 3] = -2.09e-6  # index_proximal, as the pickplace capture records it

    snapped = _validate_hand(values, "trajectory")

    assert snapped[1, 3] == 0.0
    assert np.all(snapped >= HAND_LOWER)


def test_a_real_overshoot_is_still_rejected_and_named():
    from inspire_franka_trajectory_replay.replay import HAND_UPPER, _validate_hand

    values = np.tile(HAND_UPPER, (2, 1))
    values[0, 4] = HAND_UPPER[4] + 0.05  # thumb pitch, 50 mrad past its limit

    with pytest.raises(ValueError, match="thumb_proximal_pitch_joint"):
        _validate_hand(values, "trajectory")


def test_home_that_matches_the_trajectory_start_is_accepted():
    from inspire_franka_trajectory_replay.replay import _check_home

    home = np.array([-0.0873, -0.6109, 0.0, -2.618, -0.5236, 2.9671, 0.0])
    _check_home(home, home + 0.01, "pickup.yaml", 0.1)


def test_home_from_a_different_task_config_is_refused_by_name():
    """The real mismatch: pickup.yaml against the Pickplace-Multi capture."""
    from inspire_franka_trajectory_replay.replay import _check_home

    home = np.array([-0.392613, 0.004288, -0.072713, -1.811251,
                     0.592754, 2.280553, -2.620279])
    first = np.array([-0.0873, -0.6109, 0.0, -2.618, -0.5236, 2.9671, 0.0])

    with pytest.raises(ValueError, match="fr3_joint7") as caught:
        _check_home(home, first, "pickup.yaml", 0.1)
    # Worst joint first, so the message opens with the 2.62 rad wrist swing.
    assert caught.value.args[0].index("fr3_joint7") < caught.value.args[0].index("fr3_joint5")
