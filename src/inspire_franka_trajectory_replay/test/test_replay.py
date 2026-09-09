import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inspire_franka_trajectory_replay.replay import (
    HAND_UPPER, SUPPORT_FINGER_INDICES, _close_support_fingers,
    _hand_stream_native, _prepare_arm, _stream_hand, CoordinatedReplayClient,
    PositionReplayClient,
)
from inspire_franka_trajectory_replay.joint_trajectory_client import JointTrajectoryClient
from franka_trajectory_replay.replay_client import ReplayClient, Rejected


def test_support_finger_override_changes_only_requested_hand_channels():
    hand = np.arange(18, dtype=float).reshape(3, 6) / 100.0
    home = np.arange(6, dtype=float) / 10.0
    original_hand = hand.copy()
    original_home = home.copy()

    overridden, overridden_home = _close_support_fingers(hand, home)

    untouched = [index for index in range(6) if index not in SUPPORT_FINGER_INDICES]
    assert overridden[:, SUPPORT_FINGER_INDICES] == pytest.approx(
        np.tile(HAND_UPPER[list(SUPPORT_FINGER_INDICES)], (len(hand), 1))
    )
    assert overridden_home[list(SUPPORT_FINGER_INDICES)] == pytest.approx(
        HAND_UPPER[list(SUPPORT_FINGER_INDICES)]
    )
    assert overridden[:, untouched] == pytest.approx(original_hand[:, untouched])
    assert overridden_home[untouched] == pytest.approx(original_home[untouched])
    assert hand == pytest.approx(original_hand)
    assert home == pytest.approx(original_home)


def test_support_finger_override_requires_recorded_hand_positions():
    with pytest.raises(ValueError, match="requires Inspire hand positions"):
        _close_support_fingers(None, np.zeros(6))


def test_explicit_time_scale_stretches_arm_waypoint_timing(monkeypatch):
    calls = []

    def fake_prepare(_source, **arguments):
        calls.append(arguments)
        return SimpleNamespace(
            report={"ok": True},
            params={"time_scale": arguments["time_scale"]},
        )

    monkeypatch.setattr(
        "inspire_franka_trajectory_replay.replay.prepare", fake_prepare
    )
    trajectory = SimpleNamespace(
        time=np.array([0.0, 1.0]),
        arm=np.zeros((2, 7)),
        source="capture.npz",
        duration=1.0,
    )
    config = {
        "joint_names": [f"fr3_joint{i}" for i in range(1, 8)],
        "prepare": {
            "rate": 1000,
            "cutoff_hz": 0.0,
            "hold_start": 0.5,
            "hold_end": 0.5,
            "time_scale": 1.0,
            "auto_scale": True,
            "velocity_margin": 0.8,
            "acceleration_margin": 0.5,
            "jerk_margin": 0.5,
            "lead_in": 0.5,
            "lead_out": 0.5,
            "lead_max_acceleration": 2.5,
            "interpolation": "cubic",
            "blend_time": 0.04,
        },
    }

    prepared = _prepare_arm(trajectory, config, 120.0, time_scale=5.0)

    assert calls[0]["time_scale"] == 5.0
    assert prepared.params["time_scale"] == 5.0


def test_time_scale_stretches_native_hand_waypoint_timing():
    trajectory = SimpleNamespace(
        time=np.array([0.0, 1.0 / 15.0]),
        hand=np.vstack((np.zeros(6), np.ones(6))),
    )

    stream_time, positions = _hand_stream_native(trajectory, rate=30.0, time_scale=5.0)

    assert stream_time[-1] == pytest.approx(5.0 / 15.0)
    assert len(stream_time) == 11
    assert positions[5] == pytest.approx(np.full(6, 0.5))


def test_hardware_replay_matches_the_working_example_profile():
    config_path = Path(__file__).parents[1] / "config" / "controllers_joint_impedance.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["/**"]
    manager = config["controller_manager"]["ros__parameters"]
    controller = config["trajectory_replay_controller"]["ros__parameters"]

    assert manager["trajectory_replay_controller"]["type"] == (
        "franka_trajectory_replay/TrajectoryReplayController"
    )
    assert manager["thread_priority"] == 97
    assert manager["overruns"] == {"manage": False, "print_warnings": False}
    assert controller["command_interface"] == "effort"
    assert controller["k_gains"] == [24, 24, 24, 24, 10, 6, 2]
    assert controller["d_gains"] == [2, 2, 2, 1, 1, 1, 0.5]
    assert controller["coriolis_compensation"] is False
    assert controller["pause_ramp_duration"] == 0.5
    assert controller["set_collision_behavior"] is False
    assert controller["torque_rate_limit"] == 0.0


def test_hardware_client_uses_waypoint_transport_and_sim_keeps_action_transport():
    assert issubclass(CoordinatedReplayClient, ReplayClient)
    assert not issubclass(CoordinatedReplayClient, JointTrajectoryClient)
    assert issubclass(PositionReplayClient, JointTrajectoryClient)


def test_impedance_client_rejects_a_running_position_controller_before_activation():
    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    node.controller = "trajectory_replay_controller"
    node.list_controllers = lambda: {
        node.controller: SimpleNamespace(type="joint_trajectory_controller/JointTrajectoryController")
    }
    with pytest.raises(Rejected, match="waypoint effort controller"):
        node.ensure_active()


def test_impedance_client_rejects_stale_completion_feedback():
    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    node._lock = threading.Lock()
    node._status = {"phase_name": "idle", "completed_command_id": "1"}
    node._status_stamp = time.monotonic() - 2.0
    with pytest.raises(Rejected, match="feedback stopped"):
        node.status()


def test_waypoint_client_waits_for_completion_and_releases_hand_after_acceptance(monkeypatch):
    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    pending = {"active_command_id": "0", "completed_command_id": "0",
               "rejections": "0", "phase_name": "idle", "elapsed": "0", "duration": "1"}
    accepted = dict(pending, active_command_id="1", phase_name="trajectory")
    complete = dict(accepted, completed_command_id="1", phase_name="idle")
    current = dict(pending)
    events = []
    node.status = lambda: dict(current)

    def tick(_seconds):
        current.update(accepted if not events else complete)

    monkeypatch.setattr("franka_trajectory_replay.replay_client.time.sleep", tick)
    result = node._wait_command(
        0, 0, 1.0, "trajectory", on_accept=lambda: events.append("hand-start"),
    )
    assert result == 1
    assert events == ["hand-start"]
    assert current["phase_name"] == "idle"


def test_abort_waits_for_new_acknowledgment_even_if_old_status_is_idle(monkeypatch):
    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    node._lock = threading.Lock()
    node._status = {"processed_command_id": "3", "completed_command_id": "3",
                    "phase_name": "idle"}
    node._status_stamp = time.monotonic()
    events = []
    node._abort_publisher = SimpleNamespace(publish=lambda _message: events.append("abort"))

    def tick(_seconds):
        if node._status["processed_command_id"] == "3":
            # Status can mix an old phase with a newer processed id; require
            # completion too before returning from abort.
            events.append("acknowledgment")
            node._status["processed_command_id"] = "4"
        else:
            events.append("completed")
            node._status["completed_command_id"] = "4"

    monkeypatch.setattr("franka_trajectory_replay.replay_client.time.sleep", tick)
    node.abort()
    assert events == ["abort", "acknowledgment", "completed"]


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


def test_hand_stream_follows_the_arm_trajectory_clock_across_pause():
    node = RecordingNode()
    started = threading.Event()
    started.set()
    stopped = threading.Event()
    errors = []
    clock = {"elapsed": 0.0}
    positions = np.vstack((np.zeros(6), np.ones(6), np.full(6, 2.0)))
    thread = threading.Thread(
        target=_stream_hand,
        args=(
            node,
            started,
            stopped,
            np.array([0.0, 0.1, 0.2]),
            positions,
            errors,
            lambda: clock["elapsed"],
        ),
    )
    thread.start()
    time.sleep(0.03)
    assert len(node.commands) == 1

    # A paused arm clock does not let wall time advance the hand.
    time.sleep(0.03)
    assert len(node.commands) == 1
    clock["elapsed"] = 0.1
    time.sleep(0.03)
    assert len(node.commands) == 2
    clock["elapsed"] = 0.2
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert not errors
    assert np.asarray(node.commands) == pytest.approx(positions)


@pytest.mark.parametrize("client_type", [CoordinatedReplayClient, PositionReplayClient])
def test_wait_for_hand_republishes_home_command_until_feedback_arrives(client_type):
    """Hand homing uses wait_until's progress callback to recover a lost command."""
    node = client_type.__new__(client_type)
    node._hand_lock = threading.Lock()
    node._hand_position = np.ones(6)
    commands = []

    def command_hand(target):
        commands.append(np.asarray(target))
        with node._hand_lock:
            node._hand_position = np.asarray(target)

    node.command_hand = command_hand
    target = np.zeros(6)

    node.wait_for_hand(target, timeout=0.5, tolerance=0.01)

    assert len(commands) == 1
    assert commands[0] == pytest.approx(target)


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
