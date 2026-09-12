import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from inspire_franka_trajectory_replay.replay import (
    HAND_UPPER, SUPPORT_FINGER_INDICES, _close_support_fingers,
    _hand_stream_native, _prepare_arm, _prepare_cartesian, _stream_hand,
    CoordinatedReplayClient, PositionReplayClient, main,
)
from inspire_franka_trajectory_replay.joint_trajectory_client import JointTrajectoryClient
from franka_trajectory_replay.cartesian_replay_client import (
    CONTROLLER_TYPE as CARTESIAN_CONTROLLER_TYPE, CartesianReplayClient,
)
from franka_trajectory_replay.kinematics import READY_POSE, flange_transform
from franka_trajectory_replay.replay_client import ReplayClient, Rejected
from franka_trajectory_replay.runconfig import load_config


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


def test_simulation_override_returns_requested_timing_without_auto_scaling(monkeypatch):
    calls = []

    def fake_prepare(_source, **arguments):
        calls.append(arguments)
        return SimpleNamespace(
            report={"ok": False},
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

    prepared = _prepare_arm(
        trajectory,
        config,
        120.0,
        time_scale=5.0,
        allow_limit_violations=True,
    )

    assert len(calls) == 1
    assert prepared.report["ok"] is False
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
    assert controller["stiffness_scale"] == 1.0
    assert controller["gain_ramp_duration"] == 1.0
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


def test_impedance_client_delegates_gravity_controller_switch(monkeypatch):
    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    node.controller = "trajectory_replay_controller"
    calls = []

    def switch(_self, log=print, controller=None):
        calls.append(controller)
        return ["trajectory_replay_controller"]

    monkeypatch.setattr(ReplayClient, "ensure_active", switch)

    stopped = node.ensure_active(controller="gravity_compensation_example_controller")

    assert calls == ["gravity_compensation_example_controller"]
    assert stopped == ["trajectory_replay_controller"]


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
    from inspire_franka_trajectory_replay.trajectory import scale_thumb_abduction

    node = client_type.__new__(client_type)
    node._hand_lock = threading.Lock()
    node._hand_position = np.ones(6)
    commands = []

    def command_hand(target):
        commands.append(np.asarray(target))
        with node._hand_lock:
            # Emulate the driver's universal actuator-facing overlay while
            # retaining the raw target received on its ROS command topic.
            node._hand_position = scale_thumb_abduction(target)

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


# --- Cartesian impedance path ---------------------------------------------------------

CONFIG_DIR = Path(__file__).parents[1] / "config"


def _replay_config():
    return load_config(str(CONFIG_DIR / "replay.yaml"))


def _gentle_capture(duration=3.0, rate=15.0, amplitude=0.05):
    """A 15 Hz capture that starts at the ready pose and sways gently around it."""
    time = np.arange(int(duration * rate) + 1) / rate
    arm = np.tile(READY_POSE, (len(time), 1))
    envelope = np.sin(np.pi * time / duration) ** 2
    for joint in range(7):
        arm[:, joint] += amplitude * envelope * np.sin(2 * np.pi * (0.2 + 0.05 * joint) * time)
    hand = np.full((len(time), 6), 0.1)
    return SimpleNamespace(time=time, arm=arm, hand=hand, source="synthetic.npz",
                           duration=float(time[-1]), cycle=None, segment=None)


def test_cartesian_stream_is_forward_kinematics_of_the_prepared_joint_stream():
    config = _replay_config()
    prepared = _prepare_arm(_gentle_capture(), config, 120.0)
    stream = _prepare_cartesian(prepared, config)

    assert stream.report["ok"], stream.report["violations"]
    assert np.shares_memory(stream.q_null, prepared.q)
    # Through the configured tool: the grasp centre, not the flange.
    for index in (0, 900, len(prepared.t) - 1):
        expected = (flange_transform(prepared.q[index]) @ stream.tool)[:3, 3]
        assert stream.p[index] == pytest.approx(expected, abs=1e-12)
    assert stream.t[-1] == pytest.approx(prepared.duration)
    assert stream.tool[:3, 3] == pytest.approx(config["tcp"]["offset_xyz"])
    assert stream.tool[:3, :3] == pytest.approx(np.eye(3))


def test_cartesian_margin_overrides_replace_only_the_given_values():
    config = _replay_config()
    prepared = _prepare_arm(_gentle_capture(), config, 120.0)
    stream = _prepare_cartesian(
        prepared, config,
        {"velocity_margin": 1e-6, "acceleration_margin": None, "jerk_margin": None},
    )
    assert not stream.report["ok"]
    assert all("velocity" in text for text in stream.report["violations"])
    assert stream.report["margins"]["acceleration"] == config["cartesian"]["acceleration_margin"]


def test_cartesian_stream_requires_the_flange_tcp_frame():
    config = _replay_config()
    prepared = _prepare_arm(_gentle_capture(), config, 120.0)
    config["tcp"]["frame"] = "hand_tcp"
    with pytest.raises(ValueError, match="tcp.frame"):
        _prepare_cartesian(prepared, config)


def test_cartesian_controllers_yaml_keeps_the_joint_profile_and_adds_the_example_gains():
    joint = yaml.safe_load((CONFIG_DIR / "controllers_joint_impedance.yaml").read_text())["/**"]
    both = yaml.safe_load((CONFIG_DIR / "controllers_cartesian_impedance.yaml").read_text())["/**"]
    manager = both["controller_manager"]["ros__parameters"]

    assert manager["trajectory_replay_controller"] == (
        joint["controller_manager"]["ros__parameters"]["trajectory_replay_controller"])
    assert manager["cartesian_trajectory_replay_controller"]["type"] == CARTESIAN_CONTROLLER_TYPE
    assert both["trajectory_replay_controller"] == joint["trajectory_replay_controller"]

    cartesian = both["cartesian_trajectory_replay_controller"]["ros__parameters"]
    assert cartesian["translational_stiffness"] == 150.0
    assert cartesian["rotational_stiffness"] == 10.0
    assert cartesian["nullspace_stiffness"] == 20.0
    assert cartesian["stiffness_scale"] == 1.0
    assert cartesian["target_filter"] == 0.005
    assert cartesian["nullspace_target"] == "trajectory"
    assert cartesian["coriolis_compensation"] is True
    assert cartesian["torque_rate_limit"] == 0.0
    assert cartesian["set_collision_behavior"] is False
    assert cartesian["base_frame"] == _replay_config()["cartesian"]["base_frame"]


def test_sim_impedance_yaml_runs_the_hardware_profiles_on_a_simulation_model():
    sim = yaml.safe_load((CONFIG_DIR / "controllers_sim_impedance.yaml").read_text())
    joint = yaml.safe_load((CONFIG_DIR / "controllers_joint_impedance.yaml").read_text())["/**"]
    both = yaml.safe_load((CONFIG_DIR / "controllers_cartesian_impedance.yaml").read_text())["/**"]
    manager = sim["controller_manager"]["ros__parameters"]

    assert manager["trajectory_replay_controller"] == (
        joint["controller_manager"]["ros__parameters"]["trajectory_replay_controller"])
    assert manager["cartesian_trajectory_replay_controller"]["type"] == CARTESIAN_CONTROLLER_TYPE
    assert "hand_position_forward_command_controller" in manager
    # The joint controller is the hardware profile verbatim; it needs no robot model.
    sim_joint = sim["trajectory_replay_controller"]["ros__parameters"]
    assert sim_joint == joint["trajectory_replay_controller"]["ros__parameters"]
    assert sim_joint["coriolis_compensation"] is False
    assert sim_joint["set_collision_behavior"] is False
    # The Cartesian controller differs only in its model source and the friction-driven gain.
    sim_cart = sim["cartesian_trajectory_replay_controller"]["ros__parameters"]
    hw_cart = both["cartesian_trajectory_replay_controller"]["ros__parameters"]
    assert sim_cart["model_source"] == "dh"
    assert "model_source" not in hw_cart
    differences = {key for key in set(sim_cart) | set(hw_cart) if sim_cart.get(key) != hw_cart.get(key)}
    assert differences == {"model_source", "stiffness_scale"}


def test_cartesian_client_rejects_the_joint_controller_type_before_activation():
    node = CartesianReplayClient.__new__(CartesianReplayClient)
    node.controller = "cartesian_trajectory_replay_controller"
    node.list_controllers = lambda: {
        node.controller: SimpleNamespace(type="franka_trajectory_replay/TrajectoryReplayController")
    }
    with pytest.raises(Rejected, match="arm_controller:=cartesian-impedance"):
        node.ensure_active()


def test_cartesian_client_rejects_stale_feedback_and_tracking_faults():
    node = CartesianReplayClient.__new__(CartesianReplayClient)
    node._lock = threading.Lock()
    node._status = {"phase_name": "idle", "completed_command_id": "1"}
    node._status_stamp = time.monotonic() - 2.0
    with pytest.raises(Rejected, match="feedback stopped"):
        node.status()

    node._status = {"phase_name": "idle", "tracking_fault": "true",
                    "last_fault": "position error exceeded max_position_error"}
    node._status_stamp = time.monotonic()
    with pytest.raises(Rejected, match="max_position_error"):
        node._check_fault()
    node._status["tracking_fault"] = "false"
    node._check_fault()


def _write_capture(tmp_path):
    capture = _gentle_capture()
    npz = tmp_path / "capture.npz"
    np.savez(npz, joint_pos_arm=capture.arm, joint_pos_hand=capture.hand,
             sample_time_s=capture.time)
    hand_names = [
        "pinky_proximal_joint", "ring_proximal_joint", "middle_proximal_joint",
        "index_proximal_joint", "thumb_proximal_pitch_joint", "thumb_proximal_yaw_joint",
    ]
    home = tmp_path / "homing.yaml"
    home.write_text(yaml.safe_dump({
        "joint_names": [f"fr3_joint{i}" for i in range(1, 8)] + hand_names,
        "positions": [float(v) for v in capture.arm[0]] + [0.1] * 6,
    }))
    return npz, home


def test_cartesian_dry_run_prints_both_summaries(tmp_path, capsys):
    npz, home = _write_capture(tmp_path)
    code = main([
        str(npz), "--home", str(home), "--config", str(CONFIG_DIR / "replay.yaml"),
        "--arm-controller", "cartesian-impedance", "--time-scale", "2",
        "--interactive-pause", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "time scale x2.000" in out
    assert "arm controller: cartesian-impedance" in out
    assert "cartesian:" in out and "within limits" in out
    assert "dry run" in out


def test_cartesian_only_flags_are_refused_on_the_other_paths(tmp_path):
    npz, home = _write_capture(tmp_path)
    base = [str(npz), "--home", str(home), "--config", str(CONFIG_DIR / "replay.yaml"), "--dry-run"]
    with pytest.raises(SystemExit):
        main(base + ["--stiffness-scale", "1.2"])
    with pytest.raises(SystemExit):
        main(base + ["--cartesian-velocity-margin", "0.5"])
    with pytest.raises(SystemExit):
        main(base + ["--arm-controller", "cartesian-impedance", "--no-arm"])
    with pytest.raises(SystemExit):
        main(base + ["--arm-controller", "cartesian-impedance", "--allow-unsafe-simulation"])


def test_controlled_point_is_the_grasp_centre_in_every_config():
    """replay.yaml's tcp block and both controller profiles name one frame."""
    config = _replay_config()
    hardware = yaml.safe_load(
        (CONFIG_DIR / "controllers_cartesian_impedance.yaml").read_text()
    )["/**"]["cartesian_trajectory_replay_controller"]["ros__parameters"]
    sim = yaml.safe_load(
        (CONFIG_DIR / "controllers_sim_impedance.yaml").read_text()
    )["cartesian_trajectory_replay_controller"]["ros__parameters"]

    assert config["tcp"]["frame"] == "fr3_link8"
    assert config["tcp"]["offset_xyz"] == hardware["tool_offset_xyz"] == sim["tool_offset_xyz"]
    assert config["tcp"]["offset_rpy"] == hardware["tool_offset_rpy"] == sim["tool_offset_rpy"]
    # The Inspire grasp centre, 172.7 mm from the flange.
    assert np.linalg.norm(config["tcp"]["offset_xyz"]) == pytest.approx(0.1727, abs=5e-4)


def test_prepared_pose_stream_follows_the_configured_tool():
    config = _replay_config()
    prepared = _prepare_arm(_gentle_capture(), config, 120.0)
    stream = _prepare_cartesian(prepared, config)
    expected = flange_transform(prepared.q[0]) @ stream.tool
    assert stream.p[0] == pytest.approx(expected[:3, 3], abs=1e-12)
    assert stream.tool[:3, 3] == pytest.approx(config["tcp"]["offset_xyz"])


# -- --duration --------------------------------------------------------------


def test_duration_scales_a_capture_to_the_length_asked_for():
    from inspire_franka_trajectory_replay.replay import time_scale_for_duration

    assert time_scale_for_duration(30.0, 12.0) == pytest.approx(2.5)
    assert time_scale_for_duration(12.0, 12.0) == pytest.approx(1.0)


def test_duration_refuses_to_speed_a_recording_up():
    """Only slowing down is supported, and the message says what was asked for."""
    from inspire_franka_trajectory_replay.replay import time_scale_for_duration

    with pytest.raises(ValueError, match="would speed it up"):
        time_scale_for_duration(5.0, 12.0)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_duration_rejects_a_nonsensical_target(target):
    from inspire_franka_trajectory_replay.replay import time_scale_for_duration

    with pytest.raises(ValueError):
        time_scale_for_duration(target, 12.0)


def test_duration_needs_a_trajectory_with_a_length():
    from inspire_franka_trajectory_replay.replay import time_scale_for_duration

    with pytest.raises(ValueError, match="no duration"):
        time_scale_for_duration(30.0, 0.0)


def test_duration_and_time_scale_are_mutually_exclusive():
    from inspire_franka_trajectory_replay.replay import main

    with pytest.raises(SystemExit):
        main(["traj", "--duration", "30", "--time-scale", "5"])


def test_duration_and_hand_time_scale_are_mutually_exclusive():
    from inspire_franka_trajectory_replay.replay import main

    with pytest.raises(SystemExit):
        main(["traj", "--duration", "30", "--hand-time-scale", "5", "--no-arm"])


@pytest.mark.parametrize("target", ["0", "-3", "nan"])
def test_a_nonpositive_duration_is_refused_on_the_command_line(target):
    from inspire_franka_trajectory_replay.replay import main

    with pytest.raises(SystemExit):
        main(["traj", "--duration", target])


# --- --intervene: what it refuses before anything is sent ---------------------------------


def _write_flagged_capture(tmp_path, release_sample=10):
    """A capture directory carrying release flags, as make_cycles writes them."""
    capture = _gentle_capture()
    directory = tmp_path / "flagged"
    directory.mkdir()
    np.savez(directory / "replay_data.npz", joint_pos_arm=capture.arm,
             joint_pos_hand=capture.hand, sample_time_s=capture.time)
    rate = 1.0 / float(np.median(np.diff(capture.time)))
    last = len(capture.time) - 1
    (directory / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "data_file": "replay_data.npz",
        "rate_hz": rate,
        "release_phase": "follow_waypoints",
        "cycle_index": [{
            "cycle": 1, "start_sample": 0, "end_sample": last,
            "release_sample": release_sample,
            "release_time_s": release_sample / rate,
        }],
    }))
    hand_names = [
        "pinky_proximal_joint", "ring_proximal_joint", "middle_proximal_joint",
        "index_proximal_joint", "thumb_proximal_pitch_joint", "thumb_proximal_yaw_joint",
    ]
    home = tmp_path / "flagged_home.yaml"
    home.write_text(yaml.safe_dump({
        "joint_names": [f"fr3_joint{i}" for i in range(1, 8)] + hand_names,
        "positions": [float(v) for v in capture.arm[0]] + [0.1] * 6,
    }))
    return directory, home


def test_intervene_prints_the_release_flags_it_would_rejoin_at(tmp_path, capsys):
    directory, home = _write_flagged_capture(tmp_path)

    code = main([
        str(directory), "--home", str(home),
        "--config", str(CONFIG_DIR / "replay.yaml"), "--intervene", "--dry-run",
    ])

    out = capsys.readouterr().out
    assert code == 0, out
    assert "1 cycles, each releasing in its 'follow_waypoints' phase" in out
    # The flag is in the artifact's own samples; what the operator needs is
    # where that lands on the controller's clock.
    assert "release at 10" in out and "prepared stream" in out
    assert "nothing to do in a dry run" in out


def test_intervene_refuses_a_trajectory_with_no_release_flags(tmp_path, capsys):
    npz, home = _write_capture(tmp_path)

    code = main([
        str(npz), "--home", str(home),
        "--config", str(CONFIG_DIR / "replay.yaml"), "--intervene", "--dry-run",
    ])

    out = capsys.readouterr().out
    assert code == 2
    assert "carries no release flags" in out
    assert "make_cycle_trajectory" in out


def test_intervene_refuses_flags_that_do_not_describe_the_npz(tmp_path, capsys):
    directory, home = _write_flagged_capture(tmp_path)
    document = json.loads((directory / "metadata.json").read_text())
    document["cycle_index"][0]["end_sample"] = 10_000
    (directory / "metadata.json").write_text(json.dumps(document))

    code = main([
        str(directory), "--home", str(home),
        "--config", str(CONFIG_DIR / "replay.yaml"), "--intervene", "--dry-run",
    ])

    out = capsys.readouterr().out
    assert code == 2
    assert "does not describe this NPZ" in out


def test_intervene_needs_both_devices_and_the_validated_controller(tmp_path):
    directory, home = _write_flagged_capture(tmp_path)
    base = [str(directory), "--home", str(home),
            "--config", str(CONFIG_DIR / "replay.yaml"), "--intervene", "--dry-run"]

    for extra in (
        ["--no-arm"],                                   # nothing to hand-guide
        ["--no-hand"],                                  # no hand state to record
        ["--arm-controller", "cartesian-impedance"],    # not the validated path
        ["--arm-controller", "position-jtc"],
        ["--cycle", "1"],                               # renumbers the samples
        ["--segment", "1"],
        ["--max-release-delta", "-0.1"],
        ["--correction-dwell", "0"],
        ["--correction-peak-speed", "-1"],
    ):
        with pytest.raises(SystemExit):
            main(base + extra)


def test_intervene_implies_interactive_pause(tmp_path, capsys):
    directory, home = _write_flagged_capture(tmp_path)

    # --interactive-pause is refused with --no-arm, so if --intervene did not
    # set it, this would fail on the --no-arm check instead of the arm one.
    code = main([
        str(directory), "--home", str(home),
        "--config", str(CONFIG_DIR / "replay.yaml"), "--intervene", "--dry-run",
    ])

    assert code == 0, capsys.readouterr().out


# --- the keyboard: one press to step in ---------------------------------------------------


class _KeyboardNode:
    """Records what the keyboard asks of the arm, and in what order."""

    def __init__(self, elapsed=83.0):
        self.calls = []
        self._elapsed = elapsed

    def status(self):
        self.calls.append("status")
        return {"elapsed": self._elapsed, "phase_name": "trajectory",
                "active_command_id": "2", "completed_command_id": "1"}

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def abort(self):
        self.calls.append("abort")


def _drive_keyboard(keys, allow_intervene=True, timeout=3.0):
    """Run _InteractivePause against a pty and feed it keys."""
    import os

    from inspire_franka_trajectory_replay.replay import _InteractivePause

    node = _KeyboardNode()
    master, slave = os.openpty()
    saved = sys.stdin
    sys.stdin = os.fdopen(slave, "r", buffering=1)
    try:
        started = threading.Event()
        started.set()
        controls = _InteractivePause(node, started, allow_intervene=allow_intervene)
        with controls:
            controls.segment_started()
            for key in keys:
                os.write(master, key.encode())
                time.sleep(0.15)
            deadline = time.monotonic() + timeout
            while (controls.thread.is_alive()
                   and not controls.intervene_requested.is_set()
                   and time.monotonic() < deadline):
                time.sleep(0.02)
        return node, controls
    finally:
        sys.stdin.close()
        sys.stdin = saved
        os.close(master)


@pytest.mark.parametrize("key", ["\r", "\n"])
def test_enter_steps_in_with_one_press(key):
    """Enter stops the trajectory and frees the arm without a separate pause key.

    One keystroke, but a sequenced one: the clock is ramped to zero and
    node.pause() has returned -- which only happens once the controller reports
    the trajectory actually stopped -- before the arm is handed over.
    """
    node, controls = _drive_keyboard([key])

    assert controls.intervene_requested.is_set()
    assert node.calls.index("pause") < node.calls.index("abort")
    # The paused clock is read before the abort replaces it with the stop ramp's.
    assert node.calls.index("status") < node.calls.index("abort")
    assert controls.intervene_elapsed == pytest.approx(83.0)


def test_space_still_pauses_without_freeing_the_arm():
    node, controls = _drive_keyboard([" "], timeout=0.3)

    assert node.calls.count("pause") == 1
    assert "abort" not in node.calls
    assert not controls.intervene_requested.is_set()


def test_i_steps_in_only_after_space():
    node, controls = _drive_keyboard(["i"], timeout=0.3)
    assert not controls.intervene_requested.is_set()
    assert "abort" not in node.calls

    node, controls = _drive_keyboard([" ", "i"])
    assert controls.intervene_requested.is_set()
    assert node.calls.index("pause") < node.calls.index("abort")


def test_enter_does_nothing_without_intervene_enabled():
    node, controls = _drive_keyboard(["\r"], allow_intervene=False, timeout=0.3)

    assert not controls.intervene_requested.is_set()
    assert node.calls == []


def test_q_aborts_whether_or_not_interventions_are_enabled():
    for allow in (True, False):
        node, controls = _drive_keyboard(["q"], allow_intervene=allow, timeout=0.5)
        assert controls.aborted.is_set()
        assert "abort" in node.calls
