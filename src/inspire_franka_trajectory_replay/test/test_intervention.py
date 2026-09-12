"""Hand-guided interventions: what is recorded, what is refused, what moves.

No ROS and no arm here. The node, the capture node and the keyboard are all
stood in for, because what these tests are about is the decisions -- which pose
is refusable, what the event log says afterwards, and what the correction the
handback replays actually contains.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from franka_trajectory_replay import limits
from franka_trajectory_replay.kinematics import READY_POSE
from inspire_franka_trajectory_replay.hand_presets import load_presets
from inspire_franka_trajectory_replay.intervention import (
    GRAVITY_CONTROLLER,
    Intervention,
    InterventionSession,
    RESERVED_KEYS,
    gap_report,
    key_banner,
    pose_problems,
    session_directory,
)
from inspire_franka_trajectory_replay.release_phase import CycleRelease
from inspire_franka_trajectory_replay.trajectory import ARM_JOINTS
from inspire_hand_driver import kinematics as kin

HAND_POSE = np.array([0.3, 0.3, 0.3, 0.5, 0.4, 0.2])


class _Message:
    """The parts of a JointState that ``CaptureNode._as_dict`` reads."""

    class _Stamp:
        sec = 12
        nanosec = 34

    def __init__(self, names, positions):
        self.header = type("H", (), {"stamp": _Message._Stamp()})()
        self.name = list(names)
        self.position = [float(value) for value in positions]
        self.velocity = []
        self.effort = []


class FakeCaptureNode:
    """Stands in for the recording node: snapshots and hand commands only."""

    def __init__(self, arm=None, hand=None, complete=True):
        self.arm = READY_POSE.copy() if arm is None else np.asarray(arm, dtype=float)
        self.hand = HAND_POSE.copy() if hand is None else np.asarray(hand, dtype=float)
        self.complete = complete
        self.commands = []
        self.speeds = []
        self.forces = []
        self._clock = 1_000_000_000

    def clock_ns(self):
        self._clock += 1_000_000
        return self._clock

    def snapshot(self):
        if not self.complete:
            return {"arm_joint_states": None, "hand_joint_states": None}
        from inspire_franka_trajectory_replay.capture import CaptureNode

        return {
            "arm_joint_states": CaptureNode._as_dict(_Message(ARM_JOINTS, self.arm)),
            "hand_joint_states": CaptureNode._as_dict(
                _Message(kin.DRIVEN_JOINTS, self.hand)
            ),
            "hand_state_open_ratio": None,
            "hand_grip_force": None,
        }

    def publish_hand(self, joint_names, open_ratio):
        self.commands.append(tuple(float(value) for value in open_ratio))

    def measured_open_ratios(self):
        return [0.5] * 6

    def set_speed(self, joint_names, speed, timeout=2.0):
        self.speeds.append(int(speed))
        return True, "ok"

    def set_force(self, joint_names, force, timeout=2.0):
        self.forces.append(int(force))
        return True, "ok"


class FakeReplayNode:
    """Records the controller switches an intervention makes, in order."""

    controller = "trajectory_replay_controller"

    def __init__(self, arm=None):
        self.arm = READY_POSE.copy() if arm is None else np.asarray(arm, dtype=float)
        self.switches = []

    def ensure_active(self, log=print, controller=None):
        self.switches.append(controller or self.controller)
        return ["trajectory_replay_controller"] if controller else [GRAVITY_CONTROLLER]

    def current_joint_positions(self, timeout=10.0):
        return self.arm.copy()


def _keys(*sequence):
    """A ``read_key`` that plays a fixed sequence and then blocks on nothing."""
    remaining = list(sequence)

    def read_key(timeout=0.2):
        return remaining.pop(0) if remaining else "q"

    return read_key


def _session(tmp_path, node=None, capture=None, **kwargs):
    return InterventionSession(
        tmp_path / "session",
        node or FakeReplayNode(),
        capture or FakeCaptureNode(),
        load_presets(),
        log=lambda *_: None,
        **kwargs,
    )


def _cycle():
    return CycleRelease(cycle=3, start_sample=200, end_sample=290, release_sample=246)


# --- the pose check ------------------------------------------------------------------------


def test_a_reachable_pose_has_no_problems():
    assert pose_problems(READY_POSE, 0.8) == []


def test_a_joint_outside_its_position_limit_is_named():
    pose = READY_POSE.copy()
    pose[4] = limits.POSITION_UPPER[4] + 0.1

    problems = pose_problems(pose, 0.8)

    assert len(problems) == 1
    assert "fr3_joint5" in problems[0]
    assert "outside its position limit" in problems[0]


def test_a_pose_inside_the_braking_zone_is_refused_even_though_it_is_within_limits():
    """The realistic failure: joint 5 parked hard against its stop.

    There the FR3's velocity envelope has closed onto zero: the arm can hold the
    pose but cannot move, so it can neither arrive at it nor leave it -- which is
    why this is checked when the pose is captured rather than at the handback.
    """
    pose = READY_POSE.copy()
    pose[4] = limits.POSITION_UPPER[4] - 0.002

    problems = pose_problems(pose, 0.8)

    assert len(problems) == 1
    assert "braking zone" in problems[0]
    # 0.02 rad further out the envelope is open again and the pose is usable. It is worth
    # pinning: an envelope paired with the wrong position limits refuses it, and that is
    # what put a 2.8 rad working cap on joint 5 in the first place.
    assert pose_problems(np.where(np.arange(7) == 4, 2.8563, READY_POSE), 0.8) == []
    assert pose_problems(np.where(np.arange(7) == 4, 2.80, READY_POSE), 0.8) == []


def test_every_sample_of_the_shipped_artifact_passes_the_pose_check():
    from inspire_franka_trajectory_replay.trajectory import load_trajectory

    path = Path("apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8")
    if not path.exists():  # pragma: no cover - only outside the workspace
        pytest.skip("the demo trajectories are not on this path")
    trajectory = load_trajectory(str(path))

    refused = [i for i in range(len(trajectory.arm)) if pose_problems(trajectory.arm[i], 0.8)]

    assert refused == []


# --- capturing waypoints -------------------------------------------------------------------


def test_a_capture_writes_a_pose_capture_event_extract_waypoints_can_read(tmp_path):
    from inspire_franka_trajectory_replay.waypoints import load_waypoints

    session = _session(tmp_path)
    record = Intervention(1, 246, 83.0, _cycle(), _cycle())

    session._capture(record)
    session._capture(record)
    session.close()

    assert len(record.waypoints) == 2
    # The event log is the handover to the rest of the toolchain, so it has to
    # be readable by the tool that reads a capture session's log.
    recovered = load_waypoints(session.directory)
    assert len(recovered) == 2
    assert recovered[0].arm == pytest.approx(READY_POSE)
    assert recovered[0].hand == pytest.approx(HAND_POSE)


def test_a_capture_with_no_joint_states_yet_is_refused_and_not_counted(tmp_path):
    session = _session(tmp_path, capture=FakeCaptureNode(complete=False))
    record = Intervention(1, 0, 0.0, _cycle(), _cycle())

    session._capture(record)

    assert record.waypoints == []
    assert session._captures == 0


def test_a_pose_the_arm_could_not_hold_is_recorded_as_refused_but_not_replayed(tmp_path):
    pose = READY_POSE.copy()
    pose[4] = limits.POSITION_UPPER[4] - 0.002
    session = _session(tmp_path, capture=FakeCaptureNode(arm=pose))
    record = Intervention(1, 0, 0.0, _cycle(), _cycle())

    session._capture(record)
    session.close()

    assert record.waypoints == []
    events = [json.loads(line) for line in
              (session.directory / "events.jsonl").read_text().splitlines()]
    # The keypress happened, so it is in the log; the waypoint is not in the
    # correction, because the correction would then be refused at the handback.
    assert [event["event"] for event in events] == ["pose_capture", "pose_capture_refused"]
    assert "braking zone" in events[1]["problems"][0]


# --- the keyboard loop ---------------------------------------------------------------------


def test_g_records_poses_and_hands_back(tmp_path):
    session = _session(tmp_path)
    record = Intervention(1, 246, 83.0, _cycle(), _cycle())

    session._keyboard_loop(record, _keys("\r", "\r", "g"))

    assert record.released is True
    assert record.aborted is False
    assert len(record.waypoints) == 2


def test_handing_back_with_nothing_recorded_is_allowed(tmp_path):
    """Nothing is replayed from the poses, so an empty supervision is valid.

    The operator may only have repositioned the workpiece by hand. It leaves no
    record, which is worth saying, but it is not an error.
    """
    session = _session(tmp_path)
    record = Intervention(1, 246, 83.0, _cycle(), _cycle())

    session._keyboard_loop(record, _keys("g"))

    assert record.released is True
    assert record.waypoints == []


def test_q_aborts_and_keeps_whatever_was_captured(tmp_path):
    session = _session(tmp_path)
    record = Intervention(1, 246, 83.0, _cycle(), _cycle())

    session._keyboard_loop(record, _keys("\r", "\n", "q"))

    assert record.aborted is True
    assert record.released is False
    assert len(record.waypoints) == 2


def test_a_preset_key_commands_the_hand_and_is_logged(tmp_path):
    capture = FakeCaptureNode()
    session = _session(tmp_path, capture=capture)
    record = Intervention(1, 0, 0.0, _cycle(), _cycle())
    preset = next(iter(load_presets()))

    session._keyboard_loop(record, _keys(preset.key, "\r", "g"))
    session.close()

    assert capture.commands, "the preset key published no hand command"
    events = [json.loads(line)["event"] for line in
              (session.directory / "events.jsonl").read_text().splitlines()]
    assert "hand_command" in events


@pytest.mark.parametrize("key", ["-", "=", "[", "]"])
def test_each_jog_key_commands_the_expected_hand_dof(tmp_path, key):
    capture = FakeCaptureNode()
    session = _session(tmp_path, capture=capture)
    record = Intervention(1, 0, 0.0, _cycle(), _cycle())
    control, delta = session.presets.jog.control_for(key)

    session._keyboard_loop(record, _keys(key, "g"))
    session.close()

    assert len(capture.commands) == 1
    assert capture.commands[0][control.dof] == pytest.approx(0.5 + delta)
    events = [json.loads(line) for line in
              (session.directory / "events.jsonl").read_text().splitlines()]
    jog = next(event for event in events if event["event"] == "hand_command")
    assert jog["source"] == "jog"
    assert jog["joint"] == control.joint
    assert jog["key"] == key
    assert jog["step"] == pytest.approx(delta)


def test_the_reserved_keys_are_the_ones_the_preset_file_keeps_free():
    presets = load_presets()
    claimed = {preset.key for preset in presets}
    if presets.jog is not None:
        claimed |= set(presets.jog.keys)

    # Enter, g, q and ? must not be bound to a posture, or a keypress would do
    # two things at once.
    assert claimed.isdisjoint(RESERVED_KEYS)
    assert "?" in key_banner(presets)


# --- floating and handing back -------------------------------------------------------------


def test_the_arm_is_floated_and_always_taken_back(tmp_path):
    node = FakeReplayNode()
    session = _session(tmp_path, node=node)

    session.run(246, 83.0, _cycle(), _cycle(), _keys("\r", "g"))
    session.close()

    # Gravity compensation first, the replay controller again at the end.
    assert node.switches == [GRAVITY_CONTROLLER, "trajectory_replay_controller"]


def test_the_arm_is_taken_back_even_when_the_keyboard_loop_raises(tmp_path):
    node = FakeReplayNode()
    session = _session(tmp_path, node=node)

    def explode(timeout=0.2):
        raise RuntimeError("terminal went away")

    with pytest.raises(RuntimeError):
        session.run(246, 83.0, _cycle(), _cycle(), explode)

    assert node.switches == [GRAVITY_CONTROLLER, "trajectory_replay_controller"]


def test_the_session_log_and_manifest_describe_what_happened(tmp_path):
    session = _session(tmp_path)

    session.run(246, 83.0, _cycle(), _cycle(), _keys("\r", "\r", "g"))
    manifest = json.loads(session.write_manifest({"exit_code": 0}).read_text())
    session.close()

    events = [json.loads(line) for line in
              (session.directory / "events.jsonl").read_text().splitlines()]
    names = [event["event"] for event in events]
    assert names[0] == "intervention_open"
    assert "arm_floating" in names and "arm_stiff" in names
    assert names[-1] == "arm_stiff"
    assert names.count("pose_capture") == 2
    assert "intervention_release" in names

    assert manifest["gravity_controller"] == GRAVITY_CONTROLLER
    assert manifest["interventions"][0]["waypoints"] == 2
    assert manifest["interventions"][0]["rejoin_sample"] == 246
    assert manifest["interventions"][0]["released"] is True
    assert manifest["exit_code"] == 0
    # The handback records where the arm and the hand were left, which is the
    # pose the guarded goto onto the release point starts from.
    stiff = next(event for event in events if event["event"] == "arm_stiff")
    assert stiff["measured"]["fr3_joint1"] == pytest.approx(READY_POSE[0])
    assert stiff["measured_hand"]["index_proximal_joint"] == pytest.approx(HAND_POSE[3])


def test_the_measured_pose_is_recorded_at_the_handback(tmp_path):
    parked = READY_POSE + 0.05
    session = _session(
        tmp_path, node=FakeReplayNode(arm=parked), capture=FakeCaptureNode(arm=parked)
    )

    arm, hand = session.measured_pose()

    assert arm == pytest.approx(parked)
    assert hand == pytest.approx(HAND_POSE)


def test_a_pause_past_the_last_release_has_nothing_to_rejoin(tmp_path):
    session = _session(tmp_path)

    record = session.run(1100, 360.0, None, None, _keys("\r", "g"))
    session.close()

    assert record.rejoin_sample is None
    assert record.released is True


# --- reporting -----------------------------------------------------------------------------


def test_the_gap_report_names_the_worst_joint_first():
    current = np.zeros(7)
    target = np.array([0.1, 0.0, 0.5, 0.0, 0.2, 0.0, 0.0])

    report = gap_report(current, target, "onto the release point")

    assert report.startswith("onto the release point: 0.500 rad at worst")
    assert report.index("fr3_joint3") < report.index("fr3_joint5") < report.index("fr3_joint1")
    # Joints that do not move are not listed at all.
    assert "fr3_joint2" not in report


def test_the_gap_report_says_so_when_there_is_nothing_to_move():
    report = gap_report(READY_POSE, READY_POSE, "onto the correction")

    assert "already there" in report


def test_the_session_directory_is_stamped_and_labelled(tmp_path):
    plain = session_directory(tmp_path)
    labelled = session_directory(tmp_path, "failed grasp 3")

    assert plain.parent == tmp_path
    assert plain.name.endswith("Z")
    assert labelled.name.endswith("_failed-grasp-3")
