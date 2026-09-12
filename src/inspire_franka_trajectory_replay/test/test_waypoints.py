"""Waypoint extraction, checked without a bag: the marks, the motion, the artifact."""

import json

import numpy as np
import pytest

from inspire_franka_trajectory_replay.trajectory import (
    ARM_JOINTS,
    HAND_JOINTS,
    load_trajectory,
)
from inspire_franka_trajectory_replay.waypoints import (
    DEFAULT_PEAK_SPEED,
    HAND_SETTLE_FRACTION,
    MIN_MOVE_SECONDS,
    Waypoint,
    build_arrays,
    build_path,
    load_waypoints,
    mark_events,
    move_seconds,
    write_artifact,
)


RATE = 15.0


def _joint_state(joint_names, values, prefix=""):
    """A snapshot block in the shape capture.CaptureNode._as_dict writes.

    Parallel name/position lists beside the stamp, i.e. the JointState message's
    own shape -- NOT a flattened {joint: value} mapping. Pinned here because
    reading it as a mapping is a silent fall-through to the bag, not an error.
    """
    return {
        "stamp_ns": 1,
        "name": [prefix + joint for joint in joint_names],
        "position": [float(value) for value in values],
        "velocity": [0.0] * len(joint_names),
        "effort": [0.0] * len(joint_names),
    }


def _snapshot_event(name, stamp, index, arm, hand):
    return {
        "event": name,
        "t_ros_ns": stamp,
        "index": index,
        "arm_joint_states": _joint_state(ARM_JOINTS, arm),
        "hand_joint_states": _joint_state(HAND_JOINTS, hand),
    }


def _waypoint(index, arm_value, hand_value=0.2):
    return Waypoint(
        index=index,
        stamp_ns=index * 1_000_000_000,
        arm=np.full(len(ARM_JOINTS), float(arm_value)),
        hand=np.full(len(HAND_JOINTS), float(hand_value)),
        source="test",
    )


def _session(tmp_path, events, manifest=None):
    session = tmp_path / "session"
    session.mkdir()
    (session / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (session / "manifest.json").write_text(
        json.dumps(manifest or {"note": "test"}), encoding="utf-8"
    )
    return session


# --- which events count as a mark -------------------------------------------


def test_marks_are_returned_in_time_order_not_file_order():
    events = [
        {"event": "pose_capture", "t_ros_ns": 300, "index": 2},
        {"event": "pose_capture", "t_ros_ns": 100, "index": 1},
        {"event": "segment", "t_ros_ns": 200},
    ]
    assert [mark["t_ros_ns"] for mark in mark_events(events)] == [100, 300]


def test_a_legacy_session_marked_waypoint_is_still_read():
    """Sessions recorded before the event was renamed are still on disk."""
    events = [{"event": "waypoint", "t_ros_ns": 100, "index": 1}]
    assert len(mark_events(events)) == 1


def test_pose_capture_wins_when_a_session_somehow_holds_both():
    events = [
        {"event": "waypoint", "t_ros_ns": 100, "index": 1},
        {"event": "pose_capture", "t_ros_ns": 200, "index": 1},
    ]
    marks = mark_events(events)
    assert [mark["event"] for mark in marks] == ["pose_capture"]


def test_a_session_with_no_marks_says_so_rather_than_producing_nothing(tmp_path):
    session = _session(tmp_path, [{"event": "segment", "t_ros_ns": 1}])
    with pytest.raises(ValueError, match="no operator marks"):
        load_waypoints(session)


# --- reading the snapshot ----------------------------------------------------


def test_waypoints_come_from_the_events_alone_when_the_marks_carry_state(tmp_path):
    """The whole point: a current session needs no bag at all."""
    arm = np.linspace(0.1, 0.7, len(ARM_JOINTS))
    hand = np.full(len(HAND_JOINTS), 0.3)
    session = _session(tmp_path, [_snapshot_event("pose_capture", 100, 1, arm, hand)])
    # No bag directory exists under tmp_path, so this can only have worked from
    # events.jsonl.
    points = load_waypoints(session)
    assert len(points) == 1
    np.testing.assert_allclose(points[0].arm, arm)
    np.testing.assert_allclose(points[0].hand, hand)
    assert points[0].source == "events.jsonl snapshot"


def test_a_namespaced_hand_reads_the_same_as_a_bare_one(tmp_path):
    arm = np.zeros(len(ARM_JOINTS))
    event = _snapshot_event("pose_capture", 100, 1, arm, np.full(len(HAND_JOINTS), 0.4))
    event["hand_joint_states"] = _joint_state(
        HAND_JOINTS, [0.4] * len(HAND_JOINTS), prefix="/inspire_hand/"
    )
    points = load_waypoints(_session(tmp_path, [event]))
    np.testing.assert_allclose(points[0].hand, 0.4)


def test_a_mark_missing_an_arm_joint_is_not_silently_zero_filled(tmp_path):
    """A partial snapshot must fall through to the bag, not invent a pose."""
    arm = np.zeros(len(ARM_JOINTS))
    event = _snapshot_event("pose_capture", 100, 1, arm, np.zeros(len(HAND_JOINTS)))
    event["arm_joint_states"]["name"].pop(3)
    event["arm_joint_states"]["position"].pop(3)
    session = _session(tmp_path, [event])
    with pytest.raises(FileNotFoundError, match="no joint-state snapshot"):
        load_waypoints(session)


# --- the motion --------------------------------------------------------------


def test_a_single_waypoint_is_refused_because_it_is_not_a_motion():
    with pytest.raises(ValueError, match="at least two waypoints"):
        build_path([_waypoint(1, 0.0)], rate_hz=RATE)


def test_the_path_starts_at_the_first_waypoint(tmp_path):
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5)]
    path = build_path(points, rate_hz=RATE)
    np.testing.assert_allclose(path.arm[0], points[0].arm)


def test_the_path_ends_exactly_on_the_last_waypoint():
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5), _waypoint(3, -0.2)]
    path = build_path(points, rate_hz=RATE)
    np.testing.assert_allclose(path.arm[-1], points[-1].arm)


def test_every_waypoint_is_reached_and_marked():
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5), _waypoint(3, -0.2)]
    path = build_path(points, rate_hz=RATE)
    assert len(path.arrivals) == len(points)
    for arrival, point in zip(path.arrivals, points):
        np.testing.assert_allclose(path.arm[arrival], point.arm, atol=1e-9)


def test_the_arm_holds_still_for_the_whole_dwell():
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5)]
    dwell = 2.0
    path = build_path(points, rate_hz=RATE, dwell_s=dwell)
    arrival = int(path.arrivals[-1])
    held = path.arm[arrival:]
    assert len(held) == pytest.approx(dwell * RATE, abs=1)
    np.testing.assert_allclose(held, np.repeat(points[-1].arm[None, :], len(held), axis=0))


def test_the_hand_moves_only_after_the_arm_has_stopped():
    """A grasp is commanded at the waypoint, not during the transit to it."""
    start, end = _waypoint(1, 0.0, hand_value=0.0), _waypoint(2, 0.5, hand_value=1.0)
    path = build_path([start, end], rate_hz=RATE, dwell_s=2.0)
    arrival = int(path.arrivals[-1])
    # Unchanged for the whole move.
    np.testing.assert_allclose(path.hand[:arrival], 0.0)
    # Reached by the end of the settle fraction, then held.
    np.testing.assert_allclose(path.hand[-1], 1.0)


def test_the_hand_reaches_its_posture_within_the_settle_fraction():
    start, end = _waypoint(1, 0.0, hand_value=0.0), _waypoint(2, 0.5, hand_value=1.0)
    dwell = 2.0
    path = build_path([start, end], rate_hz=RATE, dwell_s=dwell)
    arrival = int(path.arrivals[-1])
    settle = int(round((len(path.arm) - arrival) * HAND_SETTLE_FRACTION))
    np.testing.assert_allclose(path.hand[arrival + settle - 1], 1.0, atol=1e-9)


def test_move_duration_respects_the_peak_speed_not_the_average():
    """The quintic peaks at 1.875x its mean; the solve has to account for it."""
    start = np.zeros(len(ARM_JOINTS))
    end = np.full(len(ARM_JOINTS), 1.0)
    duration = move_seconds(start, end, DEFAULT_PEAK_SPEED)
    assert duration == pytest.approx(1.875 / DEFAULT_PEAK_SPEED)


def test_two_nearly_identical_waypoints_do_not_become_a_step_input():
    start = np.zeros(len(ARM_JOINTS))
    assert move_seconds(start, start + 1e-9, DEFAULT_PEAK_SPEED) == MIN_MOVE_SECONDS


def test_the_generated_path_stays_under_the_commanded_peak_speed():
    points = [_waypoint(1, 0.0), _waypoint(2, 1.2), _waypoint(3, -0.6)]
    peak = DEFAULT_PEAK_SPEED
    path = build_path(points, rate_hz=RATE, peak_speed=peak)
    speed = np.abs(np.diff(path.arm, axis=0)) * RATE
    # A little headroom for the discrete grid: the profile is sampled, not solved.
    assert speed.max() <= peak * 1.1


def test_a_slower_speed_makes_a_longer_path():
    points = [_waypoint(1, 0.0), _waypoint(2, 1.0)]
    fast = build_path(points, rate_hz=RATE, peak_speed=DEFAULT_PEAK_SPEED)
    slow = build_path(points, rate_hz=RATE, peak_speed=DEFAULT_PEAK_SPEED / 2)
    assert len(slow.time) > len(fast.time)


# --- the artifact ------------------------------------------------------------


def test_the_artifact_carries_every_commandable_joint():
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5)]
    arrays = build_arrays(build_path(points, rate_hz=RATE), RATE)
    count = len(arrays["sample_time_s"])
    assert arrays["joint_pos"].shape == (count, 1, 19)
    assert arrays["tcp_pos"].shape == (count, 1, 3)
    assert arrays["tcp_quat"].shape == (count, 1, 4)


def test_the_artifact_replays_through_the_ordinary_trajectory_loader(tmp_path):
    """The contract that keeps the replay stack unmodified."""
    points = [_waypoint(1, 0.0), _waypoint(2, 0.4), _waypoint(3, 0.1)]
    path = build_path(points, rate_hz=RATE)
    session = _session(tmp_path, [{"event": "session_start", "t_ros_ns": 1}])
    out = tmp_path / "artifact"
    write_artifact(out, points, path, session, RATE, 1.5, DEFAULT_PEAK_SPEED)

    trajectory = load_trajectory(str(out / "replay_data.npz"))
    assert trajectory.arm.shape == (len(path.time), 7)
    assert trajectory.hand is not None
    np.testing.assert_allclose(trajectory.arm[0], points[0].arm, atol=1e-9)
    np.testing.assert_allclose(trajectory.arm[-1], points[-1].arm, atol=1e-9)


def test_the_home_pose_is_the_trajectorys_own_first_sample(tmp_path):
    """A home that disagreed with sample 0 would jerk the arm on the first step."""
    yaml = pytest.importorskip("yaml")
    points = [_waypoint(1, 0.25), _waypoint(2, 0.5)]
    path = build_path(points, rate_hz=RATE)
    session = _session(tmp_path, [{"event": "session_start", "t_ros_ns": 1}])
    out = tmp_path / "artifact"
    write_artifact(out, points, path, session, RATE, 1.5, DEFAULT_PEAK_SPEED)

    home = yaml.safe_load((out / "homing.yaml").read_text(encoding="utf-8"))
    assert home["joint_names"] == list(ARM_JOINTS) + list(HAND_JOINTS)
    np.testing.assert_allclose(home["positions"][: len(ARM_JOINTS)], path.arm[0])


def test_the_metadata_never_calls_a_waypoint_replay_a_demonstration(tmp_path):
    """Nothing downstream may mistake this for the motion that was performed."""
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5)]
    path = build_path(points, rate_hz=RATE)
    session = _session(tmp_path, [{"event": "session_start", "t_ros_ns": 1}])
    out = tmp_path / "artifact"
    write_artifact(out, points, path, session, RATE, 1.5, DEFAULT_PEAK_SPEED)

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["replay"] == "hand_guided_waypoints"
    assert metadata["motion"]["kind"] == "joint_space_point_to_point"
    assert metadata["source"]["waypoint_count"] == 2


def test_a_simulated_session_keeps_its_own_word_in_the_artifact(tmp_path):
    points = [_waypoint(1, 0.0), _waypoint(2, 0.5)]
    path = build_path(points, rate_hz=RATE)
    session = _session(tmp_path, [{"event": "session_start", "t_ros_ns": 1}],
                       manifest={"simulated": True})
    out = tmp_path / "artifact"
    write_artifact(out, points, path, session, RATE, 1.5, DEFAULT_PEAK_SPEED)

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["replay"] == "hand_guided_waypoints_sim"
    assert metadata["simulated"] is True


# -- the bag fallback, for sessions whose marks carry no snapshot -------------


class _FakeReader:
    """Enough of rosbag2_py.SequentialReader to drive _sample_topic_at_marks."""

    def __init__(self, rows):
        self._rows = rows  # (stamp_ns, positions)
        self._at = 0

    def seek(self, stamp_ns):
        self._at = next(
            (i for i, (s, _) in enumerate(self._rows) if s >= stamp_ns), len(self._rows)
        )

    def has_next(self):
        return self._at < len(self._rows)

    def read_next(self):
        stamp, positions = self._rows[self._at]
        self._at += 1
        return "/topic", (stamp, positions), stamp


def _patch_reader(monkeypatch, rows):
    from inspire_franka_trajectory_replay import waypoints as module

    class _Message:
        def __init__(self, stamp, positions):
            self.name = list(ARM_JOINTS)
            self.position = positions
            self.header = type("H", (), {"stamp": stamp})()

    monkeypatch.setattr(module, "open_reader", lambda *a, **k: (_FakeReader(rows), {"/topic": object}))
    monkeypatch.setattr(module, "deserialize_message", lambda data, _: _Message(*data))
    monkeypatch.setattr(module, "stamp_to_ns", lambda stamp: stamp)
    return module


def test_the_nearest_sample_to_a_mark_is_the_one_taken(monkeypatch):
    rows = [(t, [float(t)] * len(ARM_JOINTS)) for t in (1_000, 2_000, 3_000)]
    module = _patch_reader(monkeypatch, rows)
    got = module._sample_topic_at_marks("bag", "/topic", ARM_JOINTS, [2_100])
    np.testing.assert_allclose(got[0], 2_000.0)


def test_a_mark_outside_what_the_topic_recorded_is_an_error_not_a_far_pose(monkeypatch):
    """A silently wrong waypoint is a pose the arm would actually be driven to."""
    from inspire_franka_trajectory_replay.waypoints import SEEK_WINDOW_NS

    rows = [(t, [float(t)] * len(ARM_JOINTS)) for t in (1_000, 2_000)]
    module = _patch_reader(monkeypatch, rows)
    far = 2_000 + SEEK_WINDOW_NS * 10
    with pytest.raises(ValueError, match="outside what this topic recorded"):
        module._sample_topic_at_marks("bag", "/topic", ARM_JOINTS, [far])


def test_a_snapshot_is_read_in_the_shape_capture_actually_writes(tmp_path):
    """Regression: capture writes parallel name/position lists, not a mapping.

    Reading it as {joint: value} does not raise -- it returns None and falls
    silently through to the bag, so a current session pays a bag read it should
    never have needed and the artifact says it came from the wrong place.
    """
    arm = np.linspace(0.1, 0.7, len(ARM_JOINTS))
    hand = np.full(len(HAND_JOINTS), 0.3)
    session = _session(tmp_path, [_snapshot_event("pose_capture", 100, 1, arm, hand)])
    points = load_waypoints(session)
    assert points[0].source == "events.jsonl snapshot"
    np.testing.assert_allclose(points[0].arm, arm)
    np.testing.assert_allclose(points[0].hand, hand)


def test_a_snapshot_with_mismatched_name_and_position_lengths_is_refused(tmp_path):
    arm = np.zeros(len(ARM_JOINTS))
    event = _snapshot_event("pose_capture", 100, 1, arm, np.zeros(len(HAND_JOINTS)))
    event["arm_joint_states"]["position"].pop()
    with pytest.raises(FileNotFoundError, match="no joint-state snapshot"):
        load_waypoints(_session(tmp_path, [event]))
