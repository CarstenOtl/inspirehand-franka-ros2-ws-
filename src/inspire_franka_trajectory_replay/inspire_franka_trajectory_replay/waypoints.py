"""Turn a capture session's operator marks into a replayable waypoint artifact.

    ros2 run inspire_franka_trajectory_replay extract_waypoints \
        logs/demo_capture/<stamp>_<name>

This is the *first* half of the capture loop, and it deliberately does much
less than ``extract_demo``. A hand-guided session holds two different things:
the continuous path the operator's hand happened to take, and the small set of
poses the operator actually meant -- the ones they stopped at and pressed ``c``
on. Confirming a demonstration is worth keeping is a question about the second
set, not the first, and answering it should not cost a bag read.

So this tool reads ``events.jsonl`` and nothing else, whenever it can. A
``pose_capture`` event written by a current ``capture_demo`` already carries the
arm and hand joint state at the instant the key went down, which is the whole
input. Extraction is milliseconds, and a lean capture never has to be
processed at all to be tried out.

Sessions recorded before ``capture_demo`` embedded that snapshot have bare
marks -- a timestamp and an index. Those are handled by reading the *arm and
hand joint-state topics only* out of the bag and sampling them at the marked
instants: a few seconds even on a large session, because it never touches
``FrankaRobotState``, which is where essentially all of a bag's decode cost
lives.

What comes out
--------------
``replay_data.npz``, ``metadata.json`` and ``homing.yaml``, in the same schema
``extract_demo`` writes and ``replay_trajectory`` already consumes. There is no
new replay path and no new motion code: the artifact is a dense 15 Hz
trajectory like any other, so it goes through ``prepare()``, the limit guards
and the same client as a full demonstration.

The motion it describes is joint-space point to point with a dwell at each
mark: ease from waypoint to waypoint on a quintic profile with zero velocity
and acceleration at both ends, hold still for ``--dwell`` seconds on arrival,
then move on. The hand is commanded to the posture recorded at a waypoint over
the first part of that dwell, so the fingers move once the arm has stopped
rather than during the transit.

This is not the demonstrated motion and does not pretend to be -- the metadata
says ``hand_guided_waypoints`` -- it is the reachability and grasp check you
run before spending a full-state capture on the take.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml
from rclpy.serialization import deserialize_message

from franka_trajectory_replay.kinematics import tool_transform
from franka_trajectory_replay.recording import open_reader, stamp_to_ns
from franka_trajectory_replay.runconfig import load_config
from inspire_hand_driver import kinematics as kin

from .extract import (
    ARTIFACT_JOINT_NAMES,
    DEFAULT_RATE_HZ,
    DRIVER_TO_ARTIFACT_JOINT,
    HAND_JOINT_STATES_TOPIC,
    SCHEMA_VERSION,
    _atomic_json,
    _atomic_npz,
    _atomic_text,
    arm_joint_states_topic,
    central_difference,
    expand_followers,
    load_events,
    pose16_from_joint_angles,
    tcp_from_pose16,
)
from .trajectory import ARM_JOINTS, HAND_JOINTS


#: Event names that mean "the operator marked this instant".
#:
#: ``pose_capture`` is what ``capture_demo`` writes now. ``waypoint`` is what it
#: wrote before, and sessions in that format are still on disk and still worth
#: replaying, so both are read. They are never mixed within one session: the
#: first name that appears at all is the one used, because a session that
#: somehow held both would have two different meanings of "index" in it.
MARK_EVENTS = ("pose_capture", "waypoint")

#: Peak joint speed of a waypoint-to-waypoint move, rad/s.
#:
#: Well under the FR3's limits on every joint -- this is a check run for the
#: first time on poses nobody has commanded before, and the point is to arrive,
#: not to arrive quickly. ``--speed`` scales it. The quintic profile below
#: peaks at 1.875x its average, which is accounted for when solving durations,
#: so this really is the peak and not the mean.
DEFAULT_PEAK_SPEED = 0.35

#: Seconds held still at each waypoint.
DEFAULT_DWELL = 1.5

#: Shortest a move may take however small it is, seconds. Keeps a pair of
#: nearly-identical waypoints from becoming a step input to the controller.
MIN_MOVE_SECONDS = 0.5

#: Fraction of the dwell the hand is given to reach its commanded posture. The
#: rest of the dwell is the hand holding it, which is what makes a grasp at a
#: waypoint observable rather than instantaneous.
HAND_SETTLE_FRACTION = 0.5


@dataclass(frozen=True)
class Waypoint:
    """One operator mark, with everything needed to command it again."""

    index: int
    stamp_ns: int
    #: Arm joint angles in ``ARM_JOINTS`` order, radians.
    arm: np.ndarray
    #: The six driven hand joints in ``kin.DRIVEN_JOINTS`` order, radians.
    hand: np.ndarray
    #: Where the numbers came from, recorded in the artifact's metadata.
    source: str


def _snapshot_columns(
    snapshot: Optional[dict], joint_names: Sequence[str]
) -> Optional[np.ndarray]:
    """Named joints out of one snapshot block, or None if it cannot supply them.

    ``capture.CaptureNode._as_dict`` writes a JointState as it stands -- parallel
    ``name`` and ``position`` lists beside the stamp -- rather than flattening it
    to a mapping, so that the event log keeps the message's own shape. Zipping
    them back is this function's whole job.

    Joints are matched on the last path segment, so a namespaced hand reads the
    same as a bare one; that is the rule ``extract.read_hand`` applies to the bag
    and the two have to agree or a waypoint would not match its own recording.
    """
    if not isinstance(snapshot, dict):
        return None
    names = snapshot.get("name")
    positions = snapshot.get("position")
    if not names or not positions or len(names) != len(positions):
        return None
    values = {str(name).split("/")[-1]: value for name, value in zip(names, positions)}
    if any(joint not in values for joint in joint_names):
        return None
    return np.array([float(values[joint]) for joint in joint_names])


def mark_events(events: Sequence[dict]) -> List[dict]:
    """The operator's marks, in time order, under whichever name they carry."""
    for name in MARK_EVENTS:
        marks = [event for event in events if event.get("event") == name]
        if marks:
            return sorted(marks, key=lambda event: int(event["t_ros_ns"]))
    return []


def _from_snapshot(mark: dict) -> Optional[Waypoint]:
    """A waypoint built from the event alone, or None if it has no snapshot."""
    arm = _snapshot_columns(mark.get("arm_joint_states"), ARM_JOINTS)
    hand = _snapshot_columns(mark.get("hand_joint_states"), kin.DRIVEN_JOINTS)
    if arm is None or hand is None:
        return None
    return Waypoint(
        index=int(mark.get("index", 0)),
        stamp_ns=int(mark["t_ros_ns"]),
        arm=arm,
        hand=hand,
        source="events.jsonl snapshot",
    )


#: How far either side of a mark to look for a joint-state sample, nanoseconds.
#:
#: Generous, because ``seek`` works on the bag's receive timestamps while a mark
#: and a message both carry their own header stamp, and the two differ by the
#: transport delay. Half a second at 1 kHz is a thousand candidate samples and
#: still a rounding error against a whole bag.
SEEK_WINDOW_NS = 500_000_000


def _sample_topic_at_marks(
    bag_dir, topic: str, joint_names: Sequence[str], instants_ns: Sequence[int]
) -> List[np.ndarray]:
    """The joint row nearest each instant, reading only a window around each.

    Nearest rather than interpolated on purpose: a mark is a moment the operator
    chose while the arm was stationary, so the honest answer is the sample they
    were looking at, not a blend of two.

    Seeks to just before every mark rather than streaming the topic: a session
    holds ~175k arm samples and a handful of marks, so reading all of them to
    keep three is the difference between twenty seconds and none. Falls back to
    a full scan if the bag cannot seek.
    """
    reader, classes = open_reader(bag_dir, [topic])
    if topic not in classes:
        raise ValueError(f"the bag holds no {topic} messages")
    message_class = classes[topic]
    columns: Optional[List[int]] = None

    def decode(data):
        nonlocal columns
        message = deserialize_message(data, message_class)
        if columns is None:
            suffixes = [str(name).split("/")[-1] for name in message.name]
            missing = [joint for joint in joint_names if joint not in suffixes]
            if missing:
                raise ValueError(
                    f"{topic} does not carry {missing}; it published {list(message.name)}"
                )
            columns = [suffixes.index(joint) for joint in joint_names]
        return message

    results: List[np.ndarray] = []
    for instant in instants_ns:
        try:
            reader.seek(int(instant) - SEEK_WINDOW_NS)
        except (AttributeError, RuntimeError):
            reader, classes = open_reader(bag_dir, [topic])
            message_class = classes[topic]
        best: Optional[np.ndarray] = None
        best_gap: Optional[int] = None
        while reader.has_next():
            read_topic, data, receive_ns = reader.read_next()
            if read_topic != topic:
                continue
            message = decode(data)
            stamp = stamp_to_ns(message.header.stamp) or receive_ns
            gap = abs(stamp - int(instant))
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best = np.array([float(message.position[i]) for i in columns])
            if stamp > int(instant) + SEEK_WINDOW_NS:
                break
        # A mark outside what this topic recorded has to be an error. Taking the
        # nearest sample regardless would hand back a pose from a completely
        # different moment, and a wrong waypoint is worse than no waypoint:
        # nothing downstream can tell it was wrong, and it is a pose the arm
        # will be commanded to.
        if best is None or best_gap > SEEK_WINDOW_NS:
            raise ValueError(
                f"{topic} has no sample within {SEEK_WINDOW_NS * 1e-9:.1f} s of the mark at "
                f"{instant} ns; the mark lies outside what this topic recorded"
            )
        results.append(best)
    return results


def _from_bag(marks: Sequence[dict], bag_dir) -> List[Waypoint]:
    """Waypoints for a session whose marks carry no snapshot.

    Reads the two joint-state topics and nothing else, in a window around each
    mark. That restriction is the entire reason this is fast: ``FrankaRobotState``
    is ~90% of a session bag's decode cost and holds no joint angle the arm
    topic does not.
    """
    arm_topic = arm_joint_states_topic(bag_dir)
    instants = [int(mark["t_ros_ns"]) for mark in marks]
    arm_rows = _sample_topic_at_marks(bag_dir, arm_topic, ARM_JOINTS, instants)
    hand_rows = _sample_topic_at_marks(
        bag_dir, HAND_JOINT_STATES_TOPIC, kin.DRIVEN_JOINTS, instants
    )
    return [
        Waypoint(
            index=int(mark.get("index", position + 1)),
            stamp_ns=instants[position],
            arm=arm_rows[position],
            hand=hand_rows[position],
            source=f"{arm_topic} + {HAND_JOINT_STATES_TOPIC} sampled at the mark",
        )
        for position, mark in enumerate(marks)
    ]


def load_waypoints(session_dir) -> List[Waypoint]:
    """Every operator mark in a session, from the events if possible.

    Falls back to the bag for a session recorded before the snapshot existed,
    and says which route it took in each waypoint's ``source``.
    """
    session_dir = Path(session_dir)
    events = load_events(session_dir / "events.jsonl")
    marks = mark_events(events)
    if not marks:
        raise ValueError(
            f"{session_dir} holds no operator marks; nothing was captured with the "
            "'c' key during this session, so there are no waypoints to replay"
        )

    snapshots = [_from_snapshot(mark) for mark in marks]
    if all(waypoint is not None for waypoint in snapshots):
        return list(snapshots)

    manifest_path = session_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    bag_dir = Path(manifest.get("bag_dir") or (session_dir / "bag"))
    if not bag_dir.is_dir():
        raise FileNotFoundError(
            f"{len(marks)} mark(s) in {session_dir} carry no joint-state snapshot, and "
            f"there is no bag at {bag_dir} to read them from. This session was recorded "
            "by a capture_demo too old to embed the snapshot and too lean to recover it."
        )
    print(
        f"note: {session_dir.name} predates snapshotted marks; reading the arm and hand "
        "joint-state topics out of the bag instead (seconds, not minutes -- "
        "FrankaRobotState is not touched).",
        flush=True,
    )
    return _from_bag(marks, bag_dir)


def _quintic(fraction: np.ndarray) -> np.ndarray:
    """Minimum-jerk ease, zero velocity and acceleration at both ends."""
    return 10.0 * fraction**3 - 15.0 * fraction**4 + 6.0 * fraction**5


def move_seconds(start: np.ndarray, end: np.ndarray, peak_speed: float) -> float:
    """How long a waypoint-to-waypoint move takes at a peak joint speed.

    The quintic profile's peak velocity is 1.875x its mean, so a move of
    distance ``d`` in time ``T`` peaks at ``1.875 d / T``; solving that for the
    allowed peak is what sets ``T``. The slowest joint decides, because they
    all run on one clock.
    """
    distance = float(np.max(np.abs(np.asarray(end) - np.asarray(start))))
    return max(MIN_MOVE_SECONDS, 1.875 * distance / float(peak_speed))


@dataclass(frozen=True)
class WaypointPath:
    """The dense trajectory a waypoint artifact is written from."""

    time: np.ndarray
    arm: np.ndarray
    hand: np.ndarray
    #: Sample index each waypoint lands on, for the artifact's capture markers.
    arrivals: np.ndarray


def build_path(
    waypoints: Sequence[Waypoint],
    rate_hz: float = DEFAULT_RATE_HZ,
    dwell_s: float = DEFAULT_DWELL,
    peak_speed: float = DEFAULT_PEAK_SPEED,
    lead_in_s: Optional[float] = None,
) -> WaypointPath:
    """Dense point-to-point path through the waypoints, dwelling at each.

    Starts *at* the first waypoint. Getting the arm there from wherever it
    happens to be is homing's job, and ``homing.yaml`` is written from this
    same first sample, exactly as ``extract_demo`` does it -- so the trajectory
    never contains a move the operator did not demonstrate the endpoint of.
    """
    if len(waypoints) < 2:
        raise ValueError(
            f"need at least two waypoints to build a motion; this session has "
            f"{len(waypoints)}. Press 'c' at each pose you want the replay to visit."
        )
    dt = 1.0 / float(rate_hz)
    dwell_samples = max(1, int(round(float(dwell_s) / dt)))
    settle_samples = max(1, int(round(dwell_samples * HAND_SETTLE_FRACTION)))

    arm_rows: List[np.ndarray] = []
    hand_rows: List[np.ndarray] = []
    arrivals: List[int] = []

    def dwell(at: Waypoint, previous_hand: np.ndarray) -> None:
        """Hold the arm still; ease the hand to this waypoint's posture."""
        arrivals.append(len(arm_rows))
        for step in range(dwell_samples):
            arm_rows.append(np.array(at.arm))
            if step < settle_samples:
                blend = _quintic(np.array((step + 1) / settle_samples))
                hand_rows.append(previous_hand + (at.hand - previous_hand) * float(blend))
            else:
                hand_rows.append(np.array(at.hand))

    # A lead-in dwell on the first waypoint gives the controller a stationary
    # start and gives the hand somewhere to reach its first posture from.
    first = waypoints[0]
    lead_in = dwell_s if lead_in_s is None else lead_in_s
    for _ in range(max(1, int(round(float(lead_in) / dt)))):
        arm_rows.append(np.array(first.arm))
        hand_rows.append(np.array(first.hand))
    arrivals.append(0)

    for start, end in zip(waypoints, waypoints[1:]):
        duration = move_seconds(start.arm, end.arm, peak_speed)
        steps = max(2, int(round(duration / dt)))
        # 1..steps, so the move's first emitted sample has already left the
        # waypoint and its last lands exactly on the next one.
        blend = _quintic(np.arange(1, steps + 1) / steps)[:, None]
        arm_rows.extend(start.arm + (end.arm - start.arm) * blend)
        hand_rows.extend(np.repeat(start.hand[None, :], steps, axis=0))
        dwell(end, np.array(start.hand))

    arm = np.asarray(arm_rows, dtype=float)
    hand = np.asarray(hand_rows, dtype=float)
    return WaypointPath(
        time=np.arange(len(arm)) * dt,
        arm=arm,
        hand=hand,
        arrivals=np.asarray(arrivals, dtype=np.int64),
    )


def build_arrays(path: WaypointPath, rate_hz: float) -> dict:
    """The artifact's NPZ arrays, in the layout ``replay_trajectory`` reads."""
    dt = 1.0 / float(rate_hz)
    count = len(path.time)
    columns = expand_followers(path.hand)

    block = np.empty((count, len(ARTIFACT_JOINT_NAMES)))
    block[:, : len(ARM_JOINTS)] = path.arm
    for index, name in enumerate(ARTIFACT_JOINT_NAMES[len(ARM_JOINTS):], start=len(ARM_JOINTS)):
        driver_name = next(k for k, v in DRIVER_TO_ARTIFACT_JOINT.items() if v == name)
        block[:, index] = columns[driver_name]
    joint_pos = block[:, None, :]

    # Forward kinematics: a waypoint path is a commanded trajectory, so there
    # is no recorded O_T_EE for it even when the session that produced the
    # waypoints had one.
    config = load_config(None)
    tool = tool_transform(config["tcp"]["offset_xyz"], config["tcp"]["offset_rpy"])
    tcp_pos, tcp_quat = tcp_from_pose16(pose16_from_joint_angles(path.arm), tool)
    return {
        "sample_time_s": path.time,
        "step": np.arange(count, dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": central_difference(joint_pos[:, 0, :], dt)[:, None, :],
        # A waypoint replay commands exactly what it holds -- there is no
        # measured-versus-commanded distinction to record, because none of this
        # was measured. Writing the same block twice keeps the artifact's shape
        # honest rather than inventing a second number.
        "joint_pos_target": joint_pos,
        "tcp_pos": tcp_pos[:, None, :],
        "tcp_quat": tcp_quat[:, None, :],
        "pose_capture_index": path.arrivals,
    }


def write_artifact(
    output_dir: Path,
    waypoints: Sequence[Waypoint],
    path: WaypointPath,
    session_dir: Path,
    rate_hz: float,
    dwell_s: float,
    peak_speed: float,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_dir / "replay_data.npz", build_arrays(path, rate_hz))

    manifest_path = Path(session_dir) / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    simulated = bool(manifest.get("simulated"))
    metadata = {
        # Its own word, for the same reason extract_demo has one: this is not a
        # demonstration and nothing downstream should be able to read it as one.
        "replay": "hand_guided_waypoints_sim" if simulated else "hand_guided_waypoints",
        "simulated": simulated,
        "schema_version": SCHEMA_VERSION,
        "task": manifest.get("note", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_file": "replay_data.npz",
        "rate_hz": float(rate_hz),
        "joint_names": list(ARTIFACT_JOINT_NAMES),
        "source": {
            "session_dir": str(session_dir),
            "capture_tool": "inspire_franka_trajectory_replay.capture",
            "extract_tool": "inspire_franka_trajectory_replay.waypoints",
            "full_state_bag": manifest.get("full_state"),
            "waypoint_count": len(waypoints),
            "waypoint_source": sorted({point.source for point in waypoints}),
            "waypoint_stamp_ns": [int(point.stamp_ns) for point in waypoints],
        },
        "motion": {
            "kind": "joint_space_point_to_point",
            "dwell_s": float(dwell_s),
            "peak_joint_speed_rad_s": float(peak_speed),
            "profile": "quintic, zero velocity and acceleration at each waypoint",
            "hand": "eased to the waypoint's recorded posture over the first "
                    f"{HAND_SETTLE_FRACTION:.0%} of the dwell, then held",
            "note": "This is NOT the demonstrated path. It visits the poses the operator "
                    "marked, in order, by the shortest joint-space route between them. "
                    "Use extract_demo for the motion that was actually performed.",
        },
        "hardware_orientation": {
            "joint": "fr3_joint7",
            "offset_rad": 0.0,
            "note": "Recorded joint 7 is written through unchanged and homing.yaml is read "
                    "off the same first sample. Do not apply the legacy +90-degree "
                    "compensation, retarget_flange_mount.py, or a *_flange180 derivative.",
        },
    }
    _atomic_json(output_dir / "metadata.json", metadata)

    # Read off sample 0 exactly as extract_demo does, so the home pose and the
    # trajectory's first sample cannot disagree. Sample 0 is the first waypoint.
    home = {
        "schema_version": 1,
        "name": f"{output_dir.name}_homing",
        "description": "First waypoint of a hand-guided capture.",
        "units": "radians",
        "joint_names": list(ARM_JOINTS) + list(HAND_JOINTS),
        "positions": [float(value) for value in path.arm[0]]
        + [float(value) for value in path.hand[0]],
        "source": {
            "session_dir": str(session_dir),
            "derived_from": "replay_data.npz sample 0",
            "method": "Read off the first marked waypoint, so the home and the "
                      "trajectory start cannot disagree.",
            "joint7_offset_rad": 0.0,
        },
    }
    _atomic_text(
        output_dir / "homing.yaml",
        "# Commandable joints only. The six passive Inspire joints follow\n"
        "# mechanically and must not be commanded independently.\n"
        + yaml.safe_dump(home, sort_keys=False, default_flow_style=False),
    )
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a replayable point-to-point artifact from a capture "
                    "session's operator marks, without reading the bag.",
    )
    parser.add_argument("session", help="a logs/demo_capture/<stamp>_<name> directory")
    parser.add_argument(
        "-o", "--output", default=None,
        help="artifact directory (default: <session>/waypoints)",
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE_HZ,
        help=f"output sample rate in hertz (default: {DEFAULT_RATE_HZ:g})",
    )
    parser.add_argument(
        "--dwell", type=float, default=DEFAULT_DWELL,
        help=f"seconds held still at each waypoint (default: {DEFAULT_DWELL:g})",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="scale the peak joint speed of each move "
             f"(1.0 = {DEFAULT_PEAK_SPEED:g} rad/s, deliberately slow)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.rate <= 0:
        _parser().error("--rate must be positive")
    if args.dwell < 0:
        _parser().error("--dwell may not be negative")
    if args.speed <= 0:
        _parser().error("--speed must be positive")

    session_dir = Path(args.session)
    try:
        waypoints = load_waypoints(session_dir)
        path = build_path(
            waypoints,
            rate_hz=args.rate,
            dwell_s=args.dwell,
            peak_speed=DEFAULT_PEAK_SPEED * args.speed,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output) if args.output else session_dir / "waypoints"
    write_artifact(
        output_dir, waypoints, path, session_dir,
        args.rate, args.dwell, DEFAULT_PEAK_SPEED * args.speed,
    )

    print(f"{len(waypoints)} waypoints -> {output_dir}")
    for point in waypoints:
        print(f"  {point.index:3d}  arm " + " ".join(f"{value:+.3f}" for value in point.arm))
    print(f"  {path.time[-1]:.1f} s, {len(path.time)} samples at {args.rate:g} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
