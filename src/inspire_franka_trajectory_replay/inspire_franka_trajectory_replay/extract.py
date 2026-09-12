"""Turn a hand-guided capture session into a replayable trajectory artifact.

    ros2 run inspire_franka_trajectory_replay extract_demo logs/demo_capture/<stamp>_<name>

Input is a session directory written by ``capture_demo``: the raw bag, the
``events.jsonl`` command/marker log, and ``manifest.json``. Output is a
directory holding ``replay_data.npz``, ``metadata.json`` and ``homing.yaml`` in
the schema ``replay_trajectory`` already consumes, so that

    ros2 run inspire_franka_trajectory_replay replay_trajectory <artifact> \
        --home <artifact>/homing.yaml --dry-run

works against it with no conversion step and no modification to the replay
stack.

Lean and full-state sessions
----------------------------
Both are read, without being told which is which. A ``--full-state`` capture has
``FrankaRobotState``, and the arm state, the TCP pose and the whole FCI field
set come from it. A lean one does not, and the arm comes from
``/franka/joint_states`` at the same 1 kHz with the TCP derived by forward
kinematics -- a different measurement with different error, recorded as such in
``metadata.source.tcp_source`` rather than passed off as the same number.

The artifact is otherwise identical, because it never contained the rest of the
field set anyway. What a lean session cannot do is be augmented into training
data later; see ``capture.FULL_STATE_ONLY_TOPICS``.

If you only want the poses the operator marked, use ``extract_waypoints``
instead -- it reads ``events.jsonl`` and skips the bag entirely.

What the extraction does, and why
---------------------------------
*Resampling.* The arm is recorded at the controller's 1 kHz; the artifact
convention in this workspace is 15 Hz (``dt = 1/15``), which is what every
existing capture under ``apps/traj_replay/demo_trajs`` uses and what the replay
preparation expects to spline through.

*Filtering.* Hand-guided motion is jerky in a way a policy rollout is not: the
operator's tremor and the arm's own structural ring both sit in the recording,
and both land inside the band a 15 Hz sampler can represent. Arm joint positions
are therefore low-passed with a zero-phase 4th-order Butterworth (``filtfilt``,
so there is no lag and no group delay to correct for) at
:data:`DEFAULT_CUTOFF_HZ` before being sampled onto the 15 Hz grid. That filter
does two jobs at once: it is the anti-alias filter the decimation needs (the
15 Hz grid's Nyquist is 7.5 Hz), and it is what brings the *derivatives* of a
hand-guided trajectory inside the FR3's velocity, acceleration and jerk limits.

This is the only lever this tool pulls to pass the limit guards. It changes the
trajectory -- and says by how much, in ``extraction_report.json`` and in the
printed table -- rather than relaxing a margin. If a session still fails
preparation after filtering, the answer is ``--time-scale`` on replay, or a
calmer demonstration; it is never a raised margin, and it is never dropping
``--dry-run`` to find out.

*Velocities.* ``joint_vel`` is differentiated from the filtered, resampled
positions rather than taken from the recorded ``dq``, so that the artifact's
velocity field describes the trajectory the artifact actually holds.

*The hand.* The six driven joints come from the measured
``/inspire_hand/joint_states``; the six mechanical followers are recomputed from
them through :mod:`inspire_hand_driver.kinematics` rather than being resampled
independently, so the coupling still holds exactly at every emitted sample.

*The joint-7 contract.* Recorded ``fr3_joint7`` is written through unchanged and
the homing pose is read off the same first sample, so the two cannot disagree.
``metadata.json`` records ``offset_rad: 0.0``. See
``apps/traj_replay/demo_trajs/README.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from franka_trajectory_replay.dataset import extract_bag
from franka_trajectory_replay.kinematics import (
    column_major_to_transform,
    flange_transform,
    matrix_to_quaternion,
    tool_transform,
)
from franka_trajectory_replay.recording import read_messages, stamp_to_ns
from franka_trajectory_replay.runconfig import load_config
from inspire_hand_driver import kinematics as kin

from .trajectory import ARM_JOINTS, FORGE_HAND_JOINTS, HAND_JOINTS


SCHEMA_VERSION = 1

#: The artifact convention every capture in this workspace follows.
DEFAULT_RATE_HZ = 15.0

#: Zero-phase low-pass corner for the arm joint positions, in hertz. Below the
#: 7.5 Hz Nyquist of the 15 Hz grid, and above the band hand guiding actually
#: carries: a person moving a 7 kg arm deliberately puts almost nothing above
#: 2 Hz into it, while tremor and structural ring sit well above that.
DEFAULT_CUTOFF_HZ = 2.0

ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"

#: Arm joint angles, in preference order, for a session with no FrankaRobotState.
#:
#: ``/franka/joint_states`` is joint_state_broadcaster's own 1 kHz arm-only
#: publication -- the same rate as the FCI state and a twentieth of the bytes,
#: which is what makes a lean capture worth having. ``/joint_states`` is the
#: 30 Hz merged view, and is the fallback because a simulated session has only
#: that one. Order matters: 30 Hz leaves a 15 Hz grid just two samples per
#: output, so it is used when it is all there is, not by choice.
ARM_JOINT_STATES_TOPICS = ("/franka/joint_states", "/joint_states")

#: Kept as the name the rest of this module and its tests refer to.
JOINT_STATES_TOPIC = ARM_JOINT_STATES_TOPICS[-1]
HAND_JOINT_STATES_TOPIC = "/inspire_hand/joint_states"

#: The driver's twelve joint names in the naming the replay artifacts use.
#:
#: The driven six are fixed by ``trajectory.FORGE_HAND_JOINTS``, which the
#: replay loader itself reads, and ``test_extract.py`` asserts that this table
#: agrees with it. The six followers are matched by their coupling structure,
#: which is the same in both worlds: each finger's ``_joint_1`` follows its
#: ``_joint_0``, and the thumb's ``_joint_2`` and ``_joint_3`` follow the thumb
#: bend (see MIMIC_JOINT_MAP in apps/policy_rollout/policy_rollout/forge_osc.py).
#: The coupling *ratios* differ between the two models -- ours are the vendored
#: URDF's, and this tool uses ours, because what it is describing is this
#: hardware.
DRIVER_TO_ARTIFACT_JOINT = {
    "index_proximal_joint": "index_joint_0",
    "pinky_proximal_joint": "little_joint_0",
    "middle_proximal_joint": "middle_joint_0",
    "ring_proximal_joint": "ring_joint_0",
    "thumb_proximal_yaw_joint": "thumb_joint_0",
    "index_intermediate_joint": "index_joint_1",
    "pinky_intermediate_joint": "little_joint_1",
    "middle_intermediate_joint": "middle_joint_1",
    "ring_intermediate_joint": "ring_joint_1",
    "thumb_proximal_pitch_joint": "thumb_joint_1",
    "thumb_intermediate_joint": "thumb_joint_2",
    "thumb_distal_joint": "thumb_joint_3",
}

#: The 19 columns of ``joint_pos``, in the order the existing artifacts use.
ARTIFACT_JOINT_NAMES: Tuple[str, ...] = tuple(ARM_JOINTS) + (
    "index_joint_0", "little_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
    "index_joint_1", "little_joint_1", "middle_joint_1", "ring_joint_1",
    "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
)


def _atomic_json(path: Path, document) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class HandRecording:
    """Measured hand joint angles from the bag, driven six only, radians."""

    stamp_ns: np.ndarray
    driven: np.ndarray  # (N, 6) in kin.DRIVEN_JOINTS order


def read_hand(bag_dir, topic: str = HAND_JOINT_STATES_TOPIC) -> HandRecording:
    """Pull the six driven hand joints out of the session bag.

    Read by name rather than by position: the driver publishes twelve joints and
    optionally a name prefix, so an index into the message would follow the
    wrong finger the first time either changes.
    """
    stamps: List[int] = []
    rows: List[List[float]] = []
    columns: Optional[List[int]] = None
    for _topic, message, receive_ns in read_messages(str(bag_dir), [topic]):
        if columns is None:
            names = list(message.name)
            suffixes = [name.split("/")[-1] for name in names]
            missing = [joint for joint in kin.DRIVEN_JOINTS if joint not in suffixes]
            if missing:
                # A prefixed hand is fine; a hand that is not publishing the
                # driven joints at all is not, and silently emitting zeros for
                # them would be the worst possible outcome here.
                raise ValueError(
                    f"{topic} does not carry the driven joints {missing}; it published {names}"
                )
            columns = [suffixes.index(joint) for joint in kin.DRIVEN_JOINTS]
        stamps.append(stamp_to_ns(message.header.stamp) or receive_ns)
        rows.append([float(message.position[index]) for index in columns])
    if not stamps:
        raise ValueError(f"the bag holds no {topic} messages")
    order = np.argsort(np.asarray(stamps), kind="stable")
    return HandRecording(
        stamp_ns=np.asarray(stamps, dtype=np.int64)[order],
        driven=np.asarray(rows, dtype=float)[order],
    )


def bag_topics(bag_dir) -> List[str]:
    """Topic names the bag holds, from its own metadata.yaml.

    Read off the metadata rather than opened through rosbag2: this is asked
    before any deserialisation happens, purely to decide *which* topics are
    worth deserialising, and paying a reader open to find out would defeat it.
    A bag with no readable metadata returns nothing, and the caller falls back
    to asking for everything -- the old behaviour.
    """
    path = Path(bag_dir) / "metadata.yaml"
    if not path.is_file():
        return []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    information = document.get("rosbag2_bagfile_information") or {}
    names = []
    for entry in information.get("topics_with_message_count") or []:
        metadata = (entry or {}).get("topic_metadata") or {}
        name = metadata.get("name")
        if name:
            names.append(str(name))
    return names


def arm_joint_states_topic(bag_dir) -> str:
    """Which joint_states topic carries the arm in this bag.

    Prefers the 1 kHz arm-only publication over the 30 Hz merged one; see
    :data:`ARM_JOINT_STATES_TOPICS`.
    """
    present = set(bag_topics(bag_dir))
    for topic in ARM_JOINT_STATES_TOPICS:
        if topic in present:
            return topic
    return ARM_JOINT_STATES_TOPICS[-1]


def load_events(path) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def lowpass(values: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass down each column. ``cutoff_hz <= 0`` is a no-op."""
    if not cutoff_hz or cutoff_hz <= 0.0:
        return np.array(values, dtype=float, copy=True)
    from scipy.signal import butter, sosfiltfilt

    nyquist = rate_hz / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff {cutoff_hz:g} Hz must be below the Nyquist rate {nyquist:g} Hz")

    # Second-order sections rather than the (b, a) transfer function. At the
    # cutoffs used here the two are numerically indistinguishable -- measured,
    # not assumed -- but SOS is the form that stays well conditioned if the
    # order is ever raised or the corner lowered, and it costs nothing.
    sections = butter(4, cutoff_hz / nyquist, output="sos")
    # The padding, on the other hand, matters a great deal. scipy pads by a
    # fixed few samples regardless of where the corner sits; this filter settles
    # over hundreds. On a 2 Hz corner at 1 kHz that default leaves 18 mrad of
    # transient on the very first sample -- which is the sample the homing pose
    # is read off, and the one the replay lead-in ramp attaches to. Padding by
    # the filter's own settling time instead brings it to 3e-5 rad.
    padlen = min(len(values) - 1, max(12, int(round(3.0 * rate_hz / cutoff_hz))))
    return sosfiltfilt(sections, np.asarray(values, dtype=float), axis=0, padlen=padlen)


def to_uniform(stamp_ns: np.ndarray, values: np.ndarray, rate_hz: float):
    """Interpolate an irregularly stamped stream onto a uniform grid at ``rate_hz``.

    Filtering assumes an even sample spacing, and a 1 kHz stream that dropped a
    packet somewhere does not have one. Interpolating first costs nothing at
    this rate and makes the filter's corner mean what it says.
    """
    t = (stamp_ns - stamp_ns[0]) * 1e-9
    count = int(np.floor(t[-1] * rate_hz)) + 1
    grid = np.arange(count) / rate_hz
    columns = [np.interp(grid, t, values[:, index]) for index in range(values.shape[1])]
    return grid, np.column_stack(columns)


def sample_at(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(target_t, source_t, values[:, index]) for index in range(values.shape[1])]
    )


def central_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Velocity of a sampled trajectory, matching what ``joint_pos`` holds."""
    return np.gradient(np.asarray(values, dtype=float), dt, axis=0)


def hand_command_radians(events: Sequence[dict], grid_ns: np.ndarray, fallback: np.ndarray):
    """The commanded hand pose in force at each grid instant, in radians.

    This is the demonstration's action channel and it is read from the operator's
    own keypresses, not inferred from measured finger angles. Before the first
    keypress there is no command: the hand was simply holding whatever it held,
    so ``fallback`` (the first measured pose) stands in, and the returned mask
    says which samples that applies to. Nothing here invents an action.
    """
    commands = [
        (int(event["t_ros_ns"]), event)
        for event in events
        if event.get("event") == "hand_command"
    ]
    commands.sort(key=lambda item: item[0])

    target = np.tile(np.asarray(fallback, dtype=float), (len(grid_ns), 1))
    commanded = np.zeros(len(grid_ns), dtype=bool)
    names: List[str] = []
    for stamp_ns, event in commands:
        radians = event.get("open_ratio_rad") or {}
        if not names:
            names = list(kin.DRIVEN_JOINTS)
        row = np.array([float(radians[name]) for name in names], dtype=float)
        active = grid_ns >= stamp_ns
        target[active] = row
        commanded[active] = True
    return target, commanded


def expand_followers(driven: np.ndarray) -> Dict[str, np.ndarray]:
    """Twelve hand joint columns from the six driven ones, by the driver's coupling."""
    columns: Dict[str, np.ndarray] = {}
    for index, dof in enumerate(kin.DOFS):
        columns[dof.joint] = driven[:, index]
        for coupling in dof.couplings:
            columns[coupling.joint] = np.clip(
                coupling.multiplier * driven[:, index] + coupling.offset,
                coupling.lower,
                coupling.upper,
            )
    return columns


def pose16_from_joint_angles(q: np.ndarray) -> np.ndarray:
    """Flange poses by forward kinematics, packed the way libfranka packs O_T_EE.

    The fallback for a recording that has no ``O_T_EE`` -- which means any
    simulated one, because that field is libfranka's and MuJoCo does not have
    it. On hardware the recorded pose is used instead and this is not called:
    it is the robot's own answer, and it already contains whatever the arm's
    calibration says that this nominal DH chain does not.
    """
    q = np.asarray(q, dtype=float)
    return np.stack([flange_transform(row).reshape(16, order="F") for row in q])


def tcp_from_pose16(pose16: np.ndarray, tool: np.ndarray):
    """``O_T_EE`` columns to TCP position and a WXYZ quaternion.

    Two conversions worth naming, because both are silent if got wrong:
    libfranka packs the pose column-major, and
    ``franka_trajectory_replay.kinematics.matrix_to_quaternion`` returns XYZW
    while every replay artifact's ``tcp_quat`` is WXYZ.
    """
    positions = np.empty((len(pose16), 3))
    quaternions = np.empty((len(pose16), 4))
    for index, row in enumerate(pose16):
        transform = column_major_to_transform(row) @ tool
        positions[index] = transform[:3, 3]
        x, y, z, w = matrix_to_quaternion(transform[:3, :3])
        quaternions[index] = (w, x, y, z)
    return positions, quaternions


def marker_indices(events, kind: str, grid_ns: np.ndarray, start_ns: int, end_ns: int):
    """Sample indices of one kind of operator mark, inside the extracted window.

    Snapped to the nearest emitted sample rather than interpolated: a mark is a
    moment the operator chose, and the useful thing to record is which waypoint
    it lands on.
    """
    stamps = [
        int(event["t_ros_ns"])
        for event in events
        if event.get("event") == kind and start_ns <= int(event["t_ros_ns"]) <= end_ns
    ]
    return np.array(
        [int(np.argmin(np.abs(grid_ns - stamp))) for stamp in stamps], dtype=np.int64
    )


def _segment_window(events, start_ns: int, end_ns: int, segment: Optional[int]):
    """The [start, end) instants of one marked segment, or the whole session."""
    boundaries = sorted(
        int(event["t_ros_ns"]) for event in events if event.get("event") == "segment"
    )
    edges = [start_ns, *[b for b in boundaries if start_ns < b < end_ns], end_ns]
    spans = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    if segment is None:
        if len(spans) > 1:
            print(
                f"note: this session is marked into {len(spans)} segments; "
                "extracting all of them as one trajectory. Use --segment N for one.",
                flush=True,
            )
        return start_ns, end_ns, None, len(spans)
    if not 0 <= segment < len(spans):
        raise ValueError(
            f"segment {segment} does not exist; this session has {len(spans)} "
            f"(0..{len(spans) - 1}), marked with the 's' key"
        )
    return spans[segment][0], spans[segment][1], segment, len(spans)


@dataclass
class Extraction:
    """Everything the artifact and its report are written from."""

    time: np.ndarray
    arm: np.ndarray
    arm_raw: np.ndarray
    hand_driven: np.ndarray
    hand_target: np.ndarray
    hand_commanded: np.ndarray
    tcp_pos: np.ndarray
    tcp_quat: np.ndarray
    captures: np.ndarray
    grid_ns: np.ndarray
    segment: Optional[int]
    segment_count: int


def extract_session(
    session_dir,
    rate_hz: float = DEFAULT_RATE_HZ,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    segment: Optional[int] = None,
    work_dir=None,
) -> Tuple[Extraction, dict]:
    """Read a session and build the 15 Hz trajectory, without writing the artifact."""
    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    bag_dir = Path(manifest.get("bag_dir") or (session_dir / "bag"))
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"{session_dir} has no bag at {bag_dir}")
    events = load_events(session_dir / "events.jsonl")

    # The arm goes through the shared extractor, so a full-state session's
    # data.npz carries exactly the FrankaRobotState field set a replay run's
    # does -- which is the point of recording all of it. A lean session has no
    # robot_state at all; asking for it anyway is harmless (extract_bag only
    # errors on a role that was requested *and* is the run's clock), and it is
    # what lets one code path read both kinds of bag.
    work_dir = Path(work_dir) if work_dir is not None else session_dir
    arm_npz = work_dir / "session_arm.npz"
    joint_states_topic = arm_joint_states_topic(bag_dir)
    extract_bag(
        str(bag_dir),
        {"robot_state": ROBOT_STATE_TOPIC, "joint_states": joint_states_topic},
        list(ARM_JOINTS),
        str(arm_npz),
    )
    with np.load(arm_npz, allow_pickle=False) as data:
        if "rs_stamp_ns" in data:
            # Hardware: the FCI's own state, including the pose the robot
            # reports rather than one this tool computed.
            arm_stamp_ns = np.asarray(data["rs_stamp_ns"], dtype=np.int64)
            arm_q = np.asarray(data["rs_q"], dtype=float)
            pose16 = np.asarray(data["rs_O_T_EE"], dtype=float)
            arm_source = ROBOT_STATE_TOPIC
            tcp_source = "recorded O_T_EE"
        elif "js_stamp_ns" in data:
            # A lean hardware capture or a simulated one: joint angles are all
            # there is, so the pose is forward kinematics of them. Recorded as
            # such, because it is a different measurement with different error,
            # not the same number by another route.
            arm_stamp_ns = np.asarray(data["js_stamp_ns"], dtype=np.int64)
            arm_q = np.asarray(data["js_q"], dtype=float)
            pose16 = pose16_from_joint_angles(arm_q)
            arm_source = joint_states_topic
            tcp_source = "forward kinematics of the recorded joint angles"
        else:
            raise ValueError(
                f"the bag holds neither {ROBOT_STATE_TOPIC} nor {joint_states_topic}; "
                "there is no arm state to extract"
            )

    hand = read_hand(bag_dir)

    start_ns = int(max(arm_stamp_ns[0], hand.stamp_ns[0]))
    end_ns = int(min(arm_stamp_ns[-1], hand.stamp_ns[-1]))
    if end_ns <= start_ns:
        raise ValueError("the arm and hand recordings do not overlap in time")
    start_ns, end_ns, segment, segment_count = _segment_window(events, start_ns, end_ns, segment)

    dt = 1.0 / float(rate_hz)
    count = int(np.floor((end_ns - start_ns) * 1e-9 / dt)) + 1
    if count < 4:
        raise ValueError(
            f"only {count} samples at {rate_hz:g} Hz; this session is too short to spline"
        )
    grid_ns = start_ns + (np.arange(count) * dt * 1e9).astype(np.int64)
    time = (grid_ns - grid_ns[0]) * 1e-9

    # Uniform 1 kHz -> filter -> 15 Hz. Filtering before decimating is what makes
    # the low-pass an anti-alias filter rather than a cosmetic smoother.
    arm_rate = 1.0 / float(np.median(np.diff(arm_stamp_ns)) * 1e-9)
    uniform_t, uniform_q = to_uniform(arm_stamp_ns, arm_q, arm_rate)
    filtered = lowpass(uniform_q, arm_rate, cutoff_hz)
    grid_t_arm = (grid_ns - arm_stamp_ns[0]) * 1e-9
    arm = sample_at(uniform_t, filtered, grid_t_arm)
    arm_raw = sample_at(uniform_t, uniform_q, grid_t_arm)

    hand_t = (hand.stamp_ns - hand.stamp_ns[0]) * 1e-9
    hand_driven = sample_at(hand_t, hand.driven, (grid_ns - hand.stamp_ns[0]) * 1e-9)
    hand_target, hand_commanded = hand_command_radians(events, grid_ns, hand_driven[0])

    # The TCP comes from the *unfiltered* O_T_EE sampled on the same grid: it is
    # a recorded measurement of where the hand was, and smoothing it would make
    # it disagree with the joint angles it is supposed to describe.
    config = load_config(None)
    tool = tool_transform(config["tcp"]["offset_xyz"], config["tcp"]["offset_rpy"])
    tcp_pose = sample_at((arm_stamp_ns - arm_stamp_ns[0]) * 1e-9, pose16, grid_t_arm)
    tcp_pos, tcp_quat = tcp_from_pose16(tcp_pose, tool)

    captures = marker_indices(events, "pose_capture", grid_ns, start_ns, end_ns)

    extraction = Extraction(
        time=time,
        arm=arm,
        arm_raw=arm_raw,
        hand_driven=hand_driven,
        hand_target=hand_target,
        hand_commanded=hand_commanded,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        captures=captures,
        grid_ns=grid_ns,
        segment=segment,
        segment_count=segment_count,
    )
    context = {
        "manifest": manifest,
        "events": events,
        "session_dir": str(session_dir),
        "bag_dir": str(bag_dir),
        "arm_state_topic": arm_source,
        "tcp_source": tcp_source,
        "simulated": bool(manifest.get("simulated", arm_source != ROBOT_STATE_TOPIC)),
        "arm_rate_hz": float(arm_rate),
        "arm_samples": int(len(arm_stamp_ns)),
        "hand_samples": int(len(hand.stamp_ns)),
        "arm_npz": str(arm_npz),
    }
    return extraction, context


def build_arrays(extraction: Extraction, rate_hz: float):
    """The artifact's NPZ arrays, in the (time, environment, component) layout."""
    dt = 1.0 / float(rate_hz)
    count = len(extraction.time)

    measured = expand_followers(extraction.hand_driven)
    commanded = expand_followers(extraction.hand_target)

    def nineteen(arm, hand_columns):
        block = np.empty((count, len(ARTIFACT_JOINT_NAMES)))
        block[:, : len(ARM_JOINTS)] = arm
        hand_names = ARTIFACT_JOINT_NAMES[len(ARM_JOINTS):]
        for index, name in enumerate(hand_names, start=len(ARM_JOINTS)):
            driver_name = next(k for k, v in DRIVER_TO_ARTIFACT_JOINT.items() if v == name)
            block[:, index] = hand_columns[driver_name]
        return block[:, None, :]

    joint_pos = nineteen(extraction.arm, measured)
    joint_pos_target = nineteen(extraction.arm, commanded)
    joint_vel = central_difference(joint_pos[:, 0, :], dt)[:, None, :]

    return {
        "sample_time_s": extraction.time,
        "step": np.arange(count, dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_pos_target": joint_pos_target,
        "tcp_pos": extraction.tcp_pos[:, None, :],
        "tcp_quat": extraction.tcp_quat[:, None, :],
        "hand_command_active": extraction.hand_commanded,
        "pose_capture_index": extraction.captures,
        "capture_stamp_ns": extraction.grid_ns,
    }


def filter_report(extraction: Extraction, rate_hz: float, cutoff_hz: float) -> dict:
    """How far the filter moved each joint, and what the derivatives came out at.

    This is the accounting for the one thing this tool changes about a
    demonstration, so it is written next to the artifact rather than printed and
    forgotten.
    """
    dt = 1.0 / float(rate_hz)
    deviation = extraction.arm - extraction.arm_raw
    velocity = central_difference(extraction.arm, dt)
    raw_velocity = central_difference(extraction.arm_raw, dt)
    return {
        "cutoff_hz": float(cutoff_hz),
        "rate_hz": float(rate_hz),
        "samples": int(len(extraction.time)),
        "duration_s": float(extraction.time[-1]),
        "joints": [
            {
                "joint": name,
                "filter_rms_rad": float(np.sqrt(np.mean(deviation[:, index] ** 2))),
                "filter_max_rad": float(np.max(np.abs(deviation[:, index]))),
                "range_rad": [
                    float(extraction.arm[:, index].min()),
                    float(extraction.arm[:, index].max()),
                ],
                "peak_velocity_rad_s": float(np.max(np.abs(velocity[:, index]))),
                "peak_velocity_unfiltered_rad_s": float(np.max(np.abs(raw_velocity[:, index]))),
            }
            for index, name in enumerate(ARM_JOINTS)
        ],
    }


def format_filter_table(report: dict) -> str:
    lines = [
        "",
        f"extracted {report['samples']} samples at {report['rate_hz']:g} Hz "
        f"({report['duration_s']:.2f} s), low-pass {report['cutoff_hz']:g} Hz zero-phase",
        f"  {'joint':<12} {'min':>9} {'max':>9} {'filt rms':>10} {'filt max':>10} "
        f"{'|qd| raw':>10} {'|qd| filt':>10}",
    ]
    for entry in report["joints"]:
        lines.append(
            f"  {entry['joint']:<12} {entry['range_rad'][0]:+9.4f} {entry['range_rad'][1]:+9.4f} "
            f"{entry['filter_rms_rad']:10.5f} {entry['filter_max_rad']:10.5f} "
            f"{entry['peak_velocity_unfiltered_rad_s']:10.3f} {entry['peak_velocity_rad_s']:10.3f}"
        )
    lines.append(
        "  'filt' columns are how far the filter moved the waypoints; "
        "'|qd|' are peak 15 Hz finite-difference speeds, before and after."
    )
    return "\n".join(lines)


def write_plot(path: Path, extraction: Extraction) -> Optional[Path]:
    """Raw against extracted, per arm joint. Returns None if matplotlib is absent.

    Two columns, because one is not enough: at the scale of the motion the two
    traces sit on top of each other and the plot looks like proof that the
    filter does nothing. The right column is the difference, in milliradians,
    which is the only place the filter is actually visible.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    residual = 1000.0 * (extraction.arm - extraction.arm_raw)
    figure, axes = plt.subplots(7, 2, figsize=(13, 15), sharex=True,
                                gridspec_kw={"width_ratios": [2, 1]})
    for index, name in enumerate(ARM_JOINTS):
        left, right = axes[index]
        left.plot(extraction.time, extraction.arm_raw[:, index], linewidth=1.6,
                  color="0.55", label="recorded (sampled at the artifact rate)")
        left.plot(extraction.time, extraction.arm[:, index], linewidth=1.0,
                  color="tab:orange", label="extracted waypoints")
        right.plot(extraction.time, residual[:, index], linewidth=0.9, color="tab:blue")
        right.axhline(0.0, color="0.7", linewidth=0.6)
        for sample in extraction.captures:
            left.axvline(extraction.time[sample], color="tab:green",
                         linestyle=":", linewidth=1.0)
        left.set_ylabel(name.replace("fr3_", ""))
        right.set_ylabel("mrad", fontsize="small")
        right.tick_params(labelsize="small")
        if index == 0:
            left.legend(loc="upper right", fontsize="small")
            right.set_title("extracted - recorded", fontsize="small")
    axes[-1][0].set_xlabel("time [s]  (dotted: operator pose captures)")
    axes[-1][1].set_xlabel("time [s]")
    figure.suptitle("Hand-guided capture: recording against extracted waypoints")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def write_artifact(
    output_dir: Path,
    extraction: Extraction,
    context: dict,
    rate_hz: float,
    cutoff_hz: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_arrays(extraction, rate_hz)
    report = filter_report(extraction, rate_hz, cutoff_hz)

    _atomic_npz(output_dir / "replay_data.npz", arrays)

    manifest = context["manifest"]
    simulated = bool(context.get("simulated"))
    metadata = {
        # Never the same word as a hardware capture. Everything downstream that
        # asks "what is this" gets a different answer for a rehearsal, so no
        # amount of copying directories can turn one into training data.
        "replay": "hand_guided_sim" if simulated else "hand_guided",
        "simulated": simulated,
        "schema_version": SCHEMA_VERSION,
        "task": manifest.get("note", ""),
        "source": {
            "session_dir": context["session_dir"],
            "bag_dir": context["bag_dir"],
            "capture_tool": "inspire_franka_trajectory_replay.capture",
            "bringup_command": manifest.get("bringup_command", ""),
            "gravity_compensation": manifest.get("gravity_compensation"),
            "active_controllers": manifest.get("active_controllers"),
            "hand_presets": manifest.get("hand_presets", {}).get("path"),
            "hand_presets_sha256": manifest.get("hand_presets", {}).get("sha256"),
            "arm_state_topic": context.get("arm_state_topic"),
            "tcp_source": context.get("tcp_source"),
            "arm_record_rate_hz": context["arm_rate_hz"],
            "arm_samples_recorded": context["arm_samples"],
            "hand_samples_recorded": context["hand_samples"],
            "segment": extraction.segment,
            "segment_count": extraction.segment_count,
        },
        "hardware_orientation": {
            "joint": "fr3_joint7",
            "offset_rad": 0.0,
            "note": "Recorded joint 7 is written through unchanged and homing.yaml is read "
                    "off the same first sample. Do not apply the legacy +90-degree "
                    "compensation, retarget_flange_mount.py, or a *_flange180 derivative.",
        },
        "dt": 1.0 / float(rate_hz),
        "recording_frequency_hz": float(rate_hz),
        "joint_names": list(ARTIFACT_JOINT_NAMES),
        "arm_joint_names": list(ARM_JOINTS),
        "arm_joint_ids": list(range(len(ARM_JOINTS))),
        "hand_joint_names": [DRIVER_TO_ARTIFACT_JOINT[name] for name in HAND_JOINTS],
        "hand_joint_ids": [
            ARTIFACT_JOINT_NAMES.index(DRIVER_TO_ARTIFACT_JOINT[name]) for name in HAND_JOINTS
        ],
        "home": "homing.yaml",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(len(extraction.time)),
        "data_file": "replay_data.npz",
        "fields": {key: list(np.shape(value)) for key, value in arrays.items()},
        "trajectory_layout": "time, environment, component",
        "filter": {
            "kind": "zero-phase Butterworth low-pass, order 4, scipy filtfilt",
            "cutoff_hz": float(cutoff_hz),
            "applied_to": "the seven arm joint positions, at the recorded rate, before "
                          "decimation to the artifact rate",
            "why": "anti-aliases the 15 Hz decimation (Nyquist 7.5 Hz) and removes the "
                   "operator tremor and structural ring that hand guiding injects, which is "
                   "what brings the trajectory's derivatives inside the FR3 limit guards",
            "not_applied_to": "tcp_pos/tcp_quat and the hand, which are recorded "
                              "measurements rather than a commanded path",
            "report": "extraction_report.json",
        },
        "coordinate_convention": {
            "quaternion": "WXYZ",
            "joint_position": "radians",
            "joint_velocity": "radians/second, central difference of joint_pos at dt",
            "tcp_position": "metres in the robot base frame fr3_link0 - NOT the "
                            "environment-relative frame the Forge captures use; a "
                            "hand-guided demonstration has no simulation environment origin",
            "tcp_frame": "the flange pose composed with the replay.yaml tcp offset (the "
                         "Inspire grasp centre, 173 mm from the flange); see "
                         "source.tcp_source for whether the flange pose was the robot's "
                         "own O_T_EE or forward kinematics of the joint angles",
        },
        "temporal_alignment": {
            "joint_pos_target": "the hand block is the preset in force at that instant, from "
                                "the operator's own keypresses in events.jsonl; the arm block "
                                "is joint_pos itself, because a hand-guided demonstration has "
                                "no arm command - the capture IS the reference a replay tracks",
            "hand_command_active": "false before the first keypress, where joint_pos_target's "
                                   "hand block falls back to the first measured pose",
        },
        "limitations": (
            [
                "Recorded in MuJoCo, not on the FCI. There is no measured torque, no "
                "external wrench, no collision or contact indicator, no robot mode and "
                "no load model, because franka_msgs/FrankaRobotState does not exist in "
                "simulation. tcp_pos/tcp_quat are forward kinematics of the joint "
                "angles rather than the robot's own O_T_EE. This is a rehearsal of the "
                "capture and extraction path; it is not training data.",
            ]
            if simulated
            else []
        ),
        "markers": {
            "pose_capture_index": "sample indices of the operator's 'c' keypresses; the "
                                  "full joint state recorded at each is in the session's "
                                  "events.jsonl",
            "segments_marked": extraction.segment_count,
            "note": "replay_trajectory's own --segment splits on unreachable steps, not on "
                    "these marks; select a marked segment at extraction time instead",
        },
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    _atomic_json(output_dir / "extraction_report.json", report)

    home_names = list(ARM_JOINTS) + list(HAND_JOINTS)
    home_values = [float(value) for value in extraction.arm[0]] + [
        float(value) for value in extraction.hand_driven[0]
    ]
    home = {
        "schema_version": 1,
        "name": f"{output_dir.name}_homing",
        "description": "First extracted sample of a hand-guided capture.",
        "units": "radians",
        "joint_names": home_names,
        "positions": home_values,
        "source": {
            "session_dir": context["session_dir"],
            "derived_from": "replay_data.npz sample 0",
            "method": "Read off the first extracted waypoint, so the home and the "
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

    plot = write_plot(output_dir / "extraction.png", extraction)
    return plot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("session", help="a capture session directory written by capture_demo")
    parser.add_argument(
        "-o", "--output", default=None,
        help="artifact directory (default: <session>/artifact, or artifact_segment<N>)",
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE_HZ,
        help=f"artifact sample rate in Hz "
             f"(default: {DEFAULT_RATE_HZ:g}, the workspace convention)",
    )
    parser.add_argument(
        "--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
        help=f"zero-phase low-pass corner for the arm joints (default: {DEFAULT_CUTOFF_HZ:g}); "
             "0 disables it, which will normally fail the replay limit guards",
    )
    parser.add_argument(
        "--segment", type=int, default=None,
        help="extract only the Nth span between the operator's 's' marks (0-based)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    args = _parser().parse_args(argv)
    if args.rate <= 0:
        _parser().error("--rate must be positive")
    if args.cutoff_hz < 0:
        _parser().error("--cutoff-hz must not be negative")

    session = Path(args.session).expanduser()
    try:
        extraction, context = extract_session(
            session, rate_hz=args.rate, cutoff_hz=args.cutoff_hz, segment=args.segment
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"extraction error: {exc}", file=sys.stderr)
        return 2

    suffix = "" if extraction.segment is None else f"_segment{extraction.segment}"
    output = Path(args.output) if args.output else session / f"artifact{suffix}"
    plot = write_artifact(output, extraction, context, args.rate, args.cutoff_hz)

    report = json.loads((output / "extraction_report.json").read_text(encoding="utf-8"))
    print(format_filter_table(report), flush=True)
    print(f"\nartifact: {output}", flush=True)
    if plot is not None:
        print(f"plot:     {plot}", flush=True)
    print(
        "\ncheck it against the replay stack's own guards before any motion:\n"
        f"  ros2 run inspire_franka_trajectory_replay replay_trajectory {output} \\\n"
        f"      --home {output}/homing.yaml --dry-run",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
