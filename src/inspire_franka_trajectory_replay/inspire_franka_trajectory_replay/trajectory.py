"""Load coordinated FR3/Inspire trajectories, including Forge replay artifacts."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np

from franka_trajectory_replay.limits import VELOCITY_MAX


ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)
FINGER_FLEXION_JOINTS = (
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
)
FINGER_FLEXION_INDICES = tuple(
    HAND_JOINTS.index(name) for name in FINGER_FLEXION_JOINTS
)
FORGE_HAND_JOINTS = (
    "little_joint_0",
    "ring_joint_0",
    "middle_joint_0",
    "index_joint_0",
    "thumb_joint_1",
    "thumb_joint_0",
)
PASSIVE_HAND_JOINTS = {
    "pinky_intermediate_joint",
    "ring_intermediate_joint",
    "middle_intermediate_joint",
    "index_intermediate_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
}

# A Forge recording is one file per *run*, not per episode: when the task
# resets, the simulator teleports the arm back to its start pose between two
# consecutive samples. Splining through that teleport is what produced a
# demanded 793 rad/s^2 on joint 6 -- 15863 % of the limit -- from a recording
# whose real motion never exceeds 1.2 rad/s.
#
# The threading recordings carry a ``cycle`` field that marks the boundaries.
# The pickplace ones do not, so the boundaries are found instead: a step the
# FR3 could not make even at its own velocity limit did not happen, it is a
# reset. That is a property of the arm rather than a tuned threshold -- the
# margin is wide, with real steps under 0.07 rad and resets over 0.8 rad
# against a 0.17 rad bound.
RESET_VELOCITY = VELOCITY_MAX

# Segments too short to spline are listed but not offered. Four samples is the
# smallest run a cubic fit says anything about; the fragments this drops are
# the 3-sample tails left when a recording stops mid-episode.
MIN_SEGMENT_SAMPLES = 4


@dataclass(frozen=True)
class CoordinatedTrajectory:
    time: np.ndarray
    arm: np.ndarray
    hand: Optional[np.ndarray]
    source: Path
    cycle: Optional[int] = None
    segment: Optional[int] = None

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


def scale_finger_flexion(hand, scale):
    """Scale only index- and thumb-MCP waypoint angles from their open pose."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("finger flexion scale must be finite and positive")
    if hand is None:
        raise ValueError(
            "--finger-flexion-scale requires Inspire hand positions in the trajectory"
        )
    scaled = np.array(hand, dtype=float, copy=True)
    scaled[..., list(FINGER_FLEXION_INDICES)] *= scale
    return scaled


def resolve_trajectory(path: str) -> Path:
    """Resolve an NPZ path or a trajectory directory."""
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        metadata_path = source / "metadata.json"
        data_name = "replay_data.npz"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data_name = metadata.get("data_file", data_name)
        source = source / data_name
    if source.suffix.lower() != ".npz":
        raise ValueError("trajectory must be an NPZ or a directory containing replay_data.npz")
    return source


def _metadata(source: Path) -> dict:
    path = source.parent / "metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _time_from_npz(data, count: int, rate: Optional[float], metadata: dict) -> np.ndarray:
    for key in ("sample_time_s", "t", "time"):
        if key in data:
            return np.asarray(data[key], dtype=float)
    if "dt" in data:
        dt = float(np.asarray(data["dt"]))
    elif "rate" in data:
        dt = 1.0 / float(np.asarray(data["rate"]))
    elif metadata.get("timing", {}).get("sample_period_s"):
        dt = float(metadata["timing"]["sample_period_s"])
    elif metadata.get("recording_frequency_hz"):
        dt = 1.0 / float(metadata["recording_frequency_hz"])
    elif rate:
        dt = 1.0 / rate
    else:
        raise ValueError("trajectory needs sample_time_s/t/time, dt, rate, or --rate")
    return np.arange(count, dtype=float) * dt


def _columns(names, expected, label):
    if names is None:
        return list(range(len(expected)))
    names = [str(value) for value in names]
    if any(name in PASSIVE_HAND_JOINTS for name in names):
        raise ValueError("passive Inspire hand joints must not be commanded")
    missing = [name for name in expected if name not in names]
    if missing:
        raise ValueError(f"{label} trajectory is missing joints: {missing}")
    return [names.index(name) for name in expected]


def _forge_rows(data, cycle: Optional[int]) -> tuple[np.ndarray, Optional[int]]:
    """Select one continuous rollout from a multi-cycle Forge recording."""
    count = len(data["joint_pos"])
    if "cycle" not in data:
        if cycle is not None:
            raise ValueError("--cycle was supplied, but this Forge NPZ has no cycle field")
        return np.arange(count), None

    recorded = np.asarray(data["cycle"])
    if recorded.ndim != 1 or len(recorded) != count:
        raise ValueError("Forge cycle field must have shape (time,)")
    available = [int(value) for value in np.unique(recorded)]
    if cycle is None and len(available) > 1:
        raise ValueError(
            "Forge recording contains multiple rollout cycles "
            f"{available}; select one continuous run with --cycle N"
        )
    selected = available[0] if cycle is None else cycle
    if selected not in available:
        raise ValueError(f"cycle {selected} is not present; available cycles: {available}")
    return np.flatnonzero(recorded == selected), selected


def _reset_steps(arm: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Index every sample the FR3 could not have reached from its predecessor.

    Compared against the arm's own per-joint velocity limit over the actual
    sample spacing, so it stays correct if a recording is ever made at a rate
    other than the 15 Hz these are.
    """
    reachable = RESET_VELOCITY * np.diff(time)[:, None]
    return np.flatnonzero((np.abs(np.diff(arm, axis=0)) > reachable).any(axis=1)) + 1


def _segment_rows(arm: np.ndarray, time: np.ndarray, segment: Optional[int]):
    """Split one run at its resets and pick a single continuous episode.

    Returns row indices into ``arm``, and the segment number chosen. A
    recording with no reset in it is one segment and needs no selection, which
    is why the ``cycle``-bearing threading captures are unaffected.
    """
    breaks = _reset_steps(arm, time)
    bounds = [0, *breaks.tolist(), len(arm)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if len(spans) == 1:
        # No reset in these rows, so there is nothing to choose between and no
        # opinion to have about how long a recording is allowed to be. Whether
        # it holds enough samples to replay is load_trajectory's check, not
        # this one's.
        if segment not in (None, 0):
            raise ValueError(f"segment {segment} does not exist; this recording is continuous")
        return np.arange(len(arm)), None

    usable = [
        (index, start, stop)
        for index, (start, stop) in enumerate(spans)
        if stop - start >= MIN_SEGMENT_SAMPLES
    ]
    if not usable:
        raise ValueError(
            f"this recording resets {len(spans) - 1} times and every episode "
            f"between the resets is shorter than {MIN_SEGMENT_SAMPLES} samples"
        )
    if len(usable) == 1 and segment is None:
        index, start, stop = usable[0]
        return np.arange(start, stop), index

    catalogue = ", ".join(
        f"{index}: {stop - start} samples / {time[stop - 1] - time[start]:.1f} s"
        + ("" if stop - start >= MIN_SEGMENT_SAMPLES else " (too short)")
        for index, (start, stop) in enumerate(spans)
    )
    if segment is None:
        raise ValueError(
            f"this recording holds {len(usable)} episodes separated by a reset "
            f"the FR3 cannot follow; select one continuous run with --segment N "
            f"[{catalogue}]"
        )
    chosen = [entry for entry in usable if entry[0] == segment]
    if not chosen:
        raise ValueError(f"segment {segment} is not a usable episode [{catalogue}]")
    _, start, stop = chosen[0]
    return np.arange(start, stop), segment


def _load_forge(data, metadata: dict, environment: int, cycle: Optional[int],
                segment: Optional[int]):
    positions = np.asarray(data["joint_pos"], dtype=float)
    if positions.ndim != 3:
        return None
    if not 0 <= environment < positions.shape[1]:
        raise ValueError(
            f"environment {environment} is outside [0, {positions.shape[1] - 1}]"
        )
    names = tuple(str(name) for name in metadata.get("joint_names", ()))
    if len(names) != positions.shape[2]:
        raise ValueError("metadata joint_names does not match joint_pos width")
    rows, selected_cycle = _forge_rows(data, cycle)
    selected = positions[rows, environment, :]
    arm = selected[:, _columns(names, ARM_JOINTS, "arm")]
    hand = selected[:, _columns(names, FORGE_HAND_JOINTS, "Forge hand")]
    time = _time_from_npz(data, len(positions), None, metadata)[rows]
    # After the cycle field has had its say, and whether or not it exists: a
    # reset left inside the selected rows is the one thing the preparation
    # downstream cannot survive, so it is found here rather than discovered as
    # a limit violation two steps later.
    kept, selected_segment = _segment_rows(arm, time, segment)
    return time[kept], arm[kept], hand[kept], selected_cycle, selected_segment


def load_trajectory(
    path: str,
    rate: Optional[float] = None,
    environment: int = 0,
    cycle: Optional[int] = None,
    segment: Optional[int] = None,
) -> CoordinatedTrajectory:
    """Load a coordinated NPZ or the raw ``replay_data.npz`` Forge format."""
    source = resolve_trajectory(path)
    metadata = _metadata(source)
    with np.load(source, allow_pickle=False) as data:
        forge = (
            _load_forge(data, metadata, environment, cycle, segment)
            if "joint_pos" in data
            else None
        )
        if forge is not None:
            time, arm, hand, selected_cycle, selected_segment = forge
        else:
            if cycle is not None:
                raise ValueError("--cycle is only valid for a Forge NPZ with a cycle field")
            if segment is not None:
                raise ValueError("--segment is only valid for a Forge NPZ")
            selected_cycle = None
            selected_segment = None
            arm_key = next(
                (key for key in ("joint_pos_arm", "arm", "q_arm", "q") if key in data),
                None,
            )
            if arm_key is None:
                raise ValueError("NPZ must contain joint_pos_arm/arm/q_arm/q or Forge joint_pos")
            arm = np.asarray(data[arm_key], dtype=float)
            if arm.ndim != 2 or arm.shape[1] != 7:
                raise ValueError(f"arm trajectory must have shape (N, 7), got {arm.shape}")
            names = data["arm_joint_names"] if "arm_joint_names" in data else None
            arm = arm[:, _columns(names, ARM_JOINTS, "arm")]

            hand = None
            hand_key = next(
                (key for key in ("joint_pos_hand", "hand", "q_hand") if key in data),
                None,
            )
            if hand_key is not None:
                hand = np.asarray(data[hand_key], dtype=float)
                if hand.ndim != 2 or hand.shape[1] != 6:
                    raise ValueError(f"hand trajectory must have shape (N, 6), got {hand.shape}")
                hand_names = data["hand_joint_names"] if "hand_joint_names" in data else None
                hand = hand[:, _columns(hand_names, HAND_JOINTS, "hand")]
            time = _time_from_npz(data, len(arm), rate, metadata)

    time = np.asarray(time, dtype=float)
    time -= time[0]
    if hand is not None and len(hand) != len(arm):
        raise ValueError("arm and hand trajectories must contain the same number of samples")
    if len(time) != len(arm) or len(time) < 2:
        raise ValueError("trajectory time and arm data must have at least two matching samples")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise ValueError("trajectory time must be finite and strictly increasing")
    if not np.all(np.isfinite(arm)) or (hand is not None and not np.all(np.isfinite(hand))):
        raise ValueError("trajectory positions must be finite")
    return CoordinatedTrajectory(time, arm, hand, source, selected_cycle, selected_segment)


def resample(source: CoordinatedTrajectory, rate: float = 1000.0) -> CoordinatedTrajectory:
    """Linearly resample both assets onto one common command timeline."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    count = int(round(source.duration * rate)) + 1
    time = np.arange(count, dtype=float) / rate
    time[-1] = source.duration
    arm = np.column_stack(
        [np.interp(time, source.time, source.arm[:, joint]) for joint in range(7)]
    )
    hand = None
    if source.hand is not None:
        hand = np.column_stack(
            [np.interp(time, source.time, source.hand[:, joint]) for joint in range(6)]
        )
    return CoordinatedTrajectory(time, arm, hand, source.source, source.cycle, source.segment)
