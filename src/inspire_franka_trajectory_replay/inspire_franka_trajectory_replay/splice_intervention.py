"""Replace part of a replay artifact with a recorded intervention's waypoints.

The intervention recorder stores deliberate pose captures, not the continuous
path the operator's hand took.  This command therefore builds an explicit,
auditable approximation:

* the original artifact through the sample where the operator stepped in;
* minimum-jerk point-to-point motion through every captured pose;
* a minimum-jerk handback to the recorded release pose; and
* the original artifact after that release pose.

The result is an ordinary coordinated replay artifact and goes through the
same preparation and safety checks as every other replay::

    ros2 run inspire_franka_trajectory_replay splice_intervention logs/dagger/<session>
    ros2 run inspire_franka_trajectory_replay replay_trajectory \
        logs/dagger/<session>/composite \
        --home logs/dagger/<session>/composite/homing.yaml --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

from .extract import _atomic_json, _atomic_npz, _atomic_text, load_events
from .release_phase import from_metadata
from .trajectory import ARM_JOINTS, HAND_JOINTS, CoordinatedTrajectory, load_trajectory
from .waypoints import (
    DEFAULT_DWELL,
    DEFAULT_PEAK_SPEED,
    Waypoint,
    _from_snapshot,
    build_path,
    load_waypoints,
    mark_events,
)


@dataclass(frozen=True)
class Splice:
    """The generated arrays and the indices needed to audit their provenance."""

    time: np.ndarray
    arm: np.ndarray
    hand: np.ndarray
    paused_sample: int
    original_rejoin_sample: int
    composite_rejoin_sample: int
    composite_suffix_sample: int
    waypoint_samples: np.ndarray


def _existing_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = (Path.cwd() / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} recorded by the session does not exist: {value}")


def _uniform_rate(trajectory: CoordinatedTrajectory) -> float:
    spacing = np.diff(trajectory.time)
    dt = float(np.median(spacing))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("the original trajectory has no finite positive sample period")
    if not np.allclose(spacing, dt, rtol=1e-5, atol=1e-9):
        raise ValueError(
            "the original trajectory is not uniformly sampled; resample it before splicing"
        )
    return 1.0 / dt


def _selected_intervention(manifest: dict, index: Optional[int]) -> dict:
    records = list(manifest.get("interventions") or ())
    if not records:
        raise ValueError("the session manifest contains no interventions")
    if index is None:
        if len(records) != 1:
            available = [record.get("index") for record in records]
            raise ValueError(
                f"the session contains interventions {available}; select one with --intervention"
            )
        return records[0]
    selected = [record for record in records if int(record.get("index", -1)) == int(index)]
    if not selected:
        available = [record.get("index") for record in records]
        raise ValueError(f"intervention {index} is absent; available: {available}")
    return selected[0]


def _intervention_waypoints(session: Path, events: Sequence[dict], index: int) -> list[Waypoint]:
    marks = [
        mark for mark in mark_events(events)
        if int(mark.get("intervention", index)) == int(index)
    ]
    if not marks:
        raise ValueError(f"intervention {index} contains no captured poses")
    points = [_from_snapshot(mark) for mark in marks]
    if all(point is not None for point in points):
        return list(points)
    # The bag fallback cannot separate legacy marks that do not carry an
    # intervention id.  It is unambiguous only for a one-intervention session.
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest.get("interventions") or ()) != 1:
        raise ValueError(
            "some selected marks have no embedded joint-state snapshot, and a multi-intervention "
            "session cannot assign the bag samples unambiguously"
        )
    return load_waypoints(session)


def build_splice(
    original: CoordinatedTrajectory,
    paused_sample: int,
    rejoin_sample: int,
    waypoints: Sequence[Waypoint],
    rate_hz: float,
    dwell_s: float = DEFAULT_DWELL,
    peak_speed: float = DEFAULT_PEAK_SPEED,
) -> Splice:
    """Replace ``(paused_sample, rejoin_sample]`` with a path through the marks."""
    if original.hand is None:
        raise ValueError("the original trajectory has no hand channel")
    paused_sample = int(paused_sample)
    rejoin_sample = int(rejoin_sample)
    if not 0 <= paused_sample < rejoin_sample < len(original.time):
        raise ValueError(
            f"splice bounds must satisfy 0 <= pause < rejoin < {len(original.time)}; "
            f"got {paused_sample} and {rejoin_sample}"
        )
    if len(waypoints) < 1:
        raise ValueError("an intervention needs at least one captured pose")

    start = Waypoint(
        index=0,
        stamp_ns=0,
        arm=np.array(original.arm[paused_sample]),
        hand=np.array(original.hand[paused_sample]),
        source=f"original sample {paused_sample}",
    )
    end = Waypoint(
        index=-1,
        stamp_ns=0,
        arm=np.array(original.arm[rejoin_sample]),
        hand=np.array(original.hand[rejoin_sample]),
        source=f"original release sample {rejoin_sample}",
    )
    replacement = build_path(
        [start, *waypoints, end],
        rate_hz=rate_hz,
        dwell_s=dwell_s,
        peak_speed=peak_speed,
        lead_in_s=0.0,
    )

    # replacement[0] is the pause pose already present at the end of prefix.
    # The suffix starts one sample after the release pose; the replacement's
    # final dwell already reaches and holds that pose.
    arm = np.vstack(
        [original.arm[: paused_sample + 1], replacement.arm[1:], original.arm[rejoin_sample + 1 :]]
    )
    hand = np.vstack(
        [
            original.hand[: paused_sample + 1],
            replacement.hand[1:],
            original.hand[rejoin_sample + 1 :],
        ]
    )
    time = np.arange(len(arm), dtype=float) / float(rate_hz)
    # build_path's arrivals name the first dwell sample at each anchor.  Anchor
    # zero is the original pause, the last is the original release, and the
    # anchors between them are the operator's captured poses.
    mapped_arrivals = paused_sample + replacement.arrivals
    return Splice(
        time=time,
        arm=arm,
        hand=hand,
        paused_sample=paused_sample,
        original_rejoin_sample=rejoin_sample,
        composite_rejoin_sample=int(mapped_arrivals[-1]),
        composite_suffix_sample=paused_sample + len(replacement.time),
        waypoint_samples=np.asarray(mapped_arrivals[1:-1], dtype=np.int64),
    )


def remap_cycle_index(
    metadata: dict,
    splice: Splice,
    output_count: int,
    rate_hz: float,
) -> list[dict]:
    """Carry release flags into the composite artifact's sample numbering."""
    releases = from_metadata(metadata)
    if releases is None:
        raise ValueError(
            "the original artifact has no cycle_index release flags; it cannot identify "
            "where an intervention rejoins"
        )
    selected = [
        entry for entry in releases.releases
        if entry.release_sample == splice.original_rejoin_sample
    ]
    if len(selected) != 1 or not selected[0].contains(splice.paused_sample):
        raise ValueError(
            "the session pause/rejoin samples do not describe one cycle in the original "
            "artifact's cycle_index"
        )
    cycle = selected[0]
    # Original samples after the removed release sample resume after the whole
    # synthesized replacement (including its hand-settle dwell).
    suffix_shift = output_count - len_original_from_metadata(metadata)

    remapped = []
    for entry in releases.releases:
        def shifted(value: int) -> int:
            return int(value) if value <= splice.paused_sample else int(value) + suffix_shift

        release = (
            splice.composite_rejoin_sample
            if entry.cycle == cycle.cycle
            else shifted(entry.release_sample)
        )
        remapped.append(
            {
                "cycle": int(entry.cycle),
                "start_sample": shifted(entry.start_sample),
                "end_sample": shifted(entry.end_sample),
                "release_sample": int(release),
                "release_time_s": float(release) / float(rate_hz),
            }
        )
    return remapped


def len_original_from_metadata(metadata: dict) -> int:
    """The source length recorded in metadata, required for index shifting."""
    try:
        count = int(metadata["sample_count"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("the original metadata needs sample_count for a splice") from None
    if count < 2:
        raise ValueError(f"invalid original sample_count {count}")
    return count


def write_splice(
    output: Path,
    splice: Splice,
    original: CoordinatedTrajectory,
    original_metadata: dict,
    home_path: Path,
    session: Path,
    intervention: dict,
    waypoints: Sequence[Waypoint],
    rate_hz: float,
    dwell_s: float,
    peak_speed: float,
) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; choose a new --output directory"
        )
    original_count = len(original.time)
    if len_original_from_metadata(original_metadata) != original_count:
        raise ValueError(
            f"original metadata says {original_metadata.get('sample_count')} samples but "
            f"the trajectory contains {original_count}"
        )
    cycle_index = remap_cycle_index(original_metadata, splice, len(splice.time), rate_hz)
    metadata = {
        "schema_version": 1,
        "data_file": "replay_data.npz",
        "generated_by": "inspire_franka_trajectory_replay.splice_intervention",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Original replay with one interval replaced by minimum-jerk motion through "
            "the poses marked during a hand-guided intervention. This approximates the "
            "human path; the intervention did not record continuous arm motion."
        ),
        "units": "radians",
        "recording_frequency_hz": float(rate_hz),
        "sample_count": int(len(splice.time)),
        "arm_joint_names": list(ARM_JOINTS),
        "hand_joint_names": list(HAND_JOINTS),
        "release_phase": original_metadata.get("release_phase", "follow_waypoints"),
        "cycle_index": cycle_index,
        "source": {
            "original_trajectory": str(original.source),
            "original_home": str(home_path),
            "intervention_session": str(session),
            "intervention_index": int(intervention["index"]),
        },
        "splice": {
            "paused_sample": int(splice.paused_sample),
            "original_rejoin_sample": int(splice.original_rejoin_sample),
            "composite_rejoin_sample": int(splice.composite_rejoin_sample),
            "composite_suffix_sample": int(splice.composite_suffix_sample),
            "captured_waypoints": len(waypoints),
            "composite_waypoint_samples": [int(value) for value in splice.waypoint_samples],
            "dwell_s": float(dwell_s),
            "peak_joint_speed_rad_s": float(peak_speed),
            "profile": "quintic, zero velocity and acceleration at each marked pose",
            "continuous_human_path_recorded": False,
        },
        "hardware_orientation": original_metadata.get("hardware_orientation", {}),
    }
    # Validate every source-derived field before creating the directory. A
    # refused splice must not leave something that looks like an artifact.
    output.mkdir(parents=True)
    arrays = {
        "joint_pos_arm": splice.arm,
        "joint_pos_hand": splice.hand,
        "arm_joint_names": np.asarray(ARM_JOINTS),
        "hand_joint_names": np.asarray(HAND_JOINTS),
        "sample_time_s": splice.time,
    }
    _atomic_npz(output / "replay_data.npz", arrays)
    _atomic_json(output / "metadata.json", metadata)
    _atomic_text(output / "homing.yaml", home_path.read_text(encoding="utf-8"))
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="logs/dagger/<session> directory")
    parser.add_argument(
        "-o", "--output", default=None,
        help="output artifact directory (default: <session>/composite)",
    )
    parser.add_argument(
        "--intervention", type=int, default=None,
        help="intervention index when a session contains more than one",
    )
    parser.add_argument(
        "--dwell", type=float, default=DEFAULT_DWELL,
        help=f"seconds held at each captured pose and handback (default: {DEFAULT_DWELL:g})",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help=f"replacement speed scale (1.0 = {DEFAULT_PEAK_SPEED:g} rad/s peak)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.dwell < 0:
        parser.error("--dwell may not be negative")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")

    session = Path(args.session).expanduser().resolve()
    try:
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        events = load_events(session / "events.jsonl")
        record = _selected_intervention(manifest, args.intervention)
        index = int(record["index"])
        points = _intervention_waypoints(session, events, index)
        rollout = next((event for event in events if event.get("event") == "rollout_start"), None)
        if rollout is None:
            raise ValueError("the session has no rollout_start event naming the original artifact")
        source_path = _existing_path(str(rollout["trajectory"]), "original trajectory")
        home_path = _existing_path(str(rollout["home"]), "original home")
        original = load_trajectory(str(source_path))
        metadata_path = original.source.parent / "metadata.json"
        original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rate_hz = _uniform_rate(original)
        splice = build_splice(
            original,
            int(record["paused_sample"]),
            int(record["rejoin_sample"]),
            points,
            rate_hz,
            dwell_s=float(args.dwell),
            peak_speed=DEFAULT_PEAK_SPEED * float(args.speed),
        )
        output = Path(args.output).expanduser() if args.output else session / "composite"
        write_splice(
            output, splice, original, original_metadata, home_path, session, record, points,
            rate_hz, float(args.dwell), DEFAULT_PEAK_SPEED * float(args.speed),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"original samples 0-{splice.paused_sample}")
    print(
        f"replacement through {len(points)} captured waypoints: composite samples "
        f"{splice.paused_sample + 1}-{splice.composite_suffix_sample - 1}"
    )
    print(
        f"release phase starts at composite sample {splice.composite_rejoin_sample}; "
        f"original sample {splice.original_rejoin_sample + 1} resumes at composite "
        f"sample {splice.composite_suffix_sample}"
    )
    print(f"{len(splice.time)} samples over {splice.time[-1]:.2f} s -> {output}")
    print("WARNING: marked poses were recorded, not the continuous human path; the middle is a")
    print("minimum-jerk point-to-point approximation through those marks.")
    playback_scale = float(rollout.get("time_scale", 1.0))
    duration_guard = int(
        100.0 * math.ceil((float(splice.time[-1]) * playback_scale + 5.0) / 100.0)
    )
    print("\nValidate before motion:")
    print(
        f"  ros2 run inspire_franka_trajectory_replay replay_trajectory {output} \\\n"
        f"    --home {output / 'homing.yaml'} --time-scale {playback_scale:g} \\\n"
        f"    --max-prepared-duration {duration_guard} --dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
