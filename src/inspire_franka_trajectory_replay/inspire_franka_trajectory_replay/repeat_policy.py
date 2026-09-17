"""Repeat one policy turn with a scripted release, retreat, and reset.

This builds an ordinary coordinated replay artifact; it never sends ROS
commands.  Each generated cycle is:

1. the selected continuous policy recording;
2. a cubic-Hermite path through the ordered scripted waypoints; and
3. a cubic-Hermite return to the policy's matching homing pose.

The output must still pass ``replay_trajectory --dry-run`` before hardware use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Optional, Sequence

import numpy as np
import yaml

from franka_trajectory_replay import limits
from inspire_hand_driver import kinematics as hand_kinematics

from .make_cycles import load_home
from .release_phase import RELEASE_PHASE
from .trajectory import (
    ARM_JOINTS,
    FORGE_HAND_JOINTS,
    HAND_JOINTS,
    CoordinatedTrajectory,
    load_trajectory,
    resolve_trajectory,
)


DEFAULT_WAYPOINTS = Path(
    "apps/traj_replay/scripted_waypoints/"
    "waypoints_release_and_reset_franka_20260910_current.yaml"
)
FORGE_HAND_BY_DRIVER = {
    "pinky_proximal_joint": "little_joint_0",
    "ring_proximal_joint": "ring_joint_0",
    "middle_proximal_joint": "middle_joint_0",
    "index_proximal_joint": "index_joint_0",
    "thumb_proximal_pitch_joint": "thumb_joint_1",
    "thumb_proximal_yaw_joint": "thumb_joint_0",
}


@dataclass(frozen=True)
class ScriptedRecipe:
    source: Path
    names: tuple[str, ...]
    arm: np.ndarray
    hand: np.ndarray
    waypoint_duration_s: float
    return_duration_s: float
    sha256: str


@dataclass(frozen=True)
class ReleaseSelection:
    sample: int
    method: str
    source_turn_progress_rad: Optional[float] = None
    reference: Optional[Path] = None
    reference_cycle: Optional[int] = None
    reference_handoff_sample: Optional[int] = None
    reference_turn_progress_rad: Optional[float] = None
    arm_max_delta_rad: Optional[float] = None
    hand_max_delta_rad: Optional[float] = None
    match_guard_rad: Optional[float] = None


@dataclass(frozen=True)
class RepeatedTrajectory:
    time: np.ndarray
    arm: np.ndarray
    hand: np.ndarray
    rate_hz: float
    cycle_index: tuple[dict, ...]
    phase_index: tuple[dict, ...]
    policy_samples: int
    cycle_samples: int
    policy_start_home_delta: float
    policy_end_waypoint_delta: float
    seam_delta: float
    joint5_cap_report: dict
    waypoint_duration_s: float
    return_duration_s: float


def _forge_names(source: Path) -> tuple[str, ...]:
    metadata_path = source.parent / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Forge trajectory has no metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return tuple(str(name) for name in metadata.get("joint_names", ()))


def _forge_pose_columns(source: Path) -> tuple[list[int], list[int]]:
    names = _forge_names(source)
    missing = [name for name in (*ARM_JOINTS, *FORGE_HAND_JOINTS) if name not in names]
    if missing:
        raise ValueError(f"Forge trajectory metadata is missing joints: {missing}")
    return (
        [names.index(name) for name in ARM_JOINTS],
        [names.index(name) for name in FORGE_HAND_JOINTS],
    )


def _turn_progress_for_pose(
    trajectory: CoordinatedTrajectory, sample: int, environment: int
) -> Optional[float]:
    """Find an informational turn-progress value for a loaded Forge pose."""
    source = trajectory.source
    with np.load(source, allow_pickle=False) as data:
        if "joint_pos" not in data or "threading_turn_progress_rad" not in data:
            return None
        positions = np.asarray(data["joint_pos"], dtype=float)
        if positions.ndim != 3 or not 0 <= environment < positions.shape[1]:
            return None
        arm_columns, hand_columns = _forge_pose_columns(source)
        pose = positions[:, environment]
        error = np.maximum(
            np.max(np.abs(pose[:, arm_columns] - trajectory.arm[sample]), axis=1),
            np.max(np.abs(pose[:, hand_columns] - trajectory.hand[sample]), axis=1),
        )
        raw_sample = int(np.argmin(error))
        progress = np.asarray(data["threading_turn_progress_rad"], dtype=float)
        if progress.ndim == 2:
            return float(progress[raw_sample, environment])
        if progress.ndim == 1:
            return float(progress[raw_sample])
    return None


def select_release_sample(
    policy: CoordinatedTrajectory,
    environment: int,
    sample: Optional[int] = None,
    reference: Optional[Path] = None,
    reference_cycle: int = 1,
    max_match_delta: float = 0.15,
) -> ReleaseSelection:
    """Choose an inclusive policy cutoff explicitly or from a hybrid handoff."""
    if policy.hand is None:
        raise ValueError("the policy trajectory has no Inspire hand positions")
    if (sample is None) == (reference is None):
        raise ValueError("choose exactly one of a release sample or release reference")
    if sample is not None:
        if not 1 <= sample < len(policy.arm):
            raise ValueError(
                f"release sample must be within [1, {len(policy.arm) - 1}], got {sample}"
            )
        return ReleaseSelection(
            sample=sample,
            method="explicit_sample",
            source_turn_progress_rad=_turn_progress_for_pose(policy, sample, environment),
        )

    if not np.isfinite(max_match_delta) or max_match_delta <= 0.0:
        raise ValueError("release match guard must be finite and positive")
    reference_path = Path(reference).expanduser().resolve()
    source = resolve_trajectory(str(reference_path))
    metadata_path = source.parent / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    with np.load(source, allow_pickle=False) as data:
        raw_phased = "cycle" in data and "replay_phase" in data
        if raw_phased:
            reference_trajectory = load_trajectory(
                str(reference_path), environment=environment, cycle=reference_cycle
            )
            cycles = np.asarray(data["cycle"])
            phases = np.asarray(data["replay_phase"]).astype(str)
            rows = np.flatnonzero(cycles == reference_cycle)
            if len(rows) != len(reference_trajectory.arm):
                raise ValueError(
                    "release reference cycle contains an unexpected reset segment"
                )
            local_policy = np.flatnonzero(phases[rows] == "policy")
            if not len(local_policy):
                raise ValueError(f"reference cycle {reference_cycle} has no policy phase")
            handoff_index = int(local_policy[-1])
            if (
                handoff_index + 1 >= len(rows)
                or phases[rows[handoff_index + 1]] == "policy"
            ):
                raise ValueError(
                    f"reference cycle {reference_cycle} has no policy-to-scripted handoff"
                )
            reference_handoff_sample = int(rows[handoff_index])
        else:
            reference_trajectory = load_trajectory(
                str(reference_path), environment=environment
            )
            entries = metadata.get("cycle_index") or []
            matching = [
                entry for entry in entries if int(entry.get("cycle", -1)) == reference_cycle
            ]
            if len(matching) != 1 or "release_sample" not in matching[0]:
                raise ValueError(
                    "the release reference needs raw cycle/replay_phase fields or "
                    "exactly one metadata cycle_index release flag"
                )
            release_sample = int(matching[0]["release_sample"])
            handoff_index = release_sample - 1
            if not 0 <= handoff_index < len(reference_trajectory.arm):
                raise ValueError("release reference metadata points outside its trajectory")
            reference_handoff_sample = handoff_index

        progress = None
        if raw_phased and "threading_turn_progress_rad" in data:
            values = np.asarray(data["threading_turn_progress_rad"], dtype=float)
            progress = float(
                values[reference_handoff_sample, environment]
                if values.ndim == 2
                else values[reference_handoff_sample]
            )

    if reference_trajectory.hand is None:
        raise ValueError("the release reference has no Inspire hand positions")

    arm_delta = np.max(
        np.abs(policy.arm - reference_trajectory.arm[handoff_index]), axis=1
    )
    hand_delta = np.max(
        np.abs(policy.hand - reference_trajectory.hand[handoff_index]), axis=1
    )
    combined_delta = np.maximum(arm_delta, hand_delta)
    matched = int(np.argmin(combined_delta))
    if combined_delta[matched] > max_match_delta:
        raise ValueError(
            f"nearest release-reference pose is {combined_delta[matched]:.4f} rad "
            f"away, over the {max_match_delta:.4f} rad match guard"
        )
    return ReleaseSelection(
        sample=matched,
        method="nearest_hybrid_policy_handoff",
        source_turn_progress_rad=_turn_progress_for_pose(policy, matched, environment),
        reference=source,
        reference_cycle=reference_cycle,
        reference_handoff_sample=reference_handoff_sample,
        reference_turn_progress_rad=progress,
        arm_max_delta_rad=float(arm_delta[matched]),
        hand_max_delta_rad=float(hand_delta[matched]),
        match_guard_rad=float(max_match_delta),
    )


def _uniform_rate(trajectory: CoordinatedTrajectory) -> float:
    spacing = np.diff(trajectory.time)
    dt = float(np.median(spacing))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("the policy trajectory has no finite positive sample period")
    if not np.allclose(spacing, dt, rtol=1e-5, atol=1e-9):
        raise ValueError("the policy trajectory must be uniformly sampled")
    return 1.0 / dt


def _ordered_waypoints(document: dict) -> list[tuple[str, dict]]:
    waypoint_map = document.get("waypoints")
    if not isinstance(waypoint_map, dict) or not waypoint_map:
        raise ValueError("the scripted waypoint YAML contains no waypoints")
    expected = [f"waypoint{index}" for index in range(1, len(waypoint_map) + 1)]
    if list(waypoint_map) != expected:
        raise ValueError(
            f"scripted waypoints must be ordered consecutively as {expected}; "
            f"found {list(waypoint_map)}"
        )
    return [(name, waypoint_map[name]) for name in expected]


def _commanded_pose(waypoint: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    robot = waypoint.get("robot_dofs") or {}
    names = [str(value) for value in robot.get("names", ())]
    positions = np.asarray(robot.get("position", ()), dtype=float)
    if len(names) != len(positions):
        raise ValueError(f"{label} robot_dofs names and positions differ in length")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} robot_dofs contains duplicate names")
    by_name = dict(zip(names, positions))
    missing_arm = [name for name in ARM_JOINTS if name not in by_name]
    missing_hand = [
        source for source in FORGE_HAND_BY_DRIVER.values() if source not in by_name
    ]
    if missing_arm or missing_hand:
        raise ValueError(
            f"{label} is missing commandable joints: {missing_arm + missing_hand}"
        )
    arm = np.asarray([by_name[name] for name in ARM_JOINTS], dtype=float)
    hand = np.asarray(
        [by_name[FORGE_HAND_BY_DRIVER[name]] for name in HAND_JOINTS], dtype=float
    )
    if not np.all(np.isfinite(arm)) or not np.all(np.isfinite(hand)):
        raise ValueError(f"{label} contains non-finite joint positions")
    return arm, hand


def load_recipe(path: Path) -> ScriptedRecipe:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    document = yaml.safe_load(raw) or {}
    if document.get("trajectory_space") != "robot_joint_position_pd":
        raise ValueError(
            "the repeated V2 trajectory requires robot_joint_position_pd waypoints; "
            f"{source} declares {document.get('trajectory_space')!r}"
        )
    ordered = _ordered_waypoints(document)
    poses = [_commanded_pose(value, name) for name, value in ordered]
    settings = document.get("settings") or {}
    waypoint_duration = float(settings.get("waypoint_duration_s", 0.7))
    return_duration = float(settings.get("return_duration_s", 1.0))
    if waypoint_duration <= 0.0 or return_duration <= 0.0:
        raise ValueError("scripted waypoint and return durations must be positive")
    return ScriptedRecipe(
        source=source,
        names=tuple(name for name, _ in ordered),
        arm=np.asarray([pose[0] for pose in poses]),
        hand=np.asarray([pose[1] for pose in poses]),
        waypoint_duration_s=waypoint_duration,
        return_duration_s=return_duration,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    hand_dofs = [hand_kinematics.DOFS[hand_kinematics.dof_index(name)] for name in HAND_JOINTS]
    lower = np.concatenate((limits.POSITION_LOWER, [dof.lower for dof in hand_dofs]))
    upper = np.concatenate((limits.POSITION_UPPER, [dof.upper for dof in hand_dofs]))
    return lower, upper


def _limit_safe_knot_velocities(
    knots: np.ndarray, durations: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Forge's centered C1 tangents, zeroed where they would overshoot limits."""
    velocities = np.zeros_like(knots)
    velocities[1:-1] = (knots[2:] - knots[:-2]) / (
        durations[:-1, None] + durations[1:, None]
    )
    fraction = np.linspace(0.0, 1.0, 129)[:, None]
    f2 = fraction * fraction
    f3 = f2 * fraction
    h00 = 2.0 * f3 - 3.0 * f2 + 1.0
    h10 = f3 - 2.0 * f2 + fraction
    h01 = -2.0 * f3 + 3.0 * f2
    h11 = f3 - f2
    for _ in range(3):
        changed = False
        for index, duration in enumerate(durations):
            samples = (
                h00 * knots[index]
                + h10 * duration * velocities[index]
                + h01 * knots[index + 1]
                + h11 * duration * velocities[index + 1]
            )
            unsafe = ((samples < lower - 1e-6) | (samples > upper + 1e-6)).any(axis=0)
            moving = unsafe & (
                (velocities[index] != 0.0) | (velocities[index + 1] != 0.0)
            )
            if np.any(moving):
                velocities[index : index + 2, moving] = 0.0
                changed = True
        if not changed:
            break
    return velocities


def _scripted_path(
    start: np.ndarray,
    waypoints: np.ndarray,
    home: np.ndarray,
    rate_hz: float,
    waypoint_duration_s: float,
    return_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    knots = np.vstack((start, waypoints, home))
    requested = [waypoint_duration_s] * len(waypoints) + [return_duration_s]
    # Avoid turning an exact 15-sample interval into 16 because 1 / median(dt)
    # can be 15.000000000000004 in binary floating point.
    steps = np.asarray(
        [max(2, math.ceil(value * rate_hz - 1e-9)) for value in requested]
    )
    durations = steps / float(rate_hz)
    lower, upper = _joint_bounds()
    if np.any(knots < lower - 1e-6) or np.any(knots > upper + 1e-6):
        raise ValueError("a scripted waypoint lies outside an arm or hand joint limit")
    velocities = _limit_safe_knot_velocities(knots, durations, lower, upper)
    rows = [knots[0].copy()]
    arrivals = [0]
    for index, (count, duration) in enumerate(zip(steps, durations)):
        fraction = np.arange(1, count + 1, dtype=float)[:, None] / float(count)
        f2 = fraction * fraction
        f3 = f2 * fraction
        rows.extend(
            2.0 * f3 * knots[index]
            - 3.0 * f2 * knots[index]
            + knots[index]
            + (f3 - 2.0 * f2 + fraction) * duration * velocities[index]
            + (-2.0 * f3 + 3.0 * f2) * knots[index + 1]
            + (f3 - f2) * duration * velocities[index + 1]
        )
        arrivals.append(len(rows) - 1)
    return np.asarray(rows), np.asarray(arrivals, dtype=np.int64)


def build(
    policy: CoordinatedTrajectory,
    home_arm: np.ndarray,
    home_hand: np.ndarray,
    recipe: ScriptedRecipe,
    cycles: int,
    release_sample: int,
    max_home_delta: float = 0.05,
    joint5_cap: Optional[float] = None,
    waypoint_duration_s: Optional[float] = None,
    return_duration_s: Optional[float] = None,
) -> RepeatedTrajectory:
    if cycles < 1:
        raise ValueError("cycles must be at least one")
    if policy.hand is None:
        raise ValueError("the policy trajectory has no Inspire hand positions")
    rate_hz = _uniform_rate(policy)
    if not 1 <= release_sample < len(policy.arm):
        raise ValueError(
            f"release sample must be within [1, {len(policy.arm) - 1}], "
            f"got {release_sample}"
        )
    arm = np.array(policy.arm[: release_sample + 1], dtype=float, copy=True)
    hand = np.array(policy.hand[: release_sample + 1], dtype=float, copy=True)
    cap_report = {}
    if joint5_cap is not None:
        if not np.isfinite(joint5_cap):
            raise ValueError("joint5 cap must be finite")
        recorded = arm[:, 4].copy()
        arm[:, 4] = np.minimum(arm[:, 4], float(joint5_cap))
        changed = np.flatnonzero(arm[:, 4] != recorded)
        cap_report = {
            "joint5_cap_rad": float(joint5_cap),
            "joint5_recorded_max_rad": float(recorded.max()),
            "joint5_capped_samples_per_cycle": int(len(changed)),
            "joint5_maximum_change_rad": float(np.abs(arm[:, 4] - recorded).max()),
        }
    home_arm = np.asarray(home_arm, dtype=float)
    home_hand = np.asarray(home_hand, dtype=float)
    start_delta = float(
        max(np.max(np.abs(arm[0] - home_arm)), np.max(np.abs(hand[0] - home_hand)))
    )
    if start_delta > max_home_delta:
        raise ValueError(
            f"policy start is {start_delta:.3f} rad from its home, over the "
            f"{max_home_delta:.3f} rad seam guard"
        )
    duration_waypoint = (
        recipe.waypoint_duration_s
        if waypoint_duration_s is None
        else float(waypoint_duration_s)
    )
    duration_return = (
        recipe.return_duration_s
        if return_duration_s is None
        else float(return_duration_s)
    )
    if duration_waypoint <= 0.0 or duration_return <= 0.0:
        raise ValueError("waypoint and return durations must be positive")
    policy_block = np.hstack((arm, hand))
    waypoint_block = np.hstack((recipe.arm, recipe.hand))
    home = np.concatenate((home_arm, home_hand))
    scripted, arrivals = _scripted_path(
        policy_block[-1], waypoint_block, home, rate_hz,
        duration_waypoint, duration_return,
    )
    # scripted[0] is already the final policy sample.
    one_cycle = np.vstack((policy_block, scripted[1:]))
    repeated = np.vstack([one_cycle] * cycles)
    cycle_samples = len(one_cycle)
    release_offset = len(policy_block)
    mapped_arrivals = len(policy_block) - 1 + arrivals
    cycle_index = []
    phase_index = []
    for cycle in range(1, cycles + 1):
        offset = (cycle - 1) * cycle_samples
        cycle_index.append(
            {
                "cycle": cycle,
                "start_sample": offset,
                "end_sample": offset + cycle_samples - 1,
                "release_sample": offset + release_offset,
                "release_time_s": (offset + release_offset) / rate_hz,
            }
        )
        phase_index.append(
            {
                "cycle": cycle,
                "policy_start_sample": offset,
                "policy_end_sample": offset + len(policy_block) - 1,
                "release_start_sample": offset + release_offset,
                "scripted_waypoint_samples": {
                    name: offset + int(sample)
                    for name, sample in zip(recipe.names, mapped_arrivals[1:-1])
                },
                "return_start_sample": offset + int(mapped_arrivals[-2]) + 1,
                "reset_sample": offset + int(mapped_arrivals[-1]),
            }
        )
    end_waypoint_delta = float(np.max(np.abs(policy_block[-1] - waypoint_block[0])))
    seam_delta = float(np.max(np.abs(one_cycle[-1] - one_cycle[0])))
    return RepeatedTrajectory(
        time=np.arange(len(repeated), dtype=float) / rate_hz,
        arm=repeated[:, : len(ARM_JOINTS)],
        hand=repeated[:, len(ARM_JOINTS) :],
        rate_hz=rate_hz,
        cycle_index=tuple(cycle_index),
        phase_index=tuple(phase_index),
        policy_samples=len(policy_block),
        cycle_samples=cycle_samples,
        policy_start_home_delta=start_delta,
        policy_end_waypoint_delta=end_waypoint_delta,
        seam_delta=seam_delta,
        joint5_cap_report=cap_report,
        waypoint_duration_s=duration_waypoint,
        return_duration_s=duration_return,
    )


def write(
    output: Path,
    result: RepeatedTrajectory,
    policy: CoordinatedTrajectory,
    policy_metadata: dict,
    home_path: Path,
    recipe: ScriptedRecipe,
    cycles: int,
    release: ReleaseSelection,
) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    metadata = {
        "schema_version": 1,
        "data_file": "replay_data.npz",
        "generated_by": "inspire_franka_trajectory_replay.repeat_policy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"{cycles} repetitions of one policy rollout, each followed by the "
            "vendored scripted release/retreat waypoints and a return to home."
        ),
        "units": "radians",
        "recording_frequency_hz": result.rate_hz,
        "sample_count": len(result.time),
        "arm_joint_names": list(ARM_JOINTS),
        "hand_joint_names": list(HAND_JOINTS),
        "cycles": list(range(1, cycles + 1)),
        "policy_samples_per_cycle": result.policy_samples,
        "cycle_samples": result.cycle_samples,
        "release_phase": RELEASE_PHASE,
        "cycle_index": list(result.cycle_index),
        "phase_index": list(result.phase_index),
        "source": {
            "policy": str(policy.source),
            "policy_segment": policy.segment,
            "policy_hardware_replay_status": policy_metadata.get(
                "hardware_replay_status"
            ),
            "policy_hardware_waypoint_retargeting": policy_metadata.get(
                "hardware_waypoint_retargeting"
            ),
            "home": str(Path(home_path).resolve()),
            "scripted_waypoints": str(recipe.source),
            "scripted_waypoints_sha256": recipe.sha256,
        },
        "release_selection": {
            "inclusive_policy_sample": release.sample,
            "method": release.method,
            "source_turn_progress_rad": release.source_turn_progress_rad,
            "source_turn_progress_deg": (
                math.degrees(release.source_turn_progress_rad)
                if release.source_turn_progress_rad is not None
                else None
            ),
            "reference": str(release.reference) if release.reference else None,
            "reference_cycle": release.reference_cycle,
            "reference_handoff_sample": release.reference_handoff_sample,
            "reference_turn_progress_rad": release.reference_turn_progress_rad,
            "reference_arm_max_delta_rad": release.arm_max_delta_rad,
            "reference_hand_max_delta_rad": release.hand_max_delta_rad,
            "reference_match_guard_rad": release.match_guard_rad,
        },
        "composition": {
            "profile": "cubic Hermite with centered C1 internal knot velocities",
            "waypoint_duration_s": result.waypoint_duration_s,
            "return_duration_s": result.return_duration_s,
            "policy_start_home_delta_rad": result.policy_start_home_delta,
            "policy_end_to_waypoint1_max_delta_rad": result.policy_end_waypoint_delta,
            "cycle_seam_max_delta_rad": result.seam_delta,
        },
        "hardware_orientation": policy_metadata.get("hardware_orientation", {}),
        "hardware_replay_status": "generated_requires_dry_run_and_physical_validation",
        **result.joint5_cap_report,
    }
    output.mkdir(parents=True)
    np.savez(
        output / "replay_data.npz",
        joint_pos_arm=result.arm,
        joint_pos_hand=result.hand,
        arm_joint_names=np.asarray(ARM_JOINTS),
        hand_joint_names=np.asarray(HAND_JOINTS),
        sample_time_s=result.time,
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(home_path, output / "homing.yaml")
    shutil.copyfile(recipe.source, output / "scripted_waypoints.yaml")
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="single policy recording or NPZ")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--home", default=None, help="default: homing.yaml beside source")
    parser.add_argument("--waypoints", default=str(DEFAULT_WAYPOINTS))
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--max-home-delta", type=float, default=0.05)
    parser.add_argument("--joint5-cap", type=float, default=None)
    release_group = parser.add_mutually_exclusive_group(required=True)
    release_group.add_argument(
        "--release-sample",
        type=int,
        help="inclusive sample in the selected policy trajectory",
    )
    release_group.add_argument(
        "--release-reference",
        help="hybrid multi-cycle recording whose policy handoff selects the cutoff",
    )
    parser.add_argument("--reference-cycle", type=int, default=1)
    parser.add_argument("--max-release-match-delta", type=float, default=0.15)
    parser.add_argument("--waypoint-duration", type=float, default=None)
    parser.add_argument("--return-duration", type=float, default=None)
    args = parser.parse_args(argv)
    if args.cycles < 1:
        parser.error("--cycles must be at least one")
    if args.max_home_delta < 0.0:
        parser.error("--max-home-delta must not be negative")
    source = Path(args.trajectory).expanduser().resolve()
    source_dir = source if source.is_dir() else source.parent
    home_path = Path(args.home).expanduser().resolve() if args.home else source_dir / "homing.yaml"
    try:
        policy = load_trajectory(str(source), environment=args.env)
        home_arm, home_hand = load_home(home_path)
        recipe = load_recipe(Path(args.waypoints))
        release = select_release_sample(
            policy,
            environment=args.env,
            sample=args.release_sample,
            reference=(Path(args.release_reference) if args.release_reference else None),
            reference_cycle=args.reference_cycle,
            max_match_delta=args.max_release_match_delta,
        )
        metadata_path = policy.source.parent / "metadata.json"
        policy_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        result = build(
            policy, home_arm, home_hand, recipe, args.cycles, release.sample,
            max_home_delta=args.max_home_delta,
            joint5_cap=args.joint5_cap,
            waypoint_duration_s=args.waypoint_duration,
            return_duration_s=args.return_duration,
        )
        output = write(
            Path(args.output), result, policy, policy_metadata, home_path, recipe,
            args.cycles, release,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"policy:    {policy.source} ({result.policy_samples} samples per cycle)")
    progress = (
        f", {math.degrees(release.source_turn_progress_rad):.2f} deg turn"
        if release.source_turn_progress_rad is not None
        else ""
    )
    print(
        f"release:   policy sample {release.sample} inclusive ({release.method}{progress})"
    )
    if release.reference is not None:
        print(
            f"reference: {release.reference}, cycle {release.reference_cycle}, "
            f"handoff sample {release.reference_handoff_sample}; max delta "
            f"arm={release.arm_max_delta_rad:.4f}, hand={release.hand_max_delta_rad:.4f} rad"
        )
    print(f"waypoints: {recipe.source} ({', '.join(recipe.names)})")
    print(
        f"output:    {output} ({args.cycles} cycles, {len(result.time)} samples, "
        f"{result.time[-1]:.2f} s)"
    )
    print(
        f"seams: policy start/home {result.policy_start_home_delta:.6f} rad; "
        f"policy end/waypoint1 {result.policy_end_waypoint_delta:.3f} rad; "
        f"cycle end/start {result.seam_delta:.6f} rad"
    )
    print(
        "Validate before motion:\n"
        f"  ros2 run inspire_franka_trajectory_replay replay_trajectory {output} \\\n"
        f"    --home {output / 'homing.yaml'} --time-scale 5 \\\n"
        "    --max-prepared-duration 300 --dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
