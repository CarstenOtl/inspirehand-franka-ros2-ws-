#!/usr/bin/env python3
"""Compare the arm-torque candidates a force-observing rollout recorded.

    python3 apps/policy_rollout/utils/compare_torque_candidates.py \\
        logs/policy_rollout/hardware-<stamp> [more runs] \\
        [--reference episode_000_sequential_threading.npz]

Every hardware run records each candidate the policy could have seen
(``arm_torque_{external,measured,desired,commanded}_nm``; ros-sim has measured
and commanded), whichever one ``--arm-torque-source`` fed it. This prints them
in the policy's units (N m over the FR3 effort limits, clipped to 1), per joint
and per process phase, and how they correlate with each other. With
``--reference``, a training episode recorded with force columns
(``student_arm_joint_torque``, already normalized) is summarized the same way:
the candidate whose phase means and spreads sit closest to it is the hardware
signal that matches training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.data_collection import DATA_FILENAME, METADATA_FILENAME  # noqa: E402

FR3_ARM_EFFORT_LIMITS_NM = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
CANDIDATES = ("external", "measured", "desired", "commanded")
PHASES = ("policy", "follow_waypoints", "return_to_reset")
REFERENCE_KEY = "student_arm_joint_torque"


def normalize(torque_nm) -> np.ndarray:
    return np.clip(np.asarray(torque_nm, dtype=float) / FR3_ARM_EFFORT_LIMITS_NM, -1.0, 1.0)


def load_run(run_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Normalized candidates by name, the per-step process phase, and the metadata."""

    data = np.load(run_dir / "data" / DATA_FILENAME)
    metadata_path = run_dir / "data" / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    candidates = {
        name: normalize(data[f"arm_torque_{name}_nm"])
        for name in CANDIDATES
        if f"arm_torque_{name}_nm" in data.files
    }
    if not candidates:
        raise ValueError(f"{run_dir} recorded no arm torque candidates")
    return candidates, np.asarray(data["process_phase"]).astype(str), metadata


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if REFERENCE_KEY not in data.files:
        raise ValueError(f"{path} has no {REFERENCE_KEY}; it predates the force columns")
    phase_key = next((k for k in ("process_phase", "replay_phase") if k in data.files), None)
    phases = np.asarray(data[phase_key]).astype(str) if phase_key else np.full(len(data[REFERENCE_KEY]), "")
    return np.asarray(data[REFERENCE_KEY], dtype=float), phases


def phase_rows(torque: np.ndarray, phases: np.ndarray) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """``(phase, samples, mean, std)`` per joint for all steps and each process phase."""

    rows = [("all", len(torque), torque.mean(0), torque.std(0))]
    for phase in PHASES:
        mask = np.char.find(phases, phase) >= 0
        if mask.any():
            rows.append((phase, int(mask.sum()), torque[mask].mean(0), torque[mask].std(0)))
    return rows


def correlations(candidates: dict[str, np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    """Per-joint Pearson correlation for every candidate pair (NaN on a flat channel)."""

    names = list(candidates)
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            x, y = candidates[a], candidates[b]
            xc, yc = x - x.mean(0), y - y.mean(0)
            denominator = np.sqrt((xc**2).sum(0) * (yc**2).sum(0))
            with np.errstate(invalid="ignore", divide="ignore"):
                result[(a, b)] = np.where(denominator > 0, (xc * yc).sum(0) / denominator, np.nan)
    return result


def _vector(values) -> str:
    return " ".join(f"{v:+.3f}" for v in values)


def print_summary(label: str, rows) -> None:
    print(f"  {label}")
    for phase, count, mean, std in rows:
        print(f"    {phase:<17} n={count:<5} mean {_vector(mean)}")
        print(f"    {'':<17} {'':<7} std  {_vector(std)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, help="training episode with force columns")
    args = parser.parse_args(argv)

    print("normalized torque (N m / [87 x4, 12 x3], clipped to 1), joints 1..7")
    if args.reference:
        torque, phases = load_reference(args.reference)
        print(f"\nreference {args.reference.name}")
        print_summary("training (student_arm_joint_torque)", phase_rows(torque, phases))
    for run_dir in args.runs:
        if not (run_dir / "data" / DATA_FILENAME).is_file():
            report = run_dir / "report.json"
            reason = json.loads(report.read_text()).get("error") if report.is_file() else None
            print(f"\n{run_dir.name}: no recording" + (f" ({reason})" if reason else ""))
            continue
        candidates, phases, metadata = load_run(run_dir)
        source = metadata.get("arm_joint_torque_source")
        print(f"\n{run_dir.name}  (policy input: {source})")
        for name, torque in candidates.items():
            print_summary(name + (" <- policy" if name == source else ""), phase_rows(torque, phases))
        print("  correlation per joint")
        for (a, b), value in correlations(candidates).items():
            print(f"    {a:>9} vs {b:<9} {_vector(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
