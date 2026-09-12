#!/usr/bin/env python3
"""Convert a ForgeUltra dual-supervision episode into the rollout recording schema.

This lets ``run_policy_rollout.py evaluate --reference`` compare a MuJoCo
student rollout against the recorded teacher episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.data_collection import RolloutDataCollector  # noqa: E402

DEFAULT_EPISODE = APP_ROOT / "checkpoints" / "reference_episode" / "episode_000_sequential_threading.npz"


def convert(episode_path: Path, output_dir: Path, duration_s: float = 64.6) -> Path:
    ep = np.load(episode_path, allow_pickle=False)
    proprio = ep["student_osc_proprio"].astype(np.float64)
    unified = np.clip(ep["unified_osc_action"].astype(np.float64), -1.0, 1.0)
    filtered = ep["osc_filtered_action"].astype(np.float64)
    times = ep["sample_time_s"].astype(np.float64)
    phases = ep["replay_phase"].astype(str)
    cycles = ep["cycle"].astype(int)
    collector = RolloutDataCollector(
        output_dir,
        metadata={
            "source_episode": str(episode_path),
            "collection_mode": "forgeultra_teacher_episode",
            "note": "teacher-labelled reference converted to the rollout schema",
        },
    )
    for t in range(len(times)):
        collector.record(
            sample_time_s=float(times[t]),
            joint_position=proprio[t, :10],
            joint_velocity=proprio[t, 10:20],
            proprio=proprio[t],
            policy_action=unified[t],
            filtered_native_action=filtered[t],
            clipped_elements=0,
            trajectory_progress=min(1.0, float(times[t]) / duration_s),
            process_phase=str(phases[t]),
            task_signals={
                "pickup_success": True,
                "threading_entered": True,
                "completed_cycles": int(cycles[t]) - 1,
                "terminated": False,
                "truncated": False,
                "watchdog_stop": False,
            },
        )
    return collector.close().data_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default=str(DEFAULT_EPISODE))
    parser.add_argument("--output-dir", default=str(DEFAULT_EPISODE.parent / "reference_rollout"))
    args = parser.parse_args(argv)
    path = convert(Path(args.episode), Path(args.output_dir))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
