#!/usr/bin/env python3
"""Benchmark the deployed student checkpoint without ROS or robot commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "checkpoints/sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt"
)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(step < 1 for step in args.steps):
        parser.error("steps and repeats must be positive")

    import torch

    from policy_rollout.flow_policy import FlowPolicyRunner

    runner = FlowPolicyRunner(args.checkpoint, device=args.device, integration_steps=1)
    config = runner.config
    rgb = np.zeros((3, 180, 320), dtype=np.float32)
    depth = np.full((1, 180, 320), 0.8, dtype=np.float32)
    valid = np.ones((1, 180, 320), dtype=bool)
    proprio = np.zeros(config.proprio_dim, dtype=np.float32)
    phase = None
    if config.cyclic_process_phase_conditioning:
        phase = np.zeros(len(config.cyclic_process_phase_features), dtype=np.float32)
        phase[0] = 1.0
    progress = 0.0 if config.trajectory_progress_conditioning else None

    def infer():
        runner.reset(seed=0)
        return runner.step(
            proprio=proprio,
            head_rgb=rgb,
            head_depth=depth,
            valid_mask=valid,
            trajectory_progress=progress,
            cyclic_process_phase=phase,
        )

    infer()  # one-time operator/library warm-up
    results = []
    actions = {}
    for integration_steps in args.steps:
        runner.integration_steps = integration_steps
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            action = infer()
            durations.append(time.perf_counter() - started)
            if not bool(torch.isfinite(action).all().item()):
                raise RuntimeError("student returned a non-finite action")
        actions[integration_steps] = action.numpy()
        results.append(
            {
                "integration_steps": integration_steps,
                "mean_s": statistics.mean(durations),
                "min_s": min(durations),
                "max_s": max(durations),
                "mean_hz": 1.0 / statistics.mean(durations),
            }
        )
    reference_steps = max(args.steps)
    reference = actions[reference_steps]
    for result in results:
        difference = actions[result["integration_steps"]] - reference
        result["action_l2_vs_max_steps"] = float(np.linalg.norm(difference))
        result["action_max_abs_vs_max_steps"] = float(np.max(np.abs(difference)))
    print(
        json.dumps(
            {
                "device": str(runner.device),
                "torch_threads": torch.get_num_threads(),
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
