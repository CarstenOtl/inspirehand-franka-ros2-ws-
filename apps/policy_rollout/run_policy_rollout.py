#!/usr/bin/env python3
"""Run the distilled DP3 policy on either hardware or MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parents[1]
DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "checkpoints/sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt"
)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _hardware_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_policy_rollout.py hardware",
        description="Run the distilled policy on the physical FR3 and Inspire RH56.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument(
        "--config",
        default=str(
            WORKSPACE_ROOT
            / "src/inspire_franka_trajectory_replay/config/replay.yaml"
        ),
    )
    parser.add_argument(
        "--home",
        default=str(
            WORKSPACE_ROOT / "apps/traj_replay/demo_trajs/traj_2/homing.yaml"
        ),
    )
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=2,
        help="Flow ODE steps per action (hardware default: 2 for 15 Hz CPU execution).",
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--recording-root", default="logs/policy_rollout")
    parser.add_argument("--record-rgbd", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a side-by-side viewer for the policy RGB and aligned-depth topics.",
    )
    parser.add_argument(
        "--viewer-depth-max",
        type=float,
        default=2.0,
        metavar="METRES",
        help="Maximum depth shown by --viewer (default: 2.0 m).",
    )
    parser.add_argument(
        "--viewer-hz",
        type=float,
        default=5.0,
        metavar="HZ",
        help="Matplotlib refresh rate used by --viewer (default: 5 Hz).",
    )
    parser.add_argument("--hand-topic", default="/inspire_hand/command")
    parser.add_argument("--hand-state-topic", default="/inspire_hand/joint_states")
    parser.add_argument("--hand-timeout", type=float, default=20.0)
    parser.add_argument("--hand-tolerance", type=float, default=0.08)
    parser.add_argument("--input-timeout", type=float, default=15.0)
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--max-frame-skew", type=float, default=0.04)
    parser.add_argument("--max-home-delta", type=float, default=0.01)
    return parser


def _start_camera_viewer(calibration, depth_max: float, viewer_hz: float):
    viewer = (
        WORKSPACE_ROOT
        / "apps/camera_calibration/tests/test_camera.py"
    )
    command = [
        sys.executable,
        str(viewer),
        "--no-launch",
        "--color-topic",
        calibration.color_topic,
        "--depth-topic",
        calibration.depth_topic,
        "--depth-max",
        str(depth_max),
        "--viewer-hz",
        str(viewer_hz),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    time.sleep(1.0)
    if process.poll() is not None:
        raise RuntimeError(
            f"policy camera viewer exited during startup with code {process.returncode}"
        )
    return process


def _stop_camera_viewer(process) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=3.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


def _warm_up_runner(runner, calibration, seed: int) -> float:
    """Populate CPU kernels/caches before the real-time controller is active."""

    height, width = calibration.policy_shape
    phase = None
    if runner.config.cyclic_process_phase_conditioning:
        phase = np.zeros(
            len(runner.config.cyclic_process_phase_features), dtype=np.float32
        )
        phase[0] = 1.0
    progress = 0.0 if runner.config.trajectory_progress_conditioning else None
    runner.reset(seed=seed)
    started = time.perf_counter()
    runner.step(
        proprio=np.zeros(runner.config.proprio_dim, dtype=np.float32),
        head_rgb=np.zeros((3, height, width), dtype=np.float32),
        head_depth=np.full((1, height, width), 0.8, dtype=np.float32),
        valid_mask=np.ones((1, height, width), dtype=bool),
        trajectory_progress=progress,
        cyclic_process_phase=phase,
    )
    elapsed = time.perf_counter() - started
    runner.reset(seed=seed)
    return elapsed


def _top_help() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="backend")
    commands.add_parser(
        "hardware", add_help=False, help="physical FR3 + RH56 + RealSense"
    )
    commands.add_parser(
        "mujoco", add_help=False, help="closed-loop MuJoCo simulation"
    )
    return parser


def _run_hardware(argv) -> int:
    args = _hardware_parser().parse_args(argv)
    for name in (
        "rate",
        "hand_timeout",
        "input_timeout",
        "max_state_age",
        "max_frame_skew",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.cycles < 1 or args.max_steps < 1 or args.integration_steps < 1:
        raise ValueError("--cycles, --max-steps, and --integration-steps must be positive")
    if args.hand_tolerance < 0 or args.max_home_delta < 0:
        raise ValueError("hand tolerance and max home delta must be non-negative")
    if args.viewer_depth_max <= 0 or args.viewer_hz <= 0:
        raise ValueError("--viewer-depth-max and --viewer-hz must be positive")

    from policy_rollout.hardware import assess_hardware_readiness, run_hardware_rollout
    from utils.camera_calibration import load_camera_calibration

    calibration = load_camera_calibration(args.camera_calibration)
    readiness = assess_hardware_readiness(calibration)
    if not readiness.ready:
        details = "\n".join(f"- {item}" for item in readiness.blockers)
        raise RuntimeError(
            "physical policy execution is disabled by readiness checks:\n" + details
        )

    from policy_rollout.flow_policy import FlowPolicyRunner

    runner = FlowPolicyRunner(
        args.checkpoint,
        device=args.device,
        integration_steps=args.integration_steps,
    )
    calibration.assert_checkpoint_compatible(runner.config.to_dict())
    warm_up_s = _warm_up_runner(runner, calibration, args.seed)
    print(
        f"policy model warm-up complete in {warm_up_s:.3f} s "
        f"({args.integration_steps} flow steps)"
    )
    viewer_process = None
    try:
        if args.viewer:
            viewer_process = _start_camera_viewer(
                calibration, args.viewer_depth_max, args.viewer_hz
            )
        report = run_hardware_rollout(runner, calibration, args)
    finally:
        _stop_camera_viewer(viewer_process)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"step_budget", "completed"} else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _top_help().print_help()
        return 0
    backend, backend_argv = argv[0], argv[1:]
    try:
        if backend == "hardware":
            return _run_hardware(backend_argv)
        if backend == "mujoco":
            from utils.mujoco_student_rollout import main as run_mujoco

            return run_mujoco(backend_argv)
        _top_help().error(f"unknown backend {backend!r}; choose hardware or mujoco")
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"policy rollout error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
