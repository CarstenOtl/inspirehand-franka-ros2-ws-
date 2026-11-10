#!/usr/bin/env python3
"""Run the distilled vision policy on hardware, ros-sim or MuJoCo."""

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
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import DEFAULT_CHECKPOINT  # noqa: E402


def _hardware_parser(sim: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_policy_rollout.py " + ("ros-sim" if sim else "hardware"),
        description=(
            "Run the hardware rollout path against sim_policy.launch.py (MuJoCo "
            "behind the same ROS interfaces)."
            if sim
            else "Run the distilled policy on the physical FR3 and Inspire RH56."
        ),
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "torch device for the policy, vision encoder included: 'auto' picks "
            "CUDA when the torch build sees a GPU, else CPU (default: auto)."
        ),
    )
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument(
        "--config",
        default=str(
            WORKSPACE_ROOT
            / "src/inspire_franka_trajectory_replay/config/replay.yaml"
        ),
    )
    parser.add_argument(
        "--task-profile",
        default="auto",
        help=(
            "the training contract to reproduce (reset pose, orientation-action frame, "
            "hand model); 'auto' (default) follows the checkpoint "
            "(forge_osc.CHECKPOINT_TASK_PROFILES)"
        ),
    )
    parser.add_argument(
        "--home",
        default=None,
        help="homing YAML (default: the task profile's, see POLICY_HOME_CONFIGS)",
    )
    # ros-sim paces this rate on the simulator's /clock, so a slowed simulator
    # (sim_policy.launch.py sim_speed) keeps it exact on a shared CPU.
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=None,
        help=(
            "Flow ODE steps per action (default: the 16 the student was distilled "
            "with on CUDA, 2 on CPU so the 15 Hz period still fits)."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.5,
        help=(
            "weight of a chunk proposal per policy tick of age when overlapping chunks "
            "are blended (default 0.5, Isaac's evaluate_flow default; 0 executes only "
            "the newest chunk's first action)"
        ),
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="step budget (default: 4500 in ros-sim, as Isaac's evaluation of the students; 1500 on hardware)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prev-action-seed",
        choices=["pose", "zeros"],
        default="pose",
        help=(
            "cycle 1's previous-action input: 'pose' (default) encodes the live grasp "
            "frame and wrist angles like Isaac's reset, which is how training's episodes "
            "start; 'zeros' is the start used before 2026-10-26."
        ),
    )
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--recording-root", default="logs/policy_rollout")
    parser.set_defaults(sim=sim)
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
    parser.add_argument(
        "--fingertip-force-topic",
        default=None,
        help=(
            "pad forces for a force-observing student (default: /fingertip_force, "
            "inspire_franka_sim's projected channels in N, in ros-sim; the RH56 "
            "driver's /inspire_hand/grip_force in grams, zeroed at the release "
            "posture, on hardware)"
        ),
    )
    parser.add_argument(
        "--arm-torque-source",
        choices=["auto", "external", "measured", "desired", "commanded"],
        default="auto",
        help=(
            "which arm torque a force-observing student sees; every candidate the "
            "backend has is recorded either way (arm_torque_<name>_nm). 'external': "
            "libfranka's tau_ext_hat_filtered; 'measured': tau_J minus the model's "
            "gravity; 'desired': tau_J_d; 'commanded': the controller's tau_task + "
            "tau_nullspace. 'auto' (default) is external on hardware and commanded in "
            "ros-sim, which has measured and commanded only"
        ),
    )
    parser.add_argument(
        "--arm-torque-lowpass-s",
        type=float,
        default=0.02,
        help=(
            "first-order low-pass time constant on the broadcaster torques (default "
            "0.02 s; 0 off); the controller's measured_torque_lowpass_s filters tau_J"
        ),
    )
    parser.add_argument("--hand-timeout", type=float, default=20.0)
    parser.add_argument("--hand-tolerance", type=float, default=0.08)
    parser.add_argument("--input-timeout", type=float, default=15.0)
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--max-frame-skew", type=float, default=0.04)
    parser.add_argument("--max-home-delta", type=float, default=0.01)
    parser.add_argument(
        "--max-home-grasp-error",
        type=float,
        default=0.002,
        metavar="METRES",
        help=(
            "how far the live grasp frame may sit from the trained reset grasp after "
            "homing (default 2 mm; the return gate is 5 mm). Homing is retried "
            "--home-attempts times before the run is refused."
        ),
    )
    parser.add_argument("--home-attempts", type=int, default=3)
    parser.add_argument("--home-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--hand-start",
        choices=["auto", "release", "grasp"],
        default="auto",
        help=(
            "'release' opens thumb and index to training's release posture and commands "
            "the grasp just before the first policy sample, as a settled reset's first "
            "observation shows the hand closing; 'grasp' starts from the hand resting at "
            "the grasp, as a fast reset does. 'auto' (default) follows the task profile "
            "in ros-sim and is 'release' on hardware, which needs the open hand to tare "
            "the pad forces."
        ),
    )
    parser.add_argument(
        "--expected-load-mass-kg",
        type=float,
        default=None,
        help=(
            "refuse to run unless the mass Franka Desk configures beyond the flange "
            "(FrankaRobotState inertia_total: end effector plus load) is within 0.1 kg "
            "of this; without it the mass is only logged."
        ),
    )
    parser.add_argument(
        "--return-tolerance-mm",
        type=float,
        default=None,
        help=(
            "the grasp must come back within this distance of the trained reset grasp "
            "for a cycle to count (default: 10 in ros-sim, as Isaac's evaluation of the "
            "students; 5 on hardware, where the arm settled 8-10 mm short on 2026-11-03)"
        ),
    )
    parser.add_argument(
        "--return-tolerance-deg",
        type=float,
        default=None,
        help=(
            "the grasp must come back within this rotation of the trained reset grasp "
            "(default: 10 in ros-sim, 5 on hardware)"
        ),
    )
    parser.add_argument(
        "--rehome-between-cycles",
        action="store_true",
        help=(
            "after each return, except the last cycle's, stop the policy and put the arm "
            "on the reset pose with the joint-impedance homing, then start the next cycle "
            "from there. The return gate then only has to clear the nut, so "
            "--return-tolerance-mm/-deg default to 30 and 25."
        ),
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.15,
        metavar="RAD",
        help=(
            "stop the rollout when any arm joint comes this close to its position limit, "
            "before the FR3's own velocity reflex does (successful cycles stay >= 0.30 rad "
            "away; a stuck policy winds joints 5 and 7 toward their limits)"
        ),
    )
    parser.add_argument(
        "--hold-thumb-grip",
        action="store_true",
        help=(
            "diagnostic: hold the thumb flexion target at its peak while both pads are "
            "loaded, until the policy opens the index or 3 s pass, to see whether the "
            "nut turns once the grip lasts. Mimic-hand students only."
        ),
    )
    parser.add_argument(
        "--release-signal",
        choices=("auto", "nut-twist", "finger-sweep", "grasp-yaw"),
        default="auto",
        help=(
            "what fires the scripted release. auto: the simulated nut's twist in "
            "ros-sim, finger-sweep on hardware. finger-sweep: the end of a loaded "
            "finger sweep, how the deployed student turns the nut. grasp-yaw: the "
            "grasp-frame yaw proxy, for students that turn with the wrist. Giving a "
            "hardware signal in ros-sim rehearses it against the measured twist."
        ),
    )
    parser.add_argument(
        "--release-proxy-deg",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "with --release-signal grasp-yaw: the proxy reading that fires the release, "
            "instead of 55 deg. For a student whose wrist turns less than the nut."
        ),
    )
    parser.add_argument(
        "--min-fingertip-separation",
        type=float,
        default=0.030,
        metavar="METRES",
        help=(
            "the grasp-yaw proxy may not fire the release while the fingertips are "
            "closer than this (an empty, collapsed pinch; the nut is 36 mm across flats)"
        ),
    )
    parser.add_argument(
        "--max-proxy-step-deg",
        type=float,
        default=20.0,
        help="the proxy may not fire the release on a tick where it moved more than this",
    )
    parser.add_argument(
        "--turn-log-period",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help=(
            "print the nut turn, the grasp-yaw proxy, the release gate, fingertip "
            "separation and index joint every this many seconds (0 disables). The "
            "same values stream every tick on /fr3_policy_rollout/turn."
        ),
    )
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help=(
            "publish the policy's point cloud, control frames and camera-view overlay "
            "(/fr3_policy_rollout/dp3_overlay, for rqt_image_view). Its cost is "
            "inside the policy period, so the preflight gate measures it too."
        ),
    )
    parser.add_argument(
        "--debug-viz-every",
        type=int,
        default=2,
        help="publish every Nth policy step (default 2)",
    )
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
        "ros-sim",
        add_help=False,
        help="the hardware path against sim_policy.launch.py (MuJoCo behind ROS)",
    )
    commands.add_parser(
        "mujoco", add_help=False, help="closed-loop MuJoCo simulation"
    )
    return parser


def _resolve_execution(args):
    """Pin the policy device and fill the device-dependent step default."""

    from policy_rollout.flow_policy import default_integration_steps, resolve_device

    device = resolve_device(args.device)
    args.device = str(device)
    if args.integration_steps is None:
        args.integration_steps = default_integration_steps(device)
    return device


#: Each task profile's homing YAML: its reset arm joints and entry pinch.
POLICY_HOME_CONFIGS = {
    "mimic_m24": APP_ROOT / "config/policy_home.yaml",
    "mimic_windback45_tilt30": APP_ROOT / "config/policy_home_windback45_tilt30.yaml",
}


def resolve_backend_defaults(args, sim: bool) -> None:
    """Fill the limits that differ between ros-sim (Isaac's evaluation) and hardware."""

    if args.max_steps is None:
        args.max_steps = 4500 if sim else 1500
    rehome = getattr(args, "rehome_between_cycles", False)
    if args.return_tolerance_mm is None:
        args.return_tolerance_mm = 30.0 if rehome else 10.0 if sim else 5.0
    if args.return_tolerance_deg is None:
        args.return_tolerance_deg = 25.0 if rehome else 10.0 if sim else 5.0


def resolve_hand_start(choice: str, profile, sim: bool) -> str:
    """``--hand-start``: 'auto' starts as the student's reset did, in ros-sim."""

    if choice != "auto":
        return choice
    return "grasp" if sim and profile.fast_reset else "release"


def _run_hardware(argv, sim: bool = False) -> int:
    args = _hardware_parser(sim).parse_args(argv)
    _resolve_execution(args)
    from policy_rollout.teacher import EPISODE_STEPS, is_teacher_checkpoint

    teacher = is_teacher_checkpoint(args.checkpoint)
    if teacher:
        if not sim:
            raise ValueError(
                "this checkpoint is an RL teacher: it observes the nut pose, which the "
                "workcell does not measure. Run it with ros-sim."
            )
        if getattr(args, "debug_viz", False):
            raise ValueError("--debug-viz draws the student's point cloud; the teacher has no vision")
        # One training episode: it never learned to release or return.
        args.cycles = 1
        if args.max_steps is None:
            args.max_steps = EPISODE_STEPS
    resolve_backend_defaults(args, sim)
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
    if args.max_home_grasp_error < 0 or args.home_attempts < 1 or args.home_settle_s < 0:
        raise ValueError("--max-home-grasp-error, --home-attempts and --home-settle-s must be valid")
    if not 0.0 < args.return_tolerance_mm <= 30.0 or not 0.0 < args.return_tolerance_deg <= 25.0:
        raise ValueError("--return-tolerance-mm must be in (0, 30] and --return-tolerance-deg in (0, 25]")
    if not 0.0 < args.joint_limit_margin < 1.0:
        raise ValueError("--joint-limit-margin must be in (0, 1) rad")
    if args.release_signal == "nut-twist" and not sim:
        raise ValueError("--release-signal nut-twist needs ros-sim's simulated nut")
    if args.release_proxy_deg is not None and (
        args.release_signal != "grasp-yaw" or not 0.0 < args.release_proxy_deg <= 90.0
    ):
        raise ValueError("--release-proxy-deg needs --release-signal grasp-yaw and a value in (0, 90]")
    if args.min_fingertip_separation < 0 or args.max_proxy_step_deg <= 0:
        raise ValueError("--min-fingertip-separation and --max-proxy-step-deg must be valid")

    from policy_rollout import forge_osc as fo

    profile = fo.resolve_task_profile(args.task_profile, args.checkpoint)
    args.task_profile = profile.name
    if args.hold_thumb_grip and profile.hand != "mimic":
        raise ValueError("--hold-thumb-grip applies to mimic-hand students only")
    args.hand_start = resolve_hand_start(args.hand_start, profile, sim)
    if args.home is None:
        args.home = str(POLICY_HOME_CONFIGS[profile.name])
    print(f"task profile: {profile.name} ({profile.source}); home {args.home}", flush=True)

    from policy_rollout.hardware import assess_hardware_readiness, run_hardware_rollout
    from utils.camera_calibration import load_camera_calibration

    calibration = load_camera_calibration(args.camera_calibration)
    readiness = assess_hardware_readiness(calibration)
    if not readiness.ready:
        details = "\n".join(f"- {item}" for item in readiness.blockers)
        raise RuntimeError(
            "physical policy execution is disabled by readiness checks:\n" + details
        )

    if teacher:
        from policy_rollout.teacher import TeacherPolicyRunner

        runner = TeacherPolicyRunner(args.checkpoint, device=args.device)
        print(f"RL teacher loaded on {runner.device}: one turn, then the run ends", flush=True)
    else:
        from policy_rollout.flow_policy import FlowPolicyRunner

        runner = FlowPolicyRunner(
            args.checkpoint,
            device=args.device,
            integration_steps=args.integration_steps,
            temporal_ensemble_decay=args.temporal_ensemble_decay,
        )
        calibration.assert_checkpoint_compatible(runner.config.to_dict())
        warm_up_s = _warm_up_runner(runner, calibration, args.seed)
        print(
            f"policy model warm-up complete in {warm_up_s:.3f} s "
            f"({args.integration_steps} flow steps on {runner.device})"
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
        if backend == "ros-sim":
            return _run_hardware(backend_argv, sim=True)
        if backend == "mujoco":
            from utils.mujoco_student_rollout import main as run_mujoco

            return run_mujoco(backend_argv)
        _top_help().error(
            f"unknown backend {backend!r}; choose hardware, ros-sim or mujoco"
        )
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"policy rollout error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
