#!/usr/bin/env python3
"""Inspect, dry-run, record, plot, and evaluate the distilled DP3 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout.hardware import assess_hardware_readiness, run_hardware_placeholder
from utils.camera_calibration import load_camera_calibration


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("checkpoint", help="ForgeUltra offline-flow .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument(
        "--camera-calibration",
        default=None,
        help=(
            "camera profile YAML (default: "
            "utils/camera_calibration/fr3_realsense_dp3.yaml)"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="strictly load and describe a checkpoint"
    )
    _add_common(inspect)
    dry_run = commands.add_parser("dry-run", help="run one synthetic RGB-D policy step")
    _add_common(dry_run)
    dry_run.add_argument("--integration-steps", type=int, default=16)
    dry_run.add_argument(
        "--recording-dir",
        default=None,
        help="write the synthetic step using the rollout recording schema",
    )
    dry_run.add_argument(
        "--record-rgbd",
        action="store_true",
        help="include prepared policy RGB-D arrays in --recording-dir",
    )
    run = commands.add_parser(
        "run", help="physical entrypoint placeholder; always fails closed for now"
    )
    _add_common(run)
    camera = commands.add_parser("camera-check", help="show physical camera blockers")
    camera.add_argument("--camera-calibration", default=None)
    plot = commands.add_parser("plot", help="plot an existing rollout artifact")
    plot.add_argument("recording")
    plot.add_argument("--reference", default=None)
    plot.add_argument("--output-dir", default=None)
    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate an existing rollout and optionally compare a reference",
    )
    evaluate.add_argument("recording")
    evaluate.add_argument("--reference", default=None)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--required-cycles", type=int, default=6)
    evaluate.add_argument("--no-plots", action="store_true")
    return parser


def _runner(args):
    from policy_rollout.flow_policy import FlowPolicyRunner

    calibration = load_camera_calibration(args.camera_calibration)
    runner = FlowPolicyRunner(
        args.checkpoint,
        device=args.device,
        integration_steps=getattr(args, "integration_steps", 16),
    )
    calibration.assert_checkpoint_compatible(runner.config.to_dict())
    return runner, calibration


def _summary(runner, calibration) -> dict:
    metadata = runner.metadata()
    config = metadata["config"]
    return {
        "checkpoint": metadata["checkpoint"],
        "sha256": metadata["sha256"],
        "weight_source": metadata["weight_source"],
        "epoch": metadata["epoch"],
        "student": {
            "control_domain": config["student_control_domain"],
            "action_representation": config["osc_action_representation"],
            "action_dim": config["joint_dim"],
            "action_horizon": config["action_horizon"],
            "proprio_dim": config["proprio_dim"],
            "vision_encoder": config["vision_encoder_config"]["encoder"]["type"],
            "image_shape": config["vision_encoder_config"]["input"]["image_shape"],
            "trajectory_progress_conditioning": config[
                "trajectory_progress_conditioning"
            ],
            "cyclic_process_phase_conditioning": config[
                "cyclic_process_phase_conditioning"
            ],
        },
        "camera_profile": str(calibration.source_path),
        "physical_camera_ready": calibration.hardware_ready,
        "camera_blockers": list(calibration.hardware_blockers()),
    }


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plot":
            from utils.plotting import plot_rollout

            paths = plot_rollout(
                args.recording,
                reference=args.reference,
                output_dir=args.output_dir,
            )
            print(json.dumps({"plots": [str(path) for path in paths]}, indent=2))
            return 0
        if args.command == "evaluate":
            from utils.evaluation import evaluate_rollout

            report_path, report = evaluate_rollout(
                args.recording,
                reference=args.reference,
                output_dir=args.output_dir,
                required_cycles=args.required_cycles,
            )
            plots = []
            if not args.no_plots:
                from utils.plotting import plot_rollout

                plots = plot_rollout(
                    args.recording,
                    reference=args.reference,
                    output_dir=report_path.parent,
                )
            print(
                json.dumps(
                    {
                        "evaluation": str(report_path),
                        "task_outcome": report["metrics"]["task_outcome"],
                        "plots": [str(path) for path in plots],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "camera-check":
            calibration = load_camera_calibration(args.camera_calibration)
            readiness = assess_hardware_readiness(calibration)
            print(
                json.dumps(
                    {"ready": readiness.ready, "blockers": readiness.blockers}, indent=2
                )
            )
            return 0 if readiness.ready else 2

        runner, calibration = _runner(args)
        if args.command == "inspect":
            print(json.dumps(_summary(runner, calibration), indent=2))
            return 0
        if args.command == "run":
            run_hardware_placeholder(calibration)

        from policy_rollout.session import PolicyRolloutSession

        collector = None
        if args.record_rgbd and args.recording_dir is None:
            raise ValueError("--record-rgbd requires --recording-dir")
        if args.recording_dir is not None:
            from utils.data_collection import RolloutDataCollector

            metadata = runner.metadata()
            collector = RolloutDataCollector(
                args.recording_dir,
                metadata={
                    "checkpoint": metadata["checkpoint"],
                    "checkpoint_sha256": metadata["sha256"],
                    "checkpoint_weight_source": metadata["weight_source"],
                    "camera_profile": str(calibration.source_path),
                    "collection_mode": "synthetic_dry_run",
                },
                record_rgbd=args.record_rgbd,
            )

        session = PolicyRolloutSession(runner, calibration, collector=collector)
        session.reset(previous_filtered_native_action=np.zeros(9), seed=0)
        height, width = (
            calibration.policy_intrinsics.height,
            calibration.policy_intrinsics.width,
        )
        synthetic_depth_m = float(
            calibration.dp3_point_cloud["xyz_center_m"][2]
        )
        progress = 0.0 if runner.config.trajectory_progress_conditioning else None
        phase = "policy" if runner.config.cyclic_process_phase_conditioning else None
        result = session.step(
            joint_position=np.zeros(10),
            joint_velocity=np.zeros(10),
            rgb=np.zeros((height, width, 3), dtype=np.uint8),
            depth=np.full((height, width), synthetic_depth_m, dtype=np.float32),
            depth_units="metres",
            trajectory_progress=progress,
            process_phase=phase,
            sample_time_s=0.0,
        )
        artifact = collector.close() if collector is not None else None
        print(
            json.dumps(
                {
                    **_summary(runner, calibration),
                    "dry_run": {
                        "policy_action": result.policy_action.tolist(),
                        "filtered_native_action": (
                            result.filtered_native_action.tolist()
                        ),
                        "clipped_elements": result.clipped_elements,
                        "note": (
                            "synthetic observation only; no ROS or robot command "
                            "was sent"
                        ),
                        "recording": str(artifact.data_path) if artifact else None,
                    },
                },
                indent=2,
            )
        )
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"policy rollout error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
