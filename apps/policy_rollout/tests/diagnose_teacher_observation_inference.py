#!/usr/bin/env python3
"""Evaluate the deployed flow checkpoint on recorded teacher observations.

This is an offline isolation test: it does not import ROS or command hardware.
The checkpoint receives the episode's exact RGB-D, 29-D native-OSC proprio,
trajectory progress, and cyclic phase. Its outputs are compared with the
recorded unified and filtered/native teacher actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "checkpoints/sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt"
)
DEFAULT_EPISODE = (
    APP_ROOT
    / "checkpoints/reference_episode/episode_000_sequential_threading.npz"
)
DEFAULT_OUTPUT = APP_ROOT.parents[1] / "artifacts/policy_rollout/teacher_observation_test"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    error = predicted.astype(np.float64) - target.astype(np.float64)
    per_component = []
    for index in range(error.shape[1]):
        p = predicted[:, index].astype(np.float64)
        t = target[:, index].astype(np.float64)
        correlation = (
            float(np.corrcoef(p, t)[0, 1])
            if np.std(p) > 0.0 and np.std(t) > 0.0
            else None
        )
        per_component.append(
            {
                "mae": float(np.mean(np.abs(error[:, index]))),
                "rmse": float(np.sqrt(np.mean(np.square(error[:, index])))),
                "max_abs": float(np.max(np.abs(error[:, index]))),
                "correlation": correlation,
            }
        )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs": float(np.max(np.abs(error))),
        "per_component": per_component,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episode", default=str(DEFAULT_EPISODE))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--integration-steps", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--physical-recording",
        default=None,
        help="optional rollout_data.npz used for first-frame proprio sensitivity",
    )
    args = parser.parse_args(argv)
    if args.stride < 1 or any(value < 1 for value in args.integration_steps):
        parser.error("stride and integration steps must be positive")

    from policy_rollout.flow_policy import FlowPolicyRunner
    from policy_rollout.observation import one_hot_process_phase
    from utils.camera_calibration import load_camera_calibration, prepare_rgbd

    # NpzFile is lazy: repeatedly indexing a compressed member would
    # decompress the entire tensor once per sampled row. Materialize the
    # handful of required arrays once before inference instead.
    with np.load(args.episode, allow_pickle=False) as source:
        episode = {
            name: source[name]
            for name in (
                "sample_time_s",
                "replay_phase",
                "student_osc_proprio",
                "unified_osc_action",
                "unified_osc_action_valid",
                "osc_filtered_action",
                "head_rgb",
                "head_depth",
            )
        }
    profile = load_camera_calibration()
    valid_rows = np.flatnonzero(episode["unified_osc_action_valid"])
    indices = valid_rows[:: args.stride]
    if valid_rows[-1] not in indices:
        indices = np.append(indices, valid_rows[-1])
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    action_names = (
        "tcp_dx",
        "tcp_dy",
        "tcp_dz",
        "tcp_drx",
        "tcp_dry",
        "tcp_drz",
        "thumb_yaw",
        "thumb_pitch",
        "index_pitch",
    )
    report = {
        "test": "teacher_observation_checkpoint_inference",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode": str(Path(args.episode).resolve()),
        "seed": args.seed,
        "stride": args.stride,
        "sample_count": int(len(indices)),
        "sample_indices": indices.tolist(),
        "action_names": action_names,
        "results": [],
    }

    for integration_steps in args.integration_steps:
        runner = FlowPolicyRunner(
            args.checkpoint,
            device=args.device,
            integration_steps=integration_steps,
        )
        profile.assert_checkpoint_compatible(runner.config.to_dict())
        predicted = []
        target = []
        predicted_native = []
        target_native = []
        for index in indices:
            rgb = np.ascontiguousarray(
                episode["head_rgb"][index].transpose(1, 2, 0)
            )
            depth = episode["head_depth"][index, 0]
            prepared = prepare_rgbd(rgb, depth, profile, depth_units="metres")
            progress = min(
                1.0,
                float(episode["sample_time_s"][index])
                / runner.config.trajectory_progress_duration_s,
            )
            phase = one_hot_process_phase(
                str(episode["replay_phase"][index]),
                runner.config.cyclic_process_phase_features,
            )
            # Reset makes each row an independent teacher-forced test and
            # removes temporal drift from prior predicted chunks.
            runner.reset(seed=args.seed)
            action = runner.step(
                proprio=episode["student_osc_proprio"][index],
                head_rgb=prepared.rgb,
                head_depth=prepared.depth,
                valid_mask=prepared.valid_mask,
                trajectory_progress=progress,
                cyclic_process_phase=phase,
            ).numpy()
            predicted.append(action)
            target.append(episode["unified_osc_action"][index])
            predicted_native.append(
                np.clip(action, -1.0, 1.0)
                * np.asarray(runner.config.osc_native_action_scale)
            )
            target_native.append(episode["osc_filtered_action"][index])

        predicted = np.asarray(predicted)
        target = np.asarray(target)
        predicted_native = np.asarray(predicted_native)
        target_native = np.asarray(target_native)
        result = {
            "integration_steps": integration_steps,
            "unified_action": _metrics(predicted, target),
            "native_filtered_action": _metrics(predicted_native, target_native),
        }
        report["results"].append(result)
        np.savez_compressed(
            output / f"predictions_{integration_steps}_steps.npz",
            sample_index=indices,
            sample_time_s=episode["sample_time_s"][indices],
            predicted_unified_action=predicted,
            target_unified_action=target,
            predicted_native_action=predicted_native,
            target_native_action=target_native,
        )

    if args.physical_recording is not None:
        with np.load(args.physical_recording, allow_pickle=False) as source:
            physical_proprio = source["proprio"][0].copy()
            physical_action = source["policy_action"][0].copy()
        teacher_proprio = episode["student_osc_proprio"][0].copy()
        variants = {
            "exact_teacher_proprio": teacher_proprio,
            "physical_previous_action_only": np.concatenate(
                (teacher_proprio[:20], physical_proprio[20:29])
            ),
            "physical_hand_position_velocity_only": teacher_proprio.copy(),
            "full_physical_proprio": physical_proprio,
        }
        from inspire_hand_driver import command_overlays
        from inspire_hand_driver import kinematics as hand_kinematics

        corrected_physical = physical_proprio.copy()
        corrected_physical[20:29] = 0.0
        thumb_dof = hand_kinematics.dof_index(
            command_overlays.THUMB_ABDUCTION_JOINT
        )
        physical_ratio = hand_kinematics.rad_to_open_ratio(
            thumb_dof, corrected_physical[7]
        )
        logical_ratio = command_overlays.invert_open_ratio_overlay(
            thumb_dof, physical_ratio
        )
        corrected_physical[7] = hand_kinematics.open_ratio_to_rad(
            thumb_dof, logical_ratio
        )
        corrected_physical[17] /= (
            1.0 - command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
        )
        variants["corrected_physical_proprio"] = corrected_physical
        variants["physical_hand_position_velocity_only"][7:10] = physical_proprio[7:10]
        variants["physical_hand_position_velocity_only"][17:20] = physical_proprio[17:20]
        first_rgb = np.ascontiguousarray(episode["head_rgb"][0].transpose(1, 2, 0))
        first_prepared = prepare_rgbd(
            first_rgb, episode["head_depth"][0, 0], profile, depth_units="metres"
        )
        sensitivity = []
        for integration_steps in args.integration_steps:
            runner = FlowPolicyRunner(
                args.checkpoint,
                device=args.device,
                integration_steps=integration_steps,
            )
            phase = one_hot_process_phase(
                str(episode["replay_phase"][0]),
                runner.config.cyclic_process_phase_features,
            )
            rows = []
            for name, proprio in variants.items():
                runner.reset(seed=args.seed)
                action = runner.step(
                    proprio=proprio,
                    head_rgb=first_prepared.rgb,
                    head_depth=first_prepared.depth,
                    valid_mask=first_prepared.valid_mask,
                    trajectory_progress=0.0,
                    cyclic_process_phase=phase,
                ).numpy()
                rows.append(
                    {
                        "variant": name,
                        "action": action.tolist(),
                        "mae_vs_teacher_label": float(
                            np.mean(np.abs(action - episode["unified_osc_action"][0]))
                        ),
                        "mae_vs_real_first_action": float(
                            np.mean(np.abs(action - physical_action))
                        ),
                    }
                )
            sensitivity.append(
                {"integration_steps": integration_steps, "variants": rows}
            )
        report["first_frame_proprio_sensitivity"] = {
            "physical_recording": str(Path(args.physical_recording).resolve()),
            "teacher_previous_action": teacher_proprio[20:29].tolist(),
            "physical_previous_action": physical_proprio[20:29].tolist(),
            "teacher_hand_position": teacher_proprio[7:10].tolist(),
            "physical_hand_position": physical_proprio[7:10].tolist(),
            "results": sensitivity,
        }

    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
