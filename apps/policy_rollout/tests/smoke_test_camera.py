#!/usr/bin/env python3
"""Check live policy RGB-D and run one student inference without robot commands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--integration-steps", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=0.75,
        help="Time to monitor synchronized pairs before inference (default: 0.75).",
    )
    parser.add_argument("--max-frame-skew", type=float, default=0.04)
    return parser


def _stamp(message) -> float:
    return float(message.header.stamp.sec) + 1.0e-9 * float(
        message.header.stamp.nanosec
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.timeout <= 0.0
        or args.observe_seconds <= 0.0
        or args.max_frame_skew <= 0.0
        or args.integration_steps < 1
    ):
        _parser().error("timeout, max-frame-skew, and integration-steps must be positive")

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from realsense2_camera_msgs.msg import RGBD

    from policy_rollout.flow_policy import FlowPolicyRunner
    from policy_rollout.hardware import (
        RgbdFrameSynchronizer,
        assert_policy_camera_frames,
        image_message_to_numpy,
    )
    from utils.camera_calibration import load_camera_calibration, prepare_rgbd

    profile = load_camera_calibration(args.camera_calibration)
    blockers = profile.hardware_blockers()
    if blockers:
        raise RuntimeError("camera profile is not hardware-ready: " + "; ".join(blockers))
    frames = RgbdFrameSynchronizer(args.max_frame_skew)
    messages = {"info": None}

    def put_rgbd(message):
        received = time.monotonic()
        frames.add("rgb", message.rgb, received)
        frames.add("depth", message.depth, received)
        messages["info"] = message.rgb_camera_info

    rclpy.init(args=[])
    node = rclpy.create_node("policy_camera_smoke_test")
    subscriptions = (
        node.create_subscription(
            RGBD,
            profile.rgbd_topic,
            put_rgbd,
            qos_profile_sensor_data,
        ),
    )
    try:
        ready_deadline = time.monotonic() + args.timeout
        skew = math.inf
        pair = None
        max_pair_age = 0.0
        stale_pair_observations = 0
        pair_stamps = set()
        while time.monotonic() < ready_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            pair = frames.latest_pair()
            if pair is not None and messages["info"] is not None:
                break
        else:
            rgb_queue = frames._queues["rgb"]
            depth_queue = frames._queues["depth"]
            nearest = min(
                (
                    (abs(rgb[0] - depth[0]), rgb[0], depth[0])
                    for rgb in rgb_queue
                    for depth in depth_queue
                ),
                default=None,
            )
            raise RuntimeError(
                "policy camera inputs did not synchronize; "
                f"buffered rgb={len(rgb_queue)} depth={len(depth_queue)}, "
                f"nearest timestamp pair={nearest}"
            )

        # Startup readiness and observation duration are separate windows: a
        # ten-second observation must not fail merely because the first frame
        # took part of the startup timeout to arrive.
        observation_deadline = time.monotonic() + args.observe_seconds
        while time.monotonic() < observation_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            current = frames.latest_pair()
            if current is None:
                continue
            pair = current
            pair_age = time.monotonic() - min(pair[2], pair[3])
            max_pair_age = max(max_pair_age, pair_age)
            stale_pair_observations += int(pair_age > 0.5)
            pair_stamps.add((_stamp(pair[0]), _stamp(pair[1])))

        rgb_message, depth_message, rgb_received, depth_received = pair
        skew = abs(_stamp(rgb_message) - _stamp(depth_message))
        frame_age = time.monotonic() - min(rgb_received, depth_received)
        info = messages["info"]
        assert_policy_camera_frames(rgb_message, depth_message, info, profile.frame_id)
        profile.assert_live_camera_info(info.width, info.height, info.k)
        rgb, _ = image_message_to_numpy(rgb_message)
        depth, depth_units = image_message_to_numpy(depth_message)
        prepared = prepare_rgbd(
            rgb, depth, profile, depth_units=depth_units
        )

        runner = FlowPolicyRunner(
            args.checkpoint,
            device=args.device,
            integration_steps=args.integration_steps,
        )
        profile.assert_checkpoint_compatible(runner.config.to_dict())
        runner.reset(seed=0)
        phase = None
        if runner.config.cyclic_process_phase_conditioning:
            phase = np.zeros(len(runner.config.cyclic_process_phase_features))
            phase[0] = 1.0
        progress = 0.0 if runner.config.trajectory_progress_conditioning else None
        def infer():
            runner.reset(seed=0)
            started = time.perf_counter()
            result = runner.step(
                proprio=np.zeros(runner.config.proprio_dim, dtype=np.float32),
                head_rgb=prepared.rgb,
                head_depth=prepared.depth,
                valid_mask=prepared.valid_mask,
                trajectory_progress=progress,
                cyclic_process_phase=phase,
            ).numpy()
            return result, time.perf_counter() - started

        action, cold_inference_time = infer()
        action, warm_inference_time = infer()
        if action.shape != (runner.config.joint_dim,) or not np.isfinite(action).all():
            raise RuntimeError(f"student returned an invalid action: {action}")

        valid_fraction = float(prepared.valid_mask.mean())
        print(
            json.dumps(
                {
                    "status": "pass",
                    "robot_commands_sent": False,
                    "camera_profile_ready": True,
                    "camera_serial": profile.serial_number,
                    "rgb_topic": profile.color_topic,
                    "depth_topic": profile.depth_topic,
                    "rgbd_topic": profile.rgbd_topic,
                    "frame_id": profile.frame_id,
                    "source_shape": list(rgb.shape),
                    "policy_rgb_shape": list(prepared.rgb.shape),
                    "policy_depth_shape": list(prepared.depth.shape),
                    "frame_skew_s": skew,
                    "frame_age_s": frame_age,
                    "observation_seconds": args.observe_seconds,
                    "synchronized_pair_count": len(pair_stamps),
                    "max_pair_age_s": max_pair_age,
                    "stale_pair_observations": stale_pair_observations,
                    "valid_depth_fraction": valid_fraction,
                    "student_action_shape": list(action.shape),
                    "student_action_finite": True,
                    "integration_steps": args.integration_steps,
                    "cold_inference_time_s": cold_inference_time,
                    "warm_inference_time_s": warm_inference_time,
                    "warm_inference_hz": 1.0 / warm_inference_time,
                },
                indent=2,
            )
        )
        return 0
    finally:
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
