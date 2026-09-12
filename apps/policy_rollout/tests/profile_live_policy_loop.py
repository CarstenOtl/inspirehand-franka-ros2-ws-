#!/usr/bin/env python3
"""Time the live hardware policy pipeline without publishing robot commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parents[1]
DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "checkpoints/sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt"
)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "p95_s": ordered[p95_index],
        "max_s": max(values),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument(
        "--config",
        default=str(
            WORKSPACE_ROOT
            / "src/inspire_franka_trajectory_replay/config/replay.yaml"
        ),
    )
    parser.add_argument("--hand-topic", default="/inspire_hand/command")
    parser.add_argument("--hand-state-topic", default="/inspire_hand/joint_states")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--integration-steps", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--max-frame-skew", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.integration_steps < 1 or args.iterations < 1 or args.timeout <= 0.0:
        parser.error("integration-steps, iterations, and timeout must be positive")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from franka_trajectory_replay.runconfig import load_config
    from policy_rollout.flow_policy import FlowPolicyRunner
    from policy_rollout.hardware import TrainingFrameAdapter, _hardware_node_class
    from policy_rollout.session import PolicyRolloutSession
    from utils.camera_calibration import load_camera_calibration

    calibration = load_camera_calibration(args.camera_calibration)
    config = load_config(args.config)
    runner = FlowPolicyRunner(
        args.checkpoint,
        device=args.device,
        integration_steps=args.integration_steps,
    )
    session = PolicyRolloutSession(runner, calibration)
    frame = TrainingFrameAdapter()

    rclpy.init(args=[])
    node = _hardware_node_class()(
        calibration,
        config,
        hand_command_topic=args.hand_topic,
        hand_state_topic=args.hand_state_topic,
        max_state_age_s=args.max_state_age,
        max_frame_skew_s=args.max_frame_skew,
    )
    executor = MultiThreadedExecutor(num_threads=5)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.wait_ready(args.timeout)
        session.reset(previous_filtered_native_action=np.zeros(9), seed=args.seed)

        timings = {name: [] for name in ("sample", "tf", "policy", "decode", "total")}
        first_pass = None
        for _ in range(args.iterations + 1):
            started = time.perf_counter()
            sample = node.sample()
            sampled = time.perf_counter()
            grasp_position, grasp_quaternion, _flange_quaternion = node.grasp_pose()
            transformed = time.perf_counter()
            q_hand, dq_hand = node.policy_hand_state(sample)
            result = session.step(
                joint_position=np.concatenate((sample.arm_position, q_hand)),
                joint_velocity=np.concatenate((sample.arm_velocity, dq_hand)),
                rgb=sample.rgb,
                depth=sample.depth,
                depth_units=sample.depth_units,
                trajectory_progress=0.0,
                process_phase="policy",
                sample_time_s=started,
                task_signals={"completed_cycles": 0, "watchdog_stop": False},
            )
            inferred = time.perf_counter()
            frame.controller_target(
                result.filtered_native_action.numpy(),
                grasp_position_base=grasp_position,
                grasp_quaternion_base=grasp_quaternion,
                controlled_position_base=sample.controller_measured_position,
                controlled_quaternion_base=sample.controller_measured_quaternion,
            )
            finished = time.perf_counter()
            stages = {
                "sample": sampled - started,
                "tf": transformed - sampled,
                "policy": inferred - transformed,
                "decode": finished - inferred,
                "total": finished - started,
            }
            # Report the first real-frame pass separately from steady state.
            if _ == 0:
                first_pass = stages
                print(
                    json.dumps(
                        {
                            "first_real_frame_pass_s": first_pass,
                            "robot_commands_sent": False,
                        }
                    ),
                    flush=True,
                )
            else:
                for name, value in stages.items():
                    timings[name].append(value)

        print(
            json.dumps(
                {
                    "status": "pass",
                    "robot_commands_sent": False,
                    "iterations": args.iterations,
                    "integration_steps": args.integration_steps,
                    "first_real_frame_pass_s": first_pass,
                    "stages": {name: _summary(values) for name, values in timings.items()},
                },
                indent=2,
            )
        )
        return 0
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
