"""Closed-loop student rollout in MuJoCo with the ported ForgeUltra control path.

The DP3 flow student sees the calibrated camera render and MuJoCo joint
state at 15 Hz; its unified OSC action is executed through the ported
operational-space controller and joint-PD adapter at 120 Hz; the cyclic
coordinator runs release/return exactly like ``forge_transitions.py``. The
run is recorded in the rollout schema so it can be evaluated against the
converted reference teacher episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import forge_osc as fo  # noqa: E402
from policy_rollout.mujoco_scene import add_policy_camera  # noqa: E402
from policy_rollout.mujoco_threading_env import (  # noqa: E402
    ARM_JOINTS,
    DECIMATION,
    HAND_POLICY_JOINTS,
    POLICY_JOINTS,
    CyclicCoordinator,
    ThreadingScene,
    seed_action_history,
)
from utils.camera_calibration import load_camera_calibration  # noqa: E402

DEFAULT_CHECKPOINT = APP_ROOT / "checkpoints" / "sequential_threading_cycle10_hybrid_teacher_d415_20ep" / "checkpoint.pt"
DEFAULT_EPISODE = APP_ROOT / "checkpoints" / "reference_episode" / "episode_000_sequential_threading.npz"
RESET_HAND_POSTURE = {
    **fo.THREADING_GRASP_POSTURE,
    "middle_joint_0": 1.333,
    "ring_joint_0": 1.333,
    "little_joint_0": 1.333,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_policy_rollout.py mujoco", description=__doc__
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episode", default=str(DEFAULT_EPISODE), help="reference episode (nut spawn pose)")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--output-dir", default=str(DEFAULT_EPISODE.parent / "student_rollout"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-rgbd", action="store_true")
    parser.add_argument("--preview-every", type=int, default=0, help="save a render every N steps")
    parser.add_argument("--video", action="store_true", help="record an MP4 of the rollout")
    parser.add_argument("--video-camera", default="policy_replay_front", help="MJCF camera for the video")
    parser.add_argument("--video-size", default="960x540")
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--dead-zone", choices=["none", "default"], default="none")
    args = parser.parse_args(argv)
    if args.integration_steps < 1:
        parser.error("--integration-steps must be positive")

    import mujoco
    import torch

    from policy_rollout.flow_policy import FlowPolicyRunner
    from policy_rollout.session import PolicyRolloutSession
    from utils.data_collection import RolloutDataCollector

    profile = load_camera_calibration()
    episode = np.load(args.episode, allow_pickle=False)
    nut_quat0 = episode["nut_quat"][0].astype(float)
    scene = ThreadingScene(nut_quat_wxyz=nut_quat0, camera=lambda spec: add_policy_camera(spec, profile))
    width, height = profile.source_intrinsics.width, profile.source_intrinsics.height
    renderer = mujoco.Renderer(scene.model, height, width)
    video_renderer = None
    video_frames = []
    if args.video:
        vw, vh = (int(v) for v in args.video_size.lower().split("x"))
        video_renderer = mujoco.Renderer(scene.model, vh, vw)

    runner = FlowPolicyRunner(
        args.checkpoint,
        device=args.device,
        integration_steps=args.integration_steps,
        seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = runner.metadata()
    collector = RolloutDataCollector(
        output,
        metadata={
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": metadata["sha256"],
            "checkpoint_weight_source": metadata["weight_source"],
            "flow_integration_steps": metadata["integration_steps"],
            "camera_profile": str(profile.source_path),
            "collection_mode": "mujoco_closed_loop_student",
            "reference_episode": str(args.episode),
        },
        record_rgbd=args.record_rgbd,
    )
    session = PolicyRolloutSession(runner, profile, collector=collector)
    dead_zone = fo.DEFAULT_DEAD_ZONE if args.dead_zone == "default" else None

    # --- reset (Isaac: reset joints, grasp posture, 0.25 s settle) ------------
    scene.reset(fo.FRANKA_ARM_RESET_JOINTS_M24, RESET_HAND_POSTURE)
    thumb, _ = scene.body_pose(scene.thumb_tip)
    index, _ = scene.body_pose(scene.index_tip)
    flange_pos, flange_quat = scene.body_pose(scene.flange)
    z_transport = fo.reset_z_transport(thumb, index, flange_pos, flange_quat, tilt_deg=0.0)
    reset_grasp = scene.grasp_state(z_transport)
    reset_hand = scene.q(HAND_POLICY_JOINTS)
    coordinator = CyclicCoordinator(
        max_cycles=args.cycles,
        reset_grasp_pos=reset_grasp.pos.copy(),
        reset_grasp_quat=reset_grasp.quat.copy(),
        reset_hand=reset_hand.copy(),
    )
    session.reset(previous_filtered_native_action=np.zeros(9), seed=args.seed)
    duration_s = float(runner.config.trajectory_progress_duration_s)

    def render():
        renderer.disable_depth_rendering()
        renderer.update_scene(scene.data, camera="policy_calibrated")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(scene.data, camera="policy_calibrated")
        depth = renderer.render().copy().astype(np.float32)
        return rgb, depth

    log = {k: [] for k in (
        "sim_time_s", "phase", "cycle", "turn_progress_deg", "nut_axial_m", "nut_twist_rad",
        "grasp_pos", "grasp_quat", "hand_q", "pd_command", "osc_torque", "target_pos", "target_quat",
        "return_errors", "arm_q", "policy_action", "filtered_action",
    )}
    previews = []
    start_wall = time.perf_counter()
    status = "step_budget"
    print(f"reset grasp frame {reset_grasp.pos.round(4)} hand {reset_hand.round(3)} nut yaw0 {nut_quat0.round(4)}", flush=True)
    for step in range(args.max_steps):
        rgb, depth = render()
        q10 = scene.q(POLICY_JOINTS)
        dq10 = scene.dq(POLICY_JOINTS)
        phase = coordinator.process_phase()
        progress = min(1.0, scene.sim_time / duration_s)
        result = session.step(
            joint_position=q10,
            joint_velocity=dq10,
            rgb=rgb,
            depth=depth,
            depth_units="metres",
            trajectory_progress=progress,
            process_phase=phase,
            sample_time_s=scene.sim_time,
            task_signals={
                "pickup_success": True,
                "threading_entered": True,
                "completed_cycles": coordinator.completed_cycles,
                "threading_turn_progress_rad": scene.turn_progress_rad,
                "terminated": False,
                "truncated": False,
                "watchdog_stop": False,
            },
        )
        filtered = result.filtered_native_action.numpy()
        for _ in range(DECIMATION):
            diag = scene.control_tick(filtered, z_transport, dead_zone=dead_zone)
        grasp = diag["grasp"]
        hand = scene.q(HAND_POLICY_JOINTS)
        errors = coordinator.return_errors(grasp, hand)
        event = coordinator.observe(
            step=step, turn_progress_rad=scene.turn_progress_rad, grasp=grasp, hand_joints=hand, scene=scene
        )
        if event == "cycle_completed":
            seed = seed_action_history(scene.grasp_state(z_transport), scene.q(fo.PINCH_JOINTS))
            session.action_filter.previous_filtered = torch.as_tensor(seed, dtype=torch.float32)
        if event is not None:
            print(f"[step {step:4d} t={scene.sim_time:6.2f}s] {event}: cycles={coordinator.completed_cycles} turn={np.degrees(scene.turn_progress_rad):.1f} deg return_err(p,o,h)=({errors[0]*1000:.1f} mm, {errors[1]:.1f} deg, {errors[2]:.3f} rad)", flush=True)
        log["sim_time_s"].append(scene.sim_time)
        log["phase"].append(phase)
        log["cycle"].append(coordinator.completed_cycles + 1)
        log["turn_progress_deg"].append(np.degrees(scene.turn_progress_rad))
        log["nut_axial_m"].append(scene.axial_position)
        log["nut_twist_rad"].append(scene.twist_angle)
        log["grasp_pos"].append(grasp.pos)
        log["grasp_quat"].append(grasp.quat)
        log["hand_q"].append(hand)
        log["pd_command"].append(diag["pd_command"])
        log["osc_torque"].append(diag["osc_torque"])
        log["target_pos"].append(diag["target_pos"])
        log["target_quat"].append(diag["target_quat"])
        log["return_errors"].append(np.array(errors))
        log["arm_q"].append(scene.q(ARM_JOINTS))
        log["policy_action"].append(result.policy_action.numpy())
        log["filtered_action"].append(filtered)
        if args.preview_every and step % args.preview_every == 0:
            previews.append((step, rgb))
        if video_renderer is not None:
            video_renderer.update_scene(scene.data, camera=args.video_camera)
            video_frames.append(video_renderer.render().copy())
        if step % 50 == 0:
            print(f"step {step:4d} t={scene.sim_time:6.2f}s phase={phase:16s} cycles={coordinator.completed_cycles} turn={np.degrees(scene.turn_progress_rad):6.1f} deg grasp={grasp.pos.round(3)} nut_z={fo.BOLT_TIP_POSITION[2]+scene.axial_position:.4f} wall={time.perf_counter()-start_wall:.0f}s", flush=True)
        if coordinator.limit_reached:
            status = "all_cycles_completed"
            break
        if coordinator.failed:
            status = "return_failed"
            break
        if not np.isfinite(scene.data.qpos).all():
            status = "numerical_failure"
            break

    artifact = collector.close()
    arrays = {k: np.asarray(v) for k, v in log.items()}
    np.savez_compressed(output / "sim_state.npz", **arrays)
    if previews:
        try:
            from PIL import Image

            for step, rgb in previews:
                Image.fromarray(rgb).save(output / f"preview_{step:04d}.png")
        except ImportError:
            pass
    video_path = None
    if video_frames:
        video_path = output / "rollout.mp4"
        try:
            import imageio.v2 as imageio

            imageio.mimsave(video_path, video_frames, fps=args.video_fps, macro_block_size=1)
        except ImportError:
            import subprocess

            raw = output / "frames"
            raw.mkdir(exist_ok=True)
            from PIL import Image

            for index, frame in enumerate(video_frames):
                Image.fromarray(frame).save(raw / f"{index:05d}.png")
            subprocess.run(
                ["ffmpeg", "-y", "-framerate", str(args.video_fps), "-i", str(raw / "%05d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)],
                check=True, capture_output=True,
            )
    report = {
        "status": status,
        "steps": len(log["sim_time_s"]),
        "sim_time_s": float(scene.sim_time),
        "completed_cycles": coordinator.completed_cycles,
        "events": coordinator.events,
        "final_turn_progress_deg": float(np.degrees(scene.turn_progress_rad)),
        "final_nut_twist_deg": float(np.degrees(scene.twist_angle)),
        "rollout_data": str(artifact.data_path),
        "wall_time_s": time.perf_counter() - start_wall,
        "video": None if video_path is None else str(video_path),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0
