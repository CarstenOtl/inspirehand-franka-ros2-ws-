#!/usr/bin/env python3
"""Smoke-test the student policy with MuJoCo renders from the calibrated camera.

For a few frames of a recorded training episode this script:

1. poses the replay MJCF (arm, hand, and mocap nut) from the episode,
2. adds a MuJoCo camera at the profile's fr3_link0 -> camera_color_optical_frame
   pose with the calibrated 640x480 intrinsics, and renders aligned RGB-D,
3. runs one policy step on that render and one on the episode's recorded
   320x180 RGB-D, both with the episode's proprioception,
4. compares both actions with the episode's action label and writes a
   side-by-side image for eyeballing.

Nothing here talks to ROS or the robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
WS_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout.observation import FORGE_POLICY_JOINT_NAMES  # noqa: E402
from utils.camera_calibration import load_camera_calibration  # noqa: E402

DEFAULT_MJCF = WS_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand_replay.xml"
DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "checkpoints"
    / "sequential_threading_cycle10_hybrid_teacher_d415_20ep"
    / "checkpoint.pt"
)
DEFAULT_EPISODE = (
    APP_ROOT / "checkpoints" / "reference_episode" / "episode_000_sequential_threading.npz"
)
# MuJoCo cameras look along -z with y up; a ROS optical frame looks along +z
# with y down. They differ by a half turn about x.
OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def quat_wxyz_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(m):
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, dtype=float).flatten())
    return q


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def build_scene(mjcf: Path, profile, resolution, base_plate_z: float = 0.0):
    """Compile the replay scene with a camera at the calibrated optical pose."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(mjcf))
    # The replay MJCF keeps the FR3 base plate 0.333 m below the table (the
    # pre-2026-09-09 Isaac scene). The episode used for this smoke test was
    # recorded in the corrected scene, where base plate and tabletop are both
    # at world z = 0 and joint 1 is 0.333 m above them (forgeUltra
    # HANDOVER.md, "Corrected physical FR3 hand mount"). Raise the base so the
    # recorded joint angles put the hand where the episode says it was.
    base = spec.body("base")
    base.pos = [base.pos[0], base.pos[1], base_plate_z]
    # Compile once to get fr3_link0's world pose (the base is yawed by pi).
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    link0 = model.body("fr3_link0").id
    t_world_link0 = transform(data.xmat[link0].reshape(3, 3), data.xpos[link0])

    pose = profile.training_world_pose
    assert pose.parent_frame_id == "fr3_link0"
    t_link0_optical = transform(
        quat_wxyz_to_matrix(pose.rotation_wxyz), np.asarray(pose.translation_m)
    )
    # With the base plate at table height the calibrated fr3_link0 -> camera
    # transform needs no further adjustment: a plane fit to the recorded
    # training depth puts the camera 0.491 m above the table at 22.4 deg,
    # exactly the calibrated height and tilt relative to fr3_link0.
    t_world_optical = t_world_link0 @ t_link0_optical
    t_world_mjcam = t_world_optical @ transform(OPTICAL_TO_MUJOCO, np.zeros(3))

    k = profile.source_intrinsics
    width, height = resolution
    cam = spec.worldbody.add_camera()
    cam.name = "policy_calibrated"
    cam.pos = t_world_mjcam[:3, 3]
    cam.quat = matrix_to_quat_wxyz(t_world_mjcam[:3, :3])
    fx, fy, cx, cy = k.camera_matrix[0], k.camera_matrix[4], k.camera_matrix[2], k.camera_matrix[5]
    intrinsic_mode = "fovy"
    try:
        cam.resolution = [width, height]
        # sensor size in metres for a unit focal length: pixels / focal.
        cam.sensorsize = [width / fx, height / fy]
        cam.focal = [1.0, 1.0]
        # principal point offset from the image centre, in sensor units.
        cam.principal = [(cx - width / 2.0) / fx, (cy - height / 2.0) / fy]
        intrinsic_mode = "sensorsize+principal"
    except AttributeError:
        cam.fovy = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    model = spec.compile()
    return model, t_world_link0, t_world_optical, intrinsic_mode


def pose_scene(model, data, episode, index, joint_ids, mimic_pairs, nut_body):
    import mujoco

    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    q10 = episode["student_osc_proprio"][index][:10]
    for name, value in zip(FORGE_POLICY_JOINT_NAMES, q10):
        data.qpos[model.jnt_qposadr[joint_ids[name]]] = float(value)
    for source, mimic in mimic_pairs:
        data.qpos[model.jnt_qposadr[mimic]] = data.qpos[model.jnt_qposadr[source]]
    mocap = model.body_mocapid[nut_body]
    data.mocap_pos[mocap] = episode["nut_pos"][index]
    data.mocap_quat[mocap] = episode["nut_quat"][index]  # Isaac stores wxyz
    mujoco.mj_forward(model, data)


def render(renderer, data, camera):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy().astype(np.float32)
    return rgb, depth


def policy_step(session, runner, episode, index, rgb, depth, depth_units):
    proprio = episode["student_osc_proprio"][index]
    session.reset(previous_filtered_native_action=proprio[20:29], seed=0)
    progress = float(index) / max(1, runner.config.trajectory_progress_horizon_steps)
    phase = str(episode["replay_phase"][index])
    step = session.step(
        joint_position=proprio[:10],
        joint_velocity=proprio[10:20],
        rgb=rgb,
        depth=depth,
        depth_units=depth_units,
        trajectory_progress=min(1.0, progress),
        process_phase=phase,
        sample_time_s=float(episode["sample_time_s"][index]),
    )
    return step.policy_action.numpy(), runner.last_chunk.numpy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episode", default=str(DEFAULT_EPISODE))
    parser.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    parser.add_argument("--frames", default="0,60,150,300,500,800")
    parser.add_argument("--output-dir", default=str(DEFAULT_EPISODE.parent / "smoke"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--base-plate-z",
        type=float,
        default=0.0,
        help="world z of the FR3 base plate (training scene: 0.0; replay MJCF: -0.333)",
    )
    args = parser.parse_args(argv)

    import mujoco
    import torch

    from policy_rollout.flow_policy import FlowPolicyRunner
    from policy_rollout.session import PolicyRolloutSession

    profile = load_camera_calibration()
    episode = np.load(args.episode, allow_pickle=False)
    width, height = profile.source_intrinsics.width, profile.source_intrinsics.height
    model, t_world_link0, t_world_optical, intrinsic_mode = build_scene(
        Path(args.mjcf), profile, (width, height), args.base_plate_z
    )
    data = mujoco.MjData(model)
    joint_ids = {name: model.joint(name).id for name in FORGE_POLICY_JOINT_NAMES}
    mimic_pairs = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if name.endswith("_mimic"):
            mimic_pairs.append((model.joint(name[: -len("_mimic")]).id, j))
    nut_body = model.body("recorded_nut").id
    renderer = mujoco.Renderer(model, height, width)

    runner = FlowPolicyRunner(args.checkpoint, device=args.device)
    session = PolicyRolloutSession(runner, profile)

    # Geometry check: the bolt seen from the calibrated camera should sit near
    # the checkpoint's DP3 point-cloud centre.
    mujoco.mj_forward(model, data)
    bolt_world = data.xpos[model.body("m24_bolt").id].copy()
    bolt_optical = (np.linalg.inv(t_world_optical) @ np.append(bolt_world, 1.0))[:3]
    report = {
        "intrinsic_mode": intrinsic_mode,
        "base_plate_z": args.base_plate_z,
        "fr3_link0_world_position": t_world_link0[:3, 3].round(4).tolist(),
        "camera_world_position": t_world_optical[:3, 3].round(4).tolist(),
        "bolt_in_camera_optical_m": bolt_optical.round(4).tolist(),
        "checkpoint_dp3_xyz_center_m": list(profile.dp3_point_cloud["xyz_center_m"]),
        "frames": [],
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    for index in (int(v) for v in args.frames.split(",")):
        pose_scene(model, data, episode, index, joint_ids, mimic_pairs, nut_body)
        rgb, depth = render(renderer, data, "policy_calibrated")
        action_sim, chunk_sim = policy_step(
            session, runner, episode, index, rgb, depth, "metres"
        )
        rec_rgb = np.ascontiguousarray(episode["head_rgb"][index].transpose(1, 2, 0))
        rec_depth = episode["head_depth"][index][0]
        action_rec, chunk_rec = policy_step(
            session, runner, episode, index, rec_rgb, rec_depth, "metres"
        )
        label = episode["unified_osc_action"][index]
        # what the runtime actually saw, for the side-by-side
        from utils.camera_calibration import prepare_rgbd

        seen = prepare_rgbd(rgb, depth, profile, depth_units="metres")
        sim_view = (seen.rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
        sim_depth = seen.depth[0]
        valid_sim = (sim_depth >= 0.1) & (sim_depth <= 2.0)
        valid_rec = (rec_depth >= 0.1) & (rec_depth <= 2.0)
        frame = {
            "index": index,
            "phase": str(episode["replay_phase"][index]),
            "cycle": int(episode["cycle"][index]),
            "label": label.round(3).tolist(),
            "action_from_mujoco_render": action_sim.round(3).tolist(),
            "action_from_recorded_image": action_rec.round(3).tolist(),
            "mae_mujoco_vs_label": float(np.abs(action_sim - label).mean()),
            "mae_recorded_vs_label": float(np.abs(action_rec - label).mean()),
            "mae_mujoco_vs_recorded": float(np.abs(action_sim - action_rec).mean()),
            "chunk_mae_mujoco_vs_recorded": float(np.abs(chunk_sim - chunk_rec).mean()),
            "depth_median_m": {
                "mujoco": float(np.median(sim_depth[valid_sim])) if valid_sim.any() else None,
                "recorded": float(np.median(rec_depth[valid_rec])) if valid_rec.any() else None,
            },
            "valid_depth_fraction": {
                "mujoco": float(valid_sim.mean()),
                "recorded": float(valid_rec.mean()),
            },
        }
        report["frames"].append(frame)

        def depth_panel(d):
            scaled = np.clip((d - 0.5) / 1.0, 0, 1)
            scaled[d <= 0] = 0
            return (np.stack([scaled] * 3, axis=-1) * 255).astype(np.uint8)

        panels.append(
            np.concatenate(
                (
                    np.concatenate((sim_view, depth_panel(sim_depth)), axis=1),
                    np.concatenate((rec_rgb, depth_panel(rec_depth)), axis=1),
                ),
                axis=0,
            )
        )
        np.save(output / f"frame_{index:04d}_mujoco_rgb_640x480.npy", rgb)

    grid = np.concatenate(panels, axis=0)
    try:
        from PIL import Image

        Image.fromarray(grid).save(output / "side_by_side.png")
        report["image"] = str(output / "side_by_side.png")
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(output / "side_by_side.png", grid)
        report["image"] = str(output / "side_by_side.png")
    report["image_layout"] = (
        "per frame, top row: MuJoCo render RGB | MuJoCo depth; "
        "bottom row: recorded training RGB | recorded depth (320x180 each)"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
