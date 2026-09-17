#!/usr/bin/env python3
"""Generate the ros2_control MuJoCo scene for rehearsing the student rollout.

``src/inspire_franka_sim/mjcf/inspire_franka_policy_scene.xml`` is the flange
torque scene (FR3 + Inspire hand, gravity off) placed in the student's training
environment instead of ``scene_common.xml``:

- the bench top at fr3_link0 z = 0 and the floor 0.75 m below it, as in the
  corrected training scene (``policy_rollout/mujoco_scene.py``);
- the M24 bolt at the training pose and the nut at its first recorded pose,
  both converted from the training world (FR3 base at (1.2, 0, 0) yawed by pi)
  into fr3_link0;
- ``policy_d415``: the calibrated colour camera with the exact 640x480 pinhole
  matrix (focal length and principal point, not only fovy) so that
  mujoco_ros2_control renders what the checkpoint's camera contract expects;
- keyframe ``policy_home``: the M24 reset joints and the training grasp posture.

The nut is fixed. This hand model has no collision geometry, so the simulation
checks the rollout plumbing (frames, camera, rates, controller and hand command
paths), not the contact-rich threading itself.

Regenerate after changing the camera profile; ``tests/test_ros_sim_scene.py``
fails while the checked-in file is stale:

    python3 apps/policy_rollout/utils/make_ros_sim_scene.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import forge_osc as fo  # noqa: E402
from policy_rollout.mujoco_scene import OPTICAL_TO_MUJOCO  # noqa: E402
from utils.camera_calibration import load_camera_calibration  # noqa: E402

SIM_MJCF_DIR = WORKSPACE_ROOT / "src" / "inspire_franka_sim" / "mjcf"
SCENE_NAME = "inspire_franka_policy_scene.xml"
CAMERA_NAME = "policy_d415"
KEYFRAME_NAME = "policy_home"
MESH_SOURCE = WORKSPACE_ROOT / "assets" / "fr3_inspirehand" / "forge_nut_bolt" / "m24"
MESH_SUBDIR = "forge_m24"

# First sample of checkpoints/reference_episode/episode_000_sequential_threading.npz
# (nut_pos, nut_quat wxyz) in the training world. Kept literal: the episode is
# a 250 MB ignored file.
TRAINING_NUT_POSITION = (0.61, 0.0, 0.10997997)
TRAINING_NUT_QUATERNION = (0.2588191, 0.0, 0.0, 0.96592575)

# Training reset grasp posture (utils/mujoco_student_rollout.RESET_HAND_POSTURE)
# in this workspace's RH56 joint names. The thumb yaw is the policy's logical
# coordinate; the simulator has no command overlay, so this is also the pose.
POLICY_HOME_HAND = {
    "thumb_proximal_yaw_joint": fo.THREADING_GRASP_POSTURE["thumb_joint_0"],
    "thumb_proximal_pitch_joint": fo.THREADING_GRASP_POSTURE["thumb_joint_1"],
    "index_proximal_joint": fo.THREADING_GRASP_POSTURE["index_joint_0"],
    "middle_proximal_joint": 1.333,
    "ring_proximal_joint": 1.333,
    "pinky_proximal_joint": 1.333,
}

# Pixel size of the virtual sensor. Any value works; MuJoCo only uses the
# ratios focal/sensorsize and principal/sensorsize.
SENSOR_PIXEL_M = 6.0e-6


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def training_world_to_link0(position, quaternion_wxyz):
    """Pose in the training world -> pose in fr3_link0 (base at (1.2,0,0), yaw pi)."""

    q_yaw = fo.quat_from_euler_xyz(0.0, 0.0, math.pi)
    q_inv = fo.quat_conjugate(q_yaw)
    p = fo.quat_rotate(q_inv, np.asarray(position, float) - fo.ROBOT_BASE_POSITION)
    q = fo.quat_mul(q_inv, np.asarray(quaternion_wxyz, float))
    return p, q / np.linalg.norm(q)


def camera_element(profile) -> str:
    """The calibrated colour camera as an MJCF element in fr3_link0 coordinates."""

    pose = profile.training_world_pose
    if pose.parent_frame_id != "fr3_link0":
        raise ValueError("camera profile pose must be expressed in fr3_link0")
    rotation = fo.matrix_from_quat(np.asarray(pose.rotation_wxyz, float)) @ OPTICAL_TO_MUJOCO
    quaternion = fo.quat_from_matrix(rotation)
    k = profile.source_intrinsics
    fx, fy, cx, cy = (float(k.camera_matrix[i]) for i in (0, 4, 2, 5))
    width, height = int(k.width), int(k.height)
    pixel = SENSOR_PIXEL_M
    # MuJoCo's principal offset is measured from the sensor centre towards
    # -x/+y of the image, which is (centre - c) in pixel coordinates.
    return (
        f'<camera name="{CAMERA_NAME}" pos="{_fmt(pose.translation_m)}" '
        f'quat="{_fmt(quaternion)}" resolution="{width} {height}" '
        f'sensorsize="{_fmt((width * pixel, height * pixel))}" '
        f'focal="{_fmt((fx * pixel, fy * pixel))}" '
        f'principal="{_fmt(((0.5 * width - cx) * pixel, (0.5 * height - cy) * pixel))}"/>'
    )


def scene_xml(profile, keyframe: str = "") -> str:
    bolt, _ = training_world_to_link0(fo.BOLT_BASE_POSITION, (1.0, 0.0, 0.0, 0.0))
    bolt_quat = fo.quat_from_euler_xyz(0.0, 0.0, math.pi)
    nut, nut_quat = training_world_to_link0(TRAINING_NUT_POSITION, TRAINING_NUT_QUATERNION)
    table, _ = training_world_to_link0((0.335, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    width = int(profile.source_intrinsics.width)
    height = int(profile.source_intrinsics.height)
    return f"""<mujoco model="inspire_franka_policy_scene">
  <!-- GENERATED by apps/policy_rollout/utils/make_ros_sim_scene.py from
       apps/policy_rollout/utils/camera_calibration/fr3_realsense_dp3.yaml.
       Do not edit by hand; rerun the generator.

       The flange torque scene (gravity off, as libfranka compensates it on
       the real arm) in the distilled policy's training environment, for
       inspire_franka_trajectory_replay's sim_policy.launch.py. Poses are the
       training world's converted into fr3_link0: bench top at z = 0, M24 bolt
       and first recorded nut pose, and the calibrated D415 colour camera with
       its exact pinhole matrix. The nut is a fixed visual; this hand model has
       no collision geometry. -->

  <asset>
    <model name="end_effector" file="inspire_hand_on_flange.xml"/>
    <texture type="skybox" builtin="flat" rgb1="0.92 0.92 0.92" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.31 0.31 0.31" rgb2="0.31 0.31 0.31" markrgb="0.78 0.78 0.78"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="2 2" reflectance="0.05"/>
    <material name="table_top" rgba="0.93 0.93 0.91 1" roughness="0.55"/>
    <material name="table_leg" rgba="0.13 0.14 0.15 1" roughness="0.35" metallic="0.65"/>
    <material name="forge_bolt" rgba="0.055 0.055 0.065 1"/>
    <material name="forge_nut" rgba="0.84 0.84 0.88 1"/>
    <mesh name="forge_m24_nut" file="{MESH_SUBDIR}/m24_nut.stl"/>
    <mesh name="forge_m24_bolt" file="{MESH_SUBDIR}/m24_bolt.stl"/>
  </asset>

  <include file="fr3.xml"/>

  <option gravity="0 0 0"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0 0 0"/>
    <rgba haze="0.7 0.7 0.7 1"/>
    <global azimuth="120" elevation="-20" offwidth="{max(width, 1920)}" offheight="{max(height, 1080)}"/>
  </visual>

  <worldbody>
    <geom name="floor" pos="0 0 -0.75" size="0 0 0.05" type="plane" material="groundplane"/>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" castshadow="false"/>
    <!-- mujoco_ros2_control renders every camera here, so this is the only one. -->
    {camera_element(profile)}

    <body name="table" pos="{_fmt(table)}">
      <geom name="table_top" type="box" pos="0 0 -0.015" size="0.4 0.3 0.015"
        material="table_top" contype="0" conaffinity="0"/>
      <geom name="table_leg_0" type="cylinder" pos="0.35 0.25 -0.39" size="0.015 0.36"
        material="table_leg" contype="0" conaffinity="0"/>
      <geom name="table_leg_1" type="cylinder" pos="0.35 -0.25 -0.39" size="0.015 0.36"
        material="table_leg" contype="0" conaffinity="0"/>
      <geom name="table_leg_2" type="cylinder" pos="-0.35 0.25 -0.39" size="0.015 0.36"
        material="table_leg" contype="0" conaffinity="0"/>
      <geom name="table_leg_3" type="cylinder" pos="-0.35 -0.25 -0.39" size="0.015 0.36"
        material="table_leg" contype="0" conaffinity="0"/>
    </body>
    <body name="m24_bolt" pos="{_fmt(bolt)}" quat="{_fmt(bolt_quat)}">
      <geom name="m24_bolt_geom" type="mesh" mesh="forge_m24_bolt" material="forge_bolt"
        contype="0" conaffinity="0"/>
    </body>
    <body name="m24_nut" pos="{_fmt(nut)}" quat="{_fmt(nut_quat)}">
      <geom name="m24_nut_geom" type="mesh" mesh="forge_m24_nut" material="forge_nut"
        contype="0" conaffinity="0"/>
    </body>
  </worldbody>
{keyframe}</mujoco>
"""


def _keyframe(model) -> str:
    import mujoco

    qpos = model.qpos0.copy()
    for index, value in enumerate(fo.FRANKA_ARM_RESET_JOINTS_M24, start=1):
        qpos[model.jnt_qposadr[model.joint(f"fr3_joint{index}").id]] = value
    for name, value in POLICY_HOME_HAND.items():
        qpos[model.jnt_qposadr[model.joint(name).id]] = value
    # Put the mimic followers on their couplings so the first step is quiet.
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        follower, leader = model.eq_obj1id[eq], model.eq_obj2id[eq]
        c = model.eq_data[eq]
        x = qpos[model.jnt_qposadr[leader]]
        qpos[model.jnt_qposadr[follower]] = c[0] + c[1] * x + c[2] * x**2
    return (
        f'  <keyframe>\n    <key name="{KEYFRAME_NAME}" qpos="{_fmt(qpos)}"/>\n'
        "  </keyframe>\n"
    )


def generate(profile_path=None, output_dir: Path = SIM_MJCF_DIR) -> str:
    """Return the scene text; compiles it once to lay out the keyframe."""

    import mujoco

    profile = load_camera_calibration(profile_path)
    output_dir = Path(output_dir)
    mesh_dir = output_dir / "assets" / MESH_SUBDIR
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for mesh in ("m24_nut.stl", "m24_bolt.stl"):
        if not (mesh_dir / mesh).exists():
            shutil.copyfile(MESH_SOURCE / mesh, mesh_dir / mesh)
    # Includes resolve relative to the model file, so compile beside the scene.
    probe = output_dir / f".{SCENE_NAME}.probe.xml"
    try:
        probe.write_text(scene_xml(profile))
        model = mujoco.MjModel.from_xml_path(str(probe))
    finally:
        probe.unlink(missing_ok=True)
    return scene_xml(profile, _keyframe(model))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in scene is stale")
    args = parser.parse_args(argv)
    text = generate(args.camera_calibration)
    target = SIM_MJCF_DIR / SCENE_NAME
    if args.check:
        if not target.exists() or target.read_text() != text:
            print(f"{target} is stale; rerun {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
