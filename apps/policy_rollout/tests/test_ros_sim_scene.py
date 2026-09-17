from pathlib import Path

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from utils import make_ros_sim_scene as scene
from utils.camera_calibration import load_camera_calibration


def test_checked_in_policy_scene_matches_the_camera_profile():
    pytest.importorskip("mujoco")
    target = scene.SIM_MJCF_DIR / scene.SCENE_NAME
    assert target.read_text() == scene.generate(), (
        "regenerate with apps/policy_rollout/utils/make_ros_sim_scene.py"
    )


def test_policy_scene_camera_reproduces_the_calibrated_projection():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
    mujoco.mj_forward(model, data)
    profile = load_camera_calibration()
    k = np.asarray(profile.source_intrinsics.camera_matrix, dtype=float).reshape(3, 3)
    pose = profile.training_world_pose

    camera = model.camera(scene.CAMERA_NAME).id
    optical = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(data.cam_xpos[camera], pose.translation_m, atol=1e-6)
    np.testing.assert_allclose(optical, fo.matrix_from_quat(np.asarray(pose.rotation_wxyz)), atol=1e-6)

    # MuJoCo's projection of the camera intrinsics, as the renderer uses them.
    width, height = model.cam_resolution[camera]
    sensor_w, sensor_h = model.cam_sensorsize[camera]
    fx_len, fy_len, px_len, py_len = model.cam_intrinsic[camera]
    fx, fy = fx_len * width / sensor_w, fy_len * height / sensor_h
    cx, cy = width / 2 - px_len * width / sensor_w, height / 2 - py_len * height / sensor_h
    np.testing.assert_allclose([fx, fy, cx, cy], [k[0, 0], k[1, 1], k[0, 2], k[1, 2]], atol=1e-3)


def test_policy_scene_places_the_bolt_and_home_like_training():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body("m24_bolt").id], [0.59, 0.0, 0.05], atol=1e-9)
    arm = [data.qpos[model.jnt_qposadr[model.joint(f"fr3_joint{i}").id]] for i in range(1, 8)]
    np.testing.assert_allclose(arm, fo.FRANKA_ARM_RESET_JOINTS_M24, atol=1e-8)
