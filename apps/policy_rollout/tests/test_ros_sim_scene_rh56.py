"""The RH56 students' ros-sim scene (make_ros_sim_scene.py --task-profile rh56_windback45_tilt30)."""

import math

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from utils import make_ros_sim_scene as scene

mujoco = pytest.importorskip("mujoco")
linkage = pytest.importorskip("inspire_hand_driver.rh56_linkage")

PROFILE = fo.RH56_WINDBACK45_TILT30_PROFILE
SCENE_PATH = scene.SIM_MJCF_DIR / scene.SCENE_NAMES["rh56"]


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def _numeric(model, name):
    index = model.numeric(name).id
    return np.asarray(model.numeric_data[model.numeric_adr[index]:][: model.numeric_size[index]])


def _text(model, name):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    assert index >= 0, name
    return model.text_data[model.text_adr[index]:].split(b"\0", 1)[0].decode()


def test_checked_in_rh56_scene_is_current():
    assert SCENE_PATH.read_text() == scene.generate(task_profile=PROFILE), (
        "regenerate with make_ros_sim_scene.py --task-profile rh56_windback45_tilt30"
    )


def test_rh56_hand_spans_the_linkage_strokes_with_the_rh56_drives(model):
    table = linkage.load_joint_table()
    for isaac, joint in linkage.WORKSPACE_JOINT.items():
        lower, upper = table.joint_range(isaac)
        assert model.jnt_range[model.joint(joint).id] == pytest.approx([lower, upper], abs=1e-5), joint
        actuator = model.actuator(joint).id
        assert model.actuator_ctrlrange[actuator] == pytest.approx([lower, upper], abs=1e-5), joint
        kp, kv, effort = scene.hand.RH56_DRIVES[joint]
        assert model.actuator_gainprm[actuator, 0] == pytest.approx(kp)
        assert model.actuator_biasprm[actuator, 2] == pytest.approx(-kv)
        assert model.actuator_forcerange[actuator] == pytest.approx([-effort, effort])


def test_rh56_scene_puts_the_followers_on_linkage_curves(model):
    names = [model.numeric(i).name for i in range(model.nnumeric)]
    assert not [n for n in names if "/mimic/" in n]
    assert _numeric(model, "isaac_hand/velocity_limit") == pytest.approx([scene.hand.RH56_VELOCITY_LIMIT])
    table = linkage.load_joint_table()
    curves = [n for n in names if n.startswith("isaac_hand/linkage/")]
    assert len(curves) == 6
    for channel, (leader, _, _) in table.channels.items():
        for follower, (xs, ys) in table.follower_curves(channel).items():
            key = linkage.WORKSPACE_JOINT[follower]
            assert _text(model, f"isaac_hand/linkage_leader/{key}") == linkage.WORKSPACE_JOINT[leader]
            data = _numeric(model, f"isaac_hand/linkage/{key}")
            half = data.size // 2
            # What IsaacHandPlugin interpolates is the table, to the thinning tolerance.
            assert np.max(np.abs(np.interp(xs, data[:half], data[half:]) - ys)) < scene.RH56_CURVE_TOLERANCE_RAD


def test_rh56_keyframe_is_the_profile_reset_on_the_linkage(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
    q = lambda n: data.qpos[model.jnt_qposadr[model.joint(n).id]]  # noqa: E731
    assert [q(f"fr3_joint{i}") for i in range(1, 8)] == pytest.approx(PROFILE.arm_reset_joints)
    table = linkage.load_joint_table()
    assert q("thumb_proximal_yaw_joint") == pytest.approx(PROFILE.grasp_posture["thumb_joint_0"], abs=1e-6)
    assert q("thumb_proximal_pitch_joint") == pytest.approx(PROFILE.grasp_posture["thumb_joint_1"], abs=1e-6)
    assert q("index_proximal_joint") == pytest.approx(PROFILE.grasp_posture["index_joint_0"], abs=1e-6)
    assert q("middle_proximal_joint") == pytest.approx(table.joints_at("middle", 100.0)["middle_joint_0"])
    for channel, (leader, _, _) in table.channels.items():
        for follower, (xs, ys) in table.follower_curves(channel).items():
            expected = np.interp(q(linkage.WORKSPACE_JOINT[leader]), xs, ys)
            assert q(linkage.WORKSPACE_JOINT[follower]) == pytest.approx(expected, abs=1e-6), follower
    for joint in linkage.WORKSPACE_JOINT.values():
        assert data.ctrl[model.actuator(joint).id] == pytest.approx(q(joint)), joint


def test_rh56_nut_starts_at_the_profile_spawn_yaw(model):
    body = model.body("m24_nut")
    # fr3_link0 is the training world yawed by pi: the yaw there is the spawn yaw + 180 deg.
    w, x, y, z = model.body_quat[body.id]
    yaw = math.degrees(2.0 * math.atan2(z, w))
    hex_residual = (yaw - (PROFILE.nut_spawn_yaw_deg + 180.0)) % 60.0
    assert min(hex_residual, 60.0 - hex_residual) == pytest.approx(0.0, abs=1e-4)
