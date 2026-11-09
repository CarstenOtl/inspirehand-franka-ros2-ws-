"""The old-hand wound students' ros-sim scene (make_ros_sim_scene.py --task-profile mimic_windback45_tilt30)."""

import math

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from utils import make_ros_sim_scene as scene

mujoco = pytest.importorskip("mujoco")

PROFILE = fo.MIMIC_WINDBACK45_TILT30_PROFILE
SCENE_PATH = scene.SIM_MJCF_DIR / scene.scene_file_name(PROFILE)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def test_each_profile_has_its_own_scene_file():
    names = [scene.scene_file_name(profile) for profile in fo.TASK_PROFILES.values()]
    assert len(set(names)) == len(names)
    assert scene.scene_file_name(fo.MIMIC_M24_PROFILE) == scene.SCENE_NAME


def test_checked_in_windback_scene_is_current():
    assert SCENE_PATH.read_text() == scene.generate(task_profile=PROFILE), (
        "regenerate with make_ros_sim_scene.py --task-profile mimic_windback45_tilt30"
    )


def test_windback_nut_starts_at_the_profile_spawn_yaw(model):
    w, _, _, z = model.body_quat[model.body("m24_nut").id]
    yaw = math.degrees(2.0 * math.atan2(z, w))
    # fr3_link0 is the training world yawed by pi: the yaw there is the spawn yaw + 180 deg.
    hex_residual = (yaw - (PROFILE.nut_spawn_yaw_deg + 180.0)) % 60.0
    assert min(hex_residual, 60.0 - hex_residual) == pytest.approx(0.0, abs=1e-4)


def _home_tips_about_the_nut(model):
    """Fingertip midpoint minus nut position at the keyframe, and the tip gap."""

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
    mujoco.mj_forward(model, data)
    nut = data.xpos[model.body("m24_nut").id]
    thumb = data.xpos[model.body("thumb_tip").id]
    index = data.xpos[model.body("index_tip").id]
    return 0.5 * (thumb + index) - nut, float(np.linalg.norm(thumb - index))


def test_windback_home_is_the_old_reset_wound_45_deg_about_the_bolt(model):
    """The old hand's reset, carried around the bolt axis: same height and same tip gap."""

    default = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))
    wound_offset, wound_gap = _home_tips_about_the_nut(model)
    offset, gap = _home_tips_about_the_nut(default)
    angle = math.radians(45.0)
    rotated = np.array(
        [
            math.cos(angle) * offset[0] - math.sin(angle) * offset[1],
            math.sin(angle) * offset[0] + math.cos(angle) * offset[1],
            offset[2],
        ]
    )
    assert np.linalg.norm(wound_offset - rotated) < 1e-3
    assert wound_gap == pytest.approx(gap, abs=1e-3)
