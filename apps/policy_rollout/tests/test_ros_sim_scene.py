import math
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


def _thread_parameters(model):
    """Everything about the thread pair that is frame-independent."""

    out = {}
    for name in ("nut_axial", "nut_twist"):
        joint = model.joint(name)
        out[f"{name}.type"] = int(joint.type[0])
        out[f"{name}.axis"] = tuple(np.round(joint.axis, 9))
        out[f"{name}.range"] = tuple(np.round(joint.range, 9))
        out[f"{name}.armature"] = float(joint.armature[0])
        out[f"{name}.damping"] = float(joint.damping[0])
        out[f"{name}.frictionloss"] = float(joint.frictionloss[0])
    for name in ("thread_coupling", "thread_hold"):
        index = model.equality(name).id
        out[f"{name}.type"] = int(model.eq_type[index])
        out[f"{name}.active0"] = bool(model.eq_active0[index])
        out[f"{name}.polycoef"] = tuple(np.round(model.eq_data[index][:5], 12))
        out[f"{name}.solref"] = tuple(np.round(model.eq_solref[index], 9))
        out[f"{name}.solimp"] = tuple(np.round(model.eq_solimp[index], 9))
    nut = model.geom("m24_nut_geom")
    out["nut.contype"] = int(nut.contype[0])
    out["nut.conaffinity"] = int(nut.conaffinity[0])
    out["nut.friction"] = tuple(np.round(nut.friction, 9))
    out["nut.mass"] = round(float(model.body_mass[model.body("m24_nut").id]), 9)
    out["carrier.mass"] = round(float(model.body_mass[model.body("nut_carrier").id]), 9)
    return out


def test_policy_scene_thread_pair_matches_the_training_plant():
    """The port is pinned to its source, not to numbers copied out of it.

    ``ThreadingScene`` is the MuJoCo-only loop that ran on the training asset;
    if either side of the port drifts, this fails with the parameter named.
    """

    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("yaml")
    from policy_rollout.mujoco_threading_env import ThreadingScene

    training = ThreadingScene(nut_quat_wxyz=(1.0, 0.0, 0.0, 0.0)).model
    rehearsal = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))
    assert _thread_parameters(rehearsal) == _thread_parameters(training)


def test_policy_scene_nut_is_grippable_and_the_bolt_is_only_visual():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))

    nut = model.geom("m24_nut_geom")
    assert nut.contype[0] and nut.conaffinity[0]
    bolt = model.geom("m24_bolt_geom")
    assert not bolt.contype[0] and not bolt.conaffinity[0], (
        "the thread coupling holds the nut on the bolt; meshing the hulls fights it"
    )

    # MuJoCo takes the element-wise maximum of two geoms' friction, so the pads
    # and the nut both have to carry training's value for a contact to see it.
    pads = {
        round(float(model.geom_friction[g][0]), 9)
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.body(model.geom_bodyid[g]).name.startswith(("thumb", "index"))
    }
    assert pads == {scene.NUT_FRICTION[0]}, f"finger pad friction is {pads}"
    assert round(float(nut.friction[0]), 9) == scene.NUT_FRICTION[0]


def test_policy_scene_thread_turns_the_nut_down_the_bolt():
    """A tightening turn descends exactly one thread pitch, and nothing else moves it."""

    mujoco = pytest.importorskip("mujoco")
    from policy_rollout import mujoco_threading_env as te

    model = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))
    data = mujoco.MjData(model)
    axial = model.jnt_qposadr[model.joint("nut_axial").id]
    twist = model.jnt_qposadr[model.joint("nut_twist").id]
    twist_dof = model.jnt_dofadr[model.joint("nut_twist").id]

    def run(twist_torque, seconds=1.0):
        mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
        mujoco.mj_forward(model, data)
        start = (data.qpos[axial], data.qpos[twist])
        for _ in range(int(seconds / model.opt.timestep)):
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[twist_dof] = twist_torque
            mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0.0
        return data.qpos[axial] - start[0], data.qpos[twist] - start[1]

    # The keyframe is the nut's start pose on the bolt.
    mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
    assert data.qpos[axial] == pytest.approx(te.NUT_START_AXIAL_OFFSET_M)
    assert data.qpos[twist] == pytest.approx(0.0)

    # Unloaded it stays put: the hinge's Coulomb term plus the coupling make the
    # pair self-locking against its own inertia.
    rise, turn = run(0.0)
    assert abs(turn) < 1e-6 and abs(rise) < 1e-6

    # A tightening torque is negative about the hinge's world-up z axis, and it
    # has to drive the nut DOWN by pitch per turn.
    rise, turn = run(-0.02)
    assert turn < -math.radians(90.0), "a 0.02 N m torque should turn the nut"
    assert rise < 0.0, "tightening must lower the nut onto the bolt"
    assert rise / turn == pytest.approx(
        fo.M24_THREAD_PITCH / (2.0 * math.pi), rel=0.01
    )
    # ForgeUltra's sign convention: tightening is positive progress.
    assert te.THREADING_DIRECTION_SIGN * turn > 0.0


def test_policy_scene_thread_hold_clamps_the_nut():
    """The clamp the plugin applies for release/return actually holds.

    ``inspire_franka_sim``'s thread pair plugin freezes the coupling at the
    reached axial position and pins the twist with ``thread_hold``, mirroring
    ``ThreadingScene._apply_thread_drive``. This pins the semantics that makes
    that work, so a scene edit cannot quietly break the hold: with it engaged, a
    torque that otherwise spins the nut most of a turn must not move it.
    """

    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(str(scene.SIM_MJCF_DIR / scene.SCENE_NAME))
    data = mujoco.MjData(model)
    axial = model.jnt_qposadr[model.joint("nut_axial").id]
    twist = model.jnt_qposadr[model.joint("nut_twist").id]
    twist_dof = model.jnt_dofadr[model.joint("nut_twist").id]
    coupling = model.equality("thread_coupling").id
    hold = model.equality("thread_hold").id
    slope = float(model.eq_data[coupling][1])

    def run(engage_hold):
        mujoco.mj_resetDataKeyframe(model, data, model.key(scene.KEYFRAME_NAME).id)
        mujoco.mj_forward(model, data)
        if engage_hold:
            model.eq_data[coupling][0] = data.qpos[axial]
            model.eq_data[coupling][1] = 0.0
            model.eq_data[hold][0] = data.qpos[twist]
            data.eq_active[hold] = 1
        start = (data.qpos[axial], data.qpos[twist])
        for _ in range(int(1.0 / model.opt.timestep)):
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[twist_dof] = -0.05
            mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0.0
        moved = (data.qpos[axial] - start[0], data.qpos[twist] - start[1])
        model.eq_data[coupling][0] = scene.te.NUT_START_AXIAL_OFFSET_M
        model.eq_data[coupling][1] = slope
        data.eq_active[hold] = 0
        return moved

    free_rise, free_turn = run(engage_hold=False)
    assert abs(free_turn) > math.radians(360.0), "the torque should spin an unheld nut"

    held_rise, held_turn = run(engage_hold=True)
    assert abs(held_turn) < math.radians(1.0), (
        f"held nut turned {math.degrees(held_turn):.3f} deg under the same torque"
    )
    assert abs(held_rise) < 1e-4
    assert abs(free_rise) > 1e-3
