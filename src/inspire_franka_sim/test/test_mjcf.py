"""Checks that the MJCF models and the URDF still describe the same robot.

`ros2_control` matches the description to the simulator **by joint name**, and
nothing checks that the two agree about anything else. A joint the URDF declares
and the MJCF does not is caught at startup; a joint that exists in both but sits
somewhere else, or is coupled differently, is not caught at all - the robot just
quietly does the wrong thing. These tests cover that gap.

They need the `mujoco` python module, and skip without it. The simulator itself
comes from `mujoco_vendor`, so a workspace can be built and these skipped
without the sim being broken.
"""

import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip("mujoco")

MJCF_DIR = Path(__file__).resolve().parents[1] / "mjcf"

SCENES = (
    "fr3_scene.xml",
    "inspire_hand_scene.xml",
    "inspire_franka_bench_scene.xml",
    "inspire_franka_flange_scene.xml",
)

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_DRIVEN = [
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
]
COUPLINGS = {
    "index_intermediate_joint": ("index_proximal_joint", 1.06399, -0.04545),
    "middle_intermediate_joint": ("middle_proximal_joint", 1.06399, -0.04545),
    "ring_intermediate_joint": ("ring_proximal_joint", 1.06399, -0.04545),
    "pinky_intermediate_joint": ("pinky_proximal_joint", 1.06399, -0.04545),
    "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.334, 0.0),
    "thumb_distal_joint": ("thumb_proximal_pitch_joint", 0.667, 0.0),
}

# Must match hand_bench_xyz in
# inspire_franka_description/urdf/inspire_franka.urdf.xacro.
BENCH_XYZ = (0.45, -0.35, 0.0)

# The composed flange -> palm rotation with 180-degree flange clocking (wxyz).
FLIPPED_FLANGE_TO_PALM_QUAT = (
    math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0
)


def load(scene: str):
    # MuJoCo resolves meshdir relative to the process working directory.
    cwd = os.getcwd()
    os.chdir(MJCF_DIR)
    try:
        return mujoco.MjModel.from_xml_path(scene)
    finally:
        os.chdir(cwd)


def joint_names(model):
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]


@pytest.mark.parametrize("side", ["right", "left"])
def test_reusable_hand_assets_have_no_pending_keyframes(side):
    root = ET.parse(MJCF_DIR / f"inspire_hand_{side}.xml").getroot()
    assert root.find("keyframe") is None


@pytest.mark.parametrize("scene", SCENES)
def test_every_scene_compiles(scene):
    load(scene)


@pytest.mark.parametrize(
    "scene,expected",
    [
        ("fr3_scene.xml", ARM_JOINTS),
        ("inspire_hand_scene.xml", list(COUPLINGS) + HAND_DRIVEN),
        ("inspire_franka_bench_scene.xml", ARM_JOINTS + HAND_DRIVEN + list(COUPLINGS)),
        ("inspire_franka_flange_scene.xml", ARM_JOINTS + HAND_DRIVEN + list(COUPLINGS)),
    ],
)
def test_scenes_contain_exactly_the_expected_joints(scene, expected):
    assert set(joint_names(load(scene))) == set(expected)


@pytest.mark.parametrize("scene", ["inspire_hand_scene.xml", "inspire_franka_bench_scene.xml"])
def test_only_the_driven_hand_joints_have_actuators(scene):
    model = load(scene)
    actuated = {
        joint_names(model)[model.actuator_trnid[i, 0]] for i in range(model.nu)
    }
    assert set(HAND_DRIVEN) <= actuated
    assert not (set(COUPLINGS) & actuated)


def test_the_couplings_are_the_ones_the_urdf_declares():
    # The same six multiplier/offset pairs live in the URDF's <mimic> tags and
    # in inspire_hand_driver.kinematics. If they drift apart, the simulator and
    # the driver report different fingertip positions for the same command.
    model = load("inspire_hand_scene.xml")
    names = joint_names(model)
    found = {}
    for i in range(model.neq):
        assert model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
        follower = names[model.eq_obj1id[i]]
        driver = names[model.eq_obj2id[i]]
        offset, multiplier = model.eq_data[i][0], model.eq_data[i][1]
        assert model.eq_data[i][2:5] == pytest.approx([0, 0, 0]), "coupling is not affine"
        found[follower] = (driver, multiplier, offset)
    assert found == pytest.approx(COUPLINGS)


def test_the_hand_rests_without_touching_itself():
    # The palm's collision geometry is a coarse envelope that overlaps the
    # finger roots; without the contact excludes the generator adds, the hand
    # starts jammed and will not reopen. Any contact at the rest pose means
    # those excludes have stopped covering it.
    model = load("inspire_hand_scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "open")
    )
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


@pytest.mark.parametrize(
    "scene,key", [("inspire_franka_bench_scene.xml", "start"),
                  ("inspire_franka_flange_scene.xml", "start")]
)
def test_the_combined_rest_pose_is_consistent(scene, key):
    model = load(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    )
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, "the robot starts in self-collision"

    names = joint_names(model)
    q = {n: data.qpos[model.jnt_qposadr[names.index(n)]] for n in names}
    # The arm at Franka's documented home pose...
    for joint, want in zip(ARM_JOINTS, [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]):
        assert q[joint] == pytest.approx(want, abs=1e-6)
    # ...and the hand open, followers included. A keyframe that zero-padded the
    # followers would start the model violating its own equality constraints.
    for follower, (driver, multiplier, offset) in COUPLINGS.items():
        assert q[follower] == pytest.approx(multiplier * q[driver] + offset, abs=1e-6)


def test_the_bench_hand_sits_where_the_urdf_puts_it():
    # MuJoCo places the hand from the scene file; TF places it from the URDF.
    # If these disagree, RViz and the simulator show the hand in different
    # places and nothing warns.
    model = load("inspire_franka_bench_scene.xml")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mount")
    assert body != -1, "the bench scene has no hand_mount body"
    assert model.body_pos[body] == pytest.approx(BENCH_XYZ, abs=1e-9)


def test_the_flange_mount_has_180_degree_clocking():
    model = load("inspire_franka_flange_scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link8")
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base_link")
    assert flange != -1
    assert palm != -1

    # The target mount has no translation. Compare world positions because both
    # frames are body origins and are rigidly welded in this scene.
    assert data.xpos[palm] == pytest.approx(data.xpos[flange], abs=1e-9)

    # q_relative = conjugate(q_flange) * q_palm. Quaternion signs are
    # equivalent, so compare the absolute dot product with the desired pose.
    fw, fx, fy, fz = data.xquat[flange]
    pw, px, py, pz = data.xquat[palm]
    relative = (
        fw * pw + fx * px + fy * py + fz * pz,
        fw * px - fx * pw - fy * pz + fz * py,
        fw * py + fx * pz - fy * pw - fz * px,
        fw * pz - fx * py + fy * px - fz * pw,
    )
    alignment = abs(sum(a * b for a, b in zip(
        relative, FLIPPED_FLANGE_TO_PALM_QUAT
    )))
    assert alignment == pytest.approx(1.0, abs=1e-6)


def test_every_actuator_is_a_plain_torque_source():
    # mujoco_ros2_control makes an `effort` command interface unsupported on a
    # <position> actuator, and an incompatible pairing is a hard error at
    # controller_manager startup rather than a warning. Everything must be a
    # <motor>: gaintype fixed, biastype none.
    model = load("inspire_franka_bench_scene.xml")
    for i in range(model.nu):
        assert model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
