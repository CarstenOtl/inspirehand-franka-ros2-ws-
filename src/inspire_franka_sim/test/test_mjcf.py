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
    "inspire_franka_flange_torque_scene.xml",
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
# The training hand's couplings (forgeUltra's official URDF). The thumb's
# distal joint follows the intermediate, not the pitch.
COUPLINGS = {
    "index_intermediate_joint": ("index_proximal_joint", 1.1169, 0.0),
    "middle_intermediate_joint": ("middle_proximal_joint", 1.1169, 0.0),
    "ring_intermediate_joint": ("ring_proximal_joint", 1.1169, 0.0),
    "pinky_intermediate_joint": ("pinky_proximal_joint", 1.1169, 0.0),
    "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.1425, 0.0),
    "thumb_distal_joint": ("thumb_intermediate_joint", 0.7508, 0.0),
}

# Must match hand_bench_xyz in
# inspire_franka_description/urdf/inspire_franka.urdf.xacro.
BENCH_XYZ = (0.45, -0.35, 0.0)

# The flange -> palm transform: the 10 mm adapter flange, then a 180-degree
# clocking about the flange axis (wxyz). The palm frame is hand_mount's, as in
# forgeUltra, which mounts the palm itself with exactly this transform.
FLANGE_TO_PALM_XYZ = (0.0, 0.0, 0.010)
FLIPPED_FLANGE_TO_PALM_QUAT = (0.0, 0.0, 0.0, 1.0)


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
    # The palm collides as the convex hull of its shell, which overlaps the
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

    # Both frames are body origins rigidly welded in this scene, so the palm's
    # offset is fixed in the flange frame.
    rotation = data.xmat[flange].reshape(3, 3)
    offset = rotation.T @ (data.xpos[palm] - data.xpos[flange])
    assert offset == pytest.approx(FLANGE_TO_PALM_XYZ, abs=1e-9)

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


# -- the hand is the one the policy was trained on ----------------------------


def test_the_hand_sits_on_the_flange_where_training_put_it():
    # Anchors measured on assets/fr3_inspirehand/fr3_inspirehand_replay.xml,
    # the scene policy_rollout.mujoco_scene builds the policy's grasp frame
    # from. If the description drifts, the plant and the policy's own
    # kinematics stop agreeing about where the fingers are.
    model = load("inspire_hand_on_flange.xml")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    flange = model.body("fr3_link8").id
    rotation = data.xmat[flange].reshape(3, 3)
    origin = data.xpos[flange]
    expected = {
        "thumb_proximal_yaw_joint": (-0.01696, -0.02045, 0.07635),
        "index_proximal_joint": (0.000285, -0.020275, 0.146946),
    }
    for name, reference in expected.items():
        anchor = rotation.T @ (data.xanchor[model.joint(name).id] - origin)
        assert anchor == pytest.approx(reference, abs=1.0e-5)


def test_the_left_hand_is_the_mirror_of_the_right():
    # There is no official left hand; the description mirrors the right one
    # across the hand_mount XZ plane. Any joint posture must then put every
    # tip at the right hand's position with y negated.
    right, left = load("inspire_hand_right.xml"), load("inspire_hand_left.xml")
    assert joint_names(right) == joint_names(left)
    data_r, data_l = mujoco.MjData(right), mujoco.MjData(left)
    for fraction in (0.0, 0.3, 0.7, 1.0):
        q = right.jnt_range[:, 0] + fraction * (right.jnt_range[:, 1] - right.jnt_range[:, 0])
        data_r.qpos[:] = q
        data_l.qpos[:] = q
        mujoco.mj_kinematics(right, data_r)
        mujoco.mj_kinematics(left, data_l)
        for tip in ("thumb_tip", "index_tip", "pinky_tip"):
            mirrored = data_r.xpos[right.body(tip).id] * (1.0, -1.0, 1.0)
            assert data_l.xpos[left.body(tip).id] == pytest.approx(mirrored, abs=1.0e-9)


def _hand_generator():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_hand_mjcf.py"
    spec = importlib.util.spec_from_file_location("make_hand_mjcf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_isaac_hand_carries_isaac_labs_training_dynamics():
    """inspire_hand_right_isaac.xml against make_hand_mjcf.py's ISAAC tables.

    Same geometry, masses and ranges as the reusable hand; no equality
    couplings (Isaac's mimic is target-level, in IsaacHandPlugin); one position
    servo per joint with Isaac's stiffness, damping and effort limit.
    """
    gen = _hand_generator()
    isaac = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "inspire_hand_right_isaac.xml"))
    plain = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "inspire_hand_right.xml"))

    assert isaac.neq == 0
    assert set(joint_names(isaac)) == set(joint_names(plain)) == set(gen.ISAAC_DRIVES)
    for name in joint_names(isaac):
        joint, reference = isaac.joint(name), plain.joint(name)
        assert tuple(joint.range) == pytest.approx(tuple(reference.range)), name
        assert isaac.body_mass[joint.bodyid[0]] == pytest.approx(plain.body_mass[reference.bodyid[0]])
        assert joint.armature[0] == pytest.approx(gen.ISAAC_ARMATURE[name]), name
        assert joint.damping[0] == 0.0 and joint.frictionloss[0] == 0.0, name

        stiffness, damping, effort = gen.ISAAC_DRIVES[name]
        actuator = isaac.actuator(name)
        assert actuator.trnid[0] == joint.id
        assert actuator.gainprm[0] == pytest.approx(stiffness), name
        assert tuple(actuator.biasprm[:3]) == pytest.approx((0.0, -stiffness, -damping)), name
        assert tuple(actuator.forcerange) == pytest.approx((-effort, effort)), name
        # The description's effort="1" must not survive as a joint-level clamp.
        assert tuple(isaac.jnt_actfrcrange[joint.id]) == pytest.approx((-effort, effort)), name
    assert isaac.nu == len(gen.ISAAC_DRIVES)
    # hand_joint_effort_limits_nm: the grip the old-hand students trained under.
    for name in ("thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "index_proximal_joint"):
        assert tuple(isaac.actuator(name).forcerange) == pytest.approx((-0.5, 0.5)), name


def test_the_isaac_hand_carries_isaac_labs_inertia_offset():
    """Every hand body, massless tips included, and nothing else gets the offset."""
    gen = _hand_generator()
    isaac = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "inspire_hand_right_isaac.xml"))
    plain = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "inspire_hand_right.xml"))

    offset_bodies = 0
    for body in range(isaac.nbody):
        name = isaac.body(body).name
        offset = 0.0 if name in ("world", "hand_mount") else gen.ISAAC_HAND_INERTIA_OFFSET
        offset_bodies += offset != 0.0
        expected = plain.body(name).inertia + offset
        # The generated MJCF keeps six significant digits.
        assert isaac.body(name).inertia == pytest.approx(expected, abs=2e-6), name
    assert offset_bodies == gen.ISAAC_HAND_BODY_COUNT


def test_the_isaac_flange_wrapper_is_the_flange_wrapper_on_the_isaac_hand():
    gen = _hand_generator()
    source = (MJCF_DIR / gen.FLANGE_WRAPPER).read_text()
    generated = (MJCF_DIR / "inspire_hand_on_flange_isaac.xml").read_text()
    assert generated == gen.isaac_flange_wrapper(source, "inspire_hand_right_isaac.xml"), (
        "regenerate with scripts/make_hand_mjcf.py --isaac"
    )
