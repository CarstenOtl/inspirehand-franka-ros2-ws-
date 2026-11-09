import math

import numpy as np
import pytest

from policy_rollout import DEFAULT_CHECKPOINT
from policy_rollout import forge_osc as fo
from policy_rollout.rh56_hand import RegisterPinchActions, command_open_ratios

RH56_RUN = "m24-dp3-exp17r01-windback45tilt30-rollback3-noclock-force-nohv-da-s0-20261106-132547-951759"
DRIVER_HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)


class LinearHandMap:
    """Stand-in for RH56HandMap: register 1000 at 0 rad, 0 at the end of the stroke."""

    STROKE = {"thumb_joint_0": 1.308, "thumb_joint_1": 0.716, "index_joint_0": 1.470}

    def register_from_rad(self, key, rad):
        return float(np.clip(1000.0 * (1.0 - rad / self.STROKE[key]), 0.0, 1000.0))

    def rad_from_register(self, key, register):
        return float(self.STROKE[key] * (1.0 - np.clip(register, 0.0, 1000.0) / 1000.0))

    def open_ratio_from_rad(self, key, rad):
        return self.register_from_rad(key, rad) / 1000.0


def test_checkpoints_resolve_to_their_task_profile():
    nested = f"checkpoints/{RH56_RUN}-20261106T141800Z-1-001/{RH56_RUN}/nn/checkpoint.pt"
    assert fo.task_profile_for_checkpoint(nested) is fo.RH56_WINDBACK45_TILT30_PROFILE
    assert fo.task_profile_for_checkpoint(DEFAULT_CHECKPOINT) is fo.MIMIC_M24_PROFILE
    assert fo.resolve_task_profile("auto", nested) is fo.RH56_WINDBACK45_TILT30_PROFILE
    assert fo.resolve_task_profile("mimic_m24", nested) is fo.MIMIC_M24_PROFILE
    with pytest.raises(ValueError, match="unknown task profile"):
        fo.resolve_task_profile("rh57", nested)


@pytest.mark.parametrize(
    ("run", "export"),
    (
        ("m24-dp3-exp19r01-windback45tilt30-clock-force-nohv-da-s0-20261109-105240-553825", "20261109T123227Z"),
        ("m24-dp3-exp19r01-proprio-friction-rollback3-clock-force-nohv-da-s0-20261109-143521-315193", "20261109T170730Z"),
    ),
)
def test_exp19_is_the_old_hand_at_the_wound_reset(run, export):
    nested = f"checkpoints/{run}-{export}-1-001/{run}/nn/checkpoint.pt"
    profile = fo.task_profile_for_checkpoint(nested)
    assert profile is fo.MIMIC_WINDBACK45_TILT30_PROFILE
    # The old hand: mimic, its URDF pinch ranges, joint-angle actions about the grasp.
    assert profile.hand == "mimic"
    assert profile.pinch_ranges == fo.PINCH_RANGES
    grasp = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    assert fo.pinch_targets(np.zeros(9), profile=profile) == pytest.approx(grasp)
    # The wound, tilted reset of exp_17, whichever hand holds it.
    wound = fo.RH56_WINDBACK45_TILT30_PROFILE
    assert profile.arm_reset_joints == wound.arm_reset_joints
    assert profile.ee_base_orn_euler == wound.ee_base_orn_euler
    assert profile.yaw_wrap_upper_deg == 280.0
    assert profile.nut_spawn_yaw_deg == 75.0


def test_mimic_profile_keeps_the_module_constants():
    profile = fo.MIMIC_M24_PROFILE
    assert np.allclose(profile.arm_reset_joints, fo.FRANKA_ARM_RESET_JOINTS_M24)
    assert profile.ee_base_orn_euler == fo.EE_BASE_ORN_EULER
    assert profile.yaw_wrap_upper_deg == 235.0
    assert profile.entry_pinch == fo.ISAAC_ENTRY_PINCH


def test_wound_profile_turns_the_orientation_action_frame_45_deg():
    unused = fo.GraspFrameState(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3), np.zeros((6, 7)))
    action = np.zeros(9)
    mimic = fo.decode_action_target(action, unused, clip=False)
    wound = fo.decode_action_target(action, unused, clip=False, profile=fo.RH56_WINDBACK45_TILT30_PROFILE)
    turn = fo.quat_mul(wound.quat, fo.quat_conjugate(mimic.quat))
    # base(yaw 45) x a == R_z(45) x base(yaw 0) x a: a pure +45 deg turn about world z.
    expected = fo.quat_from_angle_axis(math.pi / 4.0, [0.0, 0.0, 1.0])
    assert abs(float(np.dot(turn, expected))) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(wound.pos, mimic.pos)


def test_yaw_seam_follows_the_profile():
    assert fo.wrap_yaw(math.radians(240.0)) == pytest.approx(math.radians(-120.0))
    assert fo.wrap_yaw(math.radians(240.0), 280.0) == pytest.approx(math.radians(240.0))


def test_mimic_pinch_decoder_refuses_an_rh56_profile():
    with pytest.raises(ValueError, match="mimic"):
        fo.pinch_targets(np.zeros(9), profile=fo.RH56_WINDBACK45_TILT30_PROFILE)


def test_register_pinch_actions_hold_the_grasp_and_span_the_stroke():
    hand_map = LinearHandMap()
    actions = RegisterPinchActions(hand_map, fo.RH56_WINDBACK45_TILT30_PROFILE)
    grasp = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    # Zero action holds the grasp register, i.e. the grasp posture.
    assert actions.pinch_targets(np.zeros(9)) == pytest.approx(grasp)
    # +1 closes by half the stroke (register falls), clamped at the closed stop.
    closing = actions.registers(np.r_[np.zeros(6), np.ones(3)])
    assert np.all(closing <= actions.grasp)
    assert np.all(closing >= actions.reg_min)
    assert actions.half_span == pytest.approx(np.full(3, 500.0))
    assert actions.registers(np.r_[np.zeros(6), np.full(3, -5.0)]) == pytest.approx(actions.reg_max)


def test_register_pinch_actions_need_an_rh56_profile():
    with pytest.raises(ValueError, match="rh56"):
        RegisterPinchActions(LinearHandMap(), fo.MIMIC_M24_PROFILE)


def test_rh56_command_tucks_the_support_fingers_at_register_100():
    ratios = dict(
        zip(DRIVER_HAND_JOINTS, command_open_ratios(LinearHandMap(), [0.0, 0.716, 0.735], DRIVER_HAND_JOINTS))
    )
    assert ratios["pinky_proximal_joint"] == ratios["ring_proximal_joint"] == ratios["middle_proximal_joint"] == 0.1
    assert ratios["thumb_proximal_yaw_joint"] == pytest.approx(1.0)
    assert ratios["thumb_proximal_pitch_joint"] == pytest.approx(0.0)
    assert ratios["index_proximal_joint"] == pytest.approx(0.5)


def test_vendored_map_and_joint_table_agree_on_every_leader():
    from policy_rollout.rh56_hand import CHANNEL_OF_LEADER, load_hand_map, load_joint_table

    hand_map, table = load_hand_map(), load_joint_table()
    for leader, channel in CHANNEL_OF_LEADER.items():
        for register in (0.0, 137.5, 500.0, 912.0, 1000.0):
            rad = float(hand_map.rad_from_register(leader, register))
            assert table.joints_at(channel, register)[leader] == pytest.approx(rad, abs=1e-6)
            assert table.register_of_leader(channel, rad) == pytest.approx(register, abs=1e-3)


def test_rh56_followers_follow_the_linkage_not_the_mimic():
    from policy_rollout.rh56_hand import expand_followers, load_joint_table

    table = load_joint_table()
    joints = expand_followers({"thumb_joint_0": 1.185, "thumb_joint_1": 0.716, "index_joint_0": 0.0})
    assert set(joints) == {
        "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
        "index_joint_0", "index_joint_1", "middle_joint_0", "middle_joint_1",
        "ring_joint_0", "ring_joint_1", "little_joint_0", "little_joint_1",
    }
    assert joints["thumb_joint_0"] == pytest.approx(1.185, abs=1e-6)
    # Fully bent thumb: the linkage's distal joints, not 1.1425x / 0.7508x chains.
    assert joints["thumb_joint_2"] == pytest.approx(0.7873, abs=1e-3)
    assert joints["thumb_joint_3"] == pytest.approx(0.8106, abs=1e-3)
    assert joints["index_joint_1"] == pytest.approx(0.0, abs=1e-6)
    # No leader given: the support fingers tuck at register 100.
    assert joints["middle_joint_0"] == pytest.approx(table.joints_at("middle", 100.0)["middle_joint_0"])
    # Leaders past the stroke clamp to it.
    assert expand_followers({"index_joint_0": 2.0})["index_joint_0"] == pytest.approx(1.47, abs=1e-6)


def test_register_pinch_actions_on_the_real_map_hold_the_grasp():
    from policy_rollout.rh56_hand import load_hand_map

    actions = RegisterPinchActions(load_hand_map(), fo.RH56_WINDBACK45_TILT30_PROFILE)
    grasp = [fo.THREADING_GRASP_POSTURE[name] for name in fo.PINCH_JOINTS]
    assert actions.pinch_targets(np.zeros(9)) == pytest.approx(grasp, abs=1e-6)
    assert np.all(actions.reg_min == pytest.approx(0.0, abs=1.0))
    assert np.all(actions.reg_max == pytest.approx(1000.0, abs=1.0))


def test_rh56_readout_goes_from_registers_to_leader_angles():
    """The driver's ~/state (open ratios = registers / 1000) is what an RH56 student observes through."""

    from types import SimpleNamespace

    from policy_rollout.hardware import HAND_JOINTS, physical_hand_state_to_policy
    from policy_rollout.rh56_hand import load_hand_map, register_state_to_open_ratios

    hand_map = load_hand_map()
    pinch = [1.185078562, 0.050614548, 0.214675498]
    ratios = command_open_ratios(hand_map, pinch, DRIVER_HAND_JOINTS)
    # The hand reports in channel order 1..6 = pinky, ring, middle, index, thumb bend, thumb rotation.
    state = SimpleNamespace(name=["1", "2", "3", "4", "5", "6"], position=ratios)
    open_ratios = register_state_to_open_ratios(state, HAND_JOINTS)
    q, dq = physical_hand_state_to_policy(open_ratios, np.zeros(6), thumb_overlay=False, hand_map=hand_map)
    assert q == pytest.approx(pinch, abs=1e-6)
    assert dq.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="overlay"):
        physical_hand_state_to_policy(open_ratios, np.zeros(6), thumb_overlay=True, hand_map=hand_map)
    with pytest.raises(RuntimeError, match="six RH56 channels"):
        register_state_to_open_ratios(SimpleNamespace(name=["1"], position=[0.5]), HAND_JOINTS)



def test_linear_hand_map_is_the_drivers_linear_map_for_the_leaders():
    from inspire_hand_driver import kinematics as kin

    from policy_rollout.rh56_hand import LinearHandMap

    hand_map = LinearHandMap()
    yaw = kin.dof_index("thumb_proximal_yaw_joint")
    for register in (0.0, 69.0, 500.0, 1000.0):
        rad = hand_map.rad_from_register("thumb_joint_0", register)
        assert rad == pytest.approx(kin.open_ratio_to_rad(yaw, register / 1000.0))
        assert hand_map.register_from_rad("thumb_joint_0", rad) == pytest.approx(register, abs=1e-6)
    # A student angle past the linear stroke is sent as the closed stop.
    assert hand_map.register_from_rad("thumb_joint_0", 1.308) == pytest.approx(0.0)
    assert hand_map.rad_from_register("thumb_joint_1", 0.0) == pytest.approx(
        kin.open_ratio_to_rad(kin.dof_index("thumb_proximal_pitch_joint"), 0.0)
    )


def test_linear_hand_model_round_trips_the_student_pinch_through_the_driver():
    from types import SimpleNamespace

    from policy_rollout.hardware import HAND_JOINTS, physical_hand_state_to_policy
    from policy_rollout.rh56_hand import LinearHandMap, register_state_to_open_ratios

    hand_map = LinearHandMap()
    pinch = [1.185078562, 0.050614548, 0.214675498]
    ratios = dict(zip(DRIVER_HAND_JOINTS, command_open_ratios(hand_map, pinch, DRIVER_HAND_JOINTS)))
    # The student's angle is the linear angle the driver gets: the same open ratio
    # the old hand model would command for it.
    assert ratios["index_proximal_joint"] == pytest.approx(1.0 - 0.214675498 / 1.333, abs=1e-6)
    assert ratios["middle_proximal_joint"] == 0.1
    state = SimpleNamespace(name=["1", "2", "3", "4", "5", "6"], position=[ratios[j] for j in DRIVER_HAND_JOINTS])
    q, _ = physical_hand_state_to_policy(
        register_state_to_open_ratios(state, HAND_JOINTS), np.zeros(6), thumb_overlay=False, hand_map=hand_map
    )
    assert q == pytest.approx(pinch, abs=1e-6)


def test_mimic_followers_for_the_old_hand_model():
    from policy_rollout.rh56_hand import expand_followers_mimic

    joints = expand_followers_mimic({"thumb_joint_0": 1.2, "thumb_joint_1": 0.4, "index_joint_0": 0.5})
    assert joints["thumb_joint_2"] == pytest.approx(0.4 * 1.1425)
    assert joints["thumb_joint_3"] == pytest.approx(0.4 * 1.1425 * 0.7508)
    assert joints["index_joint_1"] == pytest.approx(0.5 * 1.1169)
    assert joints["middle_joint_0"] == pytest.approx(1.2, abs=1e-3)  # register 100 on the linear map


def test_live_tips_follow_the_hand_model_while_the_reference_stays_the_training_hand():
    from policy_rollout.hardware import TrainingHandKinematics

    profile = fo.RH56_WINDBACK45_TILT30_PROFILE
    training = TrainingHandKinematics(profile)
    old = TrainingHandKinematics(profile, followers="mimic")
    bent = [1.0, 0.45, 0.3]
    linkage, mimic = training._hand_joint_angles(bent), old._hand_joint_angles(bent)
    assert mimic["thumb_joint_2"] == pytest.approx(0.45 * 1.1425)
    assert linkage["thumb_joint_2"] != pytest.approx(mimic["thumb_joint_2"], abs=1e-3)
    # At equal leader angles the two follower models put the tips within a few mm;
    # the models differ mainly in which register reaches which leader angle.
    thumb_linkage, _ = training.tips_in_flange(bent)
    thumb_mimic, _ = old.tips_in_flange(bent)
    assert np.linalg.norm(thumb_linkage - thumb_mimic) < 5e-3
    with pytest.raises(ValueError, match="followers"):
        TrainingHandKinematics(profile, followers="linear")
