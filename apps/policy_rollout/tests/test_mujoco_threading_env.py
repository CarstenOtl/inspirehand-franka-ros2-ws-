"""The MuJoCo threading scene's arm dynamics switch."""

import numpy as np
import pytest

from policy_rollout import forge_osc as fo


def _scene(**kwargs):
    pytest.importorskip("mujoco")
    from policy_rollout.mujoco_threading_env import ThreadingScene

    nut_quat = fo.quat_from_euler_xyz(0.0, 0.0, np.deg2rad(30.0))
    return ThreadingScene(nut_quat_wxyz=nut_quat, **kwargs)


def test_the_arm_is_menageries_fr3_v2():
    # Link masses, joint ranges and armature from inspire_franka_sim's fr3v2.xml.
    scene = _scene()
    m = scene.model
    masses = [m.body(f"fr3_link{i}").mass[0] for i in range(1, 8)]
    np.testing.assert_allclose(masses, [2.4377, 2.2375, 2.18579, 2.17587, 2.17482, 1.56762, 0.682876])
    np.testing.assert_allclose(m.jnt_range[m.joint("fr3_joint6").id], [0.4398, 4.6216])
    np.testing.assert_allclose(m.dof_armature[scene.arm_dofs], 0.195)
    assert m.mesh("fr3v2_link6_9").id >= 0


def test_arm_friction_scale_sets_the_menagerie_values_on_the_arm_dofs():
    from policy_rollout.mujoco_threading_env import (
        MENAGERIE_ARM_DAMPING,
        MENAGERIE_ARM_FRICTIONLOSS,
    )

    default = _scene()
    assert default.arm_friction_scale == 1.0
    np.testing.assert_allclose(
        default.model.dof_frictionloss[default.arm_dofs], MENAGERIE_ARM_FRICTIONLOSS
    )
    frictionless = _scene(arm_friction_scale=0.0)
    np.testing.assert_allclose(frictionless.model.dof_frictionloss[frictionless.arm_dofs], 0.0)
    np.testing.assert_allclose(frictionless.model.dof_damping[frictionless.arm_dofs], 0.0)

    half = _scene(arm_friction_scale=0.5)
    np.testing.assert_allclose(
        half.model.dof_frictionloss[half.arm_dofs], 0.5 * np.asarray(MENAGERIE_ARM_FRICTIONLOSS)
    )
    np.testing.assert_allclose(
        half.model.dof_damping[half.arm_dofs], 0.5 * np.asarray(MENAGERIE_ARM_DAMPING)
    )
    # Only the arm: the hand and thread-pair dofs keep their own values.
    other = [i for i in range(half.model.nv) if i not in half.arm_dofs]
    np.testing.assert_allclose(
        half.model.dof_frictionloss[other], frictionless.model.dof_frictionloss[other]
    )
    with pytest.raises(ValueError):
        _scene(arm_friction_scale=-1.0)


class _ThreadStub:
    """The scene calls ``CyclicCoordinator.observe`` makes, on a settable twist."""

    def __init__(self):
        self.twist_angle = 0.0
        self.twist_reference = 0.0
        self.held = False

    @property
    def turn_progress_rad(self):
        return fo.THREADING_DIRECTION_SIGN * (self.twist_angle - self.twist_reference)

    def rebase_turn_reference(self):
        self.twist_reference = self.twist_angle

    def hold_thread(self):
        self.held = True

    def release_thread_hold(self):
        self.held = False


def test_the_cyclic_coordinator_aims_each_turn_at_the_60_degree_grid():
    pytest.importorskip("mujoco")
    from policy_rollout.mujoco_threading_env import CyclicCoordinator

    grasp = fo.GraspFrameState(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), np.zeros((6, 7))
    )
    coordinator = CyclicCoordinator(
        max_cycles=3,
        reset_grasp_pos=np.zeros(3),
        reset_grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        reset_hand=np.zeros(3),
    )
    scene = _ThreadStub()

    def observe(turn_deg):
        scene.twist_angle = fo.THREADING_DIRECTION_SIGN * np.deg2rad(turn_deg)
        return coordinator.observe(
            step=0, turn_progress_rad=0.0, grasp=grasp, hand_joints=np.zeros(3), scene=scene
        )

    assert observe(54.0) is None
    assert observe(55.5) == "release_started"
    # The nut stops at 56 deg; the return completes there.
    events = [observe(56.0) for _ in range(coordinator.waypoint_steps * 4 + coordinator.return_steps)]
    assert events[-1] == "cycle_completed"
    assert np.rad2deg(coordinator.target_turn_rad) == pytest.approx(64.0)
    # Cycle 2 releases 5 deg before its 64 deg target, i.e. at 115 deg in total.
    assert observe(114.0) is None
    assert observe(115.5) == "release_started"


def test_a_pinch_on_the_nut_reads_as_compression_on_both_pads():
    from policy_rollout.mujoco_threading_env import seed_action_history

    scene = _scene()
    scene.reset(
        fo.FRANKA_ARM_RESET_JOINTS_M24,
        {**fo.THREADING_GRASP_POSTURE, "middle_joint_0": 1.333, "ring_joint_0": 1.333, "little_joint_0": 1.333},
    )
    assert scene.fingertip_force_n().tolist() == [0.0, 0.0]
    thumb, _ = scene.body_pose(scene.thumb_tip)
    index, _ = scene.body_pose(scene.index_tip)
    flange_pos, flange_quat = scene.body_pose(scene.flange)
    z_transport = fo.reset_z_transport(thumb, index, flange_pos, flange_quat, tilt_deg=0.0)
    action = seed_action_history(scene.grasp_state(z_transport))
    peak = np.zeros(2)
    # Close thumb and index from the grasp onto the nut.
    for squeeze in np.linspace(0.0, 1.4, 240):
        action[6:9] = squeeze
        scene.control_tick(action, z_transport)
        peak = np.maximum(peak, scene.fingertip_force_n())
    # Without the calibrated 150 deg yaw the thumb's squeeze lands on its +X
    # and reads zero.
    assert peak[0] > 0.3 and peak[1] > 0.3
    # Holding the pose, the OSC torque is small next to the effort limits.
    assert np.all(np.abs(scene.arm_joint_torque_nm()) < [8.7] * 4 + [1.2] * 3)


def test_the_table_is_threading_v2s_legless_top_in_front_of_the_robot():
    mujoco = pytest.importorskip("mujoco")
    scene = _scene()
    m, d = scene.model, scene.data
    mujoco.mj_forward(m, d)
    top = m.geom("table_top").id
    centre, half = d.geom_xpos[top], m.geom_size[top]
    # Raised to the bolt holder's foot, 43.5 mm under the bolt base.
    np.testing.assert_allclose(centre, [0.335, 0.0, 0.0065 - 0.015], atol=1e-9)
    holder = m.body("bolt_holder").id
    np.testing.assert_allclose(d.xpos[holder], d.xpos[m.body("m24_bolt").id], atol=1e-9)
    np.testing.assert_allclose(half, [0.4, 0.6, 0.015], atol=1e-9)
    # Near ledge 46.5 cm from the robot axis (x = 1.2), bolt 12.5 cm inside it.
    assert 1.2 - (centre[0] + half[0]) == pytest.approx(0.465)
    assert centre[0] + half[0] - d.xpos[m.body("m24_bolt").id][0] == pytest.approx(0.125)
    assert not any(m.geom(g).name.startswith("table_leg") for g in range(m.ngeom))
    assert m.mat_texid[m.geom_matid[top]][mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0


def test_thread_is_one_way_like_isaac():
    from policy_rollout import forge_osc as fo
    from policy_rollout.mujoco_threading_env import NUT_START_AXIAL_OFFSET_M, ThreadingScene

    scene = ThreadingScene(nut_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    scene.reset(fo.FRANKA_ARM_RESET_JOINTS_M24, fo.THREADING_GRASP_POSTURE, settle_s=0.0)
    slope = scene.coupling_slope
    assert slope > 0.0

    # 20 N either way along the bolt deflects the coupling and does not turn the nut.
    for force in (20.0, -20.0):
        scene.data.qfrc_applied[:] = 0.0
        scene.data.qfrc_applied[scene.axial_dof] = force
        start = scene.twist_angle
        for _ in range(60):
            scene._step_physics()
        assert scene.twist_angle == pytest.approx(start, abs=1e-6)
        assert scene.axial_position == pytest.approx(NUT_START_AXIAL_OFFSET_M + slope * start, abs=2e-4)

    # A torque about the bolt turns it, and the axial position follows the twist.
    scene.data.qfrc_applied[:] = 0.0
    scene.data.qfrc_applied[scene.twist_dof] = -0.02
    for _ in range(30):
        scene._step_physics()
    assert scene.twist_angle < -0.5
    assert scene.axial_position == pytest.approx(NUT_START_AXIAL_OFFSET_M + slope * scene.twist_angle, abs=5e-4)

    # The hold pins both.
    scene.data.qfrc_applied[:] = 0.0
    scene.hold_thread()
    held = (scene.axial_position, scene.twist_angle)
    scene.data.qfrc_applied[scene.twist_dof] = -0.02
    for _ in range(30):
        scene._step_physics()
    assert scene.twist_angle == pytest.approx(held[1], abs=2e-3)
    assert scene.axial_position == pytest.approx(held[0], abs=2e-4)
