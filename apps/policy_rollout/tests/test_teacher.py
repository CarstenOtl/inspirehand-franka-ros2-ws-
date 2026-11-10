"""The exp_20 teacher's loader and observation, without ROS or a simulator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from policy_rollout import mujoco_threading_env as te
from policy_rollout import teacher as tc

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"
RUN = "m24-fr3v2-exp20-run01-hwfit-2026-11-09_23-43-16"
FOUND = sorted(CHECKPOINTS.glob(f"m24-fr3v2-exp20-run01-hwfit-extended-*/{RUN}/nn/last_*.pth"))
needs_checkpoint = pytest.mark.skipif(not FOUND, reason="exp_20 checkpoint not on disk")
PROFILE = fo.TASK_PROFILES["mimic_windback45_tilt30"]


def reset_observation(forces=(0.0, 0.0)):
    """The first observation ros-sim builds at the trained reset."""

    from policy_rollout.hardware import (
        TrainingHandKinematics,
        grasp_frame_from_tips,
        grasp_z_transport_at_reset,
    )
    from policy_rollout.mujoco_threading_env import seed_action_history

    flange, flange_quat, thumb, index = TrainingHandKinematics(PROFILE).frames_in_world(
        PROFILE.arm_reset_joints, PROFILE.entry_pinch
    )
    z_transport = grasp_z_transport_at_reset(thumb, index, flange, flange_quat)
    position, quaternion = grasp_frame_from_tips(thumb, index, flange_quat, z_transport)
    previous = seed_action_history(
        fo.GraspFrameState(position, quaternion, np.zeros(3), np.zeros(3), np.zeros((6, 7))), PROFILE
    )
    episode = tc.TeacherEpisode(15.0, PROFILE.nut_spawn_yaw_deg)
    return episode, episode.observe(
        grasp_position_world=position,
        grasp_quaternion_world=quaternion,
        axial_m=te.NUT_START_AXIAL_OFFSET_M,
        twist_rad=0.0,
        turn_progress_rad=0.0,
        fingertip_force_n=forces,
        arm_joint_torque_nm=np.zeros(7),
        previous_filtered_action=previous,
    )


def test_observation_layout_is_46_values():
    _, state = reset_observation()
    assert state.observation.shape == (tc.TEACHER_OBS_DIM,)
    assert np.isfinite(state.observation).all()
    # w and z of the fingertip quaternion are zeroed, as in training.
    assert state.observation[3] == 0.0 and state.observation[6] == 0.0


def test_target_is_one_clockwise_turn_down_the_bolt():
    start = np.array([0.61, 0.0, 0.11])
    q_start = fo.quat_from_euler_xyz(0.0, 0.0, math.radians(75.0))
    position, quaternion = tc.threading_target(start, q_start)
    assert position[2] == pytest.approx(0.11 - 0.003 / 6.0)
    expected = fo.quat_from_euler_xyz(0.0, 0.0, math.radians(15.0))
    assert abs(float(np.dot(quaternion, expected))) == pytest.approx(1.0)


def test_grip_latches_after_the_hold_and_success_needs_the_target():
    episode, _ = reset_observation()
    kwargs = dict(
        grasp_position_world=fo.BOLT_TIP_POSITION,
        grasp_quaternion_world=np.array([0.0, 1.0, 0.0, 0.0]),
        axial_m=0.0,
        twist_rad=0.0,
        fingertip_force_n=(5.0, 5.0),
        arm_joint_torque_nm=np.zeros(7),
        previous_filtered_action=np.zeros(9),
    )
    for _ in range(tc.GRASP_HOLD_STEPS - 1):
        assert not episode.observe(turn_progress_rad=0.0, **kwargs).grasped_once
    assert episode.observe(turn_progress_rad=0.0, **kwargs).grasped_once
    assert not episode.success
    for _ in range(tc.SUCCESS_HOLD_STEPS):
        state = episode.observe(turn_progress_rad=tc.TARGET_TURN_RAD, **kwargs)
    assert state.success


def test_only_pth_checkpoints_are_teachers():
    assert tc.is_teacher_checkpoint("run/nn/last_ForgeTg2_ep_200.pth")
    assert not tc.is_teacher_checkpoint("run/nn/checkpoint.pt")


def test_exp20_resolves_to_the_windback_profile():
    path = f"/x/{RUN}/nn/last_ForgeTg2_ep_200_rew_517.838.pth"
    assert fo.task_profile_for_checkpoint(path) is PROFILE


@needs_checkpoint
def test_checkpoint_loads_and_runs_with_memory():
    runner = tc.TeacherPolicyRunner(FOUND[0])
    _, state = reset_observation()
    first = runner.step(state.observation).numpy()
    second = runner.step(state.observation).numpy()
    assert first.shape == (9,)
    assert not np.allclose(first, second)  # the LSTM state advanced
    runner.reset()
    assert np.allclose(runner.step(state.observation).numpy(), first)


@needs_checkpoint
def test_reset_observation_sits_inside_training_statistics():
    """A wrong frame, sign or scale at reset shows up as a large z-score."""

    runner = tc.TeacherPolicyRunner(FOUND[0])
    _, state = reset_observation()
    mean = runner.model.obs_mean.cpu().numpy()
    std = np.sqrt(runner.model.obs_var.cpu().numpy() + 1.0e-5)
    z = (state.observation - mean) / std
    # Torque at rest and the force channels are left out: gravity compensation
    # and contact make their training distributions one-sided.
    geometry = np.r_[0:3, 4:6, 20:24, 24:31, 36:37]
    assert np.abs(z[geometry]).max() < 5.0, np.round(z, 2)
