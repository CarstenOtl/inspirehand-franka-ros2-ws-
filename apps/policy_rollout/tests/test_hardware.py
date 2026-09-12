from types import SimpleNamespace

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from policy_rollout.hardware import (
    GripCycleCoordinator,
    RgbdFrameSynchronizer,
    TrainingFrameAdapter,
    assert_policy_camera_frames,
    image_message_to_numpy,
    limit_cartesian_step,
    physical_hand_state_to_policy,
    policy_tool_transform,
    retarget_grasp_pose_to_controlled_pose,
)


def _stamped(seconds, value):
    whole = int(seconds)
    nanoseconds = round((seconds - whole) * 1.0e9)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=whole, nanosec=nanoseconds)
        ),
        value=value,
    )


def _image(array, encoding):
    return SimpleNamespace(
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        step=array.strides[0],
        data=array.tobytes(),
    )


def test_image_decoder_converts_bgr_and_depth_units():
    bgr = np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)
    rgb, units = image_message_to_numpy(_image(bgr, "bgr8"))
    assert units == "rgb"
    assert rgb.tolist() == [[[1, 2, 3], [4, 5, 6]]]

    depth_mm = np.array([[100, 200]], dtype=np.uint16)
    decoded_mm, units_mm = image_message_to_numpy(_image(depth_mm, "16UC1"))
    assert units_mm == "millimetres"
    assert decoded_mm.tolist() == [[100, 200]]

    depth_m = np.array([[0.1, 0.2]], dtype=np.float32)
    decoded_m, units_m = image_message_to_numpy(_image(depth_m, "32FC1"))
    assert units_m == "metres"
    assert decoded_m == pytest.approx(depth_m)


def test_image_decoder_ignores_row_padding():
    padded = np.array([[[1, 2, 3], [4, 5, 6], [99, 99, 99]]], dtype=np.uint8)
    message = SimpleNamespace(
        encoding="rgb8", height=1, width=2, step=9, data=padded.tobytes()
    )
    image, _ = image_message_to_numpy(message)
    assert image.tolist() == [[[1, 2, 3], [4, 5, 6]]]


def test_physical_hand_feedback_is_mapped_back_to_policy_coordinates():
    from inspire_hand_driver import command_overlays
    from inspire_hand_driver import kinematics as kin

    logical_q = np.array([1.08, 0.05, 0.22])
    logical_dq = np.array([0.4, -0.2, 0.1])
    physical_positions = np.array([1.0, 1.1, 1.2, logical_q[2], logical_q[1], 0.0])
    physical_velocities = np.array([0.0, 0.0, 0.0, logical_dq[2], logical_dq[1], 0.0])
    thumb_dof = kin.dof_index(command_overlays.THUMB_ABDUCTION_JOINT)
    logical_ratio = kin.rad_to_open_ratio(thumb_dof, logical_q[0])
    physical_ratio = command_overlays.apply_open_ratio_overlay(
        thumb_dof, logical_ratio
    )
    physical_positions[-1] = kin.open_ratio_to_rad(thumb_dof, physical_ratio)
    physical_velocities[-1] = (
        1.0 - command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    ) * logical_dq[0]

    q_policy, dq_policy = physical_hand_state_to_policy(
        physical_positions, physical_velocities
    )

    assert q_policy == pytest.approx(logical_q)
    assert dq_policy == pytest.approx(logical_dq)


def test_policy_camera_frames_must_all_be_registered_to_colour():
    def message(frame_id):
        return SimpleNamespace(header=SimpleNamespace(frame_id=frame_id))

    colour = "camera_color_optical_frame"
    assert_policy_camera_frames(message(colour), message(colour), message(colour), colour)
    with pytest.raises(RuntimeError, match="not registered.*depth='camera_depth"):
        assert_policy_camera_frames(
            message(colour), message("camera_depth_optical_frame"), message(colour), colour
        )


def test_policy_tool_transform_uses_the_kinematics_api():
    pytest.importorskip("franka_trajectory_replay.kinematics")
    tool = policy_tool_transform(
        {"tcp": {"offset_xyz": [0.01, -0.02, 0.11], "offset_rpy": [0.0, 0.0, 0.0]}}
    )
    assert tool.shape == (4, 4)
    assert tool[:3, 3] == pytest.approx([0.01, -0.02, 0.11])


def test_rgbd_synchronizer_ignores_staggered_latest_callbacks():
    frames = RgbdFrameSynchronizer(max_skew_s=0.04)
    frames.add("rgb", _stamped(10.0, "rgb-0"), 100.0)
    frames.add("depth", _stamped(10.0, "depth-0"), 100.01)
    assert tuple(message.value for message in frames.latest_pair()[:2]) == (
        "rgb-0",
        "depth-0",
    )

    # Reproduce the live failure: RGB arrives 134.4 ms ahead of the newest
    # depth callback. The last synchronized pair remains valid meanwhile.
    frames.add("rgb", _stamped(10.1344, "rgb-1"), 100.13)
    assert tuple(message.value for message in frames.latest_pair()[:2]) == (
        "rgb-0",
        "depth-0",
    )
    frames.add("depth", _stamped(10.1344, "depth-1"), 100.14)
    pair = frames.latest_pair()
    assert tuple(message.value for message in pair[:2]) == ("rgb-1", "depth-1")
    assert pair[2:] == pytest.approx((100.13, 100.14))


def test_rgbd_synchronizer_prefers_fresh_pair_over_older_exact_pair():
    frames = RgbdFrameSynchronizer(max_skew_s=0.04)
    frames.add("rgb", _stamped(10.0, "rgb-old"), 100.0)
    frames.add("depth", _stamped(10.0, "depth-old"), 100.0)
    frames.add("rgb", _stamped(10.20, "rgb-new"), 100.20)
    frames.add("depth", _stamped(10.21, "depth-new"), 100.21)

    pair = frames.latest_pair()
    assert tuple(message.value for message in pair[:2]) == ("rgb-new", "depth-new")
    assert pair[2:] == pytest.approx((100.20, 100.21))


def test_cartesian_step_limiter_uses_translation_and_quaternion_norms():
    reference_p = np.zeros(3)
    reference_q = fo.quat_from_euler_xyz(0.0, 0.0, 0.0)
    target_q = fo.quat_from_euler_xyz(0.12, 0.12, 0.12)
    position, quaternion, position_limited, orientation_limited = limit_cartesian_step(
        np.array([0.04, 0.03, 0.0]),
        target_q,
        reference_p,
        reference_q,
        max_position_step_m=0.036,
        max_orientation_step_rad=0.18,
    )
    assert position_limited
    assert orientation_limited
    assert np.linalg.norm(position - reference_p) == pytest.approx(0.036 * 0.98)
    angle = 2.0 * np.arccos(abs(float(np.dot(reference_q, quaternion))))
    assert angle == pytest.approx(0.18 * 0.98)


def test_training_frame_round_trip_and_controller_target():
    adapter = TrainingFrameAdapter()
    position = np.array([0.5, -0.1, 0.4])
    quaternion = fo.quat_from_euler_xyz(0.2, -0.3, 0.4)
    world_position, world_quaternion = adapter.pose_base_to_world(position, quaternion)
    round_position, round_quaternion = adapter.pose_world_to_base(
        world_position, world_quaternion
    )
    assert round_position == pytest.approx(position)
    assert abs(float(np.dot(round_quaternion, quaternion))) == pytest.approx(1.0)

    action = np.zeros(9)
    target_position, target_quaternion = adapter.controller_target(
        action,
        grasp_position_base=position,
        grasp_quaternion_base=quaternion,
        controlled_position_base=position,
        controlled_quaternion_base=quaternion,
    )
    expected_world = fo.decode_action_target(
        action,
        fo.GraspFrameState(
            world_position,
            world_quaternion,
            np.zeros(3),
            np.zeros(3),
            np.zeros((6, 7)),
        ),
    )
    expected_position, expected_grasp_quaternion = adapter.pose_world_to_base(
        expected_world.pos, expected_world.quat
    )
    assert target_position == pytest.approx(expected_position)

    assert abs(float(np.dot(target_quaternion, expected_grasp_quaternion))) == pytest.approx(1.0)


def test_grasp_target_is_retargeted_to_the_fixed_controller_point():
    grasp_position = np.array([0.58, 0.00, 0.26])
    grasp_quaternion = fo.quat_from_euler_xyz(0.1, -0.2, 0.3)
    controlled_in_grasp = np.array([-0.029, -0.001, -0.012])
    controlled_in_grasp_quaternion = fo.quat_from_euler_xyz(-0.2, 0.1, 0.05)
    controlled_position = grasp_position + fo.quat_rotate(
        grasp_quaternion, controlled_in_grasp
    )
    controlled_quaternion = fo.quat_mul(
        grasp_quaternion, controlled_in_grasp_quaternion
    )

    # An unchanged grasp target must reproduce the measured controller pose;
    # this is the invariant the old direct-position mapping violated by 31 mm.
    hold_position, hold_quaternion = retarget_grasp_pose_to_controlled_pose(
        grasp_position_base=grasp_position,
        grasp_quaternion_base=grasp_quaternion,
        controlled_position_base=controlled_position,
        controlled_quaternion_base=controlled_quaternion,
        target_grasp_position_base=grasp_position,
        target_grasp_quaternion_base=grasp_quaternion,
    )
    assert hold_position == pytest.approx(controlled_position)
    assert abs(float(np.dot(hold_quaternion, controlled_quaternion))) == pytest.approx(1.0)

    target_grasp_position = np.array([0.60, -0.02, 0.22])
    target_grasp_quaternion = fo.quat_from_euler_xyz(-0.3, 0.25, -0.1)
    target_position, target_quaternion = retarget_grasp_pose_to_controlled_pose(
        grasp_position_base=grasp_position,
        grasp_quaternion_base=grasp_quaternion,
        controlled_position_base=controlled_position,
        controlled_quaternion_base=controlled_quaternion,
        target_grasp_position_base=target_grasp_position,
        target_grasp_quaternion_base=target_grasp_quaternion,
    )
    assert target_position == pytest.approx(
        target_grasp_position
        + fo.quat_rotate(target_grasp_quaternion, controlled_in_grasp)
    )
    expected_quaternion = fo.quat_mul(
        target_grasp_quaternion, controlled_in_grasp_quaternion
    )
    assert abs(float(np.dot(target_quaternion, expected_quaternion))) == pytest.approx(1.0)


def test_cycle_coordinator_enters_release_after_clockwise_turn():
    reset_q = fo.quat_from_euler_xyz(0.0, 0.0, 0.0)
    coordinator = GripCycleCoordinator(
        rate_hz=10.0,
        max_cycles=1,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
    )
    turned = fo.quat_from_euler_xyz(0.0, 0.0, np.deg2rad(-56.0))
    event, progress = coordinator.update(
        position=np.zeros(3), quaternion=turned, hand=np.zeros(3)
    )
    assert progress == pytest.approx(np.deg2rad(56.0))
    assert event == "release_started"
    assert coordinator.process_phase() == "follow_waypoints"


def test_cycle_completion_rebases_without_retriggering_release():
    reset_q = fo.quat_from_euler_xyz(0.0, 0.0, 0.0)
    coordinator = GripCycleCoordinator(
        rate_hz=10.0,
        max_cycles=2,
        reset_position=np.zeros(3),
        reset_quaternion=reset_q,
        reset_hand=np.zeros(3),
    )
    coordinator.active = True
    coordinator.phase_index = 4
    coordinator.phase_steps = 9
    coordinator._unwrapped_yaw = np.deg2rad(-60.0)
    coordinator._previous_yaw = np.deg2rad(-60.0)

    event, _ = coordinator.update(
        position=np.zeros(3), quaternion=reset_q, hand=np.zeros(3)
    )

    assert event == "cycle_completed"
    assert coordinator.completed_cycles == 1
    assert not coordinator.active
