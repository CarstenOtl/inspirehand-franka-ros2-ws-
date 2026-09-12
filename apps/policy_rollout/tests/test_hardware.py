from types import SimpleNamespace

import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from policy_rollout.hardware import (
    GripCycleCoordinator,
    TrainingFrameAdapter,
    image_message_to_numpy,
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
        flange_quaternion_base=quaternion,
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

    flange_to_grasp = fo.quat_mul(fo.quat_conjugate(quaternion), quaternion)
    reconstructed_grasp = fo.quat_mul(target_quaternion, flange_to_grasp)
    assert abs(float(np.dot(reconstructed_grasp, expected_grasp_quaternion))) == pytest.approx(1.0)


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
