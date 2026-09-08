import numpy as np
import pytest

from camera_calibration.calibration_math import (
    CalibrationError,
    calibrate_eye_to_hand,
    calibrate_fixed_tag_eye_to_hand,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine],
        ]
    )


def synthetic_samples(count=18):
    random = np.random.default_rng(51)
    world_to_camera = make_transform(
        axis_angle([0.2, 0.8, -0.4], 0.7), [0.72, -0.31, 0.61]
    )
    hand_to_tag = make_transform(
        axis_angle([1.0, 0.3, 0.2], -0.35), [0.035, -0.018, 0.105]
    )
    world_to_hand = []
    camera_to_tag = []
    for _ in range(count):
        hand_pose = make_transform(
            axis_angle(random.normal(size=3), random.uniform(-1.0, 1.0)),
            random.uniform([0.2, -0.4, 0.2], [0.7, 0.4, 0.8]),
        )
        world_to_hand.append(hand_pose)
        camera_to_tag.append(invert_transform(world_to_camera) @ hand_pose @ hand_to_tag)
    return world_to_camera, hand_to_tag, world_to_hand, camera_to_tag


def test_transform_and_quaternion_round_trip():
    transform = make_transform(axis_angle([1, 2, 3], 1.2), [0.1, -0.2, 0.3])
    np.testing.assert_allclose(invert_transform(transform) @ transform, np.eye(4), atol=1e-12)
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    np.testing.assert_allclose(
        quaternion_xyzw_to_matrix(quaternion), transform[:3, :3], atol=1e-12
    )


def test_recovers_exact_camera_and_tag_transforms():
    expected_camera, expected_tag, robot, observations = synthetic_samples()
    result, retained = calibrate_eye_to_hand(robot, observations)
    np.testing.assert_allclose(result.world_to_camera, expected_camera, atol=1e-10)
    np.testing.assert_allclose(result.hand_to_tag, expected_tag, atol=1e-10)
    assert retained.all()
    assert result.translation_rmse_m < 1e-10
    assert result.rotation_rmse_deg < 1e-5


def test_rejects_one_bad_observation():
    expected_camera, _, robot, observations = synthetic_samples(20)
    observations[7] = observations[7].copy()
    observations[7][:3, 3] += [0.10, -0.08, 0.05]
    result, retained = calibrate_eye_to_hand(robot, observations)
    assert retained.sum() == 19
    assert not retained[7]
    np.testing.assert_allclose(result.world_to_camera, expected_camera, atol=1e-10)


def test_rejects_pose_set_without_rotation():
    hand_to_tag = make_transform(np.eye(3), [0, 0, 0.1])
    robot = [make_transform(np.eye(3), [0.05 * index, 0, 0.4]) for index in range(8)]
    observations = [sample @ hand_to_tag for sample in robot]
    with pytest.raises(CalibrationError, match="rotational excitation"):
        calibrate_eye_to_hand(robot, observations)


def test_fixed_tag_calibration_recovers_camera_without_estimating_mount():
    expected_camera, known_tag, robot, observations = synthetic_samples()
    result, retained = calibrate_fixed_tag_eye_to_hand(
        robot, observations, known_tag, minimum_samples=12
    )
    np.testing.assert_allclose(result.world_to_camera, expected_camera, atol=1e-10)
    np.testing.assert_allclose(result.hand_to_tag, known_tag, atol=1e-12)
    assert retained.all()


def test_fixed_tag_calibration_rejects_bad_rgb_observation():
    expected_camera, known_tag, robot, observations = synthetic_samples(20)
    observations[5] = observations[5].copy()
    observations[5][:3, 3] += [0.12, -0.09, 0.04]
    result, retained = calibrate_fixed_tag_eye_to_hand(
        robot, observations, known_tag, minimum_samples=12
    )
    assert retained.sum() == 19
    assert not retained[5]
    np.testing.assert_allclose(result.world_to_camera, expected_camera, atol=1e-10)


def test_fixed_tag_calibration_allows_translation_only_motion():
    world_to_camera = make_transform(axis_angle([0.2, -0.4, 0.7], 0.6), [0.8, 0.1, 0.7])
    known_tag = make_transform(axis_angle([0.1, 0.9, -0.2], -0.4), [0.0, 0.02, 0.092])
    robot = [make_transform(np.eye(3), [0.3 + 0.03 * index, 0.0, 0.4]) for index in range(12)]
    observations = [
        invert_transform(world_to_camera) @ world_to_hand @ known_tag
        for world_to_hand in robot
    ]
    result, _ = calibrate_fixed_tag_eye_to_hand(
        robot, observations, known_tag, minimum_samples=12
    )
    np.testing.assert_allclose(result.world_to_camera, world_to_camera, atol=1e-10)
