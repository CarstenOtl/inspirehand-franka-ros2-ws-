"""Small, ROS-independent SE(3) and hand-eye calibration routines.

Transforms are 4x4 matrices named ``T_parent_child`` and map coordinates from
the child frame into the parent frame.  Given observations

    T_world_hand[i] @ T_hand_tag == T_world_camera @ T_camera_tag[i]

we solve both fixed unknown transforms.  This is the eye-to-hand form of the
AX=XB problem; keeping it here free of ROS/OpenCV makes the convention directly
unit-testable.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class CalibrationError(ValueError):
    """Raised when samples cannot constrain a calibration."""


@dataclass(frozen=True)
class CalibrationResult:
    world_to_camera: np.ndarray
    hand_to_tag: np.ndarray
    translation_rmse_m: float
    rotation_rmse_deg: float
    sample_translation_errors_m: np.ndarray
    sample_rotation_errors_deg: np.ndarray
    translation_rank: int


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1.0e-12:
        raise CalibrationError("zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a unit quaternion (x, y, z, w)."""
    rotation = np.asarray(rotation, dtype=float)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[3] >= 0.0 else -quaternion


def _quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    x, y, z, w = matrix_to_quaternion_xyzw(rotation)
    return np.asarray([w, x, y, z])


def _left_quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def _right_quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, z, -y],
            [y, -z, w, x],
            [z, y, -x, w],
        ]
    )


def _average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=float)
    for rotation in rotations:
        quaternion = _quaternion_wxyz(rotation)
        accumulator += np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(accumulator)
    w, x, y, z = vectors[:, -1]
    return quaternion_xyzw_to_matrix([x, y, z, w])


def _relative_pairs(
    world_to_hand: Sequence[np.ndarray], camera_to_tag: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    motions_a: list[np.ndarray] = []
    motions_b: list[np.ndarray] = []
    for first in range(len(world_to_hand)):
        for second in range(first + 1, len(world_to_hand)):
            # inv(W_H[first]) W_H[second] X = X inv(C_T[first]) C_T[second]
            motions_a.append(invert_transform(world_to_hand[first]) @ world_to_hand[second])
            motions_b.append(invert_transform(camera_to_tag[first]) @ camera_to_tag[second])
    return motions_a, motions_b


def _solve_once(
    world_to_hand: Sequence[np.ndarray], camera_to_tag: Sequence[np.ndarray]
) -> CalibrationResult:
    motions_a, motions_b = _relative_pairs(world_to_hand, camera_to_tag)

    rotation_rows = []
    informative_rotations = 0
    for motion_a, motion_b in zip(motions_a, motions_b):
        if max(rotation_angle_deg(motion_a[:3, :3]), rotation_angle_deg(motion_b[:3, :3])) > 2.0:
            informative_rotations += 1
        qa = _quaternion_wxyz(motion_a[:3, :3])
        qb = _quaternion_wxyz(motion_b[:3, :3])
        rotation_rows.append(_left_quaternion_matrix(qa) - _right_quaternion_matrix(qb))

    if informative_rotations < 3:
        raise CalibrationError(
            "not enough rotational excitation; rotate the hand about at least two axes"
        )

    rotation_system = np.vstack(rotation_rows)
    _, singular_values, right_vectors = np.linalg.svd(rotation_system)
    solution_quaternion = right_vectors[-1]
    solution_quaternion /= np.linalg.norm(solution_quaternion)
    w, x, y, z = solution_quaternion
    hand_to_tag_rotation = quaternion_xyzw_to_matrix([x, y, z, w])

    translation_lhs = []
    translation_rhs = []
    for motion_a, motion_b in zip(motions_a, motions_b):
        translation_lhs.append(motion_a[:3, :3] - np.eye(3))
        translation_rhs.append(
            hand_to_tag_rotation @ motion_b[:3, 3] - motion_a[:3, 3]
        )
    translation_lhs_array = np.vstack(translation_lhs)
    translation_rank = int(np.linalg.matrix_rank(translation_lhs_array, tol=1.0e-7))
    if translation_rank < 3:
        raise CalibrationError(
            "motion is degenerate; vary roll, pitch, and yaw while keeping the tag visible"
        )
    hand_to_tag_translation, _, _, _ = np.linalg.lstsq(
        translation_lhs_array, np.hstack(translation_rhs), rcond=None
    )
    hand_to_tag = make_transform(hand_to_tag_rotation, hand_to_tag_translation)

    world_to_camera_candidates = [
        world_hand @ hand_to_tag @ invert_transform(camera_tag)
        for world_hand, camera_tag in zip(world_to_hand, camera_to_tag)
    ]
    world_to_camera = make_transform(
        _average_rotations([candidate[:3, :3] for candidate in world_to_camera_candidates]),
        np.mean([candidate[:3, 3] for candidate in world_to_camera_candidates], axis=0),
    )

    translation_errors = []
    rotation_errors = []
    for world_hand, camera_tag in zip(world_to_hand, camera_to_tag):
        predicted_world_to_tag = world_to_camera @ camera_tag
        robot_world_to_tag = world_hand @ hand_to_tag
        error = invert_transform(predicted_world_to_tag) @ robot_world_to_tag
        translation_errors.append(np.linalg.norm(error[:3, 3]))
        rotation_errors.append(rotation_angle_deg(error[:3, :3]))

    translation_errors_array = np.asarray(translation_errors)
    rotation_errors_array = np.asarray(rotation_errors)
    return CalibrationResult(
        world_to_camera=world_to_camera,
        hand_to_tag=hand_to_tag,
        translation_rmse_m=float(np.sqrt(np.mean(translation_errors_array**2))),
        rotation_rmse_deg=float(np.sqrt(np.mean(rotation_errors_array**2))),
        sample_translation_errors_m=translation_errors_array,
        sample_rotation_errors_deg=rotation_errors_array,
        translation_rank=translation_rank,
    )


def calibrate_eye_to_hand(
    world_to_hand: Sequence[np.ndarray],
    camera_to_tag: Sequence[np.ndarray],
    minimum_samples: int = 8,
    reject_outliers: bool = True,
) -> tuple[CalibrationResult, np.ndarray]:
    """Solve the fixed camera and tag poses, returning result and retained mask."""
    if len(world_to_hand) != len(camera_to_tag):
        raise CalibrationError("robot and camera sample counts differ")
    if len(world_to_hand) < minimum_samples:
        raise CalibrationError(
            f"need at least {minimum_samples} samples, have {len(world_to_hand)}"
        )
    for transform in list(world_to_hand) + list(camera_to_tag):
        if np.asarray(transform).shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise CalibrationError("every sample must be a finite 4x4 transform")

    result = _solve_once(world_to_hand, camera_to_tag)
    retained = np.ones(len(world_to_hand), dtype=bool)
    if not reject_outliers or len(world_to_hand) < minimum_samples + 3:
        return result, retained

    def robust_limit(values: np.ndarray, floor: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 3.5 * 1.4826 * mad)

    translation_limit = robust_limit(result.sample_translation_errors_m, 0.008)
    rotation_limit = robust_limit(result.sample_rotation_errors_deg, 1.5)
    retained = np.logical_and(
        result.sample_translation_errors_m <= translation_limit,
        result.sample_rotation_errors_deg <= rotation_limit,
    )
    if int(np.count_nonzero(retained)) >= minimum_samples and not np.all(retained):
        retained_world = [sample for sample, keep in zip(world_to_hand, retained) if keep]
        retained_camera = [sample for sample, keep in zip(camera_to_tag, retained) if keep]
        result = _solve_once(retained_world, retained_camera)
    else:
        retained[:] = True
    return result, retained
