"""Generating the joint configurations an automated calibration run drives to.

The set of poses is what decides whether a hand-eye calibration is well posed.
Rather than hoping a human happens to move the hand well, the program is built
backwards from the camera: sample where the tag should appear in the image and
how it should be tilted, turn that into the flange pose that puts it there, and
solve inverse kinematics for it.  Poses that leave the image, graze the tag,
exceed a joint limit, dip towards the table or duplicate an orientation already
in the set are discarded before the arm is ever asked to move.

A coarse ``world -> camera`` is needed to aim at all; it comes from a short
bootstrap pass over hand-taught seed poses, or from a previous run's result.
The forward kinematics and joint limits are the ones the replay controller
enforces, imported from ``franka_trajectory_replay`` so that there is one FR3
model in this workspace rather than two.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .calibration_math import invert_transform, make_transform, rotation_angle_deg
from .tag_pose import project_tag_corners, tag_view_angle_deg

try:
    from franka_trajectory_replay.kinematics import READY_POSE, flange_jacobian, flange_transform
    from franka_trajectory_replay.limits import POSITION_LOWER, POSITION_UPPER
except ImportError as error:  # pragma: no cover - exercised only outside the workspace
    raise RuntimeError(
        "pose generation needs franka_trajectory_replay for the FR3 model; "
        "build and source this workspace first"
    ) from error


class PoseProgramError(RuntimeError):
    """Raised when no usable pose program can be built."""


@dataclass(frozen=True)
class CameraModel:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, message) -> "CameraModel":
        return cls(
            matrix=np.asarray(message.k, dtype=float).reshape(3, 3),
            distortion=np.asarray(message.d, dtype=float),
            width=int(message.width),
            height=int(message.height),
        )


@dataclass(frozen=True)
class CalibrationPose:
    joint_positions: np.ndarray
    world_to_hand: np.ndarray
    camera_to_tag: np.ndarray
    image_corners: np.ndarray
    view_angle_deg: float
    distance_m: float
    inverse_kinematics_position_error_m: float
    inverse_kinematics_rotation_error_deg: float

    def as_dict(self) -> dict:
        return {
            "joint_positions": [float(value) for value in self.joint_positions],
            "predicted_world_to_hand": self.world_to_hand.tolist(),
            "predicted_camera_to_tag": self.camera_to_tag.tolist(),
            "predicted_image_corners": self.image_corners.tolist(),
            "view_angle_deg": self.view_angle_deg,
            "distance_m": self.distance_m,
            "inverse_kinematics_position_error_m": self.inverse_kinematics_position_error_m,
            "inverse_kinematics_rotation_error_deg": self.inverse_kinematics_rotation_error_deg,
        }


@dataclass(frozen=True)
class PoseProgramLimits:
    """Everything a candidate pose is judged against."""

    distance_range_m: tuple = (0.35, 0.85)
    max_tilt_deg: float = 35.0
    # The tag's roll about its own normal is useful excitation, but a full turn
    # of it drags joint 7 across its whole range and turns a compact program
    # into 4 rad hops between neighbouring views.
    max_roll_deg: float = 60.0
    max_view_angle_deg: float = 55.0
    image_margin_px: float = 50.0
    min_tag_edge_px: float = 22.0
    joint_margin_rad: float = 0.12
    min_flange_height_m: float = 0.15
    min_orientation_separation_deg: float = 7.0
    max_joint_step_rad: float = 2.5


def pose_residual(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Six-vector (linear, angular) taking ``current`` onto ``target``, in the base frame."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    residual = np.zeros(6)
    residual[:3] = target[:3, 3] - current[:3, 3]
    difference = target[:3, :3] @ current[:3, :3].T
    angle = np.radians(rotation_angle_deg(difference))
    if angle > 1.0e-12:
        axis = np.asarray(
            [
                difference[2, 1] - difference[1, 2],
                difference[0, 2] - difference[2, 0],
                difference[1, 0] - difference[0, 1],
            ]
        )
        norm = np.linalg.norm(axis)
        if norm > 1.0e-12:
            residual[3:] = axis / norm * angle
    return residual


def solve_inverse_kinematics(
    base_to_hand_target: np.ndarray,
    seed: Sequence[float],
    joint_margin_rad: float = 0.0,
    iterations: int = 200,
    damping: float = 0.05,
    tolerance_m: float = 1.0e-4,
    tolerance_deg: float = 0.05,
    stall_iterations: int = 12,
) -> Optional[tuple[np.ndarray, float, float]]:
    """Damped least squares onto the flange pose; ``None`` if it does not converge.

    Joint values are clamped into the limits at every step, so a solution that
    is returned is always inside them: an unreachable target simply fails to
    converge instead of producing a pose the controller would reject.  Pose
    generation throws away far more candidates than it keeps, so a target that
    is going nowhere is abandoned as soon as the error stops falling rather than
    after the full iteration budget.
    """
    lower = POSITION_LOWER + joint_margin_rad
    upper = POSITION_UPPER - joint_margin_rad
    joint_positions = np.clip(np.asarray(seed, dtype=float), lower, upper)
    best_error = np.inf
    since_improvement = 0
    for _ in range(iterations):
        current = flange_transform(joint_positions)
        residual = pose_residual(current, base_to_hand_target)
        position_error = float(np.linalg.norm(residual[:3]))
        rotation_error = float(np.degrees(np.linalg.norm(residual[3:])))
        if position_error <= tolerance_m and rotation_error <= tolerance_deg:
            return joint_positions, position_error, rotation_error
        combined = position_error + np.radians(rotation_error)
        if combined < best_error - 1.0e-6:
            best_error = combined
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= stall_iterations:
                return None
        jacobian = flange_jacobian(joint_positions)
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + (damping**2) * np.eye(6), residual
        )
        # A long step is a sign of a near-singular configuration; shorten it
        # instead of jumping across the workspace and losing the seed.
        largest = float(np.abs(step).max())
        if largest > 0.3:
            step *= 0.3 / largest
        joint_positions = np.clip(joint_positions + step, lower, upper)
    return None


def _tag_orientation(
    direction: np.ndarray, tilt_x: float, tilt_y: float, roll: float
) -> np.ndarray:
    """A tag facing back along ``direction``, then tilted about its own x and y and rolled."""
    forward = -np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    helper = np.asarray([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.asarray([1.0, 0.0, 0.0])
    right = np.cross(helper, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    rotation = np.column_stack((right, up, forward))
    return rotation @ _rotation_x(tilt_x) @ _rotation_y(tilt_y) @ _rotation_z(roll)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _image_cells(camera: CameraModel, margin: float, columns: int, rows: int):
    left, right = margin, camera.width - margin
    top, bottom = margin, camera.height - margin
    if right <= left or bottom <= top:
        raise PoseProgramError(
            f"image_margin_px ({margin}) leaves no room in a "
            f"{camera.width}x{camera.height} image"
        )
    cells = []
    for row in range(rows):
        for column in range(columns):
            cells.append(
                (
                    left + (right - left) * column / columns,
                    left + (right - left) * (column + 1) / columns,
                    top + (bottom - top) * row / rows,
                    top + (bottom - top) * (row + 1) / rows,
                )
            )
    return cells


def _corners_inside(corners: np.ndarray, camera: CameraModel, margin: float) -> bool:
    return bool(
        np.all(corners[:, 0] >= margin)
        and np.all(corners[:, 0] <= camera.width - margin)
        and np.all(corners[:, 1] >= margin)
        and np.all(corners[:, 1] <= camera.height - margin)
    )


def _shortest_edge_px(corners: np.ndarray) -> float:
    edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    return float(edges.min())


def orientation_excitation(world_to_hand: Sequence[np.ndarray]) -> tuple[float, int]:
    """How well the set's relative rotations span three axes.

    Returns the smallest eigenvalue of the angle-weighted axis scatter matrix,
    normalised by the largest, and the number of pairs rotating by more than
    2 deg.  The solver needs both to be healthy: a set that only rotates about
    one axis cannot determine the unknown flange-to-tag offset.
    """
    scatter = np.zeros((3, 3))
    informative = 0
    for first in range(len(world_to_hand)):
        for second in range(first + 1, len(world_to_hand)):
            relative = invert_transform(world_to_hand[first]) @ world_to_hand[second]
            angle = rotation_angle_deg(relative[:3, :3])
            if angle <= 2.0:
                continue
            informative += 1
            axis = np.asarray(
                [
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ]
            )
            norm = np.linalg.norm(axis)
            if norm > 1.0e-9:
                axis = axis / norm
                scatter += angle * np.outer(axis, axis)
    if informative == 0:
        return 0.0, 0
    eigenvalues = np.linalg.eigvalsh(scatter)
    largest = float(eigenvalues[-1])
    return (float(eigenvalues[0] / largest) if largest > 0.0 else 0.0), informative


def order_by_travel(
    poses: Sequence[CalibrationPose], start_joint_positions: Sequence[float]
) -> list[CalibrationPose]:
    """Greedy nearest neighbour in joint space, so the run is short and every ramp is small."""
    remaining = list(poses)
    ordered: list[CalibrationPose] = []
    current = np.asarray(start_joint_positions, dtype=float)
    while remaining:
        distances = [float(np.abs(pose.joint_positions - current).max()) for pose in remaining]
        index = int(np.argmin(distances))
        chosen = remaining.pop(index)
        ordered.append(chosen)
        current = chosen.joint_positions
    return ordered


def _largest_step(poses: Sequence[CalibrationPose]) -> float:
    if len(poses) < 2:
        return 0.0
    return max(
        float(np.abs(second.joint_positions - first.joint_positions).max())
        for first, second in zip(poses, poses[1:])
    )


def generate_pose_program(
    world_to_camera: np.ndarray,
    hand_to_tag: np.ndarray,
    camera: CameraModel,
    tag_size_m: float,
    count: int = 28,
    world_to_base: Optional[np.ndarray] = None,
    start_joint_positions: Optional[Sequence[float]] = None,
    limits: PoseProgramLimits = PoseProgramLimits(),
    seed: int = 0,
    attempts_per_pose: int = 60,
    collision_check=None,
) -> list[CalibrationPose]:
    """Build ``count`` reachable, well-spread calibration poses around a coarse camera pose.

    ``collision_check(joint_positions) -> bool`` is an optional veto, used for
    the MuJoCo self-collision test when the model is available.
    """
    if count < 8:
        raise PoseProgramError("a calibration program needs at least 8 poses")
    world_to_camera = np.asarray(world_to_camera, dtype=float).reshape(4, 4)
    hand_to_tag = np.asarray(hand_to_tag, dtype=float).reshape(4, 4)
    world_to_base = np.eye(4) if world_to_base is None else np.asarray(world_to_base, dtype=float)
    base_to_world = invert_transform(world_to_base)
    tag_to_hand = invert_transform(hand_to_tag)
    start = (
        np.asarray(READY_POSE, dtype=float)
        if start_joint_positions is None
        else np.asarray(start_joint_positions, dtype=float)
    )

    random = np.random.default_rng(seed)
    inverse_matrix = np.linalg.inv(camera.matrix)
    cells = _image_cells(camera, limits.image_margin_px, columns=3, rows=3)
    accepted: list[CalibrationPose] = []
    seed_joint_positions = start.copy()
    rejections = {
        "view_angle": 0,
        "outside_image": 0,
        "tag_too_small": 0,
        "no_inverse_kinematics": 0,
        "flange_too_low": 0,
        "collision": 0,
        "duplicate_orientation": 0,
        "too_far_from_previous": 0,
    }

    for index in range(count):
        cell = cells[index % len(cells)]
        pose = None
        for _ in range(attempts_per_pose):
            pixel = np.asarray(
                [
                    random.uniform(cell[0], cell[1]),
                    random.uniform(cell[2], cell[3]),
                    1.0,
                ]
            )
            distance = random.uniform(*limits.distance_range_m)
            ray = inverse_matrix @ pixel
            position = ray / ray[2] * distance
            tilt = np.radians(limits.max_tilt_deg)
            roll = np.radians(limits.max_roll_deg)
            rotation = _tag_orientation(
                position,
                random.uniform(-tilt, tilt),
                random.uniform(-tilt, tilt),
                random.uniform(-roll, roll),
            )
            camera_to_tag = make_transform(rotation, position)

            view_angle = tag_view_angle_deg(camera_to_tag)
            if view_angle > limits.max_view_angle_deg:
                rejections["view_angle"] += 1
                continue
            corners = project_tag_corners(
                camera_to_tag, camera.matrix, camera.distortion, tag_size_m
            )
            if not _corners_inside(corners, camera, limits.image_margin_px):
                rejections["outside_image"] += 1
                continue
            if _shortest_edge_px(corners) < limits.min_tag_edge_px:
                rejections["tag_too_small"] += 1
                continue

            world_to_hand = world_to_camera @ camera_to_tag @ tag_to_hand
            if world_to_hand[2, 3] < limits.min_flange_height_m:
                rejections["flange_too_low"] += 1
                continue
            solution = solve_inverse_kinematics(
                base_to_world @ world_to_hand,
                seed_joint_positions,
                joint_margin_rad=limits.joint_margin_rad,
            )
            if solution is None:
                rejections["no_inverse_kinematics"] += 1
                continue
            joint_positions, position_error, rotation_error = solution
            if (
                float(np.abs(joint_positions - seed_joint_positions).max())
                > limits.max_joint_step_rad
            ):
                rejections["too_far_from_previous"] += 1
                continue
            if collision_check is not None and not collision_check(joint_positions):
                rejections["collision"] += 1
                continue
            if any(
                rotation_angle_deg(
                    (invert_transform(other.world_to_hand) @ world_to_hand)[:3, :3]
                )
                < limits.min_orientation_separation_deg
                for other in accepted
            ):
                rejections["duplicate_orientation"] += 1
                continue

            pose = CalibrationPose(
                joint_positions=joint_positions,
                world_to_hand=world_to_base @ flange_transform(joint_positions),
                camera_to_tag=camera_to_tag,
                image_corners=corners,
                view_angle_deg=view_angle,
                distance_m=float(distance),
                inverse_kinematics_position_error_m=position_error,
                inverse_kinematics_rotation_error_deg=rotation_error,
            )
            break
        if pose is not None:
            accepted.append(pose)
            seed_joint_positions = pose.joint_positions

    if len(accepted) < 8:
        detail = ", ".join(f"{key} {value}" for key, value in rejections.items() if value)
        raise PoseProgramError(
            f"only {len(accepted)} of {count} poses were usable ({detail}). The coarse "
            "camera pose is probably wrong, or the camera cannot see the robot's "
            "workspace at the configured distances."
        )

    # Every accepted pose is within max_joint_step_rad of the one before it, so the
    # order they were generated in is always drivable.  Shortening the route is an
    # optimisation on top of that, and it is only taken if it stays drivable too.
    ordered = order_by_travel(accepted, start)
    if _largest_step(ordered) > limits.max_joint_step_rad:
        ordered = accepted
    # The ramp from wherever the arm is right now is a different matter: it is a
    # single goto like any other, and the controller's own max_joint_step is the
    # authority on it, so it is reported by the runner rather than refused here.

    spread, informative = orientation_excitation([pose.world_to_hand for pose in ordered])
    if spread < 0.02 or informative < 3 * len(ordered):
        raise PoseProgramError(
            f"the generated poses rotate about too few axes (axis spread {spread:.3f}, "
            f"{informative} informative pairs); raise max_tilt_deg or widen distance_range_m"
        )
    return ordered
