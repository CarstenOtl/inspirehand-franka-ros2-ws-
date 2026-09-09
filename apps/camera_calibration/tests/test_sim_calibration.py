#!/usr/bin/env python3
"""End-to-end eye-to-hand calibration in the FR3/Inspire MuJoCo scene.

The combined robot model already contains a 40 mm ``tag36h11`` ID 0 on the
dorsal side of the hand.  This script adds a fixed pinhole camera with nominal
Intel RealSense D435 RGB intrinsics, moves the FR3 through a small multi-axis
trajectory, and for every rendered RGB image:

1. detects the AprilTag with OpenCV;
2. estimates ``T_camera_tag`` with ``cv2.solvePnP``;
3. reads ``T_world_tag`` from MuJoCo forward kinematics; and
4. solves ``T_world_camera`` from
   ``T_world_tag = T_world_camera @ T_camera_tag``.

Depth is neither rendered nor used.  Run the interactive RGB view with::

    python3 apps/camera_calibration/tests/test_sim_calibration.py

For CI or a machine without a display, select a headless OpenGL backend before
starting Python, for example::

    MUJOCO_GL=egl python3 apps/camera_calibration/tests/test_sim_calibration.py --headless

The physical D435's factory intrinsics vary by device.  The simulation uses a
centered 1920x1080 pinhole model with fx=fy=1383.75 px; real calibration
continues to use the measured ``CameraInfo`` values from the camera driver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_SCENE = REPO_ROOT / "assets" / "fr3_inspirehand" / "fr3_inspirehand.xml"

# Make direct execution work without requiring the ROS overlay to be sourced.
PACKAGE_SOURCE = REPO_ROOT / "src" / "camera_calibration"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from camera_calibration.calibration_math import (  # noqa: E402
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rotation_angle_deg,
)
from camera_calibration.auto_waypoints import AUTO_WAYPOINTS  # noqa: E402
from camera_calibration.model_geometry import (  # noqa: E402
    TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW,
    TAG_BODY_TO_PRINTED_TAG_XYZ,
)


CAMERA_NAME = "rs435_rgb"
CAMERA_ROLL_DEG = 90.0
TAG_BODY_NAME = "apriltag_0"
TAG_ID = 0
TAG_SIZE_M = 0.040  # Black-square edge, excluding the white quiet zone.
ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class D435RgbIntrinsics:
    """Nominal full-resolution D435 RGB calibration for the virtual camera."""

    width: int = 1920
    height: int = 1080
    fx: float = 1383.75
    fy: float = 1383.75
    cx: float = 960.0
    cy: float = 540.0

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def fovy_deg(self) -> float:
        # A MuJoCo camera has one vertical field-of-view parameter.  Keeping
        # fx==fy makes its rendered projection identical to this OpenCV model.
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width <= 0 or self.height <= 0 or not all(np.isfinite(values)):
            raise ValueError("D435 RGB intrinsics must be finite and positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("D435 RGB focal lengths must be positive")
        if not np.isclose(self.fx, self.fy, rtol=0.0, atol=1.0e-9):
            raise ValueError("MuJoCo rendering requires fx == fy for this camera model")
        if not np.isclose(self.cx, self.width / 2.0, atol=1.0e-9) or not np.isclose(
            self.cy, self.height / 2.0, atol=1.0e-9
        ):
            raise ValueError("MuJoCo rendering requires a centered principal point")


@dataclass(frozen=True)
class TagObservation:
    camera_to_tag: np.ndarray
    image_corners: np.ndarray
    reprojection_error_px: float


@dataclass(frozen=True)
class FixedCameraCalibrationResult:
    world_to_camera: np.ndarray
    translation_rmse_m: float
    rotation_rmse_deg: float
    sample_translation_errors_m: np.ndarray
    sample_rotation_errors_deg: np.ndarray


@dataclass(frozen=True)
class SimulationCalibrationRun:
    calibration: FixedCameraCalibrationResult
    retained: np.ndarray
    expected_world_to_camera: np.ndarray
    world_to_tag: tuple[np.ndarray, ...]
    camera_to_tag: tuple[np.ndarray, ...]
    detected_frame_count: int
    motion_frame_count: int
    last_rgb: np.ndarray
    last_annotated_bgr: np.ndarray

    @property
    def camera_translation_error_m(self) -> float:
        return float(
            np.linalg.norm(
                self.calibration.world_to_camera[:3, 3]
                - self.expected_world_to_camera[:3, 3]
            )
        )

    @property
    def camera_rotation_error_deg(self) -> float:
        error = (
            self.calibration.world_to_camera[:3, :3].T
            @ self.expected_world_to_camera[:3, :3]
        )
        return rotation_angle_deg(error)


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MuJoCo is not importable. Install the Python 'mujoco' package or "
            "run in the workspace simulation environment."
        ) from exc
    return mujoco


def _named_id(mj: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mj.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {name!r}")
    return object_id


def _pose(position: np.ndarray, row_major_rotation: np.ndarray) -> np.ndarray:
    return make_transform(
        np.asarray(row_major_rotation, dtype=float).reshape(3, 3),
        np.asarray(position, dtype=float),
    )


def _reset_start_keyframe(mj: Any, model: Any, data: Any) -> None:
    key_id = _named_id(mj, model, mj.mjtObj.mjOBJ_KEY, "start")
    mj.mj_resetDataKeyframe(model, data, key_id)
    mj.mj_forward(model, data)


def _initial_camera_pose(
    mj: Any, model: Any, data: Any, distance_m: float
) -> np.ndarray:
    """Place the fixed camera opposite the tag at the trajectory's centre pose.

    MuJoCo cameras look down local -Z with +Y upward.  The camera is placed
    along the tag's outward +Z with a modest lateral offset, aimed back at the
    tag, and remains fixed after this setup.
    """
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("camera distance must be positive")
    tag_id = _named_id(mj, model, mj.mjtObj.mjOBJ_BODY, TAG_BODY_NAME)
    world_to_tag = _pose(data.xpos[tag_id], data.xmat[tag_id])
    # A modest off-axis view is intentional: an exactly fronto-parallel square
    # has two nearly indistinguishable planar PnP solutions.  The offset gives
    # IPPE useful perspective while keeping the camera opposite the palm.
    camera_position = (
        world_to_tag[:3, 3]
        + distance_m * world_to_tag[:3, 2]
        + 0.14 * world_to_tag[:3, 0]
        + 0.08 * world_to_tag[:3, 1]
    )
    camera_back = camera_position - world_to_tag[:3, 3]
    camera_back /= np.linalg.norm(camera_back)
    camera_right = (
        world_to_tag[:3, 0] - np.dot(world_to_tag[:3, 0], camera_back) * camera_back
    )
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_back, camera_right)

    # Roll the physical sensor so the workcell appears upright in the RGB
    # image.  With MuJoCo's +Y-up camera convention, right'=up and up'=-right
    # rotates the rendered image 90 degrees clockwise without changing the
    # optical axis or camera position.
    roll = math.radians(CAMERA_ROLL_DEG)
    original_right = camera_right.copy()
    camera_right = math.cos(roll) * original_right + math.sin(roll) * camera_up
    camera_up = -math.sin(roll) * original_right + math.cos(roll) * camera_up
    world_to_camera_mujoco = make_transform(
        np.column_stack((camera_right, camera_up, camera_back)), camera_position
    )
    return world_to_camera_mujoco


def _absolute_asset_xml_with_camera(
    world_to_camera_mujoco: np.ndarray, intrinsics: D435RgbIntrinsics
) -> str:
    """Return the scene XML with absolute assets and one fixed RGB camera."""
    if not ROBOT_SCENE.is_file():
        raise FileNotFoundError(f"combined MuJoCo scene not found: {ROBOT_SCENE}")
    root = ET.fromstring(ROBOT_SCENE.read_text(encoding="utf-8"))

    # Loading from an XML string otherwise loses the scene file's directory.
    # Absolute paths retain all user-owned mesh and AprilTag texture edits.
    for element in root.iter():
        filename = element.get("file")
        if filename:
            asset_path = (ROBOT_SCENE.parent / filename).resolve()
            if not asset_path.is_file():
                raise FileNotFoundError(f"MuJoCo asset not found: {asset_path}")
            element.set("file", str(asset_path))

    # The appearance model can wrap the tag over the curved hand shell, while
    # IPPE_SQUARE (and a real tag fixed to a rigid backing) assumes a plane.
    # Inject an explicitly UV-mapped 50 mm carrier.  The texture has a 640/800
    # black-square ratio, hence its metric black edge is exactly 40 mm.
    asset = root.find("asset")
    tag_geom = root.find(".//geom[@name='apriltag_36h11_id0']")
    if asset is None or tag_geom is None:
        raise RuntimeError(f"{ROBOT_SCENE} has no AprilTag asset/geometry")
    planar_mesh_name = "sim_calibration_planar_apriltag"
    asset.append(
        ET.Element(
            "mesh",
            {
                "name": planar_mesh_name,
                "vertex": (
                    "-0.025 -0.025 0.002  0.025 -0.025 0.002  "
                    "0.025 0.025 0.002  -0.025 0.025 0.002"
                ),
                # Preserve the existing tag texture's orientation relative to
                # the apriltag_0 body: texture U=-Y and texture V=+X.
                # Inline MuJoCo texcoords use the opposite U handedness to the
                # source OBJ loader, so U is flipped here (not the image).
                "texcoord": "0 0  0 1  1 1  1 0",
                "face": "0 1 2  0 2 3",
                "inertia": "shell",
            },
        )
    )
    tag_geom.set("type", "mesh")
    tag_geom.set("mesh", planar_mesh_name)
    tag_geom.attrib.pop("size", None)

    visual_global = root.find("./visual/global")
    if visual_global is None:
        raise RuntimeError(f"{ROBOT_SCENE} has no visual/global configuration")
    visual_global.set("offwidth", str(intrinsics.width))
    visual_global.set("offheight", str(intrinsics.height))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"{ROBOT_SCENE} has no worldbody")
    rotation = world_to_camera_mujoco[:3, :3]
    camera = ET.Element(
        "camera",
        {
            "name": CAMERA_NAME,
            "mode": "fixed",
            "pos": " ".join(f"{value:.12g}" for value in world_to_camera_mujoco[:3, 3]),
            "xyaxes": " ".join(
                f"{value:.12g}" for value in np.hstack((rotation[:, 0], rotation[:, 1]))
            ),
            "fovy": f"{intrinsics.fovy_deg:.12g}",
        },
    )
    worldbody.insert(0, camera)
    return ET.tostring(root, encoding="unicode")


def load_calibration_scene(
    intrinsics: D435RgbIntrinsics = D435RgbIntrinsics(),
    camera_distance_m: float = 0.50,
) -> tuple[Any, Any, Any, np.ndarray]:
    """Load the robot and return its fixed camera pose in OpenCV coordinates."""
    intrinsics.validate()
    mj = _require_mujoco()

    seed_model = mj.MjModel.from_xml_path(str(ROBOT_SCENE))
    seed_data = mj.MjData(seed_model)
    _reset_start_keyframe(mj, seed_model, seed_data)
    world_to_camera_mujoco = _initial_camera_pose(
        mj, seed_model, seed_data, camera_distance_m
    )

    xml = _absolute_asset_xml_with_camera(world_to_camera_mujoco, intrinsics)
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    _reset_start_keyframe(mj, model, data)

    camera_id = _named_id(mj, model, mj.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    world_to_camera_mujoco = _pose(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
    # OpenCV optical coordinates are +X right, +Y down, +Z forward.  MuJoCo's
    # camera coordinates are +X right, +Y up, -Z forward.
    opencv_to_mujoco = np.diag([1.0, -1.0, -1.0, 1.0])
    world_to_camera_opencv = world_to_camera_mujoco @ opencv_to_mujoco
    return mj, model, data, world_to_camera_opencv


def _arm_joint_addresses(mj: Any, model: Any) -> np.ndarray:
    addresses = []
    for name in ARM_JOINTS:
        joint_id = _named_id(mj, model, mj.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"{name!r} is not a scalar hinge")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def _set_arm_pose(
    mj: Any,
    model: Any,
    data: Any,
    qpos_template: np.ndarray,
    addresses: np.ndarray,
    arm_position: np.ndarray,
) -> None:
    data.qpos[:] = qpos_template
    data.qvel[:] = 0.0
    data.qpos[addresses] = arm_position
    mj.mj_forward(model, data)


def _project_tag_corners(
    world_to_camera: np.ndarray,
    world_to_tag: np.ndarray,
    intrinsics: D435RgbIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    half = TAG_SIZE_M * 0.5
    corners_tag = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=float,
    )
    camera_to_tag = invert_transform(world_to_camera) @ world_to_tag
    corners_camera = (
        camera_to_tag[:3, :3] @ corners_tag.T + camera_to_tag[:3, 3, None]
    ).T
    image = np.empty((4, 2), dtype=float)
    image[:, 0] = (
        intrinsics.fx * corners_camera[:, 0] / corners_camera[:, 2] + intrinsics.cx
    )
    image[:, 1] = (
        intrinsics.fy * corners_camera[:, 1] / corners_camera[:, 2] + intrinsics.cy
    )
    return image, corners_camera


def _world_to_printed_tag_from_fk(data: Any, tag_body_id: int) -> np.ndarray:
    """Return the printed tag frame in the robot/world frame from MuJoCo FK."""
    world_to_tag_body = _pose(data.xpos[tag_body_id], data.xmat[tag_body_id])
    # The in-memory planar mesh is 2 mm above the tag body.  Its UV mapping
    # rotates the printed AprilTag axes +90 degrees around the body's +Z.
    body_to_printed_tag = make_transform(
        quaternion_xyzw_to_matrix(TAG_BODY_TO_PRINTED_TAG_QUATERNION_XYZW),
        TAG_BODY_TO_PRINTED_TAG_XYZ,
    )
    return world_to_tag_body @ body_to_printed_tag


def _average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=float)
    for rotation in rotations:
        x, y, z, w = matrix_to_quaternion_xyzw(rotation)
        quaternion = np.asarray([w, x, y, z])
        accumulator += np.outer(quaternion, quaternion)
    _, eigenvectors = np.linalg.eigh(accumulator)
    w, x, y, z = eigenvectors[:, -1]
    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm < 1.0e-12:
        raise RuntimeError("camera rotation average is degenerate")
    quaternion /= quaternion_norm
    return quaternion_xyzw_to_matrix(quaternion)


def calibrate_fixed_camera_from_tag_poses(
    world_to_tag: Sequence[np.ndarray],
    camera_to_tag: Sequence[np.ndarray],
    *,
    minimum_samples: int = 8,
) -> tuple[FixedCameraCalibrationResult, np.ndarray]:
    """Solve ``T_world_camera`` from synchronized OpenCV and robot-FK poses."""
    if len(world_to_tag) != len(camera_to_tag):
        raise ValueError("robot-FK and camera sample counts differ")
    if len(world_to_tag) < minimum_samples:
        raise ValueError(
            f"need at least {minimum_samples} tag-pose pairs, have {len(world_to_tag)}"
        )
    candidates = [
        np.asarray(robot_tag, dtype=float) @ invert_transform(camera_tag)
        for robot_tag, camera_tag in zip(world_to_tag, camera_to_tag)
    ]

    def solve(indices: np.ndarray) -> FixedCameraCalibrationResult:
        selected = [candidates[index] for index in indices]
        world_to_camera = make_transform(
            _average_rotations([candidate[:3, :3] for candidate in selected]),
            np.mean([candidate[:3, 3] for candidate in selected], axis=0),
        )
        translation_errors = []
        rotation_errors = []
        for candidate in candidates:
            error = invert_transform(world_to_camera) @ candidate
            translation_errors.append(np.linalg.norm(error[:3, 3]))
            rotation_errors.append(rotation_angle_deg(error[:3, :3]))
        translation_errors_array = np.asarray(translation_errors)
        rotation_errors_array = np.asarray(rotation_errors)
        return FixedCameraCalibrationResult(
            world_to_camera=world_to_camera,
            translation_rmse_m=float(
                np.sqrt(np.mean(translation_errors_array[indices] ** 2))
            ),
            rotation_rmse_deg=float(
                np.sqrt(np.mean(rotation_errors_array[indices] ** 2))
            ),
            sample_translation_errors_m=translation_errors_array,
            sample_rotation_errors_deg=rotation_errors_array,
        )

    all_indices = np.arange(len(candidates), dtype=int)
    first = solve(all_indices)

    def robust_limit(values: np.ndarray, floor: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 3.5 * 1.4826 * mad)

    retained = np.logical_and(
        first.sample_translation_errors_m
        <= robust_limit(first.sample_translation_errors_m, 0.008),
        first.sample_rotation_errors_deg
        <= robust_limit(first.sample_rotation_errors_deg, 1.5),
    )
    if int(np.count_nonzero(retained)) < minimum_samples:
        retained[:] = True
        return first, retained
    if np.all(retained):
        return first, retained
    return solve(np.flatnonzero(retained)), retained


def _tag_is_safely_visible(
    world_to_camera: np.ndarray,
    world_to_tag: np.ndarray,
    intrinsics: D435RgbIntrinsics,
    border_px: float = 18.0,
) -> bool:
    image, corners_camera = _project_tag_corners(
        world_to_camera, world_to_tag, intrinsics
    )
    if np.any(corners_camera[:, 2] <= 0.20) or np.any(corners_camera[:, 2] >= 1.5):
        return False
    # The visible front of the printed tag has its +Z normal toward the camera.
    camera_to_tag = invert_transform(world_to_camera) @ world_to_tag
    if camera_to_tag[2, 2] > -0.20:
        return False
    if (
        np.any(image[:, 0] < border_px)
        or np.any(image[:, 0] >= intrinsics.width - border_px)
        or np.any(image[:, 1] < border_px)
        or np.any(image[:, 1] >= intrinsics.height - border_px)
    ):
        return False
    area = abs(float(cv2.contourArea(image.astype(np.float32))))
    return area >= 700.0


def generate_visible_motion(
    mj: Any,
    model: Any,
    data: Any,
    world_to_camera: np.ndarray,
    intrinsics: D435RgbIntrinsics,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth, joint-limited motion whose complete tag stays visible."""
    if frame_count < 24:
        raise ValueError("motion needs at least 24 frames for useful excitation")
    addresses = _arm_joint_addresses(mj, model)
    qpos_template = data.qpos.copy()
    centre = qpos_template[addresses].copy()
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count, endpoint=False)
    base_amplitudes = np.asarray([0.22, 0.18, 0.22, 0.18, 0.38, 0.24, 0.50])
    frequencies = np.asarray([1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 2.0])
    phase_offsets = np.asarray([0.0, 0.4, 1.1, 1.8, 0.7, 2.4, 1.3])

    tag_id = _named_id(mj, model, mj.mjtObj.mjOBJ_BODY, TAG_BODY_NAME)
    for scale in np.linspace(1.0, 0.35, 14):
        arm_motion = centre + scale * base_amplitudes * np.sin(
            phase[:, None] * frequencies + phase_offsets
        )
        for column, name in enumerate(ARM_JOINTS):
            joint_id = _named_id(mj, model, mj.mjtObj.mjOBJ_JOINT, name)
            low, high = np.asarray(model.jnt_range[joint_id], dtype=float)
            arm_motion[:, column] = np.clip(
                arm_motion[:, column], low + 0.025, high - 0.025
            )

        all_visible = True
        for arm_position in arm_motion:
            _set_arm_pose(mj, model, data, qpos_template, addresses, arm_position)
            world_to_tag = _pose(data.xpos[tag_id], data.xmat[tag_id])
            if not _tag_is_safely_visible(world_to_camera, world_to_tag, intrinsics):
                all_visible = False
                break
        if all_visible:
            _set_arm_pose(mj, model, data, qpos_template, addresses, centre)
            return arm_motion, addresses, qpos_template

    raise RuntimeError(
        "could not generate a sufficiently large trajectory with the whole tag visible"
    )


def _tag_detector() -> Any:
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError(
            "OpenCV was built without the aruco module and AprilTag dictionaries; "
            "install opencv-contrib-python"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 40
    parameters.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_tag_pose(
    rgb: np.ndarray,
    detector: Any,
    intrinsics: D435RgbIntrinsics,
) -> TagObservation | None:
    """Estimate the tag coordinate system in the D435 RGB optical frame."""
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    detected_corners, identifiers, _ = detector.detectMarkers(gray)
    if identifiers is None:
        return None
    matches = [
        np.asarray(corners, dtype=np.float64).reshape(4, 2)
        for corners, identifier in zip(detected_corners, identifiers.reshape(-1))
        if int(identifier) == TAG_ID
    ]
    if len(matches) != 1:
        return None

    half = TAG_SIZE_M * 0.5
    object_corners = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_corners,
        matches[0],
        intrinsics.matrix,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or float(translation_vector[2, 0]) <= 0.0:
        return None
    projected, _ = cv2.projectPoints(
        object_corners,
        rotation_vector,
        translation_vector,
        intrinsics.matrix,
        np.zeros(5, dtype=np.float64),
    )
    reprojection_error = float(
        np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - matches[0]) ** 2, axis=1)))
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return TagObservation(
        camera_to_tag=make_transform(rotation, translation_vector.reshape(3)),
        image_corners=matches[0],
        reprojection_error_px=reprojection_error,
    )


def _annotate_rgb(
    rgb: np.ndarray,
    observation: TagObservation | None,
    intrinsics: D435RgbIntrinsics,
    sample_number: int,
) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if observation is not None:
        outline = np.rint(observation.image_corners).astype(np.int32)
        cv2.polylines(image, [outline], True, (0, 255, 255), 2, cv2.LINE_AA)
        rotation_vector, _ = cv2.Rodrigues(observation.camera_to_tag[:3, :3])
        cv2.drawFrameAxes(
            image,
            intrinsics.matrix,
            np.zeros(5),
            rotation_vector,
            observation.camera_to_tag[:3, 3],
            TAG_SIZE_M * 0.75,
            2,
        )
    text = (
        f"D435 RGB | tag36h11 id={TAG_ID} | samples={sample_number} | "
        f"detected={'yes' if observation is not None else 'NO'}"
    )
    cv2.putText(
        image,
        text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 255, 40),
        1,
        cv2.LINE_AA,
    )
    return image


def run_simulated_calibration(
    *,
    sample_count: int = 18,
    frames_per_sample: int = 3,
    camera_distance_m: float = 0.50,
    interactive: bool = False,
    fps: float = 10.0,
    intrinsics: D435RgbIntrinsics = D435RgbIntrinsics(),
) -> SimulationCalibrationRun:
    """Render RGB, collect OpenCV/FK pose pairs, and calibrate the camera."""
    if sample_count < 8:
        raise ValueError("at least 8 calibration samples are required")
    if frames_per_sample < 1:
        raise ValueError("frames_per_sample must be positive")
    if fps <= 0.0 or not np.isfinite(fps):
        raise ValueError("fps must be positive")

    mj, model, data, expected_world_to_camera = load_calibration_scene(
        intrinsics, camera_distance_m
    )
    frame_count = sample_count * frames_per_sample
    arm_motion, arm_addresses, qpos_template = generate_visible_motion(
        mj,
        model,
        data,
        expected_world_to_camera,
        intrinsics,
        frame_count,
    )
    detector = _tag_detector()
    tag_body_id = _named_id(mj, model, mj.mjtObj.mjOBJ_BODY, TAG_BODY_NAME)

    try:
        renderer = mj.Renderer(model, height=intrinsics.height, width=intrinsics.width)
    except Exception as exc:
        raise RuntimeError(
            "MuJoCo could not create an RGB renderer. On a headless machine set "
            "MUJOCO_GL=egl before starting Python."
        ) from exc

    world_to_tag: list[np.ndarray] = []
    camera_to_tag: list[np.ndarray] = []
    reprojection_errors: list[float] = []
    detected_frame_count = 0
    last_rgb = np.empty((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
    last_annotated = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
    next_frame_time = time.monotonic()
    window_name = "RealSense D435 RGB - simulated eye-to-hand calibration"

    try:
        for frame_index, arm_position in enumerate(arm_motion):
            _set_arm_pose(
                mj,
                model,
                data,
                qpos_template,
                arm_addresses,
                arm_position,
            )
            renderer.update_scene(data, camera=CAMERA_NAME)
            last_rgb = renderer.render().copy()
            observation = detect_tag_pose(last_rgb, detector, intrinsics)
            if observation is None:
                raise RuntimeError(
                    f"AprilTag was not detected in RGB motion frame {frame_index + 1}/"
                    f"{frame_count}; the calibration motion no longer keeps it visible"
                )
            detected_frame_count += 1

            capture = frame_index % frames_per_sample == 0
            if capture:
                # This is the tag pose in the robot/world frame from FK.  No
                # rendered camera ground truth participates in the estimate.
                world_to_tag.append(_world_to_printed_tag_from_fk(data, tag_body_id))
                camera_to_tag.append(observation.camera_to_tag)
                reprojection_errors.append(observation.reprojection_error_px)

            last_annotated = _annotate_rgb(
                last_rgb, observation, intrinsics, len(world_to_tag)
            )
            if interactive:
                cv2.imshow(window_name, last_annotated)
                next_frame_time += 1.0 / fps
                delay_ms = max(1, int(1000.0 * (next_frame_time - time.monotonic())))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):
                    raise KeyboardInterrupt
    finally:
        renderer.close()
        if interactive:
            cv2.destroyWindow(window_name)

    if len(world_to_tag) != sample_count:
        raise RuntimeError(
            f"expected {sample_count} calibration samples, collected {len(world_to_tag)}"
        )
    calibration, retained = calibrate_fixed_camera_from_tag_poses(
        world_to_tag, camera_to_tag, minimum_samples=8
    )
    print(
        f"Captured {len(world_to_tag)} RGB/FK pairs; tag detected in "
        f"{detected_frame_count}/{frame_count} motion frames; mean PnP reprojection "
        f"error={np.mean(reprojection_errors):.3f} px"
    )
    return SimulationCalibrationRun(
        calibration=calibration,
        retained=retained,
        expected_world_to_camera=expected_world_to_camera,
        world_to_tag=tuple(world_to_tag),
        camera_to_tag=tuple(camera_to_tag),
        detected_frame_count=detected_frame_count,
        motion_frame_count=frame_count,
        last_rgb=last_rgb,
        last_annotated_bgr=last_annotated,
    )


def _format_transform(name: str, transform: np.ndarray) -> str:
    translation = transform[:3, 3]
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    return (
        f"{name}:\n{np.array2string(transform, precision=6, suppress_small=True)}\n"
        f"  xyz_m=[{', '.join(f'{value:.6f}' for value in translation)}]\n"
        f"  quaternion_xyzw=[{', '.join(f'{value:.7f}' for value in quaternion)}]"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="do not open the RGB window"
    )
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--frames-per-sample", type=int, default=3)
    parser.add_argument("--camera-distance", type=float, default=0.50, metavar="METRES")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="interactive motion speed in rendered frames per second (default: 10)",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="optionally save the final annotated RGB frame",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run = run_simulated_calibration(
            sample_count=args.samples,
            frames_per_sample=args.frames_per_sample,
            camera_distance_m=args.camera_distance,
            interactive=not args.headless,
            fps=args.fps,
        )
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        return 130
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        _format_transform(
            "calibrated T_world_camera_rgb_optical", run.calibration.world_to_camera
        )
    )
    print(
        _format_transform(
            "ground truth T_world_camera_rgb_optical", run.expected_world_to_camera
        )
    )
    print(
        f"camera error: translation={1000.0 * run.camera_translation_error_m:.2f} mm, "
        f"rotation={run.camera_rotation_error_deg:.3f} deg; retained "
        f"{int(np.count_nonzero(run.retained))}/{len(run.retained)} samples; "
        f"fit RMSE={1000.0 * run.calibration.translation_rmse_m:.2f} mm / "
        f"{run.calibration.rotation_rmse_deg:.3f} deg"
    )
    if args.output_image is not None:
        output = args.output_image.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), run.last_annotated_bgr):
            print(f"error: could not write {output}", file=sys.stderr)
            return 1
        print(f"Saved final annotated RGB frame: {output}")
    return 0


def test_nominal_d435_rgb_projection_contract() -> None:
    intrinsics = D435RgbIntrinsics()
    intrinsics.validate()
    assert (intrinsics.width, intrinsics.height) == (1920, 1080)
    assert intrinsics.matrix.shape == (3, 3)
    assert intrinsics.fovy_deg == pytest.approx(42.636, abs=0.01)


def test_simulated_rgb_calibration_recovers_fixed_camera() -> None:
    """Pytest smoke test of MuJoCo RGB -> OpenCV PnP -> FK -> calibration."""
    pytest.importorskip("mujoco")
    run = run_simulated_calibration(
        sample_count=12,
        frames_per_sample=2,
        interactive=False,
    )
    assert run.detected_frame_count == run.motion_frame_count
    assert int(np.count_nonzero(run.retained)) >= 8
    assert run.camera_translation_error_m < 0.025
    assert run.camera_rotation_error_deg < 4.0


def test_real_auto_waypoint_interpolations_keep_tag_in_simulated_view() -> None:
    """Check endpoints and slow controller interpolation, including the first move."""
    pytest.importorskip("mujoco")
    mj, model, data, world_to_camera = load_calibration_scene()
    addresses = _arm_joint_addresses(mj, model)
    qpos_template = data.qpos.copy()
    tag_body_id = _named_id(mj, model, mj.mjtObj.mjOBJ_BODY, TAG_BODY_NAME)
    poses = [qpos_template[addresses].copy(), *map(np.asarray, AUTO_WAYPOINTS)]

    for start, goal in zip(poses, poses[1:]):
        for fraction in np.linspace(0.0, 1.0, 31):
            arm_position = (1.0 - fraction) * start + fraction * goal
            _set_arm_pose(
                mj, model, data, qpos_template, addresses, arm_position
            )
            world_to_tag = _pose(
                data.xpos[tag_body_id], data.xmat[tag_body_id]
            )
            assert _tag_is_safely_visible(
                world_to_camera, world_to_tag, D435RgbIntrinsics()
            )


# Imported late so the direct-run path has no dependency on pytest.
try:  # pragma: no cover - only absent when the file is run outside a test image.
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    pytest = None


if __name__ == "__main__":
    raise SystemExit(main())
