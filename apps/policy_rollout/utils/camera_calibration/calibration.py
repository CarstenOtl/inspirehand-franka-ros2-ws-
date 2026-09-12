"""Single source of truth for the DP3 camera and RGB-D input conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _tuple(values: Any, width: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != width or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain {width} finite values")
    return result


@dataclass(frozen=True)
class CropRectangle:
    """Pixel rectangle kept from a larger frame before it is resized."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.width < 1 or self.height < 1:
            raise ValueError(
                "crop rectangle needs non-negative offsets and a positive size"
            )

    def fits(self, width: int, height: int) -> bool:
        return self.x + self.width <= width and self.y + self.height <= height

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if not self.fits(width, height):
            raise ValueError(f"crop {self} exceeds a {width}x{height} frame")
        return image[self.y : self.y + self.height, self.x : self.x + self.width]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    camera_matrix: tuple[float, ...]
    distortion_model: str = "plumb_bob"
    distortion_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("camera dimensions must be positive")
        if len(self.camera_matrix) != 9:
            raise ValueError("camera_matrix must contain nine values")
        if self.camera_matrix[0] <= 0 or self.camera_matrix[4] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.camera_matrix[6:] != (0.0, 0.0, 1.0):
            raise ValueError("camera_matrix must end in [0, 0, 1]")

    def cropped(self, crop: CropRectangle) -> "CameraIntrinsics":
        if not crop.fits(self.width, self.height):
            raise ValueError("crop rectangle exceeds the calibrated frame")
        matrix = list(self.camera_matrix)
        matrix[2] -= crop.x
        matrix[5] -= crop.y
        return CameraIntrinsics(
            width=crop.width,
            height=crop.height,
            camera_matrix=tuple(matrix),
            distortion_model=self.distortion_model,
            distortion_coefficients=self.distortion_coefficients,
        )

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        sx, sy = width / self.width, height / self.height
        if not math.isclose(sx, sy, abs_tol=1e-9):
            raise ValueError("camera resize must preserve the calibrated aspect ratio")
        matrix = list(self.camera_matrix)
        matrix[0] *= sx
        matrix[2] *= sx
        matrix[4] *= sy
        matrix[5] *= sy
        return CameraIntrinsics(
            width=width,
            height=height,
            camera_matrix=tuple(matrix),
            distortion_model=self.distortion_model,
            distortion_coefficients=self.distortion_coefficients,
        )

    def policy_view(
        self, crop: CropRectangle | None, width: int, height: int
    ) -> "CameraIntrinsics":
        """Intrinsics of this frame after an optional crop and an isotropic resize."""

        view = self if crop is None else self.cropped(crop)
        return view.scaled(width, height)


@dataclass(frozen=True)
class CameraPose:
    parent_frame_id: str
    child_frame_id: str
    translation_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(value * value for value in self.rotation_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError("camera pose quaternion must be normalized")


@dataclass(frozen=True)
class CameraCalibrationProfile:
    source_path: Path
    training_model: str
    frame_id: str
    source_intrinsics: CameraIntrinsics
    source_crop: CropRectangle | None
    policy_intrinsics: CameraIntrinsics
    training_world_pose: CameraPose
    dp3_point_cloud: dict[str, Any]
    color_topic: str
    depth_topic: str
    camera_info_topic: str
    physical_model: str
    serial_number: str
    physical_stream_size: tuple[int, int] | None
    physical_stream_crop: CropRectangle | None
    maximum_intrinsics_error_px: float
    measured_world_pose: CameraPose | None
    measured_pose_status: str
    maximum_translation_error_m: float
    maximum_rotation_error_deg: float

    @property
    def hardware_ready(self) -> bool:
        return not self.hardware_blockers()

    @property
    def policy_shape(self) -> tuple[int, int]:
        return (self.policy_intrinsics.height, self.policy_intrinsics.width)

    def frame_views(self) -> tuple[tuple[str, tuple[int, int], CropRectangle | None], ...]:
        """Full-frame shapes ``prepare_rgbd`` accepts, each with its crop."""

        views = [
            (
                "training source",
                (self.source_intrinsics.height, self.source_intrinsics.width),
                self.source_crop,
            )
        ]
        if self.physical_stream_size is not None:
            width, height = self.physical_stream_size
            if (height, width) != views[0][1]:
                views.append(("physical stream", (height, width), self.physical_stream_crop))
        return tuple(views)

    def crop_for_frame(self, shape: tuple[int, int]) -> CropRectangle | None:
        for _label, view_shape, crop in self.frame_views():
            if tuple(shape) == view_shape:
                return crop
        accepted = ", ".join(
            f"the {label} {view_shape}" for label, view_shape, _crop in self.frame_views()
        )
        raise ValueError(
            f"camera frame must be {accepted}, or the policy view "
            f"{self.policy_shape}; got {tuple(shape)}"
        )

    def assert_live_camera_info(
        self, width: int, height: int, camera_matrix: Any
    ) -> CameraIntrinsics:
        """Check a live ``CameraInfo`` maps onto the checkpoint's policy view."""

        if self.physical_stream_size is None:
            raise ValueError("camera profile declares no physical colour stream")
        expected_width, expected_height = self.physical_stream_size
        if (int(width), int(height)) != (expected_width, expected_height):
            raise ValueError(
                f"live colour stream is {int(width)}x{int(height)}; the profile "
                f"expects {expected_width}x{expected_height}"
            )
        live = CameraIntrinsics(
            width=int(width),
            height=int(height),
            camera_matrix=_tuple(camera_matrix, 9, "camera_matrix"),
        )
        view = live.policy_view(
            self.physical_stream_crop,
            self.policy_intrinsics.width,
            self.policy_intrinsics.height,
        )
        error = max(
            abs(actual - expected)
            for actual, expected in zip(
                view.camera_matrix, self.policy_intrinsics.camera_matrix
            )
        )
        if error > self.maximum_intrinsics_error_px:
            raise ValueError(
                f"live intrinsics map onto the policy view with {error:.3f} px error "
                f"(limit {self.maximum_intrinsics_error_px:.3f} px)"
            )
        return view

    def training_pose_error(self) -> tuple[float, float]:
        """Return physical-vs-training translation metres and rotation degrees."""

        if self.measured_world_pose is None:
            raise RuntimeError("measured camera pose is unavailable")
        expected = self.training_world_pose
        measured = self.measured_world_pose
        translation = float(
            np.linalg.norm(
                np.asarray(measured.translation_m) - np.asarray(expected.translation_m)
            )
        )
        measured_q = np.asarray(measured.rotation_wxyz, dtype=float)
        expected_q = np.asarray(expected.rotation_wxyz, dtype=float)
        dot = abs(
            float(
                np.dot(measured_q, expected_q)
                / (np.linalg.norm(measured_q) * np.linalg.norm(expected_q))
            )
        )
        rotation = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
        return translation, rotation

    def hardware_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.physical_model != self.training_model:
            blockers.append(
                f"physical camera model {self.physical_model!r} does not match "
                f"training model {self.training_model!r}"
            )
        if not self.serial_number or self.serial_number.upper() == "PLACEHOLDER":
            blockers.append("physical camera serial number is not recorded")
        if self.physical_stream_size is None:
            blockers.append("physical colour stream resolution and crop are not recorded")
        if self.measured_pose_status != "validated" or self.measured_world_pose is None:
            blockers.append(
                "measured world-to-camera pose is not validated "
                f"(status: {self.measured_pose_status})"
            )
        else:
            measured = self.measured_world_pose
            expected = self.training_world_pose
            if (
                measured.parent_frame_id != expected.parent_frame_id
                or measured.child_frame_id != expected.child_frame_id
            ):
                blockers.append("measured and training camera pose frames do not match")
            translation, rotation = self.training_pose_error()
            if translation > self.maximum_translation_error_m:
                blockers.append(
                    f"camera translation differs from training by {translation:.4f} m "
                    f"(limit {self.maximum_translation_error_m:.4f} m)"
                )
            if rotation > self.maximum_rotation_error_deg:
                blockers.append(
                    f"camera rotation differs from training by {rotation:.3f} deg "
                    f"(limit {self.maximum_rotation_error_deg:.3f} deg)"
                )
        return tuple(blockers)

    def assert_checkpoint_compatible(self, checkpoint_config: dict[str, Any]) -> None:
        if checkpoint_config.get("observation_mode") != "vision":
            raise ValueError("rollout requires a vision checkpoint")
        vision = checkpoint_config.get("vision_encoder_config") or {}
        encoder = vision.get("encoder", {})
        if encoder.get("type") != "rgb_pointcloud_dp3_encoder":
            raise ValueError("rollout requires the RGB point-cloud DP3 encoder")
        input_config = vision.get("input", {})
        expected_shape = self.policy_shape
        if tuple(input_config.get("image_shape", ())) != expected_shape:
            raise ValueError(
                f"checkpoint image shape must be {expected_shape}, got "
                f"{tuple(input_config.get('image_shape', ()))}"
            )
        actual_k = np.asarray(input_config.get("camera_matrix", ()), dtype=float)
        expected_k = np.asarray(self.policy_intrinsics.camera_matrix, dtype=float)
        if actual_k.shape != (9,) or not np.allclose(actual_k, expected_k, atol=1e-5):
            raise ValueError(
                "checkpoint camera matrix does not match calibration profile"
            )
        point_cloud = encoder.get("point_cloud", {})
        for key in (
            "crop_min_m",
            "crop_max_m",
            "xyz_center_m",
            "xyz_scale_m",
        ):
            expected = np.asarray(self.dp3_point_cloud[key], dtype=float)
            actual = np.asarray(point_cloud.get(key, ()), dtype=float)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, atol=1e-6
            ):
                raise ValueError(f"checkpoint DP3 {key} does not match camera profile")


@dataclass(frozen=True)
class PreparedRgbd:
    rgb: np.ndarray
    depth: np.ndarray
    valid_mask: np.ndarray


def default_profile_path() -> Path:
    return Path(__file__).with_name("fr3_realsense_dp3.yaml")


def _intrinsics(document: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(document["width"]),
        height=int(document["height"]),
        camera_matrix=_tuple(document["camera_matrix"], 9, "camera_matrix"),
        distortion_model=str(document.get("distortion_model", "plumb_bob")),
        distortion_coefficients=tuple(
            float(value) for value in document.get("distortion_coefficients", ())
        ),
    )


def _crop(document: dict[str, Any] | None) -> CropRectangle | None:
    if document is None:
        return None
    return CropRectangle(
        x=int(document["x"]),
        y=int(document["y"]),
        width=int(document["width"]),
        height=int(document["height"]),
    )


def _pose(document: dict[str, Any]) -> CameraPose:
    return CameraPose(
        parent_frame_id=str(document["parent_frame_id"]),
        child_frame_id=str(document["child_frame_id"]),
        translation_m=_tuple(document["translation_m"], 3, "translation_m"),
        rotation_wxyz=_tuple(document["rotation_wxyz"], 4, "rotation_wxyz"),
    )


def _assert_isotropic(
    width: int, height: int, policy: CameraIntrinsics, label: str
) -> None:
    if width * policy.height != height * policy.width:
        raise ValueError(
            f"{label} {width}x{height} does not have the policy aspect ratio "
            f"{policy.width}x{policy.height}"
        )


def load_camera_calibration(path: str | Path | None = None) -> CameraCalibrationProfile:
    source = Path(path or default_profile_path()).expanduser().resolve()
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"unsupported camera calibration schema: {source}")
    training = document["training_camera"]
    physical = document["physical_camera"]
    source_intrinsics = _intrinsics(training["color"])
    policy_document = training["policy_input"]
    policy_intrinsics = _intrinsics(policy_document)
    source_crop = _crop(policy_document.get("source_crop_px"))
    derived = source_intrinsics.policy_view(
        source_crop, policy_intrinsics.width, policy_intrinsics.height
    )
    if not np.allclose(derived.camera_matrix, policy_intrinsics.camera_matrix, atol=1e-6):
        raise ValueError(
            "policy camera matrix is not the exact cropped and scaled source calibration"
        )

    stream = physical.get("color_stream")
    stream_size = None
    stream_crop = None
    if stream is not None:
        stream_size = (int(stream["width"]), int(stream["height"]))
        stream_crop = _crop(stream.get("crop_px"))
        if stream_crop is None:
            _assert_isotropic(*stream_size, policy_intrinsics, "physical colour stream")
        else:
            if not stream_crop.fits(*stream_size):
                raise ValueError("physical stream crop exceeds the stream resolution")
            _assert_isotropic(
                stream_crop.width,
                stream_crop.height,
                policy_intrinsics,
                "physical colour stream crop",
            )

    measured = physical["measured_world_pose"]
    measured_pose = None
    if (
        measured.get("translation_m") is not None
        and measured.get("rotation_wxyz") is not None
    ):
        measured_pose = _pose(measured)
    limits = physical["maximum_training_pose_error"]
    return CameraCalibrationProfile(
        source_path=source,
        training_model=str(training["model"]),
        frame_id=str(training["frame_id"]),
        source_intrinsics=source_intrinsics,
        source_crop=source_crop,
        policy_intrinsics=policy_intrinsics,
        training_world_pose=_pose(training["world_pose"]),
        dp3_point_cloud=dict(training["dp3_point_cloud"]),
        color_topic=str(physical["color_topic"]),
        depth_topic=str(physical["depth_topic"]),
        camera_info_topic=str(physical["camera_info_topic"]),
        physical_model=str(physical["model"]),
        serial_number=str(physical.get("serial_number", "PLACEHOLDER")),
        physical_stream_size=stream_size,
        physical_stream_crop=stream_crop,
        maximum_intrinsics_error_px=float(
            physical.get("maximum_intrinsics_error_px", 0.5)
        ),
        measured_world_pose=measured_pose,
        measured_pose_status=str(measured.get("status", "PLACEHOLDER")).lower(),
        maximum_translation_error_m=float(limits["translation_m"]),
        maximum_rotation_error_deg=float(limits["rotation_deg"]),
    )


def _nearest_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    rows = np.minimum(
        ((np.arange(height) + 0.5) * source_height / height).astype(int),
        source_height - 1,
    )
    columns = np.minimum(
        ((np.arange(width) + 0.5) * source_width / width).astype(int),
        source_width - 1,
    )
    return image[rows[:, None], columns[None, :]]


def prepare_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    profile: CameraCalibrationProfile,
    *,
    depth_units: str,
) -> PreparedRgbd:
    """Validate, crop, resize, and normalize one aligned camera observation.

    Frames may arrive at the policy view directly, at the training source
    resolution, or at the physical colour-stream resolution; the latter two
    are cropped with the profile's rectangle and then resized isotropically.
    """

    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    expected_policy = profile.policy_shape
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have shape [H,W,3]")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("depth must be pixel-aligned with RGB")
    if rgb.shape[:2] != expected_policy:
        crop = profile.crop_for_frame(rgb.shape[:2])
        if crop is not None:
            rgb = crop.apply(rgb)
            depth = crop.apply(depth)
        rgb = _nearest_resize(rgb, *expected_policy)
        depth = _nearest_resize(depth, *expected_policy)
    rgb_float = rgb.astype(np.float32, copy=False)
    if np.issubdtype(rgb.dtype, np.integer):
        rgb_float = rgb_float / 255.0
    elif not np.isfinite(rgb_float).all() or rgb_float.min() < 0 or rgb_float.max() > 1:
        raise ValueError("floating-point RGB must be finite and in [0,1]")
    if depth_units == "millimetres":
        depth_m = depth.astype(np.float32) * 1.0e-3
    elif depth_units == "metres":
        depth_m = depth.astype(np.float32)
    else:
        raise ValueError("depth_units must be 'metres' or 'millimetres'")
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_m = np.where(valid, depth_m, 0.0).astype(np.float32, copy=False)
    return PreparedRgbd(
        rgb=np.ascontiguousarray(rgb_float.transpose(2, 0, 1)),
        depth=np.ascontiguousarray(depth_m[None]),
        valid_mask=np.ascontiguousarray(valid[None]),
    )


__all__ = [
    "CameraCalibrationProfile",
    "CameraIntrinsics",
    "CameraPose",
    "CropRectangle",
    "PreparedRgbd",
    "default_profile_path",
    "load_camera_calibration",
    "prepare_rgbd",
]
