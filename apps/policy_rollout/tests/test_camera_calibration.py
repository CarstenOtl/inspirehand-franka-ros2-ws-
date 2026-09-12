from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from utils.camera_calibration import (
    CropRectangle,
    load_camera_calibration,
    prepare_rgbd,
)

APP_ROOT = Path(__file__).resolve().parents[1]
STUDENT_CHECKPOINT = (
    APP_ROOT
    / "checkpoints"
    / "sequential_threading_cycle10_hybrid_teacher_d415_20ep"
    / "checkpoint.pt"
)

# /camera/camera/color/camera_info as published by this workcell's D415
# (serial 349222064041) at 640x480 on 2026-09-12.
LIVE_640X480_K = (
    602.0516357421875, 0.0, 307.9754943847656,
    0.0, 600.6934204101562, 237.92767333984375,
    0.0, 0.0, 1.0,
)


def test_checked_in_camera_profile_derives_policy_view_from_640x480_calibration():
    profile = load_camera_calibration()
    assert profile.training_model == "Intel RealSense D415"
    assert profile.physical_model == "Intel RealSense D415"
    assert (profile.source_intrinsics.width, profile.source_intrinsics.height) == (
        640,
        480,
    )
    assert profile.source_crop == CropRectangle(x=0, y=60, width=640, height=360)
    assert (profile.policy_intrinsics.width, profile.policy_intrinsics.height) == (
        320,
        180,
    )
    derived = profile.source_intrinsics.policy_view(profile.source_crop, 320, 180)
    assert derived.camera_matrix == pytest.approx(
        profile.policy_intrinsics.camera_matrix, abs=1e-9
    )
    assert profile.policy_intrinsics.camera_matrix[0] == pytest.approx(
        profile.source_intrinsics.camera_matrix[0] / 2.0
    )
    assert profile.policy_intrinsics.camera_matrix[5] == pytest.approx(
        (profile.source_intrinsics.camera_matrix[5] - 60.0) / 2.0
    )
    assert profile.training_world_pose.parent_frame_id == "fr3_link0"
    assert profile.rgbd_topic == "/camera/camera/rgbd"


def test_checked_in_camera_profile_is_hardware_ready():
    profile = load_camera_calibration()
    assert profile.hardware_ready
    blockers = profile.hardware_blockers()
    assert profile.serial_number == "349222064041"
    assert blockers == ()


def test_live_640x480_camera_info_maps_onto_the_policy_view():
    profile = load_camera_calibration()
    assert profile.physical_stream_size == (640, 480)
    assert profile.physical_stream_crop == CropRectangle(
        x=0, y=60, width=640, height=360
    )
    view = profile.assert_live_camera_info(640, 480, LIVE_640X480_K)
    assert view.camera_matrix == pytest.approx(
        profile.policy_intrinsics.camera_matrix, abs=1e-3
    )
    with pytest.raises(ValueError, match="1280x720"):
        profile.assert_live_camera_info(1280, 720, LIVE_640X480_K)
    shifted = list(LIVE_640X480_K)
    shifted[2] += 4.0  # two policy pixels
    with pytest.raises(ValueError, match="px error"):
        profile.assert_live_camera_info(640, 480, shifted)


def test_hardware_ready_requires_a_validated_matching_pose_and_camera_model():
    profile = load_camera_calibration()
    ready = replace(
        profile,
        physical_model=profile.training_model,
        serial_number="000000000000",
        measured_pose_status="validated",
        measured_world_pose=profile.training_world_pose,
    )
    assert ready.hardware_ready
    assert ready.training_pose_error() == pytest.approx((0.0, 0.0))
    shifted = replace(
        ready,
        measured_world_pose=replace(
            ready.training_world_pose,
            translation_m=tuple(
                np.asarray(ready.training_world_pose.translation_m) + (0.1, 0.0, 0.0)
            ),
        ),
    )
    assert not shifted.hardware_ready
    assert any("translation differs" in item for item in shifted.hardware_blockers())


def test_prepare_rgbd_converts_units_layout_and_invalid_depth():
    profile = load_camera_calibration()
    rgb = np.full((180, 320, 3), 255, dtype=np.uint8)
    depth = np.full((180, 320), 420, dtype=np.uint16)
    depth[0, 0] = 0
    prepared = prepare_rgbd(rgb, depth, profile, depth_units="millimetres")
    assert prepared.rgb.shape == (3, 180, 320)
    assert prepared.depth.shape == (1, 180, 320)
    assert prepared.valid_mask.shape == (1, 180, 320)
    assert prepared.rgb.dtype == np.float32
    assert prepared.rgb.max() == 1.0
    assert prepared.depth[0, 1, 1] == pytest.approx(0.42)
    assert prepared.depth[0, 0, 0] == 0.0
    assert not prepared.valid_mask[0, 0, 0]


@pytest.mark.parametrize(
    ("shape", "crop"),
    [((480, 640), CropRectangle(x=0, y=60, width=640, height=360))],
)
def test_prepare_rgbd_crops_full_frames_before_resizing(shape, crop):
    profile = load_camera_calibration()
    # Everything outside the crop is poisoned; nothing of it may survive.
    depth = np.full(shape, 9.0, dtype=np.float32)
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    depth[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width] = 0.42
    rgb[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width] = 255
    prepared = prepare_rgbd(rgb, depth, profile, depth_units="metres")
    assert prepared.depth.shape == (1, 180, 320)
    assert np.all(prepared.depth == pytest.approx(0.42))
    assert np.all(prepared.rgb == 1.0)


def test_prepare_rgbd_rejects_unmodelled_frame_shapes():
    profile = load_camera_calibration()
    with pytest.raises(ValueError, match="policy view"):
        prepare_rgbd(
            np.zeros((360, 640, 3), dtype=np.uint8),
            np.ones((360, 640), dtype=np.float32),
            profile,
            depth_units="metres",
        )


@pytest.mark.skipif(
    not STUDENT_CHECKPOINT.is_file(), reason="student checkpoint not copied in"
)
def test_checked_in_profile_matches_the_student_checkpoint():
    torch = pytest.importorskip("torch")
    payload = torch.load(STUDENT_CHECKPOINT, map_location="cpu", weights_only=False)
    load_camera_calibration().assert_checkpoint_compatible(payload["config"])
