from dataclasses import replace

import numpy as np
import pytest

from utils.camera_calibration import load_camera_calibration, prepare_rgbd


def test_checked_in_camera_profile_has_exact_policy_scaling_and_is_not_ready():
    profile = load_camera_calibration()
    assert profile.training_model == "Intel RealSense D415"
    assert profile.physical_model == "Intel RealSense D415"
    assert (profile.source_intrinsics.width, profile.source_intrinsics.height) == (
        1280,
        720,
    )
    assert (profile.policy_intrinsics.width, profile.policy_intrinsics.height) == (
        320,
        180,
    )
    assert profile.policy_intrinsics.camera_matrix[0] == pytest.approx(
        profile.source_intrinsics.camera_matrix[0] / 4.0
    )
    assert not profile.hardware_ready
    assert profile.hardware_blockers()


def test_hardware_ready_requires_a_validated_matching_pose_and_camera_model():
    profile = load_camera_calibration()
    ready = replace(
        profile,
        physical_model=profile.training_model,
        measured_pose_status="validated",
        measured_world_pose=profile.training_world_pose,
    )
    assert ready.hardware_ready
    assert ready.training_pose_error() == pytest.approx((0.0, 0.0))
    shifted = replace(
        ready,
        measured_world_pose=replace(
            ready.training_world_pose,
            translation_m=(1.0, 0.0, 0.4),
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


def test_prepare_rgbd_rejects_unmodelled_aspect_ratio():
    profile = load_camera_calibration()
    with pytest.raises(ValueError, match="source .* or policy"):
        prepare_rgbd(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640), dtype=np.float32),
            profile,
            depth_units="metres",
        )
