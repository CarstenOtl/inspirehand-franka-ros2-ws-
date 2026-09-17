from types import SimpleNamespace

import numpy as np
import pytest

from utils.camera_calibration import CameraIntrinsics, load_camera_calibration, prepare_rgbd
from utils.compare_depth_filters import (
    PolicyView,
    SoftwareDepthPipeline,
    build_parser,
    driver_launch_arguments,
    fill_fraction,
    plane_rms_mm,
    temporal_noise_mm,
)


def _plane(height=180, width=320, noise_m=0.0, seed=0):
    v, u = np.mgrid[0:height, 0:width].astype(np.float32)
    depth = 0.8 + 1e-4 * u + 2e-4 * v
    return depth + np.random.default_rng(seed).normal(0.0, noise_m, depth.shape).astype(np.float32)


def test_plane_rms_recovers_the_noise_on_a_tilted_plane_and_ignores_holes():
    depth = _plane(noise_m=0.002)
    depth[10:20, 10:20] = 0.0
    assert plane_rms_mm(_plane(), (0.0, 0.0, 1.0, 1.0)) < 1e-2
    assert plane_rms_mm(depth, (0.0, 0.0, 1.0, 1.0)) == pytest.approx(2.0, rel=0.05)
    assert plane_rms_mm(depth, None) is None
    assert plane_rms_mm(np.zeros_like(depth), (0.0, 0.0, 1.0, 1.0)) is None


def test_temporal_noise_uses_pixels_valid_in_every_frame():
    frames = [_plane(noise_m=0.001, seed=seed) for seed in range(20)]
    frames[3][:, :100] = 0.0
    assert temporal_noise_mm(frames) == pytest.approx(1.0, rel=0.1)
    assert temporal_noise_mm(frames[:2]) is None
    assert fill_fraction(frames[3]) == pytest.approx(220 / 320)


def test_policy_view_is_prepare_rgbd_for_the_profile_stream():
    profile = load_camera_calibration()
    live = profile.source_intrinsics
    view = PolicyView(profile, live)
    depth_mm = np.random.default_rng(1).integers(0, 2000, (480, 640)).astype(np.uint16)
    expected = prepare_rgbd(
        np.zeros((480, 640, 3), np.uint8), depth_mm, profile, depth_units="millimetres"
    ).depth[0]
    assert view.exact
    np.testing.assert_array_equal(view.depth_m(depth_mm), expected)
    np.testing.assert_allclose(
        view.intrinsics.camera_matrix, profile.policy_intrinsics.camera_matrix, atol=1e-6
    )


def test_policy_view_falls_back_to_a_scaled_1080p_stream_and_counts_dp3_points():
    profile = load_camera_calibration()
    k = (1380.0, 0.0, 960.0, 0.0, 1380.0, 540.0, 0.0, 0.0, 1.0)
    view = PolicyView(profile, CameraIntrinsics(width=1920, height=1080, camera_matrix=k))
    assert not view.exact and view.crop is None
    assert view.intrinsics.camera_matrix[0] == pytest.approx(230.0)
    depth = view.depth_m(np.full((1080, 1920), 900, np.uint16))
    assert depth.shape == (180, 320) and np.allclose(depth, 0.9)
    assert 0 < view.dp3_points(depth) < depth.size
    assert view.dp3_points(np.zeros_like(depth)) == 0


def test_driver_launch_arguments_name_the_realsense_parameters():
    state = {"spatial": True, "temporal": True, "hole_filling": False, "disparity": False}
    assert driver_launch_arguments(state) == "spatial_filter.enable:=true temporal_filter.enable:=true"
    assert "no filter" in driver_launch_arguments(dict.fromkeys(state, False))


def _info(width, height, f):
    return SimpleNamespace(
        width=width, height=height, k=[f, 0.0, width / 2, 0.0, f, height / 2, 0.0, 0.0, 1.0], d=[0.0] * 5
    )


def test_software_pipeline_aligns_and_the_spatial_filter_smooths():
    pytest.importorskip("pyrealsense2")
    args = build_parser().parse_args([])
    info = _info(320, 180, 230.0)
    extrinsics = SimpleNamespace(rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[0, 0, 0])
    pipeline = SoftwareDepthPipeline(info, info, extrinsics, args)
    noisy_mm = (_plane(noise_m=0.004) * 1000).astype(np.uint16)

    # Identical intrinsics and identity extrinsics: librealsense's alignment
    # keeps the depth it maps, to the 1 mm its splatting picks between
    # neighbours on a slope, and leaves a few pixel holes. (Equality with the
    # driver's own alignment is checked live at start-up.)
    smooth_mm = (_plane() * 1000).astype(np.uint16)
    aligned = pipeline.align_only(smooth_mm)
    mapped = aligned > 0
    assert aligned.shape == smooth_mm.shape and mapped.mean() > 0.9
    assert np.all(np.abs(aligned[mapped].astype(int) - smooth_mm[mapped]) <= 1)

    off = dict(spatial=False, temporal=False, hole_filling=False, disparity=False)
    unfiltered, _ = pipeline.process(noisy_mm, off)
    np.testing.assert_array_equal(unfiltered, noisy_mm)
    for state in (dict(off, spatial=True), dict(off, spatial=True, disparity=True)):
        filtered, filtered_aligned = pipeline.process(noisy_mm, state)
        roi = (0.1, 0.1, 0.9, 0.9)
        assert plane_rms_mm(filtered.astype(np.float32) / 1000, roi) < 0.8 * plane_rms_mm(
            noisy_mm.astype(np.float32) / 1000, roi
        )
        assert filtered_aligned.shape == noisy_mm.shape
