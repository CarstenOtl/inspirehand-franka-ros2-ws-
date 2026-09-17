"""The full-resolution camera launch: that it builds, and that it asks for the right stream.

Starting a RealSense needs the device; checking that this launch asks for the
D415's largest colour mode with depth aligned to it and the RGBD composite
enabled does not. Those are the choices ``--record-rgbd`` depends on, so they
are the ones under test.
"""

import pytest

launch = pytest.importorskip("launch")

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription  # noqa: E402

from inspire_franka_trajectory_replay.launch_files import rgbd_camera  # noqa: E402
from inspire_franka_trajectory_replay.rgbd_recording import (  # noqa: E402
    D415_FULL_COLOR,
    D415_FULL_DEPTH,
)


def _text(substitution):
    if isinstance(substitution, str):
        return substitution
    if isinstance(substitution, (list, tuple)):
        return "".join(_text(part) for part in substitution)
    return getattr(substitution, "text", None) or str(substitution)


def _include():
    entities = rgbd_camera.generate_launch_description().entities
    includes = [entity for entity in entities if isinstance(entity, IncludeLaunchDescription)]
    assert len(includes) == 1
    return includes[0]


def test_the_launch_description_builds():
    assert rgbd_camera.generate_launch_description() is not None


def test_the_profiles_are_the_d415s_largest_modes_at_the_recording_rate():
    assert rgbd_camera.COLOR_PROFILE == f"{D415_FULL_COLOR[0]}x{D415_FULL_COLOR[1]}x30"
    assert rgbd_camera.DEPTH_PROFILE == f"{D415_FULL_DEPTH[0]}x{D415_FULL_DEPTH[1]}x30"
    assert rgbd_camera.COLOR_PROFILE == "1920x1080x30"


def test_the_composite_sync_and_alignment_are_fixed_not_optional():
    fixed = rgbd_camera.FIXED_CAMERA_ARGUMENTS
    assert fixed["enable_rgbd"] == "true"
    assert fixed["enable_sync"] == "true"
    assert fixed["align_depth.enable"] == "true"
    assert fixed["device_type"] == "d415"
    assert fixed["pointcloud.enable"] == "false"


def test_the_realsense_launch_is_included_with_those_arguments():
    include = _include()
    arguments = {_text(name): value for name, value in include.launch_arguments}
    for name, value in rgbd_camera.FIXED_CAMERA_ARGUMENTS.items():
        assert _text(arguments[name]) == value, name
    for name in ("rgb_camera.color_profile", "depth_module.depth_profile", "serial_no"):
        assert name in arguments
    source = _text(include.launch_description_source.location)
    assert "realsense2_camera" in source and "rs_launch.py" in source


def test_the_operator_can_choose_a_lower_profile_but_gets_full_resolution_by_default():
    declared = {
        entity.name: entity
        for entity in rgbd_camera.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert _text(declared["color_profile"].default_value) == "1920x1080x30"
    assert _text(declared["depth_profile"].default_value) == "1280x720x30"
    assert {"serial_no", "camera_namespace", "camera_name", "initial_reset"} <= set(declared)
    assert _text(declared["initial_reset"].default_value) == "false"
