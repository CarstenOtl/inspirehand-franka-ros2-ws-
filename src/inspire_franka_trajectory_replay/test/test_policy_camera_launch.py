"""The policy rollout's camera launch: that it builds, and that it asks for the right stream.

Starting a RealSense needs the device; checking that this launch asks for the
calibrated 640x480 colour mode at the profile's snapshot rate, with depth
aligned to it and the RGBD composite enabled, does not. Those are the choices
the rollout depends on, so they are the ones under test -- along with
auto-exposure priority, which five hardware runs aborted without.
"""

import pytest

launch = pytest.importorskip("launch")
yaml = pytest.importorskip("yaml")

from pathlib import Path  # noqa: E402

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription  # noqa: E402

from inspire_franka_trajectory_replay.launch_files import policy_camera  # noqa: E402


def _text(substitution):
    if isinstance(substitution, str):
        return substitution
    if isinstance(substitution, (list, tuple)):
        return "".join(_text(part) for part in substitution)
    return getattr(substitution, "text", None) or str(substitution)


def _include():
    entities = policy_camera.generate_launch_description().entities
    includes = [entity for entity in entities if isinstance(entity, IncludeLaunchDescription)]
    assert len(includes) == 1
    return includes[0]


def test_the_launch_description_builds():
    assert policy_camera.generate_launch_description() is not None


def test_the_profiles_are_the_calibrated_size_at_the_policys_snapshot_rate():
    assert policy_camera.COLOR_PROFILE == "640x480x15"
    assert policy_camera.DEPTH_PROFILE == "640x480x15"


def test_the_composite_sync_and_alignment_are_fixed_not_optional():
    fixed = policy_camera.FIXED_CAMERA_ARGUMENTS
    assert fixed["enable_rgbd"] == "true"
    assert fixed["enable_sync"] == "true"
    assert fixed["align_depth.enable"] == "true"
    assert fixed["device_type"] == "d415"
    assert fixed["pointcloud.enable"] == "false"


def test_the_realsense_launch_is_included_with_those_arguments():
    include = _include()
    arguments = {_text(name): value for name, value in include.launch_arguments}
    for name, value in policy_camera.FIXED_CAMERA_ARGUMENTS.items():
        assert _text(arguments[name]) == value, name
    for name in ("rgb_camera.color_profile", "depth_module.depth_profile", "serial_no"):
        assert name in arguments
    source = _text(include.launch_description_source.location)
    assert "realsense2_camera" in source and "rs_launch.py" in source


def test_auto_exposure_priority_travels_in_a_config_file_not_a_launch_argument():
    """rs_launch.py drops launch arguments it does not declare, with a warning.

    It does merge config_file into the node's parameters, so the option has to
    arrive that way or it silently does not reach the camera at all.
    """

    arguments = {_text(name): value for name, value in _include().launch_arguments}
    assert "config_file" in arguments
    assert policy_camera.CAMERA_PARAMETER_FILE in _text(arguments["config_file"])
    assert "rgb_camera.auto_exposure_priority" not in arguments


def test_the_shipped_parameter_file_turns_auto_exposure_priority_off():
    source = (
        Path(__file__).resolve().parents[1] / "config" / policy_camera.CAMERA_PARAMETER_FILE
    )
    parameters = yaml.safe_load(source.read_text())
    assert parameters["rgb_camera.auto_exposure_priority"] is False


def test_the_operator_can_choose_another_profile_but_gets_the_calibrated_one_by_default():
    declared = {
        entity.name: entity
        for entity in policy_camera.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert _text(declared["color_profile"].default_value) == "640x480x15"
    assert _text(declared["depth_profile"].default_value) == "640x480x15"
    assert {"serial_no", "camera_namespace", "camera_name", "initial_reset"} <= set(declared)
    assert _text(declared["initial_reset"].default_value) == "false"
