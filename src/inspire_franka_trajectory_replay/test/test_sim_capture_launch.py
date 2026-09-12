"""The simulation capture launch: that it builds, and that it builds the right thing.

Launching MuJoCo needs a simulator; checking that this launch asks for an arm
nothing claims, in a scene without gravity, with the real hand driver behind it,
does not. Those four choices are the whole reason the file exists, so they are
the ones under test.
"""

import pytest

launch = pytest.importorskip("launch")


def _text(substitution):
    """Flatten a launch substitution (or a plain string) to text."""
    if isinstance(substitution, str):
        return substitution
    if isinstance(substitution, (list, tuple)):
        return "".join(_text(part) for part in substitution)
    return getattr(substitution, "text", None) or str(substitution)

from inspire_franka_trajectory_replay.launch_files import sim_capture  # noqa: E402


def test_the_launch_description_builds():
    assert sim_capture.generate_launch_description() is not None


def test_the_arm_is_claimed_by_an_effort_controller_not_left_unclaimed():
    """Unclaimed is not free: the hardware holds unclaimed joints where they were.

    Floating the arm takes the same thing it takes on the real robot -- a
    controller on the effort interface, commanded to apply nothing.
    """
    assert sim_capture.SIMULATOR_ARGUMENTS["arm_command_interface"] == "effort"
    assert sim_capture.ZERO_EFFORT_CONTROLLER == "fr3_effort_forward_command_controller"


def test_zero_torque_is_actually_published_rather_than_assumed():
    description = sim_capture.generate_launch_description()
    commands = [
        [_text(part) for part in entity.cmd]
        for entity in description.entities
        if getattr(entity, "cmd", None)
    ]
    published = [command for command in commands if command[:3] == ["ros2", "topic", "pub"]]
    assert len(published) == 1
    assert sim_capture.ZERO_EFFORT_TOPIC in published[0]
    assert any("0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0" in part for part in published[0])


def test_the_capture_tool_and_the_launch_agree_on_the_zero_effort_controller():
    from inspire_franka_trajectory_replay import capture

    assert capture.SIM_ZERO_EFFORT_CONTROLLER == sim_capture.ZERO_EFFORT_CONTROLLER


def test_the_scene_has_no_gravity_so_the_arm_floats_instead_of_collapsing():
    assert "torque_scene" in sim_capture.FLOATING_SCENE
    assert sim_capture.SIMULATOR_ARGUMENTS["mjcf"] == sim_capture.FLOATING_SCENE


def test_the_hand_runs_the_real_driver_in_mock_mode():
    """That is what puts the driver's unit conversion and overlay under test."""
    assert sim_capture.DRIVER_ARGUMENTS["mock"] == "true"


def test_the_hand_is_driven_directly_rather_than_through_a_second_interpolator():
    assert sim_capture.SIMULATOR_ARGUMENTS["hand_command_interface"] == "position_direct"


def test_the_hand_driver_keeps_the_topic_names_the_capture_tool_expects():
    assert sim_capture.DRIVER_ARGUMENTS["node_name"] == "inspire_hand"


def test_the_bridge_is_started_on_simulation_time():
    description = sim_capture.generate_launch_description()
    bridge = [
        entity
        for entity in description.entities
        if getattr(entity, "node_executable", None) == "inspire_hand_sim_bridge.py"
    ]
    assert len(bridge) == 1
