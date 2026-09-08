"""Launch the subscriber-only MuJoCo visualization/replay node.

Unlike sim.launch.py, this launch creates no controller manager, hardware
interface, command controller, or application/data publisher.  It is safe to
attach to recorded, mock, or live state topics because information only flows
into the viewer.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("inspire_franka_sim")
    default_model = os.path.join(
        share, "mjcf", "inspire_franka_flange_scene.xml"
    )

    arguments = [
        DeclareLaunchArgument("model_path", default_value=default_model),
        DeclareLaunchArgument("initial_keyframe", default_value="start"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument(
            "trajectory_topic", default_value="/mujoco_sim/joint_trajectory"
        ),
        DeclareLaunchArgument("subscribe_joint_states", default_value="true"),
        DeclareLaunchArgument("subscribe_trajectory", default_value="true"),
        DeclareLaunchArgument(
            "joint_state_preempts_trajectory", default_value="true"
        ),
        DeclareLaunchArgument(
            "joint_map_file",
            default_value="",
            description="YAML ROS-joint-name to MJCF-joint-name mapping.",
        ),
        DeclareLaunchArgument("playback_speed", default_value="1.0"),
        DeclareLaunchArgument("render_hz", default_value="60.0"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument(
            "show_left_ui",
            default_value="false",
            description="Show MuJoCo's left simulation/configuration panel.",
        ),
        DeclareLaunchArgument(
            "show_right_ui",
            default_value="false",
            description="Show MuJoCo's right rendering/watch panel.",
        ),
        DeclareLaunchArgument(
            "step_physics",
            default_value="false",
            description="Explicit opt-in to passive physics steps; kinematic replay is the default.",
        ),
        DeclareLaunchArgument("max_physics_steps_per_frame", default_value="20"),
        DeclareLaunchArgument("clamp_to_joint_limits", default_value="false"),
    ]

    bool_parameters = (
        "subscribe_joint_states",
        "subscribe_trajectory",
        "joint_state_preempts_trajectory",
        "headless",
        "show_left_ui",
        "show_right_ui",
        "step_physics",
        "clamp_to_joint_limits",
    )
    parameters = {
        "model_path": LaunchConfiguration("model_path"),
        "initial_keyframe": LaunchConfiguration("initial_keyframe"),
        "joint_state_topic": LaunchConfiguration("joint_state_topic"),
        "trajectory_topic": LaunchConfiguration("trajectory_topic"),
        "joint_map_file": LaunchConfiguration("joint_map_file"),
        "playback_speed": ParameterValue(
            LaunchConfiguration("playback_speed"), value_type=float
        ),
        "render_hz": ParameterValue(
            LaunchConfiguration("render_hz"), value_type=float
        ),
        "max_physics_steps_per_frame": ParameterValue(
            LaunchConfiguration("max_physics_steps_per_frame"), value_type=int
        ),
    }
    parameters.update(
        {
            name: ParameterValue(LaunchConfiguration(name), value_type=bool)
            for name in bool_parameters
        }
    )

    return LaunchDescription(
        arguments
        + [
            Node(
                package="inspire_franka_sim",
                executable="mujoco_vis_node.py",
                name="mujoco_vis_node",
                output="screen",
                emulate_tty=True,
                parameters=[parameters],
            )
        ]
    )
