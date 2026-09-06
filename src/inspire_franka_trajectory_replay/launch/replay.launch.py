"""Bring up the FR3 replay controller and the real Inspire hand."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("franka_trajectory_replay"), "launch", "replay.launch.py"]
            )
        ),
        launch_arguments={
            "robot_config_file": PathJoinSubstitution(
                [
                    FindPackageShare("inspire_franka_trajectory_replay"),
                    "config",
                    "robot.config.yaml",
                ]
            ),
            "robot_ips": LaunchConfiguration("robot_ip"),
        }.items(),
    )
    hand = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("inspire_franka_bringup"), "launch", "hand.launch.py"]
            )
        ),
        launch_arguments={
            "port": LaunchConfiguration("hand_port"),
            "hand_id": LaunchConfiguration("hand_id"),
            "protocol": LaunchConfiguration("hand_protocol"),
            "mock": LaunchConfiguration("hand_mock"),
            "publish_description": "true",
            "description_namespace": "hand",
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="10.7.7.7"),
            DeclareLaunchArgument("hand_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("hand_id", default_value="1"),
            DeclareLaunchArgument("hand_protocol", default_value="modbus"),
            DeclareLaunchArgument("hand_mock", default_value="false"),
            arm,
            hand,
        ]
    )
