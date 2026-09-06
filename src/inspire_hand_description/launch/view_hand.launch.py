"""Look at the hand's description on its own, with sliders for every joint.

    ros2 launch inspire_hand_description view_hand.launch.py
    ros2 launch inspire_hand_description view_hand.launch.py side:=left

No driver, no hardware, no simulation - just the URDF, so this is the quickest
way to check that a re-vendored model still looks like a hand. The sliders drive
all twelve joints independently, including the six followers, which the real
hand cannot do; use the sim or the driver to see the coupling applied.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("inspire_hand_description")
    robot_description = xacro.process_file(
        os.path.join(share, "urdf", "inspire_hand.urdf.xacro"),
        mappings={
            "side": LaunchConfiguration("side").perform(context),
            "ros2_control": "false",
        },
    ).toxml()

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="both",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="both",
            arguments=["-d", os.path.join(share, "rviz", "inspire_hand.rviz")],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "side", default_value="right", description="'left' or 'right'."
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
