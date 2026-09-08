"""Bring up the FR3 replay controller and the real Inspire hand.

Either device can be left out, because on hardware they are two independent
stacks (see the workspace README). Bringing up one alone is the whole bring-up
for that device, not a degraded version of the pair:

    ros2 launch inspire_franka_trajectory_replay replay.launch.py \
        hand_port:=/dev/ttyUSB0 arm:=false      # hand only, no FCI needed
    ros2 launch inspire_franka_trajectory_replay replay.launch.py \
        robot_ip:=172.16.0.2 hand:=false         # arm only

Match the runner to whatever was launched: `--no-arm` for a hand-only session,
`--no-hand` for an arm-only one. The runner reaches the arm through the
controller manager, so a hand-only launch plus a coordinated run would block
waiting for a controller that was never started.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
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
            "controllers_yaml": LaunchConfiguration("controllers_yaml"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("arm")),
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
            "description_namespace": LaunchConfiguration("description_namespace"),
            "state_extras_divisor": LaunchConfiguration("hand_state_extras_divisor"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("hand")),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
            # Stock JointTrajectoryController over position interfaces. This
            # makes franka_hardware use the FR3's internal joint-impedance
            # controller; no custom torque/impedance law is in the replay path.
            DeclareLaunchArgument(
                "controllers_yaml",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("inspire_franka_trajectory_replay"),
                        "config",
                        "controllers_internal_impedance.yaml",
                    ]
                ),
                description="Controller manager configuration for the stock position "
                "trajectory controller and robot-internal joint impedance.",
            ),
            DeclareLaunchArgument("hand_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("hand_id", default_value="1"),
            DeclareLaunchArgument("hand_protocol", default_value="modbus"),
            DeclareLaunchArgument("hand_mock", default_value="false"),
            # Replay is the one session that streams targets, and RS485 is
            # half-duplex, so the driver's per-publish current and force reads
            # are bandwidth taken from the command stream. Nothing in the
            # replay path reads either, so they drop to 10 Hz here while
            # joint_states stays at the full publish rate. Ordinary bringup
            # leaves the driver's own default of 1 alone.
            DeclareLaunchArgument(
                "hand_state_extras_divisor",
                default_value="5",
                description="Read the hand's current and force once per N state publishes.",
            ),
            DeclareLaunchArgument(
                "arm", default_value="true", description="Bring up the FR3 replay controller."
            ),
            DeclareLaunchArgument(
                "hand", default_value="true", description="Bring up the Inspire hand driver."
            ),
            # Two latched publishers on /robot_description means RViz shows
            # whichever it heard last, so the hand yields the global topic to the
            # arm. Alone, it has no reason to hide.
            DeclareLaunchArgument(
                "description_namespace",
                default_value=PythonExpression(
                    ["'hand' if '", LaunchConfiguration("arm"), "'.lower() in ",
                     "('true', '1', 'yes', 'on') else ''"]
                ),
                description="Namespace for the hand's robot_state_publisher.",
            ),
            arm,
            hand,
        ]
    )
