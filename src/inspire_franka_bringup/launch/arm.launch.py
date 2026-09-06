"""The Franka FR3 on real hardware, on its own.

    ros2 launch inspire_franka_bringup arm.launch.py robot_ip:=10.7.7.7
    ros2 launch inspire_franka_bringup arm.launch.py \
        robot_ip:=10.7.7.7 gravity_compensation:=true
    ros2 launch inspire_franka_bringup arm.launch.py use_fake_hardware:=true

A thin wrapper over upstream franka_bringup's franka.launch.py. It is here so
the arm, the hand and the pair are all started the same way, and so there is one
place to record the two defaults that are ours rather than Franka's:

* `robot_ip` defaults to nothing. Franka's own default is 172.16.0.3, which is
  the address on the robot's own network - not where the FCI is on a
  company-network installation. Passing the wrong address wastes a lot of time,
  so this refuses to guess. See docs/network.md.
* `load_gripper` defaults to false. Franka's two-finger gripper is not what this
  workspace is about, and on real hardware it and the Inspire hand would be
  competing for the same flange.

Before the first run, check the FCI is actually reachable. No ROS, no
controllers, no motion:

    ros2 run inspire_franka_bringup fci_check 10.7.7.7
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ARGS = (
    ("robot_ip", "", "Hostname or IP address of the FR3's FCI. Required unless "
                     "use_fake_hardware:=true."),
    ("robot_type", "fr3", "Arm model, passed through to franka_description."),
    ("arm_prefix", "", "Prefix for arm topics and joint names."),
    ("namespace", "", "Namespace for the arm's nodes."),
    ("load_gripper", "false", "Use Franka's two-finger gripper as the end effector."),
    ("use_fake_hardware", "false", "Use ros2_control mock hardware instead of the FCI."),
    ("fake_sensor_commands", "false", "Fake sensor commands (only with use_fake_hardware)."),
    ("joint_state_rate", "30", "Joint state publishing rate, Hz."),
    ("use_rviz", "false", "Start RViz."),
    (
        "gravity_compensation",
        "false",
        "Start Franka's zero-effort gravity compensation controller for hand guiding.",
    ),
)


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration("robot_ip").perform(context)
    fake = LaunchConfiguration("use_fake_hardware").perform(context).lower() in ("true", "1")
    if not robot_ip and not fake:
        raise RuntimeError(
            "robot_ip is required. There is no safe default: Franka's own "
            "(172.16.0.3) is the robot's private network, and on a company "
            "network the FCI is somewhere else entirely - see docs/network.md. "
            "Pass use_fake_hardware:=true to run without an arm."
        )

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("franka_bringup"), "launch", "franka.launch.py"]
                )
            ),
            launch_arguments={
                n: LaunchConfiguration(n)
                for n, _, _ in ARGS
                # These are options provided by this wrapper, not franka.launch.py.
                if n not in ("use_rviz", "gravity_compensation")
            }.items(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=LaunchConfiguration("namespace"),
            arguments=[
                "gravity_compensation_example_controller",
                "--controller-manager-timeout",
                "30",
            ],
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare("franka_bringup"), "config", "controllers.yaml"]
                )
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("gravity_compensation")),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [OpaqueFunction(function=launch_setup)]
    )
