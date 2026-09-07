"""Both assets on real hardware: the FR3 over the FCI, the Inspire hand over RS485.

    ros2 launch inspire_franka_bringup inspire_franka.launch.py \
        robot_ip:=10.7.7.7 hand_port:=/dev/ttyUSB0 \
        gravity_compensation:=true

    # neither piece of hardware present, everything else identical
    ros2 launch inspire_franka_bringup inspire_franka.launch.py \
        use_fake_hardware:=true hand_mock:=true start_rviz:=true

    # one asset only - the same as arm.launch.py / hand.launch.py
    ros2 launch inspire_franka_bringup inspire_franka.launch.py hand:=false robot_ip:=10.7.7.7
    ros2 launch inspire_franka_bringup inspire_franka.launch.py arm:=false

This starts the two stacks side by side. They share a ROS graph and a TF tree
and nothing else - there is no combined controller_manager, no shared clock, and
no coordinated motion primitive. Commanding both at once means publishing to
both, which is what "together" means for two mechanically independent devices
sitting on the same bench.

Why they are separate at all: the arm is a 1 kHz real-time FCI connection served
by a ros2_control hardware component, and the hand is a ~50 Hz half-duplex
Modbus link served by a plain rclpy node. Putting the hand's serial round-trip
inside the arm's control loop would stall it. See the repo README.

For coordinated control under one controller_manager - which the simulation
does have - use inspire_franka_sim.

What ends up on the graph:

    /joint_states                     the arm, from franka's joint_state_publisher
    /robot_description                the arm, from franka's robot_state_publisher
    /inspire_hand/joint_states        the hand, in radians, from the driver
    /inspire_hand/state               the hand, in open-ratio units
    /inspire_hand/command             command the hand (open ratios, 1.0 = open)
    /hand/robot_description           the hand's URDF, namespaced so it does not
                                      collide with the arm's
    /tf                               both, merged

RViz therefore needs two RobotModel displays, one per description topic. The
config in this package already has them.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# The hand's robot_state_publisher goes here rather than in the global
# namespace: franka's own already owns /robot_description, and two latched
# publishers on one topic means RViz shows whichever it heard last.
HAND_DESCRIPTION_NAMESPACE = "hand"

ARM_ARGS = (
    ("robot_ip", "", "Hostname or IP address of the FR3's FCI. Required unless "
                     "use_fake_hardware:=true. There is no default - see docs/network.md."),
    ("robot_type", "fr3", "Arm model, passed through to franka_description."),
    ("arm_prefix", "", "Prefix for arm topics and joint names."),
    ("load_gripper", "false", "Use Franka's two-finger gripper as the end effector."),
    ("use_fake_hardware", "false", "Use ros2_control mock hardware instead of the FCI."),
    ("joint_state_rate", "30", "Arm joint state publishing rate, Hz."),
    (
        "gravity_compensation",
        "false",
        "Start Franka's zero-effort gravity compensation controller for hand guiding.",
    ),
)

# Prefixed with hand_ so that `mock` and `use_fake_hardware` cannot be confused
# for each other on a command line that sets both.
HAND_ARGS = (
    ("hand_port", "port", "/dev/ttyUSB0", "Serial port of the USB-RS485 adapter."),
    ("hand_baudrate", "baudrate", "115200", "Serial baud rate."),
    ("hand_id", "hand_id", "1", "Configured hand ID on the RS485 bus."),
    ("hand_protocol", "protocol", "modbus", "Wire protocol: 'modbus' or 'legacy'."),
    ("hand_mock", "mock", "false", "Run against a simulated hand instead of hardware."),
    ("hand_side", "side", "right", "Which hand's geometry to describe."),
    ("hand_publish_rate_hz", "publish_rate_hz", "50.0", "Hand state publish rate."),
    ("hand_joint_prefix", "joint_prefix", "", "Prefix on every hand joint name."),
)


def launch_setup(context, *args, **kwargs):
    def flag(name):
        return LaunchConfiguration(name).perform(context).lower() in ("true", "1")

    with_arm, with_hand = flag("arm"), flag("hand")
    if not (with_arm or with_hand):
        raise RuntimeError("arm and hand are both false; there is nothing to bring up")

    actions = []

    if with_arm:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("inspire_franka_bringup"), "launch", "arm.launch.py"]
                    )
                ),
                launch_arguments={n: LaunchConfiguration(n) for n, _, _ in ARM_ARGS}.items(),
            )
        )

    if with_hand:
        hand_arguments = {
            inner: LaunchConfiguration(outer) for outer, inner, _, _ in HAND_ARGS
        }
        hand_arguments["publish_description"] = "true"
        # Namespaced only when the arm is also running; on its own the hand may
        # as well own the conventional topic.
        hand_arguments["description_namespace"] = (
            HAND_DESCRIPTION_NAMESPACE if with_arm else ""
        )
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("inspire_franka_bringup"), "launch", "hand.launch.py"]
                    )
                ),
                launch_arguments=hand_arguments.items(),
            )
        )

    # One RViz for both, rather than letting each sub-launch start its own.
    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="both",
            arguments=[
                "-d",
                PathJoinSubstitution(
                    [
                        FindPackageShare("inspire_franka_bringup"),
                        "rviz",
                        "inspire_franka.rviz",
                    ]
                ),
            ],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
        )
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("arm", default_value="true", description="Bring up the FR3."),
            DeclareLaunchArgument(
                "hand", default_value="true", description="Bring up the Inspire hand."
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="false",
                description="Start one RViz showing both descriptions.",
            ),
        ]
        + [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARM_ARGS]
        + [
            DeclareLaunchArgument(outer, default_value=d, description=h)
            for outer, _, d, h in HAND_ARGS
        ]
        + [OpaqueFunction(function=launch_setup)]
    )
