"""Bring up one Inspire RH56 hand: the driver, and optionally its description.

    # hardware, driver only
    ros2 launch inspire_hand_driver inspire_hand.launch.py port:=/dev/ttyUSB0

    # no hardware to hand - a simulated hand that slews to its targets
    ros2 launch inspire_hand_driver inspire_hand.launch.py mock:=true

    # with robot_state_publisher and RViz, so the hand appears in TF
    ros2 launch inspire_hand_driver inspire_hand.launch.py \
        mock:=true publish_description:=true start_rviz:=true

`publish_description` is off by default because when this launch is included by
`inspire_franka_bringup`, the arm's and the hand's descriptions are published
together by one `robot_state_publisher` rather than one each.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xacro

# (name, default, type, help). The type matters: launch substitutions are always
# strings, and the node declares typed parameters, so each value has to be
# coerced back to the type the node expects.
NODE_ARGS = (
    ("port", "/dev/ttyUSB0", str, "Serial port of the USB-RS485 adapter."),
    ("baudrate", "115200", int, "Serial baud rate."),
    ("hand_id", "1", int, "Configured hand ID on the RS485 bus."),
    ("protocol", "modbus", str, "Wire protocol: 'modbus' or 'legacy'."),
    ("mock", "false", bool, "Run against a simulated hand instead of hardware."),
    ("publish_rate_hz", "50.0", float, "State publish rate."),
    ("joint_prefix", "", str, "Prefix on every joint name; must match the description."),
    ("startup_speed", "0", int, "Speed applied to all DOF at startup (0 = leave alone)."),
    ("startup_force", "0", int, "Force threshold applied at startup (0 = leave alone)."),
)


def launch_setup(context, *args, **kwargs):
    node_name = LaunchConfiguration("node_name").perform(context)
    side = LaunchConfiguration("side").perform(context)
    prefix = LaunchConfiguration("joint_prefix").perform(context)

    nodes = [
        Node(
            package="inspire_hand_driver",
            executable="inspire_hand_node",
            name=node_name,
            output="screen",
            parameters=[
                {
                    n: ParameterValue(LaunchConfiguration(n), value_type=t)
                    for n, _, t, _ in NODE_ARGS
                }
            ],
        )
    ]

    if LaunchConfiguration("publish_description").perform(context).lower() in ("true", "1"):
        xacro_path = os.path.join(
            get_package_share_directory("inspire_hand_description"),
            "urdf",
            "inspire_hand.urdf.xacro",
        )
        robot_description = xacro.process_file(
            xacro_path, mappings={"side": side, "prefix": prefix, "ros2_control": "false"}
        ).toxml()
        # Empty (the default) publishes to the global /robot_description, which
        # is what a standalone hand wants. Running alongside the arm, franka's
        # own robot_state_publisher already owns that topic and two latched
        # publishers on it means RViz shows whichever it happened to hear last -
        # so the combined bringup passes a namespace here instead.
        description_namespace = LaunchConfiguration("description_namespace").perform(context)
        nodes += [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=description_namespace,
                output="both",
                parameters=[{"robot_description": robot_description}],
                remappings=[
                    # The driver publishes on ~/joint_states; robot_state_publisher
                    # reads a relative joint_states, so bridge the two.
                    ("joint_states", f"/{node_name}/joint_states"),
                    # TF is global by convention. Without these, a namespaced
                    # robot_state_publisher would publish to /<ns>/tf and the
                    # hand would simply be missing from RViz.
                    ("/tf", "/tf"),
                    ("/tf_static", "/tf_static"),
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="both",
                arguments=[
                    "-d",
                    os.path.join(
                        get_package_share_directory("inspire_hand_description"),
                        "rviz",
                        "inspire_hand.rviz",
                    ),
                ],
                condition=IfCondition(LaunchConfiguration("start_rviz")),
            ),
        ]

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, _, h in NODE_ARGS]
        + [
            DeclareLaunchArgument(
                "node_name",
                default_value="inspire_hand",
                description="Node name, which is also the topic namespace. Use distinct "
                "names (e.g. left_hand / right_hand) to run two hands at once.",
            ),
            DeclareLaunchArgument(
                "side",
                default_value="right",
                description="Which hand's geometry to describe: 'left' or 'right'. Affects "
                "the description only - the wire protocol is identical.",
            ),
            DeclareLaunchArgument(
                "publish_description",
                default_value="false",
                description="Also start robot_state_publisher for the hand alone. Leave off "
                "when a higher-level bringup already publishes a combined description.",
            ),
            DeclareLaunchArgument(
                "description_namespace",
                default_value="",
                description="Namespace for the hand's robot_state_publisher, and so for its "
                "robot_description topic. Empty means the global /robot_description. Set it "
                "when something else already publishes there - the combined bringup does.",
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="false",
                description="Start RViz. Requires publish_description:=true to show anything.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
