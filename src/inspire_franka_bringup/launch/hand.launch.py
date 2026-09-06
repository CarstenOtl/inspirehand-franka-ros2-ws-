"""The Inspire hand on real hardware, on its own.

    ros2 launch inspire_franka_bringup hand.launch.py port:=/dev/ttyUSB0
    ros2 launch inspire_franka_bringup hand.launch.py mock:=true start_rviz:=true

A thin wrapper over inspire_hand_driver's own launch, kept here so that the
hand, the arm and the pair are all started the same way from one package. If you
are working on the hand alone, `ros2 launch inspire_hand_driver
inspire_hand.launch.py` is the same thing with fewer layers.

Don't know the port, baud rate or protocol? The probe is read-only and cannot
move the hand:

    ros2 run inspire_hand_driver inspire_hand_probe /dev/ttyUSB0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

ARGS = (
    ("port", "/dev/ttyUSB0", "Serial port of the USB-RS485 adapter."),
    ("baudrate", "115200", "Serial baud rate. The hand accepts 115200/57600/19200/921600."),
    ("hand_id", "1", "Configured hand ID on the RS485 bus. Factory default is 1."),
    ("protocol", "modbus", "Wire protocol: 'modbus' or 'legacy'."),
    ("mock", "false", "Run against a simulated hand instead of hardware."),
    ("publish_rate_hz", "50.0", "State publish rate."),
    ("node_name", "inspire_hand", "Node name, which is also the topic namespace."),
    ("side", "right", "Which hand's geometry to describe: 'left' or 'right'."),
    ("joint_prefix", "", "Prefix on every joint name; must match the description."),
    ("publish_description", "true", "Also start robot_state_publisher for the hand."),
    ("description_namespace", "", "Namespace for the hand's robot_state_publisher. Empty "
                                  "means the global /robot_description."),
    ("start_rviz", "false", "Start RViz. Needs publish_description:=true."),
)


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
        + [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("inspire_hand_driver"),
                            "launch",
                            "inspire_hand.launch.py",
                        ]
                    )
                ),
                launch_arguments={n: LaunchConfiguration(n) for n, _, _ in ARGS}.items(),
            )
        ]
    )
