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

`arm_controller:=cartesian-impedance` selects the controllers yaml that also
declares the Cartesian impedance replay controller and loads it inactive next
to the joint controller; the runner's `--arm-controller cartesian-impedance`
homes with the joint controller and swaps for the trajectory:

    ros2 launch inspire_franka_trajectory_replay replay.launch.py \
        arm_controller:=cartesian-impedance
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

CARTESIAN_CONTROLLER = "cartesian_trajectory_replay_controller"


def generate_launch_description():
    arm_and_hand = PythonExpression(
        [
            "'", LaunchConfiguration("arm"), "'.lower() in ('true', '1', 'yes', 'on') and ",
            "'", LaunchConfiguration("hand"), "'.lower() in ('true', '1', 'yes', 'on')",
        ]
    )
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
            # With an arm, the rootless hand description below is attached to
            # fr3_link8. A hand-only replay keeps its ordinary world-mounted
            # description.
            "publish_description": PythonExpression(
                [
                    "'false' if '", LaunchConfiguration("arm"),
                    "'.lower() in ('true', '1', 'yes', 'on') else 'true'",
                ]
            ),
            "description_namespace": LaunchConfiguration("description_namespace"),
            "state_extras_divisor": LaunchConfiguration("hand_state_extras_divisor"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("hand")),
    )
    hand_description = ParameterValue(
        Command(
            [
                "xacro ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("inspire_hand_description"),
                        "urdf",
                        "inspire_hand.urdf.xacro",
                    ]
                ),
                " side:=right mount_to_world:=false ros2_control:=false",
            ]
        ),
        value_type=str,
    )
    mounted_hand_description = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="hand",
        parameters=[{"robot_description": hand_description}],
        remappings=[
            ("joint_states", "/inspire_hand/joint_states"),
            ("/tf", "/tf"),
            ("/tf_static", "/tf_static"),
        ],
        output="both",
        condition=IfCondition(arm_and_hand),
    )
    hand_mount = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="inspire_hand_flange_mount",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "3.141592653589793",
            "--frame-id", "fr3_link8", "--child-frame-id", "hand_mount",
        ],
        output="screen",
        condition=IfCondition(arm_and_hand),
    )
    loads_cartesian = PythonExpression(
        ["'", LaunchConfiguration("arm_controller"),
         "' in ('cartesian-impedance', 'policy')"]
    )
    is_policy = PythonExpression(
        ["'", LaunchConfiguration("arm_controller"), "' == 'policy'"]
    )
    cartesian_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            CARTESIAN_CONTROLLER,
            "--inactive",
            "--controller-manager-timeout",
            "30",
        ],
        parameters=[LaunchConfiguration("controllers_yaml")],
        output="screen",
        condition=IfCondition(
            PythonExpression(
                ["'", LaunchConfiguration("arm"), "'.lower() in ('true', '1', 'yes', 'on') and (",
                 loads_cartesian, ")"]
            )
        ),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
            DeclareLaunchArgument(
                "arm_controller",
                default_value="joint-impedance",
                choices=["joint-impedance", "cartesian-impedance", "policy"],
                description="joint-impedance: the validated joint-impedance replay stack. "
                "cartesian-impedance: additionally load the Cartesian impedance replay "
                "controller; policy: load it with bounded live-setpoint gains.",
            ),
            # The simple joint-impedance example's effort law and gains, with
            # the replay plugin supplying its reference from recorded waypoints.
            # The Cartesian yaml is that file plus the Cartesian controller.
            DeclareLaunchArgument(
                "controllers_yaml",
                default_value=PythonExpression(
                    ["'", PathJoinSubstitution(
                        [FindPackageShare("inspire_franka_trajectory_replay"), "config",
                         "controllers_policy.yaml"]),
                     "' if ", is_policy, " else ('",
                     PathJoinSubstitution(
                        [FindPackageShare("inspire_franka_trajectory_replay"), "config",
                         "controllers_cartesian_impedance.yaml"]),
                     "' if ", loads_cartesian, " else '",
                     PathJoinSubstitution(
                        [FindPackageShare("inspire_franka_trajectory_replay"), "config",
                         "controllers_joint_impedance.yaml"]),
                     "')"]
                ),
                description="Controller manager configuration; follows arm_controller unless "
                "given explicitly.",
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
            cartesian_spawner,
            hand,
            mounted_hand_description,
            hand_mount,
        ]
    )
