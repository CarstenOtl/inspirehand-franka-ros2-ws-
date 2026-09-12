"""The replay stack against MuJoCo instead of hardware.

    ros2 launch inspire_franka_trajectory_replay sim_replay.launch.py

Then, in a second sourced shell, the runner exactly as on hardware -- same
executable, with an explicit selection of the legacy position controller:

    ros2 run inspire_franka_trajectory_replay replay_trajectory \\
        apps/traj_replay/demo_trajs/traj_1 --cycle 1 --arm-controller position-jtc

By default this launch preserves the stock ``JointTrajectoryController``
position-based simulation. ``arm_controller:=joint-impedance`` or
``cartesian-impedance`` instead runs the hardware replay controllers over the
simulated effort interfaces, in a gravity-free copy of the scene (libfranka
compensates gravity underneath a torque controller on the real arm), with the
Cartesian controller on its built-in DH model. That exercises the whole runner
flow -- homing, the controller swap, the goto settle, the pose stream, pause and
abort -- against MuJoCo's dynamics, not the FR3's:

    ros2 launch inspire_franka_trajectory_replay sim_replay.launch.py \\
        arm_controller:=cartesian-impedance
    ros2 run inspire_franka_trajectory_replay replay_trajectory \\
        apps/traj_replay/demo_trajs/traj_2_cycle3 \\
        --home apps/traj_replay/demo_trajs/traj_2_cycle3/homing.yaml \\
        --arm-controller cartesian-impedance --time-scale 5

How the two devices get into the simulator
------------------------------------------
The arm is direct: ``inspire_franka_description``'s ``fr3.ros2_control.xacro``
exports position commands and position/velocity state under
``MujocoSystemInterface``, which is exactly what the stock trajectory controller
claims. The runner selects this action interface with --arm-controller position-jtc.

The hand is not direct, because on hardware it is not a ros2_control device at
all. The driver runs in ``mock`` mode, ``inspire_hand_sim_bridge`` forwards what
it publishes into the simulator's forward command controller, and the chain
becomes the hardware one up to the last step:

    runner --open ratios--> driver (mock) --radians--> bridge --> MuJoCo

so the unit conversion, the partial-command merge, the register quantisation
and the range rejection are all still in the path.

What this does not tell you
---------------------------
The RS485 bus: the mock transport slews at a fixed rate rather than modelling
the hand's ~0.17 s closed-loop response, and there is no serial latency, bus
contention or dropped frames -- see ``inspire_hand_driver.benchmark``. And the
arm's tracking error is MuJoCo's position actuators, not the FR3's joint
impedance; treat it as a plausibility check, not a prediction. What it does rule
out is the whole class of mistakes that matter before a first hardware run:
wrong units, wrong joint order, a trajectory in the wrong part of the workspace,
self-collision, and any limit the preparation would have let through.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CONTROLLER = "trajectory_replay_controller"
CARTESIAN_CONTROLLER = "cartesian_trajectory_replay_controller"


def generate_launch_description():
    arm_controller = LaunchConfiguration("arm_controller")
    is_torque = PythonExpression(["'", arm_controller, "' != 'position-jtc'"])
    is_cartesian = PythonExpression(["'", arm_controller, "' == 'cartesian-impedance'"])
    position_yaml = PathJoinSubstitution(
        [FindPackageShare("inspire_franka_sim"), "config", "controllers_replay.yaml"]
    )
    torque_yaml = PathJoinSubstitution(
        [FindPackageShare("inspire_franka_trajectory_replay"), "config",
         "controllers_sim_impedance.yaml"]
    )
    controllers_yaml = PythonExpression(
        ["'", torque_yaml, "' if ", is_torque, " else '", position_yaml, "'"]
    )
    # The torque scene is the flange scene without gravity; '' keeps the
    # simulator's own default for the position stack.
    mjcf = PythonExpression(
        ["'inspire_franka_flange_torque_scene.xml' if ", is_torque, " else ''"]
    )
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("inspire_franka_sim"), "launch", "sim.launch.py"]
            )
        ),
        launch_arguments={
            "mount": "flange",
            "hardware_type": "mujoco",
            "headless": LaunchConfiguration("headless"),
            "start_rviz": LaunchConfiguration("start_rviz"),
            # The arm is claimed by the replay controller, spawned below, and
            # not by either of the stock controllers this launch knows about.
            "arm_command_interface": "none",
            # Direct position on the hand: the driver has already shaped the
            # motion by the time the bridge forwards it, so a second
            # interpolator here would only add lag.
            "hand_command_interface": "position_direct",
            "controllers_config_path": controllers_yaml,
            "mjcf": mjcf,
        }.items(),
    )

    # Spawned here rather than by sim.launch.py, which only knows the stock
    # controllers. Its parameters come from the same YAML the manager was given.
    replay_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            CONTROLLER,
            "--param-file",
            controllers_yaml,
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )
    # Loaded inactive next to the joint controller; the runner homes with the
    # joint controller and swaps, exactly as on hardware.
    cartesian_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            CARTESIAN_CONTROLLER,
            "--inactive",
            "--param-file",
            controllers_yaml,
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
        condition=IfCondition(is_cartesian),
    )

    hand_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("inspire_hand_driver"), "launch", "inspire_hand.launch.py"]
            )
        ),
        launch_arguments={
            "mock": "true",
            # The node name is the topic namespace, and "inspire_hand" is what
            # the runner defaults to, so the runner needs no topic flags.
            "node_name": "inspire_hand",
            # The simulator already publishes the hand's description and TF.
            "publish_description": "false",
        }.items(),
        condition=IfCondition(LaunchConfiguration("hand")),
    )

    bridge = Node(
        package="inspire_franka_sim",
        executable="inspire_hand_sim_bridge.py",
        output="screen",
        parameters=[{"driver_joint_states": "/inspire_hand/joint_states"}],
        condition=IfCondition(LaunchConfiguration("hand")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "arm_controller",
                default_value="position-jtc",
                choices=["position-jtc", "joint-impedance", "cartesian-impedance"],
                description="position-jtc: the stock position trajectory controller "
                "(runner --arm-controller position-jtc). joint-impedance / "
                "cartesian-impedance: the hardware replay controllers over effort "
                "interfaces in the gravity-free scene (runner default / "
                "--arm-controller cartesian-impedance).",
            ),
            DeclareLaunchArgument(
                "headless", default_value="true", description="Run MuJoCo without a viewer."
            ),
            DeclareLaunchArgument(
                "start_rviz", default_value="false", description="Start RViz2."
            ),
            DeclareLaunchArgument(
                "hand",
                default_value="true",
                description="Bring up the mock hand driver and its bridge. With this off, "
                "the simulated hand is unclaimed and the runner needs --no-hand.",
            ),
            simulator,
            replay_controller,
            cartesian_controller,
            hand_driver,
            bridge,
        ]
    )
