"""The hand-guided capture stack against MuJoCo instead of the FCI.

    ros2 launch inspire_franka_trajectory_replay sim_capture.launch.py

Then, in a second sourced shell, the capture tool exactly as on hardware, with
the profile that knows MuJoCo has no FrankaRobotState:

    ros2 run inspire_franka_trajectory_replay capture_demo --profile sim \\
        --note "rehearsing the pinch"

This exists because neither of the two launches next to it is right for a
capture. ``sim.launch.py`` claims the arm with a trajectory controller and never
starts the hand driver, so there is no ``/inspire_hand/command`` to press a key
into. ``sim_replay.launch.py`` starts the driver and the bridge, but it also
spawns the replay controller, which claims the arm and drives it to a setpoint --
the opposite of an arm you can push around.

What this launch assembles instead
----------------------------------
*The arm floats.* This takes a zero-torque controller, not an unclaimed arm.
Leaving the arm unclaimed does **not** make it limp: the MuJoCo hardware holds
unclaimed joints on their last desired position, so the arm comes up rigid and
cannot be pushed. What works is the same thing that works on the real robot --
claim the arm with an effort controller and command zero torque -- in the
gravity-free scene, which is the simulator's stand-in for the gravity
compensation libfranka applies underneath a torque controller. Zero commanded
torque plus no gravity is an arm that holds its pose and moves when you push it.
With the viewer up, ctrl-drag a link to guide it by hand.

*The hand is the real driver.* ``inspire_hand_driver`` runs in ``mock`` mode and
``inspire_hand_sim_bridge`` forwards its radians into MuJoCo, so every key press
goes through the driver's unit conversion, range rejection, register
quantisation and thumb-abduction overlay exactly as it would on the bench. The
hand control is genuinely under test here; the arm is scenery.

What it cannot rehearse
-----------------------
``franka_msgs/FrankaRobotState`` is libfranka's own message. There is no
measured torque, no external wrench, no ``O_T_EE``, no collision indicator and
no load model in simulation, and none of them are faked: a session recorded here
carries ``/joint_states`` and nothing more from the arm, its TCP is forward
kinematics rather than the robot's own pose, and both the session manifest and
the extracted artifact are marked ``hand_guided_sim``. It rehearses the capture
and extraction path. It does not produce training data.

The mock transport also slews toward its target at a fixed rate rather than
modelling the hand's closed-loop response, and there is no RS485 latency or bus
contention. Timing conclusions belong on hardware.
The description is built here, under the package, rather than in the launch file
itself: a file under ``launch/`` is installed as data and cannot be imported, and
the four choices below are exactly the kind that break silently. They are stated
as data so ``test_sim_capture_launch.py`` can assert on them.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

#: The flange scene without gravity. The arm holds its pose instead of
#: collapsing, which is what makes it guidable rather than merely unclaimed.
FLOATING_SCENE = "inspire_franka_flange_torque_scene.xml"

#: The arm controller that stands in for gravity compensation: a plain forward
#: command controller on the effort interface, fed zeros. It has to *claim* the
#: arm -- an unclaimed arm is held rigid on its last desired position, which is
#: the opposite of guidable.
ZERO_EFFORT_CONTROLLER = "fr3_effort_forward_command_controller"

#: Where that controller takes its command.
ZERO_EFFORT_TOPIC = f"/{ZERO_EFFORT_CONTROLLER}/commands"

#: What the simulator is asked for. Every entry here is load-bearing:
#: ``arm_command_interface=effort`` is what puts the arm under a controller that
#: can be told to apply nothing, and the gravity-free scene is what keeps it
#: where it was put once nothing is holding it up.
SIMULATOR_ARGUMENTS = {
    "hardware_type": "mujoco",
    "arm_command_interface": "effort",
    "hand_command_interface": "position_direct",
    "mjcf": FLOATING_SCENE,
}

#: What the hand driver is asked for. ``mock`` is the point: it is the real
#: driver, with its unit conversion, range rejection, register quantisation and
#: thumb-abduction overlay all in the path, talking to a simulated transport.
DRIVER_ARGUMENTS = {
    "mock": "true",
    "node_name": "inspire_hand",
    # The simulator already publishes the hand's description into TF; a second
    # publisher would fight it.
    "publish_description": "false",
}


def generate_launch_description() -> LaunchDescription:
    sim_share = get_package_share_directory("inspire_franka_sim")

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("inspire_franka_sim"), "launch", "sim.launch.py"]
            )
        ),
        launch_arguments={
            **SIMULATOR_ARGUMENTS,
            "headless": LaunchConfiguration("headless"),
            "start_rviz": LaunchConfiguration("start_rviz"),
            # The stock config, because it is the one that defines
            # fr3_effort_forward_command_controller. controllers_replay.yaml
            # does not -- it is built around the replay controller, which is
            # exactly the thing that must not be driving the arm here.
            "controllers_config_path": os.path.join(
                sim_share, "config", "controllers.yaml"
            ),
        }.items(),
    )

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("inspire_hand_driver"), "launch", "inspire_hand.launch.py"]
            )
        ),
        launch_arguments={
            **DRIVER_ARGUMENTS,
            "publish_rate_hz": LaunchConfiguration("hand_publish_rate_hz"),
        }.items(),
    )

    bridge = Node(
        package="inspire_franka_sim",
        executable="inspire_hand_sim_bridge.py",
        name="inspire_hand_sim_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # Zero torque, republished rather than latched: the controller writes
    # nothing until it has had a command, and this way it does not matter
    # whether the publisher or the spawner wins the race. Guiding the arm is
    # then MuJoCo's ctrl-drag against an arm nothing is holding.
    zero_effort = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "-r", "10", ZERO_EFFORT_TOPIC,
            "std_msgs/msg/Float64MultiArray",
            "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}",
        ],
        output="log",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run MuJoCo without its viewer. Off by default here, unlike "
                "the rest of the stack: guiding the arm by hand means ctrl-dragging it "
                "in the window.",
            ),
            DeclareLaunchArgument(
                "start_rviz", default_value="false", description="Start RViz2."
            ),
            DeclareLaunchArgument(
                "hand_publish_rate_hz",
                default_value="50.0",
                description="Mock driver state rate. 50 Hz matches the RS485 link's.",
            ),
            simulator,
            driver,
            bridge,
            zero_effort,
        ]
    )
