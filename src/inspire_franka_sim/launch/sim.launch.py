"""Bring up the FR3 and the Inspire hand in MuJoCo, under ros2_control.

Terminal-first: nothing opens a window unless you ask for it.

    ros2 launch inspire_franka_sim sim.launch.py                    # arm + flange hand
    ros2 launch inspire_franka_sim sim.launch.py headless:=false    # MuJoCo viewer
    ros2 launch inspire_franka_sim sim.launch.py start_rviz:=true   # RViz
    ros2 launch inspire_franka_sim sim.launch.py hand:=false        # bare arm
    ros2 launch inspire_franka_sim sim.launch.py arm:=false         # hand only

The two assets are driven independently, through `arm_command_interface` and
`hand_command_interface`. Either may be `none`, which leaves that asset
unclaimed by any controller.

Close the hand (channel order: little, ring, middle, index, thumb bend, thumb
rotation - the same order the real driver uses):

    ros2 action send_goal /hand_joint_trajectory_controller/follow_joint_trajectory \
      control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names:
      [index_proximal_joint, middle_proximal_joint, ring_proximal_joint,
      pinky_proximal_joint], points: [{positions: [1.2, 1.2, 1.2, 1.2],
      time_from_start: {sec: 2}}]}}"

`allow_partial_joints_goal` is on for the hand, so a goal may name any subset of
its six driven joints. The six followers must NOT be named - they have no
command interface, and MuJoCo holds them to their coupling.

Move the arm:

    ros2 action send_goal /fr3_joint_trajectory_controller/follow_joint_trajectory \
      control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names:
      [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6,
      fr3_joint7], points: [{positions: [0.5,-0.4,0.3,-2.0,0.2,1.4,0.9],
      time_from_start: {sec: 3}}]}}"

Both GUIs need `xhost +local:root` on the host.

The current MuJoCo window provided by mujoco_ros2_control does not expose a
joint/actuator slider panel. Setting either command interface to `none` only
leaves those joints unclaimed; it does not add interactive GUI controls. Use ROS
commands to drive this simulated plant. The separate passive viewer has local
actuator sliders, but actuation is deliberately disabled there, so they do not
pose its model or publish commands to this simulation or hardware.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.parameter_descriptions import ParameterValue
import xacro

# Which MJCF goes with which combination of assets. The URDF and the MJCF have
# to describe the same robot - ros2_control matches them by joint name - so one
# set of launch arguments picks both, and they cannot drift apart by accident.
SCENES = {
    ("arm", "hand"): "inspire_franka_flange_scene.xml",
    ("arm", None): "fr3_scene.xml",
    (None, "hand"): "inspire_hand_scene.xml",
}

# Each scene's rest keyframe. The combined scenes need their own ("start"),
# because fr3.xml's "home" covers only the arm and MuJoCo zero-pads a keyframe
# to the whole model, which would put the hand followers somewhere wrong. See
# the comment in inspire_franka_flange_scene.xml.
KEYFRAMES = {
    "inspire_franka_flange_scene.xml": "start",
    "fr3_scene.xml": "home",
    "inspire_hand_scene.xml": "open",
}

# controller name -> the value of *_command_interface that spawns it.
ARM_CONTROLLERS = {
    "fr3_joint_trajectory_controller": "position",
    "fr3_position_forward_command_controller": "position_direct",
    "fr3_effort_forward_command_controller": "effort",
}
HAND_CONTROLLERS = {
    "hand_joint_trajectory_controller": "position",
    "hand_position_forward_command_controller": "position_direct",
    "hand_effort_forward_command_controller": "effort",
}


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return arg(name).lower() in ("true", "1")

    with_arm, with_hand = flag("arm"), flag("hand")
    hardware_type = arg("hardware_type")

    if not (with_arm or with_hand):
        raise RuntimeError("arm and hand are both false; there is nothing to simulate")

    # The shipped scenes all use the right hand. Silently simulating a right
    # hand while TF publishes a left one would be a genuinely confusing bug, so
    # refuse instead - and say what would fix it.
    if with_hand and hardware_type == "mujoco" and arg("hand_side") != "right" and not arg("mjcf"):
        raise RuntimeError(
            "hand_side:=left has no MuJoCo scene. mjcf/inspire_hand_left.xml is "
            "generated, but no scene binds it; write one modelled on "
            "inspire_franka_flange_scene.xml and pass it as mjcf:=<file>. "
            "hardware_type:=mock works with either side."
        )

    key = ("arm" if with_arm else None, "hand" if with_hand else None)
    scene = arg("mjcf") or SCENES[key]

    sim_share = get_package_share_directory("inspire_franka_sim")
    mjcf_path = scene if os.path.isabs(scene) else os.path.join(sim_share, "mjcf", scene)
    controllers_yaml = arg("controllers_config_path")
    pids_yaml = arg("pids_config_path")

    robot_description = xacro.process_file(
        arg("xacro_path"),
        mappings={
            "arm": "true" if with_arm else "false",
            "hand": "true" if with_hand else "false",
            "hand_side": arg("hand_side"),
            "ros2_control": "true",
            "hardware_type": hardware_type,
            "mujoco_model": mjcf_path,
            "pids_config_file": pids_yaml,
            "headless": arg("headless").lower(),
            "initial_keyframe": KEYFRAMES.get(os.path.basename(mjcf_path), "home"),
        },
    ).toxml()

    # MuJoCo owns the clock, so everything downstream has to run on sim time.
    use_sim_time = {"use_sim_time": hardware_type == "mujoco"}

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[
                {"robot_description": ParameterValue(robot_description, value_type=str)},
                use_sim_time,
            ],
        ),
        # NOTE: package="mujoco_ros2_control", NOT "controller_manager".
        # mujoco_ros2_control ships its own ros2_control_node - same executable
        # name and the same parameters, but it also owns the MuJoCo Simulate
        # app. Using controller_manager's would load the plugin with no
        # simulator behind it.
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[use_sim_time, ParameterFile(controllers_yaml, allow_substs=True)],
            # If MuJoCo dies, tear the whole launch down rather than leaving
            # orphaned spawners waiting on a controller_manager that is gone.
            on_exit=Shutdown(),
        ),
    ]

    def spawner(controller):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                controller,
                "--param-file",
                controllers_yaml,
                # The default 10 s is not enough when the machine is busy at
                # startup - MuJoCo loading meshes, or RViz coming up alongside.
                # On a timeout the spawner moves on and calls
                # configure_controller before the manager has processed the
                # load, which fails with the misleading "no controller with
                # this name exists".
                "--controller-manager-timeout",
                "60",
                "--service-call-timeout",
                "60",
            ],
            output="both",
        )

    nodes.append(spawner("joint_state_broadcaster"))

    # One controller per asset, spawned in addition to each other rather than
    # instead: they claim disjoint sets of joints.
    if with_arm:
        wanted = arg("arm_command_interface")
        nodes += [c for c, v in ARM_CONTROLLERS.items() if v == wanted]
    if with_hand:
        wanted = arg("hand_command_interface")
        nodes += [c for c, v in HAND_CONTROLLERS.items() if v == wanted]
    nodes = [spawner(n) if isinstance(n, str) else n for n in nodes]

    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", arg("rviz_config_path")],
            parameters=[use_sim_time],
            output="both",
            condition=IfCondition(LaunchConfiguration("start_rviz")),
        )
    )
    return nodes


def generate_launch_description():
    sim_share = get_package_share_directory("inspire_franka_sim")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "arm", default_value="true", description="Include the FR3."
            ),
            DeclareLaunchArgument(
                "hand", default_value="true", description="Include the Inspire hand."
            ),
            DeclareLaunchArgument(
                "hand_side",
                default_value="right",
                description="'left' or 'right'. Note the shipped MJCF scenes use the "
                "right hand; a left-handed scene needs its own MJCF.",
            ),
            DeclareLaunchArgument(
                "hardware_type",
                default_value="mujoco",
                description="ros2_control backend: 'mujoco' for physics, 'mock' for "
                "mock_components/GenericSystem (no simulator, mirrors commands back as "
                "states - useful for checking plumbing on a machine with no GPU).",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="Run MuJoCo without its Simulate window. On by default: the "
                "stack is terminal-first. Pass headless:=false for the viewer.",
            ),
            DeclareLaunchArgument(
                "arm_command_interface",
                default_value="position",
                description="Which controller owns the arm's joints (only one may): "
                "'position' -> fr3_joint_trajectory_controller; 'position_direct' -> "
                "fr3_position_forward_command_controller; 'effort' -> "
                "fr3_effort_forward_command_controller; 'none' -> leave it unclaimed.",
            ),
            DeclareLaunchArgument(
                "hand_command_interface",
                default_value="position",
                description="Which controller owns the hand's six driven joints: "
                "'position' -> hand_joint_trajectory_controller; 'position_direct' -> "
                "hand_position_forward_command_controller; 'effort' -> "
                "hand_effort_forward_command_controller; 'none' -> leave it unclaimed.",
            ),
            DeclareLaunchArgument(
                "start_rviz", default_value="false", description="Start RViz2."
            ),
            DeclareLaunchArgument(
                "mjcf",
                default_value="",
                description="Override the MuJoCo scene. A bare filename resolves inside "
                "the package's mjcf/ directory; an absolute path is used as given. Empty "
                "means pick the scene that matches the enabled arm/hand assets.",
            ),
            DeclareLaunchArgument(
                "xacro_path",
                default_value=os.path.join(
                    get_package_share_directory("inspire_franka_description"),
                    "urdf",
                    "inspire_franka.urdf.xacro",
                ),
                description="Robot description xacro.",
            ),
            DeclareLaunchArgument(
                "controllers_config_path",
                default_value=os.path.join(sim_share, "config", "controllers.yaml"),
                description="ros2_control controllers configuration.",
            ),
            DeclareLaunchArgument(
                "pids_config_path",
                default_value=os.path.join(sim_share, "config", "pids.yaml"),
                description="PID gains mujoco_ros2_control closes position and velocity "
                "loops with, on top of the MJCF's torque actuators.",
            ),
            DeclareLaunchArgument(
                "rviz_config_path",
                default_value=os.path.join(sim_share, "rviz", "inspire_franka.rviz"),
                description="RViz2 config.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
