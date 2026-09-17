"""The distilled-policy rollout stack against MuJoCo instead of hardware.

    ros2 launch inspire_franka_trajectory_replay sim_policy.launch.py

Then, in a second sourced shell, the hardware runner with its simulated backend:

    python3 apps/policy_rollout/run_policy_rollout.py ros-sim --device cpu --yes

The MuJoCo window shows the scene from a free camera; press ``]`` in it to
cycle to the calibrated ``policy_d415`` view. ``camera_view:=true`` (default)
also opens rqt_image_view on the relayed colour stream the policy consumes.

Everything the hardware backend talks to is here, under the same names:

- the FR3 in ``inspire_franka_policy_scene.xml`` (the flange torque scene in
  the student's training environment: bench, M24 bolt and nut) with
  ``trajectory_replay_controller`` active for homing and
  ``cartesian_trajectory_replay_controller`` inactive, both from
  ``controllers_sim_policy.yaml`` (the policy profile on the DH model);
- the Inspire hand driver in ``mock`` mode and ``inspire_hand_sim_bridge``, so
  ``/inspire_hand/command`` open ratios go through the driver's conversion,
  thumb-yaw overlay and quantisation before reaching MuJoCo;
- the calibrated D415: the MJCF camera is rendered by mujoco_ros2_control and
  ``policy_camera_relay`` republishes it as ``/camera/camera/rgbd`` plus the
  colour/aligned-depth topics, with the profile's camera matrix and frame, and
  a static fr3_link0 -> camera_color_optical_frame transform for RViz.

What it does not model: contact (the hand model has no collision geometry and
the nut is fixed), the FR3's own dynamics (DH kinematics, no coriolis, gravity
off), RealSense depth noise, and the hand's RS485 timing. It exercises frames,
units, rates, the controller switch, the policy command path and the camera
contract end to end, and shows in the viewer where each command sends the arm.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

CONTROLLER = "trajectory_replay_controller"
CARTESIAN_CONTROLLER = "cartesian_trajectory_replay_controller"
SCENE = "inspire_franka_policy_scene.xml"
MJCF_CAMERA = "policy_d415"


def _default_calibration() -> str:
    # install/<pkg>/share/<pkg> -> workspace root
    share = Path(get_package_share_directory("inspire_franka_trajectory_replay"))
    return str(
        share.parents[3]
        / "apps/policy_rollout/utils/camera_calibration/fr3_realsense_dp3.yaml"
    )


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    calibration_path = Path(arg("camera_calibration"))
    if not calibration_path.is_file():
        raise RuntimeError(
            f"camera profile {calibration_path} not found; pass camera_calibration:=<yaml>"
        )
    camera = yaml.safe_load(calibration_path.read_text())["training_camera"]
    physical = yaml.safe_load(calibration_path.read_text())["physical_camera"]
    rgbd_topic = physical["rgbd_topic"]
    prefix = rgbd_topic.rsplit("/", 1)[0]
    pose = camera["world_pose"]

    controllers_yaml = str(
        Path(get_package_share_directory("inspire_franka_trajectory_replay"))
        / "config"
        / "controllers_sim_policy.yaml"
    )
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("inspire_franka_sim")) / "launch" / "sim.launch.py")
        ),
        launch_arguments={
            "mount": "flange",
            "hardware_type": "mujoco",
            "headless": arg("headless"),
            "start_rviz": arg("start_rviz"),
            "arm_command_interface": "none",
            "hand_command_interface": "position_direct",
            "controllers_config_path": controllers_yaml,
            "mjcf": SCENE,
            "camera_publish_rate": arg("camera_rate"),
            "sim_speed_factor": arg("sim_speed"),
        }.items(),
    )

    def spawner(*extra):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[*extra, "--param-file", controllers_yaml, "--controller-manager-timeout", "60"],
            output="screen",
        )

    hand_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("inspire_hand_driver")) / "launch" / "inspire_hand.launch.py")
        ),
        launch_arguments={
            "mock": "true",
            "node_name": "inspire_hand",
            "publish_description": "false",
        }.items(),
    )
    bridge = Node(
        package="inspire_franka_sim",
        executable="inspire_hand_sim_bridge.py",
        output="screen",
        parameters=[{"driver_joint_states": "/inspire_hand/joint_states"}],
    )
    relay = Node(
        package="inspire_franka_sim",
        executable="policy_camera_relay.py",
        output="screen",
        parameters=[
            {
                "source": "/" + MJCF_CAMERA,
                "frame_id": camera["frame_id"],
                "camera_matrix": [float(v) for v in camera["color"]["camera_matrix"]],
                "output_prefix": prefix,
                "use_sim_time": True,
            }
        ],
    )
    w, x, y, z = (str(float(v)) for v in pose["rotation_wxyz"])
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", str(pose["translation_m"][0]),
            "--y", str(pose["translation_m"][1]),
            "--z", str(pose["translation_m"][2]),
            "--qx", x, "--qy", y, "--qz", z, "--qw", w,
            "--frame-id", pose["parent_frame_id"],
            "--child-frame-id", pose["child_frame_id"],
        ],
        parameters=[{"use_sim_time": True}],
        output="log",
    )
    camera_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        arguments=[physical["color_topic"]],
        output="log",
        condition=IfCondition(arg("camera_view")),
    )
    return [
        camera_view,
        simulator,
        spawner(CONTROLLER),
        spawner(CARTESIAN_CONTROLLER, "--inactive"),
        hand_driver,
        bridge,
        relay,
        camera_tf,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="false opens the MuJoCo viewer (needs xhost +local:root on the host).",
            ),
            DeclareLaunchArgument("start_rviz", default_value="false", description="Start RViz2."),
            DeclareLaunchArgument(
                "camera_rate",
                default_value="15.0",
                description="Render/publish rate of the simulated D415. The physical stream "
                "is 30 Hz, but the policy reads at most 15 Hz and every rendered frame "
                "costs CPU the policy loop needs.",
            ),
            DeclareLaunchArgument(
                "sim_speed",
                default_value="0.5",
                description="Simulated seconds per wall-clock second. The ros-sim runner "
                "paces the policy on /clock, so below 1 the policy keeps its trained "
                "15 Hz in simulated time while inference, rendering and the viewers "
                "share this 4-core CPU. 1.0 is real time.",
            ),
            DeclareLaunchArgument(
                "camera_view",
                default_value="true",
                description="Open rqt_image_view on the simulated D415 colour stream.",
            ),
            DeclareLaunchArgument(
                "camera_calibration",
                default_value=_default_calibration(),
                description="Camera profile YAML whose matrix, frame and topics the simulated D415 uses.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
