"""The distilled-policy rollout stack against MuJoCo instead of hardware.

    ros2 launch inspire_franka_trajectory_replay sim_policy.launch.py

Then, in a second sourced shell, the hardware runner with its simulated backend:

    python3 apps/policy_rollout/run_policy_rollout.py ros-sim --device cpu --yes

The MuJoCo window shows the scene from a free camera; press ``]`` in it to
cycle to the calibrated ``policy_d415`` view. ``camera_view:=true`` (default)
also opens rqt_image_view on the relayed colour stream the policy consumes.

Everything the hardware backend talks to is here, under the same names:

- the FR3 in ``inspire_franka_policy_scene.xml`` (the flange torque scene in
  the student's training environment: bench, M24 bolt, and the nut on
  ForgeUltra's thread pair) with
  ``trajectory_replay_controller`` active for homing and
  ``cartesian_trajectory_replay_controller`` inactive, both from
  ``controllers_sim_policy.yaml`` (the policy profile on the DH model);
- the hand as Isaac Lab simulates it in training (``hand_chain:=isaac``, the
  default): the training hand with Isaac's drives, armature and effort limits
  (``inspire_hand_right_isaac.xml``), ``inspire_franka_sim``'s IsaacHandPlugin
  for its target-level mimic, 1 rad/s velocity limit and PhysX joint friction,
  and ``isaac_hand_relay.py`` serving ``/inspire_hand/command`` and
  ``/inspire_hand/joint_states`` from the simulated joints. ``hand_chain:=driver``
  puts the Inspire hand driver in ``mock`` mode and ``inspire_hand_sim_bridge``
  in between instead, so open ratios go through the driver's conversion,
  quantisation and wall-clock slew, and the runner sees the driver's state (the
  thumb-yaw overlay is off: the simulated hand is the training hand, so the
  logical command is the pose);
- the calibrated D415: the MJCF camera is rendered by mujoco_ros2_control and
  ``policy_camera_relay`` republishes it as ``/camera/camera/rgbd`` plus the
  colour/aligned-depth topics, with the profile's camera matrix and frame, and
  a static fr3_link0 -> camera_color_optical_frame transform for RViz.

- ``inspire_franka_sim``'s thread pair plugin, loaded through the
  ``mujoco_plugins`` parameter in ``controllers_sim_policy.yaml``. The nut is
  not part of the robot, so no ros2_control interface reaches it; the plugin
  publishes its twist on ``/thread_state`` and clamps the thread on
  ``/thread_hold``, which is what lets the runner gate the release on the
  nut's real turn instead of on grasp-frame yaw.

What it does not model: the FR3's own dynamics (the controller runs on DH
kinematics with no coriolis term, and gravity is off in the scene in place of
libfranka's compensation), RealSense depth noise, and the hand's RS485 timing.
The hand is the training hand (inspire_hand_description is generated from
forgeUltra's URDF), so the simulated pads land where training's did. Whether
the physical pads do is still a bench measurement (D1 in
docs/policy_rollout_training_gap_todo.md).
"""

from pathlib import Path
import re

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
#: task_profile -> scene, for the mimic-hand students whose reset is not the default scene's.
TASK_PROFILE_SCENES = {"mimic_windback45_tilt30": "inspire_franka_policy_scene_windback45_tilt30.xml"}
MJCF_CAMERA = "policy_d415"


# The arm the scene includes: MuJoCo Menagerie's FR3 v2 under the workspace's
# fr3_ names (inspire_franka_sim/mjcf/fr3v2.xml). Its joint friction, damping and
# armature are upstream's; arm_friction_scale and arm_armature_scale rescale them.
ARM_MJCF = "fr3v2.xml"
VARIANT_SCENE = "inspire_franka_policy_scene_arm_variant.xml"
VARIANT_ARM = "fr3v2_arm_variant.xml"
_ATTRIBUTE = r'\s{name}="([^"]*)"'


def _joint_dynamics(arm: str) -> list:
    """(frictionloss, damping, armature) of fr3_joint1..7: the joint's own value, else its class default."""

    default = re.search(r'<default class="fr3v2">\s*(?:<!--.*?-->\s*)*<joint([^>]*)/>', arm, re.S)
    if default is None:
        raise RuntimeError(f"no fr3v2 joint default in {ARM_MJCF}")
    values = []
    for i in range(1, 8):
        match = re.search(rf'<joint name="fr3_joint{i}"[^>]*?/>', arm, re.S)
        if match is None:
            raise RuntimeError(f"fr3_joint{i} not found in {ARM_MJCF}")
        row = []
        for name in ("frictionloss", "damping", "armature"):
            own = re.search(_ATTRIBUTE.format(name=name), match.group(0))
            inherited = re.search(_ATTRIBUTE.format(name=name), default.group(1))
            row.append(float((own or inherited).group(1)) if (own or inherited) else 0.0)
        values.append(tuple(row))
    return values


def _friction_compensation(value: str):
    """``arm_friction_compensation``: empty for off, one number for joint 7, or seven."""

    value = value.strip()
    if not value:
        return None
    numbers = [float(v) for v in value.replace(",", " ").split()]
    if len(numbers) == 1:
        numbers = [0.0] * 6 + numbers
    if len(numbers) != 7 or any(not 0.0 <= v <= 2.0 for v in numbers):
        raise RuntimeError("arm_friction_compensation takes one (joint 7) or seven values in [0, 2] Nm")
    return numbers


#: ForgeUltra's default_dead_zone (vanilla_threading.yaml): the top of training's U(0, d).
WORST_CASE_WRENCH_DEAD_ZONE = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


def _wrench_dead_zone(value: str):
    """``arm_wrench_dead_zone``: empty for off, ``worst`` for training's maximum, or six values."""

    value = value.strip()
    if not value:
        return None
    if value == "worst":
        return list(WORST_CASE_WRENCH_DEAD_ZONE)
    numbers = [float(v) for v in value.replace(",", " ").split()]
    if len(numbers) != 6 or any(not 0.0 <= v <= (50.0 if i < 3 else 10.0) for i, v in enumerate(numbers)):
        raise RuntimeError(
            "arm_wrench_dead_zone takes 'worst' or six values (3 x N in [0, 50], 3 x N m in [0, 10])"
        )
    return numbers


def _controllers_variant(controllers_yaml: str, compensation, dead_zone) -> str:
    """Write the controller profile with friction_compensation_nm and/or task_wrench_dead_zone
    replaced, next to the original."""

    source = Path(controllers_yaml)
    text = source.read_text()
    for key, values in (("friction_compensation_nm", compensation), ("task_wrench_dead_zone", dead_zone)):
        if values is None:
            continue
        line = f"{key}: [" + ", ".join(repr(float(v)) for v in values) + "]"
        text, count = re.subn(rf"{key}: \[[^\]]*\]", line, text)
        if count != 1:
            raise RuntimeError(f"expected one {key} in {source}")
    variant = source.with_name("controllers_sim_policy_friction_compensation.yaml")
    variant.write_text(text)
    return str(variant)


def _arm_variant_scene(friction_scale: float, armature_scale: float, base_scene: str = SCENE) -> str:
    """Write the policy scene with the arm's joint friction and armature rescaled.

    Menagerie's dry friction and damping times ``friction_scale`` and its
    armature times ``armature_scale``. Both files go next to the originals in
    the installed mjcf/ directory, so the arm's meshdir and the scene's
    relative asset paths still resolve; they are regenerated on every launch.
    """

    mjcf = Path(get_package_share_directory("inspire_franka_sim")) / "mjcf"
    arm = (mjcf / ARM_MJCF).read_text()
    for i, (friction, damping, armature) in enumerate(_joint_dynamics(arm), start=1):
        match = re.search(rf'<joint name="fr3_joint{i}"[^>]*?/>', arm, re.S)
        tag = re.sub(r'\s(frictionloss|damping|armature)="[^"]*"', "", match.group(0))
        tag = tag[:-2].rstrip() + (
            f' frictionloss="{friction_scale * friction:.6g}"'
            f' damping="{friction_scale * damping:.6g}"'
            f' armature="{armature_scale * armature:.6g}"/>'
        )
        arm = arm[: match.start()] + tag + arm[match.end() :]
    (mjcf / VARIANT_ARM).write_text(arm)
    scene = (mjcf / base_scene).read_text()
    include = f'<include file="{ARM_MJCF}"/>'
    if scene.count(include) != 1:
        raise RuntimeError(f"{base_scene} does not include {ARM_MJCF} exactly once")
    scene = scene.replace(include, f'<include file="{VARIANT_ARM}"/>')
    variant = base_scene.replace(".xml", "_arm_variant.xml")
    (mjcf / variant).write_text(scene)
    return variant


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

    task_profile = arg("task_profile")
    if task_profile and task_profile not in TASK_PROFILE_SCENES:
        raise RuntimeError(
            f"task_profile must be empty or one of {sorted(TASK_PROFILE_SCENES)}, got {task_profile!r}"
        )
    base_scene = TASK_PROFILE_SCENES.get(task_profile, SCENE)
    arm_friction_scale = float(arg("arm_friction_scale"))
    arm_armature_scale = float(arg("arm_armature_scale"))
    if arm_friction_scale < 0.0 or arm_armature_scale < 0.0:
        raise RuntimeError("arm_friction_scale and arm_armature_scale must be >= 0")
    if arm_friction_scale == 1.0 and arm_armature_scale == 1.0:
        scene = base_scene
    else:
        scene = _arm_variant_scene(arm_friction_scale, arm_armature_scale, base_scene)

    controllers_yaml = str(
        Path(get_package_share_directory("inspire_franka_trajectory_replay"))
        / "config"
        / "controllers_sim_policy.yaml"
    )
    compensation = _friction_compensation(arg("arm_friction_compensation"))
    dead_zone = _wrench_dead_zone(arg("arm_wrench_dead_zone"))
    if compensation is not None or dead_zone is not None:
        controllers_yaml = _controllers_variant(controllers_yaml, compensation, dead_zone)
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("inspire_franka_sim")) / "launch" / "sim.launch.py")
        ),
        launch_arguments={
            "mount": "flange",
            "hardware_type": "mujoco",
            "headless": arg("headless"),
            "start_rviz": arg("start_rviz"),
            "rviz_config_path": arg("rviz_config"),
            "arm_command_interface": "none",
            "hand_command_interface": "position_direct",
            "controllers_config_path": controllers_yaml,
            "mjcf": scene,
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

    hand_chain = arg("hand_chain")
    if hand_chain not in ("isaac", "driver"):
        raise RuntimeError(f"hand_chain must be 'isaac' or 'driver', got {hand_chain!r}")
    if hand_chain == "isaac":
        # Isaac's way: the pinch target goes straight into the joint drives and
        # the runner observes the simulated joints (see isaac_hand_relay.py).
        hand_nodes = [
            Node(
                package="inspire_franka_sim",
                executable="isaac_hand_relay.py",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                    }
                ],
            )
        ]
    else:
        hand_driver = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(Path(get_package_share_directory("inspire_hand_driver")) / "launch" / "inspire_hand.launch.py")
            ),
            launch_arguments={
                "mock": "true",
                "node_name": "inspire_hand",
                "publish_description": "false",
                # The simulated hand's registers are the model, so it has no
                # thumb yaw register scale.
                "thumb_abduction_overlay": "false",
            }.items(),
        )
        bridge = Node(
            package="inspire_franka_sim",
            executable="inspire_hand_sim_bridge.py",
            output="screen",
            parameters=[{"driver_joint_states": "/inspire_hand/joint_states"}],
        )
        hand_nodes = [hand_driver, bridge]
    relay = Node(
        package="inspire_franka_sim",
        executable="policy_camera_relay.py",
        output="screen",
        parameters=[
            {
                "source": arg("camera_source").rstrip("/"),
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
        *hand_nodes,
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
            DeclareLaunchArgument(
                "start_rviz",
                default_value="false",
                description="Start RViz2 on rviz_config. With run_policy_rollout.py --debug-viz "
                "this shows the 4096 points the DP3 encoder actually sampled.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=str(
                    Path(get_package_share_directory("inspire_franka_sim"))
                    / "rviz"
                    / "policy_debug.rviz"
                ),
                description="RViz2 config. The default shows the policy's DP3 cloud, the points "
                "above the table in the base frame, and the control frames published by "
                "run_policy_rollout.py --debug-viz.",
            ),
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
                "hand_chain",
                default_value="isaac",
                description="'isaac' (default): policy pinch targets go straight to the "
                "hand's joint drives and the runner observes the simulated joints, as in "
                "Isaac Lab training. 'driver': through the mock RS485 driver and "
                "inspire_hand_sim_bridge.py, which slews in wall time and reports its own "
                "state rather than the simulated fingers'; rehearses the driver contract.",
            ),
            DeclareLaunchArgument(
                "task_profile",
                default_value="",
                description="Empty (default): the default policy scene. "
                "'mimic_windback45_tilt30': the old-hand exp_19 student's scene, the mimic hand "
                "at the reset wound 45 deg and tilted 30 deg, with the nut at its spawn yaw.",
            ),
            DeclareLaunchArgument(
                "camera_view",
                default_value="true",
                description="Open rqt_image_view on the simulated D415 colour stream.",
            ),
            DeclareLaunchArgument(
                "camera_source",
                default_value="/mujoco_camera_plugin/" + MJCF_CAMERA,
                description="Topic prefix mujoco_ros2_control renders the MJCF camera on. "
                "Its 0.1.2 apt release namespaces camera topics under the plugin node; "
                "older builds published them at /" + MJCF_CAMERA + ".",
            ),
            DeclareLaunchArgument(
                "camera_calibration",
                default_value=_default_calibration(),
                description="Camera profile YAML whose matrix, frame and topics the simulated D415 uses.",
            ),
            DeclareLaunchArgument(
                "arm_friction_scale",
                default_value="1.0",
                description="Scale on the arm's joint dry friction and damping as MuJoCo "
                "Menagerie's FR3 v2 has them (0.2 N m and 0.003 N m s/rad per joint). 1.0 "
                "(default) is Menagerie's model, 0 a frictionless arm. Off the defaults the "
                "launch writes " + VARIANT_SCENE + " and runs that.",
            ),
            DeclareLaunchArgument(
                "arm_friction_compensation",
                default_value="",
                description="Cartesian controller's Coulomb friction feedforward, Nm: one "
                "value for joint 7 or seven for joints 1-7. Empty (default) keeps "
                "controllers_sim_policy.yaml's (off); otherwise the launch writes "
                "controllers_sim_policy_friction_compensation.yaml next to it and uses that.",
            ),
            DeclareLaunchArgument(
                "arm_wrench_dead_zone",
                default_value="",
                description="ForgeUltra's task-wrench dead zone in the Cartesian controller: "
                "'worst' for training's full default [5 5 5 N, 1 1 1 N m] (training draws "
                "U(0, that) per reset), six values, or empty (default, off). Off the default "
                "the launch writes controllers_sim_policy_friction_compensation.yaml.",
            ),
            DeclareLaunchArgument(
                "arm_armature_scale",
                default_value="1.0",
                description="Scale on Menagerie's FR3 v2 rotor inertia (armature 0.195 kg m^2 "
                "per joint). 1.0 (default) keeps it. Below ~0.14 joint 7 goes unstable under "
                "the critically damped Cartesian controller at MuJoCo's 2 ms step.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
