"""The D415 as the policy rollout needs it.

    ros2 launch inspire_franka_trajectory_replay policy_camera.launch.py

This is the ordinary ``realsense2_camera`` launch with the arguments the
distilled student's camera contract needs, stated once here rather than retyped
on the command line:

- colour and depth both 640x480 at 15 Hz. The size is what
  ``apps/policy_rollout/utils/camera_calibration/fr3_realsense_dp3.yaml``
  calibrated and what the checkpoint's intrinsics assume; the rate is that
  profile's own ``policy_input.snapshot_rate_hz``, and the policy reads no
  faster, so a 30 Hz stream only costs the rollout process deserialisation it
  throws away;
- ``align_depth.enable``, ``enable_sync`` and ``enable_rgbd``, so one frameset
  arrives as ``realsense2_camera_msgs/RGBD`` on ``/camera/camera/rgbd``, which
  is what the rollout subscribes to;
- ``rgb_camera.auto_exposure_priority`` **false**. This one is not cosmetic.

Why auto-exposure priority is forced off. With it on -- the device default --
auto-exposure is allowed to drop the colour frame rate to reach its brightness
target, and in this workcell's light it does. Measured on ``/camera/camera/rgbd``
over 400-sample windows: 12.5-12.9 Hz with frames arriving 8 ms apart in bursts
separated by gaps of up to 0.466 s. ``rgbd`` only publishes when a colour and a
depth frame pair, so it inherits colour's gaps while depth runs on at 14.9 Hz.
The rollout rejects inputs older than its ``--max-state-age`` (0.5 s by
default), and its own subscriber is slower than a bare ``ros2 topic hz``, so
those 0.466 s gaps became over 0.5 s at the policy and aborted five hardware
runs with ``stale live inputs: rgb, depth`` between steps 6 and 43. With the
option off the same measurement reads 14.1 Hz, minimum gap 0.063 s, maximum
0.202 s -- paced rather than clumped, and 2.5x inside the guard.

Capping exposure rather than fixing it keeps auto-brightness working; the
sensor trades integration time for gain instead of for frames. That matters
because the colour image is the policy's vision input and its appearance is
already a sim-to-real gap worth not widening.

``rs_launch.py`` only forwards launch arguments it declares -- anything else
draws a warning and is dropped -- and ``auto_exposure_priority`` is not among
them. It does merge a ``config_file`` straight into the node's parameters,
which is why that option travels in :data:`CAMERA_PARAMETER_FILE` rather than
as a launch argument.

The description is built here, under the package, so its arguments can be
imported and tested; ``launch/policy_camera.launch.py`` is the entry point.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

#: The calibrated policy colour size, and the profile's own snapshot rate.
POLICY_COLOR = (640, 480)
POLICY_DEPTH = (640, 480)
POLICY_RATE_HZ = 15

COLOR_PROFILE = f"{POLICY_COLOR[0]}x{POLICY_COLOR[1]}x{POLICY_RATE_HZ}"
DEPTH_PROFILE = f"{POLICY_DEPTH[0]}x{POLICY_DEPTH[1]}x{POLICY_RATE_HZ}"

#: Parameters ``rs_launch.py`` does not declare, so they travel in a file.
CAMERA_PARAMETER_FILE = "policy_camera_params.yaml"

#: What is fixed. The composite, sync and alignment are not optional: without
#: all three the rollout has no aligned frame pair to read.
FIXED_CAMERA_ARGUMENTS = {
    "device_type": "d415",
    "enable_color": "true",
    "enable_depth": "true",
    "enable_sync": "true",
    "align_depth.enable": "true",
    "enable_rgbd": "true",
    # The policy back-projects its own cloud from the aligned depth, so the
    # driver's point cloud would be load for data the rollout never reads.
    "pointcloud.enable": "false",
}

#: What the operator may change, with the defaults the policy rollout wants.
LAUNCH_ARGUMENTS = (
    ("color_profile", COLOR_PROFILE, "colour stream WxHxFPS; the checkpoint calibrated 640x480"),
    ("depth_profile", DEPTH_PROFILE, "depth stream WxHxFPS; the aligned depth comes out at the colour size"),
    ("serial_no", "''", "the device serial, when more than one RealSense is attached"),
    ("camera_namespace", "camera", "namespace of the RealSense node"),
    ("camera_name", "camera", "name of the RealSense node"),
    ("initial_reset", "false", "reset the device once before streaming; never while another node owns it"),
)


def generate_launch_description():
    parameter_file = PathJoinSubstitution(
        [
            FindPackageShare("inspire_franka_trajectory_replay"),
            "config",
            CAMERA_PARAMETER_FILE,
        ]
    )
    launch_arguments = dict(FIXED_CAMERA_ARGUMENTS)
    launch_arguments.update(
        {
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            "serial_no": LaunchConfiguration("serial_no"),
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "camera_name": LaunchConfiguration("camera_name"),
            "initial_reset": LaunchConfiguration("initial_reset"),
            "config_file": parameter_file,
        }
    )
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments=list(launch_arguments.items()),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(name, default_value=default, description=description)
            for name, default, description in LAUNCH_ARGUMENTS
        ]
        + [camera]
    )
