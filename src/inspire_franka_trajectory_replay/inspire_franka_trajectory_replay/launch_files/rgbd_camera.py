"""The D415 at its full colour resolution, for ``replay_trajectory --record-rgbd``.

    ros2 launch inspire_franka_trajectory_replay rgbd_camera.launch.py

This is the ordinary ``realsense2_camera`` launch with the arguments a
full-resolution RGB-D recording needs, stated once here rather than retyped:

- colour at 1920x1080, the D415's largest colour mode, at 30 Hz;
- depth at 1280x720, the sensor's largest depth mode, at the same rate;
- ``align_depth.enable`` so the depth is reprojected into the colour image
  and comes out at the colour resolution, in the colour frame;
- ``enable_sync`` and ``enable_rgbd`` so colour and aligned depth from one
  frameset are published together as ``realsense2_camera_msgs/RGBD`` on
  ``/camera/camera/rgbd``, which is what the recording subscribes to.

Only one node can own the device. The policy rollout's camera runs at 640x480
and must be stopped first; ``replay_trajectory --record-rgbd`` checks the live
stream's resolution and refuses to move the arm if it finds that one.

Aligning depth to a 1080p colour image is done on the CPU by librealsense.
Whether this workcell sustains 30 Hz at that size is a bench measurement; the
recording preflight prints the delivered rate, and ``color_profile`` /
``depth_profile`` are launch arguments so a lower mode can be chosen
deliberately (then pass the matching ``--rgbd-resolution`` to the runner).

The description is built here, under the package, so its arguments can be
imported and tested; ``launch/rgbd_camera.launch.py`` is the entry point.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ..rgbd_recording import D415_FULL_COLOR, D415_FULL_DEPTH, DEFAULT_RATE_HZ

RATE = int(DEFAULT_RATE_HZ)

#: The full-resolution profiles, as the RealSense launch spells them.
COLOR_PROFILE = f"{D415_FULL_COLOR[0]}x{D415_FULL_COLOR[1]}x{RATE}"
DEPTH_PROFILE = f"{D415_FULL_DEPTH[0]}x{D415_FULL_DEPTH[1]}x{RATE}"

#: What is fixed. The RGBD composite, sync and alignment are not optional:
#: without all three the recording has no aligned frame pair to record.
FIXED_CAMERA_ARGUMENTS = {
    "device_type": "d415",
    "enable_color": "true",
    "enable_depth": "true",
    "enable_sync": "true",
    "align_depth.enable": "true",
    "enable_rgbd": "true",
    # No point cloud and no IMU: the bag holds the aligned frames, and a point
    # cloud at 1080p would triple the node's load for data the depth already is.
    "pointcloud.enable": "false",
}

#: What the operator may change, with the defaults a full-resolution take wants.
LAUNCH_ARGUMENTS = (
    ("color_profile", COLOR_PROFILE, "colour stream WxHxFPS"),
    ("depth_profile", DEPTH_PROFILE, "depth stream WxHxFPS; the aligned depth comes out at the colour size"),
    ("serial_no", "''", "the device serial, when more than one RealSense is attached"),
    ("camera_namespace", "camera", "namespace of the RealSense node"),
    ("camera_name", "camera", "name of the RealSense node"),
    ("initial_reset", "false", "reset the device once before streaming; never while another node owns it"),
)


def generate_launch_description():
    launch_arguments = dict(FIXED_CAMERA_ARGUMENTS)
    launch_arguments.update(
        {
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            "serial_no": LaunchConfiguration("serial_no"),
            "camera_namespace": LaunchConfiguration("camera_namespace"),
            "camera_name": LaunchConfiguration("camera_name"),
            "initial_reset": LaunchConfiguration("initial_reset"),
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
