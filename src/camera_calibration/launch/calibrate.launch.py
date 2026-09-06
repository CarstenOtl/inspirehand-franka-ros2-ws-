"""Start the D415 and collect eye-to-hand calibration samples."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


ARGS = (
    ("tag_id", "0", "Numeric ID printed on the AprilTag."),
    ("tag_size_m", "0.040", "Black-square edge length in metres; measure it accurately."),
    ("tag_family", "tag36h11", "AprilTag family: tag16h5, tag25h9, tag36h10, or tag36h11."),
    ("world_frame", "world", "Fixed frame at the Franka base."),
    ("hand_frame", "fr3_link8", "Moving TF frame rigidly carrying the hand and tag."),
    ("camera_mount_frame", "camera_link", "Camera frame that will become a child of world."),
    ("camera_optical_frame", "", "Optical frame; empty takes it from CameraInfo."),
    ("image_topic", "/camera/camera/color/image_raw", "Rectified or raw color image."),
    ("camera_info_topic", "/camera/camera/color/camera_info", "Matching color intrinsics."),
    ("auto_capture", "true", "Automatically collect sufficiently different poses."),
    ("minimum_samples", "12", "Smallest sample set accepted by the solver."),
)


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_camera",
                default_value="true",
                description="Also launch the RealSense D415 driver.",
            ),
            *[
                DeclareLaunchArgument(name, default_value=default, description=description)
                for name, default, description in ARGS
            ],
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
                    )
                ),
                launch_arguments={"device_type": "d415", "enable_color": "true"}.items(),
                condition=IfCondition(LaunchConfiguration("start_camera")),
            ),
            Node(
                package="camera_calibration",
                executable="calibrate",
                name="camera_calibration",
                output="screen",
                parameters=[{name: LaunchConfiguration(name) for name, _, _ in ARGS}],
            ),
        ]
    )
