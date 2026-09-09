"""Start the D415 full-resolution RGB stream and collect calibration samples."""

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
    ("world_frame", "fr3_link0", "Fixed frame at the Franka base."),
    ("hand_frame", "fr3_link8", "Moving TF frame rigidly carrying the hand and tag."),
    ("camera_mount_frame", "camera_link", "Camera frame that will become a child of world."),
    ("camera_optical_frame", "", "Optical frame; empty takes it from CameraInfo."),
    ("output_root", "logs", "Parent directory for timestamped sample/result folders."),
    ("image_topic", "/camera/camera/color/image_raw", "Rectified or raw color image."),
    ("camera_info_topic", "/camera/camera/color/camera_info", "Matching color intrinsics."),
    (
        "capture_mode",
        "manual",
        "manual records guided poses; auto moves through 12 poses; triggered waits for capture.",
    ),
    (
        "auto_motion_authorized",
        "false",
        "Required explicit safety acknowledgement before auto mode may command the robot.",
    ),
    (
        "trajectory_action",
        "/fr3_arm_controller/follow_joint_trajectory",
        "FollowJointTrajectory action used by auto mode.",
    ),
    ("auto_waypoint_duration_s", "5.0", "Seconds for each automatic waypoint move."),
    ("auto_sample_timeout_s", "8.0", "Maximum wait for a valid sample at each waypoint."),
    ("minimum_samples", "12", "Smallest sample set accepted by the solver."),
)


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_camera",
                default_value="true",
                description="Also launch the RealSense D415 RGB driver.",
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
                launch_arguments={
                    "device_type": "d415",
                    "enable_color": "true",
                    "enable_depth": "false",
                    "rgb_camera.color_profile": "1920x1080x30",
                }.items(),
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
