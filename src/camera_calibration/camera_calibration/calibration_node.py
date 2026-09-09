"""Calibrate a fixed camera from an AprilTag on the Inspire Hand's back."""

import json
import math

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from .apriltag_detector import AprilTagDetector
from .auto_waypoints import AUTO_WAYPOINTS, JOINT_NAMES
from .calibration_math import (
    CalibrationError,
    calibrate_fixed_tag_eye_to_hand,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    rotation_angle_deg,
)


APRILTAG_FAMILIES = ("tag16h5", "tag25h9", "tag36h10", "tag36h11")

# Measured T_fr3_link8_apriltag_0 from the current physical/MuJoCo asset.
# The source poses are T_fr3_link8_hand_base_link =
# (xyz 0 0 0.010, q_wxyz 0 0 0 1) and the measured
# apriltag_0 pose in assets/fr3_inspirehand/fr3_inspirehand.xml.  The final
# +90-degree in-plane rotation and 2 mm surface offset match the printed-tag
# frame used by the simulator and OpenCV solvePnP (origin at tag centre).
DEFAULT_HAND_TO_TAG_XYZ = (
    0.0187392994594,
    0.000541679480601,
    0.1020099800138,
)
DEFAULT_HAND_TO_TAG_QUATERNION_XYZW = (
    0.498190676972130,
    -0.499244910192648,
    0.501808192566114,
    -0.500748546576479,
)


def _clean_frame(frame: str) -> str:
    return frame.lstrip("/")


def _transform_from_message(message) -> np.ndarray:
    translation = message.translation
    rotation = message.rotation
    return make_transform(
        quaternion_xyzw_to_matrix([rotation.x, rotation.y, rotation.z, rotation.w]),
        [translation.x, translation.y, translation.z],
    )


class CameraCalibrationNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_calibration")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("hand_frame", "fr3_link8")
        self.declare_parameter("camera_mount_frame", "camera_link")
        self.declare_parameter("camera_optical_frame", "")
        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("tag_id", 0)
        self.declare_parameter("tag_size_m", 0.040)
        self.declare_parameter("capture_mode", "manual")
        self.declare_parameter("auto_motion_authorized", False)
        self.declare_parameter("minimum_samples", 12)
        self.declare_parameter("min_translation_m", 0.025)
        self.declare_parameter("min_rotation_deg", 8.0)
        self.declare_parameter("min_sample_interval_s", 0.35)
        self.declare_parameter("max_reprojection_error_px", 1.5)
        self.declare_parameter("tf_timeout_s", 0.08)
        self.declare_parameter(
            "trajectory_action", "/fr3_arm_controller/follow_joint_trajectory"
        )
        self.declare_parameter("auto_waypoint_duration_s", 5.0)
        self.declare_parameter("auto_sample_timeout_s", 8.0)
        self.declare_parameter("hand_to_tag_x_m", DEFAULT_HAND_TO_TAG_XYZ[0])
        self.declare_parameter("hand_to_tag_y_m", DEFAULT_HAND_TO_TAG_XYZ[1])
        self.declare_parameter("hand_to_tag_z_m", DEFAULT_HAND_TO_TAG_XYZ[2])
        self.declare_parameter("hand_to_tag_qx", DEFAULT_HAND_TO_TAG_QUATERNION_XYZW[0])
        self.declare_parameter("hand_to_tag_qy", DEFAULT_HAND_TO_TAG_QUATERNION_XYZW[1])
        self.declare_parameter("hand_to_tag_qz", DEFAULT_HAND_TO_TAG_QUATERNION_XYZW[2])
        self.declare_parameter("hand_to_tag_qw", DEFAULT_HAND_TO_TAG_QUATERNION_XYZW[3])

        self._world_frame = _clean_frame(self._string_parameter("world_frame"))
        self._hand_frame = _clean_frame(self._string_parameter("hand_frame"))
        self._camera_mount_frame = _clean_frame(self._string_parameter("camera_mount_frame"))
        configured_optical = self._string_parameter("camera_optical_frame")
        self._camera_optical_frame = _clean_frame(configured_optical) if configured_optical else ""
        self._tag_id = int(self.get_parameter("tag_id").value)
        self._tag_size = float(self.get_parameter("tag_size_m").value)
        self._minimum_samples = int(self.get_parameter("minimum_samples").value)
        self._capture_mode = self._string_parameter("capture_mode").lower()
        self._auto_motion_authorized = bool(
            self.get_parameter("auto_motion_authorized").value
        )
        self._min_translation = float(self.get_parameter("min_translation_m").value)
        self._min_rotation = float(self.get_parameter("min_rotation_deg").value)
        self._min_interval_ns = int(
            float(self.get_parameter("min_sample_interval_s").value) * 1.0e9
        )
        self._max_reprojection_error = float(
            self.get_parameter("max_reprojection_error_px").value
        )
        self._tf_timeout = Duration(
            seconds=float(self.get_parameter("tf_timeout_s").value)
        )
        self._trajectory_action = self._string_parameter("trajectory_action")
        self._auto_waypoint_duration = float(
            self.get_parameter("auto_waypoint_duration_s").value
        )
        self._auto_sample_timeout_ns = int(
            float(self.get_parameter("auto_sample_timeout_s").value) * 1.0e9
        )
        self._hand_to_tag = make_transform(
            quaternion_xyzw_to_matrix(
                [
                    float(self.get_parameter("hand_to_tag_qx").value),
                    float(self.get_parameter("hand_to_tag_qy").value),
                    float(self.get_parameter("hand_to_tag_qz").value),
                    float(self.get_parameter("hand_to_tag_qw").value),
                ]
            ),
            [
                float(self.get_parameter("hand_to_tag_x_m").value),
                float(self.get_parameter("hand_to_tag_y_m").value),
                float(self.get_parameter("hand_to_tag_z_m").value),
            ],
        )
        if self._tag_size <= 0.0:
            raise ValueError("tag_size_m must be positive")
        if self._minimum_samples < 4:
            raise ValueError("minimum_samples must be at least 4")
        if not self._world_frame or not self._hand_frame or not self._camera_mount_frame:
            raise ValueError("world_frame, hand_frame, and camera_mount_frame cannot be empty")
        if self._capture_mode not in ("manual", "auto", "triggered"):
            raise ValueError("capture_mode must be 'manual', 'auto', or 'triggered'")
        if self._capture_mode == "auto" and self._minimum_samples != len(AUTO_WAYPOINTS):
            raise ValueError(
                f"auto mode requires minimum_samples={len(AUTO_WAYPOINTS)} "
                "(one sample per waypoint)"
            )
        if self._capture_mode == "auto" and not self._auto_motion_authorized:
            raise ValueError(
                "auto mode requires auto_motion_authorized=true after explicit operator confirmation"
            )
        if self._auto_waypoint_duration <= 0.0 or self._auto_sample_timeout_ns <= 0:
            raise ValueError("automatic motion durations must be positive")

        family = self._string_parameter("tag_family").lower()
        if family not in APRILTAG_FAMILIES:
            choices = ", ".join(APRILTAG_FAMILIES)
            raise ValueError(f"unsupported tag_family '{family}'; choose one of {choices}")
        self._detector = AprilTagDetector(family)

        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._camera_matrix = None
        self._distortion = None
        self._world_to_hand_samples: list[np.ndarray] = []
        self._camera_to_tag_samples: list[np.ndarray] = []
        self._sample_stamps_ns: list[int] = []
        self._last_sample_hand = None
        self._capture_requested = False
        self._calibration_complete = False
        self._last_notice_ns: dict[str, int] = {}
        self._auto_action_client = None
        self._auto_state = "disabled"
        self._auto_waypoint_index = 0
        self._auto_sample_deadline_ns = 0
        self._auto_goal_handle = None

        image_topic = self._string_parameter("image_topic")
        camera_info_topic = self._string_parameter("camera_info_topic")
        self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data
        )
        self.create_subscription(Image, image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_service(Trigger, "~/capture", self._capture_callback)
        self.create_service(Trigger, "~/solve", self._solve_callback)
        self.create_service(Trigger, "~/reset", self._reset_callback)

        if self._capture_mode == "auto":
            self._auto_action_client = ActionClient(
                self, FollowJointTrajectory, self._trajectory_action
            )
            self._auto_state = "waiting_controller"
            self.create_timer(0.25, self._auto_motion_tick)

        mode = (
            "manual hand-guiding with automatic valid-sample capture"
            if self._capture_mode == "manual"
            else (
                f"automatic {len(AUTO_WAYPOINTS)}-waypoint robot motion"
                if self._capture_mode == "auto"
                else "service-triggered (/camera_calibration/capture)"
            )
        )
        self.get_logger().info(
            f"Watching tag {self._tag_id} ({family}, {self._tag_size:.4f} m) on "
            f"{image_topic}; sampling is {mode}."
        )
        self.get_logger().info(
            "Passive recording is active; this node sends no motion commands. "
            f"Guide the hand slowly through varied positions and roll/pitch/yaw; "
            f"its motion is tracked through {_clean_frame(self._hand_frame)}. Keep "
            "the complete tag visible."
        )
        self.get_logger().info(
            f"Calibration will solve automatically after {self._minimum_samples} "
            "accepted samples. The /camera_calibration/solve service remains available."
        )
        self.get_logger().info(
            f"Using the fixed measured transform {self._hand_frame} -> "
            f"apriltag_{self._tag_id}; only the camera pose will be calibrated."
        )
        if self._capture_mode == "auto":
            self.get_logger().info(
                f"Waiting for trajectory action {self._trajectory_action}. Each waypoint "
                f"will take {self._auto_waypoint_duration:.1f} s; a sample is accepted "
                "only after its trajectory goal succeeds."
            )

    def _string_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _notice(self, key: str, message: str, period_s: float = 5.0) -> None:
        now = self.get_clock().now().nanoseconds
        if now - self._last_notice_ns.get(key, 0) >= int(period_s * 1.0e9):
            self.get_logger().warn(message)
            self._last_notice_ns[key] = now

    def _auto_motion_tick(self) -> None:
        if self._capture_mode != "auto" or self._calibration_complete:
            return
        if self._auto_state == "waiting_controller":
            if not self._auto_action_client.server_is_ready():
                self._notice(
                    "trajectory_controller",
                    f"Waiting for {self._trajectory_action}. Start the FR3 trajectory "
                    "controller; gravity-compensation alone cannot run auto mode.",
                )
                return
            self._send_auto_waypoint()
        elif (
            self._auto_state == "waiting_sample"
            and self.get_clock().now().nanoseconds > self._auto_sample_deadline_ns
        ):
            self._capture_requested = False
            self._auto_state = "failed"
            self.get_logger().error(
                f"Auto calibration stopped at waypoint {self._auto_waypoint_index + 1}: "
                "no valid RGB/FK sample arrived before the timeout. Keep the whole tag "
                "visible, then reset and retry."
            )

    def _send_auto_waypoint(self) -> None:
        waypoint_number = self._auto_waypoint_index + 1
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = list(AUTO_WAYPOINTS[self._auto_waypoint_index])
        duration_seconds = int(self._auto_waypoint_duration)
        point.time_from_start.sec = duration_seconds
        point.time_from_start.nanosec = int(
            (self._auto_waypoint_duration - duration_seconds) * 1.0e9
        )
        goal.trajectory.points = [point]
        self._auto_state = "sending_goal"
        self.get_logger().info(
            f"Auto waypoint {waypoint_number}/{len(AUTO_WAYPOINTS)}: moving slowly."
        )
        future = self._auto_action_client.send_goal_async(goal)
        future.add_done_callback(self._auto_goal_response_callback)

    def _auto_goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:
            self._fail_auto_motion(f"trajectory goal request failed: {error}")
            return
        if not goal_handle.accepted:
            self._fail_auto_motion("trajectory controller rejected the goal")
            return
        self._auto_goal_handle = goal_handle
        self._auto_state = "moving"
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._auto_goal_result_callback)

    def _auto_goal_result_callback(self, future) -> None:
        self._auto_goal_handle = None
        try:
            wrapped_result = future.result()
        except Exception as error:
            self._fail_auto_motion(f"trajectory execution failed: {error}")
            return
        result = wrapped_result.result
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            detail = result.error_string or f"error code {result.error_code}"
            self._fail_auto_motion(f"trajectory did not succeed: {detail}")
            return
        self._auto_state = "waiting_sample"
        self._capture_requested = True
        self._auto_sample_deadline_ns = (
            self.get_clock().now().nanoseconds + self._auto_sample_timeout_ns
        )
        self.get_logger().info(
            f"Auto waypoint {self._auto_waypoint_index + 1} reached; waiting for one "
            "valid settled RGB/FK sample."
        )

    def _fail_auto_motion(self, reason: str) -> None:
        self._capture_requested = False
        self._auto_state = "failed"
        self.get_logger().error(
            f"Auto calibration stopped at waypoint {self._auto_waypoint_index + 1}: {reason}."
        )

    def cancel_auto_motion(self):
        """Cancel an active automatic trajectory, returning its cancel future."""
        self._capture_requested = False
        self._auto_state = "cancelled"
        if self._auto_goal_handle is None:
            return None
        self.get_logger().warn("Cancelling the active automatic calibration trajectory.")
        return self._auto_goal_handle.cancel_goal_async()

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if message.width == 0 or message.height == 0:
            return
        self._camera_matrix = np.asarray(message.k, dtype=float).reshape(3, 3)
        self._distortion = np.asarray(message.d, dtype=float)
        if not self._camera_optical_frame:
            self._camera_optical_frame = _clean_frame(message.header.frame_id)

    def _detect_tag(self, gray_image: np.ndarray):
        matches = [
            detection
            for detection in self._detector.detect(gray_image)
            if detection.identifier == self._tag_id
        ]
        if len(matches) != 1:
            return None
        return matches[0].corners

    def _estimate_tag_pose(self, image_corners: np.ndarray):
        half = self._tag_size * 0.5
        object_corners = np.asarray(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_corners,
            image_corners,
            self._camera_matrix,
            self._distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None, math.inf
        if float(translation_vector[2]) <= 0.0:
            return None, math.inf
        projected, _ = cv2.projectPoints(
            object_corners,
            rotation_vector,
            translation_vector,
            self._camera_matrix,
            self._distortion,
        )
        reprojection_error = float(
            np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_corners) ** 2, axis=1)))
        )
        rotation, _ = cv2.Rodrigues(rotation_vector)
        return make_transform(rotation, translation_vector.reshape(3)), reprojection_error

    def _image_callback(self, message: Image) -> None:
        if self._camera_matrix is None:
            self._notice("intrinsics", "Waiting for CameraInfo before detecting the tag.")
            return
        message_frame = _clean_frame(message.header.frame_id)
        if not self._camera_optical_frame:
            self._camera_optical_frame = message_frame
        if message_frame and message_frame != self._camera_optical_frame:
            self._notice(
                "frame",
                f"Image frame is {message_frame}, but camera_optical_frame is "
                f"{self._camera_optical_frame}; ignoring it.",
            )
            return

        try:
            color_image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:  # cv_bridge exception types differ by ROS release
            self._notice("cv_bridge", f"Cannot decode image: {error}")
            return
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        image_corners = self._detect_tag(gray_image)
        if image_corners is None:
            self._notice(
                "tag_visibility",
                f"No single tag {self._tag_id} detection; keep its complete border visible.",
                period_s=2.0,
            )
            return
        camera_to_tag, reprojection_error = self._estimate_tag_pose(image_corners)
        if camera_to_tag is None:
            self._notice("pnp", "Tag detected, but OpenCV solvePnP failed.", period_s=2.0)
            return
        if reprojection_error > self._max_reprojection_error:
            self._notice(
                "reprojection",
                f"Tag pose rejected: {reprojection_error:.2f} px reprojection error "
                f"exceeds {self._max_reprojection_error:.2f} px.",
                period_s=2.0,
            )
            return
        self._try_capture(message, camera_to_tag, reprojection_error)

    def _try_capture(
        self, image_message: Image, camera_to_tag: np.ndarray, reprojection_error: float
    ) -> None:
        if self._calibration_complete:
            return
        if self._capture_mode in ("auto", "triggered") and not self._capture_requested:
            return
        stamp = Time.from_msg(image_message.header.stamp)
        try:
            transform = self._tf_buffer.lookup_transform(
                self._world_frame, self._hand_frame, stamp, timeout=self._tf_timeout
            )
        except TransformException as error:
            self._notice(
                "robot_tf",
                f"No {self._world_frame} -> {self._hand_frame} TF at the image time: {error}",
            )
            return
        world_to_hand = _transform_from_message(transform.transform)
        stamp_ns = stamp.nanoseconds

        if self._sample_stamps_ns and stamp_ns - self._sample_stamps_ns[-1] < self._min_interval_ns:
            return
        if self._last_sample_hand is not None:
            delta = invert_transform(self._last_sample_hand) @ world_to_hand
            translation_delta = float(np.linalg.norm(delta[:3, 3]))
            rotation_delta = rotation_angle_deg(delta[:3, :3])
            if translation_delta < self._min_translation and rotation_delta < self._min_rotation:
                self._notice(
                    "pose_difference",
                    "Valid tag/FK observation, but the pose is too similar to the last "
                    "sample; move at least 2.5 cm or rotate at least 8 degrees.",
                    period_s=2.0,
                )
                return

        self._world_to_hand_samples.append(world_to_hand)
        self._camera_to_tag_samples.append(camera_to_tag)
        self._sample_stamps_ns.append(stamp_ns)
        self._last_sample_hand = world_to_hand
        self._capture_requested = False
        count = len(self._world_to_hand_samples)
        self.get_logger().info(
            f"Accepted valid sample {count}/{self._minimum_samples} from timestamped "
            f"{self._world_frame} -> {self._hand_frame} FK "
            f"(tag reprojection error {reprojection_error:.2f} px)."
        )
        if self._capture_mode == "auto":
            self._auto_waypoint_index += 1
            if self._auto_waypoint_index < len(AUTO_WAYPOINTS):
                self._auto_state = "waiting_controller"
            else:
                self._auto_state = "finished"
        if count >= self._minimum_samples:
            self.get_logger().info(
                "Minimum sample count reached; solving the camera calibration now."
            )
            solve_response = Trigger.Response()
            self._solve_callback(None, solve_response)
            if solve_response.success:
                self._calibration_complete = True
                self.get_logger().info(
                    "Automatic calibration is complete. Call /camera_calibration/reset "
                    "to collect a new calibration."
                )

    def _capture_callback(self, request, response):
        del request
        if self._capture_mode == "auto":
            response.success = False
            response.message = "Capture is sequenced automatically in auto mode."
            return response
        self._capture_requested = True
        response.success = True
        response.message = "The next valid, sufficiently different tag observation will be captured."
        return response

    def _reset_callback(self, request, response):
        del request
        if self._capture_mode == "auto" and self._auto_state in (
            "sending_goal",
            "moving",
        ):
            response.success = False
            response.message = "Cannot reset while an automatic trajectory is moving."
            return response
        count = len(self._world_to_hand_samples)
        self._world_to_hand_samples.clear()
        self._camera_to_tag_samples.clear()
        self._sample_stamps_ns.clear()
        self._last_sample_hand = None
        self._capture_requested = False
        self._calibration_complete = False
        if self._capture_mode == "auto":
            self._auto_waypoint_index = 0
            self._auto_state = "waiting_controller"
            self._auto_sample_deadline_ns = 0
        response.success = True
        response.message = f"Cleared {count} samples."
        return response

    def _solve_callback(self, request, response):
        del request
        try:
            result, retained = calibrate_fixed_tag_eye_to_hand(
                self._world_to_hand_samples,
                self._camera_to_tag_samples,
                self._hand_to_tag,
                minimum_samples=self._minimum_samples,
            )
            if self._camera_mount_frame == self._camera_optical_frame:
                mount_to_optical = np.eye(4)
            else:
                if not self._camera_optical_frame:
                    raise CalibrationError("camera optical frame is not known yet")
                try:
                    mount_to_optical_message = self._tf_buffer.lookup_transform(
                        self._camera_mount_frame,
                        self._camera_optical_frame,
                        Time(),
                        timeout=Duration(seconds=1.0),
                    )
                except TransformException as error:
                    raise CalibrationError(
                        f"cannot look up camera-internal TF {self._camera_mount_frame} -> "
                        f"{self._camera_optical_frame}: {error}"
                    ) from error
                mount_to_optical = _transform_from_message(mount_to_optical_message.transform)

            world_to_mount = result.world_to_camera @ invert_transform(mount_to_optical)
            retained_count = int(np.count_nonzero(retained))
            calibration = self._calibration_log(result, world_to_mount, retained_count)
        except (CalibrationError, ValueError) as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(f"Calibration failed: {error}")
            return response

        rejected = len(retained) - int(np.count_nonzero(retained))
        response.success = True
        response.message = (
            f"Calibrated {self._world_frame} -> {self._camera_mount_frame}; "
            f"{int(np.count_nonzero(retained))} samples used, {rejected} rejected, "
            f"RMSE {result.translation_rmse_m * 1000.0:.1f} mm / "
            f"{result.rotation_rmse_deg:.2f} deg. Result written to the node log."
        )
        self.get_logger().info(response.message)
        self.get_logger().info(
            "CALIBRATION_RESULT " + json.dumps(calibration, sort_keys=True)
        )
        return response

    def _calibration_log(
        self, result, world_to_mount: np.ndarray, retained_count: int
    ) -> dict:
        def transform_dict(transform: np.ndarray) -> dict:
            quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
            return {
                "translation": {
                    "x": float(transform[0, 3]),
                    "y": float(transform[1, 3]),
                    "z": float(transform[2, 3]),
                },
                "rotation_xyzw": {
                    "x": float(quaternion[0]),
                    "y": float(quaternion[1]),
                    "z": float(quaternion[2]),
                    "w": float(quaternion[3]),
                },
                "matrix_4x4": transform.tolist(),
            }

        return {
            "schema_version": 1,
            "parent_frame": self._world_frame,
            "child_frame": self._camera_mount_frame,
            "camera_optical_frame": self._camera_optical_frame,
            "camera_intrinsics": {
                "matrix_3x3": self._camera_matrix.tolist(),
                "distortion": self._distortion.tolist(),
            },
            "transform": transform_dict(world_to_mount),
            "fixed_carrier_to_tag": {
                "parent_frame": self._hand_frame,
                "child_frame": f"apriltag_{self._tag_id}",
                **transform_dict(result.hand_to_tag),
            },
            "quality": {
                "samples_collected": len(self._world_to_hand_samples),
                "samples_used": retained_count,
                "translation_rmse_m": result.translation_rmse_m,
                "rotation_rmse_deg": result.rotation_rmse_deg,
            },
            "tag": {"id": self._tag_id, "size_m": self._tag_size},
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = CameraCalibrationNode()
        # TF may arrive a few milliseconds after its matching image.  A second
        # executor thread lets TransformListener fill the buffer while the
        # image callback waits for the timestamped transform.
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None and executor is not None:
            cancel_future = node.cancel_auto_motion()
            if cancel_future is not None:
                try:
                    executor.spin_until_future_complete(cancel_future, timeout_sec=1.0)
                    if not cancel_future.done():
                        node.get_logger().error(
                            "Trajectory cancellation was not confirmed; use the emergency "
                            "stop if the robot is still moving."
                        )
                except Exception:
                    node.get_logger().error(
                        "Could not confirm trajectory cancellation; use the emergency stop "
                        "if the robot is still moving."
                    )
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
