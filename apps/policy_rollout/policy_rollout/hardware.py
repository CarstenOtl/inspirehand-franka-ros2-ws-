"""Real FR3/Inspire/RealSense adapter for the distilled threading policy.

The non-realtime policy process publishes bounded Cartesian setpoints. The
``CartesianTrajectoryReplayController`` owns the FR3 effort interfaces and
executes those setpoints in its 1 kHz loop. The RH56 remains on the same
independent ``/inspire_hand/command`` link used by trajectory replay.

ROS imports are deliberately lazy so unit tests continue to work on machines
without a ROS installation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from . import forge_osc as fo
from utils.camera_calibration import CameraCalibrationProfile


ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)
SUPPORT_FINGER_TARGETS = np.array([1.333, 1.333, 1.333], dtype=float)
POLICY_HAND_JOINTS = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
)


def physical_hand_state_to_policy(
    hand_position, hand_velocity
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical RH56 feedback into the checkpoint's logical coordinates.

    The driver contracts thumb-yaw commands into the physically usable final
    75 percent of travel. Its joint-state topic correctly reports that physical
    pose for TF, but the checkpoint was trained on the pre-overlay logical
    coordinate. Only the policy scalar and its derivative are inverted here;
    the physical TF tree remains untouched.
    """

    from inspire_hand_driver import command_overlays
    from inspire_hand_driver import kinematics as kin

    positions = np.asarray(hand_position, dtype=float).reshape(-1)
    velocities = np.asarray(hand_velocity, dtype=float).reshape(-1)
    if positions.shape != (len(HAND_JOINTS),) or velocities.shape != (
        len(HAND_JOINTS),
    ):
        raise ValueError("physical hand state must contain all six driven joints")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("physical hand state contains non-finite values")
    columns = {name: index for index, name in enumerate(HAND_JOINTS)}
    indices = [columns[name] for name in POLICY_HAND_JOINTS]
    q_policy = positions[indices].copy()
    dq_policy = velocities[indices].copy()

    thumb_dof = kin.dof_index(command_overlays.THUMB_ABDUCTION_JOINT)
    physical_ratio = kin.rad_to_open_ratio(thumb_dof, q_policy[0])
    logical_ratio = command_overlays.invert_open_ratio_overlay(
        thumb_dof, physical_ratio
    )
    if not -1.0e-3 <= logical_ratio <= 1.0 + 1.0e-3:
        raise ValueError(
            "physical thumb-yaw feedback is outside the overlaid policy range"
        )
    q_policy[0] = kin.open_ratio_to_rad(thumb_dof, logical_ratio)
    travel_scale = 1.0 - command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    dq_policy[0] /= travel_scale
    return q_policy, dq_policy


@dataclass(frozen=True)
class HardwareReadiness:
    ready: bool
    blockers: tuple[str, ...]


def assess_hardware_readiness(
    calibration: CameraCalibrationProfile,
) -> HardwareReadiness:
    """Static readiness only; graph/controller checks happen at run time."""

    blockers = tuple(calibration.hardware_blockers())
    return HardwareReadiness(ready=not blockers, blockers=blockers)


def image_message_to_numpy(message) -> tuple[np.ndarray, str]:
    """Decode the RealSense Image encodings used here without cv_bridge."""

    encoding = str(message.encoding).lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    raw = memoryview(message.data)
    if encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        channels = 4 if "a" in encoding else 3
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)[..., :3]
        if encoding.startswith("bgr"):
            image = image[..., ::-1]
        return np.ascontiguousarray(image), "rgb"
    if encoding in {"16uc1", "mono16"}:
        rows = np.frombuffer(raw, dtype=np.uint16).reshape(height, step // 2)
        return np.ascontiguousarray(rows[:, :width]), "millimetres"
    if encoding == "32fc1":
        rows = np.frombuffer(raw, dtype=np.float32).reshape(height, step // 4)
        return np.ascontiguousarray(rows[:, :width]), "metres"
    raise ValueError(f"unsupported RealSense image encoding {message.encoding!r}")


def _stamp_seconds(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)


def assert_policy_camera_frames(rgb, depth, camera_info, expected_frame_id: str) -> None:
    """Require every RGB-D input to use the calibrated colour pixel frame."""

    frames = {
        "RGB": str(rgb.header.frame_id),
        "depth": str(depth.header.frame_id),
        "CameraInfo": str(camera_info.header.frame_id),
    }
    mismatches = [
        f"{label}={frame_id!r}"
        for label, frame_id in frames.items()
        if frame_id != expected_frame_id
    ]
    if mismatches:
        raise RuntimeError(
            "policy RGB-D is not registered to the calibrated colour frame "
            f"{expected_frame_id!r}: " + ", ".join(mismatches)
        )


def policy_tool_transform(config) -> np.ndarray:
    """Build the configured flange-to-tool transform through the kinematics API."""

    from franka_trajectory_replay import kinematics

    return kinematics.tool_transform(
        config["tcp"]["offset_xyz"], config["tcp"]["offset_rpy"]
    )


def limit_cartesian_step(
    target_position,
    target_quaternion,
    reference_position,
    reference_quaternion,
    *,
    max_position_step_m: float,
    max_orientation_step_rad: float,
    margin: float = 0.98,
):
    """Contract a target to the controller's norm-based policy-step guard."""

    if not 0.0 < margin < 1.0:
        raise ValueError("Cartesian step margin must be in (0, 1)")
    if max_position_step_m <= 0.0 or max_orientation_step_rad <= 0.0:
        raise ValueError("Cartesian step limits must be positive")
    position = np.asarray(target_position, dtype=float).copy()
    reference_p = np.asarray(reference_position, dtype=float)
    if (
        position.shape != (3,)
        or reference_p.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(reference_p).all()
    ):
        raise ValueError("Cartesian positions must contain three finite values")
    delta = position - reference_p
    distance = float(np.linalg.norm(delta))
    position_limit = float(max_position_step_m) * margin
    position_limited = distance > position_limit
    if position_limited:
        position = reference_p + delta * (position_limit / distance)

    quaternion = np.asarray(target_quaternion, dtype=float).copy()
    reference_q = np.asarray(reference_quaternion, dtype=float).copy()
    quaternion_norm = float(np.linalg.norm(quaternion))
    reference_norm = float(np.linalg.norm(reference_q))
    if (
        quaternion.shape != (4,)
        or reference_q.shape != (4,)
        or not np.isfinite(quaternion).all()
        or not np.isfinite(reference_q).all()
        or quaternion_norm < 1.0e-8
        or reference_norm < 1.0e-8
    ):
        raise ValueError("Cartesian quaternions must contain four finite values")
    quaternion /= quaternion_norm
    reference_q /= reference_norm
    dot = float(np.dot(reference_q, quaternion))
    if dot < 0.0:
        quaternion = -quaternion
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * math.acos(dot)
    orientation_limit = float(max_orientation_step_rad) * margin
    orientation_limited = angle > orientation_limit
    if orientation_limited:
        fraction = orientation_limit / angle
        half_angle = math.acos(dot)
        if half_angle < 1.0e-8:
            quaternion = reference_q
        else:
            quaternion = (
                math.sin((1.0 - fraction) * half_angle) * reference_q
                + math.sin(fraction * half_angle) * quaternion
            ) / math.sin(half_angle)
            quaternion /= np.linalg.norm(quaternion)
    return position, quaternion, position_limited, orientation_limited


class RgbdFrameSynchronizer:
    """Keep recent camera messages and return the closest timestamped pair."""

    def __init__(self, max_skew_s: float, queue_size: int = 8) -> None:
        if max_skew_s <= 0.0 or queue_size < 1:
            raise ValueError("RGB-D synchronization settings must be positive")
        self.max_skew_s = float(max_skew_s)
        self.queue_size = int(queue_size)
        self._queues = {"rgb": [], "depth": []}

    def add(self, stream: str, message, receive_time_s: float) -> None:
        if stream not in self._queues:
            raise ValueError(f"unknown camera stream {stream!r}")
        queue = self._queues[stream]
        queue.append((_stamp_seconds(message), float(receive_time_s), message))
        del queue[:-self.queue_size]

    def latest_pair(self):
        candidates = []
        for rgb in self._queues["rgb"]:
            for depth in self._queues["depth"]:
                skew = abs(rgb[0] - depth[0])
                if skew <= self.max_skew_s:
                    candidates.append((skew, min(rgb[0], depth[0]), rgb, depth))
        if not candidates:
            return None
        # Prefer the newest complete pair. Use the smaller skew as the
        # tie-breaker when two candidates share the same older timestamp; that
        # avoids selecting a new RGB frame with the previous depth frame while
        # its exact partner is still between callbacks.
        _skew, _stamp, rgb, depth = max(
            candidates, key=lambda item: (item[1], -item[0])
        )
        return rgb[2], depth[2], rgb[1], depth[1]


class TrainingFrameAdapter:
    """Convert between the policy's training world and the physical FR3 base."""

    def __init__(self) -> None:
        self.world_from_base_position = fo.ROBOT_BASE_POSITION.copy()
        self.world_from_base_quaternion = fo.quat_from_euler_xyz(0.0, 0.0, math.pi)

    def pose_base_to_world(self, position, quaternion):
        q = self.world_from_base_quaternion
        return (
            self.world_from_base_position
            + fo.quat_rotate(q, np.asarray(position, dtype=float)),
            fo.quat_mul(q, np.asarray(quaternion, dtype=float)),
        )

    def pose_world_to_base(self, position, quaternion):
        q_inv = fo.quat_conjugate(self.world_from_base_quaternion)
        return (
            fo.quat_rotate(
                q_inv,
                np.asarray(position, dtype=float) - self.world_from_base_position,
            ),
            fo.quat_mul(q_inv, np.asarray(quaternion, dtype=float)),
        )

    def controller_target(
        self,
        filtered_action,
        *,
        grasp_position_base,
        grasp_quaternion_base,
        controlled_position_base,
        controlled_quaternion_base,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode Forge's grasp target into the controller's fixed tool frame.

        Forge controls the live midpoint between the thumb and index tips.  The
        Cartesian controller instead differentiates its impedance law at a
        fixed flange-relative tool transform.  Preserve the measured transform
        from the grasp frame to that controlled frame when retargeting; sending
        the grasp origin directly as the tool origin creates a centimetre-scale
        translation error whenever those frames do not coincide.
        """

        grasp_world_position, grasp_world_quaternion = self.pose_base_to_world(
            grasp_position_base, grasp_quaternion_base
        )
        grasp = fo.GraspFrameState(
            pos=grasp_world_position,
            quat=grasp_world_quaternion,
            linvel=np.zeros(3),
            angvel=np.zeros(3),
            jacobian=np.zeros((6, 7)),
        )
        target = fo.decode_action_target(filtered_action, grasp)
        target_position_base, target_grasp_quaternion_base = self.pose_world_to_base(
            target.pos, target.quat
        )
        return retarget_grasp_pose_to_controlled_pose(
            grasp_position_base=grasp_position_base,
            grasp_quaternion_base=grasp_quaternion_base,
            controlled_position_base=controlled_position_base,
            controlled_quaternion_base=controlled_quaternion_base,
            target_grasp_position_base=target_position_base,
            target_grasp_quaternion_base=target_grasp_quaternion_base,
        )


def retarget_grasp_pose_to_controlled_pose(
    *,
    grasp_position_base,
    grasp_quaternion_base,
    controlled_position_base,
    controlled_quaternion_base,
    target_grasp_position_base,
    target_grasp_quaternion_base,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the live grasp-to-controlled transform at a new grasp pose."""

    positions = tuple(
        np.asarray(value, dtype=float)
        for value in (
            grasp_position_base,
            controlled_position_base,
            target_grasp_position_base,
        )
    )
    quaternions = [
        np.asarray(value, dtype=float).copy()
        for value in (
            grasp_quaternion_base,
            controlled_quaternion_base,
            target_grasp_quaternion_base,
        )
    ]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in positions):
        raise ValueError("grasp/control positions must contain three finite values")
    if any(
        value.shape != (4,)
        or not np.isfinite(value).all()
        or np.linalg.norm(value) < 1.0e-8
        for value in quaternions
    ):
        raise ValueError("grasp/control quaternions must contain four finite values")
    for value in quaternions:
        value /= np.linalg.norm(value)

    grasp_position, controlled_position, target_grasp_position = positions
    grasp_quaternion, controlled_quaternion, target_grasp_quaternion = quaternions
    base_to_grasp_quaternion = fo.quat_conjugate(grasp_quaternion)
    controlled_in_grasp_position = fo.quat_rotate(
        base_to_grasp_quaternion, controlled_position - grasp_position
    )
    controlled_in_grasp_quaternion = fo.quat_mul(
        base_to_grasp_quaternion, controlled_quaternion
    )
    target_controlled_position = target_grasp_position + fo.quat_rotate(
        target_grasp_quaternion, controlled_in_grasp_position
    )
    target_controlled_quaternion = fo.quat_mul(
        target_grasp_quaternion, controlled_in_grasp_quaternion
    )
    target_controlled_quaternion /= np.linalg.norm(target_controlled_quaternion)
    return target_controlled_position, target_controlled_quaternion


class GripCycleCoordinator:
    """Physical release/return phase state from measured grasp rotation.

    The physical workcell has no nut-angle topic. While the nut is grasped its
    directional turn is observable as grasp-frame yaw, so that signal drives
    the same 55-degree transition threshold as the simulator. It is recorded as
    a proxy and is never promoted to task-success evidence.
    """

    def __init__(self, *, rate_hz, max_cycles, reset_position, reset_quaternion, reset_hand):
        self.rate_hz = float(rate_hz)
        self.max_cycles = int(max_cycles)
        self.reset_position = np.asarray(reset_position, dtype=float).copy()
        self.reset_quaternion = np.asarray(reset_quaternion, dtype=float).copy()
        self.reset_hand = np.asarray(reset_hand, dtype=float).copy()
        self.active = False
        self.phase_index = -1
        self.phase_steps = 0
        self.wait_steps = 0
        self.completed_cycles = 0
        self.failed = False
        self._previous_yaw = 0.0
        self._unwrapped_yaw = 0.0
        self._cycle_yaw_origin = 0.0

    def process_phase(self) -> str:
        if not self.active:
            return "policy"
        return "return_to_reset" if self.phase_index == 4 else "follow_waypoints"

    def turn_progress_rad(self, quaternion) -> float:
        relative = fo.matrix_from_quat(fo.quat_conjugate(self.reset_quaternion)) @ fo.matrix_from_quat(
            quaternion
        )
        yaw = math.atan2(relative[1, 0], relative[0, 0])
        delta = math.atan2(math.sin(yaw - self._previous_yaw), math.cos(yaw - self._previous_yaw))
        self._unwrapped_yaw += delta
        self._previous_yaw = yaw
        return -1.0 * (self._unwrapped_yaw - self._cycle_yaw_origin)

    def _returned(self, position, quaternion, hand) -> bool:
        position_error = np.linalg.norm(np.asarray(position) - self.reset_position)
        alignment = min(1.0, abs(float(np.dot(quaternion, self.reset_quaternion))))
        orientation_error = math.degrees(2.0 * math.acos(alignment))
        hand_error = np.max(np.abs(np.asarray(hand) - self.reset_hand))
        return position_error <= 0.005 and orientation_error <= 5.0 and hand_error <= 0.15

    def update(self, *, position, quaternion, hand) -> tuple[str | None, float]:
        progress = self.turn_progress_rad(quaternion)
        event = None
        if self.active:
            self.phase_steps += 1
            duration = math.ceil((0.7 if self.phase_index < 4 else 1.0) * self.rate_hz)
            if self.phase_steps >= duration:
                if self.phase_index < 4:
                    self.phase_index += 1
                    self.phase_steps = 0
                    self.wait_steps = 0
                elif self._returned(position, quaternion, hand):
                    self.completed_cycles += 1
                    self.active = False
                    self.phase_index = -1
                    self.phase_steps = 0
                    self.wait_steps = 0
                    self._cycle_yaw_origin = self._unwrapped_yaw
                    event = "cycle_completed"
                else:
                    self.wait_steps += 1
                    if self.wait_steps >= math.ceil(16.0 * self.rate_hz):
                        self.active = False
                        self.failed = True
                        event = "return_failed"
        if (
            event is None
            and not self.active
            and not self.failed
            and self.completed_cycles < self.max_cycles
            and progress >= math.radians(55.0)
        ):
            self.active = True
            self.phase_index = 0
            self.phase_steps = 0
            self.wait_steps = 0
            event = "release_started"
        return event, progress


@dataclass(frozen=True)
class LiveSample:
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    hand_position: np.ndarray
    hand_velocity: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    depth_units: str
    controller_target_position: np.ndarray
    controller_target_quaternion: np.ndarray
    controller_measured_position: np.ndarray
    controller_measured_quaternion: np.ndarray
    sample_time_s: float


def _hardware_node_class():
    """Build the Node subclass only after ROS is known to be available."""

    from rclpy.node import Node

    class HardwarePolicyNode(Node):
        def __init__(
            self,
            calibration,
            config,
            *,
            hand_command_topic,
            hand_state_topic,
            max_state_age_s,
            max_frame_skew_s,
        ):
            from control_msgs.msg import JointTrajectoryControllerState
            from diagnostic_msgs.msg import DiagnosticArray
            from franka_trajectory_replay_msgs.msg import CartesianGoto, CartesianReplayState
            from realsense2_camera_msgs.msg import RGBD
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Empty
            import tf2_ros

            super().__init__("fr3_policy_rollout")
            self.calibration = calibration
            self.config = config
            self.max_state_age_s = float(max_state_age_s)
            self.max_frame_skew_s = float(max_frame_skew_s)
            self._rgbd = RgbdFrameSynchronizer(self.max_frame_skew_s)
            controller = "/" + config["cartesian"]["controller_name"].strip("/")
            self.policy_publisher = self.create_publisher(
                CartesianGoto, controller + "/policy_command", 1
            )
            self.abort_publisher = self.create_publisher(
                Empty, controller + "/abort", 1
            )
            self.hand_publisher = self.create_publisher(
                JointState, hand_command_topic, 10
            )
            self._lock = threading.Lock()
            self._arm = self._hand = None
            self._camera_info = self._cartesian = self._controller_status = None
            self._receive_time = {}
            self.create_subscription(
                JointTrajectoryControllerState,
                controller + "/controller_state",
                lambda message: self._put("arm", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CartesianReplayState,
                controller + "/cartesian_state",
                lambda message: self._put("cartesian", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                DiagnosticArray,
                controller + "/status",
                self._put_controller_status,
                10,
            )
            self.create_subscription(
                JointState,
                hand_state_topic,
                lambda message: self._put("hand", message),
                qos_profile_sensor_data,
            )
            self.rgbd_subscription = self.create_subscription(
                RGBD,
                calibration.rgbd_topic,
                self._put_rgbd,
                qos_profile_sensor_data,
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self._camera_checked = False

        def _put(self, name, message):
            with self._lock:
                setattr(self, "_" + name, message)
                self._receive_time[name] = time.monotonic()

        def _put_rgbd(self, message):
            received = time.monotonic()
            with self._lock:
                self._rgbd.add("rgb", message.rgb, received)
                self._rgbd.add("depth", message.depth, received)
                self._camera_info = message.rgb_camera_info
                self._receive_time["camera_info"] = received

        def _put_controller_status(self, message):
            if not message.status:
                return
            values = {item.key: item.value for item in message.status[0].values}
            with self._lock:
                self._controller_status = values
                self._receive_time["status"] = time.monotonic()

        def ready(self) -> bool:
            with self._lock:
                return (
                    self._rgbd.latest_pair() is not None
                    and all(
                        value is not None
                        for value in (
                            self._arm,
                            self._cartesian,
                            self._hand,
                            self._camera_info,
                            self._controller_status,
                        )
                    )
                )

        def wait_ready(self, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self.ready() and self.policy_publisher.get_subscription_count() > 0:
                    return
                time.sleep(0.02)
            if self.rgbd_subscription.get_publisher_count() == 0:
                raise TimeoutError(
                    f"no composite RGB-D publisher on {self.calibration.rgbd_topic}; "
                    "start RealSense with enable_rgbd:=true, enable_sync:=true, "
                    "and align_depth.enable:=true"
                )
            raise TimeoutError("policy inputs/controller subscription did not become ready")

        def sample(self) -> LiveSample:
            with self._lock:
                pair = self._rgbd.latest_pair()
                messages = {
                    name: getattr(self, "_" + name)
                    for name in (
                        "arm",
                        "cartesian",
                        "hand",
                        "camera_info",
                    )
                }
                receive = dict(self._receive_time)
                if pair is not None:
                    messages["rgb"], messages["depth"] = pair[:2]
                    receive["rgb"], receive["depth"] = pair[2:]
                else:
                    messages["rgb"] = messages["depth"] = None
            now = time.monotonic()
            missing = [name for name, value in messages.items() if value is None]
            if missing:
                raise RuntimeError("missing live inputs: " + ", ".join(missing))
            stale = [
                name
                for name in ("arm", "cartesian", "hand", "rgb", "depth")
                if now - receive[name] > self.max_state_age_s
            ]
            if stale:
                raise RuntimeError("stale live inputs: " + ", ".join(stale))
            rgb_stamp = _stamp_seconds(messages["rgb"])
            depth_stamp = _stamp_seconds(messages["depth"])
            if abs(rgb_stamp - depth_stamp) > self.max_frame_skew_s:
                raise RuntimeError(
                    f"RGB/depth timestamp skew {abs(rgb_stamp-depth_stamp):.4f} s "
                    f"exceeds {self.max_frame_skew_s:.4f} s"
                )
            if not self._camera_checked:
                info = messages["camera_info"]
                assert_policy_camera_frames(
                    messages["rgb"], messages["depth"], info, self.calibration.frame_id
                )
                self.calibration.assert_live_camera_info(info.width, info.height, info.k)
                self._camera_checked = True

            arm = messages["arm"]
            arm_columns = {name: index for index, name in enumerate(arm.joint_names)}
            if any(name not in arm_columns for name in ARM_JOINTS):
                raise RuntimeError("controller state does not contain all FR3 joints")
            indices = [arm_columns[name] for name in ARM_JOINTS]
            q_arm = np.asarray(arm.feedback.positions, dtype=float)[indices]
            dq_arm = np.asarray(arm.feedback.velocities, dtype=float)[indices]
            hand = messages["hand"]
            hand_positions = dict(zip(hand.name, hand.position))
            hand_velocities = dict(zip(hand.name, hand.velocity))
            if any(name not in hand_positions for name in HAND_JOINTS):
                raise RuntimeError("hand state does not contain all driven RH56 joints")
            q_hand = np.array([hand_positions[name] for name in HAND_JOINTS])
            dq_hand = np.array([hand_velocities.get(name, 0.0) for name in HAND_JOINTS])
            rgb, _ = image_message_to_numpy(messages["rgb"])
            depth, depth_units = image_message_to_numpy(messages["depth"])
            controller_pose = messages["cartesian"].target
            controller_target_position = np.array(
                [
                    controller_pose.position.x,
                    controller_pose.position.y,
                    controller_pose.position.z,
                ],
                dtype=float,
            )
            controller_target_quaternion = np.array(
                [
                    controller_pose.orientation.w,
                    controller_pose.orientation.x,
                    controller_pose.orientation.y,
                    controller_pose.orientation.z,
                ],
                dtype=float,
            )
            controller_measured_pose = messages["cartesian"].measured
            controller_measured_position = np.array(
                [
                    controller_measured_pose.position.x,
                    controller_measured_pose.position.y,
                    controller_measured_pose.position.z,
                ],
                dtype=float,
            )
            controller_measured_quaternion = np.array(
                [
                    controller_measured_pose.orientation.w,
                    controller_measured_pose.orientation.x,
                    controller_measured_pose.orientation.y,
                    controller_measured_pose.orientation.z,
                ],
                dtype=float,
            )
            return LiveSample(
                arm_position=q_arm,
                arm_velocity=dq_arm,
                hand_position=q_hand,
                hand_velocity=dq_hand,
                rgb=rgb,
                depth=depth,
                depth_units=depth_units,
                controller_target_position=controller_target_position,
                controller_target_quaternion=controller_target_quaternion,
                controller_measured_position=controller_measured_position,
                controller_measured_quaternion=controller_measured_quaternion,
                sample_time_s=now,
            )

        def wait_for_fresh_sample(self, timeout_s: float) -> LiveSample:
            """Wait through transient callback gaps without hiding contract errors."""

            deadline = time.monotonic() + timeout_s
            last_error = None
            while time.monotonic() < deadline:
                try:
                    return self.sample()
                except RuntimeError as exc:
                    message = str(exc)
                    if not message.startswith(("missing live inputs:", "stale live inputs:")):
                        raise
                    last_error = exc
                    time.sleep(0.01)
            raise TimeoutError(
                "fresh policy inputs did not arrive"
                + ("" if last_error is None else f": {last_error}")
            )

        def controller_status(self) -> dict[str, str]:
            with self._lock:
                status = (
                    None
                    if self._controller_status is None
                    else dict(self._controller_status)
                )
                received = self._receive_time.get("status", 0.0)
            if status is None or time.monotonic() - received > 1.0:
                raise RuntimeError(
                    "Cartesian replay controller feedback stopped; check the hardware log"
                )
            return status

        @staticmethod
        def policy_hand_state(sample: LiveSample) -> tuple[np.ndarray, np.ndarray]:
            return physical_hand_state_to_policy(
                sample.hand_position, sample.hand_velocity
            )

        def grasp_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            from rclpy.time import Time

            def lookup(child):
                transform = self.tf_buffer.lookup_transform("fr3_link0", child, Time())
                t = transform.transform.translation
                r = transform.transform.rotation
                return np.array([t.x, t.y, t.z]), np.array([r.w, r.x, r.y, r.z])

            thumb, _ = lookup("thumb_tip")
            index, _ = lookup("index_tip")
            _flange_position, flange_quaternion = lookup("fr3_link8")
            direction = fo.quat_rotate(
                flange_quaternion, np.array([0.0, 0.0, -1.0])
            )
            grasp_position, grasp_rotation = fo.hand_grasp_frame(
                thumb, index, direction
            )
            return grasp_position, fo.quat_from_matrix(grasp_rotation), flange_quaternion

        def wait_for_grasp_pose(self, timeout_s: float):
            deadline = time.monotonic() + timeout_s
            last_error = None
            while time.monotonic() < deadline:
                try:
                    return self.grasp_pose()
                except Exception as exc:  # tf2 raises several lookup exception types
                    last_error = exc
                    time.sleep(0.02)
            raise TimeoutError(
                "arm-to-hand TF did not become ready"
                + ("" if last_error is None else f": {last_error}")
            )

        def publish_policy_target(self, position, quaternion_wxyz, nullspace):
            from franka_trajectory_replay_msgs.msg import CartesianGoto

            message = CartesianGoto()
            message.pose.position.x = float(position[0])
            message.pose.position.y = float(position[1])
            message.pose.position.z = float(position[2])
            q = np.asarray(quaternion_wxyz, dtype=float)
            message.pose.orientation.w = float(q[0])
            message.pose.orientation.x = float(q[1])
            message.pose.orientation.y = float(q[2])
            message.pose.orientation.z = float(q[3])
            message.nullspace_positions = [float(value) for value in nullspace]
            message.duration = 0.0
            self.policy_publisher.publish(message)

        def publish_hand_target(self, pinch):
            from inspire_hand_driver import kinematics as kin
            from sensor_msgs.msg import JointState

            radians = np.concatenate(
                (SUPPORT_FINGER_TARGETS, [pinch[2], pinch[1], pinch[0]])
            )
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = list(HAND_JOINTS)
            message.position = [
                kin.rad_to_open_ratio(kin.dof_index(name), value)
                for name, value in zip(HAND_JOINTS, radians)
            ]
            self.hand_publisher.publish(message)

        def abort(self):
            from std_msgs.msg import Empty

            self.abort_publisher.publish(Empty())

    return HardwarePolicyNode


def run_hardware_rollout(runner, calibration, args) -> dict:
    """Home through trajectory replay, switch controllers, and run the live policy."""

    readiness = assess_hardware_readiness(calibration)
    if not readiness.ready:
        details = "\n".join(f"- {item}" for item in readiness.blockers)
        raise RuntimeError(
            "physical policy execution is disabled by readiness checks:\n" + details
        )

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from franka_trajectory_replay.cartesian_replay_client import CartesianReplayClient
    from franka_trajectory_replay.replay_client import Rejected
    from franka_trajectory_replay.runconfig import load_config
    from inspire_franka_trajectory_replay.replay import (
        CoordinatedReplayClient,
        load_home,
    )
    from .mujoco_threading_env import seed_action_history
    from .session import PolicyRolloutSession
    from utils.data_collection import RolloutDataCollector, create_rollout_dir

    config = load_config(args.config)
    home_arm, home_hand = load_home(args.home)
    if np.max(np.abs(home_arm - fo.FRANKA_ARM_RESET_JOINTS_M24)) > args.max_home_delta:
        raise ValueError("home arm pose does not match the policy's M24 reset pose")

    metadata = runner.metadata()
    run_dir = create_rollout_dir(args.recording_root, "hardware")
    collector = RolloutDataCollector(
        run_dir,
        metadata={
            "checkpoint": metadata["checkpoint"],
            "checkpoint_sha256": metadata["sha256"],
            "checkpoint_weight_source": metadata["weight_source"],
            "flow_integration_steps": metadata["integration_steps"],
            "camera_profile": str(calibration.source_path),
            "collection_mode": "physical_closed_loop_student",
            "controller": config["cartesian"]["controller_name"],
        },
        record_rgbd=args.record_rgbd,
    )
    session = PolicyRolloutSession(runner, calibration, collector=collector)
    frame = TrainingFrameAdapter()

    rclpy.init(args=None)
    home_client = CoordinatedReplayClient(
        config, args.hand_topic, args.hand_state_topic
    )
    arm_client = CartesianReplayClient(config, node_name="policy_arm_setup")
    node = _hardware_node_class()(
        calibration,
        config,
        hand_command_topic=args.hand_topic,
        hand_state_topic=args.hand_state_topic,
        max_state_age_s=args.max_state_age,
        max_frame_skew_s=args.max_frame_skew,
    )
    executor = MultiThreadedExecutor(num_threads=5)
    for item in (home_client, arm_client, node):
        executor.add_node(item)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    status = "error"
    steps = 0
    missed_deadlines = 0
    limited_policy_targets = 0
    watchdog_stop = False
    live_preflight_s = []
    grasp_controlled_offset_m = None
    artifact = None
    try:
        home_client.ensure_active(print)
        if not args.yes:
            input("Move the FR3 and Inspire hand to the policy reset pose? [Enter/Ctrl-C] ")
        home_client.goto(home_arm)
        home_client.wait_for_hand_link()
        home_client.command_hand(home_hand)
        home_client.wait_for_hand(
            home_hand, args.hand_timeout, args.hand_tolerance
        )
        print("policy homing complete")

        tool = policy_tool_transform(config)
        arm_client.check_tool(tool, print)
        if arm_client.uses_dh_model():
            raise Rejected("physical rollout requires model_source=franka, not dh")
        arm_client.preflight(home_client.current_joint_positions(), print)
        arm_client.ensure_active(print)
        parameters = arm_client.controller_parameters()
        expected = {
            "translational_stiffness": 565.0,
            "rotational_stiffness": 28.0,
            "nullspace_stiffness": 10.0,
            "stiffness_scale": 1.0,
            "target_filter": 1.0,
            "torque_rate_limit": 1.0,
            "max_policy_step_m": 0.036,
            "max_policy_step_rad": 0.18,
            "policy_command_timeout": 0.5,
            "state_publish_rate": 50.0,
        }
        mismatches = [
            f"{name}={parameters.get(name)!r} (expected {value})"
            for name, value in expected.items()
            if not math.isclose(
                float(parameters.get(name, float("nan"))), value, rel_tol=1e-6
            )
        ]
        if mismatches:
            raise Rejected(
                "controller is not using the policy profile: " + "; ".join(mismatches)
            )
        if parameters.get("coriolis_compensation") is not True:
            raise Rejected("policy controller must enable coriolis_compensation")

        # Homing and setup are complete. Keeping these helper nodes in the
        # Python executor would continue deserializing joint/robot state that
        # the policy loop never reads. The main node owns the live state and
        # controller-status subscriptions from this point onward.
        executor.remove_node(home_client)
        executor.remove_node(arm_client)
        node.wait_ready(args.input_timeout)

        # Synthetic tensors do not exercise every content-dependent point-cloud
        # path. Run one warm-up and four measured real-frame passes before any
        # policy command. Physical execution requires the whole policy pass to
        # fit inside its actual command period, not merely the much looser
        # controller watchdog.
        preflight_session = PolicyRolloutSession(runner, calibration)
        for _ in range(5):
            sample = node.wait_for_fresh_sample(args.input_timeout)
            grasp_position, grasp_quaternion, _ = node.wait_for_grasp_pose(
                args.input_timeout
            )
            q_policy_hand, dq_policy_hand = node.policy_hand_state(sample)
            preflight_session.reset(
                previous_filtered_native_action=np.zeros(9), seed=args.seed
            )
            preflight_started = time.perf_counter()
            preflight_session.step(
                joint_position=np.concatenate((sample.arm_position, q_policy_hand)),
                joint_velocity=np.concatenate((sample.arm_velocity, dq_policy_hand)),
                rgb=sample.rgb,
                depth=sample.depth,
                depth_units=sample.depth_units,
                trajectory_progress=(
                    0.0 if runner.config.trajectory_progress_conditioning else None
                ),
                process_phase=(
                    "policy" if runner.config.cyclic_process_phase_conditioning else None
                ),
            )
            live_preflight_s.append(time.perf_counter() - preflight_started)

        command_period = 1.0 / args.rate
        steady_budget = 0.9 * command_period
        steady_preflight_max = max(live_preflight_s[1:])
        print(
            f"live policy preflight: first {live_preflight_s[0]:.3f} s, "
            f"steady max {steady_preflight_max:.3f} s "
            f"(required < {steady_budget:.3f} s)",
            flush=True,
        )
        if steady_preflight_max >= steady_budget:
            raise Rejected(
                f"steady live policy pass took up to {steady_preflight_max:.3f} s; "
                f"must be below {steady_budget:.3f} s for {args.rate:g} Hz physical "
                "execution; reduce --integration-steps or --rate"
            )

        # Re-seed from a fresh measured state so dry preflight cannot affect the
        # first commanded action or the temporal action ensemble.
        sample = node.wait_for_fresh_sample(args.input_timeout)
        grasp_position, grasp_quaternion, _ = node.wait_for_grasp_pose(args.input_timeout)
        q_policy_hand, _ = node.policy_hand_state(sample)
        grasp_controlled_offset_m = float(
            np.linalg.norm(sample.controller_measured_position - grasp_position)
        )
        print(
            "live grasp-to-controller offset: "
            f"{1.0e3 * grasp_controlled_offset_m:.1f} mm (compensated)",
            flush=True,
        )
        # Match the reset contract used by training and the MuJoCo rollout.
        # Pose-derived seeding here made the last nine proprioception values
        # strongly out of distribution before the first physical command.
        session.reset(previous_filtered_native_action=np.zeros(9), seed=args.seed)
        coordinator = GripCycleCoordinator(
            rate_hz=args.rate,
            max_cycles=args.cycles,
            reset_position=grasp_position,
            reset_quaternion=grasp_quaternion,
            reset_hand=q_policy_hand,
        )
        if not args.yes:
            input("Start closed-loop physical policy execution? [Enter/Ctrl-C] ")

        period = 1.0 / args.rate
        start = time.monotonic()
        next_tick = start
        status = "step_budget"
        for steps in range(1, args.max_steps + 1):
            tick_started = time.monotonic()
            sample = node.sample()
            grasp_position, grasp_quaternion, _flange_quaternion = node.grasp_pose()
            q_hand, dq_hand = node.policy_hand_state(sample)
            q10 = np.concatenate((sample.arm_position, q_hand))
            dq10 = np.concatenate((sample.arm_velocity, dq_hand))
            phase = coordinator.process_phase()
            progress = min(
                1.0,
                (tick_started - start)
                / runner.config.trajectory_progress_duration_s,
            )
            result = session.step(
                joint_position=q10,
                joint_velocity=dq10,
                rgb=sample.rgb,
                depth=sample.depth,
                depth_units=sample.depth_units,
                trajectory_progress=(
                    progress
                    if runner.config.trajectory_progress_conditioning
                    else None
                ),
                process_phase=(
                    phase if runner.config.cyclic_process_phase_conditioning else None
                ),
                sample_time_s=tick_started,
                task_signals={
                    "completed_cycles": coordinator.completed_cycles,
                    "watchdog_stop": False,
                },
            )
            filtered = result.filtered_native_action.numpy()
            target_position, target_quaternion = frame.controller_target(
                filtered,
                grasp_position_base=grasp_position,
                grasp_quaternion_base=grasp_quaternion,
                controlled_position_base=sample.controller_measured_position,
                controlled_quaternion_base=sample.controller_measured_quaternion,
            )
            (
                target_position,
                target_quaternion,
                position_limited,
                orientation_limited,
            ) = limit_cartesian_step(
                target_position,
                target_quaternion,
                sample.controller_target_position,
                sample.controller_target_quaternion,
                max_position_step_m=expected["max_policy_step_m"],
                max_orientation_step_rad=expected["max_policy_step_rad"],
            )
            if position_limited or orientation_limited:
                limited_policy_targets += 1
            node.publish_policy_target(target_position, target_quaternion, home_arm)
            node.publish_hand_target(fo.pinch_targets(filtered))
            event, turn_progress = coordinator.update(
                position=grasp_position,
                quaternion=grasp_quaternion,
                hand=q_hand,
            )
            if event is not None:
                print(
                    f"policy event: {event}; cycles={coordinator.completed_cycles}, "
                    f"grasp-turn proxy={math.degrees(turn_progress):.1f} deg",
                    flush=True,
                )
            if event == "cycle_completed":
                world_position, world_quaternion = frame.pose_base_to_world(
                    grasp_position, grasp_quaternion
                )
                session.action_filter.reset(
                    seed_action_history(
                        fo.GraspFrameState(
                            world_position,
                            world_quaternion,
                            np.zeros(3),
                            np.zeros(3),
                            np.zeros((6, 7)),
                        ),
                        q_hand,
                    )
                )
            if coordinator.failed:
                status = "return_failed"
                break
            if coordinator.completed_cycles >= args.cycles:
                status = "completed"
                break
            controller_status = node.controller_status()
            if controller_status.get("tracking_fault") == "true":
                raise Rejected(
                    "the Cartesian controller stopped on a tracking fault: "
                    + controller_status.get("last_fault", "unknown")
                )
            if controller_status.get("policy_watchdog_stop") == "true":
                watchdog_stop = True
                status = "controller_watchdog"
                raise RuntimeError("Cartesian controller policy-command watchdog stopped")
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                missed_deadlines += 1
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        status = "interrupted"
        node.abort()
    except Exception:
        node.abort()
        raise
    finally:
        # Stop accepting live policy motion before tearing down the ROS graph.
        # The controller ramps its phase clock to zero and then holds.
        node.abort()
        time.sleep(0.6)
        artifact = collector.close() if collector.sample_count else None
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        for item in (node, arm_client, home_client):
            item.destroy_node()
        rclpy.shutdown()
    return {
        "status": status,
        "steps": steps,
        "completed_cycles": (
            coordinator.completed_cycles if "coordinator" in locals() else 0
        ),
        "missed_policy_deadlines": missed_deadlines,
        "limited_policy_targets": limited_policy_targets,
        "watchdog_stop": watchdog_stop,
        "live_preflight_s": live_preflight_s,
        "grasp_controlled_offset_m": grasp_controlled_offset_m,
        "recording": None if artifact is None else str(artifact.data_path),
    }


__all__ = [
    "HardwareReadiness",
    "RgbdFrameSynchronizer",
    "TrainingFrameAdapter",
    "assess_hardware_readiness",
    "assert_policy_camera_frames",
    "image_message_to_numpy",
    "limit_cartesian_step",
    "physical_hand_state_to_policy",
    "policy_tool_transform",
    "retarget_grasp_pose_to_controlled_pose",
    "run_hardware_rollout",
]
