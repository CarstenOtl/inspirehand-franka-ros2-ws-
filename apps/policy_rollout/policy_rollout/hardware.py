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
        flange_quaternion_base,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode Forge's grasp target to base-frame tool position/flange attitude."""

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
        flange_to_grasp = fo.quat_mul(
            fo.quat_conjugate(np.asarray(flange_quaternion_base, dtype=float)),
            np.asarray(grasp_quaternion_base, dtype=float),
        )
        target_flange_quaternion = fo.quat_mul(
            target_grasp_quaternion_base, fo.quat_conjugate(flange_to_grasp)
        )
        target_flange_quaternion /= np.linalg.norm(target_flange_quaternion)
        return target_position_base, target_flange_quaternion


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
            from franka_trajectory_replay_msgs.msg import CartesianGoto, CartesianReplayState
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image, JointState
            from std_msgs.msg import Empty
            import tf2_ros

            super().__init__("fr3_policy_rollout")
            self.calibration = calibration
            self.config = config
            self.max_state_age_s = float(max_state_age_s)
            self.max_frame_skew_s = float(max_frame_skew_s)
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
            self._arm = self._hand = self._rgb = self._depth = None
            self._camera_info = self._cartesian = None
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
                JointState,
                hand_state_topic,
                lambda message: self._put("hand", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                calibration.color_topic,
                lambda message: self._put("rgb", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                calibration.depth_topic,
                lambda message: self._put("depth", message),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                calibration.camera_info_topic,
                lambda message: self._put("camera_info", message),
                qos_profile_sensor_data,
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self._camera_checked = False

        def _put(self, name, message):
            with self._lock:
                setattr(self, "_" + name, message)
                self._receive_time[name] = time.monotonic()

        def ready(self) -> bool:
            with self._lock:
                return all(
                    value is not None
                    for value in (
                        self._arm,
                        self._cartesian,
                        self._hand,
                        self._rgb,
                        self._depth,
                        self._camera_info,
                    )
                )

        def wait_ready(self, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self.ready() and self.policy_publisher.get_subscription_count() > 0:
                    return
                time.sleep(0.02)
            raise TimeoutError("policy inputs/controller subscription did not become ready")

        def sample(self) -> LiveSample:
            with self._lock:
                messages = {
                    name: getattr(self, "_" + name)
                    for name in (
                        "arm",
                        "cartesian",
                        "hand",
                        "rgb",
                        "depth",
                        "camera_info",
                    )
                }
                receive = dict(self._receive_time)
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
            return LiveSample(q_arm, dq_arm, q_hand, dq_hand, rgb, depth, depth_units, now)

        @staticmethod
        def policy_hand_state(sample: LiveSample) -> tuple[np.ndarray, np.ndarray]:
            columns = {name: index for index, name in enumerate(HAND_JOINTS)}
            indices = [columns[name] for name in POLICY_HAND_JOINTS]
            return sample.hand_position[indices], sample.hand_velocity[indices]

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
    from franka_trajectory_replay import cartesian
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
    watchdog_stop = False
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

        tool = cartesian.tool_transform(
            config["tcp"]["offset_xyz"], config["tcp"]["offset_rpy"]
        )
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
        node.wait_ready(args.input_timeout)
        sample = node.sample()
        grasp_position, grasp_quaternion, _ = node.wait_for_grasp_pose(
            args.input_timeout
        )
        q_policy_hand, _ = node.policy_hand_state(sample)
        grasp_world_position, grasp_world_quaternion = frame.pose_base_to_world(
            grasp_position, grasp_quaternion
        )
        live_grasp = fo.GraspFrameState(
            grasp_world_position,
            grasp_world_quaternion,
            np.zeros(3),
            np.zeros(3),
            np.zeros((6, 7)),
        )
        previous_action = seed_action_history(live_grasp, q_policy_hand)
        session.reset(previous_filtered_native_action=previous_action, seed=args.seed)
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
            grasp_position, grasp_quaternion, flange_quaternion = node.grasp_pose()
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
                flange_quaternion_base=flange_quaternion,
            )
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
            arm_client._check_fault()
            controller_status = arm_client.status() or {}
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
        "watchdog_stop": watchdog_stop,
        "recording": None if artifact is None else str(artifact.data_path),
    }


__all__ = [
    "HardwareReadiness",
    "TrainingFrameAdapter",
    "assess_hardware_readiness",
    "image_message_to_numpy",
    "run_hardware_rollout",
]
