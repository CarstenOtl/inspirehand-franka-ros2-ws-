#!/usr/bin/env python3
"""Passive ROS 2 to MuJoCo joint-state and trajectory visualizer.

This node is deliberately not a ros2_control hardware component.  It creates
no application publishers and never writes an actuator command.  ROS callbacks
prepare immutable commands on an executor thread while the main thread
exclusively owns MuJoCo and its passive viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


# Viewer frame and label overlays, by the MuJoCo enum member they select.
FRAME_OPTIONS = {
    "none": "mjFRAME_NONE",
    "body": "mjFRAME_BODY",
    "geom": "mjFRAME_GEOM",
    "site": "mjFRAME_SITE",
    "camera": "mjFRAME_CAMERA",
    "light": "mjFRAME_LIGHT",
    "contact": "mjFRAME_CONTACT",
    "world": "mjFRAME_WORLD",
}
LABEL_OPTIONS = {
    "none": "mjLABEL_NONE",
    "body": "mjLABEL_BODY",
    "joint": "mjLABEL_JOINT",
    "geom": "mjLABEL_GEOM",
    "site": "mjLABEL_SITE",
    "camera": "mjLABEL_CAMERA",
    "actuator": "mjLABEL_ACTUATOR",
    "tendon": "mjLABEL_TENDON",
    "constraint": "mjLABEL_CONSTRAINT",
    "contact": "mjLABEL_CONTACTPOINT",
}
SITE_GROUP_COUNT = 6


@dataclass(frozen=True)
class ViewerOptions:
    """Overlay settings applied to the passive viewer once it exists."""

    frame_name: str
    label_name: str
    site_groups: tuple[int, ...] | None  # None keeps the viewer's own default (0, 1, 2)
    frame_scale: float


def parse_viewer_options(
    frame: str, label: str, site_groups: str, frame_scale: float = 1.0
) -> ViewerOptions:
    """Validate the overlay parameters before any window exists.

    ``site_groups`` is a comma- or space-separated list of MuJoCo site groups
    (0-5) to render; the flange ``attachment_site`` in fr3.xml sits in group 4,
    which the viewer hides by default. Empty keeps the default.
    """
    frame_key = frame.strip().lower()
    if frame_key not in FRAME_OPTIONS:
        raise ValueError(
            f"frame must be one of {sorted(FRAME_OPTIONS)}, got {frame!r}"
        )
    label_key = label.strip().lower()
    if label_key not in LABEL_OPTIONS:
        raise ValueError(
            f"label must be one of {sorted(LABEL_OPTIONS)}, got {label!r}"
        )
    groups: tuple[int, ...] | None = None
    text = site_groups.replace(",", " ").split()
    if text:
        parsed = []
        for token in text:
            if not token.isdigit() or not 0 <= int(token) < SITE_GROUP_COUNT:
                raise ValueError(
                    f"site_groups entries must be integers in [0, {SITE_GROUP_COUNT - 1}], "
                    f"got {site_groups!r}"
                )
            parsed.append(int(token))
        groups = tuple(sorted(set(parsed)))
    if not math.isfinite(frame_scale) or frame_scale <= 0.0:
        raise ValueError("frame_scale must be finite and positive")
    return ViewerOptions(
        FRAME_OPTIONS[frame_key], LABEL_OPTIONS[label_key], groups, float(frame_scale)
    )


@dataclass(frozen=True)
class JointBinding:
    """One scalar MuJoCo joint and its independent state addresses."""

    name: str
    joint_id: int
    qpos_address: int
    qvel_address: int


@dataclass(frozen=True)
class JointCommand:
    """A sparse, validated kinematic update keyed by MuJoCo joint name."""

    positions: Mapping[str, float]
    velocities: Mapping[str, float]


@dataclass(frozen=True)
class TrajectoryPlan:
    """A trajectory converted to MuJoCo joint order and monotonic time."""

    names: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    started_at: float
    start_positions: np.ndarray
    start_velocities: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, now: float) -> tuple[JointCommand, bool]:
        """Interpolate at wall-monotonic time; use cubic data when available."""
        elapsed = max(0.0, now - self.started_at)
        if elapsed >= self.duration:
            positions = self.positions[-1]
            # A replayed pose is held at completion.  Keeping a non-zero final
            # velocity would make the visual state internally inconsistent.
            velocities = np.zeros(len(self.names), dtype=float)
            return self._command(positions, velocities), True

        upper = int(np.searchsorted(self.times, elapsed, side="right"))
        if upper == 0:
            t0 = 0.0
            q0 = self.start_positions
            v0 = self.start_velocities
        else:
            t0 = float(self.times[upper - 1])
            q0 = self.positions[upper - 1]
            v0 = self.velocities[upper - 1]

        t1 = float(self.times[upper])
        q1 = self.positions[upper]
        v1 = self.velocities[upper]
        duration = t1 - t0
        if duration <= 0.0:
            return self._command(q1, np.zeros(len(self.names))), False

        phase = min(max((elapsed - t0) / duration, 0.0), 1.0)
        has_endpoint_velocities = np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))
        if has_endpoint_velocities:
            # Cubic Hermite interpolation preserves the velocities supplied in
            # JointTrajectoryPoint instead of treating them as decoration.
            s2 = phase * phase
            s3 = s2 * phase
            positions = (
                (2.0 * s3 - 3.0 * s2 + 1.0) * q0
                + (s3 - 2.0 * s2 + phase) * duration * v0
                + (-2.0 * s3 + 3.0 * s2) * q1
                + (s3 - s2) * duration * v1
            )
            velocities = (
                (6.0 * s2 - 6.0 * phase) * q0 / duration
                + (3.0 * s2 - 4.0 * phase + 1.0) * v0
                + (-6.0 * s2 + 6.0 * phase) * q1 / duration
                + (3.0 * s2 - 2.0 * phase) * v1
            )
        else:
            positions = q0 + phase * (q1 - q0)
            velocities = (q1 - q0) / duration
        return self._command(positions, velocities), False

    def _command(self, positions: np.ndarray, velocities: np.ndarray) -> JointCommand:
        return JointCommand(
            positions=dict(zip(self.names, positions, strict=True)),
            velocities=dict(zip(self.names, velocities, strict=True)),
        )


def duration_seconds(duration: Any) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1.0e-9


def parse_mapping_entries(entries: Iterable[str]) -> dict[str, str]:
    """Parse ``ROS_NAME=MUJOCO_NAME`` entries and reject ambiguous aliases."""
    result: dict[str, str] = {}
    for raw_entry in entries:
        source, separator, target = str(raw_entry).partition("=")
        source, target = source.strip(), target.strip()
        if not separator or not source or not target:
            raise ValueError(
                f"invalid joint mapping {raw_entry!r}; expected ROS_NAME=MUJOCO_NAME"
            )
        if source in result and result[source] != target:
            raise ValueError(f"joint {source!r} has more than one MuJoCo mapping")
        result[source] = target
    return result


def load_mapping_file(path: str) -> dict[str, str]:
    """Load either a YAML mapping or a top-level ``joint_map`` mapping."""
    if not path:
        return {}
    import yaml

    mapping_path = Path(path).expanduser()
    with mapping_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if isinstance(document, dict) and "joint_map" in document:
        document = document["joint_map"]
    if not isinstance(document, dict):
        raise ValueError(f"{mapping_path} must contain a YAML mapping")
    result = {}
    for source, target in document.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"{mapping_path} joint names must be strings")
        result[source] = target
    return result


class JointIndex:
    """Resolve ROS names without assuming MuJoCo's joint/qpos/dof ordering."""

    def __init__(self, mujoco: Any, model: Any, aliases: Mapping[str, str]) -> None:
        self.bindings: dict[str, JointBinding] = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                continue
            next_qpos = model.nq if joint_id + 1 == model.njnt else model.jnt_qposadr[joint_id + 1]
            next_qvel = model.nv if joint_id + 1 == model.njnt else model.jnt_dofadr[joint_id + 1]
            qpos_address = int(model.jnt_qposadr[joint_id])
            qvel_address = int(model.jnt_dofadr[joint_id])
            if int(next_qpos) - qpos_address != 1 or int(next_qvel) - qvel_address != 1:
                # A ROS JointState element is scalar.  Mapping it to a MuJoCo
                # free/ball joint would silently corrupt quaternion coordinates.
                continue
            self.bindings[name] = JointBinding(
                name=name,
                joint_id=joint_id,
                qpos_address=qpos_address,
                qvel_address=qvel_address,
            )

        self.aliases = dict(aliases)
        unknown_targets = sorted(set(self.aliases.values()) - set(self.bindings))
        if unknown_targets:
            raise ValueError(f"joint map targets are absent/non-scalar in MJCF: {unknown_targets}")

        # Equality joint constraints are the MJCF equivalent of URDF mimic
        # joints.  Kinematic replay does not solve constraints, so apply their
        # polynomial explicitly whenever a follower was not itself reported.
        self.couplings: list[tuple[JointBinding, JointBinding, np.ndarray]] = []
        joint_equality = int(mujoco.mjtEq.mjEQ_JOINT)
        for equality_id in range(model.neq):
            if int(model.eq_type[equality_id]) != joint_equality:
                continue
            follower_id = int(model.eq_obj1id[equality_id])
            driver_id = int(model.eq_obj2id[equality_id])
            follower_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, follower_id
            )
            driver_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, driver_id
            )
            if follower_name in self.bindings and driver_name in self.bindings:
                self.couplings.append(
                    (
                        self.bindings[follower_name],
                        self.bindings[driver_name],
                        np.asarray(model.eq_data[equality_id, :5], dtype=float).copy(),
                    )
                )

    def resolve(self, ros_name: str) -> JointBinding | None:
        return self.bindings.get(self.aliases.get(ros_name, ros_name))

    def resolve_names(
        self, ros_names: Sequence[str]
    ) -> tuple[list[tuple[int, JointBinding]], list[str]]:
        resolved: list[tuple[int, JointBinding]] = []
        unknown: list[str] = []
        destinations: set[str] = set()
        for source_index, source_name in enumerate(ros_names):
            binding = self.resolve(source_name)
            if binding is None:
                unknown.append(source_name)
                continue
            if binding.name in destinations:
                raise ValueError(
                    f"message maps more than one source joint to {binding.name!r}"
                )
            destinations.add(binding.name)
            resolved.append((source_index, binding))
        return resolved, unknown

    def apply_couplings(self, data: Any, explicitly_commanded: set[str]) -> None:
        for _ in range(max(1, len(self.couplings))):
            for follower, driver, coefficients in self.couplings:
                if follower.name in explicitly_commanded:
                    continue
                q = float(data.qpos[driver.qpos_address])
                dq = float(data.qvel[driver.qvel_address])
                powers = np.array((1.0, q, q * q, q**3, q**4))
                derivative = (
                    coefficients[1]
                    + 2.0 * coefficients[2] * q
                    + 3.0 * coefficients[3] * q * q
                    + 4.0 * coefficients[4] * q**3
                )
                data.qpos[follower.qpos_address] = float(np.dot(coefficients, powers))
                data.qvel[follower.qvel_address] = derivative * dq


class MujocoVisNode(Node):
    """Subscriber-only node whose main thread owns the MuJoCo render loop."""

    def __init__(self) -> None:
        super().__init__("mujoco_vis_node", namespace="mujoco_sim")
        default_model = str(
            Path(get_package_share_directory("inspire_franka_sim"))
            / "mjcf"
            / "inspire_franka_flange_scene.xml"
        )
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("initial_keyframe", "start")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("trajectory_topic", "/mujoco_sim/joint_trajectory")
        self.declare_parameter("subscribe_joint_states", True)
        self.declare_parameter("subscribe_trajectory", True)
        self.declare_parameter("joint_state_preempts_trajectory", True)
        self.declare_parameter("joint_map_file", "")
        joint_map_parameter = self.declare_parameter(
            "joint_map", Parameter.Type.STRING_ARRAY
        )
        self.declare_parameter("playback_speed", 1.0)
        self.declare_parameter("render_hz", 60.0)
        self.declare_parameter("headless", False)
        self.declare_parameter("show_left_ui", False)
        self.declare_parameter("show_right_ui", False)
        self.declare_parameter("step_physics", False)
        self.declare_parameter("max_physics_steps_per_frame", 20)
        self.declare_parameter("clamp_to_joint_limits", False)
        # Overlays: coordinate frames and labels drawn by the viewer, and which
        # site groups it renders. Nothing here touches the model or the data.
        self.declare_parameter("frame", "none")
        self.declare_parameter("label", "none")
        # Dynamically typed so `-p site_groups:=4` and `-p frame_scale:=2` on a
        # ros2 run command line (parsed as integers) work like the launch
        # arguments; the values are coerced below.
        self.declare_parameter(
            "site_groups", "", ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter(
            "frame_scale", 1.0, ParameterDescriptor(dynamic_typing=True)
        )

        site_groups = self.get_parameter("site_groups").value
        if isinstance(site_groups, (list, tuple)):
            site_groups = " ".join(str(group) for group in site_groups)
        self.viewer_options = parse_viewer_options(
            str(self.get_parameter("frame").value),
            str(self.get_parameter("label").value),
            "" if site_groups is None else str(site_groups),
            float(self.get_parameter("frame_scale").value),
        )
        self.playback_speed = float(self.get_parameter("playback_speed").value)
        self.render_hz = float(self.get_parameter("render_hz").value)
        if not math.isfinite(self.playback_speed) or self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be finite and positive")
        if not math.isfinite(self.render_hz) or self.render_hz <= 0.0:
            raise ValueError("render_hz must be finite and positive")
        self.headless = bool(self.get_parameter("headless").value)
        self.show_left_ui = bool(self.get_parameter("show_left_ui").value)
        self.show_right_ui = bool(self.get_parameter("show_right_ui").value)
        self.step_physics = bool(self.get_parameter("step_physics").value)
        self.max_physics_steps = int(
            self.get_parameter("max_physics_steps_per_frame").value
        )
        if self.max_physics_steps < 1:
            raise ValueError("max_physics_steps_per_frame must be positive")
        self.clamp_limits = bool(self.get_parameter("clamp_to_joint_limits").value)
        self.joint_state_preempts = bool(
            self.get_parameter("joint_state_preempts_trajectory").value
        )

        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "the MuJoCo Python bindings are not installed; use the project "
                "container or install the pinned mujoco version"
            ) from exc
        self.mujoco = mujoco
        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        # Make the safety boundary true in MuJoCo as well as in ROS.  Even when
        # step_physics is explicitly enabled, actuator bias terms and viewer UI
        # controls cannot generate generalized forces.
        self.model.opt.disableflags |= int(
            mujoco.mjtDisableBit.mjDSBL_ACTUATION
        )
        keyframe = str(self.get_parameter("initial_keyframe").value)
        if keyframe:
            keyframe_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            )
            if keyframe_id < 0:
                raise ValueError(f"MJCF has no keyframe named {keyframe!r}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        aliases = load_mapping_file(str(self.get_parameter("joint_map_file").value))
        # Jazzy returns a NOT_SET Parameter here when the typed array has no
        # override.  Calling get_parameter() for it raises
        # ParameterUninitializedException; the declaration result safely
        # exposes value=None, which is the empty mapping we want by default.
        aliases.update(parse_mapping_entries(joint_map_parameter.value or ()))
        self.joints = JointIndex(mujoco, self.model, aliases)
        self._lock = threading.Lock()
        self._latched_command: JointCommand | None = None
        self._trajectory: TrajectoryPlan | None = None
        self._display_positions = {
            name: float(self.data.qpos[binding.qpos_address])
            for name, binding in self.joints.bindings.items()
        }
        self._display_velocities = {
            name: float(self.data.qvel[binding.qvel_address])
            for name, binding in self.joints.bindings.items()
        }
        self._warned_unknown: set[tuple[str, ...]] = set()
        self._last_render_time = time.monotonic()
        self._physics_accumulator = 0.0

        # Do not call this `_subscriptions`: that is an rclpy.Node internal
        # collection, and shadowing/appending to it corrupts destroy_node().
        self._input_subscriptions = []
        if bool(self.get_parameter("subscribe_joint_states").value):
            topic = str(self.get_parameter("joint_state_topic").value)
            self._input_subscriptions.append(
                self.create_subscription(
                    JointState, topic, self._on_joint_state, qos_profile_sensor_data
                )
            )
            self.get_logger().info(f"subscribed to joint states on {topic}")
        if bool(self.get_parameter("subscribe_trajectory").value):
            topic = str(self.get_parameter("trajectory_topic").value)
            self._input_subscriptions.append(
                self.create_subscription(JointTrajectory, topic, self._on_trajectory, 10)
            )
            self.get_logger().info(f"subscribed to trajectories on {topic}")
        if not self._input_subscriptions:
            raise ValueError("both input subscribers are disabled")

        mode = "passive physics stepping" if self.step_physics else "kinematic mj_forward"
        self.get_logger().info(
            f"loaded {model_path} with {len(self.joints.bindings)} scalar joints; "
            f"render mode: {mode}; this node has no application publishers"
        )

    def _warn_unknown(self, unknown: Sequence[str]) -> None:
        key = tuple(sorted(set(unknown)))
        if key and key not in self._warned_unknown:
            self._warned_unknown.add(key)
            self.get_logger().warn(
                f"ignoring joints absent from the MJCF/joint map: {list(key)}"
            )

    def _on_joint_state(self, message: JointState) -> None:
        count = len(message.name)
        if len(set(message.name)) != count:
            self.get_logger().error("discarding JointState with duplicate names")
            return
        if len(message.position) not in (0, count) or len(message.velocity) not in (0, count):
            self.get_logger().error(
                "discarding JointState: position/velocity must be empty or match name length"
            )
            return
        try:
            resolved, unknown = self.joints.resolve_names(message.name)
        except ValueError as exc:
            self.get_logger().error(f"discarding JointState: {exc}")
            return
        self._warn_unknown(unknown)
        positions = {
            binding.name: float(message.position[index])
            for index, binding in resolved
            if message.position
        }
        velocities = {
            binding.name: float(message.velocity[index])
            for index, binding in resolved
            if message.velocity
        }
        if not all(math.isfinite(value) for value in (*positions.values(), *velocities.values())):
            self.get_logger().error("discarding JointState containing NaN/Inf")
            return
        if not positions and not velocities:
            return
        with self._lock:
            self._latched_command = JointCommand(positions, velocities)
            if self.joint_state_preempts:
                self._trajectory = None

    def _on_trajectory(self, message: JointTrajectory) -> None:
        count = len(message.joint_names)
        if not count or not message.points:
            self.get_logger().error("discarding empty JointTrajectory")
            return
        if len(set(message.joint_names)) != count:
            self.get_logger().error("discarding JointTrajectory with duplicate names")
            return
        try:
            resolved, unknown = self.joints.resolve_names(message.joint_names)
        except ValueError as exc:
            self.get_logger().error(f"discarding JointTrajectory: {exc}")
            return
        self._warn_unknown(unknown)
        if not resolved:
            self.get_logger().error("discarding trajectory with no joints present in the MJCF")
            return

        names = tuple(binding.name for _, binding in resolved)
        times: list[float] = []
        rows: list[list[float]] = []
        velocity_rows: list[list[float]] = []
        for point_index, point in enumerate(message.points):
            if len(point.positions) != count:
                self.get_logger().error(
                    f"discarding trajectory: point {point_index} positions do not match joint_names"
                )
                return
            if len(point.velocities) not in (0, count):
                self.get_logger().error(
                    f"discarding trajectory: point {point_index} velocities do not match joint_names"
                )
                return
            timestamp = duration_seconds(point.time_from_start)
            row = [float(point.positions[index]) for index, _ in resolved]
            velocity_row = (
                [float(point.velocities[index]) for index, _ in resolved]
                if point.velocities
                else [math.nan] * len(resolved)
            )
            if not math.isfinite(timestamp) or timestamp < 0.0:
                self.get_logger().error("discarding trajectory with invalid time_from_start")
                return
            if times and timestamp <= times[-1]:
                self.get_logger().error(
                    "discarding trajectory: time_from_start must be strictly increasing"
                )
                return
            if not all(math.isfinite(value) for value in row):
                self.get_logger().error("discarding trajectory containing NaN/Inf positions")
                return
            if not all(math.isnan(value) or math.isfinite(value) for value in velocity_row):
                self.get_logger().error("discarding trajectory containing Inf velocities")
                return
            times.append(timestamp / self.playback_speed)
            rows.append(row)
            velocity_rows.append(
                [value * self.playback_speed for value in velocity_row]
            )

        with self._lock:
            start_positions = np.asarray(
                [self._display_positions[name] for name in names], dtype=float
            )
            start_velocities = np.asarray(
                [self._display_velocities[name] for name in names], dtype=float
            )
            if times[0] == 0.0:
                # searchsorted(side=right) expects no zero-duration start
                # segment.  The first pose is still rendered immediately.
                start_positions = np.asarray(rows[0], dtype=float)
                first_velocity = np.asarray(velocity_rows[0], dtype=float)
                start_velocities = np.where(np.isfinite(first_velocity), first_velocity, 0.0)
            self._trajectory = TrajectoryPlan(
                names=names,
                times=np.asarray(times, dtype=float),
                positions=np.asarray(rows, dtype=float),
                velocities=np.asarray(velocity_rows, dtype=float),
                started_at=time.monotonic(),
                start_positions=start_positions,
                start_velocities=start_velocities,
            )
            self._latched_command = None
        self.get_logger().info(
            f"accepted {len(message.points)}-point trajectory for {list(names)} "
            f"({times[-1]:.3f} s at {self.playback_speed:g}x)"
        )

    def _current_command(self, now: float) -> JointCommand | None:
        with self._lock:
            if self._trajectory is not None:
                command, complete = self._trajectory.sample(now)
                if complete:
                    self._trajectory = None
                    self._latched_command = command
                return command
            return self._latched_command

    def _apply_command(self, command: JointCommand | None) -> None:
        if command is None:
            return
        commanded = set(command.positions) | set(command.velocities)
        for name, value in command.positions.items():
            binding = self.joints.bindings[name]
            if self.clamp_limits and bool(self.model.jnt_limited[binding.joint_id]):
                low, high = self.model.jnt_range[binding.joint_id]
                value = min(max(value, float(low)), float(high))
            self.data.qpos[binding.qpos_address] = value
        # Positions without an accompanying velocity are poses, not an
        # instruction to preserve a stale velocity from an older message.
        for name in command.positions:
            binding = self.joints.bindings[name]
            self.data.qvel[binding.qvel_address] = command.velocities.get(name, 0.0)
        for name, value in command.velocities.items():
            binding = self.joints.bindings[name]
            self.data.qvel[binding.qvel_address] = value
        self.joints.apply_couplings(self.data, commanded)
        if self.data.qacc.size:
            self.data.qacc[:] = 0.0
        if self.data.qacc_warmstart.size:
            self.data.qacc_warmstart[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        with self._lock:
            for name, binding in self.joints.bindings.items():
                self._display_positions[name] = float(self.data.qpos[binding.qpos_address])
                self._display_velocities[name] = float(self.data.qvel[binding.qvel_address])

    def _render_once(self, now: float) -> None:
        if self.step_physics:
            elapsed = max(0.0, now - self._last_render_time)
            timestep = float(self.model.opt.timestep)
            # Carry fractional timesteps across render frames.  Cap accumulated
            # lag so a paused window cannot trigger an unbounded catch-up burst.
            self._physics_accumulator = min(
                self._physics_accumulator + elapsed,
                self.max_physics_steps * timestep,
            )
            steps = int(self._physics_accumulator / timestep)
            self._physics_accumulator -= steps * timestep
            # Passive means no actuator or externally applied force is allowed
            # to survive from the MJCF/UI into a simulation step.
            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            for _ in range(steps):
                self.mujoco.mj_step(self.model, self.data)
        self._last_render_time = now
        self._apply_command(self._current_command(now))

    def _apply_viewer_options(self, mujoco: Any, viewer: Any) -> None:
        """Frame/label overlays and site-group visibility, under the viewer lock."""
        options = self.viewer_options
        viewer.opt.frame = getattr(mujoco.mjtFrame, options.frame_name)
        viewer.opt.label = getattr(mujoco.mjtLabel, options.label_name)
        if options.site_groups is not None:
            for group in range(len(viewer.opt.sitegroup)):
                viewer.opt.sitegroup[group] = 1 if group in options.site_groups else 0
        if options.frame_scale != 1.0:
            # Frame axes are drawn at model.vis.scale.framelength x meansize; the
            # scale is visual-only and never enters the kinematics.
            self.model.vis.scale.framelength *= options.frame_scale
            self.model.vis.scale.framewidth *= options.frame_scale
        self.get_logger().info(
            "viewer overlays: frame=%s label=%s site_groups=%s frame_scale=%.2f"
            % (options.frame_name, options.label_name,
               "default" if options.site_groups is None else list(options.site_groups),
               options.frame_scale)
        )

    def run(self) -> None:
        """Spin ROS separately while this thread services MuJoCo and the UI."""
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        ros_thread = threading.Thread(
            target=executor.spin, name="mujoco-vis-ros-executor", daemon=True
        )
        ros_thread.start()
        frame_period = 1.0 / self.render_hz
        try:
            if self.headless:
                while rclpy.ok():
                    started = time.monotonic()
                    self._render_once(started)
                    time.sleep(max(0.0, frame_period - (time.monotonic() - started)))
                return

            import mujoco.viewer

            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=self.show_left_ui,
                show_right_ui=self.show_right_ui,
            ) as viewer:
                with viewer.lock():
                    self._apply_viewer_options(mujoco, viewer)
                while rclpy.ok() and viewer.is_running():
                    started = time.monotonic()
                    with viewer.lock():
                        self._render_once(started)
                    viewer.sync()
                    time.sleep(max(0.0, frame_period - (time.monotonic() - started)))
        finally:
            executor.shutdown()
            ros_thread.join(timeout=2.0)


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node: MujocoVisNode | None = None
    try:
        node = MujocoVisNode()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                # ros2 launch forwards its own SIGINT to the child. It can
                # arrive after rclpy's handler stopped the run loop but while
                # the node is destroying its internal parameter publisher.
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
