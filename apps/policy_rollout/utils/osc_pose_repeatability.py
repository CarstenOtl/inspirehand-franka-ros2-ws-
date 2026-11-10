#!/usr/bin/env python3
"""Accuracy and repeatability of the policy rollout's impedance controller.

Drives the FR3 between free-space poses through ``~/policy_goal``, the same input
the policy uses, so the 1 kHz clip, gains, friction feedforward and controlled
point are the rollout's. The hand is not commanded.

Every pose is reached from both sides along the A-B line, which splits the rest
error into two parts that have different causes:

* ``hysteresis`` (the two approach directions disagree): joint friction.
* ``bias`` (the mean of the two directions): gravity/load or model error.

The measured pose is the controller's own, from the joint encoders and the
robot model. It shows repeatability and how close the controller gets to its
goal in that model, not absolute accuracy against an external reference.

Run inside the container with the workspace sourced, after
``ros2 launch inspire_franka_trajectory_replay replay.launch.py arm_controller:=policy``::

    python3 apps/policy_rollout/utils/osc_pose_repeatability.py hardware --plan-only
    python3 apps/policy_rollout/utils/osc_pose_repeatability.py hardware
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from franka_trajectory_replay import kinematics as kin  # noqa: E402
from franka_trajectory_replay.limits import POSITION_LOWER, POSITION_UPPER  # noqa: E402
from policy_rollout import DEFAULT_CHECKPOINT, forge_osc as fo  # noqa: E402
from policy_rollout.hardware import (  # noqa: E402
    POLICY_TOOL_OFFSET_XYZ,
    TrainingFrameAdapter,
    _configured_load,
    policy_tool_offset_config,
    policy_tool_transform,
)

GRAVITY = 9.81
# The peak of a quintic blend, as multiples of (distance / duration) and
# (distance / duration^2).
QUINTIC_PEAK_SPEED = 1.875
QUINTIC_PEAK_ACCELERATION = 5.7735
ERROR_CHECKPOINTS_S = (0.25, 0.5, 1.0)
MAX_START_POSTURE_RESIDUAL_RAD = 0.12
DEFAULT_ORIENTATIONS_DEG = (
    (0.0, 0.0, 0.0),
    (20.0, 0.0, 0.0),
    (0.0, 20.0, 0.0),
    (0.0, 0.0, 30.0),
)


# --- rotations ------------------------------------------------------------------
def rotation_from_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def rotation_vector(rotation) -> np.ndarray:
    x, y, z, w = kin.matrix_to_quaternion(rotation)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector = np.array([x, y, z])
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return 2.0 * vector
    return vector / norm * 2.0 * math.atan2(norm, w)


def slerp(start, end, s: float) -> np.ndarray:
    return start @ rotation_from_vector(s * rotation_vector(start.T @ end))


def clip_euler(rotation, seam_deg: float, frame_yaw: float = math.pi) -> np.ndarray:
    """The controller's Euler angles for its orientation clip (roll, pitch, wrapped yaw)."""

    x, y, z, w = kin.matrix_to_quaternion(rotation_from_vector([0.0, 0.0, frame_yaw]) @ rotation)
    roll, pitch, yaw = fo.get_euler_xyz((w, x, y, z))
    if pitch > math.pi:
        pitch -= 2.0 * math.pi
    return np.array([roll, pitch, fo.wrap_yaw(yaw, seam_deg)])


# --- the motion -----------------------------------------------------------------
def quintic(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def move_duration(distance, angle, speed, acceleration, angular_speed, angular_acceleration) -> float:
    return max(
        QUINTIC_PEAK_SPEED * distance / speed,
        QUINTIC_PEAK_SPEED * angle / angular_speed,
        math.sqrt(QUINTIC_PEAK_ACCELERATION * distance / acceleration),
        math.sqrt(QUINTIC_PEAK_ACCELERATION * angle / angular_acceleration),
        0.05,
    )


@dataclass
class Move:
    position: np.ndarray
    rotation: np.ndarray
    position_label: str | None = None
    orientation_label: str | None = None
    # Sign of the final motion along A->B; None for a waypoint.
    approach: int | None = None
    cycle: int = 0
    q_nullspace: np.ndarray | None = None
    start_position: np.ndarray | None = None
    start_rotation: np.ndarray | None = None
    start_q: np.ndarray | None = None
    duration: float = 0.0
    # The slow move to the start pose: its own speeds, and it may begin at the bolt.
    transit: bool = False
    # IK along the line, start included, for the nullspace target.
    q_path: np.ndarray | None = None
    # When set, the redundant elbow is carried to these joints along the move.
    posture_goal: np.ndarray | None = None
    # Keep holding the goal until the arm is really there: the orientation clip
    # turns the wrist only a few deg/s, so a fixed dwell would be too short.
    wait_arrival: bool = False

    @property
    def is_arrival(self) -> bool:
        return self.position_label is not None


def build_moves(a, b, rotations, repeats: int, overshoot_m: float) -> list[Move]:
    """Four arrivals per orientation per repeat: A and B, each from both sides.

    The arm starts at A with the first orientation. A direct move approaches
    along the line; going through a waypoint beyond the pose reverses the side.
    """

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    axis = (b - a) / np.linalg.norm(b - a)
    points = {"A": a, "B": b}
    outside = {"A": a - overshoot_m * axis, "B": b + overshoot_m * axis}
    moves = []
    at = "A"
    for cycle in range(repeats):
        for orientation_label, rotation in rotations:
            other = "B" if at == "A" else "A"
            # Turn in place first and slowly, so the arrivals measure one orientation.
            moves.append(_orientation_change(points[at], rotation))
            # Direct to the far pose, direct back, then each pose again from beyond it.
            for label, via_outside in ((other, False), (at, False), (at, True), (other, True)):
                approach = 1 if label == "B" else -1
                if via_outside:
                    moves.append(Move(outside[label].copy(), rotation))
                    approach = -approach
                moves.append(
                    Move(points[label].copy(), rotation, label, orientation_label, approach, cycle)
                )
            at = other
    return moves


def _orientation_change(position, rotation) -> Move:
    return Move(np.asarray(position, dtype=float).copy(), rotation, transit=True, wait_arrival=True)


def build_transit(current_q, goal_position, goal_rotation, goal_q, tool, lift_m: float) -> list[Move]:
    """Straight up with the orientation held, then a straight line to the start pose.

    The line also brings the elbow to ``goal_q``, so the test always runs from the
    same arm posture, the one it was checked in, wherever the arm started.
    """

    pose = kin.flange_transform(np.asarray(current_q, dtype=float)) @ tool
    lifted = pose[:3, 3] + np.array([0.0, 0.0, lift_m])
    return [
        Move(lifted, pose[:3, :3].copy(), transit=True),
        Move(
            np.asarray(goal_position, dtype=float).copy(), goal_rotation.copy(), transit=True,
            posture_goal=np.asarray(goal_q, dtype=float).copy(), wait_arrival=True,
        ),
    ]


def solve_ik(position, rotation, seed, tool, iterations: int = 600, posture=None) -> np.ndarray | None:
    """IK from ``seed``; with ``posture``, the elbow is settled on it as well."""

    q = np.asarray(seed, dtype=float).copy()
    target = q.copy() if posture is None else np.asarray(posture, dtype=float)
    gain = 0.05 if posture is None else 0.3
    for _ in range(iterations):
        pose = kin.flange_transform(q) @ tool
        error = np.concatenate(
            (position - pose[:3, 3], rotation_vector(rotation @ pose[:3, :3].T))
        )
        jacobian = kin.flange_jacobian(q, tool)
        # A truncated, not damped, inverse: a damped one is no projector, and the
        # posture term would leak into the pose and stall the convergence.
        inverse = np.linalg.pinv(jacobian, rcond=1.0e-3)
        nullspace = (np.eye(7) - inverse @ jacobian) @ (target - q)
        if (
            np.linalg.norm(error[:3]) < 1.0e-5
            and np.linalg.norm(error[3:]) < 1.0e-4
            and (posture is None or np.linalg.norm(nullspace) < 1.0e-3)
        ):
            return q
        step = inverse @ error + gain * nullspace
        norm = float(np.max(np.abs(step)))
        q = np.clip(q + step * min(1.0, 0.2 / max(norm, 1.0e-9)), POSITION_LOWER, POSITION_UPPER)
    return None


def _check_tcp(position, args, where: str) -> None:
    if position[2] < args.min_tcp_z:
        raise ValueError(f"{where}: TCP z {position[2]:.3f} m is below --min-tcp-z")
    if np.hypot(position[0], position[1]) < args.min_tcp_radius:
        raise ValueError(f"{where}: TCP is within --min-tcp-radius of the robot's axis")
    if _distance_to_bolt_m(position) < args.bolt_keepout:
        raise ValueError(f"{where}: TCP is within --bolt-keepout of the bolt")


def _check_transit_tcp(position, last_bolt_distance, start_z, args, where: str) -> float:
    """A transit may start at the bolt; low down it may only move away from it."""

    if position[2] < min(args.min_tcp_z, start_z) - 1.0e-6:
        raise ValueError(f"{where}: TCP z {position[2]:.3f} m is below --min-tcp-z")
    if np.hypot(position[0], position[1]) < args.min_tcp_radius:
        raise ValueError(f"{where}: TCP is within --min-tcp-radius of the robot's axis")
    distance = _distance_to_bolt_m(position)
    if (
        distance < args.bolt_keepout
        and distance < last_bolt_distance - 1.0e-4
        and position[2] < args.bolt_clear_z
    ):
        raise ValueError(
            f"{where}: the path moves towards the bolt at z {position[2]:.3f} m, "
            "below --bolt-clear-z; increase --lift"
        )
    return distance


def _distance_to_bolt_m(position) -> float:
    # Training world is the FR3 base yawed by pi, with the base at (1.2, 0, 0).
    bolt_x = fo.ROBOT_BASE_POSITION[0] - fo.BOLT_BASE_POSITION[0]
    return float(np.hypot(position[0] - bolt_x, position[1] - fo.BOLT_BASE_POSITION[1]))


def plan(moves, start_q, tool, args) -> dict:
    """Solve IK along every straight line and refuse a path that is not clean."""

    start_pose = kin.flange_transform(start_q) @ tool
    previous_position, previous_rotation, q = start_pose[:3, 3], start_pose[:3, :3], start_q.copy()
    previous_euler = clip_euler(previous_rotation, args.yaw_seam_deg)
    min_margin = min_sigma = math.inf
    max_joint_travel = 0.0
    for index, move in enumerate(moves):
        distance = float(np.linalg.norm(move.position - previous_position))
        angle = float(np.linalg.norm(rotation_vector(previous_rotation.T @ move.rotation)))
        move.start_position, move.start_rotation, move.start_q = (
            previous_position.copy(), previous_rotation.copy(), q.copy()
        )
        move.duration = move_duration(
            distance, angle,
            args.transit_speed if move.transit else args.speed,
            args.acceleration,
            args.transit_angular_speed if move.transit else args.angular_speed,
            args.angular_acceleration,
        )
        q_path = [q.copy()]
        bolt_distance = _distance_to_bolt_m(previous_position)
        start_z = float(previous_position[2])
        where = f"move {index} to {move.position_label or 'waypoint'} {move.orientation_label or ''}"
        steps = max(1, int(math.ceil(distance / 0.01)), int(math.ceil(angle / 0.02)))
        for s in np.linspace(0.0, 1.0, steps + 1)[1:]:
            position = previous_position + s * (move.position - previous_position)
            rotation = slerp(previous_rotation, move.rotation, float(s))
            if move.transit:
                bolt_distance = _check_transit_tcp(position, bolt_distance, start_z, args, where)
            else:
                _check_tcp(position, args, where)
            posture = (
                None if move.posture_goal is None
                else move.start_q + float(s) * (move.posture_goal - move.start_q)
            )
            solved = solve_ik(position, rotation, q, tool, posture=posture)
            if solved is None:
                raise ValueError(f"{where}: no IK at {np.array2string(position, precision=3)}")
            max_joint_travel = max(max_joint_travel, float(np.max(np.abs(solved - q))))
            q = solved
            q_path.append(q.copy())
            margin = float(min(np.min(q - POSITION_LOWER), np.min(POSITION_UPPER - q)))
            sigma = float(np.linalg.svd(kin.flange_jacobian(q, tool), compute_uv=False)[-1])
            min_margin, min_sigma = min(min_margin, margin), min(min_sigma, sigma)
            if margin < args.min_joint_margin:
                raise ValueError(f"{where}: {margin:.3f} rad from a joint limit")
            if sigma < args.min_singular_value:
                raise ValueError(f"{where}: near a singularity (sigma {sigma:.3f})")
            # The clip works per Euler angle with a yaw seam. A jump between two
            # close poses would make the controller turn the long way round.
            euler = clip_euler(rotation, args.yaw_seam_deg)
            if np.max(np.abs(euler - previous_euler)) > 0.5:
                raise ValueError(f"{where}: the path crosses the clip's Euler seam")
            previous_euler = euler
        if move.posture_goal is not None and np.max(np.abs(q - move.posture_goal)) > 0.02:
            raise ValueError(
                f"{where}: cannot reach the start posture along the line "
                f"(off by {np.max(np.abs(q - move.posture_goal)):.2f} rad)"
            )
        move.q_nullspace = q.copy()
        move.q_path = np.array(q_path)
        previous_position, previous_rotation = move.position.copy(), move.rotation.copy()
    return {
        "min_joint_margin_rad": min_margin,
        "min_singular_value": min_sigma,
        "max_joint_step_between_path_samples_rad": max_joint_travel,
        "estimated_duration_s": sum(
            m.duration + _dwell(m, args) for m in moves
        ),
    }


def _dwell(move, args) -> float:
    if move.wait_arrival:
        return 0.0
    return args.settle if move.is_arrival else args.waypoint_dwell


def expected_clip_speed_m_s(stiffness: float, clip_m: float) -> float:
    """The steady speed the clipped spring can pull against damping 2 sqrt(k), per axis."""

    return stiffness * clip_m / (2.0 * math.sqrt(stiffness))


# --- analysis -------------------------------------------------------------------
def analyze_arrival(samples: dict, arrival: dict, args) -> dict:
    time_s = samples["time_s"]
    motion = (time_s >= arrival["t_start"]) & (time_s <= arrival["t_end"])
    settled = (time_s >= arrival["t_goal_end"]) & (time_s <= arrival["t_end"])
    window = (time_s >= arrival["t_end"] - args.settle_window) & (time_s <= arrival["t_end"])
    if np.count_nonzero(window) < 2:
        raise ValueError("no controller state in an arrival's settle window")
    goal_position = np.asarray(arrival["goal_position"], dtype=float)
    goal_rotation = np.asarray(arrival["goal_rotation"], dtype=float)
    position_error = samples["position"] - goal_position
    rotation_error = np.array(
        [
            rotation_vector(kin.quaternion_to_matrix(q) @ goal_rotation.T)
            for q in samples["quaternion_xyzw"][motion]
        ]
    )
    rotation_all = np.full((len(time_s), 3), np.nan)
    rotation_all[motion] = rotation_error
    distance = np.linalg.norm(position_error, axis=1)
    angle = np.linalg.norm(rotation_all, axis=1)
    inside = (distance <= args.return_tolerance_mm / 1000.0) & (
        angle <= math.radians(args.return_tolerance_deg)
    )
    time_to_gate = None
    settled_times = time_s[settled]
    outside = np.flatnonzero(~inside[settled])
    if len(settled_times) and inside[settled][-1]:
        first = 0 if len(outside) == 0 else int(outside[-1]) + 1
        time_to_gate = float(settled_times[first] - arrival["t_goal_end"])
    checkpoints = {}
    for offset in ERROR_CHECKPOINTS_S:
        nearest = int(np.argmin(np.abs(time_s - (arrival["t_goal_end"] + offset))))
        checkpoints[f"{offset:g}"] = float(1.0e3 * distance[nearest])
    moving = time_s[motion]
    speed = 0.0
    if len(moving) > 1:
        speed = float(
            np.max(np.linalg.norm(np.diff(samples["position"][motion], axis=0), axis=1)
                   / np.maximum(np.diff(moving), 1.0e-4))
        )
    return {
        **{name: arrival[name] for name in ("position_label", "orientation_label", "approach", "cycle")},
        "position_error_m": position_error[window].mean(axis=0).tolist(),
        "rotation_error_rad": rotation_all[window].mean(axis=0).tolist(),
        "distance_mm_after_goal_end": checkpoints,
        "time_to_return_gate_s": time_to_gate,
        "in_return_gate_at_end": bool(inside[window][-1]),
        "peak_speed_m_s": speed,
        "position_clipped_fraction": float(np.mean(samples["position_clipped"][motion])),
        "rest_tau_task_nm": samples["tau_task"][window].mean(axis=0).tolist(),
        "rest_joint_position_rad": samples["q"][window].mean(axis=0).tolist(),
    }


def iso_repeatability(values) -> float:
    """ISO 9283 pose repeatability: mean distance to the barycentre plus 3 sigma."""

    values = np.asarray(values, dtype=float)
    distance = np.linalg.norm(values - values.mean(axis=0), axis=1)
    return float(distance.mean() + 3.0 * (distance.std(ddof=1) if len(values) > 1 else 0.0))


def summarize(results: list[dict], stiffness: float) -> dict:
    groups = {}
    for result in results:
        groups.setdefault((result["position_label"], result["orientation_label"]), []).append(result)
    targets = {}
    for (position_label, orientation_label), group in sorted(groups.items()):
        position = np.array([r["position_error_m"] for r in group])
        rotation = np.array([r["rotation_error_rad"] for r in group])
        sides = {}
        for side in (-1, 1):
            chosen = [r for r in group if r["approach"] == side]
            if chosen:
                sides[side] = (
                    np.mean([r["position_error_m"] for r in chosen], axis=0),
                    np.mean([r["rotation_error_rad"] for r in chosen], axis=0),
                    iso_repeatability([r["position_error_m"] for r in chosen]),
                )
        entry = {
            "arrivals": len(group),
            "accuracy_mm": float(1.0e3 * np.linalg.norm(position.mean(axis=0))),
            "accuracy_deg": float(np.degrees(np.linalg.norm(rotation.mean(axis=0)))),
            "repeatability_mm": float(1.0e3 * iso_repeatability(position)),
            "repeatability_deg": float(np.degrees(iso_repeatability(rotation))),
            "return_gate_pass_fraction": float(np.mean([r["in_return_gate_at_end"] for r in group])),
            "mean_peak_speed_m_s": float(np.mean([r["peak_speed_m_s"] for r in group])),
        }
        if len(sides) == 2:
            bias = 0.5 * (sides[-1][0] + sides[1][0])
            hysteresis = sides[1][0] - sides[-1][0]
            entry.update(
                {
                    "bias_mm": (1.0e3 * bias).tolist(),
                    "hysteresis_mm": (1.0e3 * hysteresis).tolist(),
                    "hysteresis_norm_mm": float(1.0e3 * np.linalg.norm(hysteresis)),
                    "hysteresis_deg": float(np.degrees(np.linalg.norm(sides[1][1] - sides[-1][1]))),
                    "repeatability_one_side_mm": float(1.0e3 * max(sides[-1][2], sides[1][2])),
                    # At rest the z spring carries whatever weight the model does not.
                    "implied_extra_load_kg": float(-stiffness * bias[2] / GRAVITY),
                }
            )
        targets[f"{position_label}/{orientation_label}"] = entry
    both = [t for t in targets.values() if "bias_mm" in t]
    overall = {
        "arrivals": len(results),
        "return_gate_pass_fraction": float(np.mean([r["in_return_gate_at_end"] for r in results])),
        "peak_speed_m_s": max(r["peak_speed_m_s"] for r in results),
        "accuracy_mm_mean": float(np.mean([t["accuracy_mm"] for t in targets.values()])),
        "repeatability_mm_worst": max(t["repeatability_mm"] for t in targets.values()),
        "repeatability_deg_worst": max(t["repeatability_deg"] for t in targets.values()),
    }
    if both:
        overall["hysteresis_norm_mm_mean"] = float(np.mean([t["hysteresis_norm_mm"] for t in both]))
        overall["implied_extra_load_kg_mean"] = float(np.mean([t["implied_extra_load_kg"] for t in both]))
        overall["implied_extra_load_kg_range"] = [
            min(t["implied_extra_load_kg"] for t in both),
            max(t["implied_extra_load_kg"] for t in both),
        ]
    return {"overall": overall, "targets": targets}


def print_summary(summary: dict, args) -> None:
    print(
        f"\n{'target':<16}{'n':>4}{'acc mm':>8}{'RP mm':>8}{'acc deg':>9}{'RP deg':>8}"
        f"{'gate':>7}{'hyst mm':>9}{'bias x/y/z mm':>24}{'dm kg':>8}"
    )
    for name, t in summary["targets"].items():
        bias = (
            "/".join(f"{value:+.1f}" for value in t["bias_mm"]) if "bias_mm" in t else "-"
        )
        print(
            f"{name:<16}{t['arrivals']:>4}{t['accuracy_mm']:>8.2f}{t['repeatability_mm']:>8.2f}"
            f"{t['accuracy_deg']:>9.2f}{t['repeatability_deg']:>8.2f}"
            f"{100.0 * t['return_gate_pass_fraction']:>6.0f}%"
            f"{t.get('hysteresis_norm_mm', float('nan')):>9.2f}{bias:>24}"
            f"{t.get('implied_extra_load_kg', float('nan')):>8.2f}"
        )
    o = summary["overall"]
    print(
        f"\n{o['arrivals']} arrivals; {100.0 * o['return_gate_pass_fraction']:.0f}% inside the "
        f"{args.return_tolerance_mm:g} mm / {args.return_tolerance_deg:g} deg return gate; "
        f"peak measured speed {o['peak_speed_m_s']:.3f} m/s (goal limit {args.speed:g})."
    )
    print(
        f"Worst ISO 9283 repeatability {o['repeatability_mm_worst']:.2f} mm / "
        f"{o['repeatability_deg_worst']:.2f} deg; mean accuracy {o['accuracy_mm_mean']:.2f} mm."
    )
    if "hysteresis_norm_mm_mean" in o:
        low, high = o["implied_extra_load_kg_range"]
        print(
            f"Friction signature: {o['hysteresis_norm_mm_mean']:.2f} mm mean difference between "
            "approach directions."
        )
        print(
            f"Load signature: the z bias implies {o['implied_extra_load_kg_mean']:+.2f} kg of "
            f"load the model does not carry (range {low:+.2f} to {high:+.2f} over the targets; "
            "a range that moves with orientation points at the centre of mass)."
        )


# --- live -----------------------------------------------------------------------
def _make_node(config, use_sim_time: bool):
    from control_msgs.msg import JointTrajectoryControllerState
    from franka_forge_cartesian_impedance_msgs.msg import CartesianReplayState, PolicyGoal
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data

    class RepeatabilityNode(Node):
        def __init__(self):
            super().__init__(
                "osc_pose_repeatability",
                parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)],
            )
            controller = "/" + config["cartesian"]["controller_name"].strip("/")
            self.joint_names = list(config["joint_names"])
            self.message_type = PolicyGoal
            self.goal_publisher = self.create_publisher(PolicyGoal, controller + "/policy_goal", 1)
            self.lock = threading.Lock()
            self.samples = []
            self.latest = None
            self.q = self.dq = None
            self.create_subscription(
                CartesianReplayState, controller + "/cartesian_state", self.on_state,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointTrajectoryControllerState, controller + "/controller_state", self.on_joints,
                qos_profile_sensor_data,
            )

        def now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1.0e-9

        def sleep_until(self, deadline: float) -> None:
            while self.now_s() < deadline:
                time.sleep(0.001)

        def on_joints(self, message):
            indices = [list(message.joint_names).index(name) for name in self.joint_names]
            with self.lock:
                self.q = np.asarray(message.feedback.positions, dtype=float)[indices]
                self.dq = np.asarray(message.feedback.velocities, dtype=float)[indices]

        def on_state(self, message):
            pose = message.measured
            entry = {
                "time_s": self.now_s(),
                "phase": message.phase,
                "position": (pose.position.x, pose.position.y, pose.position.z),
                "quaternion_xyzw": (
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
                ),
                "position_clipped": message.position_clipped,
                "orientation_clipped": message.orientation_clipped,
                "tau_task": tuple(message.tau_task),
                "tau_nullspace": tuple(message.tau_nullspace),
                "tau_gravity": tuple(message.tau_gravity),
                "tool_in_flange": tuple(message.tool_in_flange),
            }
            with self.lock:
                entry["q"] = None if self.q is None else self.q.copy()
                entry["dq"] = None if self.dq is None else self.dq.copy()
                self.latest = entry
                if entry["q"] is not None:
                    self.samples.append(entry)

        def latest_state(self, timeout_s: float = 10.0) -> dict:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                with self.lock:
                    if self.latest is not None and self.latest["q"] is not None:
                        return dict(self.latest)
                time.sleep(0.01)
            raise TimeoutError("no state from the Cartesian controller")

        def publish_goal(self, position, rotation, tool, nullspace):
            x, y, z, w = kin.matrix_to_quaternion(rotation)
            message = PolicyGoal()
            message.goal.position.x = float(position[0])
            message.goal.position.y = float(position[1])
            message.goal.position.z = float(position[2])
            message.goal.orientation.x = float(x)
            message.goal.orientation.y = float(y)
            message.goal.orientation.z = float(z)
            message.goal.orientation.w = float(w)
            message.tool_in_flange = [float(value) for value in tool]
            message.nullspace_positions = [float(value) for value in nullspace]
            self.goal_publisher.publish(message)

        def arrays(self) -> dict:
            with self.lock:
                samples = list(self.samples)
            return {
                name: np.array([s[name] for s in samples], dtype=float)
                for name in (
                    "time_s", "phase", "position", "quaternion_xyzw", "position_clipped",
                    "orientation_clipped", "tau_task", "tau_nullspace", "tau_gravity", "q", "dq",
                )
            }

    return RepeatabilityNode()


def _check_policy_settings(parameters: dict, args) -> None:
    from franka_trajectory_replay.replay_client import Rejected

    frame = TrainingFrameAdapter()
    expected = {
        "translational_stiffness": float(fo.DEFAULT_TASK_PROP_GAINS[0]),
        "rotational_stiffness": float(fo.DEFAULT_TASK_PROP_GAINS[3]),
        "nullspace_stiffness": fo.KP_NULL,
        "nullspace_damping_lambda": 0.0,
        "stiffness_scale": 1.0,
        "target_filter": 1.0,
        "policy_clip_position_m": float(fo.POS_ACTION_THRESHOLD[0]),
        "policy_clip_orientation_rad": float(fo.ROT_ACTION_THRESHOLD[0]),
        "policy_clip_frame_yaw": frame.world_from_base_yaw,
    }
    wrong = [
        f"{name}={parameters.get(name)!r} (expected {value})"
        for name, value in expected.items()
        if not math.isclose(float(parameters.get(name, float("nan"))), value, rel_tol=1.0e-6)
    ]
    if str(parameters.get("rotation_error")) != "axis_angle":
        wrong.append(f"rotation_error={parameters.get('rotation_error')!r}")
    if parameters.get("mass_weighted_nullspace") is not True:
        wrong.append(f"mass_weighted_nullspace={parameters.get('mass_weighted_nullspace')!r}")
    if parameters.get("coriolis_compensation") is not False:
        wrong.append(f"coriolis_compensation={parameters.get('coriolis_compensation')!r}")
    if wrong:
        raise Rejected("controller is not on the policy profile: " + "; ".join(wrong))


def _check_controller_running(arm_client, rejections_before: int, stream_established: bool) -> None:
    from franka_trajectory_replay.replay_client import Rejected

    status = arm_client.status() or {}
    if int(status.get("rejections", rejections_before)) > rejections_before:
        raise Rejected(f"goal rejected: {status.get('last_rejection')}")
    # A pause between moves trips the watchdog on purpose, and the status lags the
    # first goal after it, so these only mean something once the stream is running.
    if not stream_established:
        return
    if status.get("tracking_fault") == "true":
        raise Rejected(f"tracking fault: {status.get('last_fault')}")
    if status.get("policy_watchdog_stop") == "true":
        raise Rejected("the controller's policy watchdog stopped the goal stream")
    if status.get("phase_name") != "policy":
        raise Rejected(f"controller left the policy phase: {status.get('phase_name')}")


def _interpolate_path(q_path, s: float) -> np.ndarray:
    position = s * (len(q_path) - 1)
    lower = min(int(position), len(q_path) - 2)
    return q_path[lower] + (position - lower) * (q_path[lower + 1] - q_path[lower])


def _arrived(state, move, args) -> bool:
    distance = float(np.linalg.norm(np.asarray(state["position"], dtype=float) - move.position))
    measured = kin.quaternion_to_matrix(state["quaternion_xyzw"])
    angle = float(np.linalg.norm(rotation_vector(measured @ move.rotation.T)))
    return (
        distance <= args.arrival_tolerance_mm / 1000.0
        and angle <= math.radians(args.arrival_tolerance_deg)
    )


def _stream_move(node, arm_client, move, tool, args, rejections) -> tuple[float, float, float]:
    period = 1.0 / args.goal_rate
    dwell = _dwell(move, args)
    started = node.now_s()
    goal_end = started + move.duration
    end = goal_end + dwell
    next_tick = started
    while True:
        now = node.now_s()
        if now >= end:
            if not move.wait_arrival or _arrived(node.latest_state(), move, args):
                return started, goal_end, now
            if now >= end + args.arrival_timeout:
                print(f"  not within the arrival tolerance after {args.arrival_timeout:g} s", flush=True)
                return started, goal_end, now
        s = quintic((now - started) / move.duration)
        node.publish_goal(
            move.start_position + s * (move.position - move.start_position),
            slerp(move.start_rotation, move.rotation, s),
            tool,
            _interpolate_path(move.q_path, s),
        )
        _check_controller_running(arm_client, rejections, now - started > 0.5)
        next_tick += period
        node.sleep_until(next_tick)


def check_posture_move(current_q, start_q, tool, args) -> None:
    """The joint move that finishes the posture: short, and the TCP nearly stands still."""

    current_q, start_q = np.asarray(current_q, dtype=float), np.asarray(start_q, dtype=float)
    largest = float(np.max(np.abs(start_q - current_q)))
    if largest > args.max_start_joint_error:
        raise ValueError(
            f"a joint is {largest:.2f} rad from the start posture "
            f"(limit {args.max_start_joint_error:g}, --max-start-joint-error)"
        )
    end = (kin.flange_transform(start_q) @ tool)[:3, 3]
    for s in np.linspace(0.0, 1.0, 51):
        position = (kin.flange_transform(current_q + s * (start_q - current_q)) @ tool)[:3, 3]
        if position[2] < args.min_tcp_z:
            raise ValueError(f"the posture move takes the TCP to z {position[2]:.3f} m")
        if np.linalg.norm(position - end) > args.max_posture_move_tcp:
            raise ValueError(
                f"the posture move takes the TCP {np.linalg.norm(position - end):.3f} m from the "
                f"start pose (limit {args.max_posture_move_tcp:g}, --max-posture-move-tcp)"
            )


def plan_run(current_q, start_q, a, b, rotations, tool, args):
    """The slow move from the arm's joints to the start posture, then the test from there."""

    transit = build_transit(current_q, a, rotations[0][1], start_q, tool, args.lift)
    transit_report = plan(transit, np.asarray(current_q, dtype=float), tool, args)
    moves = build_moves(a, b, rotations, args.repeats, args.overshoot)
    return transit, moves, transit_report, plan(moves, start_q, tool, args)


def describe_transit(transit, report) -> str:
    lift, line = transit
    distance = float(np.linalg.norm(line.position - line.start_position))
    angle = float(np.linalg.norm(rotation_vector(line.start_rotation.T @ line.rotation)))
    return (
        f"move to start: up {100.0 * np.linalg.norm(lift.position - lift.start_position):.0f} cm, "
        f"then a straight line of {100.0 * distance:.0f} cm while turning {np.degrees(angle):.0f} deg; "
        f"{lift.duration + line.duration:.0f} s, min joint margin "
        f"{report['min_joint_margin_rad']:.2f} rad, min singular value "
        f"{report['min_singular_value']:.3f}"
    )


def run_live(args, start_q, a, b, rotations, tool_matrix, output_dir: Path) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from franka_trajectory_replay.cartesian_replay_client import CartesianReplayClient
    from franka_trajectory_replay.replay_client import Rejected, ReplayClient
    from franka_trajectory_replay.runconfig import load_config

    config = load_config(args.config)
    # Ctrl-C must reach the handlers below with ROS still up, so the abort is sent.
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    joint_client = ReplayClient(config, node_name="osc_repeatability_home")
    arm_client = CartesianReplayClient(config, node_name="osc_repeatability_arm")
    node = _make_node(config, args.sim)
    executor = MultiThreadedExecutor(num_threads=4)
    for member in (joint_client, arm_client, node):
        executor.add_node(member)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    moving = False
    report = {
        "backend": "ros_sim" if args.sim else "hardware",
        "arguments": {k: v for k, v in vars(args).items() if k != "func"},
        "status": "error",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    arrivals = []
    try:
        while node.now_s() <= 0.0:
            time.sleep(0.05)
        arm_client.ensure_active(print)
        arm_client.check_tool(policy_tool_transform(policy_tool_offset_config()), print)
        if args.sim:
            if not arm_client.uses_dh_model():
                raise Rejected("ros-sim expects the simulator profile (model_source=dh)")
        else:
            if arm_client.uses_dh_model():
                raise Rejected("the physical test requires model_source=franka, not dh")
            arm_client.preflight(arm_client.current_joint_positions(), print)
            report["configured_load"] = _configured_load(arm_client, 10.0, print)
        # Same call as the rollout: no goal is active yet, so the controller takes it.
        arm_client.set_parameter_remote("policy_yaw_wrap_upper_deg", args.yaw_seam_deg)
        parameters = arm_client.controller_parameters()
        _check_policy_settings(parameters, args)
        report["controller_parameters"] = {k: _plain(v) for k, v in parameters.items()}
        stiffness = float(parameters["translational_stiffness"])
        clip_speed = expected_clip_speed_m_s(stiffness, float(parameters["policy_clip_position_m"]))
        print(
            f"controller: {stiffness:g} N/m, {float(parameters['rotational_stiffness']):g} Nm/rad; "
            f"the {1.0e3 * float(parameters['policy_clip_position_m']):g} mm clip allows about "
            f"{clip_speed:.2f} m/s per axis against its damping",
            flush=True,
        )
        tool_position = np.asarray(POLICY_TOOL_OFFSET_XYZ, dtype=float)
        tool = [*tool_position, 0.0, 0.0, 0.0, 1.0]

        state = node.latest_state()
        measured = np.asarray(state["position"], dtype=float)
        current_q = np.asarray(state["q"], dtype=float)
        modelled = kin.flange_transform(current_q) @ tool_matrix
        offset = float(np.linalg.norm(measured - modelled[:3, 3]))
        print(f"the controller's EE point is {1.0e3 * offset:.1f} mm from the planner's", flush=True)
        if offset > args.max_start_offset:
            raise Rejected(
                f"measured EE point {1.0e3 * offset:.1f} mm from the planner's "
                f"(limit {1.0e3 * args.max_start_offset:.0f} mm): tool or kinematics mismatch"
            )
        transit, moves, transit_report, plan_report = plan_run(
            current_q, start_q, a, b, rotations, tool_matrix, args
        )
        report["transit_plan"] = transit_report
        report["plan"] = plan_report
        print(describe_transit(transit, transit_report), flush=True)
        if _distance_to_bolt_m(measured) < args.bolt_keepout:
            message = "the hand starts at the bolt: open the fingers if they are around the nut"
            if args.yes:
                raise Rejected(message + " (refused under --yes)")
            print(f"WARNING: {message}.", flush=True)
        if not args.yes:
            input("Make this move? Keep the E-stop ready. [Enter/Ctrl-C] ")
        rejections = int((arm_client.status() or {}).get("rejections", 0))
        moving = True
        for move in transit:
            _stream_move(node, arm_client, move, tool, args, rejections)
        state = node.latest_state()
        error = np.asarray(state["position"], dtype=float) - transit[-1].position
        print(f"at the start pose, {1.0e3 * np.linalg.norm(error):.1f} mm from it", flush=True)
        # The nullspace spring is too soft to bring the elbow in against friction
        # (2026-11-10: 0.71 rad short), so the validated joint controller finishes it.
        current_q = np.asarray(state["q"], dtype=float)
        check_posture_move(current_q, start_q, tool_matrix, args)
        print(
            f"posture move: largest joint change {np.max(np.abs(start_q - current_q)):.2f} rad, "
            "with the joint controller",
            flush=True,
        )
        if not args.yes:
            input("Make this move? Keep the E-stop ready. [Enter/Ctrl-C] ")
        joint_client.ensure_active(print)
        joint_client.goto(start_q)
        # The Cartesian controller's joint feedback stopped with it: nothing cached is current.
        with node.lock:
            node.latest = node.q = node.dq = None
        arm_client.ensure_active(print)
        arm_client.set_parameter_remote("policy_yaw_wrap_upper_deg", args.yaw_seam_deg)
        state = node.latest_state()
        residual = float(np.max(np.abs(np.asarray(state["q"], dtype=float) - start_q)))
        print(f"back on the Cartesian controller, joints within {residual:.3f} rad", flush=True)
        if residual > MAX_START_POSTURE_RESIDUAL_RAD:
            raise Rejected(f"the joint move left a joint {residual:.3f} rad from the start posture")
        rejections = int((arm_client.status() or {}).get("rejections", 0))
        if not args.yes:
            input(
                f"Start {len([m for m in moves if m.is_arrival])} arrivals, about "
                f"{plan_report['estimated_duration_s'] / 60.0:.1f} min? [Enter/Ctrl-C] "
            )
        state = node.latest_state()
        # Start from where the arm really is, not from the plan.
        moves[0].start_position = np.asarray(state["position"], dtype=float)
        moves[0].start_rotation = kin.quaternion_to_matrix(state["quaternion_xyzw"])
        total = sum(1 for m in moves if m.is_arrival)
        for move in moves:
            started, goal_end, end = _stream_move(node, arm_client, move, tool, args, rejections)
            if move.is_arrival:
                arrivals.append(
                    {
                        "position_label": move.position_label,
                        "orientation_label": move.orientation_label,
                        "approach": move.approach,
                        "cycle": move.cycle,
                        "goal_position": move.position.tolist(),
                        "goal_rotation": move.rotation.tolist(),
                        "t_start": started,
                        "t_goal_end": goal_end,
                        "t_end": end,
                    }
                )
                state = node.latest_state()
                error = np.asarray(state["position"], dtype=float) - move.position
                print(
                    f"[{len(arrivals):3d}/{total}] {move.position_label}/{move.orientation_label:<8}"
                    f" approach {move.approach:+d}: {1.0e3 * np.linalg.norm(error):6.2f} mm",
                    flush=True,
                )
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("interrupted: stopping the arm", flush=True)
    except Exception as exc:  # noqa: BLE001 - recorded, and the arm is stopped below
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"stopping: {report['error']}", flush=True)
    finally:
        if moving:
            try:
                arm_client.abort()
            except Exception as exc:  # noqa: BLE001 - the data below must still be saved
                print(f"abort failed: {exc}", flush=True)
        samples = node.arrays() if node.samples else None
        if samples is not None:
            np.savez_compressed(output_dir / "samples.npz", **samples)
        if samples is not None and arrivals:
            results = [analyze_arrival(samples, arrival, args) for arrival in arrivals]
            stiffness = float(report.get("controller_parameters", {}).get("translational_stiffness", 565.0))
            summary = summarize(results, stiffness)
            report["arrivals"] = results
            report["summary"] = summary
            print_summary(summary, args)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=_plain) + "\n")
        print(f"\nwrote {output_dir}", flush=True)
        executor.shutdown()
        spin.join(timeout=5.0)
        rclpy.try_shutdown()
    if report["status"] == "error":
        raise SystemExit(1)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def default_orientation() -> np.ndarray:
    """The flange pointing straight down, as in the FR3's ready posture."""

    return kin.flange_transform(kin.READY_POSE)[:3, :3]


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("backend", choices=("hardware", "ros-sim"))
    parser.add_argument("--plan-only", action="store_true", help="check the path; no ROS, no motion")
    parser.add_argument("--yes", action="store_true", help="skip the confirmations")
    parser.add_argument("--pose-a", type=float, nargs=3, default=(0.30, 0.20, 0.30), metavar=("X", "Y", "Z"))
    parser.add_argument("--pose-b", type=float, nargs=3, default=(0.30, 0.40, 0.30), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--orientation-deg", type=float, nargs=3, action="append", metavar=("RX", "RY", "RZ"),
        help="rotation vector in degrees, base frame, on top of flange-down; repeat for several",
    )
    parser.add_argument("--task-profile", default="auto", help="sets the clip's yaw seam, as in the rollout")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--speed", type=float, default=0.5, help="peak goal speed, m/s")
    parser.add_argument("--acceleration", type=float, default=2.5, help="peak goal acceleration, m/s^2")
    parser.add_argument("--angular-speed", type=float, default=1.5, help="peak goal rate, rad/s")
    parser.add_argument("--angular-acceleration", type=float, default=8.0, help="rad/s^2")
    parser.add_argument("--transit-speed", type=float, default=0.05, help="to the start pose, m/s")
    parser.add_argument("--transit-angular-speed", type=float, default=0.10, help="rad/s")
    parser.add_argument("--arrival-timeout", type=float, default=15.0, help="s")
    parser.add_argument("--arrival-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--arrival-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--max-posture-move-tcp", type=float, default=0.10, help="m")
    parser.add_argument("--lift", type=float, default=0.10, help="straight up before leaving, m")
    parser.add_argument(
        "--start-joints", type=float, nargs=7, metavar="Q",
        help="with --plan-only: also check the move to the start pose from these joints",
    )
    parser.add_argument("--overshoot", type=float, default=0.05, help="waypoint beyond a pose, m")
    parser.add_argument("--settle", type=float, default=2.0, help="hold after the goal stops, s")
    parser.add_argument("--settle-window", type=float, default=0.5, help="averaged for the rest pose, s")
    parser.add_argument("--waypoint-dwell", type=float, default=0.15)
    parser.add_argument("--goal-rate", type=float, default=50.0, help="Hz")
    parser.add_argument("--return-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--return-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--min-tcp-z", type=float, default=0.15)
    parser.add_argument("--min-tcp-radius", type=float, default=0.20)
    parser.add_argument("--bolt-keepout", type=float, default=0.15)
    parser.add_argument(
        "--bolt-clear-z", type=float, default=0.30,
        help="the move to the start may pass over the bolt only above this TCP height, m",
    )
    parser.add_argument("--min-joint-margin", type=float, default=0.10)
    parser.add_argument("--min-singular-value", type=float, default=0.03)
    parser.add_argument("--max-start-offset", type=float, default=0.02)
    parser.add_argument("--max-start-joint-error", type=float, default=1.0, help="rad")
    parser.add_argument("--yaw-seam-deg", type=float, default=None, help="default: the task profile's")
    parser.add_argument(
        "--config",
        default=str(WORKSPACE_ROOT / "src/inspire_franka_trajectory_replay/config/replay.yaml"),
    )
    parser.add_argument("--output-root", default=str(WORKSPACE_ROOT / "debug/osc_repeatability"))
    return parser


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    args.sim = args.backend == "ros-sim"
    if not 0.0 < args.speed <= 0.6:
        raise SystemExit("--speed must be in (0, 0.6] m/s")
    if args.settle <= args.settle_window:
        raise SystemExit("--settle must be longer than --settle-window")
    profile = fo.resolve_task_profile(args.task_profile, DEFAULT_CHECKPOINT)
    if args.yaw_seam_deg is None:
        args.yaw_seam_deg = float(profile.yaw_wrap_upper_deg)

    tool = policy_tool_transform(policy_tool_offset_config())
    nominal = default_orientation()
    offsets = args.orientation_deg or DEFAULT_ORIENTATIONS_DEG
    rotations = [
        (f"o{i}", rotation_from_vector(np.radians(offset)) @ nominal)
        for i, offset in enumerate(offsets)
    ]
    a = np.asarray(args.pose_a, dtype=float)
    b = np.asarray(args.pose_b, dtype=float)
    print(
        f"poses {float(np.linalg.norm(b - a)) * 100.0:.0f} cm apart; {len(rotations)} orientations; "
        f"{args.repeats * len(rotations) * 4} arrivals"
    )
    # One fixed arm posture for the test: the elbow on the side of the targets.
    seed = kin.READY_POSE.copy()
    seed[0] = math.atan2(a[1], a[0])
    start_q = solve_ik(a, rotations[0][1], seed, tool)
    if start_q is None:
        raise SystemExit(f"no IK for pose A at {a}")
    print(f"start posture: {np.array2string(start_q, precision=3)}")
    if args.plan_only:
        try:
            _plan_only(start_q, a, b, rotations, tool, args)
        except ValueError as exc:
            raise SystemExit(f"the planned path is refused: {exc}") from exc
        return
    if not 0.0 < args.transit_speed <= 0.15:
        raise SystemExit("--transit-speed must be in (0, 0.15] m/s")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_live(args, start_q, a, b, rotations, tool, Path(args.output_root) / f"{args.backend}-{stamp}")


def _plan_only(start_q, a, b, rotations, tool, args) -> None:
    if args.start_joints is not None:
        transit, _, transit_report, report = plan_run(
            np.asarray(args.start_joints, dtype=float), start_q, a, b, rotations, tool, args
        )
        print(describe_transit(transit, transit_report))
    else:
        report = plan(build_moves(a, b, rotations, args.repeats, args.overshoot), start_q, tool, args)
    print(
        f"test path: min joint margin {report['min_joint_margin_rad']:.2f} rad, "
        f"min singular value {report['min_singular_value']:.3f}, "
        f"about {report['estimated_duration_s'] / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()
