#!/usr/bin/env python3
"""OSC characterization, part A: dead-zone probe, static payload holds, wrist spin.

Design: docs/osc_characterization_test_design.md. One launch fixes friction
compensation (controllers_policy.yaml or controllers_policy_no_friction_comp.yaml);
this script then runs, for each payload level:

* **static holds** at three seeded random orientations (joint controller, so the
  joints sit exactly on the planned posture). On hardware, Franka's
  ``tau_ext_hat_filtered`` at rest is link-side and free of gearbox friction, so
  it carries the payload model error.
* **dead-zone probe** at the same poses (deployment Cartesian controller through
  ``~/policy_goal``): the goal walks slowly along one base axis at a time in a
  triangle, and the spring wrench (-K x error) at which the arm starts to follow
  is the dead zone in N / N m. Per-goal clip 0.05 m / 0.25 rad so the clip never
  caps the spring before the arm breaks away.
* **wrist spin**: flange axis horizontal, j7 stepped through its range with a
  hold at each stop, which isolates the hand's centre of mass from arm model error.

The payload is switched with ``service_server/set_load`` on top of a zero-mass
copy of the Desk end-effector profile (same identity flange-to-EE transform).
Everything is recorded with ``ros2 bag record`` (MCAP); the event markers are in
the bag and in ``report.json``. Analyse with ``osc_characterization_analysis.py``.

Run in the container with the workspace sourced, after
``ros2 launch inspire_franka_trajectory_replay replay.launch.py arm_controller:=policy``
(hardware) or the ros-sim policy launch::

    python3 apps/policy_rollout/utils/osc_characterization.py hardware --plan-only
    python3 apps/policy_rollout/utils/osc_characterization.py hardware --payload modelled none
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
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
    policy_tool_offset_config,
    policy_tool_transform,
)
from utils import osc_pose_repeatability as rep  # noqa: E402

SCENE = WORKSPACE_ROOT / "src/inspire_franka_sim/mjcf/inspire_franka_policy_scene_windback45_tilt30.xml"
# Desk values on 2026-11-10 (robot_state inertia_ee): hand + 40 g cable, flange frame.
DESK_HAND_MASS = 0.5789
DESK_HAND_COM = (0.00358, -0.00413, 0.09805)
DESK_HAND_INERTIA = (0.000408, 0.000538, 0.000197)
# Flange pointing horizontally outward over the table, 0.78 m up; checked against the
# MuJoCo scene for j7 in [-2.7, 2.7] (hand 85 mm from the arm at the closest).
SPIN_POSTURE = (0.0, -0.3, 0.0, -2.0, 0.0, 3.3)
# Links whose geoms never move relative to the scene; leaving them out keeps the
# clearance number about the moving arm.
FIXED_LINKS = ("fr3_link0", "fr3_link1")
AXIS_NAMES = ("x", "y", "z", "rx", "ry", "rz")
EVENT_TOPIC = "/osc_characterization/events"


# --- scene clearance ------------------------------------------------------------
class SceneClearance:
    """Minimum distance from the arm and hand to the MuJoCo policy scene."""

    def __init__(self, path=SCENE):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("policy_home").id)
        names = [self.model.body(i).name for i in range(self.model.nbody)]
        root = names.index("fr3_link0")
        hand = names.index("hand_mount")

        def under(body, ancestor):
            while body:
                if body == ancestor:
                    return True
                body = self.model.body_parentid[body]
            return False

        collides = [
            g for g in range(self.model.ngeom)
            if self.model.geom_contype[g] or self.model.geom_conaffinity[g]
        ]
        robot = [g for g in collides if under(self.model.geom_bodyid[g], root)]
        self.environment = [g for g in collides if g not in robot]
        self.moving = [g for g in robot if names[self.model.geom_bodyid[g]] not in FIXED_LINKS]
        self.hand = [g for g in robot if under(self.model.geom_bodyid[g], hand)]
        upper_arm = {f"fr3_link{i}" for i in range(6)}
        self.arm = [g for g in robot if names[self.model.geom_bodyid[g]] in upper_arm]
        self._fromto = np.zeros(6)

    def _distance(self, first, second, limit=0.5):
        best, pair = limit, None
        for a in first:
            for b in second:
                value = self.mujoco.mj_geomDistance(self.model, self.data, a, b, limit, self._fromto)
                if value < best:
                    best, pair = value, (a, b)
        return best, pair

    def check(self, q) -> tuple[float, str, float]:
        """(environment clearance, closest environment geom, hand-to-arm clearance)."""

        self.data.qpos[:7] = np.asarray(q, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)
        environment, pair = self._distance(self.moving, self.environment)
        own, _ = self._distance(self.hand, self.arm)
        name = "-" if pair is None else self.model.geom(pair[1]).name
        return float(environment), name, float(own)


# --- planning -------------------------------------------------------------------
def tilted_orientation(tilt_axis_angle, tilt, yaw) -> np.ndarray:
    axis = np.array([math.cos(tilt_axis_angle), math.sin(tilt_axis_angle), 0.0])
    return (
        rep.rotation_from_vector(tilt * axis)
        @ rep.rotation_from_vector([0.0, 0.0, yaw])
        @ rep.default_orientation()
    )


def probe_offsets(args) -> list[tuple[np.ndarray, np.ndarray]]:
    """The extremes of the probe: (translation, rotation vector) per axis and sign."""

    amplitude = [args.probe_translation_mm / 1000.0] * 3 + [math.radians(args.probe_rotation_deg)] * 3
    out = []
    for axis in range(6):
        for sign in (-1.0, 1.0):
            offset = np.zeros(6)
            offset[axis] = sign * amplitude[axis]
            out.append((offset[:3], offset[3:]))
    return out


def offset_pose(position, rotation, translation, rotation_vector):
    return (
        np.asarray(position, dtype=float) + translation,
        rep.rotation_from_vector(rotation_vector) @ rotation,
    )


def check_posture(q, tool, scene, args, where: str) -> dict:
    margin = float(min(np.min(q - POSITION_LOWER), np.min(POSITION_UPPER - q)))
    sigma = float(np.linalg.svd(kin.flange_jacobian(q, tool), compute_uv=False)[-1])
    environment, closest, own = scene.check(q)
    if margin < args.min_joint_margin:
        raise ValueError(f"{where}: {margin:.3f} rad from a joint limit")
    if sigma < args.min_singular_value:
        raise ValueError(f"{where}: near a singularity (sigma {sigma:.3f})")
    if environment < args.clearance:
        raise ValueError(f"{where}: {1000 * environment:.0f} mm from {closest} in the MuJoCo scene")
    if own < args.self_clearance:
        raise ValueError(f"{where}: hand {1000 * own:.0f} mm from the arm")
    return {"joint_margin_rad": margin, "sigma": sigma, "clearance_m": environment,
            "closest": closest, "self_clearance_m": own}


def check_joint_path(q_from, q_to, scene, args, where: str, leaving: bool = False) -> float:
    """Clearance along the straight joint-space line the joint controller drives.

    ``leaving``: the line may start inside the clearance (the arm parked at the nut
    after a rollout), as long as it only moves away and stays above
    ``--min-start-clearance`` until it is out.
    """

    q_from, q_to = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    steps = max(2, int(math.ceil(np.max(np.abs(q_to - q_from)) / 0.02)))
    lowest, previous = math.inf, -math.inf
    for s in np.linspace(0.0, 1.0, steps + 1):
        q = q_from + s * (q_to - q_from)
        environment, closest, own = scene.check(q)
        lowest = min(lowest, environment)
        if own < args.self_clearance:
            raise ValueError(f"{where}: hand {1000 * own:.0f} mm from the arm at s = {s:.2f}")
        if leaving and environment < args.clearance:
            if environment < args.min_start_clearance or environment < previous - 1.0e-3:
                raise ValueError(
                    f"{where}: starting {1000 * environment:.0f} mm from {closest}, the joint path "
                    f"does not move away from it (s = {s:.2f})"
                )
            previous = environment
            continue
        leaving = False
        if environment < args.clearance:
            raise ValueError(
                f"{where}: the joint path comes {1000 * environment:.0f} mm from {closest} "
                f"(hand to arm {1000 * own:.0f} mm) at s = {s:.2f}"
            )
    return lowest


def plan_start(current, q_first, scene, args) -> tuple[list[np.ndarray], str]:
    """Joint waypoints from wherever the arm is to the first pose.

    Straight to it if that line is clean; otherwise first lift the flange straight up
    with its orientation held, which is how the arm leaves the nut.
    """

    current = np.asarray(current, dtype=float)
    try:
        lowest = check_joint_path(current, q_first, scene, args, "move to the first pose", leaving=True)
        return [q_first], f"joint-space line, min clearance {1000 * lowest:.0f} mm once clear"
    except ValueError as direct:
        flange = kin.flange_transform(current)
        lifted = solve_lift(current, flange, args.lift)
        if lifted is None:
            raise ValueError(f"{direct}; and no IK for a {100 * args.lift:.0f} cm lift") from direct
        check_joint_path(current, lifted, scene, args, "lift", leaving=True)
        lowest = check_joint_path(lifted, q_first, scene, args, "lift -> first pose")
        return [lifted, q_first], (
            f"lift {100 * args.lift:.0f} cm, then a joint-space line (min clearance "
            f"{1000 * lowest:.0f} mm); the direct line was refused: {direct}"
        )


def read_robot_joints(timeout_s: float = 10.0) -> np.ndarray:
    """The arm's joints from /joint_states, once (for --plan-only --start-from-robot)."""

    import rclpy
    from sensor_msgs.msg import JointState

    names = [f"fr3_joint{i}" for i in range(1, 8)]
    rclpy.init()
    node = rclpy.create_node("osc_characterization_joints_once")
    latest = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.update(zip(m.name, m.position)), 10)
    deadline = time.monotonic() + timeout_s
    try:
        while not all(n in latest for n in names):
            if time.monotonic() > deadline:
                raise SystemExit("no fr3 joints on /joint_states: is the launch running, and is "
                                 "ROS_DOMAIN_ID the launch's?")
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return np.array([latest[n] for n in names])


def solve_lift(current, flange, lift: float):
    target = flange[:3, 3] + np.array([0.0, 0.0, lift])
    return rep.solve_ik(target, flange[:3, :3], current, np.eye(4))


def plan_probe_pose(position, rotation, seed_q, tool, scene, args, where: str) -> dict:
    q = rep.solve_ik(position, rotation, seed_q, tool)
    if q is None:
        raise ValueError(f"{where}: no IK")
    report = check_posture(q, tool, scene, args, where)
    yaw_deg = math.degrees(rep.clip_euler(rotation, args.yaw_seam_deg)[2])
    seam_margin = (args.yaw_seam_deg - yaw_deg) % 360.0
    seam_margin = min(seam_margin, 360.0 - seam_margin)
    if seam_margin < args.probe_rotation_deg + 10.0:
        raise ValueError(f"{where}: yaw is {seam_margin:.0f} deg from the clip's seam")
    extremes = []
    for translation, rotation_vector in probe_offsets(args):
        p, r = offset_pose(position, rotation, translation, rotation_vector)
        q_extreme = rep.solve_ik(p, r, q, tool)
        if q_extreme is None:
            raise ValueError(f"{where}: no IK at a probe extreme")
        extremes.append(check_posture(q_extreme, tool, scene, args, where + " probe extreme"))
    return {
        "q": q,
        "posture": report,
        "probe_min_clearance_m": min(e["clearance_m"] for e in extremes),
        "probe_min_joint_margin_rad": min(e["joint_margin_rad"] for e in extremes),
        "seam_margin_deg": seam_margin,
    }


def sample_poses(args, tool, scene) -> list[dict]:
    """Seeded random orientations that pass every check, and differ enough for the payload fit."""

    rng = np.random.default_rng(args.seed)
    position = np.asarray(args.pose, dtype=float)
    seed_q = kin.READY_POSE.copy()
    seed_q[0] = math.atan2(position[1], position[0])
    poses, rejected = [], 0
    while len(poses) < args.orientations:
        if rejected > args.max_draws:
            raise ValueError(f"only {len(poses)} of {args.orientations} orientations after "
                             f"{args.max_draws} draws; widen the limits or change --seed")
        draw = {
            "tilt_axis_deg": float(rng.uniform(0.0, 360.0)),
            "tilt_deg": float(rng.uniform(args.min_tilt_deg, args.max_tilt_deg)),
            "yaw_deg": float(rng.uniform(-args.max_yaw_deg, args.max_yaw_deg)),
        }
        rotation = tilted_orientation(
            math.radians(draw["tilt_axis_deg"]), math.radians(draw["tilt_deg"]),
            math.radians(draw["yaw_deg"]),
        )
        # Gravity seen in the flange frame: the payload fit needs it to point different ways.
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        if any(
            math.degrees(math.acos(np.clip(gravity @ p["gravity_in_flange"], -1.0, 1.0)))
            < args.min_gravity_separation_deg
            for p in poses
        ):
            rejected += 1
            continue
        label = f"o{len(poses)}"
        previous = seed_q if not poses else poses[-1]["q"]
        try:
            planned = plan_probe_pose(position, rotation, previous, tool, scene, args, label)
            if poses:
                check_joint_path(poses[-1]["q"], planned["q"], scene, args, f"{poses[-1]['label']} -> {label}")
        except ValueError:
            rejected += 1
            continue
        poses.append({"label": label, "draw": draw, "position": position, "rotation": rotation,
                      "gravity_in_flange": gravity, **planned})
    return poses


def spin_stops(args) -> list[np.ndarray]:
    angles = np.linspace(-args.spin_range_rad, args.spin_range_rad, args.spin_stops)
    return [np.array([*SPIN_POSTURE, float(a)]) for a in angles]


def plan_spin(last_q, scene, args) -> dict:
    stops = spin_stops(args)
    reports = [check_posture(q, None, scene, args, f"spin j7 {q[6]:+.2f}") for q in stops]
    lowest = check_joint_path(last_q, stops[0], scene, args, "move to the spin posture")
    for a, b in zip(stops[:-1], stops[1:]):
        lowest = min(lowest, check_joint_path(a, b, scene, args, "spin"))
    return {"stops": stops, "min_clearance_m": min(lowest, min(r["clearance_m"] for r in reports))}


def chunks(q_from, q_to, max_step: float) -> list[np.ndarray]:
    """Waypoints so that no single joint goto exceeds ``max_step`` on any joint."""

    q_from, q_to = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    count = max(1, int(math.ceil(np.max(np.abs(q_to - q_from)) / max_step)))
    return [q_from + (i / count) * (q_to - q_from) for i in range(1, count + 1)]


def probe_profile(args) -> list[tuple[str, float, float]]:
    """Legs of the triangle as (name, start fraction, end fraction) of the amplitude."""

    legs = [("out_plus", 0.0, 1.0), ("plus_to_minus", 1.0, -1.0), ("minus_to_zero", -1.0, 0.0)]
    return legs * args.probe_cycles


# --- live -----------------------------------------------------------------------
def _make_node(config, use_sim_time: bool):
    from franka_forge_cartesian_impedance_msgs.msg import CartesianReplayState, PolicyGoal
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String

    class CharacterizationNode(Node):
        def __init__(self):
            super().__init__(
                "osc_characterization",
                parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)],
            )
            controller = "/" + config["cartesian"]["controller_name"].strip("/")
            self.goal_publisher = self.create_publisher(PolicyGoal, controller + "/policy_goal", 1)
            self.event_publisher = self.create_publisher(String, EVENT_TOPIC, 10)
            self.lock = threading.Lock()
            self.latest = None
            self.events = []
            self.create_subscription(
                CartesianReplayState, controller + "/cartesian_state", self.on_state,
                qos_profile_sensor_data,
            )

        def now_s(self) -> float:
            return self.get_clock().now().nanoseconds * 1.0e-9

        def sleep_until(self, deadline: float) -> None:
            while self.now_s() < deadline:
                time.sleep(0.001)

        def on_state(self, message):
            pose = message.measured
            with self.lock:
                self.latest = {
                    "time_s": self.now_s(),
                    "phase": message.phase,
                    "position": np.array([pose.position.x, pose.position.y, pose.position.z]),
                    "quaternion_xyzw": (pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z, pose.orientation.w),
                }

        def latest_state(self, timeout_s: float = 10.0) -> dict:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                with self.lock:
                    if self.latest is not None and self.now_s() - self.latest["time_s"] < 1.0:
                        return dict(self.latest)
                time.sleep(0.01)
            raise TimeoutError("no current state from the Cartesian controller")

        def forget_state(self):
            with self.lock:
                self.latest = None

        def event(self, name: str, **fields) -> dict:
            entry = {"event": name, "t": self.now_s(), **{k: rep._plain(v) for k, v in fields.items()}}
            self.events.append(entry)
            self.event_publisher.publish(String(data=json.dumps(entry)))
            print(f"  [{name}] " + " ".join(f"{k}={v}" for k, v in fields.items() if k != "q"),
                  flush=True)
            return entry

        def publish_goal(self, position, rotation, tool, nullspace, args):
            x, y, z, w = kin.matrix_to_quaternion(rotation)
            message = PolicyGoal()
            message.goal.position.x, message.goal.position.y, message.goal.position.z = (
                float(position[0]), float(position[1]), float(position[2]))
            message.goal.orientation.x, message.goal.orientation.y = float(x), float(y)
            message.goal.orientation.z, message.goal.orientation.w = float(z), float(w)
            message.tool_in_flange = [float(v) for v in tool]
            message.nullspace_positions = [float(v) for v in nullspace]
            message.clip_position_m = float(args.probe_clip[0])
            message.clip_orientation_rad = float(args.probe_clip[1])
            message.max_position_error = float(args.probe_max_error[0])
            message.max_orientation_error = float(args.probe_max_error[1])
            self.goal_publisher.publish(message)

    return CharacterizationNode()


class BagRecorder:
    """``ros2 bag record`` in its own process group, stopped with SIGINT."""

    def __init__(self, output: Path, topics: list[str]):
        self.output, self.topics, self.process = output, topics, None

    def start(self):
        # The recorder's keyboard controls read the terminal; with them on, it swallows the
        # operator's answers and the script's next prompt gets EOF (2026-11-11).
        command = ["ros2", "bag", "record", "-s", "mcap", "--storage-preset-profile", "zstd_fast",
                   "--disable-keyboard-controls", "-o", str(self.output), *self.topics]
        self.process = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(2.0)
        if self.process.poll() is not None:
            raise RuntimeError(f"ros2 bag record exited at once ({self.process.returncode})")
        print(f"recording {len(self.topics)} topics to {self.output}", flush=True)

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, 15.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
            os.killpg(self.process.pid, sig)
            try:
                self.process.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                print(f"bag recorder still running after {sig.name}", flush=True)


def deactivate_arm_controllers(client, log=print):
    """Leave nothing commanding the arm, so libfranka takes configuration commands."""

    from controller_manager_msgs.srv import SwitchController
    from rclpy.duration import Duration

    joints = set(client.config["joint_names"])
    to_stop = [
        name for name, info in client.list_controllers().items()
        if info.state == "active" and "broadcaster" not in info.type.lower()
        and {i.split("/")[0] for i in info.claimed_interfaces} & joints
    ]
    if not to_stop:
        return []
    request = SwitchController.Request()
    request.deactivate_controllers = to_stop
    request.strictness = SwitchController.Request.STRICT
    request.activate_asap = True
    request.timeout = Duration(seconds=10.0).to_msg()
    response = client.call(SwitchController, client.manager_ns + "/switch_controller", request, 30.0)
    if not response.ok:
        raise RuntimeError(f"switch_controller refused to deactivate {to_stop}")
    log(f"deactivated {', '.join(to_stop)}; the robot is idle")
    return to_stop


def load_readback(arm_client) -> dict:
    state = arm_client.robot_state_once(10.0)

    def inertia(stamped):
        i = stamped.inertia
        return {"m": float(i.m), "com": [float(i.com.x), float(i.com.y), float(i.com.z)],
                "ixx_iyy_izz": [float(i.ixx), float(i.iyy), float(i.izz)]}

    pose = state.f_t_ee.pose
    return {
        "ee": inertia(state.inertia_ee),
        "load": inertia(state.inertia_load),
        "total": inertia(state.inertia_total),
        "f_t_ee_position": [pose.position.x, pose.position.y, pose.position.z],
        "f_t_ee_quaternion_xyzw": [pose.orientation.x, pose.orientation.y,
                                   pose.orientation.z, pose.orientation.w],
    }


def check_desk_profile(readback, args) -> None:
    from franka_trajectory_replay.replay_client import Rejected

    position = np.asarray(readback["f_t_ee_position"])
    quaternion = np.abs(np.asarray(readback["f_t_ee_quaternion_xyzw"]))
    if np.linalg.norm(position) > 1.0e-4 or abs(quaternion[3] - 1.0) > 1.0e-4:
        raise Rejected(f"Desk flange-to-EE transform is not identity: {readback}")
    if abs(readback["ee"]["m"]) > 1.0e-3:
        raise Rejected(
            f"the Desk end-effector profile carries {readback['ee']['m']:.3f} kg. Activate the "
            "zero-mass copy of the hand profile for this test (or pass --payload modelled "
            "--no-set-load to keep the normal profile)."
        )


def set_payload(level, joint_client, arm_client, args, log=print) -> dict:
    """Apply a payload level with nothing commanding the arm, then read it back."""

    from franka_msgs.srv import SetLoad
    from franka_trajectory_replay.replay_client import Rejected

    mass, com, inertia = (
        (args.hand_mass, args.hand_com, args.hand_inertia) if level == "modelled"
        else (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    request = SetLoad.Request()
    request.mass = float(mass)
    request.center_of_mass = [float(v) for v in com]
    matrix = np.diag(inertia)
    request.load_inertia = [float(v) for v in matrix.flatten(order="F")]
    deactivate_arm_controllers(joint_client, log)
    try:
        response = joint_client.call(SetLoad, args.set_load_service, request, 15.0)
        if not response.success:
            raise Rejected(f"set_load refused: {response.error}")
        time.sleep(0.5)
        readback = load_readback(arm_client)
    finally:
        joint_client.ensure_active(log)
    if abs(readback["load"]["m"] - mass) > 1.0e-3 or np.max(
        np.abs(np.asarray(readback["load"]["com"]) - np.asarray(com))
    ) > 1.0e-4:
        raise Rejected(f"set_load read back {readback['load']}, asked for {mass} kg at {com}")
    log(f"payload '{level}': load {readback['load']['m']:.3f} kg, total "
        f"{readback['total']['m']:.3f} kg", flush=True)
    return readback


def joint_move(joint_client, node, q_target, args, label: str) -> None:
    current = np.asarray(joint_client.current_joint_positions(), dtype=float)
    for waypoint in chunks(current, q_target, args.max_goto_step):
        joint_client.goto(waypoint)
    node.event("joint_move_done", label=label, q=q_target)


def static_hold(joint_client, node, q, args, label: str, payload: str) -> None:
    node.event("static_start", label=label, payload=payload, q=q)
    node.sleep_until(node.now_s() + args.hold_s)
    reached = np.asarray(joint_client.current_joint_positions(), dtype=float)
    node.event("static_end", label=label, payload=payload, q=reached,
               max_joint_error_rad=float(np.max(np.abs(reached - q))))


def run_probe(node, arm_client, pose, tool, args, rejections) -> None:
    """The triangle along each base axis, streamed through ~/policy_goal."""

    state = node.latest_state()
    position0 = np.asarray(state["position"], dtype=float)
    rotation0 = kin.quaternion_to_matrix(state["quaternion_xyzw"])
    period = 1.0 / args.goal_rate
    amplitude = [args.probe_translation_mm / 1000.0] * 3 + [math.radians(args.probe_rotation_deg)] * 3
    speed = [args.probe_speed_mm_s / 1000.0] * 3 + [math.radians(args.probe_rate_deg_s)] * 3

    first_goal = node.now_s()

    def established() -> bool:
        # The status lags the first goal after activation.
        return node.now_s() - first_goal > 0.5

    def hold(seconds):
        end = node.now_s() + seconds
        tick = node.now_s()
        while node.now_s() < end:
            node.publish_goal(position0, rotation0, tool, pose["q"], args)
            rep._check_controller_running(arm_client, rejections, established())
            tick += period
            node.sleep_until(tick)

    hold(args.probe_settle_s)
    node.event("probe_hold_start", label=pose["label"])
    hold(args.hold_s)
    node.event("probe_hold_end", label=pose["label"])
    for axis in range(6):
        node.event("probe_axis_start", label=pose["label"], axis=AXIS_NAMES[axis],
                   amplitude=amplitude[axis], speed=speed[axis])
        for leg, start, end in probe_profile(args):
            duration = abs(end - start) * amplitude[axis] / speed[axis]
            node.event("probe_leg_start", label=pose["label"], axis=AXIS_NAMES[axis], leg=leg,
                       direction=int(np.sign(end - start)))
            t0 = node.now_s()
            tick = t0
            while True:
                s = min((node.now_s() - t0) / duration, 1.0)
                offset = np.zeros(6)
                offset[axis] = (start + s * (end - start)) * amplitude[axis]
                position, rotation = offset_pose(position0, rotation0, offset[:3], offset[3:])
                node.publish_goal(position, rotation, tool, pose["q"], args)
                rep._check_controller_running(arm_client, rejections, True)
                measured = node.latest_state()
                error_mm = 1000.0 * float(np.linalg.norm(measured["position"] - position))
                error_deg = math.degrees(float(np.linalg.norm(rep.rotation_vector(
                    kin.quaternion_to_matrix(measured["quaternion_xyzw"]) @ rotation.T))))
                if error_mm > args.abort_mm or error_deg > args.abort_deg:
                    raise RuntimeError(f"probe tracking error {error_mm:.1f} mm / {error_deg:.1f} deg "
                                       "above the abort limit")
                if s >= 1.0:
                    break
                tick += period
                node.sleep_until(tick)
        hold(args.probe_settle_s)
        node.event("probe_axis_end", label=pose["label"], axis=AXIS_NAMES[axis])


def run_live(args, poses, spin, tool_matrix, output_dir: Path) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from franka_trajectory_replay.cartesian_replay_client import CartesianReplayClient
    from franka_trajectory_replay.replay_client import Rejected, ReplayClient
    from franka_trajectory_replay.runconfig import load_config

    config = load_config(args.config)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    joint_client = ReplayClient(config, node_name="osc_characterization_joint")
    arm_client = CartesianReplayClient(config, node_name="osc_characterization_arm")
    node = _make_node(config, args.sim)
    executor = MultiThreadedExecutor(num_threads=4)
    for member in (joint_client, arm_client, node):
        executor.add_node(member)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = "/" + config["cartesian"]["controller_name"].strip("/")
    topics = [controller + "/cartesian_state_fast", controller + "/policy_goal", "/joint_states",
              EVENT_TOPIC]
    if not args.sim:
        topics.append("/" + config["cartesian"]["robot_state_topic"].strip("/"))
    recorder = BagRecorder(output_dir / "bag", topics)
    report = {
        "backend": "ros_sim" if args.sim else "hardware",
        "arguments": {k: rep._plain(v) for k, v in vars(args).items() if k != "func"},
        "poses": [_pose_report(p) for p in poses],
        "spin_stops": [q.tolist() for q in spin["stops"]],
        "status": "error",
        "payloads": {},
    }
    moving = False
    cartesian_in_charge = False
    tool = [*np.asarray(POLICY_TOOL_OFFSET_XYZ, dtype=float), 0.0, 0.0, 0.0, 1.0]
    try:
        while node.now_s() <= 0.0:
            time.sleep(0.05)
        arm_client.ensure_active(print)
        arm_client.check_tool(policy_tool_transform(policy_tool_offset_config()), print)
        if args.sim != arm_client.uses_dh_model():
            raise Rejected("model_source does not match the backend (dh in ros-sim, franka on hardware)")
        arm_client.set_parameter_remote("policy_yaw_wrap_upper_deg", args.yaw_seam_deg)
        parameters = arm_client.controller_parameters()
        parameters.update(arm_client.remote_parameters(arm_client.controller_ns, [
            "friction_compensation_nm", "friction_compensation_velocity",
            "friction_compensation_torque", "task_wrench_dead_zone", "fast_state_publish_rate",
        ]))
        rep._check_policy_settings(parameters, args)
        if float(parameters.get("fast_state_publish_rate") or 0.0) < 500.0:
            raise Rejected("fast_state_publish_rate is below 500 Hz; the probe analysis needs ~/cartesian_state_fast")
        for name, value in (("policy_clip_position_max_m", args.probe_clip[0]),
                            ("policy_clip_orientation_max_rad", args.probe_clip[1]),
                            ("policy_max_position_error_max", args.probe_max_error[0]),
                            ("policy_max_orientation_error_max", args.probe_max_error[1])):
            if float(parameters.get(name, 0.0)) + 1.0e-9 < value:
                raise Rejected(f"{name} is {parameters.get(name)}, below the probe's {value}; "
                               "restart the launch with the current profile")
        report["controller_parameters"] = {k: rep._plain(v) for k, v in parameters.items()}
        friction = parameters.get("friction_compensation_nm")
        print(f"friction compensation: {friction}", flush=True)
        if not args.sim:
            readback = load_readback(arm_client)
            report["desk_readback_at_start"] = readback
            if args.set_load:
                check_desk_profile(readback, args)
        current = np.asarray(joint_client.current_joint_positions(), dtype=float)
        start_path, description = plan_start(current, poses[0]["q"], SceneClearance(), args)
        print(f"move to {poses[0]['label']}: {description}. The MuJoCo scene has no back table, "
              "shelf, camera rig or cables.", flush=True)
        if not args.yes:
            input("Start? Keep the E-stop in hand. [Enter/Ctrl-C] ")
        recorder.start()
        node.event("session_start", backend=report["backend"], friction=friction)
        moving = True
        joint_client.ensure_active(print)
        for waypoint in start_path[:-1]:
            joint_move(joint_client, node, waypoint, args, "lift")
        stations = [
            {"label": p["label"], "q": p["q"], "pose": p, "probe": i < args.probe_poses}
            for i, p in enumerate(poses)
        ] + [{"label": f"spin {q[6]:+.2f}", "q": q} for q in spin["stops"]]
        for index, level in enumerate(args.payload):
            node.event("payload_start", payload=level)
            joint_client.ensure_active(print)
            if args.set_load:
                if not args.yes and level == args.payload[0]:
                    input(f"Set payload '{level}' (controllers off for about a second)? [Enter/Ctrl-C] ")
                report["payloads"][level] = set_payload(level, joint_client, arm_client, args)
            # Odd levels walk the stations backwards, so no move is spent getting back.
            for station in stations if index % 2 == 0 else stations[::-1]:
                joint_move(joint_client, node, station["q"], args, station["label"])
                static_hold(joint_client, node, station["q"], args, station["label"], level)
                if station.get("probe") and level in args.probe_payloads:
                    pose = station["pose"]
                    node.forget_state()
                    arm_client.ensure_active(print)
                    cartesian_in_charge = True
                    arm_client.set_parameter_remote("policy_yaw_wrap_upper_deg", args.yaw_seam_deg)
                    rejections = int((arm_client.status() or {}).get("rejections", 0))
                    node.event("probe_start", label=pose["label"], payload=level)
                    run_probe(node, arm_client, pose, tool, args, rejections)
                    node.event("probe_end", label=pose["label"], payload=level)
                    arm_client.abort()
                    joint_client.ensure_active(print)
                    cartesian_in_charge = False
            node.event("payload_end", payload=level)
        if args.set_load and args.payload[-1] != "modelled":
            report["payloads"]["restored"] = set_payload("modelled", joint_client, arm_client, args)
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("interrupted: stopping the arm", flush=True)
    except Exception as exc:  # noqa: BLE001 - recorded, and the arm is stopped below
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"stopping: {report['error']}", flush=True)
    finally:
        if moving and cartesian_in_charge:
            try:
                arm_client.abort()
            except Exception as exc:  # noqa: BLE001
                print(f"abort failed: {exc}", flush=True)
        try:
            node.event("session_end", status=report["status"])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
        recorder.stop()
        report["events"] = node.events
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=rep._plain) + "\n")
        print(f"\nwrote {output_dir}", flush=True)
        if args.set_load:
            print("REMINDER: switch Desk back to the normal hand profile before any policy rollout.",
                  flush=True)
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        rclpy.try_shutdown()
    if report["status"] == "error":
        raise SystemExit(1)


def _pose_report(pose) -> dict:
    return {
        "label": pose["label"],
        "draw": pose["draw"],
        "position": pose["position"].tolist(),
        "rotation": pose["rotation"].tolist(),
        "q": pose["q"].tolist(),
        "gravity_in_flange": pose["gravity_in_flange"].tolist(),
        "posture": pose["posture"],
        "probe_min_clearance_m": pose["probe_min_clearance_m"],
        "probe_min_joint_margin_rad": pose["probe_min_joint_margin_rad"],
        "seam_margin_deg": pose["seam_margin_deg"],
    }


# --- entry ----------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("backend", choices=("hardware", "ros-sim"))
    p.add_argument("--plan-only", action="store_true", help="plan and check; no ROS, no motion")
    p.add_argument("--yes", action="store_true", help="skip the confirmations")
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--pose", type=float, nargs=3, default=(0.30, 0.20, 0.30), metavar=("X", "Y", "Z"),
                   help="grasp-point position of the probe poses, base frame, m")
    p.add_argument("--orientations", type=int, default=3)
    p.add_argument("--min-tilt-deg", type=float, default=10.0)
    p.add_argument("--max-tilt-deg", type=float, default=35.0)
    p.add_argument("--max-yaw-deg", type=float, default=60.0)
    p.add_argument("--min-gravity-separation-deg", type=float, default=15.0)
    p.add_argument("--max-draws", type=int, default=400)
    p.add_argument("--payload", nargs="+", choices=("modelled", "none"), default=["modelled", "none"])
    p.add_argument("--probe-poses", type=int, default=1, help="how many of the poses get the probe")
    p.add_argument("--probe-payloads", nargs="*", choices=("modelled", "none"), default=["modelled"],
                   help="payload levels that also get the dead-zone probe")
    p.add_argument("--no-set-load", dest="set_load", action="store_false",
                   help="leave the payload alone (ros-sim always does)")
    p.add_argument("--set-load-service", default="/service_server/set_load")
    p.add_argument("--hand-mass", type=float, default=DESK_HAND_MASS)
    p.add_argument("--hand-com", type=float, nargs=3, default=DESK_HAND_COM)
    p.add_argument("--hand-inertia", type=float, nargs=3, default=DESK_HAND_INERTIA,
                   metavar=("IXX", "IYY", "IZZ"))
    p.add_argument("--probe-translation-mm", type=float, default=15.0)
    p.add_argument("--probe-rotation-deg", type=float, default=8.0)
    p.add_argument("--probe-speed-mm-s", type=float, default=6.0)
    p.add_argument("--probe-rate-deg-s", type=float, default=2.0)
    p.add_argument("--probe-cycles", type=int, default=1)
    p.add_argument("--probe-settle-s", type=float, default=1.0)
    p.add_argument("--probe-clip", type=float, nargs=2, default=(0.05, 0.25), metavar=("M", "RAD"))
    p.add_argument("--probe-max-error", type=float, nargs=2, default=(0.1, 0.5), metavar=("M", "RAD"))
    p.add_argument("--abort-mm", type=float, default=45.0)
    p.add_argument("--abort-deg", type=float, default=20.0)
    p.add_argument("--hold-s", type=float, default=2.0)
    p.add_argument("--spin-stops", type=int, default=5, help="0 skips the wrist spin")
    p.add_argument("--spin-range-rad", type=float, default=2.7)
    p.add_argument("--max-goto-step", type=float, default=2.5,
                   help="rad per joint goto (the controller refuses 3.0)")
    p.add_argument("--goal-rate", type=float, default=100.0, help="Hz")
    p.add_argument("--clearance", type=float, default=0.05, help="m, MuJoCo scene")
    p.add_argument("--min-start-clearance", type=float, default=0.01,
                   help="m; the move off a start inside --clearance must stay above this")
    p.add_argument("--lift", type=float, default=0.12, help="m, straight up if the direct start is refused")
    p.add_argument("--start-joints", type=float, nargs=7, metavar="Q",
                   help="with --plan-only: also check the move from these joints to the first pose")
    p.add_argument("--start-from-robot", action="store_true",
                   help="with --plan-only: read the start joints from /joint_states")
    p.add_argument("--self-clearance", type=float, default=0.03, help="m, hand to arm")
    p.add_argument("--min-joint-margin", type=float, default=0.15)
    p.add_argument("--min-singular-value", type=float, default=0.03)
    p.add_argument("--task-profile", default="auto")
    p.add_argument("--yaw-seam-deg", type=float, default=None)
    p.add_argument("--config",
                   default=str(WORKSPACE_ROOT / "src/inspire_franka_trajectory_replay/config/replay.yaml"))
    p.add_argument("--output-root", default=str(WORKSPACE_ROOT / "debug/osc_characterization"))
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    args.sim = args.backend == "ros-sim"
    if args.sim:
        args.set_load = False
        args.payload = ["modelled"]
    if not args.set_load and len(args.payload) > 1:
        raise SystemExit("more than one payload level needs set_load")
    profile = fo.resolve_task_profile(args.task_profile, DEFAULT_CHECKPOINT)
    if args.yaw_seam_deg is None:
        args.yaw_seam_deg = float(profile.yaw_wrap_upper_deg)
    tool = policy_tool_transform(policy_tool_offset_config())
    scene = SceneClearance()
    try:
        poses = sample_poses(args, tool, scene)
        spin = plan_spin(poses[-1]["q"], scene, args) if args.spin_stops > 0 else {"stops": []}
    except ValueError as exc:
        raise SystemExit(f"the plan is refused: {exc}") from exc
    for pose in poses:
        d = pose["draw"]
        print(f"{pose['label']}: tilt {d['tilt_deg']:.0f} deg about {d['tilt_axis_deg']:.0f} deg, "
              f"yaw {d['yaw_deg']:+.0f} deg; q {np.array2string(pose['q'], precision=2)}; "
              f"clearance {1000 * pose['posture']['clearance_m']:.0f} mm "
              f"({pose['posture']['closest']}), probe extremes "
              f"{1000 * pose['probe_min_clearance_m']:.0f} mm, joint margin "
              f"{pose['probe_min_joint_margin_rad']:.2f} rad, seam {pose['seam_margin_deg']:.0f} deg")
    if spin["stops"]:
        print(f"wrist spin: {len(spin['stops'])} stops, min clearance "
              f"{1000 * spin['min_clearance_m']:.0f} mm")
    per_axis = (4 * (args.probe_translation_mm / args.probe_speed_mm_s) * 3
                + 4 * (args.probe_rotation_deg / args.probe_rate_deg_s) * 3) * args.probe_cycles
    probes = min(args.probe_poses, len(poses)) * len(args.probe_payloads)
    probe_s = probes * (per_axis + 7 * args.probe_settle_s + args.hold_s + 4.0)
    stations = len(poses) + len(spin["stops"])
    # The joint controller takes at least 5 s per goto; a hold follows each.
    moves_s = len(args.payload) * stations * (5.0 + args.hold_s + 0.5)
    loads_s = 4.0 * (len(args.payload) + 1) if args.set_load else 0.0
    print(f"estimated {(probe_s + moves_s + loads_s) / 60:.1f} min: probe {probe_s / 60:.1f}, "
          f"moves and holds {moves_s / 60:.1f}, set_load {loads_s / 60:.1f}")
    if args.plan_only:
        if args.start_from_robot:
            args.start_joints = read_robot_joints()
            print(f"robot joints: {' '.join(f'{v:.4f}' for v in args.start_joints)}")
        if args.start_joints is not None:
            try:
                path, description = plan_start(args.start_joints, poses[0]["q"], scene, args)
            except ValueError as exc:
                raise SystemExit(f"the start is refused: {exc}") from exc
            print(f"from --start-joints: {description}")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_live(args, poses, spin, tool, Path(args.output_root) / f"{args.backend}-{stamp}")


if __name__ == "__main__":
    main()
