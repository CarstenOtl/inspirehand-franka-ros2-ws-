"""Home and replay a coordinated Forge trajectory on an FR3 and Inspire RH56."""

import argparse
from contextlib import nullcontext
import dataclasses
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from franka_trajectory_replay import cartesian
from franka_trajectory_replay.cartesian_replay_client import CartesianReplayClient
from franka_trajectory_replay.kinematics import tool_transform
from franka_trajectory_replay.prepare import prepare, summarize
from franka_trajectory_replay import limits
from franka_trajectory_replay.replay_client import Rejected, ReplayClient
from franka_trajectory_replay.runconfig import load_config
from franka_trajectory_replay.trajectory_io import Trajectory as ArmTrajectory
from inspire_hand_driver import kinematics as kin

from .joint_trajectory_client import JointTrajectoryClient
from .trajectory import (
    ARM_JOINTS,
    FINGER_FLEXION_JOINTS,
    HAND_JOINTS,
    THUMB_ABDUCTION_DOF,
    THUMB_ABDUCTION_JOINT,
    THUMB_ABDUCTION_ZERO_OPEN_RATIO,
    scale_thumb_abduction,
    load_trajectory,
    scale_finger_flexion,
)


# Where each commanded hand joint sits in the driver's own DOF table. Resolved
# by name so this cannot silently follow the wrong channel if either ordering
# is ever changed.
HAND_DOF = tuple(kin.dof_index(name) for name in HAND_JOINTS)

# The joint limits come from that same table rather than a copy: they are what
# the radian-to-ratio conversion below divides by, so a copy that drifted would
# not fail a comparison, it would scale every hand command wrongly.
HAND_LOWER = np.array([kin.DOFS[index].lower for index in HAND_DOF])
HAND_UPPER = np.array([kin.DOFS[index].upper for index in HAND_DOF])

# These are the three support fingers used by the threading task. Keep this
# override separate from trajectory generation/retargeting: applying it at the
# replay boundary guarantees that all seven recorded FR3 joints remain exactly
# as loaded, including joint 7.
SUPPORT_FINGER_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
)
SUPPORT_FINGER_INDICES = tuple(
    HAND_JOINTS.index(name) for name in SUPPORT_FINGER_JOINTS
)

# How far outside those limits a sample may sit and still be treated as being
# *at* the limit. Forge writes float32, so a joint the policy drove hard onto
# its own stop arrives a couple of micro-radians past it -- the pickplace
# capture reaches -2.09e-6 rad on index_proximal. Rejecting a whole trajectory
# over that is wrong, and so is a tolerance that hides a real overshoot: 1e-4
# rad is 0.07 of the hand's 1/1000 register step, so anything snapped by it
# was going to round to the same command anyway.
HAND_LIMIT_TOLERANCE = 1e-4


def _packaged(name):
    return str(
        Path(get_package_share_directory("inspire_franka_trajectory_replay"))
        / "config"
        / name
    )


def load_home(path):
    """Read the commandable arm and hand positions from a homing YAML."""
    with open(path, encoding="utf-8") as stream:
        pose = yaml.safe_load(stream) or {}
    names = [str(name) for name in pose.get("joint_names", ())]
    positions = np.asarray(pose.get("positions", ()), dtype=float)
    if len(names) != len(positions):
        raise ValueError("homing YAML joint_names and positions have different lengths")
    columns = {name: index for index, name in enumerate(names)}
    missing = [name for name in ARM_JOINTS + HAND_JOINTS if name not in columns]
    if missing:
        raise ValueError(f"homing YAML is missing commandable joints: {missing}")
    arm = positions[[columns[name] for name in ARM_JOINTS]]
    hand = positions[[columns[name] for name in HAND_JOINTS]]
    return arm, hand


def _prepare_arm(
    trajectory,
    config,
    max_duration,
    time_scale=None,
    allow_limit_violations=False,
):
    settings = config["prepare"]
    requested_time_scale = (
        settings["time_scale"] if time_scale is None else float(time_scale)
    )
    source = ArmTrajectory(
        t=trajectory.time,
        q=trajectory.arm,
        source=str(trajectory.source),
    )
    arguments = dict(
        rate=settings["rate"],
        cutoff_hz=settings["cutoff_hz"],
        hold_start=settings["hold_start"],
        hold_end=settings["hold_end"],
        time_scale=requested_time_scale,
        auto_scale=False,
        velocity_margin=settings["velocity_margin"],
        acceleration_margin=settings["acceleration_margin"],
        jerk_margin=settings["jerk_margin"],
        joint_names=config["joint_names"],
        lead_in=settings["lead_in"],
        lead_out=settings["lead_out"],
        lead_max_acceleration=settings["lead_max_acceleration"],
        interpolation=settings["interpolation"],
        blend_time=settings["blend_time"],
    )
    prepared = prepare(source, **arguments)
    if (
        prepared.report["ok"]
        or allow_limit_violations
        or not settings["auto_scale"]
    ):
        return prepared

    required = 1.02 * limits.required_time_scale(prepared.report)
    scaled_duration = trajectory.duration * requested_time_scale * required
    scaled_duration += (
        settings["hold_start"]
        + settings["hold_end"]
        + settings["lead_in"]
        + settings["lead_out"]
    )
    if scaled_duration > max_duration:
        worst = max(
            max(prepared.report["velocity_fraction"]),
            max(prepared.report["acceleration_fraction"]),
            max(prepared.report["jerk_fraction"]),
        )
        raise ValueError(
            "FR3 safety preparation would need approximately "
            f"x{required:.2f} time scaling ({scaled_duration:.1f} s), above the "
            f"{max_duration:.1f} s guard; worst limit fraction is {worst:.1f}x. "
            "Safety violations: "
            + "; ".join(prepared.report["violations"])
        )
    arguments["time_scale"] = requested_time_scale * required
    arguments["auto_scale"] = True
    return prepare(source, **arguments)


def _prepare_cartesian(prepared, config, margins=None):
    """The pose stream for the Cartesian controller: FK of the prepared joint stream.

    The joint-space preparation stays the first stage on purpose. It is the part
    validated on hardware and where the FR3 limits and the automatic time scaling
    live; forward kinematics of that stream through the configured tool describes
    a motion the arm can make, and the stream itself is the nullspace target.
    """
    tcp = config["tcp"]
    if tcp["frame"] != "fr3_link8":
        raise ValueError(
            f"tcp.frame is {tcp['frame']!r}; the pose stream is forward kinematics of "
            "the flange (fr3_link8) through tcp.offset_xyz/offset_rpy"
        )
    tool = tool_transform(tcp["offset_xyz"], tcp["offset_rpy"])
    settings = dict(config["cartesian"])
    settings.update({key: value for key, value in (margins or {}).items() if value is not None})
    stream = cartesian.from_joint_stream(prepared, tool)
    cartesian.check_cartesian_limits(
        stream,
        settings["velocity_margin"],
        settings["acceleration_margin"],
        settings["jerk_margin"],
    )
    return stream


def _hand_stream(trajectory, prepared, rate):
    """Place the low-rate hand samples on the arm controller's prepared clock."""
    if trajectory.hand is None:
        return None, None
    count = int(np.floor(prepared.duration * rate)) + 1
    stream_time = np.arange(count, dtype=float) / rate
    if stream_time[-1] < prepared.duration:
        stream_time = np.append(stream_time, prepared.duration)
    capture_start = prepared.t[prepared.params["capture_start_index"]]
    source_time = capture_start + trajectory.time * prepared.params["time_scale"]
    positions = np.column_stack(
        [
            np.interp(stream_time, source_time, trajectory.hand[:, joint])
            for joint in range(len(HAND_JOINTS))
        ]
    )
    return stream_time, positions


def _hand_stream_native(trajectory, rate, time_scale):
    """Resample the hand samples onto their own clock, with no arm involved.

    The coordinated path places the hand on the arm's prepared clock, which is
    time-scaled to stay inside the FR3's limits. Hand-only replay has no arm to
    respect, so the recording's own timing is preserved and ``time_scale`` is
    whatever the caller asked for.
    """
    source_time = (trajectory.time - trajectory.time[0]) * time_scale
    duration = float(source_time[-1])
    count = int(np.floor(duration * rate)) + 1
    stream_time = np.arange(count, dtype=float) / rate
    if stream_time[-1] < duration:
        stream_time = np.append(stream_time, duration)
    positions = np.column_stack(
        [
            np.interp(stream_time, source_time, trajectory.hand[:, joint])
            for joint in range(len(HAND_JOINTS))
        ]
    )
    return stream_time, positions


def _validate_hand(values, label):
    """Reject a hand trajectory that leaves the URDF's limits; return it snapped.

    Snapping only ever moves a sample that was already within
    :data:`HAND_LIMIT_TOLERANCE` of a limit, and it happens *after* the check,
    so it can never mask the overshoot it would otherwise hide.
    """
    values = np.asarray(values, dtype=float)
    low = values.min(axis=0) if values.ndim == 2 else values
    high = values.max(axis=0) if values.ndim == 2 else values
    outside = [
        f"{HAND_JOINTS[index]} "
        f"[{low[index]:.6f}, {high[index]:.6f}] outside "
        f"[{HAND_LOWER[index]:g}, {HAND_UPPER[index]:g}]"
        for index in range(6)
        if low[index] < HAND_LOWER[index] - HAND_LIMIT_TOLERANCE
        or high[index] > HAND_UPPER[index] + HAND_LIMIT_TOLERANCE
    ]
    if outside:
        raise ValueError(f"{label} exceeds Inspire hand limits: {'; '.join(outside)}")
    return np.clip(values, HAND_LOWER, HAND_UPPER)


def _close_support_fingers(hand, home_hand):
    """Override only pinky, ring, and middle with their closed joint limits."""
    if hand is None:
        raise ValueError(
            "--close-support-fingers requires Inspire hand positions in the trajectory"
        )
    overridden = np.array(hand, dtype=float, copy=True)
    overridden_home = np.array(home_hand, dtype=float, copy=True)
    for index in SUPPORT_FINGER_INDICES:
        overridden[..., index] = HAND_UPPER[index]
        overridden_home[index] = HAND_UPPER[index]
    return overridden, overridden_home


def _check_home(home_arm, first, home_path, tolerance):
    """Refuse a homing pose that is not where the trajectory actually begins.

    Replay homes to the YAML, then moves to the trajectory's first point. When
    the two agree that second move is nothing; when they do not, the arm makes
    an unplanned trip at the moment replay starts, and -- worse -- the scene
    the policy was recorded against is not the scene it is being replayed into.
    The threading YAML matches its capture to the last digit, so a mismatch
    here means the YAML belongs to a different task configuration rather than
    that a tolerance is too tight.
    """
    delta = np.abs(np.asarray(home_arm, dtype=float) - np.asarray(first, dtype=float))
    if delta.max() <= tolerance:
        return
    offenders = "; ".join(
        f"{ARM_JOINTS[index]} home {home_arm[index]:+.6f} vs trajectory "
        f"{first[index]:+.6f} ({delta[index]:.3f} rad)"
        for index in np.argsort(-delta)
        if delta[index] > tolerance
    )
    raise ValueError(
        f"the homing pose in {home_path} is {delta.max():.3f} rad from the "
        f"trajectory's first point, over the {tolerance:g} rad limit: {offenders}. "
        f"Point --home at the YAML recorded with this trajectory, or raise "
        f"--max-home-delta to accept the move."
    )


class HandReplayMixin:
    """The hand's position link shared by the hardware and legacy sim clients."""

    def __init__(self, config, hand_topic, hand_state_topic):
        super().__init__(
            config,
            node_name="inspire_franka_trajectory_replay",
        )
        self._hand_publisher = self.create_publisher(JointState, hand_topic, 10)
        self._hand_lock = threading.Lock()
        self._hand_position = None
        self.create_subscription(JointState, hand_state_topic, self._on_hand_state, 10)

    def _on_hand_state(self, message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in HAND_JOINTS):
            with self._hand_lock:
                self._hand_position = np.array(
                    [positions[name] for name in HAND_JOINTS], dtype=float
                )

    def command_hand(self, positions):
        """Publish one hand target, converting radians to the driver's ratios.

        Everything above this line -- the Forge trajectories, the homing YAMLs,
        the tracking comparison against ``joint_states`` -- is in radians,
        because those files also carry the FR3's seven joints and because the
        URDF defines the hand in radians. The driver commands in open ratios,
        running the opposite way (1.0 is open, 0.0 rad is open). This method is
        the single place the two conventions meet.
        """
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(HAND_JOINTS)
        message.position = [
            kin.rad_to_open_ratio(HAND_DOF[channel], float(value))
            for channel, value in enumerate(positions)
        ]
        self._hand_publisher.publish(message)

    def wait_for_hand_link(self, timeout=10.0):
        """Block until the hand driver is on the graph and publishing state.

        A publisher is not matched to its subscriber the instant it is created,
        so a single command sent immediately afterwards is dropped with no
        error anywhere: the driver never sees it. The coordinated path never
        hit this because ``ensure_active``'s service round-trips gave discovery
        several seconds first. Hand-only replay has nothing else to wait on, so
        it has to wait here explicitly.
        """

        def linked():
            with self._hand_lock:
                have_state = self._hand_position is not None
            return self._hand_publisher.get_subscription_count() > 0 and have_state

        self.wait_until(linked, timeout, "the Inspire hand driver on the ROS graph")

    def wait_for_hand(self, target, timeout, tolerance):
        target = np.asarray(target, dtype=float)
        # Publish the original target. Only the driver rescales thumb
        # abduction onto its calibrated travel; feedback is compared with that
        # effective physical target so homing does not wait for an intentionally
        # unreachable pre-overlay position.
        effective_target = scale_thumb_abduction(target)

        def arrived():
            with self._hand_lock:
                return self._hand_position is not None and np.max(
                    np.abs(self._hand_position - effective_target)
                ) <= tolerance

        # The driver holds its last target, so repeating it costs nothing and
        # recovers a homing command lost to anything transient on the graph.
        self.wait_until(
            arrived,
            timeout,
            "the Inspire hand to reach its home pose",
            progress=lambda: self.command_hand(target),
        )


class CoordinatedReplayClient(HandReplayMixin, ReplayClient):
    """Example joint-impedance waypoint client plus the hand's position link."""

    def ensure_active(self, log=print):
        controllers = self.list_controllers()
        controller = controllers.get(self.controller)
        if controller is None or controller.type != (
            "franka_trajectory_replay/TrajectoryReplayController"
        ):
            raise Rejected(
                "joint-impedance replay requires the waypoint effort controller; "
                "restart replay.launch.py with its default controllers_joint_impedance.yaml. "
                "For the legacy MuJoCo position stack use --arm-controller position-jtc."
            )
        parameters = self.controller_parameters()
        if parameters["command_interface"] != "effort" or parameters["coriolis_compensation"]:
            raise Rejected(
                "expected the simple joint-impedance example profile "
                "(command_interface=effort, coriolis_compensation=false)"
            )
        super().ensure_active(log)
        self.wait_until(
            lambda: all(publisher.get_subscription_count() > 0 for publisher in (
                self._goto_publisher, self._trajectory_publisher,
                self._pause_publisher, self._resume_publisher, self._abort_publisher
            )),
            10.0,
            "the waypoint controller's command subscriptions",
        )
        if "processed_command_id" not in self.wait_for_status():
            raise Rejected("rebuild franka_trajectory_replay: controller lacks abort acknowledgment")
        log(
            "waypoint replay uses the example's joint-impedance law over effort "
            f"interfaces (stiffness scale {parameters.get('stiffness_scale', 1.0):g})"
        )

    def status(self):
        # A deactivated controller stops publishing: never treat old 'idle'
        # feedback as successful completion after a hardware reflex.
        with self._lock:
            if self._status is not None and time.monotonic() - self._status_stamp > 1.0:
                raise Rejected("waypoint controller feedback stopped; check the hardware log")
            return dict(self._status) if self._status else None

    def abort(self):
        with self._lock:
            before = int((self._status or {}).get("processed_command_id", 0))
        super().abort()
        # Keep the executor alive while the reference decelerates to a hold.
        # On a reflex the status stream may already have stopped; report that
        # without replacing the original exception during cleanup.
        try:
            def stopped():
                status = self.status() or {}
                processed = int(status.get("processed_command_id", 0))
                return (processed > before
                        and int(status.get("completed_command_id", 0)) >= processed
                        and status.get("phase_name") == "idle")

            self.wait_until(
                stopped,
                2.0, "the impedance reference to stop",
            )
        except (Rejected, TimeoutError) as exc:
            self.get_logger().warning(f"could not confirm arm hold: {exc}")


class PositionReplayClient(HandReplayMixin, JointTrajectoryClient):
    """Explicit compatibility path for the existing MuJoCo position stack."""


def _stream_hand(node, started, stopped, stream_time, positions, errors,
                 trajectory_clock=None):
    try:
        deadline = time.monotonic() + 20.0
        while not started.is_set() and time.monotonic() < deadline:
            if stopped.is_set():
                return
            started.wait(timeout=0.05)
        if not started.is_set():
            raise TimeoutError("arm controller did not accept the trajectory")
        epoch_ns = None if trajectory_clock is not None else node.get_clock().now().nanoseconds
        for target_time, target in zip(stream_time, positions):
            if trajectory_clock is None:
                target_ns = epoch_ns + int(float(target_time) * 1e9)
                while True:
                    remaining = (target_ns - node.get_clock().now().nanoseconds) / 1e9
                    if remaining <= 0:
                        break
                    if stopped.wait(timeout=min(remaining, 0.02)):
                        return
            else:
                # The arm owns the prepared trajectory clock. It slows and then
                # freezes that clock during an interactive pause, so waiting on
                # elapsed trajectory time keeps the independent 50 Hz hand link
                # aligned with the exact arm sample across an arbitrary pause.
                while float(trajectory_clock()) < float(target_time):
                    if stopped.wait(timeout=0.01):
                        return
            if stopped.is_set():
                return
            node.command_hand(target)
    except Exception as exc:  # reported by the main thread after the arm stops
        errors.append(exc)


class _InteractivePause:
    """Read one-key pause/resume commands without interfering with ROS threads."""

    def __init__(self, node, started, coordinated_stop=None):
        self.node = node
        self.started = started
        self.coordinated_stop = coordinated_stop
        self.stopped = threading.Event()
        self.errors = []
        self.aborted = threading.Event()
        self.paused = False
        self.fd = None
        self.terminal_settings = None
        self.thread = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("--interactive-pause requires a terminal on stdin")
        self.fd = sys.stdin.fileno()
        self.terminal_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("Interactive replay: SPACE pauses/resumes; q aborts.", flush=True)
        return self

    def _run(self):
        try:
            while not self.started.wait(timeout=0.05):
                if self.stopped.is_set():
                    return
            while not self.stopped.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                key = sys.stdin.read(1)
                if key == " ":
                    if self.paused:
                        self.node.resume()
                        self.paused = False
                        print("\nRESUMED", flush=True)
                    else:
                        print("\nPause requested; wait for PAUSED before approaching.", flush=True)
                        self.node.pause()
                        self.paused = True
                        print("PAUSED: arm and hand are holding. SPACE resumes; q aborts.",
                              flush=True)
                elif key.lower() == "q":
                    print("\nAbort requested.", flush=True)
                    self.aborted.set()
                    if self.coordinated_stop is not None:
                        self.coordinated_stop.set()
                    self.node.abort()
                    return
        except Exception as exc:
            self.errors.append(exc)
            if self.coordinated_stop is not None:
                self.coordinated_stop.set()
            try:
                self.node.abort()
            except Exception:
                pass

    def __exit__(self, _exc_type, _exc, _traceback):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.terminal_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.terminal_settings)


def _gate(enabled, prompt):
    if not enabled:
        return
    input(f"{prompt} [Enter to continue, Ctrl-C to abort] ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="replay_data.npz or its trajectory directory")
    parser.add_argument("--home", default=None, help="homing YAML (default: threading.yaml)")
    parser.add_argument("--config", default=None, help="Franka replay configuration YAML")
    parser.add_argument(
        "--arm-controller",
        choices=("joint-impedance", "cartesian-impedance", "position-jtc"),
        default="joint-impedance",
        help="joint-impedance: example joint-impedance effort law (default); "
             "cartesian-impedance: home with the joint controller, then replay the "
             "pose stream with the example Cartesian impedance law; "
             "position-jtc: legacy MuJoCo stack",
    )
    parser.add_argument(
        "--cartesian-velocity-margin", type=float, default=None,
        help="override replay.yaml's cartesian.velocity_margin (cartesian-impedance only)",
    )
    parser.add_argument(
        "--cartesian-acceleration-margin", type=float, default=None,
        help="override replay.yaml's cartesian.acceleration_margin",
    )
    parser.add_argument(
        "--cartesian-jerk-margin", type=float, default=None,
        help="override replay.yaml's cartesian.jerk_margin",
    )
    parser.add_argument(
        "--stiffness-scale", type=float, default=None,
        help="set the Cartesian controller's live stiffness_scale before replay "
             "(cartesian-impedance only)",
    )
    parser.add_argument("--rate", type=float, default=None, help="input rate if absent")
    parser.add_argument("--env", type=int, default=0, help="environment in a batched Forge NPZ")
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        help="one rollout cycle from a multi-cycle Forge recording",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="one episode from a Forge recording that resets without a cycle field",
    )
    parser.add_argument("--hand-rate", type=float, default=50.0)
    parser.add_argument("--hand-topic", default=None)
    parser.add_argument("--hand-state-topic", default=None)
    parser.add_argument("--hand-timeout", type=float, default=20.0)
    parser.add_argument("--hand-tolerance", type=float, default=0.08)
    parser.add_argument(
        "--max-home-delta",
        type=float,
        default=0.1,
        help="reject a homing YAML that does not start where the trajectory does (rad)",
    )
    parser.add_argument(
        "--timeout-margin",
        type=float,
        default=15.0,
        help="wall-clock grace beyond the trajectory's own duration before giving up. "
             "The controller's watchdogs are deliberately on the wall clock, so a "
             "simulator running below real time needs this raised even though nothing "
             "is wrong: MuJoCo sits near 0.9x, so allow roughly 0.3 * duration.",
    )
    parser.add_argument(
        "--max-prepared-duration",
        type=float,
        default=120.0,
        help="reject excessive automatic slow-down before allocating the 1 kHz stream",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help="playback duration multiplier (5 plays five times slower); applies to "
             "the arm and coordinated hand, or to the hand with --no-arm",
    )
    parser.add_argument(
        "--hand-time-scale",
        type=float,
        default=None,
        help="deprecated hand-only alias for --time-scale; requires --no-arm",
    )
    parser.add_argument("--no-hand", action="store_true", help="arm only")
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="hand only: the arm is neither prepared nor commanded",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unsafe-simulation",
        action="store_true",
        help="send a trajectory that fails FR3 preparation only to the explicit "
             "position-jtc MuJoCo path; never permitted with joint-impedance hardware",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="skip motion prompts")
    parser.add_argument(
        "--close-support-fingers",
        action="store_true",
        help="hold pinky, ring, and middle at their fully closed limits during "
             "homing and replay; arm waypoints are unchanged",
    )
    parser.add_argument(
        "--finger-flexion-scale",
        type=float,
        default=1.0,
        help="multiply only the index- and thumb-MCP waypoint flexion; "
             "1.3 commands 30%% more and 1.5 commands 50%% more (default: 1.0)",
    )
    parser.add_argument(
        "--interactive-pause",
        action="store_true",
        help="during replay, SPACE pauses/resumes both devices and q aborts",
    )
    # `ros2 run` appends "--ros-args ..." for node parameters, and the whole
    # tail belongs to rclpy rather than to argparse. Only the tail is dropped,
    # so a mistyped flag before it is still an error rather than being quietly
    # ignored. This is what lets a simulated run take use_sim_time: MuJoCo runs
    # near but not at real time, and every timeout below is measured on this
    # node's clock, so without it a long trajectory times out against a
    # simulator that is merely slow.
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    args = parser.parse_args(argv)

    if args.hand_rate <= 0:
        parser.error("--hand-rate must be positive")
    if args.max_prepared_duration <= 0:
        parser.error("--max-prepared-duration must be positive")
    if args.time_scale is not None and args.time_scale <= 0:
        parser.error("--time-scale must be positive")
    if args.hand_time_scale is not None and args.hand_time_scale <= 0:
        parser.error("--hand-time-scale must be positive")
    if not np.isfinite(args.finger_flexion_scale) or args.finger_flexion_scale <= 0:
        parser.error("--finger-flexion-scale must be finite and positive")
    if args.max_home_delta < 0:
        parser.error("--max-home-delta must not be negative")
    if args.timeout_margin < 0:
        parser.error("--timeout-margin must not be negative")
    if args.no_arm and args.no_hand:
        parser.error("--no-arm and --no-hand together leave nothing to replay")
    if args.close_support_fingers and args.no_hand:
        parser.error("--close-support-fingers cannot be used with --no-hand")
    if args.finger_flexion_scale != 1.0 and args.no_hand:
        parser.error("--finger-flexion-scale cannot be used with --no-hand")
    if args.interactive_pause and args.no_arm:
        parser.error("--interactive-pause requires the arm trajectory clock")
    if args.interactive_pause and args.arm_controller == "position-jtc":
        parser.error("--interactive-pause is unavailable on the position-jtc path")
    cartesian_mode = args.arm_controller == "cartesian-impedance"
    cartesian_margins = {
        "velocity_margin": args.cartesian_velocity_margin,
        "acceleration_margin": args.cartesian_acceleration_margin,
        "jerk_margin": args.cartesian_jerk_margin,
    }
    for name, value in cartesian_margins.items():
        if value is not None and not cartesian_mode:
            parser.error(f"--cartesian-{name.replace('_', '-')} requires "
                         "--arm-controller cartesian-impedance")
        if value is not None and not (np.isfinite(value) and value > 0):
            parser.error(f"--cartesian-{name.replace('_', '-')} must be finite and positive")
    if args.stiffness_scale is not None:
        if not cartesian_mode:
            parser.error("--stiffness-scale requires --arm-controller cartesian-impedance")
        if not np.isfinite(args.stiffness_scale) or args.stiffness_scale < 0:
            parser.error("--stiffness-scale must be finite and non-negative")
    if cartesian_mode and args.no_arm:
        parser.error("--arm-controller cartesian-impedance needs the arm; drop --no-arm")
    if args.allow_unsafe_simulation and args.arm_controller != "position-jtc":
        parser.error(
            "--allow-unsafe-simulation requires --arm-controller position-jtc"
        )
    if args.hand_time_scale is not None and not args.no_arm:
        parser.error("--hand-time-scale only applies to --no-arm; the coordinated "
                     "hand stream follows the arm's prepared clock")
    if args.time_scale is not None and args.hand_time_scale is not None:
        parser.error("use --time-scale or --hand-time-scale, not both")
    if not args.no_arm and args.time_scale is not None and args.time_scale < 1.0:
        parser.error("--time-scale must be at least 1 for arm replay")
    playback_time_scale = (
        args.time_scale if args.time_scale is not None
        else args.hand_time_scale if args.hand_time_scale is not None
        else 1.0
    )
    home_path = args.home or _packaged("threading.yaml")
    config_path = args.config or _packaged("replay.yaml")
    hand_topic = args.hand_topic or "/inspire_hand/command"
    hand_state_topic = args.hand_state_topic or "/inspire_hand/joint_states"

    try:
        trajectory = load_trajectory(
            args.trajectory, args.rate, args.env, args.cycle, args.segment
        )
        home_arm, home_hand = load_home(home_path)
        if args.finger_flexion_scale != 1.0:
            trajectory = dataclasses.replace(
                trajectory,
                hand=scale_finger_flexion(
                    trajectory.hand, args.finger_flexion_scale
                ),
            )
        if args.close_support_fingers:
            overridden_hand, home_hand = _close_support_fingers(
                trajectory.hand, home_hand
            )
            trajectory = dataclasses.replace(trajectory, hand=overridden_hand)
        prepared = None
        pose_stream = None
        if not args.no_arm:
            prepared = _prepare_arm(
                trajectory,
                load_config(config_path),
                args.max_prepared_duration,
                time_scale=args.time_scale,
                allow_limit_violations=args.allow_unsafe_simulation,
            )
            if not prepared.report["ok"] and not args.allow_unsafe_simulation:
                raise ValueError("prepared arm trajectory violates FR3 limits")
        if prepared is not None and cartesian_mode:
            pose_stream = _prepare_cartesian(prepared, load_config(config_path), cartesian_margins)
            if not pose_stream.report["ok"]:
                raise ValueError(
                    "prepared pose stream violates the Cartesian limits: "
                    + "; ".join(pose_stream.report["violations"])
                )
        if not args.no_arm:
            _check_home(home_arm, trajectory.arm[0], home_path, args.max_home_delta)
        if trajectory.hand is None and not args.no_hand:
            raise ValueError("trajectory has no Inspire hand positions; pass --no-hand for arm-only")
        if trajectory.hand is not None:
            trajectory = dataclasses.replace(
                trajectory, hand=_validate_hand(trajectory.hand, "trajectory")
            )
        home_hand = _validate_hand(home_hand, "homing pose")
    except (OSError, ValueError, KeyError) as exc:
        print(f"trajectory error: {exc}")
        return 2

    config = load_config(config_path)
    stream_time, hand_positions = (None, None)
    if not args.no_hand:
        if args.no_arm:
            stream_time, hand_positions = _hand_stream_native(
                trajectory, args.hand_rate, playback_time_scale
            )
        else:
            stream_time, hand_positions = _hand_stream(trajectory, prepared, args.hand_rate)

    print(f"source: {trajectory.source}")
    if trajectory.cycle is not None:
        print(f"Forge rollout cycle: {trajectory.cycle}")
    if trajectory.segment is not None:
        print(f"Forge episode segment: {trajectory.segment}")
    if args.close_support_fingers:
        values = ", ".join(
            f"{name}={HAND_UPPER[index]:g} rad"
            for name, index in zip(SUPPORT_FINGER_JOINTS, SUPPORT_FINGER_INDICES)
        )
        print(f"hand-only override: {values}; no arm retarget applied")
    if args.allow_unsafe_simulation:
        print(
            "WARNING: FR3 limit violations are being sent to the position-JTC "
            "simulation path only; this trajectory must not be sent to hardware"
        )
    if args.finger_flexion_scale != 1.0:
        joints = ", ".join(FINGER_FLEXION_JOINTS)
        print(
            f"hand waypoint flexion scale: {args.finger_flexion_scale:g}x "
            f"for {joints}; homing pose unchanged"
        )
    if not args.no_hand:
        thumb_abduction_zero_rad = kin.open_ratio_to_rad(
            THUMB_ABDUCTION_DOF, THUMB_ABDUCTION_ZERO_OPEN_RATIO
        )
        print(
            "driver thumb abduction overlay: raw replay commands preserved; "
            f"{THUMB_ABDUCTION_JOINT} rescaled onto open ratio "
            f"[{THUMB_ABDUCTION_ZERO_OPEN_RATIO:g}, 1] "
            f"(physical yaw [0, {thumb_abduction_zero_rad:g}] rad)"
        )
    print(
        f"coordinated source: {len(trajectory.time)} samples, "
        f"{trajectory.duration:.2f} s, hand={'yes' if hand_positions is not None else 'no'}"
    )
    if prepared is not None:
        replay_duration = prepared.duration
        print(summarize(prepared, config["joint_names"]))
        if pose_stream is not None:
            print(
                "arm controller: cartesian-impedance; pose stream is forward kinematics "
                f"of the prepared joint stream through tcp offset {pose_stream.tool[:3, 3]} m "
                f"(frame {config['tcp']['frame']}); the joint stream is the nullspace target"
            )
            print(cartesian.summarize_cartesian(pose_stream))
        print(
            "home-to-first-source max arm delta: "
            f"{np.max(np.abs(home_arm - trajectory.arm[0])):.6f} rad"
        )
    else:
        replay_duration = float(stream_time[-1])
        print(
            "hand-only: the arm is neither prepared nor commanded, so no FR3 "
            "limit check applies"
        )
        print(
            f"hand stream: {len(stream_time)} samples at {args.hand_rate:g} Hz "
            f"over {replay_duration:.2f} s (time scale {playback_time_scale:g})"
        )
        print(
            "home-to-first-source max hand delta: "
            f"{np.max(np.abs(home_hand - trajectory.hand[0])):.6f} rad"
        )
    if args.dry_run:
        print("dry run: validated only; no ROS commands sent")
        return 0

    rclpy.init(args=None)
    client_type = (PositionReplayClient if args.arm_controller == "position-jtc"
                   else CoordinatedReplayClient)
    node = client_type(
        config,
        hand_topic,
        hand_state_topic,
    )
    # In Cartesian mode the joint client homes the arm and drives the hand; the
    # Cartesian client takes the arm over for the trajectory. Whichever holds
    # the arm right now is the one a pause or an abort has to reach.
    arm = CartesianReplayClient(config) if cartesian_mode else None
    active_arm = {"node": node}
    executor = MultiThreadedExecutor(num_threads=4 if arm is not None else 3)
    executor.add_node(node)
    if arm is not None:
        executor.add_node(arm)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    exit_code = 0
    hand_thread = None
    hand_stop = threading.Event()
    try:
        if not args.no_arm:
            node.ensure_active(print)
        devices = " and ".join(
            ([] if args.no_arm else ["the FR3"]) + ([] if args.no_hand else ["the Inspire hand"])
        )
        _gate(not args.yes, f"Move {devices} to the YAML home pose?")
        if not args.no_arm:
            node.goto(home_arm)
        if not args.no_hand:
            node.wait_for_hand_link()
            node.command_hand(home_hand)
            node.wait_for_hand(home_hand, args.hand_timeout, args.hand_tolerance)
        print("homing complete")

        if arm is not None:
            # Homed by the validated joint controller; now hand the arm to the
            # Cartesian controller. Both claim the effort interfaces, so the
            # swap keeps franka_hardware in torque control. The preflight refuses
            # a robot whose end-effector frame is not the one the stream assumes.
            if arm.uses_dh_model():
                print(
                    "SIMULATION MODEL: the Cartesian controller computes its pose and "
                    "Jacobian from its built-in DH model (model_source dh), so there is "
                    "no robot frame to check; skipping the F_T_EE / O_T_EE preflight. "
                    "Never run the real arm with model_source dh."
                )
            else:
                arm.preflight(node.current_joint_positions(), pose_stream.tool, print)
            arm.ensure_active(print)
            active_arm["node"] = arm
            if args.stiffness_scale is not None:
                arm.set_stiffness_scale(args.stiffness_scale)
            # A millimetre-scale goto onto the stream's first pose absorbs the
            # at-rest tracking offset of the compliant joint controller; the
            # controller reports idle only once its reference filter has settled.
            arm.goto(pose_stream.p[0], pose_stream.quat[0], pose_stream.q_null[0])
        # The prepared arm stream may start slightly before the source's first
        # point to blend a non-zero initial velocity without a discontinuity.
        elif prepared is not None and np.max(np.abs(prepared.q[0] - home_arm)) > 1e-6:
            node.goto(prepared.q[0])
        if hand_positions is not None:
            node.command_hand(hand_positions[0])

        label = "coordinated" if not (args.no_arm or args.no_hand) else (
            "hand-only" if args.no_arm else "arm-only"
        )
        _gate(not args.yes, f"Replay the {label} {replay_duration:.1f} s trajectory?")
        started = threading.Event()
        hand_errors = []
        trajectory_progress = {"elapsed": 0.0, "command_id": None}
        def trajectory_clock():
            status = active_arm["node"].status()
            if status is not None and status.get("phase_name") == "trajectory":
                trajectory_progress["command_id"] = int(status["active_command_id"])
                trajectory_progress["elapsed"] = max(
                    trajectory_progress["elapsed"], float(status["elapsed"])
                )
            elif (
                status is not None
                and status.get("phase_name") == "idle"
                and trajectory_progress["command_id"] is not None
                and int(status["completed_command_id"]) == trajectory_progress["command_id"]
            ):
                # The status timer may observe the final transition only after
                # the phase has become idle. Release the hand's final sample on
                # normal completion, but not after a newer abort command.
                trajectory_progress["elapsed"] = replay_duration
            return trajectory_progress["elapsed"]

        if hand_positions is not None:
            hand_thread = threading.Thread(
                target=_stream_hand,
                args=(
                    node,
                    started,
                    hand_stop,
                    stream_time,
                    hand_positions,
                    hand_errors,
                    trajectory_clock
                    if not args.no_arm and args.arm_controller != "position-jtc"
                    else None,
                ),
                daemon=True,
            )
            hand_thread.start()
        if prepared is not None:
            keyboard = (_InteractivePause(active_arm["node"], started, hand_stop)
                        if args.interactive_pause else nullcontext())
            with keyboard as controls:
                if arm is not None:
                    arm.send_trajectory(
                        pose_stream,
                        config["cartesian"]["send_rate"],
                        timeout_margin=args.timeout_margin,
                        on_accept=started.set,
                        allow_pauses=args.interactive_pause,
                    )
                else:
                    node.send_trajectory(
                        prepared,
                        config["prepare"]["send_rate"],
                        timeout_margin=args.timeout_margin,
                        on_accept=started.set,
                        allow_pauses=args.interactive_pause,
                    )
            if args.interactive_pause:
                if controls.errors:
                    raise RuntimeError(f"interactive pause failed: {controls.errors[0]}")
                if controls.aborted.is_set():
                    raise Rejected("interactive replay aborted")
        else:
            # With no arm controller to acknowledge a trajectory, there is
            # nothing for the hand thread to wait on: release it immediately.
            started.set()
        if hand_thread is not None:
            hand_thread.join(timeout=replay_duration + args.timeout_margin)
            if hand_thread.is_alive():
                raise TimeoutError("hand replay thread did not finish")
            if hand_errors:
                raise hand_errors[0]
        print(f"{label} trajectory complete")
    except KeyboardInterrupt:
        exit_code = 130
        hand_stop.set()
        if not args.no_arm:
            active_arm["node"].abort()
            print("interrupted: arm abort sent; hand holds its last target")
        else:
            print("interrupted: hand holds its last target")
    except (Rejected, RuntimeError, TimeoutError) as exc:
        exit_code = 1
        hand_stop.set()
        if not args.no_arm:
            active_arm["node"].abort()
        print(f"ERROR: {exc}")
    finally:
        hand_stop.set()
        if hand_thread is not None:
            hand_thread.join(timeout=5.0)
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        if arm is not None:
            arm.destroy_node()
        rclpy.shutdown()
    return exit_code
