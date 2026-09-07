"""Home and replay a coordinated Forge trajectory on an FR3 and Inspire RH56."""

import argparse
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from franka_trajectory_replay.prepare import prepare, summarize
from franka_trajectory_replay import limits
from franka_trajectory_replay.replay_client import Rejected, ReplayClient
from franka_trajectory_replay.runconfig import load_config
from franka_trajectory_replay.trajectory_io import Trajectory as ArmTrajectory
from inspire_hand_driver import kinematics as kin

from .trajectory import ARM_JOINTS, HAND_JOINTS, load_trajectory


# Where each commanded hand joint sits in the driver's own DOF table. Resolved
# by name so this cannot silently follow the wrong channel if either ordering
# is ever changed.
HAND_DOF = tuple(kin.dof_index(name) for name in HAND_JOINTS)

# The joint limits come from that same table rather than a copy: they are what
# the radian-to-ratio conversion below divides by, so a copy that drifted would
# not fail a comparison, it would scale every hand command wrongly.
HAND_LOWER = np.array([kin.DOFS[index].lower for index in HAND_DOF])
HAND_UPPER = np.array([kin.DOFS[index].upper for index in HAND_DOF])


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


def _prepare_arm(trajectory, config, max_duration):
    settings = config["prepare"]
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
        time_scale=settings["time_scale"],
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
    if prepared.report["ok"] or not settings["auto_scale"]:
        return prepared

    required = 1.02 * limits.required_time_scale(prepared.report)
    scaled_duration = trajectory.duration * settings["time_scale"] * required
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
    arguments["time_scale"] = settings["time_scale"] * required
    arguments["auto_scale"] = True
    return prepare(source, **arguments)


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
    values = np.asarray(values, dtype=float)
    low = values.min(axis=0) if values.ndim == 2 else values
    high = values.max(axis=0) if values.ndim == 2 else values
    outside = [
        HAND_JOINTS[index]
        for index in range(6)
        if low[index] < HAND_LOWER[index] - 1e-6
        or high[index] > HAND_UPPER[index] + 1e-6
    ]
    if outside:
        raise ValueError(f"{label} exceeds Inspire hand limits for: {outside}")


class CoordinatedReplayClient(ReplayClient):
    """The proven Franka replay client plus the hand's low-rate position link."""

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

        def arrived():
            with self._hand_lock:
                return self._hand_position is not None and np.max(
                    np.abs(self._hand_position - target)
                ) <= tolerance

        # The driver holds its last target, so repeating it costs nothing and
        # recovers a homing command lost to anything transient on the graph.
        self.wait_until(
            arrived,
            timeout,
            "the Inspire hand to reach its home pose",
            progress=lambda: self.command_hand(target),
        )


def _stream_hand(node, started, stopped, stream_time, positions, errors):
    try:
        deadline = time.monotonic() + 20.0
        while not started.is_set() and time.monotonic() < deadline:
            if stopped.is_set():
                return
            started.wait(timeout=0.05)
        if not started.is_set():
            raise TimeoutError("arm controller did not accept the trajectory")
        epoch_ns = node.get_clock().now().nanoseconds
        for target_time, target in zip(stream_time, positions):
            target_ns = epoch_ns + int(float(target_time) * 1e9)
            while True:
                remaining = (target_ns - node.get_clock().now().nanoseconds) / 1e9
                if remaining <= 0:
                    break
                if stopped.wait(timeout=min(remaining, 0.02)):
                    return
            if stopped.is_set():
                return
            node.command_hand(target)
    except Exception as exc:  # reported by the main thread after the arm stops
        errors.append(exc)


def _gate(enabled, prompt):
    if not enabled:
        return
    input(f"{prompt} [Enter to continue, Ctrl-C to abort] ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="replay_data.npz or its trajectory directory")
    parser.add_argument("--home", default=None, help="homing YAML (default: threading.yaml)")
    parser.add_argument("--config", default=None, help="Franka replay configuration YAML")
    parser.add_argument("--rate", type=float, default=None, help="input rate if absent")
    parser.add_argument("--env", type=int, default=0, help="environment in a batched Forge NPZ")
    parser.add_argument(
        "--cycle",
        type=int,
        default=None,
        help="one rollout cycle from a multi-cycle Forge recording",
    )
    parser.add_argument("--hand-rate", type=float, default=50.0)
    parser.add_argument("--hand-topic", default=None)
    parser.add_argument("--hand-state-topic", default=None)
    parser.add_argument("--hand-timeout", type=float, default=20.0)
    parser.add_argument("--hand-tolerance", type=float, default=0.08)
    parser.add_argument(
        "--max-prepared-duration",
        type=float,
        default=120.0,
        help="reject excessive automatic slow-down before allocating the 1 kHz stream",
    )
    parser.add_argument(
        "--hand-time-scale",
        type=float,
        default=1.0,
        help="stretch (>1) or compress (<1) the hand's own clock; --no-arm only",
    )
    parser.add_argument("--no-hand", action="store_true", help="arm only")
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="hand only: the arm is neither prepared nor commanded",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true", help="skip motion prompts")
    args = parser.parse_args(argv)

    if args.hand_rate <= 0:
        parser.error("--hand-rate must be positive")
    if args.max_prepared_duration <= 0:
        parser.error("--max-prepared-duration must be positive")
    if args.hand_time_scale <= 0:
        parser.error("--hand-time-scale must be positive")
    if args.no_arm and args.no_hand:
        parser.error("--no-arm and --no-hand together leave nothing to replay")
    if args.hand_time_scale != 1.0 and not args.no_arm:
        parser.error("--hand-time-scale only applies to --no-arm; the coordinated "
                     "hand stream follows the arm's prepared clock")
    home_path = args.home or _packaged("threading.yaml")
    config_path = args.config or _packaged("replay.yaml")
    hand_topic = args.hand_topic or "/inspire_hand/command"
    hand_state_topic = args.hand_state_topic or "/inspire_hand/joint_states"

    try:
        trajectory = load_trajectory(args.trajectory, args.rate, args.env, args.cycle)
        home_arm, home_hand = load_home(home_path)
        prepared = None
        if not args.no_arm:
            prepared = _prepare_arm(
                trajectory, load_config(config_path), args.max_prepared_duration
            )
            if not prepared.report["ok"]:
                raise ValueError("prepared arm trajectory violates FR3 limits")
        if trajectory.hand is None and not args.no_hand:
            raise ValueError("trajectory has no Inspire hand positions; pass --no-hand for arm-only")
        if trajectory.hand is not None:
            _validate_hand(trajectory.hand, "trajectory")
        _validate_hand(home_hand, "homing pose")
    except (OSError, ValueError, KeyError) as exc:
        print(f"trajectory error: {exc}")
        return 2

    config = load_config(config_path)
    stream_time, hand_positions = (None, None)
    if not args.no_hand:
        if args.no_arm:
            stream_time, hand_positions = _hand_stream_native(
                trajectory, args.hand_rate, args.hand_time_scale
            )
        else:
            stream_time, hand_positions = _hand_stream(trajectory, prepared, args.hand_rate)

    print(f"source: {trajectory.source}")
    if trajectory.cycle is not None:
        print(f"Forge rollout cycle: {trajectory.cycle}")
    print(
        f"coordinated source: {len(trajectory.time)} samples, "
        f"{trajectory.duration:.2f} s, hand={'yes' if hand_positions is not None else 'no'}"
    )
    if prepared is not None:
        replay_duration = prepared.duration
        print(summarize(prepared, config["joint_names"]))
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
            f"over {replay_duration:.2f} s (time scale {args.hand_time_scale:g})"
        )
        print(
            "home-to-first-source max hand delta: "
            f"{np.max(np.abs(home_hand - trajectory.hand[0])):.6f} rad"
        )
    if args.dry_run:
        print("dry run: validated only; no ROS commands sent")
        return 0

    rclpy.init(args=None)
    node = CoordinatedReplayClient(
        config,
        hand_topic,
        hand_state_topic,
    )
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
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

        # The prepared arm stream may start slightly before the source's first
        # point to blend a non-zero initial velocity without a discontinuity.
        if prepared is not None and np.max(np.abs(prepared.q[0] - home_arm)) > 1e-6:
            node.goto(prepared.q[0])
        if hand_positions is not None:
            node.command_hand(hand_positions[0])

        label = "coordinated" if not (args.no_arm or args.no_hand) else (
            "hand-only" if args.no_arm else "arm-only"
        )
        _gate(not args.yes, f"Replay the {label} {replay_duration:.1f} s trajectory?")
        started = threading.Event()
        hand_errors = []
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
                ),
                daemon=True,
            )
            hand_thread.start()
        if prepared is not None:
            node.send_trajectory(
                prepared,
                config["prepare"]["send_rate"],
                on_accept=started.set,
            )
        else:
            # With no arm controller to acknowledge a trajectory, there is
            # nothing for the hand thread to wait on: release it immediately.
            started.set()
        if hand_thread is not None:
            hand_thread.join(timeout=replay_duration + 5.0)
            if hand_thread.is_alive():
                raise TimeoutError("hand replay thread did not finish")
            if hand_errors:
                raise hand_errors[0]
        print(f"{label} trajectory complete")
    except KeyboardInterrupt:
        exit_code = 130
        hand_stop.set()
        if not args.no_arm:
            node.abort()
            print("interrupted: arm abort sent; hand holds its last target")
        else:
            print("interrupted: hand holds its last target")
    except (Rejected, RuntimeError, TimeoutError) as exc:
        exit_code = 1
        hand_stop.set()
        if not args.no_arm:
            node.abort()
        print(f"ERROR: {exc}")
    finally:
        hand_stop.set()
        if hand_thread is not None:
            hand_thread.join(timeout=5.0)
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()
    return exit_code
