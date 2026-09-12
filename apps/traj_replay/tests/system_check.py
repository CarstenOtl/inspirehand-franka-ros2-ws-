#!/usr/bin/env python3
"""Bring up the real system and wait for live telemetry from every device.

Run this from a shell in which the workspace has been built and sourced::

    ./apps/traj_replay/tests/system_check.py --robot-ip 172.16.0.2

By default the check does not send motion commands.  It launches the FR3 and
Inspire hand through ``inspire_franka_bringup``, launches the RealSense wrapper
constrained to a D415, and requires fresh telemetry from all three devices.  Use
``--gravity-compensation`` to opt into Franka's zero-effort controller for hand
guiding.  On success it keeps the launch processes alive until Ctrl-C; use
``--exit-after-check`` for a one-shot test which tears them down after printing
the report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable, Optional, Sequence


@dataclass
class DeviceStatus:
    """Latest result for one hardware subsystem."""

    name: str
    ready: bool = False
    detail: str = "waiting for telemetry"
    elapsed: Optional[float] = None


def _process_group_is_running(process_group: int) -> bool:
    """Return whether a non-zombie process still belongs to the launch group."""
    proc = Path("/proc")
    if proc.is_dir():
        for stat_path in proc.glob("[0-9]*/stat"):
            try:
                fields = stat_path.read_text().rsplit(")", 1)[1].split()
                state, group = fields[0], int(fields[2])
            except (IndexError, OSError, ValueError):
                continue
            if group == process_group and state != "Z":
                return True
        return False

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


class ManagedLaunch:
    """A ROS launch process which can be stopped together with its children."""

    def __init__(self, label: str, command: Sequence[str]) -> None:
        self.label = label
        self.command = list(command)
        self.process: Optional[subprocess.Popen[bytes]] = None

    def start(self) -> None:
        print(f"Starting {self.label}: {shlex.join(self.command)}", flush=True)
        guard = Path(__file__).resolve().parents[2] / "process_guard.py"
        command = [sys.executable, str(guard), *self.command]
        self.process = subprocess.Popen(command, start_new_session=True)

    def returncode(self) -> Optional[int]:
        return None if self.process is None else self.process.poll()

    def stop(self) -> None:
        if self.process is None:
            return

        process_group = self.process.pid
        for sig, timeout in ((signal.SIGINT, 8.0), (signal.SIGTERM, 3.0)):
            if not _process_group_is_running(process_group):
                break
            try:
                os.killpg(process_group, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + timeout
            while (
                _process_group_is_running(process_group)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)

        if _process_group_is_running(process_group):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _topic(namespace: str, node_name: str, suffix: str) -> str:
    parts = [part.strip("/") for part in (namespace, node_name, suffix) if part.strip("/")]
    return "/" + "/".join(parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-ip",
        default=os.environ.get("FR3_ROBOT_IP", ""),
        help="FR3 FCI hostname/IP (or set FR3_ROBOT_IP). Required for real hardware.",
    )
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument(
        "--hand-protocol", choices=("modbus", "legacy"), default="modbus"
    )
    parser.add_argument("--camera-serial", default="", help="Optional D415 serial number.")
    parser.add_argument("--camera-namespace", default="camera")
    parser.add_argument("--camera-name", default="camera")
    parser.add_argument("--color-profile", default="640x480x30")
    parser.add_argument("--depth-profile", default="640x480x30")
    parser.add_argument(
        "--camera-initial-reset",
        action="store_true",
        help="Reset the D415 before opening it; use after a depth-stream failure.",
    )
    parser.add_argument("--arm-prefix", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--fake-arm", action="store_true", help="Use ros2_control fake hardware."
    )
    parser.add_argument(
        "--gravity-compensation",
        action="store_true",
        help="Start Franka's zero-effort gravity compensation controller.",
    )
    parser.add_argument(
        "--mock-hand", action="store_true", help="Use the hand's mock transport."
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Only inspect an already-running ROS graph; do not start devices.",
    )
    parser.add_argument(
        "--exit-after-check",
        action="store_true",
        help="Stop processes and exit after the report instead of staying up.",
    )
    return parser


def _launch_commands(args: argparse.Namespace) -> list[ManagedLaunch]:
    bringup = [
        "ros2",
        "launch",
        "inspire_franka_bringup",
        "inspire_franka.launch.py",
        f"use_fake_hardware:={'true' if args.fake_arm else 'false'}",
        f"gravity_compensation:={'true' if args.gravity_compensation else 'false'}",
        f"hand_port:={args.hand_port}",
        f"hand_id:={args.hand_id}",
        f"hand_protocol:={args.hand_protocol}",
        f"hand_mock:={'true' if args.mock_hand else 'false'}",
        "start_rviz:=false",
    ]
    if args.robot_ip:
        bringup.append(f"robot_ip:={args.robot_ip}")
    if args.arm_prefix:
        bringup.append(f"arm_prefix:={args.arm_prefix}")
    camera = [
        "ros2",
        "launch",
        "realsense2_camera",
        "rs_launch.py",
        "device_type:=d415",
        f"camera_namespace:={args.camera_namespace}",
        f"camera_name:={args.camera_name}",
        "enable_depth:=true",
        "enable_color:=true",
        f"rgb_camera.color_profile:={args.color_profile}",
        f"depth_module.depth_profile:={args.depth_profile}",
        f"wait_for_device_timeout:={args.timeout}",
    ]
    if args.camera_initial_reset:
        camera.append("initial_reset:=true")
    if args.camera_serial:
        # The upstream launch file requires an underscore to stop ROS from
        # interpreting an all-numeric serial as an integer parameter.
        serial = args.camera_serial
        camera.append(f"serial_no:={serial if serial.startswith('_') else '_' + serial}")
    return [ManagedLaunch("FR3 + Inspire hand", bringup), ManagedLaunch("D415", camera)]


def _print_report(statuses: Sequence[DeviceStatus], ok: bool) -> None:
    print("\nSystem check", flush=True)
    print("------------", flush=True)
    for status in statuses:
        result = "PASS" if status.ready else "FAIL"
        timing = f" ({status.elapsed:.2f}s)" if status.elapsed is not None else ""
        print(f"{status.name:<18} {result:<4}  {status.detail}{timing}", flush=True)
    print(f"Overall            {'PASS' if ok else 'FAIL'}", flush=True)


def _observe(args: argparse.Namespace, launches: Sequence[ManagedLaunch]) -> bool:
    # Import after argument/preflight handling so --help remains useful even in
    # a host shell which has not sourced ROS.
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    started = time.monotonic()
    arm = DeviceStatus("FR3")
    hand = DeviceStatus("Inspire hand")
    depth = DeviceStatus("D415 depth")
    color = DeviceStatus("D415 color")
    statuses = [arm, hand, depth, color]

    expected_arm_joints = {f"{args.arm_prefix}fr3_joint{i}" for i in range(1, 8)}

    def mark(status: DeviceStatus, detail: str) -> None:
        if not status.ready:
            status.ready = True
            status.detail = detail
            status.elapsed = time.monotonic() - started

    def arm_callback(message: JointState) -> None:
        present = expected_arm_joints.intersection(message.name)
        if present == expected_arm_joints:
            mark(arm, "live joint state for all 7 joints")
        elif present:
            arm.detail = f"only {len(present)}/7 expected joints received"

    def hand_callback(message: JointState) -> None:
        channels = set(message.name)
        values_ok = len(message.position) >= 6 and all(
            math.isfinite(value) for value in message.position[:6]
        )
        if set("123456").issubset(channels) and values_ok:
            mark(hand, "live state for all 6 actuator channels")
        else:
            hand.detail = "state received, but channel/value data is incomplete"

    def image_callback(status: DeviceStatus, stream: str) -> Callable[[Image], None]:
        def callback(message: Image) -> None:
            if message.width > 0 and message.height > 0 and message.data:
                mark(status, f"live {stream} frames ({message.width}x{message.height})")
            else:
                status.detail = f"{stream} message received, but frame is empty"

        return callback

    camera_base = (args.camera_namespace, args.camera_name)
    node = None
    subscriptions = []
    # argparse already consumed this script's options. Do not let rclpy parse
    # --robot-ip, --exit-after-check, and the other non-ROS arguments again.
    rclpy.init(args=[])
    try:
        node = rclpy.create_node("traj_replay_system_check")
        subscriptions = [
            node.create_subscription(
                JointState, "/joint_states", arm_callback, qos_profile_sensor_data
            ),
            node.create_subscription(
                JointState,
                "/inspire_hand/state",
                hand_callback,
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                Image,
                _topic(*camera_base, "depth/image_rect_raw"),
                image_callback(depth, "depth"),
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                Image,
                _topic(*camera_base, "color/image_raw"),
                image_callback(color, "color"),
                qos_profile_sensor_data,
            ),
        ]

        deadline = started + args.timeout
        next_progress = started
        while time.monotonic() < deadline and not all(status.ready for status in statuses):
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if now >= next_progress:
                summary = ", ".join(
                    f"{status.name}={'ready' if status.ready else 'waiting'}"
                    for status in statuses
                )
                print(f"[{now - started:5.1f}s] {summary}", flush=True)
                next_progress = now + 2.0

            exited = [
                f"{launch.label} (exit {launch.returncode()})"
                for launch in launches
                if launch.returncode() is not None
            ]
            if exited:
                reason = "launch exited: " + ", ".join(exited)
                for status in statuses:
                    if not status.ready:
                        status.detail = reason
                break
    finally:
        # Retain the subscriptions for the full observation window.
        subscriptions.clear()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    for status in statuses:
        if not status.ready and status.detail == "waiting for telemetry":
            status.detail = f"no valid telemetry within {args.timeout:g}s"

    ok = all(status.ready for status in statuses)
    _print_report(statuses, ok)
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not args.no_launch and not args.fake_arm and not args.robot_ip:
        parser.error("--robot-ip (or FR3_ROBOT_IP) is required for the real FR3")
    if shutil.which("ros2") is None:
        print("ERROR: ros2 is not on PATH; source ROS and the workspace first.", file=sys.stderr)
        return 2

    launches: list[ManagedLaunch] = [] if args.no_launch else _launch_commands(args)
    try:
        for launch in launches:
            launch.start()
        ok = _observe(args, launches)
        if not ok:
            return 1

        if launches and not args.exit_after_check:
            print("\nAll systems are enabled. Press Ctrl-C to stop them.", flush=True)
            while all(launch.returncode() is None for launch in launches):
                time.sleep(0.5)
            stopped = [
                f"{launch.label} (exit {launch.returncode()})"
                for launch in launches
                if launch.returncode() is not None
            ]
            print("ERROR: " + ", ".join(stopped), file=sys.stderr)
            return 1
        return 0
    except KeyboardInterrupt:
        print("\nStopping system check...", flush=True)
        return 130
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        for launch in reversed(launches):
            launch.stop()


if __name__ == "__main__":
    raise SystemExit(main())
