#!/usr/bin/env python3
"""Launch a RealSense D415 and display its color and depth streams.

Run from a terminal in which the ROS 2 workspace has been sourced::

    ./apps/camera_calibration/tests/test_camera.py

Close the Matplotlib window or press Ctrl-C to stop the camera.  If the camera
driver is already running, pass ``--no-launch``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional, Sequence


def _topic(namespace: str, camera_name: str, suffix: str) -> str:
    parts = [
        part.strip("/")
        for part in (namespace, camera_name, suffix)
        if part.strip("/")
    ]
    return "/" + "/".join(parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-namespace", default="camera")
    parser.add_argument("--camera-name", default="camera")
    parser.add_argument(
        "--serial",
        default="",
        help="Optional RealSense serial number when more than one camera is connected.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Subscribe to an already-running camera instead of launching one.",
    )
    parser.add_argument(
        "--initial-reset",
        action="store_true",
        help="Reset the D415 before opening it; use after a stream-start failure.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=2.0,
        metavar="METRES",
        help="Maximum distance shown by the depth color scale (default: 2.0).",
    )
    parser.add_argument(
        "--color-profile",
        default="640x480x30",
        help="RealSense color profile WIDTHxHEIGHTxFPS (default: 640x480x30).",
    )
    parser.add_argument(
        "--depth-profile",
        default="640x480x30",
        help="RealSense depth profile WIDTHxHEIGHTxFPS (default: 640x480x30).",
    )
    parser.add_argument(
        "--viewer-hz",
        type=float,
        default=30.0,
        metavar="HZ",
        help="Maximum GUI refresh rate; this does not change camera FPS (default: 30).",
    )
    return parser


def _start_camera(args: argparse.Namespace) -> Optional[subprocess.Popen[bytes]]:
    if args.no_launch:
        return None

    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError(
            "ros2 was not found. Source /opt/ros/$ROS_DISTRO/setup.bash and this "
            "workspace's install/setup.bash before running the test."
        )
    package_check = subprocess.run(
        [ros2, "pkg", "prefix", "realsense2_camera"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if package_check.returncode != 0:
        workspace = Path(__file__).resolve().parents[3]
        raise RuntimeError(
            "ROS 2 cannot find realsense2_camera. In this terminal run:\n"
            f"  source {workspace}/install/setup.bash\n"
            "Then start this test again."
        )

    camera_node = _topic(args.camera_namespace, args.camera_name, "")
    try:
        node_list = subprocess.run(
            [ros2, "node", "list"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        node_list = None
    if node_list is not None and camera_node in node_list.stdout.splitlines():
        raise RuntimeError(
            f"A camera node is already running at {camera_node}. "
            "Use --no-launch to view its streams instead of starting a second "
            "RealSense driver."
        )

    command = [
        ros2,
        "launch",
        "realsense2_camera",
        "rs_launch.py",
        "device_type:=d415",
        f"camera_namespace:={args.camera_namespace}",
        f"camera_name:={args.camera_name}",
        "enable_color:=true",
        "enable_depth:=true",
        f"rgb_camera.color_profile:={args.color_profile}",
        f"depth_module.depth_profile:={args.depth_profile}",
    ]
    if args.initial_reset:
        command.append("initial_reset:=true")
    if args.serial:
        # An underscore forces an all-numeric serial to remain a ROS string.
        serial = args.serial
        command.append(f"serial_no:={serial if serial.startswith('_') else '_' + serial}")

    # Keep cleanup outside the GUI process as native libraries can terminate it
    # without running Python's finally block.
    guard = Path(__file__).resolve().parents[2] / "process_guard.py"
    command = [sys.executable, str(guard), *command]

    print("Starting RealSense D415...", flush=True)
    return subprocess.Popen(command, start_new_session=True)


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


def _stop_camera(process: Optional[subprocess.Popen[bytes]]) -> None:
    if process is None:
        return

    process_group = process.pid
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
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.depth_max <= 0.0:
        _parser().error("--depth-max must be greater than zero")
    if args.viewer_hz <= 0.0:
        _parser().error("--viewer-hz must be greater than zero")

    # Keep --help usable on hosts where ROS or GUI dependencies are unavailable.
    import matplotlib.pyplot as plt
    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image

    camera_process = None
    node = None
    executor = None
    spin_thread = None
    stop_spinning = threading.Event()
    shutdown_requested = threading.Event()
    previous_sigint_handler = None
    frame_lock = threading.Lock()
    frames: dict[str, Optional[np.ndarray]] = {"color": None, "depth": None}
    bridge = CvBridge()

    def color_callback(message: Image) -> None:
        try:
            frame = bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
        except Exception as exc:  # Keep the viewer alive after a malformed frame.
            if node is not None:
                node.get_logger().error(f"Could not convert color frame: {exc}")
            return
        with frame_lock:
            frames["color"] = np.asarray(frame).copy()

    def depth_callback(message: Image) -> None:
        try:
            frame = np.asarray(
                bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            # RealSense Z16 is millimetres; 32FC1 is already metres.
            depth_metres = (
                frame.astype(np.float32) * 0.001
                if message.encoding.lower() in {"16uc1", "mono16"}
                else frame.astype(np.float32)
            )
            depth_metres[~np.isfinite(depth_metres)] = 0.0
        except Exception as exc:
            if node is not None:
                node.get_logger().error(f"Could not convert depth frame: {exc}")
            return
        with frame_lock:
            frames["depth"] = depth_metres

    def request_shutdown(_signum: int, _frame: object) -> None:
        shutdown_requested.set()

    try:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_shutdown)
        # Start inside the protected region so Ctrl-C during ROS/node setup still
        # tears down the complete launch process group.
        camera_process = _start_camera(args)
        # The script's CLI arguments are for argparse, not for ROS remapping.
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("realsense_matplotlib_viewer")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        color_topic = _topic(
            args.camera_namespace, args.camera_name, "color/image_raw"
        )
        depth_topic = _topic(
            args.camera_namespace, args.camera_name, "depth/image_rect_raw"
        )
        subscriptions = [
            node.create_subscription(
                Image, color_topic, color_callback, qos_profile_sensor_data
            ),
            node.create_subscription(
                Image, depth_topic, depth_callback, qos_profile_sensor_data
            ),
        ]
        node.get_logger().info(f"Waiting for RGB frames on {color_topic}")
        node.get_logger().info(f"Waiting for depth frames on {depth_topic}")

        def spin() -> None:
            try:
                while not stop_spinning.is_set() and rclpy.ok():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass

        spin_thread = threading.Thread(target=spin, name="ros-spin", daemon=True)
        spin_thread.start()

        figure, (color_axis, depth_axis) = plt.subplots(1, 2, figsize=(13, 6))
        figure.canvas.manager.set_window_title("RealSense RGB + Depth")
        color_axis.set_title("RGB — waiting for frames")
        depth_axis.set_title("Depth — waiting for frames")
        color_axis.axis("off")
        depth_axis.axis("off")

        color_artist = color_axis.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        depth_artist = depth_axis.imshow(
            np.zeros((480, 640), dtype=np.float32),
            cmap="turbo",
            vmin=0.0,
            vmax=args.depth_max,
        )
        colorbar = figure.colorbar(depth_artist, ax=depth_axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Distance (m)")
        figure.tight_layout()

        def update_view() -> None:
            if shutdown_requested.is_set():
                plt.close(figure)
                return

            with frame_lock:
                color = None if frames["color"] is None else frames["color"].copy()
                depth = None if frames["depth"] is None else frames["depth"].copy()

            if color is not None:
                color_artist.set_data(color)
                color_axis.set_title(f"RGB ({color.shape[1]} × {color.shape[0]})")
            if depth is not None:
                # Mask invalid zero readings so they appear dark/transparent.
                depth_artist.set_data(np.ma.masked_less_equal(depth, 0.0))
                depth_axis.set_title(f"Depth ({depth.shape[1]} × {depth.shape[0]})")

            if camera_process is not None and camera_process.poll() is not None:
                figure.suptitle(
                    f"Camera launch exited with code {camera_process.returncode}",
                    color="red",
                )
            figure.canvas.draw_idle()

        timer = figure.canvas.new_timer(interval=max(1, round(1000.0 / args.viewer_hz)))
        timer.add_callback(update_view)
        timer.start()
        plt.show()

        # Keep references alive for the lifetime of the node and GUI.
        del subscriptions, timer
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        stop_spinning.set()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _stop_camera(camera_process)
        if previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    raise SystemExit(main())
