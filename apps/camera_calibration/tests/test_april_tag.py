#!/usr/bin/env python3
"""Launch a D415, detect an AprilTag, and draw its frame on RGB and depth.

The default target matches the calibration package: tag36h11, ID 0, with a
40 mm black-square edge.  Run this from a shell where the workspace is sourced::

    ./apps/camera_calibration/tests/test_april_tag.py

The displayed axes follow OpenCV's convention: X is red, Y green, and Z blue.
Close the Matplotlib window or press Ctrl-C to stop the camera.
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


APRILTAG_FAMILIES = ("tag16h5", "tag25h9", "tag36h10", "tag36h11")


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
        help="Optional RealSense serial number when multiple cameras are connected.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help=(
            "Use an existing camera node. Its align_depth filter must be enabled."
        ),
    )
    parser.add_argument(
        "--initial-reset",
        action="store_true",
        help="Reset the D415 before opening it; use after a stream-start failure.",
    )
    parser.add_argument(
        "--tag-family",
        choices=APRILTAG_FAMILIES,
        default="tag36h11",
    )
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.040,
        metavar="METRES",
        help="Measured black-square edge length (default: 0.040 m).",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.030,
        metavar="METRES",
        help="Length of each displayed coordinate axis (default: 0.030 m).",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=2.0,
        metavar="METRES",
        help="Maximum distance represented by the depth colors (default: 2.0 m).",
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
            "Use --no-launch to detect tags from its streams instead of starting "
            "a second RealSense driver."
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
        "enable_sync:=true",
        "align_depth.enable:=true",
    ]
    if args.initial_reset:
        command.append("initial_reset:=true")
    if args.serial:
        # Preserve an all-numeric serial as a ROS string parameter.
        serial = args.serial
        command.append(f"serial_no:={serial if serial.startswith('_') else '_' + serial}")

    # Keep cleanup outside the GUI process as native libraries can terminate it
    # without running Python's finally block.
    guard = Path(__file__).resolve().parents[2] / "process_guard.py"
    command = [sys.executable, str(guard), *command]

    print("Starting RealSense D415 with aligned depth...", flush=True)
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
    parser = _parser()
    args = parser.parse_args(argv)
    if args.tag_size <= 0.0:
        parser.error("--tag-size must be greater than zero")
    if args.axis_length <= 0.0:
        parser.error("--axis-length must be greater than zero")
    if args.depth_max <= 0.0:
        parser.error("--depth-max must be greater than zero")
    if args.viewer_hz <= 0.0:
        parser.error("--viewer-hz must be greater than zero")

    # Delay optional imports so --help works even outside the ROS environment.
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import rclpy
    from camera_calibration.apriltag_detector import AprilTagDetector
    from cv_bridge import CvBridge
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CameraInfo, Image

    detector = AprilTagDetector(args.tag_family)

    half = args.tag_size * 0.5
    object_corners = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    camera_process = None
    node = None
    executor = None
    spin_thread = None
    stop_spinning = threading.Event()
    shutdown_requested = threading.Event()
    previous_sigint_handler = None
    state_lock = threading.Lock()
    bridge = CvBridge()

    pending_color: Optional[np.ndarray] = None
    pending_depth: Optional[np.ndarray] = None
    color_frame: Optional[np.ndarray] = None
    depth_frame: Optional[np.ndarray] = None
    camera_matrix: Optional[np.ndarray] = None
    distortion: Optional[np.ndarray] = None
    camera_size: Optional[tuple[int, int]] = None
    tag_pose: Optional[tuple[np.ndarray, np.ndarray]] = None
    tag_corners: Optional[np.ndarray] = None
    detection_status = "waiting for color image and CameraInfo"

    def draw_text(
        image: np.ndarray, text: str, color: tuple[int, int, int]
    ) -> None:
        cv2.putText(
            image,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    def camera_info_callback(message: CameraInfo) -> None:
        nonlocal camera_matrix, distortion, camera_size
        if message.width == 0 or message.height == 0:
            return
        matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        coefficients = np.asarray(message.d, dtype=np.float64)
        with state_lock:
            camera_matrix = matrix
            distortion = coefficients
            camera_size = (int(message.width), int(message.height))

    def process_color(image: np.ndarray) -> None:
        nonlocal color_frame, tag_pose, tag_corners, detection_status
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)
        for detection in detections:
            outline = np.rint(detection.corners).astype(np.int32)
            cv2.polylines(image, [outline], True, (0, 255, 0), 2, cv2.LINE_AA)
            center = np.rint(np.mean(detection.corners, axis=0)).astype(int)
            cv2.putText(
                image,
                str(detection.identifier),
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        with state_lock:
            matrix = None if camera_matrix is None else camera_matrix.copy()
            coefficients = None if distortion is None else distortion.copy()

        pose = None
        selected_corners = None
        status = f"tag {args.tag_id} not detected"
        text_color = (0, 0, 255)

        matches = [
            detection
            for detection in detections
            if detection.identifier == args.tag_id
        ]
        if matches:
            selected_corners = matches[0].corners
            if matrix is None:
                status = f"tag {args.tag_id} detected; waiting for CameraInfo"
                text_color = (0, 200, 255)
            else:
                success, rotation, translation = cv2.solvePnP(
                    object_corners,
                    selected_corners,
                    matrix,
                    coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if success and float(translation.reshape(3)[2]) > 0.0:
                    projected, _ = cv2.projectPoints(
                        object_corners,
                        rotation,
                        translation,
                        matrix,
                        coefficients,
                    )
                    error = float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (
                                        projected.reshape(4, 2)
                                        - selected_corners
                                    )
                                    ** 2,
                                    axis=1,
                                )
                            )
                        )
                    )
                    cv2.drawFrameAxes(
                        image,
                        matrix,
                        coefficients,
                        rotation,
                        translation,
                        args.axis_length,
                        3,
                    )
                    xyz = translation.reshape(3)
                    status = (
                        f"tag {args.tag_id}: x={xyz[0]:+.3f}, "
                        f"y={xyz[1]:+.3f}, z={xyz[2]:.3f} m; "
                        f"error={error:.2f}px"
                    )
                    text_color = (0, 220, 0)
                    pose = (rotation.copy(), translation.copy())
                else:
                    status = f"tag {args.tag_id} detected; pose failed"

        draw_text(image, status, text_color)
        with state_lock:
            color_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tag_pose = pose
            tag_corners = (
                None if selected_corners is None else selected_corners.copy()
            )
            detection_status = status

    def color_callback(message: Image) -> None:
        nonlocal pending_color
        try:
            image = np.asarray(
                bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ).copy()
        except Exception as exc:
            if node is not None:
                node.get_logger().error(f"Could not decode color image: {exc}")
            return
        with state_lock:
            pending_color = image

    def process_depth(depth_metres: np.ndarray) -> None:
        nonlocal depth_frame
        normalized = np.clip(
            depth_metres * (255.0 / args.depth_max), 0.0, 255.0
        ).astype(np.uint8)
        annotation = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        annotation[depth_metres <= 0.0] = 0

        with state_lock:
            pose = (
                None
                if tag_pose is None
                else (tag_pose[0].copy(), tag_pose[1].copy())
            )
            corners = None if tag_corners is None else tag_corners.copy()
            matrix = None if camera_matrix is None else camera_matrix.copy()
            coefficients = None if distortion is None else distortion.copy()
            source_size = camera_size
            status = detection_status

        if matrix is not None and source_size is not None:
            source_width, source_height = source_size
            scale_x = annotation.shape[1] / source_width
            scale_y = annotation.shape[0] / source_height
            depth_matrix = matrix.copy()
            depth_matrix[0, 0] *= scale_x
            depth_matrix[0, 2] *= scale_x
            depth_matrix[1, 1] *= scale_y
            depth_matrix[1, 2] *= scale_y

            if corners is not None:
                scaled_corners = corners.copy()
                scaled_corners[:, 0] *= scale_x
                scaled_corners[:, 1] *= scale_y
                cv2.polylines(
                    annotation,
                    [np.rint(scaled_corners).astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            if pose is not None:
                cv2.drawFrameAxes(
                    annotation,
                    depth_matrix,
                    coefficients,
                    pose[0],
                    pose[1],
                    args.axis_length,
                    3,
                )

        draw_text(annotation, status, (0, 220, 0) if pose is not None else (0, 0, 255))
        with state_lock:
            depth_frame = cv2.cvtColor(annotation, cv2.COLOR_BGR2RGB)

    def depth_callback(message: Image) -> None:
        nonlocal pending_depth
        try:
            raw_depth = np.asarray(
                bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            depth_metres = (
                raw_depth.astype(np.float32) * 0.001
                if message.encoding.lower() in {"16uc1", "mono16"}
                else raw_depth.astype(np.float32)
            )
            depth_metres[~np.isfinite(depth_metres)] = 0.0
        except Exception as exc:
            if node is not None:
                node.get_logger().error(f"Could not decode depth image: {exc}")
            return
        with state_lock:
            pending_depth = depth_metres

    def request_shutdown(_signum: int, _frame: object) -> None:
        shutdown_requested.set()

    try:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_shutdown)
        # Start inside the protected region so Ctrl-C during ROS/node setup still
        # tears down the complete launch process group.
        camera_process = _start_camera(args)
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("apriltag_camera_viewer")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        color_topic = _topic(
            args.camera_namespace, args.camera_name, "color/image_raw"
        )
        camera_info_topic = _topic(
            args.camera_namespace, args.camera_name, "color/camera_info"
        )
        depth_topic = _topic(
            args.camera_namespace,
            args.camera_name,
            "aligned_depth_to_color/image_raw",
        )
        subscriptions = [
            node.create_subscription(
                CameraInfo,
                camera_info_topic,
                camera_info_callback,
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                Image, color_topic, color_callback, qos_profile_sensor_data
            ),
            node.create_subscription(
                Image, depth_topic, depth_callback, qos_profile_sensor_data
            ),
        ]
        node.get_logger().info(
            f"Looking for {args.tag_family} tag {args.tag_id} "
            f"({args.tag_size:.4f} m)"
        )
        node.get_logger().info(f"RGB: {color_topic}")
        node.get_logger().info(f"Aligned depth: {depth_topic}")
        node.get_logger().info(f"Intrinsics: {camera_info_topic}")

        def spin() -> None:
            try:
                while not stop_spinning.is_set() and rclpy.ok():
                    executor.spin_once(timeout_sec=0.1)
            except ExternalShutdownException:
                pass

        spin_thread = threading.Thread(target=spin, name="ros-spin", daemon=True)
        spin_thread.start()

        figure, (color_axis, depth_axis) = plt.subplots(1, 2, figsize=(14, 6))
        figure.canvas.manager.set_window_title("RealSense AprilTag pose")
        color_axis.set_title("RGB — waiting for frames")
        depth_axis.set_title("Aligned depth — waiting for frames")
        color_axis.axis("off")
        depth_axis.axis("off")
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        color_artist = color_axis.imshow(placeholder)
        depth_artist = depth_axis.imshow(placeholder)
        colorbar = figure.colorbar(
            ScalarMappable(
                norm=Normalize(vmin=0.0, vmax=args.depth_max), cmap="turbo"
            ),
            ax=depth_axis,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("Distance (m)")
        figure.text(
            0.5,
            0.01,
            "Coordinate frame: X red  |  Y green  |  Z blue",
            ha="center",
        )
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

        def update_view() -> None:
            nonlocal pending_color, pending_depth
            if shutdown_requested.is_set():
                plt.close(figure)
                return

            with state_lock:
                next_color = pending_color
                next_depth = pending_depth
                pending_color = None
                pending_depth = None

            # Keep OpenCV's ArUco and drawing calls on the GUI thread. OpenCV
            # 4.6 can segfault when detection runs concurrently with TkAgg.
            if next_color is not None:
                process_color(next_color)
            if next_depth is not None:
                process_depth(next_depth)

            with state_lock:
                color = None if color_frame is None else color_frame.copy()
                depth = None if depth_frame is None else depth_frame.copy()

            if color is not None:
                color_artist.set_data(color)
                color_axis.set_title(
                    f"RGB + tag pose ({color.shape[1]} × {color.shape[0]})"
                )
            if depth is not None:
                depth_artist.set_data(depth)
                depth_axis.set_title(
                    f"Aligned depth + tag pose ({depth.shape[1]} × {depth.shape[0]})"
                )
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

        # These references intentionally remain alive until the window closes.
        _ = subscriptions, timer
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
