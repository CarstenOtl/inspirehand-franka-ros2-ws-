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
import threading
from typing import Optional, Sequence


APRILTAG_DICTIONARIES = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


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
        "--tag-family",
        choices=tuple(APRILTAG_DICTIONARIES),
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
    return parser


def _start_camera(args: argparse.Namespace) -> Optional[subprocess.Popen[bytes]]:
    if args.no_launch:
        return None

    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError(
            "ros2 was not found. Source /opt/ros/humble/setup.bash and this "
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
        "enable_sync:=true",
        "align_depth.enable:=true",
    ]
    if args.serial:
        # Preserve an all-numeric serial as a ROS string parameter.
        serial = args.serial
        command.append(f"serial_no:={serial if serial.startswith('_') else '_' + serial}")

    print("Starting RealSense D415 with aligned depth...", flush=True)
    return subprocess.Popen(command, start_new_session=True)


def _stop_camera(process: Optional[subprocess.Popen[bytes]]) -> None:
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 8.0), (signal.SIGTERM, 3.0)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
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

    # Delay optional imports so --help works even outside the ROS environment.
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV has no aruco module; install the workspace's python3-opencv dependency"
        )

    dictionary_id = getattr(
        cv2.aruco, APRILTAG_DICTIONARIES[args.tag_family]
    )
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector_parameters = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )
    detector = (
        cv2.aruco.ArucoDetector(dictionary, detector_parameters)
        if hasattr(cv2.aruco, "ArucoDetector")
        else None
    )

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

    camera_process = _start_camera(args)
    node = None
    spin_thread = None
    stop_spinning = threading.Event()
    state_lock = threading.Lock()
    bridge = CvBridge()

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

    def detect_markers(gray_image: np.ndarray):
        if detector is not None:
            return detector.detectMarkers(gray_image)
        return cv2.aruco.detectMarkers(
            gray_image, dictionary, parameters=detector_parameters
        )

    def color_callback(message: Image) -> None:
        nonlocal color_frame, tag_pose, tag_corners, detection_status
        try:
            image = np.asarray(
                bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ).copy()
        except Exception as exc:
            if node is not None:
                node.get_logger().error(f"Could not decode color image: {exc}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, identifiers, _ = detect_markers(gray)
        if identifiers is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, identifiers)

        with state_lock:
            matrix = None if camera_matrix is None else camera_matrix.copy()
            coefficients = None if distortion is None else distortion.copy()

        pose = None
        selected_corners = None
        status = f"tag {args.tag_id} not detected"
        text_color = (0, 0, 255)

        if identifiers is not None:
            matches = np.flatnonzero(identifiers.reshape(-1) == args.tag_id)
            if len(matches):
                selected_corners = np.asarray(
                    corners[int(matches[0])], dtype=np.float64
                ).reshape(4, 2)
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

    def depth_callback(message: Image) -> None:
        nonlocal depth_frame
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

    try:
        rclpy.init(args=[])
        node = rclpy.create_node("apriltag_camera_viewer")
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
            while not stop_spinning.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

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

        timer = figure.canvas.new_timer(interval=50)
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
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _stop_camera(camera_process)


if __name__ == "__main__":
    raise SystemExit(main())
