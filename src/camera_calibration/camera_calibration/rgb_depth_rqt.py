"""rqt plugin showing the RealSense color and depth streams side by side."""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Optional, Sequence

import cv2
from cv_bridge import CvBridge
import numpy as np
from python_qt_binding.QtCore import QTimer, Qt
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from rclpy.qos import qos_profile_sensor_data
from rqt_gui_py.plugin import Plugin
from sensor_msgs.msg import Image


def _topic(namespace: str, camera_name: str, suffix: str) -> str:
    parts = [
        part.strip("/")
        for part in (namespace, camera_name, suffix)
        if part.strip("/")
    ]
    return "/" + "/".join(parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgbd_view",
        description="Show the RealSense RGB and depth topics side by side.",
    )
    _add_arguments(parser)
    return parser


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-namespace", default="camera")
    parser.add_argument("--camera-name", default="camera")
    parser.add_argument(
        "--depth-max",
        type=float,
        default=2.0,
        metavar="METRES",
        help=(
            "Maximum distance represented by the depth colors "
            "(default: 2.0)."
        ),
    )


def _qimage(rgb: np.ndarray) -> QImage:
    """Return a detached QImage which owns its RGB pixel buffer."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        int(rgb.strides[0]),
        QImage.Format_RGB888,
    ).copy()


def _colorize_depth(depth_metres: np.ndarray, depth_max: float) -> np.ndarray:
    """Convert a metric depth image to an RGB turbo-colormap image."""
    valid = np.isfinite(depth_metres) & (depth_metres > 0.0)
    scaled = np.zeros(depth_metres.shape, dtype=np.uint8)
    scaled[valid] = np.clip(
        depth_metres[valid] * (255.0 / depth_max), 0.0, 255.0
    ).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


class _ImagePanel(QWidget):
    def __init__(self, title: str, waiting_text: str) -> None:
        super().__init__()
        self._image: Optional[QImage] = None

        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.image = QLabel(waiting_text)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(320, 240)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.status = QLabel(waiting_text)
        self.status.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.status)

    def set_frame(self, frame: QImage, status: str) -> None:
        self._image = frame
        self.status.setText(status)
        self._scale_frame()

    def _scale_frame(self) -> None:
        if self._image is None:
            return
        self.image.setPixmap(
            QPixmap.fromImage(self._image).scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        super().resizeEvent(event)
        self._scale_frame()


class RgbDepthView(Plugin):
    """An rqt plugin with automatic D415 RGB and depth subscriptions."""

    def __init__(self, context) -> None:
        """Create the viewer and subscribe to the configured image topics."""
        super().__init__(context)
        self.setObjectName("RgbDepthView")
        self._context = context
        self._node = context.node
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None

        args, _unknown = _parser().parse_known_args(context.argv())
        if args.depth_max <= 0.0:
            raise ValueError("--depth-max must be greater than zero")
        self._depth_max = args.depth_max
        self._color_topic = _topic(
            args.camera_namespace, args.camera_name, "color/image_raw"
        )
        self._depth_topic = _topic(
            args.camera_namespace, args.camera_name, "depth/image_rect_raw"
        )

        self._widget = QWidget()
        self._widget.setWindowTitle("RealSense RGB + Depth")
        self._color_panel = _ImagePanel(
            "RGB", f"Waiting for {self._color_topic}"
        )
        self._depth_panel = _ImagePanel(
            "Depth", f"Waiting for {self._depth_topic}"
        )
        layout = QHBoxLayout(self._widget)
        layout.addWidget(self._color_panel, 1)
        layout.addWidget(self._depth_panel, 1)
        context.add_widget(self._widget)

        self._subscriptions = [
            self._node.create_subscription(
                Image,
                self._color_topic,
                self._on_color,
                qos_profile_sensor_data,
            ),
            self._node.create_subscription(
                Image,
                self._depth_topic,
                self._on_depth,
                qos_profile_sensor_data,
            ),
        ]
        self._node.get_logger().info(
            f"RGB/depth viewer subscribed to {self._color_topic} and "
            f"{self._depth_topic}"
        )

        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)

    def _on_color(self, message: Image) -> None:
        try:
            frame = np.asarray(
                self._bridge.imgmsg_to_cv2(
                    message, desired_encoding="rgb8"
                )
            ).copy()
        except Exception as exc:
            self._node.get_logger().error(
                f"Could not convert RGB frame: {exc}"
            )
            return
        with self._lock:
            self._latest_color = frame

    def _on_depth(self, message: Image) -> None:
        try:
            frame = np.asarray(
                self._bridge.imgmsg_to_cv2(
                    message, desired_encoding="passthrough"
                )
            )
            depth_metres = (
                frame.astype(np.float32) * 0.001
                if message.encoding.lower() in {"16uc1", "mono16"}
                else frame.astype(np.float32)
            )
            depth_metres[~np.isfinite(depth_metres)] = 0.0
        except Exception as exc:
            self._node.get_logger().error(
                f"Could not convert depth frame: {exc}"
            )
            return
        with self._lock:
            self._latest_depth = depth_metres

    def _refresh(self) -> None:
        with self._lock:
            color = self._latest_color
            depth = self._latest_depth
            self._latest_color = None
            self._latest_depth = None

        if color is not None:
            self._color_panel.set_frame(
                _qimage(color),
                f"{color.shape[1]} x {color.shape[0]}  |  live",
            )
        if depth is not None:
            status = (
                f"{depth.shape[1]} x {depth.shape[0]}  |  "
                f"0-{self._depth_max:g} m"
            )
            self._depth_panel.set_frame(
                _qimage(_colorize_depth(depth, self._depth_max)),
                status,
            )

    def shutdown_plugin(self) -> None:
        """Release subscriptions and stop GUI refreshes."""
        self._timer.stop()
        for subscription in self._subscriptions:
            self._node.destroy_subscription(subscription)
        self._subscriptions.clear()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run this plugin in its own rqt window."""
    from rqt_gui.main import Main

    plugin = "camera_calibration.rgb_depth_rqt.RgbDepthView"
    cli = list(sys.argv if argv is None else argv)
    raise SystemExit(
        Main(filename=plugin).main(
            cli,
            standalone=plugin,
            plugin_argument_provider=_add_arguments,
        )
    )


if __name__ == "__main__":
    main()
