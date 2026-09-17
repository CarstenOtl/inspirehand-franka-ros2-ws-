#!/usr/bin/env python3
"""Republish a mujoco_ros2_control camera as the workcell's RealSense D415.

    ros2 run inspire_franka_sim policy_camera_relay.py --ros-args \\
        -p source:=/policy_d415 -p camera_matrix:=[fx,0,cx,0,fy,cy,0,0,1]

mujoco_ros2_control renders every MJCF camera to ``<name>/color`` (rgb8),
``<name>/depth`` (32FC1 metres) and ``<name>/camera_info``. Its CameraInfo is
derived from fovy alone (square pixels, principal point at the image centre)
and its frame is ``<name>_frame``, so a consumer checking the calibrated
camera contract rejects it even when the MJCF camera renders the calibrated
pinhole exactly. This node pairs colour and depth by stamp and publishes what
``realsense2_camera`` publishes with ``align_depth.enable`` and
``enable_rgbd``: colour, 16UC1 millimetre depth aligned to colour, the colour
CameraInfo, and the composite ``realsense2_camera_msgs/RGBD``, all in
``frame_id`` with the configured camera matrix.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from realsense2_camera_msgs.msg import RGBD
from sensor_msgs.msg import CameraInfo, Image


class PolicyCameraRelay(Node):
    def __init__(self):
        super().__init__("policy_camera_relay")
        source = self.declare_parameter("source", "/policy_d415").value.rstrip("/")
        self.frame_id = self.declare_parameter("frame_id", "camera_color_optical_frame").value
        k = [float(v) for v in self.declare_parameter("camera_matrix", [0.0] * 9).value]
        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0:
            raise ValueError("camera_matrix must be a 3x3 row-major matrix with positive focal lengths")
        self.k = k
        # MuJoCo reports the far clip distance where nothing was hit; the D415
        # reports 0 (invalid) there, which the policy's valid mask excludes.
        self.max_depth_m = float(self.declare_parameter("max_depth_m", 10.0).value)
        prefix = self.declare_parameter("output_prefix", "/camera/camera").value.rstrip("/")

        # Reliable like realsense2_camera's default, so both the policy's
        # best-effort subscription and image viewers (reliable) can connect.
        output_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.color_pub = self.create_publisher(Image, prefix + "/color/image_raw", output_qos)
        self.info_pub = self.create_publisher(CameraInfo, prefix + "/color/camera_info", output_qos)
        self.depth_pub = self.create_publisher(
            Image, prefix + "/aligned_depth_to_color/image_raw", output_qos
        )
        self.rgbd_pub = self.create_publisher(RGBD, prefix + "/rgbd", output_qos)
        # The plugin publishes reliably. A best-effort subscription lost about a
        # quarter of the 1.2 MB float depth frames (gaps up to 0.7 s) under
        # rollout CPU load, which starved the pairing below.
        source_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, source + "/color", self._on_color, source_qos)
        self.create_subscription(Image, source + "/depth", self._on_depth, source_qos)
        self.create_subscription(CameraInfo, source + "/camera_info", self._on_info, source_qos)
        self._color = {}
        self._depth = {}
        self._checked_info = False
        self.get_logger().info(
            f"relaying {source} -> {prefix}/rgbd in {self.frame_id} "
            f"(fx={k[0]:.3f} fy={k[4]:.3f} cx={k[2]:.3f} cy={k[5]:.3f})"
        )

    @staticmethod
    def _key(message):
        return (message.header.stamp.sec, message.header.stamp.nanosec)

    def _on_info(self, message):
        if self._checked_info:
            return
        self._checked_info = True
        # The plugin's own matrix is fovy-derived; only its size is authoritative.
        self.width, self.height = message.width, message.height
        if abs(message.k[4] - self.k[4]) > 0.5:
            self.get_logger().warn(
                f"MJCF camera fy {message.k[4]:.3f} differs from the calibrated "
                f"{self.k[4]:.3f}; regenerate the scene from the camera profile"
            )

    def _on_color(self, message):
        self._color[self._key(message)] = message
        self._publish()

    def _on_depth(self, message):
        self._depth[self._key(message)] = message
        self._publish()

    def _camera_info(self, header, width, height):
        info = CameraInfo()
        info.header = header
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = self.k
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.k[0], 0.0, self.k[2], 0.0, 0.0, self.k[4], self.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _publish(self):
        common = sorted(set(self._color) & set(self._depth))
        if not common:
            for queue in (self._color, self._depth):
                for key in sorted(queue)[:-4]:
                    del queue[key]
            return
        key = common[-1]
        color, depth = self._color[key], self._depth[key]
        for queue in (self._color, self._depth):
            for stale in [k for k in queue if k <= key]:
                del queue[stale]

        header = color.header
        header.frame_id = self.frame_id
        if color.encoding != "rgb8" or depth.encoding != "32FC1":
            self.get_logger().error(
                f"unexpected encodings color={color.encoding} depth={depth.encoding}",
                throttle_duration_sec=5.0,
            )
            return
        metres = np.frombuffer(depth.data, dtype=np.float32).reshape(depth.height, -1)[:, : depth.width]
        valid = np.isfinite(metres) & (metres > 0.0) & (metres < self.max_depth_m)
        millimetres = np.where(valid, np.rint(metres * 1000.0), 0.0).clip(0, 65535).astype(np.uint16)

        depth_out = Image()
        depth_out.header = header
        depth_out.height, depth_out.width = depth.height, depth.width
        depth_out.encoding = "16UC1"
        depth_out.is_bigendian = 0
        depth_out.step = 2 * depth.width
        depth_out.data = millimetres.tobytes()
        color.header = header
        info = self._camera_info(header, color.width, color.height)

        rgbd = RGBD()
        rgbd.header = header
        rgbd.rgb_camera_info = info
        rgbd.depth_camera_info = info
        rgbd.rgb = color
        rgbd.depth = depth_out
        self.rgbd_pub.publish(rgbd)
        # Serialising 1.5 MB twice more costs most of a core at 30 Hz; only do
        # it for a viewer that is actually listening.
        if self.color_pub.get_subscription_count():
            self.color_pub.publish(color)
        if self.depth_pub.get_subscription_count():
            self.depth_pub.publish(depth_out)
        if self.info_pub.get_subscription_count():
            self.info_pub.publish(info)


def main():
    rclpy.init()
    node = PolicyCameraRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
