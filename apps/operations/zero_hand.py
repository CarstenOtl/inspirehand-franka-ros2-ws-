#!/usr/bin/env python3
"""Move all Inspire hand joints to zero radians (fully open) and verify it."""

from __future__ import annotations

import argparse
import sys
import time

from operation_common import HAND_CHANNELS, HAND_ZERO_RATIOS, maximum_error, positions_by_name

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ModuleNotFoundError as exc:  # lets --dry-run work outside the ROS container
    ROS_IMPORT_ERROR = exc
    rclpy = None
    Node = object


class ZeroHandNode(Node):
    def __init__(self, command_topic: str, state_topic: str) -> None:
        super().__init__("zero_inspire_hand")
        self.publisher = self.create_publisher(JointState, command_topic, 10)
        self.state = None
        self.create_subscription(JointState, state_topic, self._on_state, 10)

    def _on_state(self, message: JointState) -> None:
        self.state = positions_by_name(message.name, message.position)

    def publish_zero(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(HAND_CHANNELS)
        # The hand wire protocol is an open ratio. Ratio 1.0 corresponds to
        # zero radians in the URDF; ratio 0.0 would fully close the hand.
        message.position = list(HAND_ZERO_RATIOS)
        self.publisher.publish(message)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--command-topic", default="/inspire_hand/command")
    result.add_argument("--state-topic", default="/inspire_hand/state")
    result.add_argument("--timeout", type=float, default=15.0)
    result.add_argument("--tolerance", type=float, default=0.03)
    result.add_argument(
        "--dry-run", action="store_true", help="print the command without connecting to ROS"
    )
    return result


def main(argv=None) -> int:
    raw = sys.argv if argv is None else [sys.argv[0], *argv]
    application_args = raw[1:]
    ros_args = [raw[0]]
    if "--ros-args" in application_args:
        ros_index = application_args.index("--ros-args")
        ros_args += application_args[ros_index:]
        application_args = application_args[:ros_index]
    args = parser().parse_args(application_args)
    if args.timeout <= 0:
        parser().error("--timeout must be positive")
    if args.tolerance < 0:
        parser().error("--tolerance must not be negative")

    print("Inspire target: all driven joints = 0 rad (open ratio 1.0, fully open)")
    if args.dry_run:
        print("dry run: no ROS command sent")
        return 0
    if rclpy is None:
        print(f"ERROR: ROS 2 Python modules are unavailable: {ROS_IMPORT_ERROR}", file=sys.stderr)
        return 1

    rclpy.init(args=ros_args)
    node = ZeroHandNode(args.command_topic, args.state_topic)
    deadline = time.monotonic() + args.timeout
    last_publish = 0.0
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            # Repeating the target makes a one-shot utility robust to discovery
            # delay and an occasional dropped RS485 transaction.
            if now - last_publish >= 0.2:
                node.publish_zero()
                last_publish = now
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.state is None:
                continue
            error = maximum_error(node.state, HAND_CHANNELS, HAND_ZERO_RATIOS)
            if error is not None and error <= args.tolerance:
                print(f"Hand zero complete (maximum open-ratio error {error:.3f})")
                return 0
        if node.publisher.get_subscription_count() == 0:
            print(
                f"ERROR: no hand driver subscribes to {args.command_topic}; "
                "is inspire_franka_bringup running?",
                file=sys.stderr,
            )
        elif node.state is None:
            print(f"ERROR: no hand feedback received on {args.state_topic}", file=sys.stderr)
        else:
            error = maximum_error(node.state, HAND_CHANNELS, HAND_ZERO_RATIOS)
            print(
                f"ERROR: hand did not reach zero within {args.timeout:g} s "
                f"(maximum open-ratio error {error:.3f})",
                file=sys.stderr,
            )
        return 1
    except KeyboardInterrupt:
        print("Interrupted; the hand holds the last zero target", file=sys.stderr)
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
