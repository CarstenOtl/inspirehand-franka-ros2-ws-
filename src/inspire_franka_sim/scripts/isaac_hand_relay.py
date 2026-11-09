#!/usr/bin/env python3
"""The Inspire hand's ROS interface, served the way Isaac Lab drives the hand.

    ros2 run inspire_franka_sim isaac_hand_relay.py

Stands in for the RS485 driver in ``sim_policy.launch.py hand_chain:=isaac``.
The policy runner talks to the same two topics it uses on hardware, but the
hand behind them is the training simulator's:

- ``/inspire_hand/command`` (open ratios, the driver's contract) is converted to
  joint targets with the driver's own ``open_ratio_to_rad`` and sent to the
  position actuators at once. Isaac writes the policy's pinch target straight
  into the joint drive; there is no slew, no register quantisation and no wall
  clock in between, which is what the driver chain (``hand_chain:=driver``:
  mock driver plus ``inspire_hand_sim_bridge.py``) adds.
- ``/inspire_hand/joint_states`` republishes the SIMULATED hand joints from
  ``/joint_states``. Isaac's student observes its finger joints, which stop on
  the nut; the mock driver reports where its slew has got to, contact or not.

The followers are not commanded here: IsaacHandPlugin points them at their
measured leaders inside the simulator.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from inspire_hand_driver import kinematics as kin

# The forward command controller's joint order (controllers_sim_policy.yaml),
# which is the driver's register order.
DRIVEN_JOINTS = tuple(kin.DRIVEN_JOINTS)


class IsaacHandRelay(Node):
    def __init__(self) -> None:
        super().__init__("inspire_hand")
        self.declare_parameter(
            "command_topic", "/hand_position_forward_command_controller/commands"
        )
        # /joint_states arrives at the controller manager's 1 kHz. Republishing
        # all of it buried the runner's Python loop in callbacks (the policy fell
        # to ~5 Hz and tripped the controller watchdog), so publish at the
        # driver's own rate (inspire_hand_driver publish_rate_hz), in sim time.
        self.declare_parameter("publish_rate_hz", 50.0)
        self._period_ns = int(1.0e9 / float(self.get_parameter("publish_rate_hz").value))
        self._last_ns = None
        self._targets = None
        self._publisher = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 10
        )
        self._state_publisher = self.create_publisher(JointState, "~/joint_states", 10)
        self.create_subscription(JointState, "~/command", self._on_command, 10)
        self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self.get_logger().info(
            "Isaac hand chain: ~/command open ratios -> joint targets, "
            "/joint_states -> ~/joint_states"
        )

    def _on_state(self, message: JointState) -> None:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if (
            self._targets is not None
            and self._last_ns is not None
            and 0 <= stamp_ns - self._last_ns < self._period_ns
        ):
            return
        self._last_ns = stamp_ns
        names = list(message.name)
        hand = [n for n in kin.ALL_JOINTS if n in names]
        if not hand:
            return
        index = {n: i for i, n in enumerate(names)}
        if self._targets is None and all(n in index for n in DRIVEN_JOINTS):
            # Until the first command, hold where the hand is.
            self._targets = [float(message.position[index[n]]) for n in DRIVEN_JOINTS]
        out = JointState()
        out.header = message.header
        out.name = hand
        out.position = [float(message.position[index[n]]) for n in hand]
        if len(message.velocity) == len(names):
            out.velocity = [float(message.velocity[index[n]]) for n in hand]
        self._state_publisher.publish(out)

    def _publish_targets(self, targets) -> None:
        command = Float64MultiArray()
        command.data = [float(v) for v in targets]
        self._publisher.publish(command)

    def _on_command(self, message: JointState) -> None:
        if self._targets is None:
            self.get_logger().warn(
                "command before the first /joint_states; ignored", throttle_duration_sec=2.0
            )
            return
        for name, ratio in zip(message.name, message.position):
            try:
                dof = kin.dof_index(name)
            except KeyError:
                self.get_logger().warn(f"not a driven joint: {name!r}", throttle_duration_sec=2.0)
                continue
            self._targets[DRIVEN_JOINTS.index(kin.DOFS[dof].joint)] = kin.open_ratio_to_rad(
                dof, ratio
            )
        self._publish_targets(self._targets)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacHandRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
