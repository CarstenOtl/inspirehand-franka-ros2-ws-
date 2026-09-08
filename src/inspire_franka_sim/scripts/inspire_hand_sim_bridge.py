#!/usr/bin/env python3
"""Drive the simulated Inspire hand from the real driver's own output.

    ros2 run inspire_franka_sim inspire_hand_sim_bridge.py

On hardware the hand is not a ros2_control device at all: it is an RS485 driver
that takes open ratios on ``/inspire_hand/command`` and reports radians on
``/inspire_hand/joint_states``. In MuJoCo it *is* a ros2_control device, which
is convenient but skips everything the driver does -- the unit conversion, the
partial-command merge, the register quantisation, the range rejection.

Running the driver in ``mock:=true`` alongside the simulator and bridging here
puts all of that back. The chain becomes exactly the hardware one up to the last
step:

    replay runner  --open ratios-->  driver (mock)  --radians-->  this  -->  MuJoCo

So a unit error, a mis-ordered channel or an out-of-range target fails in
simulation the same way it would on the bench, and the hand is still a physical
object in the scene with contacts and its mimic linkage.

What it does not reproduce: the RS485 bus. The mock transport slews toward its
target at a fixed rate rather than modelling the hand's ~0.17 s closed-loop
response, and there is no serial latency, no bus contention with state polling,
and no dropped frames. Timing conclusions belong on hardware.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# The six driven joints, in the order inspire_hand_driver.kinematics.DRIVEN_JOINTS
# uses -- which is the hand's own register order, and the order the simulator's
# forward command controller is configured with in controllers_replay.yaml. The
# names are restated rather than imported so this script does not drag the
# driver package into the simulator's dependencies; the assertion below is what
# keeps them honest.
DRIVEN_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)


class HandSimBridge(Node):
    def __init__(self) -> None:
        super().__init__("inspire_hand_sim_bridge")
        self.declare_parameter("driver_joint_states", "/inspire_hand/joint_states")
        self.declare_parameter(
            "command_topic", "/hand_position_forward_command_controller/commands"
        )
        # Prefixed onto every joint name the driver publishes; must match the
        # driver's own joint_prefix and the description's.
        self.declare_parameter("joint_prefix", "")

        self._prefix = str(self.get_parameter("joint_prefix").value)
        self._expected = [self._prefix + name for name in DRIVEN_JOINTS]
        self._warned = False

        self._publisher = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("command_topic").value), 10
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("driver_joint_states").value),
            self._on_state,
            10,
        )
        self.get_logger().info(
            f"bridging {self.get_parameter('driver_joint_states').value} -> "
            f"{self.get_parameter('command_topic').value}"
        )

    def _on_state(self, message: JointState) -> None:
        """Forward the driven six, in controller order, dropping the followers.

        The driver publishes all twelve joints because robot_state_publisher
        needs them for TF, but the simulator holds the six followers with its
        own equality constraints -- the same way the linkage holds them on the
        real hand. Commanding a follower here would fight that constraint.
        """
        positions = dict(zip(message.name, message.position))
        missing = [name for name in self._expected if name not in positions]
        if missing:
            if not self._warned:
                self.get_logger().warn(
                    f"driver joint_states is missing {missing}; is joint_prefix "
                    f"{self._prefix!r} right? (warned once)"
                )
                self._warned = True
            return
        self._warned = False
        command = Float64MultiArray()
        command.data = [float(positions[name]) for name in self._expected]
        self._publisher.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandSimBridge()
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
