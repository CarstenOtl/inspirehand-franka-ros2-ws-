"""ROS 2 driver node for the Inspire Robotics RH56 dexterous hand over RS485.

Interface
---------
Everything is relative to the node's name, so two hands coexist by running two
nodes with different names (``left_hand``, ``right_hand``) and different serial
ports -- there is no global naming baked in.

Published
    ``~/joint_states``   ``sensor_msgs/JointState``
        All twelve URDF joints, **in radians**, ready for
        ``robot_state_publisher``. The six followers are computed from the
        driven six through :mod:`inspire_hand_driver.kinematics`; without them
        the fingertips get no TF frames at all. ``effort`` carries raw actuator
        current for the driven six and 0 for the followers.
    ``~/state``          ``sensor_msgs/JointState``
        The same reading in the hand's own units: channels ``"1".."6"``,
        position as an **open ratio** (1.0 open .. 0.0 closed). This is the
        convenient form for scripting a grasp, and the form the registers use.
    ``~/grip_force``     ``sensor_msgs/JointState``
        Measured grip force per channel, in the hand's register units.

Subscribed
    ``~/command``        ``sensor_msgs/JointState``
        Names may be channel ids (``"1".."6"``, positions read as open ratios)
        or driven joint names (positions read as radians). The two sets are
        disjoint, so the message is self-describing; mixing them in one message
        is rejected. Any subset may be addressed -- unnamed DOF hold.

Services
    ``~/set_angles``     ``inspire_hand_msgs/srv/SetAngles``
    ``~/set_speed``      ``inspire_hand_msgs/srv/SetSpeed``
    ``~/set_force``      ``inspire_hand_msgs/srv/SetForce``

Why the driven/follower split is visible in the interface: only six things can
be commanded, but twelve have to be published or TF breaks. Commands therefore
accept only driven names, while state carries all twelve.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from inspire_hand_msgs.srv import SetAngles, SetForce, SetSpeed

from . import kinematics as kin
from .protocol import (
    ANGLE_INVALID,
    ANGLE_MAX,
    CHANNEL_IDS,
    REG_CURRENT,
    HandCommunicationError,
    HandTransport,
    MockTransport,
)


def angle_to_open_ratio(angle: int) -> float:
    """Convert a raw register angle (0 closed .. 1000 open) to an open ratio."""
    return max(0.0, min(1.0, float(angle) / float(ANGLE_MAX)))


def open_ratio_to_angle(ratio: float) -> int:
    """Convert an open ratio (1.0 open .. 0.0 closed) to a raw register angle."""
    return int(round(max(0.0, min(1.0, float(ratio))) * ANGLE_MAX))


class InspireHandNode(Node):
    def __init__(self) -> None:
        super().__init__("inspire_hand_driver")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("hand_id", 1)
        self.declare_parameter("protocol", "modbus")
        self.declare_parameter("mock", False)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("serial_timeout_sec", 0.25)
        # Prefixed onto every joint name published and accepted, so a two-handed
        # setup can put both hands in one URDF. Must match the description's
        # `prefix` argument exactly.
        self.declare_parameter("joint_prefix", "")
        # Applied once at startup; the hand keeps these in volatile registers.
        self.declare_parameter("startup_speed", 0)
        self.declare_parameter("startup_force", 0)
        # Consecutive read failures tolerated before the node reports the hand
        # as lost. RS485 drops the odd frame under EMI; one miss is not a fault.
        self.declare_parameter("max_read_failures", 5)

        self._mock = bool(self.get_parameter("mock").value)
        self._prefix = str(self.get_parameter("joint_prefix").value)
        self._max_failures = int(self.get_parameter("max_read_failures").value)
        self._failures = 0
        self._last_command: Optional[List[int]] = None

        self._joint_names = [self._prefix + j for j in kin.ALL_JOINTS]
        self._driven_names = [self._prefix + j for j in kin.DRIVEN_JOINTS]

        self._transport = self._build_transport()
        self._connect()

        self._joint_state_pub = self.create_publisher(JointState, "~/joint_states", 10)
        self._state_pub = self.create_publisher(JointState, "~/state", 10)
        self._force_pub = self.create_publisher(JointState, "~/grip_force", 10)
        self._command_sub = self.create_subscription(
            JointState, "~/command", self._on_command, 10
        )
        self._angles_srv = self.create_service(
            SetAngles, "~/set_angles", self._on_set_angles
        )
        self._speed_srv = self.create_service(SetSpeed, "~/set_speed", self._on_set_speed)
        self._force_srv = self.create_service(SetForce, "~/set_force", self._on_set_force)

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().info(
            f"inspire_hand_driver up: "
            f"transport={'mock' if self._mock else self._transport.port} "
            f"protocol={self._transport.protocol} id={self._transport.hand_id} "
            f"rate={rate:.0f}Hz prefix={self._prefix!r}"
        )

    # -- setup -------------------------------------------------------------
    def _build_transport(self) -> HandTransport:
        common = dict(
            baudrate=int(self.get_parameter("baudrate").value),
            hand_id=int(self.get_parameter("hand_id").value),
            protocol=str(self.get_parameter("protocol").value),
            timeout=float(self.get_parameter("serial_timeout_sec").value),
        )
        if self._mock:
            return MockTransport(**common)
        return HandTransport(port=str(self.get_parameter("port").value), **common)

    def _connect(self) -> None:
        self._transport.connect()
        if self._mock:
            return
        hand_id = self._transport.ping()
        if hand_id is None:
            self.get_logger().error(
                f"hand did not answer on {self._transport.port} @ "
                f"{self._transport.baudrate} baud, id={self._transport.hand_id}, "
                f"protocol={self._transport.protocol}. Check that the hand has its "
                f"24V supply, that RS485 A/B are not swapped, and try the other "
                f"protocol. Run 'ros2 run inspire_hand_driver inspire_hand_probe' to scan."
            )
            return
        self.get_logger().info(f"hand answered, reports ID {hand_id}")
        speed = int(self.get_parameter("startup_speed").value)
        force = int(self.get_parameter("startup_force").value)
        try:
            if speed > 0:
                self._transport.write_speeds([speed] * 6)
            if force > 0:
                self._transport.write_forces([force] * 6)
        except HandCommunicationError as exc:
            self.get_logger().warn(f"failed to apply startup speed/force: {exc}")

    # -- state publishing --------------------------------------------------
    def _on_timer(self) -> None:
        try:
            angles = self._transport.read_angles()
            currents = self._transport.read_registers(REG_CURRENT, 6)
            forces = self._transport.read_forces()
        except HandCommunicationError as exc:
            self._failures += 1
            if self._failures == self._max_failures:
                self.get_logger().error(f"hand unresponsive after {self._failures} reads: {exc}")
            elif self._failures < self._max_failures:
                self.get_logger().debug(f"read failed ({self._failures}): {exc}")
            return

        if self._failures >= self._max_failures:
            self.get_logger().info("hand responsive again")
        self._failures = 0

        stamp = self.get_clock().now().to_msg()

        # A DOF reporting the invalid sentinel is published as fully closed
        # rather than as a spurious jump: 0xFFFF would otherwise ratio to well
        # past 1.0 and clamp to "open", which is the dangerous direction to
        # guess wrong in.
        ratios = [
            angle_to_open_ratio(a) if a != ANGLE_INVALID else 0.0 for a in angles
        ]

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.name = list(self._joint_names)
        joint_state.position = kin.joint_positions(ratios)
        # Current is only measured on the driven DOF; the followers have no motor.
        joint_state.effort = [float(c) for c in currents] + [0.0] * len(kin.PASSIVE_JOINTS)
        self._joint_state_pub.publish(joint_state)

        state = JointState()
        state.header.stamp = stamp
        state.header.frame_id = "open_ratio"
        state.name = list(CHANNEL_IDS)
        state.position = ratios
        state.effort = [float(c) for c in currents]
        self._state_pub.publish(state)

        force_msg = JointState()
        force_msg.header.stamp = stamp
        force_msg.header.frame_id = "grip_force"
        force_msg.name = list(CHANNEL_IDS)
        force_msg.effort = [float(f) for f in forces]
        self._force_pub.publish(force_msg)

    # -- command paths -----------------------------------------------------
    def _resolve(self, name: str) -> int:
        """Resolve a channel id or (prefixed) driven joint name to a DOF index."""
        key = str(name)
        if self._prefix and key.startswith(self._prefix):
            key = key[len(self._prefix) :]
        return kin.dof_index(key)

    def _merge(
        self, names: Sequence[str], values: Sequence[float], radians: bool
    ) -> Tuple[Optional[List[int]], str]:
        """Merge a partial set of named targets onto the last command.

        Unaddressed DOF keep their current target rather than snapping to a
        default -- which is what makes a partial command safe to send.
        """
        if len(names) != len(values):
            return None, f"name has {len(names)} entries but value has {len(values)}"

        base = self._last_command
        if base is None:
            # First command of the session: start from where the hand actually
            # is, so unaddressed DOF hold their physical pose rather than
            # jumping to an assumed one.
            try:
                base = [
                    a if a != ANGLE_INVALID else ANGLE_MAX
                    for a in self._transport.read_angles()
                ]
            except HandCommunicationError:
                base = [ANGLE_MAX] * 6

        angles = list(base)
        unknown: List[str] = []
        touched = False
        for name, value in zip(names, values):
            try:
                index = self._resolve(name)
            except KeyError:
                unknown.append(str(name))
                continue
            ratio = (
                kin.rad_to_open_ratio(index, value) if radians else float(value)
            )
            angles[index] = open_ratio_to_angle(ratio)
            touched = True

        if unknown:
            self.get_logger().warn(f"ignoring unknown hand channels: {', '.join(unknown)}")
        if not touched:
            return None, f"no recognised channel named (got: {', '.join(map(str, names))})"
        return angles, ""

    def _classify(self, names: Sequence[str]) -> Tuple[Optional[bool], str]:
        """Decide whether `names` are channel ids or joint names.

        Returns ``(radians, error)``: True when the caller is speaking joint
        names and radians, False for channel ids and open ratios.
        """
        stripped = [
            n[len(self._prefix) :] if self._prefix and n.startswith(self._prefix) else n
            for n in map(str, names)
        ]
        channels = any(n in CHANNEL_IDS for n in stripped)
        joints = any(n in kin.DRIVEN_JOINTS for n in stripped)
        if channels and joints:
            return None, "message mixes channel ids and joint names; use one or the other"
        # Unrecognised names fall through as channel ids and get reported by _merge.
        return joints, ""

    def _send(self, angles: List[int]) -> Tuple[bool, str]:
        try:
            self._transport.write_angles(angles)
        except HandCommunicationError as exc:
            self.get_logger().warn(f"failed to write angles: {exc}")
            return False, str(exc)
        self._last_command = angles
        return True, ""

    def _apply(self, names: Sequence[str], values: Sequence[float]) -> Tuple[bool, str]:
        if not names:
            return False, "no channels named"
        radians, error = self._classify(names)
        if error:
            return False, error
        angles, error = self._merge(names, values, radians=bool(radians))
        if angles is None:
            return False, error
        return self._send(angles)

    def _on_command(self, msg: JointState) -> None:
        if not msg.position:
            return
        names = list(msg.name) if msg.name else list(CHANNEL_IDS)
        accepted, message = self._apply(names, list(msg.position))
        if not accepted and message:
            self.get_logger().warn(f"command rejected: {message}")

    def _on_set_angles(self, request, response):
        response.accepted, response.message = self._apply(
            list(request.name), list(request.open_ratio)
        )
        return response

    # Speed and force take the same shape as angles but address different
    # registers and have no "hold the rest" semantics to preserve -- the hand
    # keeps whatever it was last told per DOF, so a partial write is a partial
    # write with no merge needed.
    def _on_set_speed(self, request, response):
        response.accepted, response.message = self._write_limits(
            list(request.name), list(request.speed), self._transport.write_speeds, "speed"
        )
        return response

    def _on_set_force(self, request, response):
        response.accepted, response.message = self._write_limits(
            list(request.name), list(request.force), self._transport.write_forces, "force"
        )
        return response

    def _write_limits(self, names, values, writer, what: str) -> Tuple[bool, str]:
        if len(names) != len(values):
            return False, f"name has {len(names)} entries but {what} has {len(values)}"
        if not names:
            return False, "no channels named"
        # Read-modify-write is not possible (these registers are write-only on
        # some firmware), so an unaddressed DOF is rewritten with the value this
        # node last sent, defaulting to the hand's own mid-scale.
        cache = getattr(self, f"_{what}_cache", None) or [500] * 6
        out = list(cache)
        unknown, touched = [], False
        for name, value in zip(names, values):
            try:
                index = self._resolve(name)
            except KeyError:
                unknown.append(str(name))
                continue
            out[index] = max(0, min(1000, int(value)))
            touched = True
        if unknown:
            self.get_logger().warn(f"ignoring unknown hand channels: {', '.join(unknown)}")
        if not touched:
            return False, f"no recognised channel named (got: {', '.join(map(str, names))})"
        try:
            writer(out)
        except HandCommunicationError as exc:
            return False, str(exc)
        setattr(self, f"_{what}_cache", out)
        return True, ""

    def destroy_node(self) -> bool:
        try:
            self._transport.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = InspireHandNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
