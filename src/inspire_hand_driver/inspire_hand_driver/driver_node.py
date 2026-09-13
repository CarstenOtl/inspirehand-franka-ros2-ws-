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
    ``~/diagnostics``    ``diagnostic_msgs/DiagnosticArray``
        One entry per DOF with the firmware's STATUS code, ERROR bits and
        actuator temperature, plus a summary entry. Level ``ERROR`` on any
        DOF the firmware has stopped on a fault. Published at the
        ``state_extras_divisor`` cadence, with the current and force reads.

Subscribed
    ``~/command``        ``sensor_msgs/JointState``
        ``position`` is **always an open ratio**: ``1.0`` fully open, ``0.0``
        fully closed, matching ``~/state`` and the registers. Names only say
        *which* DOF -- channel ids (``"1".."6"``) or driven joint names, mixed
        freely. Any subset may be addressed; unnamed DOF hold. A target outside
        ``[0.0, 1.0]`` rejects the whole message.

Services
    ``~/set_angles``     ``inspire_hand_msgs/srv/SetAngles``
    ``~/set_speed``      ``inspire_hand_msgs/srv/SetSpeed``
    ``~/set_force``      ``inspire_hand_msgs/srv/SetForce``
    ``~/clear_errors``   ``std_srvs/srv/Trigger``
        Write CLEAR_ERROR by hand. The stall guard below does this on its
        own; the service is for when it is off, or for a look at whether a
        finger that stopped can be talked to without a power cycle.

Force threshold and stall guard
-------------------------------
The hand is position controlled, and a finger that cannot reach its target
does not merely wait: it pushes until the actuator's protection trips, the
firmware latches a locked-rotor or over-current error, and the DOF then
ignores every target until CLEAR_ERROR is written or the hand is
power-cycled. On the bench this was "the finger died and stayed dead until
reboot". Two things in this node address it.

*The force threshold* (``startup_force``, FORCE_SET in the registers) is the
firmware's own answer: a DOF whose fingertip force reaches it stops there,
cleanly, with status "stopped at force threshold", and no error. It is on by
default now, at 500 g of the 1000 g scale, and re-applied whenever the hand
appears to have forgotten it (a reconnect after the hand went silent, or a
periodic readback of FORCE_SET that disagrees with what was written). It is
still volatile -- never committed to flash -- and ``~/set_force`` still
overrides it per DOF, so a capture preset asking for a gentler pinch gets one.

*The stall guard* (``stall_guard``) covers what the threshold cannot: contact
away from the fingertip sensor, a thumb rotation jammed against the palm, a
finger stalled before the threshold had been applied. Every extras cycle the
node reads the STATUS and ERROR blocks; a DOF the firmware has stopped on a
fault is backed off -- its target is moved to where it actually is plus
``stall_backoff`` register counts towards open -- and then CLEAR_ERROR is
written, at most once per ``clear_error_interval_sec``. Backing off comes
first so that clearing does not simply drive the finger into the same
obstacle again. For ``stall_holdoff_sec`` afterwards, commands that would
take that DOF back past the backed-off angle are clamped to it; a 50 Hz
stream re-sending the same unreachable target therefore stalls the finger at
most once per hold-off rather than continuously. The clamp is logged, and the
finger's target in ``~/state`` shows where it was held.

Neither mechanism commands anything on its own beyond that backoff, and
neither ever writes SAVE: the register that clears errors shares a word with
the one that commits to flash, and :meth:`HandTransport.clear_errors` is the
only writer.

Thumb-abduction calibration
----------------------------
Every angle command path receives the final overlays from
:mod:`inspire_hand_driver.command_overlays`. At open ratio ``0.0`` the thumb
swings past the palm plane, so the bottom of its commanded range is unusable.
The current overlay therefore treats ``0.25`` as the thumb's zero and rescales
the whole command onto the usable travel::

    physical = 0.25 + 0.75 * commanded

A command of ``0.0`` reaches the hand as ``0.25`` and ``1.0`` still reaches it
as ``1.0``, so the mapping stays monotonic and every commanded value remains
distinct -- unlike a floor, which would collapse the bottom quarter of the
range onto one pose. The other five DOF are unchanged. Inputs are still
validated against the public ``[0, 1]`` command range before the overlay is
applied.

``~/joint_states`` reports the hand's *physical* pose, not the pre-overlay
command, because it feeds ``robot_state_publisher`` and TF. Commanding thumb
abduction ``0.0`` therefore reads back ``0.25``; callers that need to compare
feedback against a command should map the command forward with
:func:`~inspire_hand_driver.command_overlays.apply_open_ratio_overlay`.

Commanding in ratios, not radians
---------------------------------
``~/joint_states`` publishes radians because that is what ``robot_state_publisher``
and the URDF need, and there ``0.0`` is the *open* pose. Commands deliberately
do not follow it. The two conventions therefore run opposite ways, which is
worth stating once rather than discovering: **a rising joint_states value means
a closing hand, and a rising commanded ratio means an opening one.**

The reason is that the previous scheme -- infer the unit from the naming, so
joint names meant radians and channel ids meant ratios -- put both conventions
on one ``position`` field. ``1.5`` addressed as a channel id clamped to a fully
open hand; the same ``1.5`` addressed by joint name clamped to a fully closed
one. One number, opposite ends of travel, no warning either way. Ratios are now
the single commanding unit: one range, ``[0, 1]``, identical for all six DOF,
and out-of-range is an error instead of a full-travel move.

Callers holding radians convert at this boundary with
:func:`inspire_hand_driver.kinematics.rad_to_open_ratio`, which is what
``inspire_franka_trajectory_replay`` does -- its trajectories and homing YAMLs
stay in radians, because those files also carry the FR3's seven joints.

Why the driven/follower split is visible in the interface: only six things can
be commanded, but twelve have to be published or TF breaks. Commands therefore
accept only driven names, while state carries all twelve.

Sharing the bus with a command stream
-------------------------------------
RS485 is half-duplex, so every register read is time the line cannot be
carrying a target. Publishing costs four reads -- angles, current, force, and
the status/error block the stall guard watches -- and at the default 50 Hz
that is well over a third of the bus at best. Only the angles are needed to
publish joint states, so ``state_extras_divisor`` fetches the other three once
per N publishes and holds them in between; ``1`` reads everything every cycle
and is the default, while a replay streaming targets wants 5 or more.
``inspire_hand_driver.benchmark`` models and measures what the bus actually
costs.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from inspire_hand_msgs.srv import SetAngles, SetForce, SetSpeed

from . import command_overlays
from . import kinematics as kin
from .protocol import (
    ANGLE_INVALID,
    ANGLE_MAX,
    CHANNEL_IDS,
    DOF_ORDER,
    REG_CURRENT,
    STATUS_AT_FORCE,
    HandCommunicationError,
    HandHealth,
    HandProtocolError,
    HandTransport,
    MockTransport,
    describe_error,
    describe_status,
)


# The commandable range, in open-ratio units. Every DOF shares it, which is
# the point of commanding in ratios: the equivalent radian limits differ per
# joint (1.47, 0.6, 1.308) and belong to the description, not to this wire
# protocol. Targets outside it are rejected rather than clamped -- silently
# clamping turned a typo into a full-travel move with no warning anywhere.
RATIO_OPEN = 1.0
RATIO_CLOSED = 0.0


def out_of_range(names: Sequence[str], values: Sequence[float]) -> List[str]:
    """Name every target outside the commandable open-ratio range.

    NaN compares false against both bounds and so is reported, which is what
    we want: it must never reach the registers.
    """
    return [
        f"{name}={float(value):g}"
        for name, value in zip(names, values)
        if not RATIO_CLOSED <= float(value) <= RATIO_OPEN
    ]


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
        # Applied at startup and re-applied whenever the hand looks to have
        # lost them; the hand keeps these in volatile registers. 0 = leave the
        # hand's own power-on value alone.
        self.declare_parameter("startup_speed", 0)
        # The grip-force threshold, in the hand's 0..1000 (gram) units. A DOF
        # stops closing when its fingertip force reaches this, instead of
        # pushing until the actuator's protection latches an error. 500 is
        # half scale: firm enough for every grasp this rig has needed, and
        # the capture presets lower it per posture through ~/set_force.
        self.declare_parameter("startup_force", 500)
        # How often FORCE_SET is read back and compared with what was written.
        # A hand that rebooted in the gap between two reads comes back with
        # its power-on defaults, and this is what notices. 0 disables.
        self.declare_parameter("limits_check_interval_sec", 2.0)
        # Stall guard: see the module docstring.
        self.declare_parameter("stall_guard", True)
        # Register counts (0..1000 scale) to back a stalled DOF off towards
        # open from where it stopped, so that clearing the error does not
        # drive it straight back into the same obstacle.
        self.declare_parameter("stall_backoff", 30)
        # For this long after a stall, commands for that DOF are clamped to
        # the backed-off angle.
        self.declare_parameter("stall_holdoff_sec", 1.0)
        # Minimum gap between CLEAR_ERROR writes.
        self.declare_parameter("clear_error_interval_sec", 1.0)
        # Consecutive read failures tolerated before the node reports the hand
        # as lost. RS485 drops the odd frame under EMI; one miss is not a fault.
        self.declare_parameter("max_read_failures", 5)
        # Publishing costs three read transactions -- angles, current, force --
        # and RS485 is half-duplex, so those three are time the bus cannot be
        # carrying commands. Only the angles are needed to publish joint states;
        # this divides how often the other two are fetched, holding their last
        # value in between. 1 reads everything every cycle, which is what the
        # driver has always done and stays the default; a replay streaming
        # targets at 50 Hz wants 5 or more. See inspire_hand_driver.benchmark
        # for what the bus actually costs.
        self.declare_parameter("state_extras_divisor", 1)

        self._mock = bool(self.get_parameter("mock").value)
        self._prefix = str(self.get_parameter("joint_prefix").value)
        self._max_failures = int(self.get_parameter("max_read_failures").value)
        self._failures = 0
        self._last_command: Optional[List[int]] = None
        self._extras_divisor = max(1, int(self.get_parameter("state_extras_divisor").value))
        self._ticks = 0
        # Held between fetches when the divisor is above 1, and on the very
        # first tick before either has been read once.
        self._currents: List[int] = [0] * 6
        self._forces: List[int] = [0] * 6
        self._health: Optional[HandHealth] = None

        # The speed and force limits this node believes the hand is holding.
        # None means "never told it anything, leave its power-on value".
        speed = int(self.get_parameter("startup_speed").value)
        force = int(self.get_parameter("startup_force").value)
        self._speed_cache: Optional[List[int]] = [speed] * 6 if speed > 0 else None
        self._force_cache: Optional[List[int]] = [force] * 6 if force > 0 else None
        self._limits_check_interval = float(
            self.get_parameter("limits_check_interval_sec").value
        )
        self._last_limits_check = -math.inf
        self._limits_readback_supported = True

        self._stall_guard = bool(self.get_parameter("stall_guard").value)
        self._stall_backoff = max(0, int(self.get_parameter("stall_backoff").value))
        self._stall_holdoff = max(0.0, float(self.get_parameter("stall_holdoff_sec").value))
        self._clear_interval = max(
            0.0, float(self.get_parameter("clear_error_interval_sec").value)
        )
        # Per DOF: (minimum angle, monotonic expiry) while a stall is being
        # held off, else None.
        self._stall_floor: List[Optional[Tuple[int, float]]] = [None] * 6
        self._stalled: Set[int] = set()
        self._last_clear = -math.inf
        self._clears = 0
        self._stalls = 0
        self._last_clamp_log = -math.inf

        self._joint_names = [self._prefix + j for j in kin.ALL_JOINTS]
        self._driven_names = [self._prefix + j for j in kin.DRIVEN_JOINTS]

        self._transport = self._build_transport()
        self._connect()

        self._joint_state_pub = self.create_publisher(JointState, "~/joint_states", 10)
        self._state_pub = self.create_publisher(JointState, "~/state", 10)
        self._force_pub = self.create_publisher(JointState, "~/grip_force", 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, "~/diagnostics", 10)
        self._command_sub = self.create_subscription(
            JointState, "~/command", self._on_command, 10
        )
        self._angles_srv = self.create_service(
            SetAngles, "~/set_angles", self._on_set_angles
        )
        self._speed_srv = self.create_service(SetSpeed, "~/set_speed", self._on_set_speed)
        self._force_srv = self.create_service(SetForce, "~/set_force", self._on_set_force)
        self._clear_srv = self.create_service(Trigger, "~/clear_errors", self._on_clear_errors)

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().info(
            f"inspire_hand_driver up: "
            f"transport={'mock' if self._mock else self._transport.port} "
            f"protocol={self._transport.protocol} id={self._transport.hand_id} "
            f"rate={rate:.0f}Hz extras=1/{self._extras_divisor} prefix={self._prefix!r} "
            f"force_threshold={force if force > 0 else 'hand default'} "
            f"stall_guard={'on' if self._stall_guard else 'off'}"
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
        if not self._mock:
            self.get_logger().info(f"hand answered, reports ID {hand_id}")
        self._apply_limits("startup")

    def _apply_limits(self, reason: str) -> bool:
        """(Re)write the cached speed and force limits to the hand.

        Called at startup, when the hand comes back after going silent, and
        when a readback shows the force threshold is not what was written --
        all three are how a power-cycled hand, back on its power-on defaults,
        gets its threshold back without anyone noticing it was gone.
        """
        try:
            if self._speed_cache is not None:
                self._transport.write_speeds(self._speed_cache)
            if self._force_cache is not None:
                self._transport.write_forces(self._force_cache)
        except HandCommunicationError as exc:
            self.get_logger().warn(f"failed to apply speed/force limits ({reason}): {exc}")
            return False
        self._last_limits_check = time.monotonic()
        if self._force_cache is not None:
            self.get_logger().info(
                f"force threshold {self._force_cache} applied ({reason})"
            )
        return True

    def _check_limits(self) -> None:
        """Read FORCE_SET back and re-apply if the hand has lost it."""
        if (
            self._force_cache is None
            or not self._limits_readback_supported
            or self._limits_check_interval <= 0
            or time.monotonic() - self._last_limits_check < self._limits_check_interval
        ):
            return
        try:
            actual = self._transport.read_force_thresholds()
        except HandProtocolError as exc:
            # Some firmware makes FORCE_SET write-only; nothing to be done.
            self._limits_readback_supported = False
            self.get_logger().warn(f"hand refuses FORCE_SET readback, not checking again: {exc}")
            return
        except HandCommunicationError:
            return
        self._last_limits_check = time.monotonic()
        if list(actual) != list(self._force_cache):
            self.get_logger().warn(
                f"force threshold on the hand is {list(actual)}, expected "
                f"{self._force_cache}: the hand has probably rebooted; re-applying"
            )
            self._apply_limits("readback mismatch")

    # -- state publishing --------------------------------------------------
    def _on_timer(self) -> None:
        # The angles are read every cycle because they are what joint_states
        # is; current and force ride the divisor.
        extras = self._ticks % self._extras_divisor == 0
        self._ticks += 1
        try:
            angles = self._transport.read_angles()
            if extras:
                self._currents = self._transport.read_registers(REG_CURRENT, 6)
                self._forces = self._transport.read_forces()
                self._health = self._transport.read_health()
            currents, forces = self._currents, self._forces
        except HandCommunicationError as exc:
            self._failures += 1
            if self._failures == self._max_failures:
                self.get_logger().error(f"hand unresponsive after {self._failures} reads: {exc}")
            elif self._failures < self._max_failures:
                self.get_logger().debug(f"read failed ({self._failures}): {exc}")
            return

        if self._failures >= self._max_failures:
            self.get_logger().info("hand responsive again")
            # Silence long enough to count as lost is, in practice, a power
            # cycle, which wipes the volatile limits.
            self._apply_limits("hand back after being unresponsive")
        self._failures = 0

        if extras:
            if self._health is not None:
                self._supervise(angles, self._health)
            self._check_limits()

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

        if extras and self._health is not None:
            self._diag_pub.publish(self._diagnostics(stamp, angles, currents, forces, self._health))

    # -- stall guard -------------------------------------------------------
    def _dof_label(self, index: int) -> str:
        return f"{DOF_ORDER[index]} (channel {CHANNEL_IDS[index]})"

    def _supervise(self, angles: Sequence[int], health: HandHealth) -> None:
        """Back off and clear any DOF the firmware has stopped on a fault."""
        now = time.monotonic()
        stalled = health.stalled()
        for index in sorted(self._stalled.difference(stalled)):
            self.get_logger().info(
                f"{self._dof_label(index)} recovered: {describe_status(health.status[index])}"
            )
        fresh = [index for index in stalled if index not in self._stalled]
        self._stalled = set(stalled)
        if not stalled:
            return

        for index in fresh:
            self._stalls += 1
            self.get_logger().warn(
                f"{self._dof_label(index)} stalled: status "
                f"'{describe_status(health.status[index])}', error "
                f"{describe_error(health.errors[index])}"
                + ("" if self._stall_guard else " (stall_guard off: not intervening)")
            )
        if not self._stall_guard:
            return

        # Back off first, so that the cleared finger is not immediately driven
        # back into whatever stopped it.
        base = list(self._last_command) if self._last_command is not None else [
            a if a != ANGLE_INVALID else ANGLE_MAX for a in angles
        ]
        moved = []
        for index in stalled:
            actual = angles[index] if angles[index] != ANGLE_INVALID else base[index]
            floor = min(ANGLE_MAX, int(actual) + self._stall_backoff)
            self._stall_floor[index] = (floor, now + self._stall_holdoff)
            if base[index] < floor:
                base[index] = floor
                moved.append(f"{self._dof_label(index)} -> {angle_to_open_ratio(floor):.3f}")
        if moved:
            self.get_logger().warn("backing off stalled DOF: " + ", ".join(moved))
            self._send(base)

        if now - self._last_clear >= self._clear_interval:
            try:
                self._transport.clear_errors()
            except HandCommunicationError as exc:
                self.get_logger().warn(f"CLEAR_ERROR write failed: {exc}")
                return
            self._last_clear = now
            self._clears += 1
            self.get_logger().warn(
                f"wrote CLEAR_ERROR for {', '.join(self._dof_label(i) for i in stalled)}"
            )

    def _clamp_to_stall_floors(self, angles: List[int]) -> List[int]:
        """Hold DOF that recently stalled at their backed-off angle.

        Returns the DOF indices that were clamped, and drops floors whose
        hold-off has expired.
        """
        now = time.monotonic()
        clamped = []
        for index, floor in enumerate(self._stall_floor):
            if floor is None:
                continue
            minimum, expires = floor
            if now >= expires:
                self._stall_floor[index] = None
                continue
            if angles[index] < minimum:
                angles[index] = minimum
                clamped.append(index)
        if clamped and now - self._last_clamp_log >= 1.0:
            self._last_clamp_log = now
            self.get_logger().warn(
                "holding recently stalled DOF at their backed-off angle: "
                + ", ".join(
                    f"{self._dof_label(i)} >= {angle_to_open_ratio(self._stall_floor[i][0]):.3f}"
                    for i in clamped
                )
            )
        return clamped

    def _on_clear_errors(self, request, response):
        try:
            self._transport.clear_errors()
        except HandCommunicationError as exc:
            response.success, response.message = False, str(exc)
            return response
        self._last_clear = time.monotonic()
        self._clears += 1
        response.success = True
        response.message = (
            "CLEAR_ERROR written"
            if self._health is None
            else "CLEAR_ERROR written; before: " + ", ".join(
                f"{DOF_ORDER[i]}={describe_status(self._health.status[i])}/"
                f"{describe_error(self._health.errors[i])}"
                for i in range(6)
            )
        )
        return response

    def _diagnostics(
        self,
        stamp,
        angles: Sequence[int],
        currents: Sequence[int],
        forces: Sequence[int],
        health: HandHealth,
    ) -> DiagnosticArray:
        hardware = f"{self._transport.port}#{self._transport.hand_id}"
        targets = self._last_command
        thresholds = self._force_cache
        array = DiagnosticArray()
        array.header.stamp = stamp
        stalled = set(health.stalled())
        worst = DiagnosticStatus.OK
        for index in range(6):
            status = DiagnosticStatus()
            status.name = f"{self.get_name()}: {DOF_ORDER[index]}"
            status.hardware_id = f"{hardware}/{CHANNEL_IDS[index]}"
            if index in stalled:
                status.level = DiagnosticStatus.ERROR
            elif health.errors[index] or health.status[index] == STATUS_AT_FORCE:
                status.level = DiagnosticStatus.WARN
            else:
                status.level = DiagnosticStatus.OK
            worst = max(worst, status.level)
            status.message = describe_status(health.status[index])
            if health.errors[index]:
                status.message += f"; error {describe_error(health.errors[index])}"
            if index in stalled and self._stall_floor[index] is not None:
                status.message += "; backed off, held"
            status.values = [
                KeyValue(key="status", value=str(health.status[index])),
                KeyValue(key="error", value=str(health.errors[index])),
                KeyValue(key="temperature_c", value=str(health.temperatures[index])),
                KeyValue(key="current", value=str(currents[index])),
                KeyValue(key="force", value=str(forces[index])),
                KeyValue(
                    key="force_threshold",
                    value=str(thresholds[index]) if thresholds is not None else "hand default",
                ),
                KeyValue(
                    key="open_ratio",
                    value=f"{angle_to_open_ratio(angles[index]):.3f}"
                    if angles[index] != ANGLE_INVALID else "invalid",
                ),
                KeyValue(
                    key="target_open_ratio",
                    value=f"{angle_to_open_ratio(targets[index]):.3f}"
                    if targets is not None else "none",
                ),
            ]
            array.status.append(status)

        summary = DiagnosticStatus()
        summary.name = f"{self.get_name()}: hand"
        summary.hardware_id = hardware
        summary.level = worst
        summary.message = (
            "ok" if not stalled
            else f"{len(stalled)} DOF stalled: " + ", ".join(DOF_ORDER[i] for i in sorted(stalled))
        )
        summary.values = [
            KeyValue(key="stall_guard", value="on" if self._stall_guard else "off"),
            KeyValue(key="stalls_seen", value=str(self._stalls)),
            KeyValue(key="errors_cleared", value=str(self._clears)),
            KeyValue(key="read_failures", value=str(self._failures)),
        ]
        array.status.insert(0, summary)
        return array

    # -- command paths -----------------------------------------------------
    def _resolve(self, name: str) -> int:
        """Resolve a channel id or (prefixed) driven joint name to a DOF index."""
        key = str(name)
        if self._prefix and key.startswith(self._prefix):
            key = key[len(self._prefix) :]
        return kin.dof_index(key)

    def _merge(
        self, names: Sequence[str], values: Sequence[float]
    ) -> Tuple[Optional[List[int]], str]:
        """Merge a partial set of named open-ratio targets onto the last command.

        Unaddressed DOF keep their current target rather than snapping to a
        default -- which is what makes a partial command safe to send.
        """
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
            ratio = command_overlays.apply_open_ratio_overlay(index, float(value))
            angles[index] = open_ratio_to_angle(ratio)
            touched = True

        if unknown:
            self.get_logger().warn(f"ignoring unknown hand channels: {', '.join(unknown)}")
        if not touched:
            return None, f"no recognised channel named (got: {', '.join(map(str, names))})"
        self._clamp_to_stall_floors(angles)
        return angles, ""

    def _send(self, angles: List[int]) -> Tuple[bool, str]:
        try:
            self._transport.write_angles(angles)
        except HandCommunicationError as exc:
            self.get_logger().warn(f"failed to write angles: {exc}")
            return False, str(exc)
        self._last_command = angles
        return True, ""

    def _apply(
        self, names: Sequence[str], values: Sequence[float]
    ) -> Tuple[bool, str]:
        """Write a partial set of named open-ratio targets.

        Names address DOF and nothing else: either channel ids or driven joint
        names, and the two may be mixed freely because they no longer select a
        unit. ``values`` are always open ratios, 1.0 fully open. That used to
        be inferred from the naming, so the same number meant opposite ends of
        travel depending on how the DOF was addressed.
        """
        if not names:
            return False, "no channels named"
        if len(names) != len(values):
            return False, f"name has {len(names)} entries but value has {len(values)}"
        bad = out_of_range(names, values)
        if bad:
            return False, (
                f"open ratio out of the commandable range "
                f"[{RATIO_CLOSED:g}, {RATIO_OPEN:g}] "
                f"(1.0 = fully open, 0.0 = fully closed): {', '.join(bad)}"
            )
        angles, error = self._merge(names, values)
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
        # Read-modify-write is not relied on (these registers are write-only
        # on some firmware), so an unaddressed DOF is rewritten with the value
        # this node last sent, defaulting to the hand's own mid-scale.
        cache = getattr(self, f"_{what}_cache") or [500] * 6
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
