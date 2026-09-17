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
    ``~/set_compliance`` ``std_srvs/srv/SetBool``
        Enter or leave compliant mode -- see below.
    ``~/tare_force``     ``std_srvs/srv/Trigger``
        Take the current fingertip readings as "nothing is touching me".
        Entering compliant mode does this on its own; this is for when the
        zero has moved since, which it does after a heavy push.
    ``~/calibrate_force`` ``std_srvs/srv/Trigger``
        Run the hand's *own* force-sensor calibration -- see below. The node
        opens the hand clear first, then the hand moves by itself for six
        seconds; nothing may be touching it.

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

Compliant mode
--------------
``~/set_compliance`` (or ``compliance:=true`` at launch) puts the hand into a
mode where pushing on a fingertip opens that finger, so a grasp can be adjusted
by hand without fighting it. It is the nearest thing this hand has to the arm's
gravity-compensation controller and it is not very near: the FR3 floats because
it is backdrivable and commands zero torque, while the RH56 takes positions and
nothing else. The give is therefore manufactured -- fingertip force is read
every cycle and the finger's target is retreated towards open in proportion to
it. :mod:`inspire_hand_driver.compliance` carries the law, the tuning, and what
the fingertip sensor can and cannot feel; this node carries the plumbing.

Force is measured against a captured zero, not against nought, because the
fingertip sensors do not return to the same reading after a heavy load: on this
rig a pad that rested at -11 g sat at +219 g after being pushed to 2511 g, and
stayed there. Engaging compliant mode takes a tare, so the hand must not be
touched at that moment; ``~/tare_force`` takes another whenever the zero has
moved since. Without it a finger holds a permanent partial yield and never
comes home, which is what the bench check reports as "gave, did not come back".

The yield is an *offset*, never a command. ``~/command`` and ``~/set_angles``
still set the rest position, the offset is added on the way to the registers,
and ``~/state`` still reports where the fingers physically are. A partial
command merges onto the commanded pose rather than onto wherever a finger has
been pushed to, so a grasp held through a compliant episode returns to exactly
the grasp that was commanded.

Force-sensor calibration
------------------------
``~/calibrate_force`` triggers GESTURE_FORCE_CLB, the routine the Inspire
desktop app calls calibration, and it is worth being clear about how little it
has in common with ``~/tare_force``. The tare is arithmetic in this node: a
baseline is captured and subtracted, only the compliance law sees it, and it is
undone by taking another. The calibration is the hand rewriting what
``FORCE_ACT`` reports at the source, for every reader, permanently.

**It jammed the hand this was written against**, which is why
``calibration_mode`` defaults to ``none`` and the service refuses. A hand that
drives itself for six seconds with the stall guard stood down is not something
to leave an open service to on the strength of a manual paragraph.

The routine is one register write and a fixed sequence, so it cannot be asked
to calibrate the fingers and leave the thumb alone. The only lever this node
has is *when to stop it*. The sequence runs fingers first -- open all five,
bend the four -- and reaches the thumb last, so ``calibration_mode="fingers"``
stops it at ``calibration_finger_sec`` by writing GESTURE_FORCE_CLB back to 0
before the thumb steps. That is the mode that jammed, and two undocumented
things are the likely reason. Whether the firmware honours a 0 written
mid-sequence is unknown, so after the stop the thumb is watched for
``THUMB_SETTLE_SEC``: if it has not settled onto this node's own target by
then it is still being driven by the routine, and the log says so in as many
words -- two writers on one set of actuators is exactly what a jam looks like.
And whether a routine cut short commits the finger calibration at all is
equally unknown, so the mode may buy nothing even when it behaves. Compare
resting ``~/grip_force`` before and after to find out.

The hand is also posed before the register is written, waited for rather than
assumed, so the sequence starts from the pose its first step asks for. Which
DOF get posed follows ``calibration_mode``: with the thumb taking part, all
six open, which swings thumb rotation clear of the fingers' sweep; with it out,
only the four fingers, because otherwise this node would be the only reason the
thumb moves at all. A DOF that will not reach the pose within
``calibration_clearance_sec`` abandons the calibration rather than starting it
anyway; ``calibration_clearance:=false`` skips the staging for anyone who
wants to pose the hand themselves.

None of which makes the thumb perfectly still. The routine's own first step is
"hold five fingers fully open", and five includes the thumb, so the thumb
*extends* whatever this node does. What ``"fingers"`` keeps it out of
is the bending, three steps later, which is where it meets the fingers.

For those seconds the *hand* is the one commanding: it opens all five fingers,
bends the four, bends the thumb, extends it again. So this node stops writing
for the duration -- every command path returns "force calibration in progress"
rather than queueing, and the stall guard and the limits readback both stand
down, because a routine that drives fingers into their own limits is exactly
what the stall guard exists to interrupt. State keeps being published
throughout, so the motion is visible in ``~/joint_states`` as it happens rather
than as a six-second freeze followed by a jump.

When the routine ends -- or is stopped -- the node takes a fresh tare (the zero it was
holding describes sensors that no longer exist) and re-sends the pose that was
commanded before staging opened the hand, since the routine leaves the fingers
wherever its last step put them. Compliant mode is refused while a calibration runs and a calibration
is refused while compliance is doing anything, in either case because a yield
derived from readings taken mid-calibration is a number about nothing.

It costs a register read. Force normally rides ``state_extras_divisor``, which
under the replay launch's divisor of 5 would sample it at 10 Hz; compliant mode
reads it every cycle regardless, so a cycle becomes angles + force + a write
rather than angles alone. That is roughly 7-21 ms of a 20 ms slot at 50 Hz (see
:mod:`inspire_hand_driver.benchmark`), so a hand being streamed targets at the
same time wants a lower ``publish_rate_hz`` while it is compliant.

Every gain is a dynamic parameter, so ``ros2 param set`` retunes the spring
between one cycle and the next -- which is the only workable way to find
numbers for something whose test is pushing on it with a finger. Values the law
rejects are refused by the parameter callback instead of being applied.

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
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

from inspire_hand_msgs.srv import SetAngles, SetForce, SetSpeed

from . import command_overlays
from . import compliance
from . import kinematics as kin
from .protocol import (
    ANGLE_INVALID,
    ANGLE_MAX,
    CHANNEL_IDS,
    DOF_ORDER,
    FORCE_CALIBRATION_SECONDS,
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

#: The two DOF the firmware's calibration routine moves last, and the ones
#: ``calibration_mode="fingers"`` exists to keep still.
THUMB_DOF = (DOF_ORDER.index("thumb_bend"), DOF_ORDER.index("thumb_rotation"))
FINGER_DOF = tuple(i for i in range(6) if i not in THUMB_DOF)

#: What ``~/calibrate_force`` is allowed to do.
#:
#: ``none``    Refuse. The default, because on the hand this was developed
#:             against the routine jams the fingers, and a self-driving hand
#:             that jams is not something to leave a service open for.
#: ``fingers`` Start the routine and write GESTURE_FORCE_CLB back to 0 before
#:             the thumb steps. **Known to jam this hand**: a routine cut
#:             short does not obviously stop, and the fingers were left
#:             driving closed. Kept because it is the only shape a
#:             finger-only calibration could take, and because a different
#:             firmware may honour the write.
#: ``full``    Let the routine run all four of its steps, thumb included.
#:             What the Inspire desktop app does.
#: Not "off": YAML reads that as the boolean False, so `calibration_mode:=off`
#: on a command line never reaches this node as a string at all.
CALIBRATION_MODES = ("none", "fingers", "full")


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
        # Compliant mode: pushing a fingertip opens that finger. Off unless
        # asked for. See "Compliant mode" above, and
        # inspire_hand_driver.compliance for the law and what limits it.
        # Clearance staging for ~/calibrate_force -- see "Force-sensor
        # calibration" above for why the hand has to be posed before the
        # firmware's routine can be allowed to run.
        self.declare_parameter("calibration_clearance", True)
        self.declare_parameter("calibration_clearance_sec", 3.0)
        self.declare_parameter("calibration_mode", "none")
        self.declare_parameter("calibration_finger_sec", 3.0)
        self.declare_parameter("compliance", False)
        # Which DOF give. Thumb rotation (channel 6) is left out because it
        # carries no fingertip pad, so its force reading has nothing to say
        # about anyone pushing on it.
        self.declare_parameter("compliance_channels", ["1", "2", "3", "4", "5"])
        self.declare_parameter("compliance_deadband", compliance.DEFAULT_DEADBAND)
        self.declare_parameter("compliance_counts_per_gram", compliance.DEFAULT_COUNTS_PER_GRAM)
        self.declare_parameter("compliance_max_yield", compliance.DEFAULT_MAX_YIELD)
        self.declare_parameter("compliance_yield_rate", compliance.DEFAULT_YIELD_RATE)
        self.declare_parameter("compliance_return_rate", compliance.DEFAULT_RETURN_RATE)

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
        # _forces starts as zeros that were never measured; a tare against
        # those would be a fiction, so it waits for the first real read.
        self._had_force_read = False
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

        # The compliance spring, and the opening offset it currently applies.
        # The offset is added to targets on the way out in _send; nothing else
        # in the node knows about it, so every command path carries it without
        # having to opt in.
        self._spring = compliance.FingerSpring(
            self._gains_from_parameters(), self._compliance_mask()
        )
        self._yield: List[int] = [0] * 6
        self._last_spring_tick: Optional[float] = None
        # Set when compliance is engaged before any force has been read -- at
        # launch, say. The tare then happens on the first cycle that has a
        # reading to take, because taring against the initial zeros would make
        # every real resting offset look like a push.
        self._tare_pending = False
        self._last_yield_log = -math.inf
        # Monotonic deadline while the hand is running its own force-sensor
        # calibration, else None. Every write in this node is suspended until
        # it passes -- see "Force-sensor calibration" above.
        self._calibrating_until: Optional[float] = None
        # While the hand is being opened into the clearance pose, before the
        # routine is triggered. Separate from _calibrating_until because this
        # node is still the one commanding during it.
        self._staging_until: Optional[float] = None
        # The pose to put back afterwards. Not _last_command, which by then is
        # the clearance pose this node commanded on the caller's behalf.
        self._pose_before_calibration: Optional[List[int]] = None
        self._calibrations = 0
        self._clearance = bool(self.get_parameter("calibration_clearance").value)
        self._clearance_sec = max(0.0, float(self.get_parameter("calibration_clearance_sec").value))
        self._calibration_mode = str(self.get_parameter("calibration_mode").value).lower()
        if self._calibration_mode not in CALIBRATION_MODES:
            self.get_logger().error(
                f"calibration_mode={self._calibration_mode!r} is not one of "
                f"{', '.join(CALIBRATION_MODES)}; refusing calibrations"
            )
            self._calibration_mode = "none"
        self._calibrate_thumb = self._calibration_mode == "full"
        self._finger_sec = max(0.5, float(self.get_parameter("calibration_finger_sec").value))
        # Monotonic deadline after an early stop, by which the thumb should
        # have settled onto whatever this node last commanded. The firmware is
        # not documented to honour a stop request, so the node checks rather
        # than assuming, and says which way it went.
        self._thumb_watch: Optional[float] = None
        #: DOF indices the pending calibration posed, and so the only ones its
        #: clearance check may wait on.
        self._staged: List[int] = []

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
        self._compliance_srv = self.create_service(
            SetBool, "~/set_compliance", self._on_set_compliance
        )
        self._tare_srv = self.create_service(Trigger, "~/tare_force", self._on_tare_force)
        self._calibrate_srv = self.create_service(
            Trigger, "~/calibrate_force", self._on_calibrate_force
        )
        self.add_on_set_parameters_callback(self._on_set_parameters)

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._period = 1.0 / rate
        self._timer = self.create_timer(self._period, self._on_timer)
        if bool(self.get_parameter("compliance").value):
            self._set_compliance(True, "launch parameter")

        self.get_logger().info(
            f"inspire_hand_driver up: "
            f"transport={'mock' if self._mock else self._transport.port} "
            f"protocol={self._transport.protocol} id={self._transport.hand_id} "
            f"rate={rate:.0f}Hz extras=1/{self._extras_divisor} prefix={self._prefix!r} "
            f"force_threshold={force if force > 0 else 'hand default'} "
            f"stall_guard={'on' if self._stall_guard else 'off'} "
            f"compliance={'on' if self._spring.engaged else 'off'}"
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
        # is; current and force ride the divisor. Force is the exception while
        # the spring is working: it is that loop's only input, and sampling it
        # at a fifth of the rate the loop runs at would hand the spring a
        # staircase to differentiate.
        extras = self._ticks % self._extras_divisor == 0
        self._ticks += 1
        compliant = self._spring.active
        try:
            angles = self._transport.read_angles()
            if extras or compliant:
                self._forces = self._transport.read_forces()
                self._had_force_read = True
            if extras:
                self._currents = self._transport.read_registers(REG_CURRENT, 6)
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
            # cycle, which wipes the volatile limits. It also makes the yield
            # meaningless -- it was derived from a force reading against a
            # pose this hand no longer holds -- so it goes, while the mode
            # stays and re-derives itself from the next reading.
            self._apply_limits("hand back after being unresponsive")
            self._spring.reset()
            self._yield = [0] * 6
            self._last_spring_tick = None
        self._failures = 0

        if self._staging_until is not None:
            self._advance_staging(angles)
        if self._thumb_watch is not None:
            self._check_thumb_released(angles)

        calibrating = self._calibrating_until is not None
        if calibrating and time.monotonic() >= self._calibrating_until:
            self._finish_calibration(angles)
            calibrating = False

        # Both of these write, and the stall guard would read a hand driving
        # its own fingers into their limits as six faults to intervene in.
        if extras and not calibrating:
            if self._health is not None:
                self._supervise(angles, self._health)
            self._check_limits()

        if self._tare_pending and self._had_force_read:
            self._tare("deferred from when compliance was enabled")

        # After the stall guard, so that a DOF backed off this cycle is what
        # the yield is added to rather than the target it stalled on.
        if compliant:
            self._update_compliance(angles)

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

    # -- compliant mode ----------------------------------------------------
    #: Spring gain -> the parameter that carries it. One table so that the
    #: startup build and the live retune cannot come to disagree about which
    #: parameter means what.
    GAIN_PARAMETERS = {
        "deadband": "compliance_deadband",
        "counts_per_gram": "compliance_counts_per_gram",
        "max_yield": "compliance_max_yield",
        "yield_rate": "compliance_yield_rate",
        "return_rate": "compliance_return_rate",
    }

    def _gains_from_parameters(
        self, overrides: Optional[Dict[str, object]] = None
    ) -> compliance.SpringGains:
        """Build the gains from the parameters, with pending values applied.

        ``overrides`` exists for the parameter callback: it runs *before* the
        new values are stored, so reading them back would validate the ones
        being replaced.
        """
        overrides = overrides or {}
        return compliance.SpringGains(
            **{
                field: float(
                    overrides[param] if param in overrides else self.get_parameter(param).value
                )
                for field, param in self.GAIN_PARAMETERS.items()
            }
        )

    def _compliance_mask(self, names: Optional[Sequence[str]] = None) -> List[bool]:
        """Resolve ``compliance_channels`` to the six flags the spring takes."""
        if names is None:
            names = list(self.get_parameter("compliance_channels").value or [])
        indices, unknown = [], []
        for name in names:
            try:
                indices.append(self._resolve(name))
            except KeyError:
                unknown.append(str(name))
        if unknown:
            self.get_logger().warn(
                f"ignoring unknown compliance channels: {', '.join(unknown)}"
            )
        return compliance.channel_mask(indices)

    def _set_compliance(self, enable: bool, reason: str) -> str:
        if enable and (self._calibrating_until is not None or self._staging_until is not None):
            self.get_logger().warn(f"compliant mode refused ({reason}): {self.CALIBRATING_MESSAGE}")
            return self.CALIBRATING_MESSAGE
        if enable == self._spring.engaged:
            return f"compliance was already {'on' if enable else 'off'}"
        if enable:
            self._spring.engage()
            # The first cycle of a new episode has no previous tick to measure
            # against, and must not integrate however long the mode was off.
            self._last_spring_tick = None
            # Whatever the fingertips read now is what "untouched" means for
            # this episode. The sensors' zero moves after a heavy push, so a
            # fixed nought would hold every pushed finger permanently open.
            self._tare("entering compliant mode")
            giving = [
                DOF_ORDER[i] for i, on in enumerate(self._spring.channels) if on
            ] or ["(none)"]
            message = (
                f"compliant mode on ({reason}): {', '.join(giving)} give to fingertip "
                f"force; {compliance.describe(self._spring.gains)}"
            )
        else:
            self._spring.release()
            message = f"compliant mode off ({reason}): ramping the yield back out"
        self.get_logger().info(message)
        return message

    def _tare(self, reason: str) -> str:
        """Zero the fingertip readings, once there is a reading to zero against."""
        if not self._had_force_read:
            self._tare_pending = True
            return f"tare deferred until the first force read ({reason})"
        zeros = self._spring.tare(self._forces)
        self._tare_pending = False
        message = (
            f"fingertip zero set ({reason}): "
            + ", ".join(f"{DOF_ORDER[i]} {zeros[i]:.0f}g" for i in range(6))
        )
        self.get_logger().info(message)
        return message

    def _on_tare_force(self, request, response):
        response.message = self._tare("service call")
        response.success = True
        return response

    def _on_set_compliance(self, request, response):
        response.message = self._set_compliance(bool(request.data), "service call")
        # Read the answer off the spring rather than assuming it: the request
        # can be refused, and a refusal reported as success is how a caller
        # ends up believing a hand is compliant when it is not.
        response.success = self._spring.engaged == bool(request.data)
        return response

    def _on_set_parameters(self, params) -> SetParametersResult:
        """Retune the spring live; refuse values the law will not take.

        Tuning a spring you test by pushing on it means changing a gain and
        pushing again, so this has to work on a running hand. Rejecting rather
        than clamping matters more here than elsewhere: a silently clamped gain
        is indistinguishable, by feel, from one that did nothing.
        """
        pending = {p.name: p.value for p in params if p.name.startswith("compliance")}
        if not pending:
            return SetParametersResult(successful=True)
        try:
            gains = self._gains_from_parameters(pending)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        if "compliance_channels" in pending:
            try:
                self._spring.set_channels(self._compliance_mask(pending["compliance_channels"]))
            except (TypeError, ValueError) as exc:
                return SetParametersResult(successful=False, reason=str(exc))
        if gains != self._spring.gains:
            self._spring.retune(gains)
            self.get_logger().info(f"compliance retuned: {compliance.describe(gains)}")
        if "compliance" in pending:
            self._set_compliance(bool(pending["compliance"]), "parameter set")
        return SetParametersResult(successful=True)

    def _update_compliance(self, angles: Sequence[int]) -> None:
        """Advance the yield against the latest force, and rewrite if it moved."""
        now = time.monotonic()
        # A first cycle, or one after a run of dropped reads, must not
        # integrate the whole gap: the rate limits are there to bound how far
        # a finger moves per cycle, and a long dt would step straight past them.
        if self._last_spring_tick is None:
            dt = self._period
        else:
            dt = min(4.0 * self._period, now - self._last_spring_tick)
        self._last_spring_tick = now

        previous = self._yield
        self._yield = self._spring.update(self._forces, dt)
        if self._yield == previous:
            return
        anchor = self._last_command
        if anchor is None:
            # Nothing has commanded the hand this session, so its own pose is
            # the rest position -- the same choice _merge makes.
            anchor = [a if a != ANGLE_INVALID else ANGLE_MAX for a in angles]
        self._send(list(anchor))
        self._log_yield(now)

    def _log_yield(self, now: float) -> None:
        """Say what is giving, at most once a second. This is a tuning aid."""
        giving = [i for i, counts in enumerate(self._yield) if counts]
        if not giving or now - self._last_yield_log < 1.0:
            return
        self._last_yield_log = now
        saturated = (
            " -- at max_yield"
            if any(self._yield[i] >= self._spring.gains.max_yield for i in giving)
            else ""
        )
        self.get_logger().info(
            "giving to fingertip force: "
            + ", ".join(
                f"{DOF_ORDER[i]} {self._yield[i]} counts @ {self._forces[i]}g" for i in giving
            )
            + saturated
        )

    # -- force-sensor calibration -----------------------------------------
    #: How close a DOF has to be to the clearance pose, in counts, before the
    #: routine may start. 3 % of travel: tight enough that a thumb still lying
    #: across the palm fails it, loose enough that a healthy hand always
    #: passes without waiting on the last few counts of a slew.
    CLEARANCE_TOLERANCE = 30

    def _on_calibrate_force(self, request, response):
        response.success, response.message = self._start_calibration("service call")
        return response

    def _start_calibration(self, reason: str) -> Tuple[bool, str]:
        """Open the hand clear, then hand the next six seconds to the firmware."""
        if self._calibrating_until is not None:
            remaining = self._calibrating_until - time.monotonic()
            return False, f"a force calibration is already running ({remaining:.1f}s left)"
        if self._staging_until is not None:
            return False, "a force calibration is already being staged"
        if self._calibration_mode == "none":
            return False, (
                "calibration_mode is 'none'. The hand's own routine drives its fingers "
                "for six seconds and cannot be told to skip the thumb; stopping it "
                "early (calibration_mode:='fingers') jammed the fingers on this hand, "
                "and running it whole (calibration_mode:='full') moves the thumb. "
                "~/tare_force does the zeroing compliant mode actually needs, without "
                "moving anything."
            )
        if self._spring.active:
            # Not merely tidiness. The routine drives the fingers, the spring
            # would read the resulting fingertip loads as someone pushing, and
            # the two would write opposing targets to the same registers.
            return False, (
                "compliant mode is still working: turn it off with ~/set_compliance "
                "and let the yield ramp out before calibrating"
            )

        # Captured before staging moves it, because staging is a command like
        # any other and overwrites _last_command on its way out.
        self._pose_before_calibration = (
            list(self._last_command) if self._last_command is not None else None
        )
        if not self._clearance:
            return self._trigger_calibration(reason)

        # Staging must not move what the calibration is being kept away from.
        # Opening all six is right when the thumb is taking part -- it swings
        # thumb rotation clear of the fingers' sweep -- and is exactly the
        # wrong thing when it is not, because then this node is the only
        # reason the thumb moves at all.
        self._staged = list(range(6)) if self._calibrate_thumb else list(FINGER_DOF)
        accepted, error = self._apply(
            [CHANNEL_IDS[i] for i in self._staged], [RATIO_OPEN] * len(self._staged)
        )
        if not accepted:
            self._pose_before_calibration = None
            return False, f"could not pose the hand for calibration: {error}"
        self._staging_until = time.monotonic() + self._clearance_sec
        posed = (
            "the whole hand"
            if self._calibrate_thumb
            else "the four fingers, leaving the thumb where it is"
        )
        self.get_logger().info(
            f"force-sensor calibration requested ({reason}): first opening {posed}"
        )
        seconds = FORCE_CALIBRATION_SECONDS if self._calibrate_thumb else self._finger_sec
        return True, (
            f"opening {posed}, then calibrating for about {seconds:.1f}s. Nothing may "
            "touch the hand. This service returns before any of it happens -- watch the "
            "log for 'force-sensor calibration finished'."
        )

    def _advance_staging(self, angles: Sequence[int]) -> None:
        """Wait for the clearance pose, then trigger -- or give up and say why.

        The firmware's routine never commands thumb rotation, so whatever a
        previous grasp left there is where the thumb bends *from*. That is the
        one axis that can put the thumb in the fingers' way, and this is the
        only chance to move it.
        """
        off = [
            i
            for i in self._staged
            if angles[i] == ANGLE_INVALID
            or abs(int(angles[i]) - ANGLE_MAX) > self.CLEARANCE_TOLERANCE
        ]
        if not off:
            self._staging_until = None
            ok, message = self._trigger_calibration("staged")
            if not ok:
                self.get_logger().warn(f"force calibration could not be started: {message}")
                self._pose_before_calibration = None
            return
        if time.monotonic() < self._staging_until:
            return
        # Refusing here rather than calibrating anyway: a DOF that will not
        # open is either jammed or stalled, and running the routine with one
        # finger out of place is the collision this staging exists to prevent.
        self._staging_until = None
        self._pose_before_calibration = None
        self.get_logger().error(
            "force calibration abandoned: "
            + ", ".join(
                f"{self._dof_label(i)} is at "
                + (
                    "an invalid reading"
                    if angles[i] == ANGLE_INVALID
                    else f"{angle_to_open_ratio(angles[i]):.3f}"
                )
                for i in off
            )
            + f" after {self._clearance_sec:.1f}s and will not open. The hand is left "
            "open; clear whatever is holding those DOF and try again."
        )

    #: How long the thumb is given to settle onto this node's own target after
    #: an early stop, and how far off it may be and still count as settled.
    THUMB_SETTLE_SEC = 1.5
    THUMB_SETTLE_TOLERANCE = 60

    def _check_thumb_released(self, angles: Sequence[int]) -> None:
        """Say whether the firmware actually let go when asked to stop.

        The question this answers is not "did the thumb move" -- this node
        moves it itself, restoring the pose -- but "is the thumb following
        this node or still following the routine". A thumb sitting far from
        the target a second and a half after the stop is being driven by
        something else.
        """
        if time.monotonic() < self._thumb_watch:
            return
        self._thumb_watch = None
        if self._last_command is None:
            return
        adrift = [
            i
            for i in THUMB_DOF
            if angles[i] == ANGLE_INVALID
            or abs(int(angles[i]) - self._last_command[i]) > self.THUMB_SETTLE_TOLERANCE
        ]
        if not adrift:
            self.get_logger().info(
                "the routine released the thumb when asked: it is following commands again"
            )
            return
        self.get_logger().error(
            "the hand did NOT stop its calibration routine when asked: "
            + ", ".join(
                f"{self._dof_label(i)} is at "
                + (
                    "an invalid reading"
                    if angles[i] == ANGLE_INVALID
                    else f"{angle_to_open_ratio(angles[i]):.3f}"
                )
                + f" against a commanded {angle_to_open_ratio(self._last_command[i]):.3f}"
                for i in adrift
            )
            + ". Writing 0 to GESTURE_FORCE_CLB does not abort it on this firmware, so the "
            "thumb cannot be kept out of a calibration -- either accept it with "
            "calibration_mode:=full, or do not use ~/calibrate_force on this hand."
        )

    def _trigger_calibration(self, reason: str) -> Tuple[bool, str]:
        """Write GESTURE_FORCE_CLB. The hand is the one commanding from here."""
        try:
            self._transport.calibrate_force_sensors()
        except HandCommunicationError as exc:
            self.get_logger().warn(f"force calibration could not be started: {exc}")
            return False, str(exc)

        seconds = FORCE_CALIBRATION_SECONDS if self._calibrate_thumb else self._finger_sec
        self._calibrating_until = time.monotonic() + seconds
        self._calibrations += 1
        thumb = (
            "all six DOF"
            if self._calibrate_thumb
            else f"the four fingers, then stopping at {seconds:.1f}s before the thumb steps"
        )
        self.get_logger().warn(
            f"force-sensor calibration started ({reason}): the hand will move its own "
            f"fingers for {seconds:.1f}s and must not be touched -- {thumb}. "
            f"Commands are refused until it finishes."
        )
        return True, (
            f"calibration started; the hand moves on its own for about {seconds:.1f}s "
            f"({thumb}) and nothing may touch it. This service returns before the "
            f"routine does -- watch for 'force-sensor calibration finished' in the log."
        )

    def _finish_calibration(self, angles: Sequence[int]) -> None:
        """Resume writing, re-zero against the new sensors, restore the pose."""
        self._calibrating_until = None
        if self._calibrate_thumb:
            self.get_logger().info("force-sensor calibration finished")
        else:
            # The routine is not over; we are cutting it short before its
            # thumb steps. Whether the firmware lets go when asked is not
            # documented, so ask, then watch the thumb and say what happened.
            try:
                self._transport.stop_force_calibration()
            except HandCommunicationError as exc:
                self.get_logger().warn(f"could not ask the routine to stop: {exc}")
            self._thumb_watch = time.monotonic() + self.THUMB_SETTLE_SEC
            self.get_logger().info(
                f"force-sensor calibration stopped at {self._finger_sec:.1f}s, before the "
                f"thumb steps (calibration_mode is 'fingers')"
            )
        # The old zero described sensors that no longer exist. Taking a new one
        # is only valid because the calibration required an untouched hand in
        # the first place, so the precondition is already met.
        if self._had_force_read:
            self.get_logger().info(self._tare("after force calibration"))
        else:
            self._tare_pending = True
        # The routine leaves the fingers wherever its last step put them, which
        # is not what anyone commanded. Put them back -- to the pose from
        # before staging opened the hand, not to the clearance pose.
        restore, self._pose_before_calibration = self._pose_before_calibration, None
        if restore is not None:
            ok, error = self._send(list(restore))
            if ok:
                self.get_logger().info("restored the commanded pose after calibration")
            else:
                self.get_logger().warn(f"could not restore the commanded pose: {error}")

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
            if self._yield[index]:
                status.message += f"; giving {self._yield[index]} counts to fingertip force"
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
                # Counts of opening added to that target by compliant mode, so
                # a target and a pose that disagree have somewhere to say why.
                KeyValue(key="compliance_yield", value=str(self._yield[index])),
                # What this channel currently calls "nothing touching me". It
                # is not 0, and it moves: see "Compliant mode" above.
                KeyValue(key="force_zero", value=f"{self._spring.zeros[index]:.0f}"),
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
            KeyValue(key="compliance", value="on" if self._spring.engaged else "off"),
            KeyValue(
                key="force_calibration",
                value=(
                    "running"
                    if self._calibrating_until is not None
                    else "staging" if self._staging_until is not None else "idle"
                ),
            ),
            KeyValue(key="force_calibrations", value=str(self._calibrations)),
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

    #: What every write path says while the hand is calibrating its own force
    #: sensors. Refused rather than queued: six seconds later the command is
    #: stale, and a caller told "no" can decide that for itself.
    CALIBRATING_MESSAGE = (
        "force calibration in progress: the hand is running its own motion and "
        "this node is not writing until it finishes"
    )
    #: And while the hand is being posed for one. Short-lived, but a command
    #: landing here would undo the clearance a moment before the firmware
    #: starts swinging the thumb through it.
    STAGING_MESSAGE = "the hand is being posed for a force calibration"

    def _send(self, angles: List[int]) -> Tuple[bool, str]:
        if self._calibrating_until is not None:
            return False, self.CALIBRATING_MESSAGE
        try:
            self._transport.write_angles(self._with_yield(angles))
        except HandCommunicationError as exc:
            self.get_logger().warn(f"failed to write angles: {exc}")
            return False, str(exc)
        self._last_command = angles
        return True, ""

    def _with_yield(self, angles: Sequence[int]) -> List[int]:
        """Add the compliance yield to a set of targets, towards open.

        ``_last_command`` deliberately keeps the *unyielded* targets. The
        commanded pose is the spring's rest position, so a partial command
        merges onto where the caller asked a finger to be and not onto
        wherever someone has pushed it; and leaving compliant mode returns the
        hand to the grasp that was commanded, with no bookkeeping anywhere
        else in the node.
        """
        if not any(self._yield):
            return list(angles)
        return [min(ANGLE_MAX, a + y) for a, y in zip(angles, self._yield)]

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
        if self._staging_until is not None:
            return False, self.STAGING_MESSAGE
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
        if self._calibrating_until is not None:
            return False, self.CALIBRATING_MESSAGE
        if self._staging_until is not None:
            return False, self.STAGING_MESSAGE
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
