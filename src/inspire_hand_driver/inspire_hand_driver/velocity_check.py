"""Bench check for the joint velocities the driver differences from ANGLE_ACT.

Two things in one tool, because they answer the same question:

* ``--live`` (the default) prints the speed of every DOF as it moves, bars
  growing either side of zero. With ``--compliant`` the fingers give, so you
  can push them by hand and watch the reading follow.
* ``--sweep`` commands one DOF across its range and measures how good the
  reading is, in numbers rather than by eye.

    ros2 run inspire_hand_driver inspire_hand_velocity_check
    ros2 run inspire_hand_driver inspire_hand_velocity_check --sweep
    ros2 run inspire_hand_driver inspire_hand_velocity_check --sweep \\
        --channel 4 --speed 400 --filter 0

What "accurate" can mean here, and what it cannot
-------------------------------------------------
The hand has no speed sensor. ``JointState.velocity`` is a backward difference
of the ANGLE readings, so it cannot be checked against a measurement of the
same quantity -- there is none. What it can be checked against:

* **The position stream and the clock.** A centred difference of the same
  positions is a strictly better estimator than the driver's backward one: no
  half-sample lag, and it averages two intervals of quantisation noise instead
  of trusting one. The driver's number is compared against that, which is what
  isolates the filter's lag, the clamp, and any mishandled sample interval.
* **Displacement over the move.** The integral of a correct rate is the
  distance travelled, and the distance is read off the positions directly. An
  integral that comes up short is a filter or a clamp eating real motion.
* **Inspire's own specification.** Manual 2.4.8: at ``SPEED_SET`` 1000 an
  unloaded DOF crosses its whole range in 800 ms. That bound comes from
  outside this workspace entirely, so comparing the measured peak against it
  tests :func:`~inspire_hand_driver.kinematics.speed_counts_to_rad_per_s`, the
  one place where a number was assumed rather than measured.
* **Standstill.** With the hand still, every honest rate is zero. What it
  actually reports is the noise floor, and that is the figure that decides
  whether this signal is usable as controller feedback at all.

What none of this can catch: a position reading that is itself stale or wrong.
Rate and reference are differenced from the same registers, so they would be
wrong together. Only the travel-time comparison against the specification sees
across that, and only loosely.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import sys
import termios
import time
from typing import List, NamedTuple, Optional, Sequence, Tuple

import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from inspire_hand_msgs.srv import SetSpeed

from . import command_overlays
from . import kinematics as kin
from .compliance_check import Block
from .protocol import CHANNEL_IDS, DOF_ORDER

#: How many samples of history the live view keeps, for its own reference rate,
#: its interval statistics and the one-DOF trace. One column of that trace is
#: one sample, so this has to cover the widest terminal anyone will use it in
#: with seconds to spare: 1000 is 20 s at 50 Hz.
HISTORY = 1000

#: Fraction of the peak rate that still counts as "moving" when picking the
#: window to analyse. Low enough to keep the whole ramp, high enough to leave
#: out the samples either side where the finger is sitting still.
MOTION_FRACTION = 0.05


class Sample(NamedTuple):
    """One ``~/joint_states`` message, reduced to the six driven DOF."""

    stamp: float
    """Header stamp, seconds. The driver divides by this clock, so the rate it
    reports can only be checked against intervals measured on it."""

    wall: float
    """When this process saw the message. Only for the live view's own Hz."""

    position: Tuple[float, ...]
    velocity: Tuple[float, ...]
    has_velocity: bool


# -- analysis, all pure ------------------------------------------------------
def central_difference(times: Sequence[float], values: Sequence[float]) -> List[float]:
    """Differentiate a sampled signal, centred where there is room.

    The reference the driver is judged against. Centred because it has no
    lag to subtract out and half the quantisation noise of a one-sided
    difference; the two ends fall back to one-sided for want of a neighbour.
    """
    count = len(values)
    if count < 2:
        return [0.0] * count
    rates = []
    for index in range(count):
        low = max(0, index - 1)
        high = min(count - 1, index + 1)
        span = times[high] - times[low]
        rates.append(0.0 if span <= 0.0 else (values[high] - values[low]) / span)
    return rates


def trapezoid(times: Sequence[float], values: Sequence[float]) -> float:
    """Integrate a sampled signal. Trapezoid, not rectangles: a rate ramping
    linearly between two samples is exactly what the hand does, and summing
    rectangles would bias every leg by half a sample of its own slope."""
    total = 0.0
    for index in range(1, len(values)):
        total += (
            0.5
            * (values[index] + values[index - 1])
            * (times[index] - times[index - 1])
        )
    return total


def rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Pearson's r, or 0.0 if either side never varies.

    A flat signal has no correlation with anything, and saying so beats
    dividing by its zero spread.
    """
    count = min(len(first), len(second))
    if count < 2:
        return 0.0
    mean_first = sum(first[:count]) / count
    mean_second = sum(second[:count]) / count
    a = [first[i] - mean_first for i in range(count)]
    b = [second[i] - mean_second for i in range(count)]
    spread = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if spread <= 0.0:
        return 0.0
    return sum(a[i] * b[i] for i in range(count)) / spread


def best_lag_samples(
    reference: Sequence[float], measured: Sequence[float], max_shift: int = 10
) -> Optional[float]:
    """How many samples ``measured`` trails ``reference`` by, fractionally.

    Found by sliding one against the other for the smallest RMS difference,
    then fitting a parabola through the best integer shift and its neighbours
    -- the true lag is not a whole number of samples, and rounding it to one
    would report a 10 Hz pole at 50 Hz as either 0 or 20 ms.

    Negative shifts are searched too. A difference of past positions cannot
    lead the signal it differences, so a negative answer is evidence that
    something else is wrong, and hiding it would be the wrong kindness.
    """
    count = min(len(reference), len(measured))
    if count < 8:
        return None
    errors = {}
    for shift in range(-max_shift, max_shift + 1):
        pairs = [
            (reference[i], measured[i + shift])
            for i in range(count)
            if 0 <= i + shift < count
        ]
        if len(pairs) < 4:
            continue
        errors[shift] = rms([m - r for r, m in pairs])
    if not errors:
        return None
    best = min(errors, key=lambda shift: errors[shift])
    low, high = errors.get(best - 1), errors.get(best + 1)
    if low is None or high is None:
        # The best shift is at the edge of the search, so the minimum is
        # somewhere outside it and this is not a measurement of anything. It
        # happens when the trace has been clipped flat and there is no feature
        # left to align; reporting the edge as the answer would dress up a
        # failed search as a 200 ms lead.
        return None
    denominator = low - 2.0 * errors[best] + high
    if denominator <= 0.0:
        return float(best)
    return best + 0.5 * (low - high) / denominator


def shifted_rms(
    reference: Sequence[float], measured: Sequence[float], shift: float
) -> float:
    """RMS difference with ``measured`` slid back by ``shift`` samples.

    The raw difference between the two series is mostly the lag: during a ramp
    of 11 rad/s^2, 23 ms of delay is a quarter of a rad/s of error all by
    itself, which says nothing about the reading's noise. Sliding the series
    by the lag that was measured separates the two, so the residual is what is
    left after a consumer compensates for a delay it knows about.

    Interpolated rather than rounded to whole samples, because the lag is not
    a whole number of them and rounding would leave a sawtooth of its own.
    """
    errors = []
    for index in range(len(reference)):
        position = index + shift
        low = int(math.floor(position))
        high = low + 1
        if low < 0 or high >= len(measured):
            continue
        fraction = position - low
        value = measured[low] + fraction * (measured[high] - measured[low])
        errors.append(value - reference[index])
    return rms(errors)


def motion_window(
    rates: Sequence[float], fraction: float = MOTION_FRACTION
) -> Optional[Tuple[int, int]]:
    """The span of samples where the DOF was actually travelling.

    Everything outside it is the finger sitting at one end of the move, and
    averaging those in would dilute every figure in the report by however long
    the tool happened to wait. The ends of the span are kept, not trimmed: the
    acceleration is part of the signal being measured.
    """
    if not rates:
        return None
    peak = max(abs(rate) for rate in rates)
    if peak <= 0.0:
        return None
    threshold = fraction * peak
    moving = [i for i, rate in enumerate(rates) if abs(rate) >= threshold]
    return (moving[0], moving[-1])


def interval_stats(times: Sequence[float]) -> Tuple[float, float, float, float]:
    """Mean, standard deviation, min and max of the sample interval, seconds.

    The interval is the denominator of every rate here, so its spread is a
    multiplicative error on each single-sample reading, not a detail.
    """
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    if not gaps:
        return (0.0, 0.0, 0.0, 0.0)
    mean = sum(gaps) / len(gaps)
    variance = sum((gap - mean) ** 2 for gap in gaps) / len(gaps)
    return (mean, math.sqrt(variance), min(gaps), max(gaps))


def count_resolution(index: int) -> float:
    """Radians in one ANGLE register count, for one DOF.

    The ANGLE registers are 0..1000 across the whole range, so this is the
    smallest position change the hand can report -- and divided by the sample
    interval, the smallest non-zero rate a difference of them can produce.
    """
    dof = kin.DOFS[index]
    return (dof.upper - dof.lower) / 1000.0


def signed_bar(
    value: float,
    full: float,
    width: int = 31,
    peak_low: Optional[float] = None,
    peak_high: Optional[float] = None,
) -> str:
    """A bar that grows either side of a zero line, for a signed reading.

    Not :func:`~inspire_hand_driver.compliance_check.bar` with an offset: a
    velocity's sign is the most important thing about it, and a bar whose fill
    starts at the left edge makes "slowly opening" and "slowly closing" look
    alike. ``|`` marks the fastest seen each way, as in the force bars; the
    zero line is ``:``.
    """
    half = (width - 1) // 2

    def cells_from_centre(magnitude: float) -> int:
        if full <= 0.0:
            return 0
        return max(0, min(half, int(round(half * abs(magnitude) / full))))

    cells = ["."] * width
    centre = half
    cells[centre] = ":"
    filled = cells_from_centre(value)
    for offset in range(1, filled + 1):
        cells[centre + offset if value >= 0 else centre - offset] = "#"
    for peak, sign in ((peak_low, -1), (peak_high, 1)):
        if peak is None or peak == 0.0:
            continue
        mark = cells_from_centre(peak)
        if mark == 0:
            continue
        position = centre + sign * mark
        # Only where the fill does not already reach, so the mark stays a
        # high-water mark rather than a bite out of the current reading.
        if cells[position] == ".":
            cells[position] = "|"
    return "".join(cells)


# -- commanding a pose from the keyboard -------------------------------------
#: Nudge sizes, in open ratio. 0.01 is ten ANGLE counts, which is about the
#: smallest move worth asking for: the registers hold 1000 counts of travel and
#: the hand's own tracking error is a few of them.
STEPS: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.25)

KEY_HELP = (
    "1-6 pick DOF  a all  +/- nudge  o open  c close  h half  [ ] step  "
    "t compliance  r reset peaks  q quit"
)


class Keys:
    """Single keypresses, without waiting for a newline and without blocking.

    ``cbreak`` rather than raw mode, deliberately: it leaves output
    post-processing alone, so the display's own newlines still work, and it
    leaves ``ISIG`` alone, so Ctrl-C still raises ``KeyboardInterrupt`` through
    the same path as every other exit.

    On top of cbreak, ``VMIN``/``VTIME`` are both zeroed, which is what makes a
    read return immediately with nothing when nothing has been typed. Gating a
    blocking read behind ``select`` is not the same thing and does not work
    here: a pty can report itself readable while a ``VMIN``-1 read on it still
    blocks, and when it does, the display stops dead on the first keystroke
    poll. That is how this first went wrong.

    A terminal left in cbreak echoes nothing, so restoring it is not optional
    -- hence the context manager, and the restore in the caller's ``finally``
    as well.
    """

    def __init__(self) -> None:
        self.tty = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.tty else None
        self._saved = None
        self._saved_sigttin = None

    def _write_mode(self, mode) -> bool:
        """Install terminal settings, without being suspended for doing it.

        A process that is not in the terminal's foreground group gets SIGTTOU
        for changing its settings, and SIGTTOU's default action **stops the
        process**. Which is how a display that works perfectly in a terminal
        freezes solid, drawing nothing, the moment it is run from a script or a
        pipeline -- not an error, not a message, just a stopped process until
        something kills it. Ignoring the signal across the call makes the
        change go through instead.

        ``TCSANOW``, not ``TCSAFLUSH`` or ``TCSADRAIN``: those wait on the
        terminal, and there is nothing here worth waiting for.
        """
        previous = None
        try:
            previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        except ValueError:
            # Not the main thread, so the handler cannot be touched. The call
            # below is still worth trying.
            pass
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, mode)
            return True
        except termios.error:
            return False
        finally:
            if previous is not None:
                signal.signal(signal.SIGTTOU, previous)

    def __enter__(self) -> "Keys":
        if not self.tty:
            return self
        saved = termios.tcgetattr(self._fd)
        mode = list(saved)
        # cbreak by hand rather than through ``tty.cfmakecbreak``, which only
        # exists from Python 3.12: clear line buffering and echo, and leave
        # everything else -- output translation, and ISIG so Ctrl-C still
        # raises KeyboardInterrupt through the ordinary exit path.
        mode[3] &= ~(termios.ECHO | termios.ICANON)
        # Both zeroed so a read returns immediately with whatever is there,
        # including nothing at all. The default VMIN of 1 would block.
        mode[6] = list(mode[6])
        mode[6][termios.VMIN] = 0
        mode[6][termios.VTIME] = 0
        # SIGTTOU's sibling, for the reads rather than the setup: a background
        # process *reading* its controlling terminal is stopped just as
        # silently. Ignored, so the read fails with an error that ``read``
        # turns into a watch-only run instead.
        try:
            self._saved_sigttin = signal.signal(signal.SIGTTIN, signal.SIG_IGN)
        except ValueError:
            pass
        if self._write_mode(mode):
            self._saved = saved
        else:
            # The terminal refused the change, so there is no point reading
            # single keys off it; carry on as a watch-only run.
            self.tty = False
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def restore(self) -> None:
        """Put the terminal back. Must not raise: this runs from a ``finally``."""
        if self._saved_sigttin is not None:
            try:
                signal.signal(signal.SIGTTIN, self._saved_sigttin)
            except ValueError:
                pass
            self._saved_sigttin = None
        if self._saved is None:
            return
        saved, self._saved = self._saved, None
        self._write_mode(saved)

    def read(self) -> str:
        """Everything typed since the last call, or ``""``."""
        if not self.tty:
            return ""
        typed = ""
        while True:
            try:
                chunk = os.read(self._fd, 16)
            except OSError:
                # The terminal went away -- the other end of a pty closed, say.
                # Stop asking rather than raising into the display loop; the
                # run carries on as a watch-only one.
                self.tty = False
                break
            if not chunk:
                # Nothing pending. With VMIN 0 this is the ordinary answer, not
                # an end of file.
                break
            typed += chunk.decode("utf-8", "replace")
        return typed


class Pose:
    """What each DOF has been asked to hold, and the keys that change it.

    Kept apart from the terminal and from ROS so the bindings can be tested by
    feeding them a string. ``target`` holds **commands**, not physical
    positions: channel 6 passes through the thumb-abduction overlay, so a
    command of 0.0 is a physical 0.25, and conflating the two would report a
    250-count error on a thumb that is exactly where it was told to be.
    """

    def __init__(self, step: float = 0.05, selected: Optional[int] = None) -> None:
        self.target: List[Optional[float]] = [None] * len(kin.DOFS)
        #: ``None`` means every DOF at once.
        self.selected = selected
        self.step = step
        self.message = ""
        self.quit = False
        self.reset_peaks = False
        self.toggle_compliance = False

    def indices(self) -> List[int]:
        if self.selected is None:
            return list(range(len(kin.DOFS)))
        return [self.selected]

    def label(self) -> str:
        if self.selected is None:
            return "all"
        return f"{DOF_ORDER[self.selected]} ({CHANNEL_IDS[self.selected]})"

    def expected(self, index: int) -> Optional[float]:
        """The physical open ratio the command for this DOF should produce."""
        if self.target[index] is None:
            return None
        return command_overlays.apply_open_ratio_overlay(index, self.target[index])

    def apply(self, typed: str, measured: Sequence[float]) -> List[int]:
        """Act on keypresses; return the DOF whose target moved.

        ``measured`` seeds a nudge on a DOF this tool has not commanded yet, so
        the first ``+`` moves relative to where the finger actually is rather
        than jumping from an assumed pose. It is a physical ratio, so the
        overlay comes off it before it is kept as a command.
        """
        changed: List[int] = []

        def nudge(delta: float) -> None:
            for index in self.indices():
                current = self.target[index]
                if current is None:
                    physical = measured[index] if index < len(measured) else 1.0
                    current = command_overlays.invert_open_ratio_overlay(
                        index, physical
                    )
                self.target[index] = max(0.0, min(1.0, current + delta))
                changed.append(index)

        def hold(value: float) -> None:
            for index in self.indices():
                self.target[index] = value
                changed.append(index)

        for key in typed:
            if key in CHANNEL_IDS:
                self.selected = CHANNEL_IDS.index(key)
                self.message = f"{self.label()} selected"
            elif key == "a":
                self.selected = None
                self.message = "all six selected"
            elif key in "+=":
                nudge(self.step)
                self.message = f"{self.label()} +{self.step:.2f}"
            elif key in "-_":
                nudge(-self.step)
                self.message = f"{self.label()} -{self.step:.2f}"
            elif key == "o":
                hold(1.0)
                self.message = f"{self.label()} to open"
            elif key == "c":
                hold(0.0)
                self.message = f"{self.label()} to closed"
            elif key == "h":
                hold(0.5)
                self.message = f"{self.label()} to half"
            elif key in "[]":
                order = list(STEPS)
                at = min(
                    range(len(order)), key=lambda i: abs(order[i] - self.step)
                )
                at = max(0, at - 1) if key == "[" else min(len(order) - 1, at + 1)
                self.step = order[at]
                self.message = f"step {self.step:.2f}"
            elif key == "r":
                self.reset_peaks = True
                self.message = "peaks reset"
            elif key == "t":
                self.toggle_compliance = True
            elif key in "qQ":
                self.quit = True
        # Deduplicated, in DOF order: one command per DOF per frame however
        # many keys arrived, so holding a key down cannot queue up a burst of
        # writes the RS485 bus has to drain.
        return sorted(set(changed))


def parse_pose(text: str) -> dict:
    """``"0.3"`` for every DOF, or ``"4:0.3,6:0.8"`` per channel.

    Commands, in the same convention as ``~/command``: 1.0 open, 0.0 closed.
    """
    out = {}
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" in piece:
            channel, _, value = piece.partition(":")
            channel = channel.strip()
            if channel not in CHANNEL_IDS:
                raise ValueError(f"{channel!r} is not a channel; expected 1..6")
            indices = [CHANNEL_IDS.index(channel)]
        else:
            value = piece
            indices = list(range(len(kin.DOFS)))
        ratio = float(value)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"open ratio {ratio} is outside 0..1")
        for index in indices:
            out[index] = ratio
    if not out:
        raise ValueError("no pose given")
    return out


# -- the node ----------------------------------------------------------------
class VelocityCheck(Node):
    """Reads ``~/joint_states``, and can drive one DOF through a known move."""

    def __init__(self, target: str) -> None:
        super().__init__("inspire_hand_velocity_check")
        self.history: List[Sample] = []
        self.recorded: List[Sample] = []
        self.recording = False
        #: True once a message has arrived with no velocity field at all,
        #: which is what ``publish_velocity:=false`` looks like from here.
        self.velocity_missing = False
        self._columns: Optional[List[int]] = None
        self._target = target
        #: The rate each DOF's velocity is clamped to, as the driver reports
        #: it. Read rather than inferred from ``startup_speed``: ``~/set_speed``
        #: moves it, and a clamp mistaken for noise sends someone hunting the
        #: wrong bug. None until the first diagnostics message lands.
        self.ceiling: Optional[List[float]] = None
        #: Physical open ratios as the hand reports them, from ``~/state``.
        #: Read from there rather than converted back out of the radians in
        #: ``joint_states``, because this is the register value itself: one
        #: ANGLE count is 1/1000 of it, and the point of showing it is to see
        #: the counts the velocity is differenced from.
        self.ratios: Optional[List[float]] = None
        self.create_subscription(
            JointState, f"{target}/joint_states", self._on_joint_states, 50
        )
        self.create_subscription(JointState, f"{target}/state", self._on_state, 20)
        self.create_subscription(
            DiagnosticArray, f"{target}/diagnostics", self._on_diagnostics, 10
        )
        self._command = self.create_publisher(JointState, f"{target}/command", 10)
        self._compliance = self.create_client(SetBool, f"{target}/set_compliance")
        self._speed = self.create_client(SetSpeed, f"{target}/set_speed")
        self._set_params = self.create_client(SetParameters, f"{target}/set_parameters")
        self._get_params = self.create_client(GetParameters, f"{target}/get_parameters")

    def _resolve_columns(self, names: Sequence[str]) -> Optional[List[int]]:
        """Which columns of the message are the six driven joints.

        By name rather than by position, and by suffix because
        ``joint_prefix`` is a driver parameter: a two-hand rig publishes
        ``left_index_proximal_joint``, and indexing blind would quietly
        measure whichever joint happened to be in slot 3.
        """
        columns = []
        for joint in kin.DRIVEN_JOINTS:
            matches = [i for i, name in enumerate(names) if name.endswith(joint)]
            if len(matches) != 1:
                return None
            columns.append(matches[0])
        return columns

    def _on_joint_states(self, msg: JointState) -> None:
        if self._columns is None:
            self._columns = self._resolve_columns(list(msg.name))
            if self._columns is None:
                return
        if not msg.velocity:
            self.velocity_missing = True
        sample = Sample(
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9,
            wall=time.monotonic(),
            position=tuple(msg.position[c] for c in self._columns),
            velocity=tuple(
                msg.velocity[c] if msg.velocity else 0.0 for c in self._columns
            ),
            has_velocity=bool(msg.velocity),
        )
        self.history.append(sample)
        del self.history[:-HISTORY]
        if self.recording:
            self.recorded.append(sample)

    def _on_state(self, msg: JointState) -> None:
        if len(msg.position) == len(CHANNEL_IDS):
            self.ratios = list(msg.position)

    def _on_diagnostics(self, msg: DiagnosticArray) -> None:
        found = {}
        for status in msg.status:
            channel = status.hardware_id.rsplit("/", 1)[-1]
            if channel not in CHANNEL_IDS:
                continue
            for entry in status.values:
                if entry.key == "speed_ceiling_rad_s":
                    found[CHANNEL_IDS.index(channel)] = float(entry.value)
        if len(found) == len(CHANNEL_IDS):
            self.ceiling = [found[i] for i in range(len(CHANNEL_IDS))]

    # -- driving -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return bool(self.history)

    def wait_for_hand(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            if self.ready:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def command(self, index: int, ratio: float) -> None:
        self._command.publish(
            JointState(name=[CHANNEL_IDS[index]], position=[float(ratio)])
        )

    def set_compliance(self, enable: bool) -> str:
        if not self._compliance.wait_for_service(timeout_sec=5.0):
            return "no ~/set_compliance service"
        future = self._compliance.call_async(SetBool.Request(data=enable))
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            return "no answer from ~/set_compliance"
        return future.result().message

    def set_speed(self, counts: int) -> str:
        """Write one SPEED_SET value to all six DOF.

        All six, named explicitly, because the driver rewrites any DOF a call
        leaves out from its own cache -- defaulting to mid-scale when it has
        never written one. Naming a single channel would therefore move the
        other five to 500 as a side effect, which is the sort of surprise a
        measurement tool must not spring on a rig.
        """
        if not self._speed.wait_for_service(timeout_sec=5.0):
            return "no ~/set_speed service"
        request = SetSpeed.Request(
            name=list(CHANNEL_IDS), speed=[int(counts)] * len(CHANNEL_IDS)
        )
        future = self._speed.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            return "no answer from ~/set_speed"
        return result.message or ("accepted" if result.accepted else "refused")

    def set_parameters(self, values: dict) -> str:
        """Set driver parameters, returning why any were refused."""
        if not values:
            return ""
        if not self._set_params.wait_for_service(timeout_sec=5.0):
            return "no parameter service"
        request = SetParameters.Request(
            parameters=[
                Parameter(name, value=value).to_parameter_msg()
                for name, value in values.items()
            ]
        )
        future = self._set_params.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            return "no answer from the parameter service"
        return "; ".join(r.reason for r in result.results if not r.successful)

    def get_parameters_from_driver(self, names: Sequence[str]) -> dict:
        """Read driver parameters, so the report can state what it ran against.

        Best-effort: a missing answer is reported as a missing answer rather
        than defaulted, because every one of these changes what the numbers
        below mean and guessing one would make the report lie.
        """
        if not self._get_params.wait_for_service(timeout_sec=5.0):
            return {}
        future = self._get_params.call_async(
            GetParameters.Request(names=list(names))
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or len(result.values) != len(names):
            return {}
        out = {}
        for name, value in zip(names, result.values):
            # PARAMETER_NOT_SET is type 0, and reads as "the driver does not
            # have this parameter" -- an older driver, with no velocity code.
            if value.type == 1:
                out[name] = value.bool_value
            elif value.type == 2:
                out[name] = value.integer_value
            elif value.type == 3:
                out[name] = value.double_value
        return out

    def start_recording(self) -> None:
        self.recorded = []
        self.recording = True

    def stop_recording(self) -> List[Sample]:
        self.recording = False
        return list(self.recorded)

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def spin_until_still(
        self,
        index: int,
        timeout: float,
        still_for: float = 0.5,
        epsilon: float = 1e-4,
    ) -> bool:
        """Spin until one DOF stops moving. True if it stopped, False on timeout.

        Watches the positions, not the velocities: the signal under test must
        not decide when the test is over.
        """
        deadline = time.monotonic() + timeout
        last = self.history[-1].position[index]
        still_since = time.monotonic()
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.monotonic()
            current = self.history[-1].position[index]
            if abs(current - last) > epsilon:
                still_since = now
            last = current
            if now - still_since >= still_for:
                return True
        return False


# -- the measurement ---------------------------------------------------------
def analyse_leg(
    index: int, samples: Sequence[Sample], ceiling: Optional[float] = None
) -> Optional[dict]:
    """Reduce one recorded move of one DOF to the figures worth printing.

    Returns ``None`` if the DOF never moved, which is a result in itself and
    not an error: it means the command did not take, or the finger was held.
    """
    if len(samples) < 8:
        return None
    times = [s.stamp for s in samples]
    positions = [s.position[index] for s in samples]
    reference_all = central_difference(times, positions)
    window = motion_window(reference_all)
    if window is None:
        return None
    first, last = window
    # One sample of margin either side where there is room, so the ramp in and
    # out of the move is inside the window rather than clipped by the
    # threshold that found it.
    first = max(0, first - 1)
    last = min(len(samples) - 1, last + 1)
    times = times[first : last + 1]
    positions = positions[first : last + 1]
    reference = reference_all[first : last + 1]
    driver = [s.velocity[index] for s in samples[first : last + 1]]
    if len(times) < 4:
        return None

    mean_dt, jitter, min_dt, max_dt = interval_stats(times)
    duration = times[-1] - times[0]
    travel = positions[-1] - positions[0]
    integral = trapezoid(times, driver)
    lag = best_lag_samples(reference, driver)
    direction = 1.0 if travel >= 0.0 else -1.0
    clamped = 0
    if ceiling is not None and ceiling > 0.0:
        clamped = sum(
            1
            for r, d in zip(reference, driver)
            if abs(d) >= ceiling * 0.999 and abs(r) > ceiling
        )
    return {
        "samples": len(times),
        "duration": duration,
        "travel": travel,
        "mean_dt": mean_dt,
        "jitter": jitter,
        "min_dt": min_dt,
        "max_dt": max_dt,
        # The smallest non-zero rate a difference of ANGLE counts can produce.
        # Every figure above it is a multiple of it, which is the real limit on
        # a single sample and the reason the driver filters at all.
        "resolution": count_resolution(index) / mean_dt if mean_dt > 0 else 0.0,
        "reference_peak": max(reference, key=abs) if reference else 0.0,
        "driver_peak": max(driver, key=abs) if driver else 0.0,
        # Mean over the window, which for a move that starts and ends at rest
        # is travel/duration whatever the profile in between.
        "reference_mean": sum(reference) / len(reference),
        "driver_mean": sum(driver) / len(driver),
        "truth_mean": travel / duration if duration > 0 else 0.0,
        "rms_error": rms([d - r for r, d in zip(reference, driver)]),
        # The same residual with the measured lag taken out, which is the one
        # that reflects how noisy the reading is rather than how late.
        "rms_aligned": (
            None if lag is None else shifted_rms(reference, driver, lag)
        ),
        "correlation": correlation(reference, driver),
        "lag_s": None if lag is None else lag * mean_dt,
        "integral": integral,
        "integral_error": integral - travel,
        "clamped": clamped,
        "clamped_fraction": clamped / len(driver),
        # A rate that is exactly zero while the finger is travelling is the
        # quantisation gap the driver's docstring describes: a count that has
        # not flipped yet, not a finger that stopped.
        "zero_fraction": sum(1 for d in driver if d == 0.0) / len(driver),
        "direction": direction,
    }


def print_leg(label: str, index: int, figures: Optional[dict], speed: Optional[int]) -> None:
    """Print one leg's figures, with each one next to what it is judged against."""
    print(f"\n  {label}: {kin.DOFS[index].joint}")
    if figures is None:
        print("    never moved. The command did not take, or the finger was held.")
        return
    f = figures
    full_scale = kin.speed_counts_to_rad_per_s(index, 1000)
    print(
        f"    {f['samples']} samples over {f['duration']:.2f} s, "
        f"travelled {f['travel']:+.3f} rad"
    )
    print(
        f"    interval        {f['mean_dt'] * 1e3:6.1f} ms +-{f['jitter'] * 1e3:.1f} "
        f"(min {f['min_dt'] * 1e3:.1f}, max {f['max_dt'] * 1e3:.1f})"
    )
    print(
        f"    resolution      {f['resolution']:6.3f} rad/s -- one ANGLE count per sample"
    )
    print(
        f"    peak rate       {f['driver_peak']:+6.3f} reported, "
        f"{f['reference_peak']:+6.3f} from the positions"
    )
    print(
        f"    mean rate       {f['driver_mean']:+6.3f} reported, "
        f"{f['truth_mean']:+6.3f} travel over time"
    )
    peak = max(1e-9, abs(f["reference_peak"]))
    print(
        f"    vs the positions  rms {f['rms_error']:.3f} rad/s"
        f" ({100.0 * f['rms_error'] / peak:.1f}% of peak)"
        + (
            ""
            if f["rms_aligned"] is None
            else f", {f['rms_aligned']:.3f} ({100.0 * f['rms_aligned'] / peak:.1f}%)"
            " once the lag is taken out"
        )
    )
    # Reported, not judged. A move that holds a constant rate has almost no
    # variance for a correlation to find, so on the plateau r is noise against
    # noise and reads low however faithful the signal is.
    print(
        f"    correlation     r = {f['correlation']:.3f} (only meaningful where "
        f"the rate varies)"
    )
    if f["lag_s"] is not None:
        print(
            f"    lag             {f['lag_s'] * 1e3:6.1f} ms behind a centred difference"
        )
    else:
        print(
            "    lag             not measurable -- no alignment inside the search "
            "fits, which is what a trace clipped flat looks like"
        )
    print(
        f"    integral        {f['integral']:+.3f} rad against {f['travel']:+.3f} "
        f"travelled, {100.0 * f['integral_error'] / max(1e-9, abs(f['travel'])):+.1f}%"
    )
    if f["zero_fraction"] > 0.0:
        print(
            f"    exact zeros     {100.0 * f['zero_fraction']:.0f}% of samples while moving"
        )
    if f["clamped"]:
        print(
            f"    clamped         {f['clamped']} samples held at the speed ceiling"
        )
    # Inspire's bound, and what the measurement implies about the setting in
    # effect. The second one is the useful direction when nothing in the stack
    # wrote SPEED_SET: the hand is running on its flash default, and this is
    # the only way to find out what that is.
    implied = 1000.0 * abs(f["reference_peak"]) / full_scale if full_scale > 0 else 0.0
    if speed:
        bound = kin.speed_counts_to_rad_per_s(index, speed)
        print(
            f"    vs the manual   peak {abs(f['reference_peak']):.3f} against a "
            f"{bound:.3f} rad/s bound at SPEED_SET {speed} "
            f"({100.0 * abs(f['reference_peak']) / max(1e-9, bound):.0f}% of it)"
        )
    else:
        print(
            f"    vs the manual   peak {abs(f['reference_peak']):.3f} rad/s implies "
            f"SPEED_SET near {implied:.0f}, against {full_scale:.3f} at 1000"
        )


def print_standstill(samples: Sequence[Sample], seconds: float) -> List[float]:
    """Print what the velocities read with the hand still, and return the floor.

    The figure that decides whether this signal is usable as feedback: a
    controller differentiating a position it cannot trust at rest will chatter
    about noise, and no amount of filtering downstream recovers information
    the register never had.
    """
    print(f"\n  standstill, {seconds:.1f} s, {len(samples)} samples")
    if not samples:
        print("    nothing arrived.")
        return [0.0] * len(kin.DOFS)
    floors = []
    for index in range(len(kin.DOFS)):
        values = [s.velocity[index] for s in samples]
        worst = max((abs(v) for v in values), default=0.0)
        floors.append(worst)
        print(
            f"    {DOF_ORDER[index]:<14} max |v| {worst:.4f}   "
            f"rms {rms(values):.4f} rad/s"
        )
    return floors


#: How large the lag-compensated residual may be, as a multiple of one ANGLE
#: count per sample. A difference of quantised positions cannot be quieter
#: than its own quantisation, and the centred reference carries some of that
#: noise too, so the floor is around one count and the allowance is two.
MAX_RESIDUAL_COUNTS = 2.0

#: And never tighter than this fraction of the peak, so a slow move -- where
#: one count per sample is most of the signal -- is not failed for arithmetic
#: that is doing as well as the register allows.
MAX_RESIDUAL_FRACTION = 0.10

#: How much of the travel the integral of the reported rate may miss. A filter
#: costs a little at each end of a move and nothing in between, so a few per
#: cent is the pole; ten is something eating motion.
MAX_INTEGRAL_ERROR = 0.05

#: Lag a consumer can absorb. The policy runs at 15 Hz, so a 66 ms budget;
#: half of it leaves room for the rest of the path.
MAX_LAG_S = 0.033


#: Above this share of the window held at the speed ceiling, the clamp is the
#: story and everything else measured is downstream of it.
MAX_CLAMPED_FRACTION = 0.05


def verdict(
    legs: Sequence[Tuple[str, Optional[dict]]],
    floors: Sequence[float],
    mock: bool = False,
) -> int:
    """Say whether the reported rate is good enough to feed a controller."""
    print("\n  verdict")
    problems = []
    for label, figures in legs:
        if figures is None:
            problems.append(f"the {label} leg never moved")
            continue
        if figures["clamped_fraction"] > MAX_CLAMPED_FRACTION:
            # Named on its own, and the rest of this leg's figures left
            # unreported: a residual, a short integral and an unfindable lag
            # are all the same clamp seen from three sides, and listing them
            # as separate faults would send someone looking for three bugs.
            problems.append(
                f"{label}: {100.0 * figures['clamped_fraction']:.0f}% of the window is "
                f"held at the speed ceiling, so the hand moved faster than SPEED_SET "
                f"says it can. Either the setting did not take, or "
                f"FULL_TRAVEL_TIME_S is wrong"
                + (
                    " -- and on the mock transport it is the former: it ignores "
                    "SPEED_SET and slews at full rate"
                    if mock
                    else ""
                )
            )
            continue
        residual = figures["rms_aligned"]
        if residual is not None:
            allowed = max(
                MAX_RESIDUAL_COUNTS * figures["resolution"],
                MAX_RESIDUAL_FRACTION * abs(figures["reference_peak"]),
            )
            if residual > allowed:
                problems.append(
                    f"{label}: {residual:.3f} rad/s of residual against the positions "
                    f"once the lag is out, over the {allowed:.3f} the quantisation "
                    f"explains"
                )
        relative = abs(figures["integral_error"]) / max(1e-9, abs(figures["travel"]))
        if relative > MAX_INTEGRAL_ERROR:
            problems.append(
                f"{label}: the integral misses {100.0 * relative:.1f}% of the travel, "
                f"over {100.0 * MAX_INTEGRAL_ERROR:.0f}%"
            )
        lag = figures["lag_s"]
        if lag is not None and lag > MAX_LAG_S:
            problems.append(
                f"{label}: {lag * 1e3:.0f} ms of lag, over the {MAX_LAG_S * 1e3:.0f} ms budget"
            )
        if lag is not None and lag < -0.5 * figures["mean_dt"]:
            problems.append(
                f"{label}: the reported rate leads the positions by "
                f"{-lag * 1e3:.0f} ms, which a difference of past samples cannot do"
            )
    worst_floor = max(floors) if floors else 0.0
    if worst_floor > 0.0:
        # Two significant figures, not four decimals: a floor of 3e-5 is a
        # real reading and printing it as "0.0000" next to the claim that it
        # is not zero reads as a bug in the tool.
        worst_channel = DOF_ORDER[list(floors).index(worst_floor)]
        print(
            f"    at rest the worst channel ({worst_channel}) reads {worst_floor:.2g} "
            f"rad/s, so treat anything under that as no motion."
        )
    else:
        print("    at rest every channel reads exactly zero.")
    if problems:
        for problem in problems:
            print(f"    FAIL: {problem}")
        return 1
    print("    PASS: the reported rate tracks the positions, keeps the travel, and")
    print("          lags by less than the budget.")
    return 0


def sweep(node: VelocityCheck, args) -> int:
    """Command one DOF across its range and measure the rate that comes back."""
    index = CHANNEL_IDS.index(args.channel)
    params = node.get_parameters_from_driver(
        [
            "publish_velocity",
            "velocity_filter_hz",
            "publish_rate_hz",
            "startup_speed",
            "mock",
        ]
    )
    if node.velocity_missing or params.get("publish_velocity") is False:
        print(
            "\n  The driver is publishing no velocities. Relaunch it with\n"
            "  publish_velocity:=true, or if it has no such parameter it predates\n"
            "  this measurement entirely."
        )
        return 2

    if args.filter is not None:
        refused = node.set_parameters({"velocity_filter_hz": float(args.filter)})
        if refused:
            print(f"\n  the driver refused velocity_filter_hz: {refused}")
            return 2
        params["velocity_filter_hz"] = float(args.filter)

    # Compliant mode would push back against every commanded move, so the
    # sweep insists on a stiff hand rather than quietly measuring a fight.
    node.set_compliance(False)

    speed = args.speed or int(params.get("startup_speed") or 0) or None
    if args.speed:
        answer = node.set_speed(args.speed)
        print(f"\n  speed: {answer}")

    filter_hz = params.get("velocity_filter_hz")
    rate_hz = params.get("publish_rate_hz")
    described_speed = speed if speed else "the hand's own flash default"
    print(
        f"\n  driver: {'%.0f Hz' % rate_hz if rate_hz else 'rate unknown'}, "
        f"filter {('%.1f Hz' % filter_hz) if filter_hz else 'off' if filter_hz == 0 else 'unknown'}"
        f", speed {described_speed}"
    )
    # The driver's own clamp, read off its diagnostics. Only if it is not
    # publishing them does this fall back to working the number out, and if
    # neither is available the report says the clamp could not be checked
    # rather than quietly reporting its effects as noise.
    node.spin_for(0.5)
    if node.ceiling is not None:
        ceiling = node.ceiling[index]
        print(f"  clamp: {ceiling:.3f} rad/s on {kin.DOFS[index].name}, per the driver")
    elif speed:
        ceiling = kin.speed_counts_to_rad_per_s(index, speed)
        print(f"  clamp: {ceiling:.3f} rad/s, worked out from SPEED_SET {speed}")
    else:
        ceiling = None
        print(
            "  clamp: unknown -- the driver publishes no diagnostics and no speed\n"
            "         was set, so a clamped sample cannot be told from a slow one."
        )

    print(f"\n1. Parking {kin.DOFS[index].name} open, then holding still.")
    node.command(index, 1.0)
    node.spin_until_still(index, timeout=6.0)
    node.start_recording()
    node.spin_for(args.still)
    standstill = node.stop_recording()
    floors = print_standstill(standstill, args.still)
    logged = 0
    if args.log:
        rows = log_rows("standstill", standstill, [index])
        write_log(args.log, rows, header=True)
        logged += len(rows)

    print(f"\n2. Closing to {args.close:.2f} and back, measuring both legs.")
    legs = []
    for label, target in (("closing", args.close), ("opening", 1.0)):
        node.start_recording()
        node.command(index, target)
        if not node.spin_until_still(index, timeout=args.timeout):
            print(f"    {label}: it was still moving after {args.timeout:.0f} s")
        samples = node.stop_recording()
        legs.append((label, analyse_leg(index, samples, ceiling)))
        if args.log:
            rows = log_rows(label, samples, [index])
            write_log(args.log, rows, header=False)
            logged += len(rows)

    for label, figures in legs:
        print_leg(label, index, figures, speed)
    if args.log:
        print(f"\n  {logged} rows written to {args.log}")
    return verdict(legs, floors, mock=bool(params.get("mock")))


#: Columns of the CSV ``--log`` writes. ``reference`` is this tool's centred
#: difference of the positions in the same row's series, not anything the hand
#: said, so a plot can show the reading against what it is judged by.
LOG_HEADER = "phase,stamp,channel,joint,count,rad,velocity,reference"


def log_rows(
    phase: str, samples: Sequence[Sample], indices: Sequence[int]
) -> List[str]:
    """One CSV row per sample per DOF, newest last.

    The terminal is the wrong place to look closely -- one column is one
    sample and one row is a tenth of the scale -- so the samples go out whole
    and get plotted properly. ``count`` is the ANGLE register value the rate
    was differenced from, which is the column that shows the quantisation for
    what it is.
    """
    times = [sample.stamp for sample in samples]
    rows = []
    for index in indices:
        reference = central_difference(
            times, [sample.position[index] for sample in samples]
        )
        for place, sample in enumerate(samples):
            rows.append(
                f"{phase},{sample.stamp:.6f},{CHANNEL_IDS[index]},"
                f"{kin.DOFS[index].joint},"
                f"{round(kin.rad_to_open_ratio(index, sample.position[index]) * 1000.0)},"
                f"{sample.position[index]:.6f},{sample.velocity[index]:.6f},"
                f"{reference[place]:.6f}"
            )
    return rows


def write_log(path: str, rows: Sequence[str], header: bool) -> None:
    """Append rows to the log, writing the header only on a fresh file."""
    with open(path, "w" if header else "a") as handle:
        if header:
            handle.write(LOG_HEADER + "\n")
        handle.write("\n".join(rows) + ("\n" if rows else ""))


def trace_size(columns: int, lines: int) -> Tuple[int, int]:
    """Width and height for the rate trace, given the terminal's size.

    Fixed at 70x9 the trace threw away most of a modern window, and one column
    is one sample, so width is resolution in the most direct sense.

    The subtractions are the rest of the frame: 11 columns for the axis label
    and the two frame bars, and 12 lines for the eight rows of the one-DOF
    frame that are not the trace, the two the live view adds, and a line of
    margin -- a frame exactly as tall as the window scrolls its own top away on
    every redraw.
    """
    width = max(20, min(400, int(columns) - 11))
    height = max(5, min(41, int(lines) - 12))
    return width, height | 1  # odd, so the zero line has a row of its own


def velocity_trace(
    driver: Sequence[float],
    reference: Sequence[float],
    scale: float,
    width: int = 70,
    height: int = 9,
) -> List[str]:
    """A little signed plot of one DOF's rate over its recent history.

    Worth the rows when only one finger is on screen: a bar shows the rate now,
    and the two things most worth seeing about a *derived* rate are its shape
    over time -- the lag against the reference, and the quantisation arriving
    as steps rather than a curve. Neither is visible one sample at a time.

    ``#`` is what the driver reports, ``+`` this tool's centred difference of
    the same positions, and ``-`` the zero line. Newest sample on the right.
    """
    rows = max(3, height | 1)
    centre = rows // 2
    # Always the full width, with the history right-aligned inside it, rather
    # than a plot that grows a column per sample: a frame whose width changes
    # every 20 ms is unreadable for the first few seconds of every run.
    columns = max(1, width)
    grid = [[" "] * columns for _ in range(rows)]
    for column in range(columns):
        grid[centre][column] = "-"

    def place(series: Sequence[float], mark: str, over: bool) -> None:
        tail = list(series)[-columns:]
        # Right-aligned, so the newest sample is always at the right edge
        # however little history there is yet.
        offset = columns - len(tail)
        for position, value in enumerate(tail):
            if scale <= 0.0:
                row = centre
            else:
                step = int(round(centre * value / scale))
                row = centre - max(-centre, min(centre, step))
            column = offset + position
            if over or grid[row][column] in (" ", "-"):
                grid[row][column] = mark

    place(reference, "+", over=False)
    place(driver, "#", over=True)

    labels = {0: f"{scale:+5.2f}", centre: f"{0.0:+5.2f}", rows - 1: f"{-scale:+5.2f}"}
    return [
        f"  {labels.get(index, ''):>6} |" + "".join(grid[index]) + "|"
        for index in range(rows)
    ]


#: The position columns' header, built from the same widths the rows use so the
#: two cannot drift apart.
POSE_HEADER = (
    "   " + f"{'ch':>2} " + f"{'name':<15}" + f"{'cmd':>5}" + f"{'pos':>7}"
    + f"{'count':>6}" + f"{'err':>5}"
)


def pose_rows(pose: Pose, ratios: Optional[Sequence[float]]) -> List[str]:
    """The position half of a row, per DOF: what was asked for and what came back.

    ``cmd`` is the command sent, ``pos`` the physical ratio the hand reports,
    ``count`` that same reading as the ANGLE register value it came from, and
    ``err`` the difference in counts against what the command should produce
    -- after the overlay, so channel 6 does not show a standing 250-count
    error for sitting exactly where it was told.
    """
    rows = []
    for index in range(len(kin.DOFS)):
        measured = None if ratios is None else ratios[index]
        expected = pose.expected(index)
        # Rounded to an integer before formatting, not formatted with no
        # decimals: a hair under zero would otherwise print as "-0", which
        # reads as a direction rather than as the zero it is.
        error = (
            f"{round((measured - expected) * 1000.0):+5d}"
            if measured is not None and expected is not None
            else f"{'':>5}"
        )
        rows.append(
            ("> " if pose.selected == index else "  ")
            + f"{CHANNEL_IDS[index]:>2} "
            + f"{DOF_ORDER[index]:<15}"
            + (f"{'--':>5}" if expected is None else f"{pose.target[index]:5.2f}")
            + (f"{'--':>7}" if measured is None else f"{measured:7.3f}")
            + (f"{'--':>6}" if measured is None else f"{round(measured * 1000.0):6d}")
            + error
        )
    return rows


def single_rows(
    index: int,
    pose: Pose,
    ratios: Optional[Sequence[float]],
    history: Sequence[Sample],
    peak_low: float,
    peak_high: float,
    scale: float,
    reference: Sequence[float],
    size: Tuple[int, int] = (70, 9),
) -> List[str]:
    """The whole frame for one DOF: where it is, how fast, and the recent shape."""
    latest = history[-1]
    dof = kin.DOFS[index]
    measured = None if ratios is None else ratios[index]
    expected = pose.expected(index)
    rate = latest.velocity[index]
    reference_now = reference[-1] if len(reference) else 0.0

    position = (
        f"  cmd {'--' if expected is None else format(pose.target[index], '.2f')}"
        f"   pos {'--' if measured is None else format(measured, '.3f')}"
        f"   count {'--' if measured is None else round(measured * 1000.0)}"
        f"   err {'--' if measured is None or expected is None else format(round((measured - expected) * 1000.0), '+d')}"
        f"   rad {latest.position[index]:.3f}"
        f"  of {dof.lower:.2f}..{dof.upper:.2f}"
    )
    mean_dt, jitter, _, max_dt = interval_stats([s.stamp for s in history])
    # Narrower than the trace by the width of its axis label, so the two line
    # up on the left and the bar does not run past the plot on the right.
    bar_width = max(11, min(size[0] - 8, 81)) | 1
    rows = [
        f"  {dof.name} (channel {dof.channel}, {dof.joint})",
        position,
        "",
        f"  speed {rate:+6.2f} rad/s  |"
        + signed_bar(
            rate, scale, width=bar_width, peak_low=peak_low, peak_high=peak_high
        )
        + f"|  peak {peak_low:+.2f} {peak_high:+.2f}",
        f"  ref   {reference_now:+6.2f} rad/s  |"
        + signed_bar(reference_now, scale, width=bar_width)
        + "|  centred, one sample behind",
        "",
    ]
    width, height = size
    rows += velocity_trace(
        [s.velocity[index] for s in history], reference, scale, width, height
    )
    seconds = len(history) * mean_dt
    rows.append(
        f"  the last {seconds:.1f} s, newest at the right."
        f"  # reported, + from the positions, - the zero line"
    )
    rows.append(
        f"  {1.0 / mean_dt if mean_dt > 0 else 0.0:.0f} Hz, dt {mean_dt * 1e3:.1f} "
        f"+-{jitter * 1e3:.1f} ms (max {max_dt * 1e3:.0f})"
        f"   1 count = {count_resolution(index) / mean_dt if mean_dt > 0 else 0.0:.3f} rad/s"
        f"   full scale +-{scale:.2f}"
    )
    return rows


def live(node: VelocityCheck, args) -> int:
    """Position and speed for every DOF, with the keyboard driving the fingers."""
    params = node.get_parameters_from_driver(
        ["publish_velocity", "velocity_filter_hz", "publish_rate_hz"]
    )
    filter_hz = params.get("velocity_filter_hz")
    print(
        f"\n  {args.node} -- speed in rad/s, + closing, - opening."
        + (f"  filter {filter_hz:.1f} Hz." if filter_hz else "")
    )
    if node.velocity_missing or params.get("publish_velocity") is False:
        print(
            "\n  !! The driver is publishing no velocities at all. Relaunch it with\n"
            "     publish_velocity:=true."
        )
        return 2
    print(
        "  cmd is what was commanded, pos/count what the hand reports back (count is\n"
        "  the ANGLE register, 0..1000, the value the speed is differenced from), err\n"
        "  the counts between them. ref is this tool's own centred difference of the\n"
        "  same positions: the number the reading is judged against."
    )

    only = CHANNEL_IDS.index(args.only) if args.only else None
    pose = Pose(
        step=args.step,
        selected=only if only is not None else CHANNEL_IDS.index(args.channel),
    )
    compliant = args.compliant
    if compliant:
        print(f"\n  {node.set_compliance(True)}")
        print(
            "  Compliant: a fingertip push opens that finger and it returns on its\n"
            "  own, so a commanded position is a rest position, not a hard target."
        )
    else:
        print("\n  Stiff: a commanded position is held against whatever pushes back.")

    if args.pose:
        for index, ratio in parse_pose(args.pose).items():
            pose.target[index] = ratio
            node.command(index, ratio)
        print(f"  commanded {args.pose}")

    if args.log:
        # Recorded from the start, so the log covers the whole run and not
        # just whatever is still inside the display's rolling history.
        node.start_recording()
        print(f"  logging every sample to {args.log}")
    keys = Keys()
    if not keys.tty:
        print(
            "\n  Not a terminal, so no keys: this run only watches. Give --pose to\n"
            "  command a position non-interactively."
        )
    print("")

    peak_low = [0.0] * len(kin.DOFS)
    peak_high = [0.0] * len(kin.DOFS)
    block = Block()
    rows: List[str] = []
    deadline = None if args.duration <= 0 else time.monotonic() + args.duration
    try:
        with keys:
            while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
                rclpy.spin_once(node, timeout_sec=0.02)
                if not node.history:
                    continue

                for index in pose.apply(
                    keys.read(), node.ratios or [1.0] * len(kin.DOFS)
                ):
                    node.command(index, pose.target[index])
                if pose.quit:
                    break
                if pose.reset_peaks:
                    peak_low = [0.0] * len(kin.DOFS)
                    peak_high = [0.0] * len(kin.DOFS)
                    pose.reset_peaks = False
                if pose.toggle_compliance:
                    pose.toggle_compliance = False
                    compliant = not compliant
                    # Blocking, but only for as long as one service call takes,
                    # and only on a keypress. Engaging also re-tares, which is
                    # the whole reason to be able to do it from here.
                    pose.message = f"compliance {'on' if compliant else 'off'}: " + (
                        node.set_compliance(compliant)
                    )

                latest = node.history[-1]
                recent = node.history[-9:]
                times = [sample.stamp for sample in recent]
                # Not smoothed for display. The driver's own filter is the
                # thing under test, and a second one here would hide what it
                # leaves.
                reference = [
                    central_difference(times, [s.position[i] for s in recent])[-2]
                    if len(recent) >= 3
                    else 0.0
                    for i in range(len(kin.DOFS))
                ]
                # The whole history differenced, for the one-DOF trace. Only
                # the selected DOF, because this is per-sample work at the
                # display's rate and five more of them would buy nothing.
                trace_reference: List[float] = []
                if only is not None:
                    trace_reference = central_difference(
                        [sample.stamp for sample in node.history],
                        [sample.position[only] for sample in node.history],
                    )
                for i, value in enumerate(latest.velocity):
                    peak_low[i] = min(peak_low[i], value)
                    peak_high[i] = max(peak_high[i], value)
                # Auto-ranged, so a finger nudged by hand is not an invisible
                # sliver on a scale picked for a full-speed close. On one DOF
                # the range follows that DOF alone, so another finger moving
                # cannot shrink the one being watched.
                if only is not None:
                    span = max(peak_high[only], -peak_low[only])
                else:
                    span = max(max(peak_high), -min(peak_low))
                scale = args.scale or max(0.25, span, 1e-9)
                if only is not None:
                    # Measured every frame rather than once: a window resized
                    # mid-run should change the plot, not corrupt it.
                    window = shutil.get_terminal_size(fallback=(100, 30))
                    rows = single_rows(
                        only,
                        pose,
                        node.ratios,
                        node.history,
                        peak_low[only],
                        peak_high[only],
                        scale,
                        trace_reference,
                        trace_size(window.columns, window.lines),
                    )
                else:
                    positions = pose_rows(pose, node.ratios)
                    rows = [
                        POSE_HEADER + f"{'speed':>8}  {'':<21}{'peak':>12}"
                    ]
                    rows += [
                        positions[i]
                        + f"{latest.velocity[i]:+8.2f}  "
                        + "|"
                        + signed_bar(
                            latest.velocity[i],
                            scale,
                            width=21,
                            peak_low=peak_low[i],
                            peak_high=peak_high[i],
                        )
                        + "|"
                        + f"{peak_low[i]:+6.2f}{peak_high[i]:+6.2f}"
                        + f"  ref {reference[i]:+6.2f}"
                        for i in range(len(kin.DOFS))
                    ]
                    mean_dt, jitter, _, max_dt = interval_stats(
                        [sample.stamp for sample in node.history]
                    )
                    # One ANGLE count per actual sample interval: the smallest
                    # non-zero rate that exists to report. Measured rather than
                    # assumed from the publish rate, because the interval is
                    # right here and a hand that is dropping reads has a wider
                    # one than it was configured for.
                    resolutions = [
                        count_resolution(i) / mean_dt if mean_dt > 0 else 0.0
                        for i in range(len(kin.DOFS))
                    ]
                    rows.append(
                        f"  full scale +-{scale:.2f}"
                        f"   {1.0 / mean_dt if mean_dt > 0 else 0.0:.0f} Hz, dt "
                        f"{mean_dt * 1e3:.1f} +-{jitter * 1e3:.1f} ms "
                        f"(max {max_dt * 1e3:.0f})"
                        f"   1 count = {min(resolutions):.2f}..{max(resolutions):.2f} rad/s"
                    )
                if keys.tty:
                    rows.append(
                        f"  {pose.label()}, step {pose.step:.2f}, "
                        f"{'compliant' if compliant else 'stiff'}"
                        + (f"  --  {pose.message}" if pose.message else "")
                    )
                    rows.append(f"  {KEY_HELP}")
                block.update(rows)
            if rows:
                block.update(rows, force=True)
    finally:
        keys.restore()
        block.finish()
        if args.log:
            recorded = node.stop_recording()
            rows = log_rows(
                "live",
                recorded,
                [only] if only is not None else list(range(len(kin.DOFS))),
            )
            write_log(args.log, rows, header=True)
            print(f"\n  {len(rows)} rows written to {args.log}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--node", default="/inspire_hand", help="Driver node to read.")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Command one DOF across its range and measure the reading.",
    )
    parser.add_argument(
        "--channel", default="4", choices=list(CHANNEL_IDS),
        help="Which DOF to sweep, and which starts selected in the live view. "
             "4 is the index finger.",
    )
    parser.add_argument(
        "--close", type=float, default=0.10,
        help="Open ratio to close to. 0.10 stops short of the hard limit, so the "
             "stall guard stays out of the measurement; 0.0 is full travel.",
    )
    parser.add_argument(
        "--speed", type=int, default=None,
        help="Write this SPEED_SET (0..1000) to all six DOF before sweeping, so "
             "the measured peak can be held against the manual's bound. Volatile: "
             "the hand forgets it on power cycle.",
    )
    parser.add_argument(
        "--filter", type=float, default=None,
        help="Set the driver's velocity_filter_hz first. 0 measures the raw "
             "difference, with no pole in the way.",
    )
    parser.add_argument(
        "--still", type=float, default=3.0,
        help="Seconds of standstill to measure the noise floor over.",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="Seconds to allow one leg of the sweep.",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to show the live view for. 0 runs until Ctrl-C.",
    )
    parser.add_argument(
        "--scale", type=float, default=0.0,
        help="Fix the live bars' full scale, in rad/s, instead of auto-ranging.",
    )
    parser.add_argument(
        "--log", default=None, metavar="PATH",
        help="Write every sample to this CSV as well: phase, stamp, channel, "
             "joint, ANGLE count, radians, the reported rate and this tool's "
             "centred difference of the same positions. For looking closer than "
             "a terminal can.",
    )
    parser.add_argument(
        "--only", default=None, choices=list(CHANNEL_IDS),
        help="Show one DOF alone, with a trace of its rate over the recent history "
             "instead of one row among six. 4 is the index finger.",
    )
    parser.add_argument(
        "--pose", default=None,
        help="Command a pose on the way in: '0.3' for every DOF, or '4:0.3,6:0.8' "
             "per channel. Open ratios, 1.0 open and 0.0 closed, same as ~/command. "
             "Works without a terminal, which is how to set a pose from a script.",
    )
    parser.add_argument(
        "--step", type=float, default=0.05,
        help="Starting nudge size for +/- in the live view, in open ratio. "
             "0.01 is ten ANGLE counts.",
    )
    parser.add_argument(
        "--compliant", action="store_true", default=True,
        help="Engage compliant mode for the live view, so fingers can be pushed "
             "by hand. On by default.",
    )
    parser.add_argument(
        "--stiff", dest="compliant", action="store_false",
        help="Leave the hand stiff in the live view.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.pose is not None:
        # Parsed before connecting, so a typo is a message rather than a hand
        # that has already been sent somewhere while the error is printed.
        try:
            parse_pose(args.pose)
        except ValueError as exc:
            print(f"--pose: {exc}")
            return 2

    rclpy.init()
    node = VelocityCheck(args.node.rstrip("/"))
    code = 1
    try:
        if not node.wait_for_hand():
            print(
                f"Nothing on {args.node}/joint_states. Launch the driver first:\n"
                f"  ros2 launch inspire_hand_driver inspire_hand.launch.py "
                f"port:=/dev/ttyUSB0"
            )
            return 2
        code = sweep(node, args) if args.sweep else live(node, args)
    except KeyboardInterrupt:
        print("\n  stopped")
        code = 0
    except ExternalShutdownException:
        print("\n  shut down")
    finally:
        # Same contract as the compliance check: never leave a hand compliant
        # or a terminal without its cursor, whichever way this ended.
        try:
            sys.stdout.write("\x1b[?25h")
            if rclpy.ok():
                node.set_compliance(False)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
