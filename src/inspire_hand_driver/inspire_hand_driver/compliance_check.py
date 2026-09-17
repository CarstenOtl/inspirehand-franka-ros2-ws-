"""Bench check for compliant mode: push a finger, let go, see it come back.

The test this runs is the one worth running by hand, made repeatable and given
a verdict:

1. Measure what the fingertip sensors read with nothing touching them. That is
   the noise floor, and the deadband has to clear it or a grip will open
   itself. Nothing else in this workspace has measured it.
2. Command one finger to a rest position and let it settle.
3. Turn compliant mode on and wait for a push. The finger should give.
4. Let go. The finger should come back to the pose it settled at in step 2.

Passing means both halves happened: it gave by at least ``--min-open``, and it
came back to within ``--tolerance`` of where it started. Either half alone is a
failure and says something different -- a finger that gives and stays open is a
return rate that never ran, one that never gives is a deadband above what your
push produces, or a gain too small to see.

    ros2 run inspire_hand_driver inspire_hand_compliance_check
    ros2 run inspire_hand_driver inspire_hand_compliance_check --characterize
    ros2 run inspire_hand_driver inspire_hand_compliance_check \\
        --channel 3 --rest 0.4 --deadband 40 --gain 1.0

It drives the running driver through its ordinary interface -- ``~/command``,
``~/set_compliance``, and the parameter services -- so it proves the path a
caller would use, not a private one. Compliant mode is turned off and the rest
command re-sent on the way out, including on Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence

import rclpy
from rcl_interfaces.srv import SetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from .protocol import CHANNEL_IDS, DOF_ORDER

#: Long enough to see the sensor's own drift, short enough that nobody skips it.
BASELINE_SEC = 3.0


class ComplianceCheck(Node):
    def __init__(self, target: str) -> None:
        super().__init__("inspire_hand_compliance_check")
        self.state: Optional[List[float]] = None
        #: Empty until the first message lands. Not a list of zeros: a
        #: placeholder that looks like a reading gets sampled as one, and a
        #: baseline that averages in a 0 nobody measured reports a resting
        #: band of hundreds of grams on a pad that is sitting perfectly still.
        self.force: List[float] = []
        #: The driver's own per-channel zero. Read rather than assumed: the law
        #: compares force against this, so a tool comparing against anything
        #: else can disagree with the controller it is testing about whether a
        #: finger is being touched at all.
        self.zero: List[float] = [0.0] * 6
        self.create_subscription(JointState, f"{target}/state", self._on_state, 10)
        self.create_subscription(JointState, f"{target}/grip_force", self._on_force, 10)
        self.create_subscription(
            DiagnosticArray, f"{target}/diagnostics", self._on_diagnostics, 10
        )
        self._command = self.create_publisher(JointState, f"{target}/command", 10)
        self._compliance = self.create_client(SetBool, f"{target}/set_compliance")
        self._params = self.create_client(SetParameters, f"{target}/set_parameters")

    def _on_state(self, msg: JointState) -> None:
        self.state = list(msg.position)

    def _on_force(self, msg: JointState) -> None:
        self.force = list(msg.effort)

    def _on_diagnostics(self, msg: DiagnosticArray) -> None:
        for status in msg.status:
            for entry in status.values:
                if entry.key != "force_zero":
                    continue
                channel = status.hardware_id.rsplit("/", 1)[-1]
                if channel in CHANNEL_IDS:
                    self.zero[CHANNEL_IDS.index(channel)] = float(entry.value)

    def touch(self, index: int) -> float:
        """Force on one fingertip, against the zero the driver is using."""
        return self.force[index] - self.zero[index]

    # -- driving the hand --------------------------------------------------
    @property
    def ready(self) -> bool:
        """Every input this tool reads has arrived at least once.

        Force is in here deliberately. Everything below samples it as though
        it were a measurement, so starting before the first message means
        measuring a placeholder.
        """
        return (
            self.state is not None
            and bool(self.force)
            and self._compliance.service_is_ready()
        )

    def wait_for_hand(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def command(self, index: int, ratio: float) -> None:
        self._command.publish(
            JointState(name=[CHANNEL_IDS[index]], position=[float(ratio)])
        )

    def set_compliance(self, enable: bool) -> str:
        future = self._compliance.call_async(SetBool.Request(data=enable))
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            return "no answer from ~/set_compliance"
        return future.result().message

    def set_parameters(self, values: dict) -> str:
        if not values:
            return ""
        if not self._params.wait_for_service(timeout_sec=5.0):
            return "no parameter service"
        request = SetParameters.Request(
            parameters=[
                Parameter(name, value=value).to_parameter_msg()
                for name, value in values.items()
            ]
        )
        future = self._params.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            return "no answer from the parameter service"
        refused = [r.reason for r in result.results if not r.successful]
        return "; ".join(refused)

    def settle(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def settle_until_still(
        self, index: int, timeout: float, still_for: float = 0.4, epsilon: float = 0.002
    ) -> float:
        """Wait for one DOF to stop moving, and return where it stopped.

        A fixed wait is not good enough here: the rest position is what the
        whole verdict is measured against, so reading it while the finger is
        still travelling would mark a finger that came back perfectly as having
        failed to. Falls back to the last reading on timeout.
        """
        deadline = time.monotonic() + timeout
        last = self.state[index]
        still_since = time.monotonic()
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            if abs(self.state[index] - last) > epsilon:
                still_since = now
            last = self.state[index]
            if now - still_since >= still_for:
                return last
        return last


def bar(value: float, full: float, width: int = 26, peak: Optional[float] = None) -> str:
    """A fill bar, with an optional high-water mark left where the peak was."""
    def cell(v: float) -> int:
        return max(0, min(width, int(round(width * v / full)))) if full > 0 else 0

    filled = cell(value)
    cells = ["#"] * filled + ["."] * (width - filled)
    if peak is not None:
        mark = min(width - 1, max(0, cell(peak) - 1))
        if mark >= filled:
            cells[mark] = "|"
    return "".join(cells)


class Block:
    """Redraw the same few lines in place, instead of racing a carriage return.

    Three things were making the old single-line readout unreadable: it redrew
    on every message rather than at a rate an eye can follow, a bare ``\r``
    left the tail of whatever longer line came before it, and the terminal's
    cursor blinked its way along behind the text. This throttles, clears each
    line as it goes, and parks the cursor out of sight until the block is done.

    Not a terminal? Then nothing is in place: frames are printed plainly and
    rarely, so a piped or logged run stays readable.
    """

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.tty = sys.stdout.isatty()
        self._lines = 0
        self._last = 0.0
        if self.tty:
            sys.stdout.write("\x1b[?25l")  # hide the cursor
            sys.stdout.flush()

    def update(self, rows: List[str], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return
        if not self.tty:
            # Plain mode: one frame every couple of seconds, no cursor tricks.
            if force or now - self._last >= 2.0:
                self._last = now
                print("\n".join(rows), flush=True)
            return
        self._last = now
        if self._lines:
            sys.stdout.write(f"\x1b[{self._lines}A")
        sys.stdout.write("".join(f"\x1b[2K{row}\n" for row in rows))
        sys.stdout.flush()
        self._lines = len(rows)

    def finish(self) -> None:
        """Leave the last frame on screen and give the cursor back."""
        if self.tty:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        self._lines = 0


def quiet_band(node: ComplianceCheck, seconds: float) -> List[float]:
    """Measure how far each channel wanders with nothing touching it.

    The deadband exists to clear this and very little else, so it has to be
    measured on its own. Mixed in with the push phase it cannot be: the lowest
    reading of a channel someone is pushing is a push artefact, not a resting
    level -- thumb rotation swings to -243 g under load and rests near -80.
    """
    low = [float("inf")] * 6
    high = [float("-inf")] * 6
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        for i, force in enumerate(node.force):
            low[i] = min(low[i], force)
            high[i] = max(high[i], force)
    return [
        max(abs(low[i]), abs(high[i])) if low[i] != float("inf") else 0.0
        for i in range(6)
    ]


def characterize(node: ComplianceCheck, seconds: float) -> None:
    """Show what every fingertip reads, and turn that into gains worth trying."""
    print(f"\n1. Hands off for {BASELINE_SEC:.0f} s -- measuring the resting band.")
    band = quiet_band(node, BASELINE_SEC)
    print("   " + "  ".join(f"{DOF_ORDER[i]} +-{band[i]:.0f}" for i in range(6)))

    print(
        f"\n2. Now push each fingertip in turn, for {seconds:.0f} s. Push about as hard as\n"
        "   you would want the finger to give way under.\n"
        "   The | in each bar is the highest that channel has read.\n"
    )
    peak = [0.0] * 6
    smooth = [0.0] * 6
    rows: List[str] = []
    block = Block()
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            for i, force in enumerate(node.force):
                peak[i] = max(peak[i], force)
                # Display only: the peaks and the summary below are raw, but a
                # bar redrawn off a noisy reading is a bar nobody can read.
                smooth[i] += 0.35 * (force - smooth[i])
            # Auto-range, so a 30 g push is not an invisible sliver on a scale
            # picked for a grip that never happens.
            scale = max(200.0, *peak)
            rows = [
                f"  {DOF_ORDER[i]:<16}{smooth[i]:5.0f} g  "
                f"|{bar(smooth[i], scale, peak=peak[i])}|  peak {peak[i]:4.0f}"
                for i in range(6)
            ]
            rows.append(f"  {'':<16}full scale {scale:.0f} g"
                        f"{'':<12}{max(0.0, deadline - time.monotonic()):4.0f} s left")
            block.update(rows)
        if rows:
            block.update(rows, force=True)
    finally:
        block.finish()

    print(f"\n  {'channel':<18}{'resting +-':>12}{'peak push':>11}")
    for i in range(6):
        print(f"  {DOF_ORDER[i]:<18}{band[i]:>12.0f}{peak[i]:>11.0f}")

    recommend(band, peak)


def recommend(band: Sequence[float], peak: Sequence[float]) -> None:
    """Turn a characterisation into a deadband and a gain worth trying."""
    default = [CHANNEL_IDS.index(c) for c in ("1", "2", "3", "4", "5")]
    if not any(peak[i] > 0 for i in default):
        print("\n  Nothing read a push at all. Either nothing was pushed, or the sensor is")
        print("  not seeing it -- FORCE_ACT reads the fingertip pad only, not the phalanx.")
        return

    # The deadband has one job: sit above the resting band, with margin for the
    # preload of whatever the hand is gripping. It is emphatically not half the
    # push -- that would spend most of the signal before anything moved.
    deadband = max(20.0, max(band[i] for i in default) * 2.0)
    deadband = round(deadband / 10.0) * 10.0

    # The gain is what decides how far a finger gives. Aim for a moderate push
    # -- half the hardest one measured -- to give about 150 counts, 15 % of
    # travel: enough to feel and to see, with headroom under max_yield.
    usable = sorted(peak[i] for i in default if peak[i] > deadband + 50)
    if not usable:
        print(
            f"\n  Every channel's peak is under {deadband + 50:.0f} g, so none of them would "
            f"give.\n  Either push harder, or these pads are not reporting usefully."
        )
        return
    typical = usable[len(usable) // 2]
    gain = 150.0 / max(1.0, 0.5 * typical - deadband)

    print(f"\n  Suggested starting point:")
    print(f"    --deadband {deadband:.0f} --gain {gain:.2f}")
    print(
        f"\n  The deadband clears a resting band of +-{max(band[i] for i in default):.0f} g "
        f"with margin.\n"
        f"  The gain gives ~150 counts (0.15 ratio) at {0.5 * typical:.0f} g, half the "
        f"hardest\n  push measured; a full {typical:.0f} g push would ask for "
        f"{(typical - deadband) * gain:.0f} counts and be\n"
        f"  capped at max_yield (300)."
    )

    weak = [i for i in default if peak[i] <= deadband + 50]
    if weak:
        print(
            f"\n  Too weak to give: {', '.join(DOF_ORDER[i] for i in weak)} "
            f"(peak {', '.join(f'{peak[i]:.0f}' for i in weak)} g).\n"
            f"  Leave them out with --channel, or check the pad is what you pushed."
        )


def run_test(node: ComplianceCheck, args) -> int:
    index = CHANNEL_IDS.index(args.channel)
    name = DOF_ORDER[index]

    print(f"\n1. Baseline: {BASELINE_SEC:.0f} s with nothing touching the hand.")
    samples = []
    deadline = time.monotonic() + BASELINE_SEC
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        samples.append(node.force[index])
    low, high = (min(samples), max(samples)) if samples else (0.0, 0.0)
    band = high - low
    print(f"   {name} reads {low:.0f}..{high:.0f} g untouched, a band of {band:.0f} g.")
    # The absolute level does not matter -- the driver tares against it when
    # compliant mode is engaged, which is what makes a sensor whose zero has
    # walked to +219 g usable at all. What the deadband has to clear is how
    # far the reading wanders around that level.
    if band >= args.deadband:
        print(f"   !! That band alone reaches the {args.deadband:.0f} g deadband, so noise")
        print("      would open the finger. Raise --deadband above it and re-run.")
        return 2
    if abs(low) > 50:
        print(
            f"   note: this pad's zero has walked to {low:.0f} g. Engaging compliant mode\n"
            f"         tares it away, so keep your hands off until it says it is on."
        )

    refused = node.set_parameters(
        {
            "compliance_deadband": float(args.deadband),
            "compliance_counts_per_gram": float(args.gain),
            "compliance_channels": [args.channel],
        }
    )
    if refused:
        print(f"   !! the driver refused a gain: {refused}")
        return 2
    print(f"   gains set: deadband={args.deadband:g} g, {args.gain:g} counts/g")

    print(f"\n2. Commanding {name} to open ratio {args.rest:.2f} and letting it settle.")
    node.command(index, args.rest)
    rest = node.settle_until_still(index, timeout=max(5.0, args.settle * 3))
    print(f"   settled at {rest:.3f} (commanded {args.rest:.2f}).")
    if abs(rest - args.rest) > 0.05:
        print(
            f"   note: {abs(rest - args.rest):.3f} of tracking offset. The test measures "
            f"against\n         where it actually settled, not the commanded value."
        )

    print(f"\n3. {node.set_compliance(True)}")
    # Engaging takes the tare; wait for the diagnostics that report it, so
    # that what follows measures force the same way the controller does.
    node.settle(1.0)
    print(f"   fingertip zero for {name} is {node.zero[index]:.0f} g; "
          f"force below is measured against it.")
    print(
        f"\n   Push the {name} fingertip, hold it, then let go.\n"
        f"   Pass needs it to give {args.min_open:.2f} and come back within "
        f"{args.tolerance:.2f}.  Ctrl-C to stop.\n"
    )

    gave = False
    returned = False
    peak_force = 0.0
    peak_open = 0.0
    smooth_force = 0.0
    block = Block()
    quiet_since: Optional[float] = None
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not returned:
        rclpy.spin_once(node, timeout_sec=0.05)
        force = node.touch(index)
        opened = node.state[index] - rest
        peak_force = max(peak_force, force)
        peak_open = max(peak_open, opened)
        if opened >= args.min_open:
            gave = True
        if force < args.deadband:
            quiet_since = quiet_since or time.monotonic()
            if (
                gave
                and time.monotonic() - quiet_since >= args.settle
                and abs(node.state[index] - rest) <= args.tolerance
            ):
                returned = True
        else:
            quiet_since = None
        smooth_force += 0.35 * (force - smooth_force)
        if not gave:
            status = "waiting for a push"
        elif quiet_since is None:
            status = "giving -- let go when you are ready"
        else:
            status = "released, coming back"
        block.update([
            f"   force    {smooth_force:5.0f} g  |{bar(smooth_force, 500, peak=peak_force)}|"
            f"  peak {peak_force:4.0f} g",
            f"   opening  {opened:+.3f}  |{bar(opened, 0.30, peak=peak_open)}|"
            f"  peak {peak_open:+.3f}",
            f"   {status:<58}",
        ])
    block.finish()

    print("")
    print(f"   peak force {peak_force:.0f} g over the zero, "
          f"peak opening {peak_open:+.3f} ratio")
    print(f"   returned to {node.state[index]:.3f} against a rest of {rest:.3f}")

    if gave and returned:
        print("\n   PASS: the finger gave to the push and came back to where it was.")
        return 0
    print("")
    if not gave:
        print(f"   FAIL: the finger never opened {args.min_open:.2f}.")
        if peak_force <= args.deadband:
            print(
                f"         Peak force {peak_force:.0f} g never cleared the "
                f"{args.deadband:.0f} g deadband -- lower it, or the sensor is not"
            )
            print("         seeing the push (it only reads the pad, not the phalanx).")
        else:
            print(
                f"         Force got to {peak_force:.0f} g, so the gain is what is short: "
                f"{args.gain:g} counts/g"
            )
            print(
                f"         gives {(peak_force - args.deadband) * args.gain:.0f} counts "
                f"({(peak_force - args.deadband) * args.gain / 1000:.3f} ratio). Raise --gain."
            )
    elif not returned:
        print(f"   FAIL: it gave {peak_open:+.3f} but did not come back within "
              f"{args.tolerance:.2f}.")
        print("         Check compliance_return_rate is not 0, and that the finger is not")
        print("         resting against something that stops it closing again.")
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--node", default="/inspire_hand", help="Driver node to drive.")
    parser.add_argument(
        "--channel", default="4", choices=list(CHANNEL_IDS),
        help="Which DOF to test. 4 is the index finger.",
    )
    parser.add_argument("--rest", type=float, default=0.30,
                        help="Open ratio to hold as the rest position.")
    parser.add_argument("--deadband", type=float, default=40.0,
                        help="Grams of fingertip force to ignore.")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Angle counts of opening per gram above the deadband.")
    parser.add_argument("--min-open", type=float, default=0.03,
                        help="Open ratio the finger must give by to count as giving.")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="How close to the rest position counts as reinstated.")
    parser.add_argument("--settle", type=float, default=1.5,
                        help="Seconds to wait for a move to finish, and for release.")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Seconds to wait for a push and a release.")
    parser.add_argument("--characterize", action="store_true",
                        help="Only measure what the fingertips read; command nothing.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = ComplianceCheck(args.node.rstrip("/"))
    code = 1
    try:
        if not node.wait_for_hand():
            print(
                f"No hand on {args.node}. Launch it first:\n"
                f"  ros2 launch inspire_hand_driver inspire_hand.launch.py port:=/dev/ttyUSB0"
            )
            return 2
        if args.characterize:
            characterize(node, 20.0)
            return 0
        code = run_test(node, args)
    except KeyboardInterrupt:
        print("\n  stopped")
    except ExternalShutdownException:
        # SIGTERM, or the launch going down underneath us. Falls through to the
        # same cleanup as Ctrl-C: a tool that leaves a hand compliant and a
        # terminal with no cursor because it was killed is not finished.
        print("\n  shut down")
    finally:
        # Leave the hand stiff and where it was told to be, whatever happened,
        # and the terminal with a cursor. Guarded because this runs on paths
        # where rclpy is already going down and cannot service a call.
        try:
            sys.stdout.write("\x1b[?25h")
            if not args.characterize and rclpy.ok():
                node.set_compliance(False)
                node.settle(1.0)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
