"""How fast the RH56 can actually be commanded, modelled and measured.

    ros2 run inspire_hand_driver inspire_hand_benchmark [PORT]

The question this answers is what rate to stream hand targets at during a
coordinated replay. Three separate ceilings apply, and the lowest of them wins:

**The wire.** RS485 is half-duplex and every transaction is a request followed
by a reply, so the two frames add. At 8N1 a byte costs ten bit times, which the
frame sizes below turn into a hard floor no firmware can beat. This part is
arithmetic, so :func:`wire_time` reports it whether or not a hand is attached.

**The device turnaround.** The gap between the last byte of a request and the
first byte of the reply is the hand's own business and is not in any datasheet.
It is what the measurement here is for.

**The bus budget.** Commands share the line with the driver's state polling,
and :meth:`~inspire_hand_driver.driver_node.InspireHandNode._on_timer` spends
three read transactions per publish -- angles, currents, forces -- so at
``publish_rate_hz`` those cost ``3 * T_read * rate`` of every second before a
single command is sent. :func:`budget` divides what is left by the cost of one
write.

Above all three sits the hand itself: bench replay measured about 0.17 s of lag
between a commanded step and the joint arriving, which is the actuator's own
closed-loop response. Commanding faster than that resolves does not make the
hand track better, it just fills the bus. Rate selection should be driven by
the smoothness the trajectory needs, not by this ceiling.

Nothing here writes to the hand by default: the benchmark's write path
re-commands the pose the hand is already holding, so the measurement does not
move it. ``--move`` is required before it will command anything else.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Callable, List, Sequence

from .protocol import (
    ANGLE_INVALID,
    ANGLE_MAX,
    REG_CURRENT,
    HandCommunicationError,
    HandTransport,
    MockTransport,
)

# Modbus RTU frame sizes, in bytes, for a block of N holding registers.
# Read (FC 0x03):  request  id fc addr(2) count(2) crc(2)            = 8
#                  reply    id fc nbytes data(2N) crc(2)             = 5 + 2N
# Write (FC 0x10): request  id fc addr(2) count(2) nbytes data(2N) crc(2) = 9 + 2N
#                  reply    id fc addr(2) count(2) crc(2)            = 8
READ_REQUEST_BYTES = 8
READ_REPLY_BYTES = 5
WRITE_REQUEST_BYTES = 9
WRITE_REPLY_BYTES = 8

# 8N1: one start bit, eight data bits, no parity, one stop bit.
BITS_PER_BYTE = 10

# Modbus RTU separates frames by 3.5 character times, which the spec fixes at
# 1.750 ms for any baud rate above 19200 rather than letting it shrink. Two
# gaps bracket each transaction. Whether a given device actually enforces this
# is exactly the sort of thing the measurement settles.
T35_FIXED_SEC = 0.00175
T35_FIXED_ABOVE_BAUD = 19200


def frame_time(byte_count: int, baudrate: int) -> float:
    """Seconds to put ``byte_count`` bytes on the wire at ``baudrate``, 8N1."""
    return byte_count * BITS_PER_BYTE / float(baudrate)


def t35(baudrate: int) -> float:
    """The inter-frame silence Modbus RTU requires at ``baudrate``."""
    if baudrate > T35_FIXED_ABOVE_BAUD:
        return T35_FIXED_SEC
    return frame_time(3.5, baudrate)


def wire_time(baudrate: int, registers: int = 6, write: bool = False,
              silence: bool = False) -> float:
    """One transaction's time on the wire: both frames, optionally plus silences.

    With ``silence=False`` this is the bytes alone -- a floor nothing can beat.
    With ``silence=True`` it adds the two t3.5 gaps the Modbus spec mandates,
    which at 115200 baud cost more than the bytes do. Whether the hand and the
    USB-serial adapter actually impose them is not knowable from here, so the
    two are reported as a range and the measurement settles which end is real.
    """
    if write:
        request = WRITE_REQUEST_BYTES + 2 * registers
        reply = WRITE_REPLY_BYTES
    else:
        request = READ_REQUEST_BYTES
        reply = READ_REPLY_BYTES + 2 * registers
    bytes_time = frame_time(request + reply, baudrate)
    return bytes_time + (2 * t35(baudrate) if silence else 0.0)


def budget(read_time: float, write_time: float, publish_rate: float, reads_per_publish: int = 3):
    """Commands per second left after the driver's state polling takes its share.

    Returns ``(commands_per_second, fraction_of_bus_spent_polling)``. A polling
    fraction at or above 1.0 means the driver cannot even sustain its own
    publish rate, and the command rate is reported as 0.
    """
    polling = reads_per_publish * read_time * publish_rate
    if polling >= 1.0:
        return 0.0, polling
    return (1.0 - polling) / write_time, polling


def _time_calls(call: Callable[[], object], count: int) -> List[float]:
    """Run ``call`` ``count`` times, returning each duration in seconds."""
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return samples


def _report(label: str, samples: Sequence[float], floor: float) -> None:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(
        f"  {label:22s} median {1000 * median:6.2f} ms  p95 {1000 * p95:6.2f} ms  "
        f"max {1000 * ordered[-1]:6.2f} ms   -> {1.0 / median:6.1f} Hz"
    )
    if median < floor:
        print(
            f"  {'':22s} faster than the {1000 * floor:.2f} ms wire floor, so nothing "
            f"was actually put on a wire"
        )
    else:
        print(
            f"  {'':22s} wire floor {1000 * floor:5.2f} ms, so the hand's own "
            f"turnaround is about {1000 * (median - floor):.2f} ms"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--protocol", default="modbus", choices=("modbus", "legacy"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--publish-rate", type=float, default=50.0,
                        help="the driver's publish_rate_hz, for the bus budget")
    parser.add_argument("--mock", action="store_true", help="model only, no serial port")
    parser.add_argument(
        "--move",
        action="store_true",
        help="allow the write test to command a pose other than the one held",
    )
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    read_floor = wire_time(args.baudrate, 6, write=False)
    write_floor = wire_time(args.baudrate, 6, write=True)
    read_slow = wire_time(args.baudrate, 6, write=False, silence=True)
    write_slow = wire_time(args.baudrate, 6, write=True, silence=True)
    print(f"Wire model at {args.baudrate} baud, 8N1, 6 registers per block")
    print(f"  read  (FC 0x03): {READ_REQUEST_BYTES} B out + {READ_REPLY_BYTES + 12} B back"
          f"  -> {1000 * read_floor:5.2f} ms bytes, {1000 * read_slow:5.2f} ms with t3.5 gaps")
    print(f"  write (FC 0x10): {WRITE_REQUEST_BYTES + 12} B out + {WRITE_REPLY_BYTES} B back"
          f"  -> {1000 * write_floor:5.2f} ms bytes, {1000 * write_slow:5.2f} ms with t3.5 gaps")
    fast, fast_poll = budget(read_floor, write_floor, args.publish_rate)
    slow, slow_poll = budget(read_slow, write_slow, args.publish_rate)
    print(
        f"  the driver polls 3 blocks per publish, which at {args.publish_rate:g} Hz "
        f"costs {100 * fast_poll:.0f} - {100 * slow_poll:.0f} % of the bus,"
    )
    print(f"  leaving roughly {slow:.0f} - {fast:.0f} commands/s for targets.")
    if slow_poll > 0.5:
        print(
            f"  NOTE: polling alone is over half the bus at the pessimistic end. Two of\n"
            f"  those three blocks are current and force, which a replay does not read;\n"
            f"  see the driver's state_extras_divisor parameter."
        )

    transport = (
        MockTransport(baudrate=args.baudrate, hand_id=args.hand_id, protocol=args.protocol)
        if args.mock
        else HandTransport(port=args.port, baudrate=args.baudrate,
                           hand_id=args.hand_id, protocol=args.protocol)
    )
    try:
        transport.connect()
    except Exception as exc:  # noqa: BLE001 - a bad port is a normal outcome here
        print(f"\ncannot open {args.port}: {exc}")
        return 2
    if transport.ping() is None:
        print(
            f"\nNo reply from the hand on {args.port} @ {args.baudrate} baud, "
            f"id={args.hand_id}, protocol={args.protocol}. The wire model above "
            f"still holds; the measured section needs a powered hand. Run "
            f"'ros2 run inspire_hand_driver inspire_hand_probe' to find it."
        )
        transport.close()
        return 1

    where = "an in-memory mock" if args.mock else args.port
    print(f"\nMeasured over {args.samples} transactions on {where}")
    if args.mock:
        print(
            "  The mock has no wire and no actuator, so what follows is this "
            "driver's own\n  software overhead and nothing else. It says whether "
            "the tool works, not\n  what the hand can do -- for that, run it "
            "against a powered hand."
        )
    try:
        held = [a if a != ANGLE_INVALID else ANGLE_MAX for a in transport.read_angles()]
        target = held if not args.move else [ANGLE_MAX] * 6
        if not args.move:
            print(f"  (writes re-command the held pose {held}, so the hand does not move)")

        _report("read_angles", _time_calls(transport.read_angles, args.samples), read_floor)
        _report(
            "read currents",
            _time_calls(lambda: transport.read_registers(REG_CURRENT, 6), args.samples),
            read_floor,
        )
        _report("write_angles", _time_calls(lambda: transport.write_angles(target), args.samples),
                write_floor)

        def publish_cycle():
            transport.read_angles()
            transport.read_registers(REG_CURRENT, 6)
            transport.read_forces()

        cycle = _time_calls(publish_cycle, max(20, args.samples // 4))
        _report("driver publish cycle", cycle, 3 * read_floor)

        write_samples = _time_calls(lambda: transport.write_angles(target), args.samples)
        measured_read = statistics.median(
            _time_calls(transport.read_angles, max(20, args.samples // 4))
        )
        measured_write = statistics.median(write_samples)
        if args.mock:
            return 0
        commands, polling = budget(measured_read, measured_write, args.publish_rate)
        print(
            f"\n  Measured budget: state polling takes {100 * polling:.0f} % of the bus at "
            f"{args.publish_rate:g} Hz, leaving about {commands:.0f} commands/s."
        )
        print(
            "  The hand's own closed-loop response (~0.17 s to settle a step, measured\n"
            "  on the bench) is far slower than any of this, so pick the stream rate for\n"
            "  trajectory smoothness rather than for the ceiling."
        )
    except HandCommunicationError as exc:
        print(f"  transaction failed: {exc}")
        return 1
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
