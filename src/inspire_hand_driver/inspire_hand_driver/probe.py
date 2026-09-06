"""Read-only bring-up probe for an Inspire hand on a serial port.

Scans both wire protocols across the plausible baud rates and hand IDs and
reports what answers. Never issues a write, so it cannot move the hand.

    ros2 run inspire_hand_driver inspire_hand_probe [PORT]
"""

from __future__ import annotations

import argparse
import sys

from .protocol import (
    REG_ANGLE_ACT,
    REG_ERROR,
    REG_FORCE_ACT,
    REG_HAND_ID,
    REG_STATUS,
    REG_TEMP,
    HandCommunicationError,
    HandTransport,
)

BAUD_RATES = (115200, 57600, 19200, 9600, 460800, 921600)
HAND_IDS = (1, 2, 3, 4, 5)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default="/dev/ttyUSB0")
    # ROS injects --ros-args when launched via 'ros2 run'; ignore it.
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    print(f"Probing {args.port} (read-only, no motion)\n")
    hits = []
    for protocol in ("modbus", "legacy"):
        for baud in BAUD_RATES:
            for hand_id in HAND_IDS:
                transport = HandTransport(
                    port=args.port,
                    baudrate=baud,
                    hand_id=hand_id,
                    protocol=protocol,
                    timeout=0.15,
                )
                try:
                    transport.connect()
                except Exception as exc:  # noqa: BLE001
                    print(f"cannot open {args.port}: {exc}")
                    return 1
                reported = transport.ping()
                transport.close()
                if reported is not None:
                    print(f"  *** {protocol:6s} baud={baud:<7d} id={hand_id} -> HAND_ID={reported}")
                    hits.append((protocol, baud, hand_id))

    if not hits:
        print("  no response on any protocol/baud/id combination\n")
        print("Silence here means the frames left the adapter but nothing came")
        print("back. Work down the physical layer, most likely first:")
        print("")
        print("  1. WRONG WIRES. The Lemo 8-pin cable carries four signal pairs and")
        print("     only ONE of them is RS485 (per the RH56 series manual):")
        print("        yellow       = 485_A / CAN_H   <-- adapter terminal A")
        print("        green        = 485_B / CAN_L   <-- adapter terminal B")
        print("        red (thick)  = VCC 24V")
        print("        black(thick) = GND             <-- adapter GND, if it has one")
        print("        white / blue / red(thin) / black(thin) = Ethernet TX+/-, RX+/-")
        print("     Landing white+blue on A/B looks exactly like this failure.")
        print("  2. A and B swapped. Harmless to try; a reversed pair idles low and")
        print("     shows up as a solid RX LED plus stray 0x00 bytes.")
        print("  3. No common ground between adapter and the hand's 24V supply.")
        print("  4. A wire is loose in the screw terminal.")
        print("  5. The hand is a CAN variant. The RH56 series ships RS485 or CAN;")
        print("     yellow/green carry CAN_H/CAN_L on those units and no RS485")
        print("     adapter will ever reach them.")
        print("")
        print("Manual defaults: ID=1, 115200 8N1, Modbus RTU. Baud register accepts")
        print("only 115200 / 57600 / 19200 / 921600 - all of which this probe scans.")
        return 2

    protocol, baud, hand_id = hits[0]
    print(f"\n=== State at protocol={protocol} baud={baud} id={hand_id} ===")
    transport = HandTransport(port=args.port, baudrate=baud, hand_id=hand_id, protocol=protocol)
    transport.connect()
    for label, addr, count in (
        ("HAND_ID", REG_HAND_ID, 1),
        ("ANGLE_ACT", REG_ANGLE_ACT, 6),
        ("FORCE_ACT", REG_FORCE_ACT, 6),
        ("ERROR", REG_ERROR, 3),
        ("STATUS", REG_STATUS, 3),
        ("TEMP", REG_TEMP, 3),
    ):
        try:
            print(f"  {label:10s} = {transport.read_registers(addr, count)}")
        except HandCommunicationError as exc:
            print(f"  {label:10s} = ERROR: {exc}")
    transport.close()
    print(f"\nLaunch with:\n  ros2 launch inspire_hand_driver inspire_hand.launch.py \\\n"
          f"    port:={args.port} baudrate:={baud} hand_id:={hand_id} protocol:={protocol}")
    print("\nOr, alongside the arm:\n"
          f"  ros2 launch inspire_franka_bringup inspire_franka.launch.py \\\n"
          f"    hand_port:={args.port} hand_baudrate:={baud} hand_id:={hand_id} "
          f"hand_protocol:={protocol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
