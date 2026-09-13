"""Protocol-level tests that need neither hardware nor a built ROS workspace."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver.protocol import (  # noqa: E402
    ANGLE_MAX,
    DOF_ORDER,
    CHANNEL_IDS,
    REG_ANGLE_ACT,
    REG_ANGLE_SET,
    HandTransport,
    MockTransport,
    crc16_modbus,
)


def test_crc16_matches_modbus_reference_vector():
    # Canonical Modbus RTU example: read 1 register at 0 from slave 1.
    assert crc16_modbus(bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01])) == bytes([0x84, 0x0A])


def test_crc16_detects_single_bit_corruption():
    frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x06])
    corrupted = bytes([frame[0] ^ 0x01]) + frame[1:]
    assert crc16_modbus(frame) != crc16_modbus(corrupted)


def test_channel_ids_align_with_register_dof_order():
    # The hand numbers its channels 1..6 in exactly the order it lays out
    # each 6-register block. If these ever diverge, commands would silently
    # drive the wrong fingers.
    assert len(CHANNEL_IDS) == len(DOF_ORDER) == 6
    assert CHANNEL_IDS == ("1", "2", "3", "4", "5", "6")


def test_modbus_read_frame_is_well_formed():
    t = HandTransport(hand_id=2, protocol="modbus")
    sent = {}

    class FakeSerial:
        is_open = True

        def reset_input_buffer(self):
            pass

        def write(self, data):
            sent["req"] = data

        def flush(self):
            pass

        def read(self, n):
            # slave, fc, bytecount, 6 registers big-endian, crc
            body = bytes([2, 0x03, 12]) + b"".join(
                v.to_bytes(2, "big") for v in (1000, 900, 800, 700, 600, 500)
            )
            return body + crc16_modbus(body)

    t._serial = FakeSerial()
    vals = t.read_registers(1546, 6)
    assert vals == [1000, 900, 800, 700, 600, 500]
    req = sent["req"]
    assert req[0] == 2 and req[1] == 0x03
    assert int.from_bytes(req[2:4], "big") == 1546
    assert int.from_bytes(req[4:6], "big") == 6
    assert crc16_modbus(req[:-2]) == req[-2:]


def test_legacy_read_frame_header_and_checksum():
    t = HandTransport(hand_id=1, protocol="legacy")
    sent = {}

    class FakeSerial:
        is_open = True

        def reset_input_buffer(self):
            pass

        def write(self, data):
            sent["req"] = data

        def flush(self):
            pass

        def read(self, n):
            return bytes([0xEB, 0x90, 1, 15, 0x11, 0x0A, 0x06]) + b"\x00" * 12

    t._serial = FakeSerial()
    t.read_registers(1546, 6)
    req = sent["req"]
    assert req[0] == 0xEB and req[1] == 0x90
    assert req[2] == 1 and req[4] == 0x11
    assert int.from_bytes(req[5:7], "little") == 1546
    assert req[-1] == sum(req[2:-1]) & 0xFF


def test_write_angles_clamps_out_of_range_values():
    m = MockTransport()
    m.connect()
    m.write_angles([-500, 5000, 0, 1000, 500, 250])
    assert m._targets == [0.0, 1000.0, 0.0, 1000.0, 500.0, 250.0]


def test_write_angles_rejects_wrong_length():
    m = MockTransport()
    m.connect()
    with pytest.raises(ValueError):
        m.write_angles([0, 0, 0])


def test_mock_slews_toward_target_rather_than_jumping():
    m = MockTransport(slew_per_sec=1000.0)
    m.connect()
    assert m.read_angles() == [ANGLE_MAX] * 6  # starts fully open
    m.write_angles([0] * 6)
    time.sleep(0.1)
    mid = m.read_angles()
    assert all(0 < a < ANGLE_MAX for a in mid), f"expected mid-travel, got {mid}"
    time.sleep(1.2)
    assert m.read_angles() == [0] * 6  # eventually arrives


def test_unknown_protocol_rejected_at_construction():
    with pytest.raises(ValueError):
        HandTransport(protocol="ethercat")


def test_modbus_words_are_big_endian_against_captured_hardware_reply():
    """Byte-for-byte reply captured from the bench RH56, fully open at rest.

    Modbus RTU register values are big-endian on the wire. Decoding them
    little-endian is not a cosmetic error: it read HAND_ID 1 as 256 and an
    angle of 1000 as 59395, and the same swap on the write path turned every
    commanded angle into an out-of-range value. Both directions are asserted
    here with real bytes so a fake reply can never re-encode the driver's own
    assumption.
    """
    t = HandTransport(hand_id=1, protocol="modbus")
    sent = {}

    class FakeSerial:
        is_open = True

        def reset_input_buffer(self):
            pass

        def write(self, data):
            sent["req"] = data

        def flush(self):
            pass

        def read(self, n):
            # ANGLE_ACT, six DOF, as captured: 03 e8 = 1000 = fully open.
            body = bytes([1, 0x03, 12]) + bytes.fromhex("03e803e803e803e803e403dd")
            return body + crc16_modbus(body)

    t._serial = FakeSerial()
    assert t.read_registers(REG_ANGLE_ACT, 6) == [1000, 1000, 1000, 1000, 996, 989]

    t.write_angles([1000, 900, 800, 700, 600, 500])
    payload = sent["req"][7:-2]
    assert payload == bytes.fromhex("03e80384032002bc025801f4")


def test_wire_model_matches_the_modbus_frame_sizes():
    """The floor is arithmetic: 8N1 bytes on the wire, both frames."""
    from inspire_hand_driver.benchmark import budget, frame_time, wire_time

    # A 6-register read is 8 bytes out and 17 back; a write is 21 out and 8 back.
    assert wire_time(115200, 6, write=False) == pytest.approx(frame_time(25, 115200))
    assert wire_time(115200, 6, write=True) == pytest.approx(frame_time(29, 115200))
    # Doubling the baud rate halves the byte time.
    assert wire_time(230400, 6) == pytest.approx(wire_time(115200, 6) / 2)
    # The spec's inter-frame silence is additional, and at 115200 it dominates.
    assert wire_time(115200, 6, silence=True) > 2 * wire_time(115200, 6)


def test_command_budget_is_what_polling_leaves_behind():
    from inspire_hand_driver.benchmark import budget

    # Three 2 ms reads per publish at 50 Hz is 0.3 s of every second.
    commands, polling = budget(0.002, 0.002, 50.0)
    assert polling == pytest.approx(0.3)
    assert commands == pytest.approx(350.0)

    # Polling that cannot even fit in a second leaves nothing for commands.
    commands, polling = budget(0.010, 0.002, 50.0)
    assert polling > 1.0
    assert commands == 0.0


# -- health block, error clearing, and the mock's stall model ------------------

def test_byte_registers_unpack_lower_address_from_the_low_byte():
    """ERROR/STATUS/TEMP are one byte per DOF, two to a Modbus register.

    The vendor's own example unpacks the lower address from the low byte;
    getting this backwards would swap every finger's status with its
    neighbour's and the stall guard would back off the wrong DOF.
    """
    from inspire_hand_driver.protocol import pack_bytes, unpack_bytes

    assert unpack_bytes([0x0201, 0x0403, 0x0605]) == [1, 2, 3, 4, 5, 6]
    assert pack_bytes([1, 2, 3, 4, 5, 6]) == [0x0201, 0x0403, 0x0605]
    assert unpack_bytes(pack_bytes([7, 0, 0, 0, 0, 3])) == [7, 0, 0, 0, 0, 3]


class _HealthSerial:
    """Answers reads at 1606 either with all nine registers or a refusal."""

    is_open = True

    def __init__(self, refuse_span: bool):
        self.refuse_span = refuse_span
        self.requests = []
        self._reply = b""

    def reset_input_buffer(self):
        pass

    def flush(self):
        pass

    def write(self, data):
        self.requests.append(data)
        addr = int.from_bytes(data[2:4], "big")
        count = int.from_bytes(data[4:6], "big")
        if self.refuse_span and count > 3:
            body = bytes([1, 0x83, 0x02])  # illegal data address
        else:
            block = {1606: [0x01, 0, 0, 0x04, 0, 0], 1612: [6, 2, 2, 5, 3, 0],
                     1618: [40, 41, 42, 43, 44, 45]}
            raw = block[1606] + block[1612] + block[1618]
            raw = raw[addr - 1606: addr - 1606 + 2 * count]
            from inspire_hand_driver.protocol import pack_bytes
            words = pack_bytes(raw)
            body = bytes([1, 0x03, 2 * len(words)]) + b"".join(
                w.to_bytes(2, "big") for w in words
            )
        self._reply = body + crc16_modbus(body)

    def read(self, n):
        out, self._reply = self._reply[:n], self._reply[n:]
        return out


def test_read_health_fetches_error_status_and_temperature_in_one_read():
    t = HandTransport(hand_id=1, protocol="modbus")
    t._serial = _HealthSerial(refuse_span=False)
    health = t.read_health()
    assert health.errors == [1, 0, 0, 4, 0, 0]
    assert health.status == [6, 2, 2, 5, 3, 0]
    assert health.temperatures == [40, 41, 42, 43, 44, 45]
    assert len(t._serial.requests) == 1
    # locked rotor on the pinky, current protection on the index; the thumb
    # stopped at its force threshold is not a stall.
    assert health.stalled() == [0, 3]


def test_read_health_falls_back_to_three_reads_if_the_hand_refuses_the_span():
    from inspire_hand_driver.protocol import HandProtocolError

    t = HandTransport(hand_id=1, protocol="modbus")
    t._serial = _HealthSerial(refuse_span=True)
    health = t.read_health()
    assert health.status == [6, 2, 2, 5, 3, 0]
    assert t._split_health_reads, "the refusal must be remembered"
    assert len(t._serial.requests) == 4  # one refused, then three
    t.read_health()
    assert len(t._serial.requests) == 7, "and not retried every time"
    with pytest.raises(HandProtocolError):
        t.read_registers(1606, 9)


def test_clear_errors_writes_exactly_one_and_never_touches_save():
    """1004 CLEAR_ERROR and 1005 SAVE share a register; only the low byte may be set."""
    from inspire_hand_driver.protocol import REG_CLEAR_ERROR

    t = HandTransport(hand_id=1, protocol="modbus")
    sent = {}

    class FakeSerial:
        is_open = True

        def reset_input_buffer(self):
            pass

        def write(self, data):
            sent["req"] = data

        def flush(self):
            pass

        def read(self, n):
            body = bytes([1, 0x10]) + (1004).to_bytes(2, "big") + (1).to_bytes(2, "big")
            return body + crc16_modbus(body)

    t._serial = FakeSerial()
    t.clear_errors()
    req = sent["req"]
    assert req[1] == 0x10
    assert int.from_bytes(req[2:4], "big") == REG_CLEAR_ERROR
    assert int.from_bytes(req[4:6], "big") == 1
    assert req[7:9] == b"\x00\x01", "high byte is SAVE and must stay clear"


def test_mock_finger_stops_at_the_force_threshold_without_an_error():
    from inspire_hand_driver.protocol import STATUS_AT_FORCE

    m = MockTransport(slew_per_sec=1e6, stall_after_sec=0.01)
    m.connect()
    m.write_forces([500] * 6)
    m.obstacles[3] = 400  # something in the index finger's way
    m.write_angles([0] * 6)
    time.sleep(0.05)
    assert m.read_angles()[3] == 400
    time.sleep(0.05)
    assert m.read_angles()[3] == 400
    assert m.status_codes()[3] == STATUS_AT_FORCE
    assert m.read_forces()[3] == 500
    assert m.latched_errors() == [0] * 6


def test_mock_finger_without_a_threshold_latches_until_cleared():
    from inspire_hand_driver.protocol import (
        ERROR_LOCKED_ROTOR,
        STATUS_AT_TARGET,
        STATUS_LOCKED_ROTOR,
    )

    m = MockTransport(slew_per_sec=1e6, stall_after_sec=0.01)
    m.connect()
    m.obstacles[3] = 400
    m.write_angles([0] * 6)
    time.sleep(0.02)
    m.read_angles()  # arrives at the obstacle and starts pushing
    time.sleep(0.02)
    assert m.status_codes()[3] == STATUS_LOCKED_ROTOR
    assert m.read_health().errors[3] == ERROR_LOCKED_ROTOR
    # Dead: a reachable target does nothing.
    m.write_angles([1000] * 6)
    time.sleep(0.02)
    assert m.read_angles()[3] == 400
    m.clear_errors()
    time.sleep(0.02)
    assert m.read_angles()[3] == 1000
    assert m.status_codes()[3] == STATUS_AT_TARGET
    assert m.clear_error_writes == 1 and m.flash_saves == 0
