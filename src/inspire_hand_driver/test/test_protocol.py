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
