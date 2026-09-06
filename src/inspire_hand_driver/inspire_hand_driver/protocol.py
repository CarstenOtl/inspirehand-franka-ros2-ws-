"""Transport layer for the Inspire Robotics RH56 dexterous hand.

The RH56 family ships with one of two mutually incompatible serial wire
formats, depending on firmware vintage:

``modbus``
    Standard Modbus RTU (FC 0x03 read / 0x10 write) with CRC16. Used by the
    newer RH56DFX / RH56DFTP firmware.
``legacy``
    Inspire's own ``EB 90`` framing with an 8-bit additive checksum. Used by
    older units and by the vendor's ROS 1 ``inspire_hand`` package.

Both address the same holding-register map, so everything above
:class:`HandTransport` is protocol agnostic.

Register values use the vendor convention of ``0`` = fully closed and
``1000`` = fully open. Conversion to the open-ratio convention used on the
ROS side lives in :mod:`inspire_hand_driver.kinematics`.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence

# Holding-register addresses. Each per-finger block is 6 contiguous registers
# in DOF_ORDER. Transcribed from the RH56 series user manual.
REG_HAND_ID = 1000
REG_CLEAR_ERROR = 1004
REG_SAVE_FLASH = 1005
REG_POS_SET = 1474
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522
REG_POS_ACT = 1534
REG_ANGLE_ACT = 1546
REG_FORCE_ACT = 1582
REG_CURRENT = 1594
REG_ERROR = 1606
REG_STATUS = 1612
REG_TEMP = 1618

# Order the hand expects the six DOF values in, for every 6-register block.
DOF_ORDER = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")

# Channel ids, index-aligned with DOF_ORDER. The hand numbers its DOF 1..6
# little -> thumb-rotation, which is exactly the register layout, so these are
# the hand's own names for its channels and not an invention of this driver.
CHANNEL_IDS = ("1", "2", "3", "4", "5", "6")

ANGLE_MIN = 0
ANGLE_MAX = 1000

# Sentinel the hand reports for a DOF whose value is not currently available.
ANGLE_INVALID = 0xFFFF


class HandCommunicationError(RuntimeError):
    """Raised when the hand does not answer or answers with a bad frame."""


def crc16_modbus(data: bytes) -> bytes:
    """Return the little-endian Modbus RTU CRC16 of ``data``."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, "little")


class HandTransport:
    """Serial transport to one hand.

    Access is serialised with a lock: the ROS node reads state from a timer
    thread while service callbacks write targets, and the RS485 bus is
    half-duplex, so overlapping transactions would interleave frames.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        hand_id: int = 1,
        protocol: str = "modbus",
        timeout: float = 0.25,
    ) -> None:
        if protocol not in ("modbus", "legacy"):
            raise ValueError(f"unknown protocol {protocol!r}, expected 'modbus' or 'legacy'")
        self.port = port
        self.baudrate = baudrate
        self.hand_id = hand_id
        self.protocol = protocol
        self.timeout = timeout
        self._serial = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        import serial  # imported lazily so mock mode needs no pyserial

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
        )

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    @property
    def connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    # -- framing -----------------------------------------------------------
    def _txn(self, request: bytes, expected: int) -> bytes:
        if not self.connected:
            raise HandCommunicationError("serial port is not open")
        self._serial.reset_input_buffer()
        self._serial.write(request)
        self._serial.flush()
        deadline = time.monotonic() + self.timeout
        buf = bytearray()
        while len(buf) < expected and time.monotonic() < deadline:
            chunk = self._serial.read(expected - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        if not buf:
            raise HandCommunicationError(
                f"no reply from hand id={self.hand_id} on {self.port} @ {self.baudrate} baud"
            )
        return bytes(buf)

    def _read_modbus(self, addr: int, count: int) -> List[int]:
        req = bytes([self.hand_id, 0x03]) + addr.to_bytes(2, "big") + count.to_bytes(2, "big")
        req += crc16_modbus(req)
        resp = self._txn(req, 5 + 2 * count)
        if len(resp) < 5:
            raise HandCommunicationError(f"short reply ({len(resp)} bytes)")
        if resp[1] & 0x80:
            raise HandCommunicationError(f"modbus exception 0x{resp[2]:02x}")
        if crc16_modbus(resp[:-2]) != resp[-2:]:
            raise HandCommunicationError("CRC mismatch in reply")
        nbytes = resp[2]
        return [
            int.from_bytes(resp[3 + 2 * i : 5 + 2 * i], "little") for i in range(nbytes // 2)
        ]

    def _write_modbus(self, addr: int, values: Sequence[int]) -> None:
        payload = b"".join(int(v).to_bytes(2, "little") for v in values)
        req = (
            bytes([self.hand_id, 0x10])
            + addr.to_bytes(2, "big")
            + len(values).to_bytes(2, "big")
            + bytes([len(payload)])
            + payload
        )
        req += crc16_modbus(req)
        resp = self._txn(req, 8)
        if len(resp) >= 2 and resp[1] & 0x80:
            raise HandCommunicationError(f"modbus exception 0x{resp[2]:02x}")

    def _read_legacy(self, addr: int, count: int) -> List[int]:
        nbytes = 2 * count
        body = [self.hand_id, 0x04, 0x11, addr & 0xFF, (addr >> 8) & 0xFF, nbytes]
        req = bytes([0xEB, 0x90] + body + [sum(body) & 0xFF])
        resp = self._txn(req, 7 + nbytes)
        if len(resp) < 7 or resp[0] != 0xEB or resp[1] != 0x90:
            raise HandCommunicationError(f"bad legacy frame header: {resp[:4].hex(' ')}")
        data = resp[7 : 7 + nbytes]
        return [int.from_bytes(data[2 * i : 2 * i + 2], "little") for i in range(len(data) // 2)]

    def _write_legacy(self, addr: int, values: Sequence[int]) -> None:
        payload = [b for v in values for b in int(v).to_bytes(2, "little")]
        body = [self.hand_id, len(payload) + 3, 0x12, addr & 0xFF, (addr >> 8) & 0xFF] + payload
        req = bytes([0xEB, 0x90] + body + [sum(body) & 0xFF])
        # The hand acknowledges writes; drain the ack but tolerate silence.
        try:
            self._txn(req, 9)
        except HandCommunicationError:
            pass

    # -- register access ---------------------------------------------------
    def read_registers(self, addr: int, count: int) -> List[int]:
        with self._lock:
            if self.protocol == "modbus":
                return self._read_modbus(addr, count)
            return self._read_legacy(addr, count)

    def write_registers(self, addr: int, values: Sequence[int]) -> None:
        with self._lock:
            if self.protocol == "modbus":
                self._write_modbus(addr, values)
            else:
                self._write_legacy(addr, values)

    # -- hand-level operations --------------------------------------------
    def read_angles(self) -> List[int]:
        """Return the six measured DOF angles in DOF_ORDER."""
        return self.read_registers(REG_ANGLE_ACT, 6)

    def read_forces(self) -> List[int]:
        return self.read_registers(REG_FORCE_ACT, 6)

    def read_errors(self) -> List[int]:
        return self.read_registers(REG_ERROR, 3)

    def write_angles(self, angles: Sequence[int]) -> None:
        """Command the six DOF angles in DOF_ORDER (0 closed .. 1000 open)."""
        if len(angles) != 6:
            raise ValueError(f"expected 6 angles, got {len(angles)}")
        clamped = [max(ANGLE_MIN, min(ANGLE_MAX, int(a))) for a in angles]
        self.write_registers(REG_ANGLE_SET, clamped)

    def write_speeds(self, speeds: Sequence[int]) -> None:
        self.write_registers(REG_SPEED_SET, [max(0, min(1000, int(s))) for s in speeds])

    def write_forces(self, forces: Sequence[int]) -> None:
        self.write_registers(REG_FORCE_SET, [max(0, min(1000, int(f))) for f in forces])

    def ping(self) -> Optional[int]:
        """Return the hand's configured ID, or None if it does not answer."""
        try:
            return self.read_registers(REG_HAND_ID, 1)[0]
        except HandCommunicationError:
            return None


class MockTransport(HandTransport):
    """In-memory stand-in that models the hand's first-order response.

    Lets the whole ROS pipeline be exercised without hardware: commanded
    angles are approached at a finite rate rather than applied instantly, so
    consumers see the same settling behaviour they would on a real hand.
    """

    def __init__(self, slew_per_sec: float = 1200.0, **kwargs) -> None:
        kwargs.setdefault("port", "<mock>")
        super().__init__(**kwargs)
        self._angles = [float(ANGLE_MAX)] * 6
        self._targets = [float(ANGLE_MAX)] * 6
        self._slew = slew_per_sec
        self._last = time.monotonic()
        self._open = False

    def connect(self) -> None:
        self._open = True
        self._last = time.monotonic()

    def close(self) -> None:
        self._open = False

    @property
    def connected(self) -> bool:
        return self._open

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        step = self._slew * dt
        for i, target in enumerate(self._targets):
            delta = target - self._angles[i]
            self._angles[i] += max(-step, min(step, delta))

    def read_registers(self, addr: int, count: int) -> List[int]:
        with self._lock:
            self._integrate()
            if addr == REG_ANGLE_ACT or addr == REG_POS_ACT:
                return [int(round(a)) for a in self._angles[:count]]
            if addr == REG_HAND_ID:
                return [self.hand_id]
            return [0] * count

    def write_registers(self, addr: int, values: Sequence[int]) -> None:
        with self._lock:
            self._integrate()
            if addr == REG_ANGLE_SET:
                for i, v in enumerate(values[:6]):
                    self._targets[i] = float(max(ANGLE_MIN, min(ANGLE_MAX, int(v))))
