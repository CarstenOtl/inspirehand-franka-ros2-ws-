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
from typing import List, NamedTuple, Optional, Sequence

# Holding-register addresses. Each per-finger block is 6 contiguous registers
# in DOF_ORDER. Transcribed from the RH56 series user manual.
#
# The manual's addresses are *byte* addresses. A "short" register such as
# ANGLE_SET occupies two of them, so a six-DOF block is 12 addresses, and one
# Modbus register (16 bits) covers two addresses. The single-byte registers --
# CLEAR_ERROR, ERROR, STATUS, TEMP -- therefore share Modbus registers in
# pairs, with the lower address in the low byte. See :func:`unpack_bytes` and
# :meth:`HandTransport.clear_errors` for the two places that matters.
REG_HAND_ID = 1000
REG_CLEAR_ERROR = 1004
REG_SAVE_FLASH = 1005
REG_CURRENT_LIMIT = 1020
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

# REG_ERROR: one byte per DOF, a bit set per fault (manual 2.4.18).
ERROR_LOCKED_ROTOR = 0x01
ERROR_OVER_TEMPERATURE = 0x02
ERROR_OVER_CURRENT = 0x04
ERROR_MOTOR_ABNORMAL = 0x08
ERROR_COMMUNICATION = 0x10
ERROR_NAMES = {
    ERROR_LOCKED_ROTOR: "locked-rotor",
    ERROR_OVER_TEMPERATURE: "over-temperature",
    ERROR_OVER_CURRENT: "over-current",
    ERROR_MOTOR_ABNORMAL: "motor-abnormal",
    ERROR_COMMUNICATION: "communication",
}
# What a CLEAR_ERROR write actually clears. Over-temperature is not on the
# list: the manual says it clears itself once the actuator has cooled.
ERROR_CLEARABLE = (
    ERROR_LOCKED_ROTOR | ERROR_OVER_CURRENT | ERROR_MOTOR_ABNORMAL | ERROR_COMMUNICATION
)

# REG_STATUS: one byte per DOF (manual 2.4.19). 4 is not assigned.
STATUS_OPENING = 0
STATUS_CLOSING = 1
STATUS_AT_TARGET = 2
STATUS_AT_FORCE = 3
STATUS_CURRENT_PROTECTION = 5
STATUS_LOCKED_ROTOR = 6
STATUS_FAULT = 7
STATUS_NAMES = {
    STATUS_OPENING: "opening",
    STATUS_CLOSING: "closing",
    STATUS_AT_TARGET: "at target",
    STATUS_AT_FORCE: "stopped at force threshold",
    STATUS_CURRENT_PROTECTION: "stopped by current protection",
    STATUS_LOCKED_ROTOR: "stopped, locked rotor",
    STATUS_FAULT: "stopped, actuator fault",
}
# The states in which the firmware has given up on a DOF rather than parked
# it. A DOF here does not respond to new targets until the error is cleared
# (or the hand is power-cycled), which is exactly what the driver's stall
# guard exists to handle. Stopping at the force threshold is *not* one of
# these: that is the hand doing what it was asked.
STATUS_STALLED = frozenset(
    (STATUS_CURRENT_PROTECTION, STATUS_LOCKED_ROTOR, STATUS_FAULT)
)


def describe_error(bits: int) -> str:
    """Name every fault bit set in one DOF's ERROR byte, or ``"none"``."""
    names = [name for bit, name in ERROR_NAMES.items() if bits & bit]
    if bits & ~sum(ERROR_NAMES):
        names.append(f"unknown(0x{bits & ~sum(ERROR_NAMES):02x})")
    return "+".join(names) if names else "none"


def describe_status(code: int) -> str:
    return STATUS_NAMES.get(code, f"unknown({code})")


def unpack_bytes(words: Sequence[int]) -> List[int]:
    """Split 16-bit register values into the byte registers they carry.

    The lower address is the low byte, which is how the vendor's own Modbus
    example unpacks ERROR/STATUS/TEMP and how the legacy framing lays the
    bytes on the wire.
    """
    out: List[int] = []
    for word in words:
        out.append(int(word) & 0xFF)
        out.append((int(word) >> 8) & 0xFF)
    return out


def pack_bytes(values: Sequence[int]) -> List[int]:
    """Inverse of :func:`unpack_bytes`; a trailing odd byte is zero-padded."""
    values = list(values) + ([0] if len(values) % 2 else [])
    return [(values[i] & 0xFF) | ((values[i + 1] & 0xFF) << 8) for i in range(0, len(values), 2)]


class HandHealth(NamedTuple):
    """The three byte-per-DOF status blocks, each in DOF_ORDER."""

    errors: List[int]
    status: List[int]
    temperatures: List[int]

    def stalled(self) -> List[int]:
        """Indices of the DOF the firmware has stopped on a fault."""
        return [
            i
            for i in range(6)
            if self.status[i] in STATUS_STALLED or self.errors[i] & ERROR_CLEARABLE
        ]


class HandCommunicationError(RuntimeError):
    """Raised when the hand does not answer or answers with a bad frame."""


class HandProtocolError(HandCommunicationError):
    """The hand answered, but refused the request (a Modbus exception reply).

    Distinguished from silence because it says something about the request
    rather than the wiring: the firmware does not support that access.
    """


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
        # ERROR, STATUS and TEMP are adjacent, so one read normally fetches all
        # three. A firmware that refuses a read spanning blocks flips this and
        # the driver pays three transactions instead, permanently.
        self._split_health_reads = False

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
            raise HandProtocolError(f"modbus exception 0x{resp[2]:02x}")
        if crc16_modbus(resp[:-2]) != resp[-2:]:
            raise HandCommunicationError("CRC mismatch in reply")
        nbytes = resp[2]
        return [
            int.from_bytes(resp[3 + 2 * i : 5 + 2 * i], "big") for i in range(nbytes // 2)
        ]

    def _write_modbus(self, addr: int, values: Sequence[int]) -> None:
        payload = b"".join(int(v).to_bytes(2, "big") for v in values)
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
            raise HandProtocolError(f"modbus exception 0x{resp[2]:02x}")

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
        """Return the six ERROR bytes in DOF_ORDER."""
        return unpack_bytes(self.read_registers(REG_ERROR, 3))

    def read_health(self) -> HandHealth:
        """Return ERROR, STATUS and TEMP for all six DOF.

        One transaction where the firmware allows it (the three blocks are 18
        contiguous bytes), three otherwise.
        """
        if not self._split_health_reads:
            try:
                raw = unpack_bytes(self.read_registers(REG_ERROR, 9))
                return HandHealth(raw[0:6], raw[6:12], raw[12:18])
            except HandProtocolError:
                self._split_health_reads = True
        return HandHealth(
            unpack_bytes(self.read_registers(REG_ERROR, 3)),
            unpack_bytes(self.read_registers(REG_STATUS, 3)),
            unpack_bytes(self.read_registers(REG_TEMP, 3)),
        )

    def read_force_thresholds(self) -> List[int]:
        """Read back FORCE_SET, the per-DOF force threshold currently in effect."""
        return self.read_registers(REG_FORCE_SET, 6)

    def clear_errors(self) -> None:
        """Clear latched actuator errors (locked-rotor, over-current, ...).

        CLEAR_ERROR at 1004 shares its Modbus register with SAVE at 1005, and
        SAVE=1 commits every volatile parameter to flash. The value written
        here must therefore be exactly 1 -- low byte set, high byte clear --
        and nothing in this driver may ever write anything else to 1004.
        """
        self.write_registers(REG_CLEAR_ERROR, [1])

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

    It also models the two ways a finger stops short of its target, because
    the driver's stall guard has to be testable without jamming a real hand:

    * ``obstacles[i]`` is an angle the DOF cannot close past -- an object in
      the grip. On contact, with a force threshold set (FORCE_SET > 0), the
      finger stops there cleanly and reports ``STATUS_AT_FORCE``.
    * Without a threshold it keeps pushing, and after ``stall_after_sec`` the
      firmware's locked-rotor protection latches: the ERROR bit is set, the
      status reads ``STATUS_LOCKED_ROTOR``, and the DOF ignores every target
      until CLEAR_ERROR is written. That is the "finger dies until reboot"
      behaviour seen on the bench.
    """

    def __init__(
        self, slew_per_sec: float = 1200.0, stall_after_sec: float = 0.2, **kwargs
    ) -> None:
        kwargs.setdefault("port", "<mock>")
        super().__init__(**kwargs)
        self._angles = [float(ANGLE_MAX)] * 6
        self._targets = [float(ANGLE_MAX)] * 6
        self._slew = slew_per_sec
        self._last = time.monotonic()
        self._open = False
        self.stall_after_sec = stall_after_sec
        #: Per DOF: an angle the finger cannot close past, or None for free travel.
        self.obstacles: List[Optional[int]] = [None] * 6
        self.force_thresholds = [0] * 6
        self.speeds = [0] * 6
        self.temperatures = [35] * 6
        self._errors = [0] * 6
        self._status = [STATUS_AT_TARGET] * 6
        self._force_act = [0] * 6
        self._current = [0] * 6
        self._pushing_since: List[Optional[float]] = [None] * 6
        #: Counts of the two writes to the 1004/1005 register pair, so a test
        #: can assert the driver clears errors and never commits to flash.
        self.clear_error_writes = 0
        self.flash_saves = 0

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
            if self._errors[i] & ERROR_CLEARABLE:
                # Latched: the firmware has stopped driving this DOF.
                self._status[i] = STATUS_LOCKED_ROTOR
                self._current[i] = 0
                continue
            obstacle = self.obstacles[i]
            blocked = obstacle is not None and target < obstacle
            reachable = float(obstacle) if blocked else target
            delta = reachable - self._angles[i]
            self._angles[i] += max(-step, min(step, delta))
            arrived = abs(reachable - self._angles[i]) < 0.5
            if not arrived:
                self._status[i] = STATUS_CLOSING if delta < 0 else STATUS_OPENING
                self._current[i] = 600
                self._force_act[i] = 0
                self._pushing_since[i] = None
            elif not blocked:
                self._status[i] = STATUS_AT_TARGET
                self._current[i] = 0
                self._force_act[i] = 0
                self._pushing_since[i] = None
            elif self.force_thresholds[i] > 0:
                # Contact, and a threshold to stop on: a clean force stop.
                self._status[i] = STATUS_AT_FORCE
                self._force_act[i] = self.force_thresholds[i]
                self._current[i] = 300
                self._pushing_since[i] = None
            else:
                # Contact with nothing to stop on: push until protection trips.
                self._status[i] = STATUS_CLOSING
                self._force_act[i] = 1000
                self._current[i] = 1500
                if self._pushing_since[i] is None:
                    self._pushing_since[i] = now
                elif now - self._pushing_since[i] >= self.stall_after_sec:
                    self._errors[i] |= ERROR_LOCKED_ROTOR
                    self._status[i] = STATUS_LOCKED_ROTOR
                    self._pushing_since[i] = None

    def _health_bytes(self) -> List[int]:
        return list(self._errors) + list(self._status) + list(self.temperatures)

    def read_registers(self, addr: int, count: int) -> List[int]:
        with self._lock:
            self._integrate()
            if addr == REG_ANGLE_ACT or addr == REG_POS_ACT:
                return [int(round(a)) for a in self._angles[:count]]
            if addr == REG_HAND_ID:
                return [self.hand_id]
            if addr == REG_FORCE_ACT:
                return list(self._force_act[:count])
            if addr == REG_CURRENT:
                return list(self._current[:count])
            if addr == REG_FORCE_SET:
                return list(self.force_thresholds[:count])
            if addr == REG_SPEED_SET:
                return list(self.speeds[:count])
            if REG_ERROR <= addr < REG_TEMP + 6:
                first = addr - REG_ERROR
                return pack_bytes(self._health_bytes()[first : first + 2 * count])
            return [0] * count

    def write_registers(self, addr: int, values: Sequence[int]) -> None:
        with self._lock:
            self._integrate()
            if addr == REG_ANGLE_SET:
                for i, v in enumerate(values[:6]):
                    self._targets[i] = float(max(ANGLE_MIN, min(ANGLE_MAX, int(v))))
            elif addr == REG_FORCE_SET:
                for i, v in enumerate(values[:6]):
                    self.force_thresholds[i] = int(v)
            elif addr == REG_SPEED_SET:
                for i, v in enumerate(values[:6]):
                    self.speeds[i] = int(v)
            elif addr == REG_CLEAR_ERROR:
                word = int(values[0])
                if word & 0xFF == 1:
                    self.clear_error_writes += 1
                    self._errors = [e & ~ERROR_CLEARABLE for e in self._errors]
                    for i in range(6):
                        if not self._errors[i]:
                            self._status[i] = STATUS_AT_TARGET
                if word >> 8:
                    self.flash_saves += 1
            elif addr == REG_SAVE_FLASH:
                self.flash_saves += 1

    # -- test hooks ------------------------------------------------------
    def latched_errors(self) -> List[int]:
        with self._lock:
            return list(self._errors)

    def status_codes(self) -> List[int]:
        with self._lock:
            self._integrate()
            return list(self._status)
