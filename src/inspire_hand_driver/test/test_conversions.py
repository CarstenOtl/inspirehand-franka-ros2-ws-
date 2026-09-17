"""Open-ratio conversion tests.

An open ratio of 1.0 means a fully open hand and maps to register angle 1000.
Getting this backwards would close the hand on a command to open it, which is
the failure that breaks fingers and whatever they are holding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver.protocol import ANGLE_MAX  # noqa: E402


def angle_to_open_ratio(angle):
    return max(0.0, min(1.0, float(angle) / float(ANGLE_MAX)))


def open_ratio_to_angle(ratio):
    return int(round(max(0.0, min(1.0, float(ratio))) * ANGLE_MAX))


def test_open_ratio_one_is_fully_open():
    assert open_ratio_to_angle(1.0) == ANGLE_MAX


def test_open_ratio_zero_is_fully_closed():
    assert open_ratio_to_angle(0.0) == 0


def test_conversions_round_trip():
    for angle in (0, 1, 250, 500, 999, ANGLE_MAX):
        assert open_ratio_to_angle(angle_to_open_ratio(angle)) == angle


def test_out_of_range_values_are_clamped():
    assert open_ratio_to_angle(2.0) == ANGLE_MAX
    assert open_ratio_to_angle(-1.0) == 0
    assert angle_to_open_ratio(5000) == 1.0
    assert angle_to_open_ratio(-5) == 0.0


# -- signed force -------------------------------------------------------------
# FORCE_ACT is signed and was read unsigned until a real hand said otherwise:
# its six fingertips at rest read -2, -12, -26, -11, +1 and -90 g, which
# unsigned are 65534, 65524, 65510, 65525, 1 and 65446. Anything comparing
# force against a threshold sees a hand under enormous load instead of one
# holding nothing, so compliant mode would have opened every finger the moment
# it was switched on.

def test_an_unloaded_fingertip_reads_as_the_small_negative_it_is():
    from inspire_hand_driver.protocol import to_signed16

    assert [to_signed16(v) for v in (65534, 65524, 65510, 65525, 1, 65446)] == [
        -2, -12, -26, -11, 1, -90,
    ]


def test_the_signed_boundary_is_where_the_manual_puts_it():
    from inspire_hand_driver.protocol import to_signed16

    assert to_signed16(0) == 0
    assert to_signed16(1000) == 1000, "the working range is untouched"
    assert to_signed16(0x7FFF) == 32767
    assert to_signed16(0x8000) == -32768
    assert to_signed16(0xFFFF) == -1


def test_read_forces_signs_what_the_wire_carried():
    from inspire_hand_driver.protocol import HandTransport, REG_FORCE_ACT

    class Wire(HandTransport):
        def read_registers(self, addr, count):
            assert (addr, count) == (REG_FORCE_ACT, 6)
            return [65534, 65524, 65510, 65525, 1, 65446]

    assert Wire().read_forces() == [-2, -12, -26, -11, 1, -90]


def test_the_force_threshold_readback_stays_unsigned():
    # FORCE_SET is a commanded 0..1000 threshold, not a measurement; signing it
    # would turn a threshold the driver itself wrote into a negative number.
    from inspire_hand_driver.protocol import HandTransport, REG_FORCE_SET

    class Wire(HandTransport):
        def read_registers(self, addr, count):
            assert addr == REG_FORCE_SET
            return [500] * 6

    assert Wire().read_force_thresholds() == [500] * 6
