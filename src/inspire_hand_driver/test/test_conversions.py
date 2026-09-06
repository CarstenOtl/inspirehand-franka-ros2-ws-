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
