"""The compliance law: a finger that gives when its fingertip is pushed.

Exact time steps, because a rate limit is only testable against one. The other
half of compliant mode -- that the driver applies the yield as an offset on top
of the commanded pose and never as a command of its own -- is in
``test_compliance_node.py``, which needs rclpy and this does not.
"""


import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspire_hand_driver import kinematics as kin  # noqa: E402
from inspire_hand_driver.compliance import (  # noqa: E402
    FingerSpring,
    SpringGains,
    channel_mask,
    describe,
)

INDEX = kin.dof_index("index_proximal_joint")
PINKY = kin.dof_index("pinky_proximal_joint")
THUMB_ROTATION = kin.dof_index("thumb_proximal_yaw_joint")

#: A push worth 160 counts of yield at :data:`GAIN`: 400 g at the tip, 80 of
#: which the deadband eats.
PUSH = 400
#: The gain these tests do their arithmetic in. Deliberately not the default:
#: the default is a tuning choice that moves with the hand it was measured on,
#: and the law is not supposed to move with it. One test below pins the
#: default itself, so changing it stays a deliberate act.
GAIN = 0.5


def forces(by_index=None):
    """A six-DOF force reading, in grams, with the named DOF pushed on."""
    out = [0] * 6
    for index, value in (by_index or {}).items():
        out[int(index)] = value
    return out


def spring(channels=None, **gains):
    gains.setdefault("counts_per_gram", GAIN)
    s = FingerSpring(SpringGains(**gains), channels)
    s.engage()
    return s


# -- the law ------------------------------------------------------------------

def test_a_push_below_the_deadband_moves_nothing():
    s = spring(deadband=80.0)
    for _ in range(50):
        assert s.update(forces({INDEX: 79}), 0.02) == [0] * 6


def test_a_push_above_the_deadband_opens_that_finger_and_only_that_finger():
    s = spring()
    for _ in range(100):
        counts = s.update(forces({INDEX: PUSH}), 0.02)
    assert counts[INDEX] == 160, "(400 - 80) g at 0.5 counts/g"
    assert [c for i, c in enumerate(counts) if i != INDEX] == [0] * 5


def test_the_shipped_default_gain_is_what_the_bench_settled_on():
    """0.6 counts/g, chosen against this hand. Changing it is a claim about
    hardware rather than a refactor, so it is pinned here: the rest of these
    tests do their arithmetic in GAIN and would not notice it moving."""
    assert SpringGains().counts_per_gram == 0.6


def test_the_yield_is_rate_limited_on_the_way_out():
    # The rate limit, not the gain, is what the first seconds are governed by:
    # 400 counts/s over a 20 ms cycle is 8 counts, whatever the push.
    s = spring(yield_rate=400.0)
    assert s.update(forces({INDEX: 1000}), 0.02)[INDEX] == 8
    assert s.update(forces({INDEX: 1000}), 0.02)[INDEX] == 16


def test_the_yield_is_capped_however_hard_the_push():
    s = spring(max_yield=100.0)
    for _ in range(500):
        counts = s.update(forces({INDEX: 1000}), 0.02)
    assert counts[INDEX] == 100


def test_releasing_the_fingertip_closes_the_finger_back_at_the_return_rate():
    s = spring(yield_rate=1e6, return_rate=400.0)
    assert s.update(forces({INDEX: PUSH}), 0.02)[INDEX] == 160
    assert s.update(forces(), 0.02)[INDEX] == 152, "8 counts back per 20 ms cycle"
    for _ in range(100):
        counts = s.update(forces(), 0.02)
    assert counts == [0] * 6


def test_a_zero_return_rate_is_a_clutch_that_still_lets_go_of_the_mode():
    s = spring(yield_rate=1e6, return_rate=0.0)
    s.update(forces({INDEX: PUSH}), 0.02)
    for _ in range(100):
        counts = s.update(forces(), 0.02)
    assert counts[INDEX] == 160, "what the push opened stays open"
    # Leaving compliant mode must never be able to strand a finger, so the
    # ramp out borrows the yield rate rather than honouring the hold.
    s.release()
    assert s.update(forces(), 0.02) == [0] * 6
    assert not s.active


def test_a_finger_never_closes_on_its_own():
    # The fingertip sensor reads compression of the pad and nothing else, so
    # the law has one direction. A yield below zero would be a finger tightening
    # its grip because of a sensor reading, which nothing here may ever do.
    s = spring()
    for force in (0, 1, 80, 1000, 0):
        assert min(s.update(forces({INDEX: force}), 0.02)) >= 0


def test_an_unloaded_fingertip_reading_below_zero_is_not_a_push():
    # The sensor sits a little under its own zero when nothing touches it --
    # this rig's six read -2 to -90 g at rest -- so the law has to take a
    # negative reading as "no push" and not as a reason to do anything.
    s = spring()
    for _ in range(50):
        assert s.update(forces({INDEX: -90}), 0.02) == [0] * 6


def test_a_shifted_zero_is_tared_away_rather_than_held_as_a_push():
    # The measured case: the index pad rested at -11 g, was pushed to 2511 g,
    # and then sat at +219 g untouched. Against a 30 g deadband that is a
    # standing push, and the finger never comes home.
    s = spring(deadband=30.0, counts_per_gram=0.18, yield_rate=1e6)
    for _ in range(10):
        counts = s.update(forces({INDEX: 219}), 0.02)
    assert counts[INDEX] > 0, "untared, the shifted zero reads as a real push"

    s.tare(forces({INDEX: 219}))
    for _ in range(200):
        counts = s.update(forces({INDEX: 219}), 0.02)
    assert counts == [0] * 6, "tared, the same reading means nothing is touching it"


def test_a_push_on_top_of_a_tared_zero_still_gives_proportionally():
    s = spring(deadband=30.0, counts_per_gram=0.18, yield_rate=1e6)
    s.tare(forces({INDEX: 219}))
    counts = s.update(forces({INDEX: 219 + 530}), 0.02)
    assert counts[INDEX] == 90, "(530 - 30) g at 0.18 counts/g"


def test_taring_a_resting_offset_below_zero_works_the_same_way():
    # The unloaded state is a small negative, not nought, on every channel.
    s = spring(deadband=30.0, counts_per_gram=0.5, yield_rate=1e6)
    s.tare(forces({INDEX: -11}))
    assert s.update(forces({INDEX: -11}), 0.02) == [0] * 6
    assert s.update(forces({INDEX: -11 + 130}), 0.02)[INDEX] == 50


def test_the_zero_is_reported_so_a_moved_one_can_be_seen():
    s = spring()
    assert s.zeros == [0.0] * 6
    assert s.tare([1, 2, 3, 4, 5, 6]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert s.zeros == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_channels_left_out_do_not_give():
    s = spring(channels=channel_mask([PINKY]))
    counts = s.update(forces({INDEX: 1000, PINKY: 1000, THUMB_ROTATION: 1000}), 1.0)
    assert counts[PINKY] > 0
    assert counts[INDEX] == 0 and counts[THUMB_ROTATION] == 0


def test_a_spring_that_is_not_engaged_ignores_force():
    s = FingerSpring()
    assert s.update(forces({INDEX: 1000}), 1.0) == [0] * 6
    assert not s.active


def test_active_stays_true_until_the_yield_is_back_out():
    # What the driver polls force on: releasing is not instant, and the cycles
    # that ramp the yield out still have to be driven.
    s = spring(yield_rate=1e6, return_rate=100.0)
    s.update(forces({INDEX: PUSH}), 0.02)
    s.release()
    assert s.active and not s.engaged
    for _ in range(200):
        s.update(forces(), 0.02)
    assert not s.active


@pytest.mark.parametrize(
    "gains",
    [
        {"deadband": -1.0},
        {"counts_per_gram": -0.1},
        {"max_yield": -5.0},
        {"return_rate": -1.0},
        {"yield_rate": 0.0},
        {"yield_rate": -1.0},
    ],
)
def test_gains_that_cannot_mean_anything_are_refused(gains):
    with pytest.raises(ValueError):
        SpringGains(**gains)


def test_retuning_keeps_the_yield_already_applied():
    # Re-deriving it would step every compliant finger at once, under the hand
    # of whoever is tuning by feel. The rate limits govern the walk to the new
    # gain's target as they govern everything else, so softening a spring that
    # is currently holding someone's finger open lets it down rather than
    # dropping it.
    s = spring(yield_rate=1e6, return_rate=200.0)
    s.update(forces({INDEX: PUSH}), 0.02)
    s.retune(SpringGains(counts_per_gram=0.1, yield_rate=1e6, return_rate=200.0))
    assert s.counts[INDEX] == 160, "not re-derived on the spot"
    assert s.update(forces({INDEX: PUSH}), 0.02)[INDEX] == 156, "4 counts per cycle"
    for _ in range(200):
        counts = s.update(forces({INDEX: PUSH}), 0.02)
    assert counts[INDEX] == 32, "and the new gain's target in the end"


def test_describe_names_every_gain():
    text = describe(SpringGains())
    for token in ("deadband", "gain", "max_yield", "rates"):
        assert token in text
