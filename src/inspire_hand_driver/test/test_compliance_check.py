"""The bench check: its live readout, and the gains it recommends.

The readout is here because a display that flickers is not a cosmetic problem
in a tool whose whole job is to be watched while someone pushes a finger; the
first version redrew on every message behind a blinking cursor.

The recommendation is here because the first version of that was wrong in a
way that mattered more. It suggested a deadband halfway between rest and the
hardest push measured -- 409 g against a real hand -- which would have spent
most of the signal before a finger moved at all. The numbers in
``test_suggests_a_usable_tune_for_the_real_hand`` are what the rig actually
measured, so a regression has something real to fail against.
"""

import io
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("rclpy")  # the module imports rclpy, the display does not

from inspire_hand_driver.compliance_check import Block, bar, recommend  # noqa: E402

CURSOR_HIDE = "\x1b[?25l"
CURSOR_SHOW = "\x1b[?25h"
CLEAR_LINE = "\x1b[2K"


class FakeTerminal(io.StringIO):
    def __init__(self, tty=True):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.fixture
def terminal(monkeypatch):
    def make(tty=True):
        fake = FakeTerminal(tty)
        monkeypatch.setattr(sys, "stdout", fake)
        return fake

    return make


# -- the bar ------------------------------------------------------------------

def test_the_bar_fills_in_proportion():
    assert bar(0, 100, width=10) == "." * 10
    assert bar(50, 100, width=10) == "#" * 5 + "." * 5
    assert bar(100, 100, width=10) == "#" * 10


def test_the_bar_clamps_rather_than_overflowing():
    # Force can read past the scale the bar was ranged for; a wider line would
    # wrap and take the in-place redraw with it.
    assert len(bar(5000, 100, width=10)) == 10
    assert len(bar(-5, 100, width=10)) == 10


def test_the_peak_is_marked_where_the_fill_no_longer_reaches():
    marked = bar(20, 100, width=10, peak=80)
    assert marked.count("|") == 1
    assert marked.index("|") == 7
    # Once the fill has caught up with the peak there is nothing to mark.
    assert "|" not in bar(80, 100, width=10, peak=80)


def test_a_zero_scale_does_not_divide_by_it():
    assert bar(0, 0, width=8) == "." * 8


# -- the block ----------------------------------------------------------------

def test_the_first_frame_hides_the_cursor_and_clears_each_line(terminal):
    out = terminal()
    block = Block(interval=0.0)
    block.update(["one", "two"])
    text = out.getvalue()
    assert text.startswith(CURSOR_HIDE)
    assert text.count(CLEAR_LINE) == 2
    assert "\x1b[2A" not in text, "nothing to move up over yet"


def test_later_frames_redraw_in_place_instead_of_scrolling(terminal):
    out = terminal()
    block = Block(interval=0.0)
    block.update(["one", "two", "three"])
    out.truncate(0), out.seek(0)
    block.update(["four", "five", "six"])
    text = out.getvalue()
    assert text.startswith("\x1b[3A"), "back up over exactly the rows drawn"
    assert text.count(CLEAR_LINE) == 3, "and clear each, so no tail survives"
    assert text.count("\n") == 3


def test_frames_are_throttled_to_a_rate_an_eye_can_follow(terminal):
    out = terminal()
    block = Block(interval=10.0)
    block.update(["first"])
    out.truncate(0), out.seek(0)
    for _ in range(50):
        block.update(["flood"])
    assert out.getvalue() == "", "50 messages, no redraw"
    block.update(["forced"], force=True)
    assert "forced" in out.getvalue()


def test_the_cursor_comes_back_when_the_block_is_done(terminal):
    out = terminal()
    block = Block(interval=0.0)
    block.update(["one"])
    block.finish()
    assert out.getvalue().endswith(CURSOR_SHOW)


def test_a_piped_run_gets_plain_text_and_no_escapes(terminal):
    out = terminal(tty=False)
    block = Block(interval=0.0)
    block.update(["one", "two"], force=True)
    block.finish()
    text = out.getvalue()
    assert "\x1b" not in text
    assert text == "one\ntwo\n"


def test_a_piped_run_does_not_print_a_frame_per_message(terminal):
    out = terminal(tty=False)
    block = Block(interval=0.0)
    block.update(["one"], force=True)
    out.truncate(0), out.seek(0)
    for _ in range(50):
        block.update(["two"])
    assert out.getvalue() == "", "logs get a frame every couple of seconds, not 50"


# -- what it recommends -------------------------------------------------------

#: Measured on the rig on 2026-09-16, in DOF order. The resting band is what
#: each channel wandered by untouched; the peak is a firm push on each pad.
REAL_BAND = [16.0, 16.0, 33.0, 11.0, 2.0, 243.0]
REAL_PEAK = [725.0, 820.0, 660.0, 507.0, 59.0, 267.0]


def advice(band, peak, capsys):
    recommend(band, peak)
    return capsys.readouterr().out


def test_suggests_a_usable_tune_for_the_real_hand(capsys):
    out = advice(REAL_BAND, REAL_PEAK, capsys)
    line = [l for l in out.splitlines() if "--deadband" in l][0]
    deadband = float(line.split("--deadband")[1].split()[0])
    gain = float(line.split("--gain")[1].split()[0])

    # Clear of the 33 g resting band, nowhere near the 409 g the old formula
    # produced, and leaving most of a 500-800 g push as usable signal.
    assert 40 <= deadband <= 120, out
    assert deadband < min(REAL_PEAK[:4]) / 3, "most of the push must survive it"
    # A moderate push should ask for a real but not saturating opening.
    moderate = 0.5 * REAL_PEAK[3]
    assert 80 <= (moderate - deadband) * gain <= 300, out


def test_it_names_a_pad_too_weak_to_give(capsys):
    # thumb_bend peaked at 59 g where the fingers reached 500-800.
    out = advice(REAL_BAND, REAL_PEAK, capsys)
    assert "Too weak to give: thumb_bend" in out
    assert "pinky" not in out.split("Too weak to give:")[1]


def test_the_deadband_clears_the_resting_band_it_was_given(capsys):
    noisy = [90.0] * 6
    out = advice(noisy, [900.0] * 6, capsys)
    deadband = float(out.split("--deadband")[1].split()[0])
    assert deadband > 90.0, "a deadband inside the resting band opens the hand by itself"


def test_thumb_rotations_wild_resting_band_does_not_set_the_deadband(capsys):
    # It rests near -80 and swings to -243 under load, but it carries no pad
    # and is not in the default channels, so it must not drag the deadband up.
    out = advice(REAL_BAND, REAL_PEAK, capsys)
    deadband = float(out.split("--deadband")[1].split()[0])
    assert deadband < 243.0, out


def test_nothing_pushed_says_so_rather_than_recommending_a_number(capsys):
    out = advice([5.0] * 6, [0.0] * 6, capsys)
    assert "--deadband" not in out
    assert "Nothing read a push" in out


def test_pushes_too_faint_to_use_say_so_too(capsys):
    out = advice([10.0] * 6, [45.0] * 6, capsys)
    assert "--deadband" not in out
    assert "none of them would give" in out


# -- the tool and the controller must agree on what "touched" means -----------

def test_it_measures_force_against_the_drivers_own_zero():
    # The bug this pins: the tool compared raw force against the deadband while
    # the driver compared tared force. With a pad whose zero had walked to
    # 276 g, raw force never fell below a 100 g deadband, so the release
    # detector could never arm and a finger that plainly came back was failed.
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    from inspire_hand_driver.compliance_check import ComplianceCheck

    rclpy.init()
    try:
        node = ComplianceCheck("/inspire_hand")
        node.force = [0.0, 0.0, 0.0, 276.0, 0.0, 0.0]
        assert node.touch(3) == 276.0, "no zero reported yet, so raw"

        node._on_diagnostics(
            DiagnosticArray(
                status=[
                    DiagnosticStatus(
                        hardware_id="/dev/ttyUSB0#1/4",
                        values=[KeyValue(key="force_zero", value="209")],
                    )
                ]
            )
        )
        assert node.zero[3] == 209.0
        assert node.touch(3) == 67.0, "which is under a 100 g deadband, as the driver sees it"
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_it_will_not_start_before_a_force_message_has_landed():
    # The bug this pins: force started as six zeros, the baseline began
    # sampling immediately, and a pad sitting still at 278 g was reported as a
    # 278 g resting band -- 0 to 278 -- because the placeholder was sampled as
    # a reading. A band that size refuses the run outright.
    import rclpy

    from inspire_hand_driver.compliance_check import ComplianceCheck

    rclpy.init()
    try:
        node = ComplianceCheck("/inspire_hand")
        node._compliance.service_is_ready = lambda: True
        node.state = [0.5] * 6
        assert not node.ready, "state and a service are not enough to start measuring"
        node.force = [0.0, 0.0, 0.0, 278.0, 0.0, 0.0]
        assert node.ready
        node.destroy_node()
    finally:
        rclpy.shutdown()
