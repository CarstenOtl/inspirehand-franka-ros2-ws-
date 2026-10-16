"""The arithmetic and the display behind the velocity bench check.

The tool itself needs a hand to say anything, but what it concludes from a
recording does not, and that is the part that can be wrong quietly. These
tests feed it recordings whose true rate is known by construction -- including
one built the way the driver builds its reading, backward-differenced and
filtered -- and check that the figures it reports are the ones the recording
actually contains.
"""

import math
import sys
import termios
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from inspire_hand_driver import kinematics as kin
from inspire_hand_driver.velocity_check import (
    Sample,
    analyse_leg,
    best_lag_samples,
    central_difference,
    correlation,
    count_resolution,
    interval_stats,
    motion_window,
    rms,
    shifted_rms,
    signed_bar,
    trapezoid,
    verdict,
)

INDEX = kin.dof_index("index_proximal_joint")
DT = 0.02
FILTER_HZ = 10.0


def trapezoid_profile(count: int, peak: float, ramp: int) -> list:
    """A rate that ramps up, holds, and ramps down -- what a finger does."""
    rates = []
    for i in range(count):
        if i < ramp:
            rates.append(peak * (i + 1) / ramp)
        elif i >= count - ramp:
            rates.append(peak * (count - i) / ramp)
        else:
            rates.append(peak)
    return rates


def recording(
    peak: float = 1.8,
    moving: int = 40,
    still: int = 10,
    filter_hz: float = FILTER_HZ,
    quantise: bool = True,
    ceiling: float = None,
    ramp: int = 8,
) -> list:
    """A move recorded the way the driver reports one.

    Positions are integrated from a known rate profile and rounded to ANGLE
    counts, because the quantisation is the dominant error on any one sample
    and a test on smooth positions would not see it. Velocities are then the
    backward difference of exactly those positions, filtered with one pole --
    the driver's own arithmetic, so what the analysis extracts can be held
    against what was put in.
    """
    step = count_resolution(INDEX)
    rates = [0.0] * still + trapezoid_profile(moving, peak, ramp) + [0.0] * still
    positions, angle = [], 0.0
    for rate in rates:
        angle += rate * DT
        positions.append(round(angle / step) * step if quantise else angle)

    samples, filtered = [], 0.0
    tau = 0.0 if filter_hz <= 0 else 1.0 / (2.0 * math.pi * filter_hz)
    for i, position in enumerate(positions):
        raw = 0.0 if i == 0 else (position - positions[i - 1]) / DT
        if ceiling is not None:
            raw = max(-ceiling, min(ceiling, raw))
        if tau > 0.0:
            filtered += DT / (DT + tau) * (raw - filtered)
        else:
            filtered = raw
        velocity = [0.0] * len(kin.DOFS)
        velocity[INDEX] = filtered
        place = [0.0] * len(kin.DOFS)
        place[INDEX] = position
        samples.append(
            Sample(
                stamp=i * DT,
                wall=i * DT,
                position=tuple(place),
                velocity=tuple(velocity),
                has_velocity=True,
            )
        )
    return samples


# -- the estimators ----------------------------------------------------------
def test_a_straight_ramp_differences_to_its_own_slope():
    times = [i * DT for i in range(10)]
    values = [0.5 * t for t in times]
    assert central_difference(times, values) == pytest.approx([0.5] * 10)


def test_a_still_signal_differences_to_zero():
    times = [i * DT for i in range(5)]
    assert central_difference(times, [1.3] * 5) == pytest.approx([0.0] * 5)


def test_one_sample_has_no_slope_rather_than_an_exception():
    assert central_difference([0.0], [1.0]) == [0.0]


def test_a_repeated_stamp_is_not_an_infinite_slope():
    assert central_difference([0.0, 0.0], [0.0, 1.0]) == [0.0, 0.0]


def test_the_integral_of_a_constant_rate_is_rate_times_time():
    times = [i * DT for i in range(11)]
    assert trapezoid(times, [2.0] * 11) == pytest.approx(2.0 * 10 * DT)


def test_the_integral_of_a_ramp_is_its_triangle():
    times = [i * 0.1 for i in range(11)]
    assert trapezoid(times, [t for t in times]) == pytest.approx(0.5 * 1.0 * 1.0)


def test_correlation_is_one_against_itself_and_minus_one_against_its_negative():
    values = [0.0, 1.0, 2.0, 1.5, -1.0]
    assert correlation(values, values) == pytest.approx(1.0)
    assert correlation(values, [-v for v in values]) == pytest.approx(-1.0)


def test_a_flat_signal_correlates_with_nothing_rather_than_dividing_by_zero():
    assert correlation([1.0] * 5, [0.0, 1.0, 2.0, 3.0, 4.0]) == 0.0


def test_the_interval_statistics_are_of_the_gaps_not_the_stamps():
    mean, jitter, low, high = interval_stats([0.0, 0.02, 0.05, 0.07])
    assert mean == pytest.approx(0.07 / 3)
    assert (low, high) == pytest.approx((0.02, 0.03))
    assert jitter > 0.0


def test_one_angle_count_is_a_thousandth_of_the_range():
    dof = kin.DOFS[INDEX]
    assert count_resolution(INDEX) == pytest.approx((dof.upper - dof.lower) / 1000.0)


# -- finding the move, and the lag -------------------------------------------
def test_the_window_is_the_part_that_moved():
    rates = [0.0] * 5 + [1.0] * 10 + [0.0] * 5
    assert motion_window(rates) == (5, 14)


def test_a_recording_with_no_motion_has_no_window():
    assert motion_window([0.0] * 20) is None
    assert motion_window([]) is None


def test_a_known_shift_comes_back_as_that_shift():
    reference = [math.sin(i * 0.3) for i in range(60)]
    delayed = [0.0, 0.0, 0.0] + reference[:-3]
    assert best_lag_samples(reference, delayed) == pytest.approx(3.0, abs=0.2)


def test_an_unshifted_signal_has_no_lag():
    reference = [math.sin(i * 0.3) for i in range(60)]
    assert best_lag_samples(reference, reference) == pytest.approx(0.0, abs=0.05)


def test_a_fractional_lag_is_not_rounded_to_a_whole_sample():
    """Two signals half a sample apart must not both read as zero lag."""
    reference = [math.sin(i * 0.3) for i in range(80)]
    half = [math.sin((i - 0.5) * 0.3) for i in range(80)]
    assert best_lag_samples(reference, half) == pytest.approx(0.5, abs=0.15)


def test_too_short_to_judge_says_so():
    assert best_lag_samples([1.0, 2.0], [1.0, 2.0]) is None


# -- what the tool concludes from a recording --------------------------------
def test_a_recording_that_never_moved_is_reported_as_such():
    still = [
        Sample(i * DT, i * DT, (0.0,) * 6, (0.0,) * 6, True) for i in range(40)
    ]
    assert analyse_leg(INDEX, still) is None


def test_too_few_samples_to_judge_is_not_a_verdict():
    assert analyse_leg(INDEX, recording()[:4]) is None


def test_it_recovers_the_travel_and_the_peak_that_went_in():
    figures = analyse_leg(INDEX, recording(peak=1.8))
    travelled = sum(trapezoid_profile(40, 1.8, 8)) * DT
    assert figures["travel"] == pytest.approx(travelled, rel=0.02)
    # The reference is a centred difference of quantised positions, so it sees
    # the peak of the profile to within a count or so, not exactly.
    assert figures["reference_peak"] == pytest.approx(1.8, rel=0.1)
    assert figures["direction"] == 1.0


def test_the_window_excludes_the_standstill_either_side():
    samples = recording(moving=40, still=25)
    figures = analyse_leg(INDEX, samples)
    # The move itself plus the sample of margin each side, nowhere near the 90
    # that were recorded.
    assert 40 <= figures["samples"] <= 46


def test_the_integral_of_the_reported_rate_is_the_distance_travelled():
    figures = analyse_leg(INDEX, recording())
    assert figures["integral"] == pytest.approx(figures["travel"], rel=0.05)


def test_the_filters_lag_is_measured_not_assumed():
    """One pole at 10 Hz is 1/(2 pi 10) = 16 ms, on top of half a sample."""
    figures = analyse_leg(INDEX, recording(filter_hz=FILTER_HZ))
    expected = 1.0 / (2.0 * math.pi * FILTER_HZ) + 0.5 * DT
    assert figures["lag_s"] == pytest.approx(expected, abs=0.008)


def test_the_unfiltered_difference_lags_by_only_its_half_sample():
    figures = analyse_leg(INDEX, recording(filter_hz=0.0))
    assert figures["lag_s"] == pytest.approx(0.5 * DT, abs=0.006)


def test_quantisation_shows_up_as_exact_zeros_while_moving():
    """A rate below one count per sample reports in bursts, not continuously."""
    slow = count_resolution(INDEX) / DT * 0.4
    figures = analyse_leg(INDEX, recording(peak=slow, filter_hz=0.0, moving=60))
    assert figures["zero_fraction"] > 0.2
    # The pole fills the gaps in: what zeros are left are the margin samples
    # at the edges of the window, not holes in the middle of the motion.
    smooth = analyse_leg(INDEX, recording(peak=slow, filter_hz=10.0, moving=60))
    assert smooth["zero_fraction"] < 0.25 * figures["zero_fraction"]


def test_a_clamped_recording_is_counted_as_clamped():
    ceiling = 0.9
    figures = analyse_leg(
        INDEX, recording(peak=1.8, filter_hz=0.0, ceiling=ceiling), ceiling=ceiling
    )
    assert figures["clamped"] > 0
    assert abs(figures["driver_peak"]) <= ceiling + 1e-9
    # And the integral is then short of the travel, which is the point of
    # measuring it: a clamp that bites is motion the consumer never sees.
    assert abs(figures["integral"]) < abs(figures["travel"])


def test_an_unclamped_recording_counts_none():
    assert analyse_leg(INDEX, recording(peak=0.4), ceiling=1.84)["clamped"] == 0


def test_the_resolution_is_one_count_per_sample_interval():
    figures = analyse_leg(INDEX, recording())
    assert figures["resolution"] == pytest.approx(count_resolution(INDEX) / DT, rel=0.05)


# -- the verdict -------------------------------------------------------------
def test_a_faithful_recording_passes(capsys):
    legs = [(label, analyse_leg(INDEX, recording())) for label in ("closing", "opening")]
    assert verdict(legs, [0.0] * 6) == 0
    assert "PASS" in capsys.readouterr().out


def test_a_leg_that_never_moved_fails(capsys):
    assert verdict([("closing", None)], [0.0] * 6) == 1
    assert "never moved" in capsys.readouterr().out


def test_a_constant_rate_move_is_not_failed_for_having_no_variance(capsys):
    """r is near zero on a plateau however faithful the reading is.

    The mock hand slews at a fixed rate, so this is not a hypothetical: the
    first run of this tool against it failed both legs on a correlation of
    0.7 while every other figure agreed to within 1%.
    """
    samples = recording(peak=1.8, moving=40, ramp=1)
    figures = analyse_leg(INDEX, samples)
    assert abs(figures["correlation"]) < 0.9
    assert verdict([("closing", figures)], [0.0] * 6) == 0


def test_the_residual_is_measured_with_the_lag_taken_out(capsys):
    """A filter's delay is not noise, and must not be counted as some."""
    figures = analyse_leg(INDEX, recording(filter_hz=FILTER_HZ))
    assert figures["rms_aligned"] < figures["rms_error"]
    # One count per sample is the floor a difference of ANGLE counts has.
    assert figures["rms_aligned"] < 2.0 * figures["resolution"]


def test_sliding_a_series_onto_itself_leaves_no_residual():
    values = [math.sin(i * 0.3) for i in range(40)]
    delayed = [math.sin((i - 2.0) * 0.3) for i in range(40)]
    assert shifted_rms(values, delayed, 2.0) == pytest.approx(0.0, abs=1e-9)
    assert shifted_rms(values, delayed, 0.0) > 0.1


def test_a_fractional_slide_interpolates_rather_than_rounding():
    values = [float(i) for i in range(20)]
    # A straight ramp slid by half a sample is exactly half a step out.
    assert shifted_rms(values, values, 0.5) == pytest.approx(0.5)


def test_a_reading_noisier_than_its_own_quantisation_fails(capsys):
    figures = analyse_leg(INDEX, recording())
    figures["rms_aligned"] = 10.0 * figures["resolution"]
    assert verdict([("closing", figures)], [0.0] * 6) == 1
    assert "residual" in capsys.readouterr().out


def test_a_clamp_that_bites_is_reported_as_one_cause_not_three(capsys):
    """A residual, a short integral and a lost lag are all the same clamp.

    Listing them separately sends someone looking for three bugs. The first
    hardware-shaped run of this tool did exactly that: SPEED_SET 400 against a
    transport that ignores it produced three failures with one cause.
    """
    figures = analyse_leg(INDEX, recording(peak=1.8, ceiling=0.6), ceiling=0.6)
    assert verdict([("closing", figures)], [0.0] * 6) == 1
    printed = capsys.readouterr().out
    assert "speed ceiling" in printed
    assert "FULL_TRAVEL_TIME_S" in printed
    assert printed.count("FAIL") == 1


def test_against_a_mock_the_clamp_says_which_of_the_two_causes_it_is(capsys):
    figures = analyse_leg(INDEX, recording(peak=1.8, ceiling=0.6), ceiling=0.6)
    verdict([("closing", figures)], [0.0] * 6, mock=True)
    assert "ignores SPEED_SET" in capsys.readouterr().out


def test_a_search_that_ends_at_its_own_edge_is_not_an_answer():
    """Where the clamped mock run's "200 ms lead" came from.

    A difference of past samples cannot lead the signal it differences. That
    figure was the search reporting its own boundary after a clamp flattened
    the trace and left nothing to align, so the boundary now reads as a failed
    search rather than as an answer.
    """
    reference = [math.sin(i * 0.3) for i in range(60)]
    far = [math.sin((i - 30) * 0.3) for i in range(60)]
    assert best_lag_samples(reference, far, max_shift=3) is None


def test_too_much_lag_fails(capsys):
    figures = analyse_leg(INDEX, recording(filter_hz=1.0))
    assert verdict([("closing", figures)], [0.0] * 6) == 1
    assert "lag" in capsys.readouterr().out


def test_the_noise_floor_is_reported_as_the_threshold_to_believe(capsys):
    verdict([("closing", analyse_leg(INDEX, recording()))], [0.0, 0.07, 0.0, 0.0, 0.0, 0.0])
    printed = capsys.readouterr().out
    assert "0.07 rad/s" in printed
    assert "ring" in printed, "the channel it came from, so it can be looked at"


def test_a_floor_too_small_for_four_decimals_is_not_printed_as_zero(capsys):
    """Claiming a non-zero floor and printing 0.0000 reads as a bug in the tool."""
    verdict([("closing", analyse_leg(INDEX, recording()))], [3e-5] + [0.0] * 5)
    printed = capsys.readouterr().out
    assert "3e-05 rad/s" in printed


# -- the display -------------------------------------------------------------
def test_zero_sits_on_the_centre_line_with_no_fill():
    drawn = signed_bar(0.0, 1.0, width=11)
    assert drawn == ".....:....."


def test_closing_fills_to_the_right_and_opening_to_the_left():
    assert signed_bar(1.0, 1.0, width=11) == ".....:#####"
    assert signed_bar(-1.0, 1.0, width=11) == "#####:....."


def test_the_fill_is_proportional_to_the_reading():
    assert signed_bar(0.4, 1.0, width=11) == ".....:##..."


def test_a_reading_past_full_scale_clamps_rather_than_overflowing():
    assert len(signed_bar(9.0, 1.0, width=11)) == 11
    assert signed_bar(9.0, 1.0, width=11) == ".....:#####"


def test_the_peaks_are_marked_on_their_own_sides():
    drawn = signed_bar(0.2, 1.0, width=11, peak_low=-0.8, peak_high=0.8)
    assert drawn == ".|...:#..|."


def test_a_peak_the_fill_has_reached_is_not_a_gap_in_it():
    assert signed_bar(1.0, 1.0, width=11, peak_high=1.0) == ".....:#####"


def test_a_zero_scale_does_not_divide_by_it():
    assert signed_bar(1.0, 0.0, width=11) == ".....:....."


def test_rms_of_nothing_is_zero_not_a_division():
    assert rms([]) == 0.0


# -- commanding a pose from the keyboard -------------------------------------
from inspire_hand_driver import command_overlays  # noqa: E402
from inspire_hand_driver.velocity_check import (  # noqa: E402
    STEPS,
    Keys,
    Pose,
    parse_pose,
    pose_rows,
)

THUMB_YAW = command_overlays.THUMB_ABDUCTION_DOF
OPEN = [1.0] * 6


def test_nothing_is_commanded_until_a_key_asks():
    pose = Pose()
    assert pose.target == [None] * 6
    assert pose.apply("", OPEN) == []


def test_a_digit_selects_that_channel_and_nothing_else_moves():
    pose = Pose()
    assert pose.apply("3", OPEN) == []
    assert pose.selected == kin.dof_index("3")
    assert pose.target == [None] * 6


def test_a_nudge_moves_only_the_selected_dof():
    pose = Pose(step=0.05, selected=INDEX)
    assert pose.apply("-", OPEN) == [INDEX]
    assert pose.target[INDEX] == pytest.approx(0.95)
    assert [t for i, t in enumerate(pose.target) if i != INDEX] == [None] * 5


def test_a_nudge_starts_from_where_the_finger_actually_is():
    """Otherwise the first keypress jumps the finger from an assumed pose."""
    pose = Pose(step=0.10, selected=INDEX)
    measured = list(OPEN)
    measured[INDEX] = 0.42
    pose.apply("+", measured)
    assert pose.target[INDEX] == pytest.approx(0.52)


def test_a_nudge_cannot_be_walked_outside_the_range():
    pose = Pose(step=0.25, selected=INDEX)
    pose.apply("-" * 10, OPEN)
    assert pose.target[INDEX] == 0.0
    pose.apply("+" * 10, OPEN)
    assert pose.target[INDEX] == 1.0


def test_a_selects_every_dof_at_once():
    pose = Pose()
    pose.apply("ac", OPEN)
    assert pose.target == [0.0] * 6


def test_open_close_and_half_are_absolute():
    pose = Pose(selected=INDEX)
    for key, expected in (("o", 1.0), ("c", 0.0), ("h", 0.5)):
        pose.apply(key, OPEN)
        assert pose.target[INDEX] == expected


def test_the_step_size_walks_the_ladder_and_stops_at_its_ends():
    pose = Pose(step=STEPS[0])
    pose.apply("[", OPEN)
    assert pose.step == STEPS[0], "already at the smallest"
    pose.apply("]" * 10, OPEN)
    assert pose.step == STEPS[-1]


def test_one_command_per_dof_however_many_keys_arrived():
    """A held-down key must not queue a burst of writes onto the RS485 bus."""
    pose = Pose(selected=INDEX)
    assert pose.apply("+++++", OPEN) == [INDEX]


def test_the_flags_the_caller_has_to_act_on_are_raised():
    pose = Pose()
    pose.apply("t", OPEN)
    assert pose.toggle_compliance
    pose.apply("r", OPEN)
    assert pose.reset_peaks
    pose.apply("q", OPEN)
    assert pose.quit


def test_an_unbound_key_does_nothing_rather_than_erroring():
    pose = Pose(selected=INDEX)
    assert pose.apply("xyz\n\x1b", OPEN) == []


# -- the overlay, which is the trap here -------------------------------------
def test_the_thumb_yaws_overlay_is_applied_to_what_the_command_should_produce():
    """A command of 0.0 on channel 6 is a physical 0.25, by calibration.

    Comparing the command against the readback directly would show a standing
    250-count error on a thumb sitting exactly where it was told to sit.
    """
    pose = Pose(selected=THUMB_YAW)
    pose.apply("c", OPEN)
    assert pose.target[THUMB_YAW] == 0.0
    assert pose.expected(THUMB_YAW) == pytest.approx(
        command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    )


def test_a_thumb_where_it_was_told_to_be_shows_no_error():
    pose = Pose(selected=THUMB_YAW)
    pose.apply("c", OPEN)
    measured = list(OPEN)
    measured[THUMB_YAW] = command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    row = pose_rows(pose, measured)[THUMB_YAW]
    assert "  +0" in row or "   0" in row, row


def test_a_finger_has_no_overlay_to_apply():
    pose = Pose(selected=INDEX)
    pose.apply("c", OPEN)
    assert pose.expected(INDEX) == 0.0


def test_seeding_a_thumb_nudge_takes_the_overlay_back_off():
    """The hand reports physical; the target holds commands. One or the other."""
    pose = Pose(step=0.0, selected=THUMB_YAW)
    measured = list(OPEN)
    measured[THUMB_YAW] = command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO
    pose.apply("+", measured)
    assert pose.target[THUMB_YAW] == pytest.approx(0.0)
    # And a round trip puts it back where it was, rather than walking away.
    assert pose.expected(THUMB_YAW) == pytest.approx(measured[THUMB_YAW])


# -- the rows, and the pose parser -------------------------------------------
def test_a_row_shows_the_register_count_the_speed_is_differenced_from():
    pose = Pose(selected=INDEX)
    pose.apply("h", OPEN)
    measured = list(OPEN)
    measured[INDEX] = 0.487
    row = pose_rows(pose, measured)[INDEX]
    assert "0.487" in row
    assert "487" in row, "the ANGLE register value"
    assert "-13" in row, "13 counts short of the commanded 0.5"


def test_an_uncommanded_dof_shows_no_error_rather_than_a_fake_zero():
    row = pose_rows(Pose(), OPEN)[INDEX]
    assert "--" in row


def test_a_row_before_any_state_has_arrived_says_so():
    row = pose_rows(Pose(), None)[INDEX]
    assert row.count("--") >= 2


def test_the_selected_dof_is_marked():
    rows = pose_rows(Pose(selected=INDEX), OPEN)
    assert rows[INDEX].startswith(">")
    assert not rows[0].startswith(">")


def test_one_number_is_a_pose_for_every_dof():
    assert parse_pose("0.3") == {i: 0.3 for i in range(6)}


def test_channels_can_be_named_one_by_one():
    assert parse_pose("4:0.3, 6:0.8") == {kin.dof_index("4"): 0.3, kin.dof_index("6"): 0.8}


def test_a_pose_outside_the_range_is_refused():
    with pytest.raises(ValueError):
        parse_pose("1.4")
    with pytest.raises(ValueError):
        parse_pose("4:-0.1")


def test_an_unknown_channel_is_refused_rather_than_ignored():
    with pytest.raises(ValueError):
        parse_pose("7:0.3")


def test_an_empty_pose_is_refused():
    with pytest.raises(ValueError):
        parse_pose("")


def test_keys_are_a_no_op_when_nothing_is_a_terminal(monkeypatch):
    """A piped run must still display, and must not touch termios."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with Keys() as keys:
        assert not keys.tty
        assert keys.read() == ""


def test_a_keystroke_poll_on_a_real_terminal_returns_at_once(monkeypatch):
    """The regression that stopped the display dead on the first poll.

    Gating a blocking read behind select is not enough: a pty can report
    itself readable while a VMIN-1 read on it still blocks, and the display
    then hangs before it has drawn a single frame. If this test ever hangs
    rather than fails, that is the bug back again.
    """
    import os
    import signal

    master, slave = os.openpty()

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return slave

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    # A hang here is the failure mode under test, so bound it rather than
    # letting the suite stop.
    signal.alarm(10)
    try:
        with Keys() as keys:
            assert keys.tty
            assert keys.read() == "", "nothing typed yet, and it must not wait"
            os.write(master, b"3h")
            typed = ""
            for _ in range(100):
                typed += keys.read()
                if typed:
                    break
                time.sleep(0.01)
            assert typed == "3h"
            assert keys.read() == "", "and nothing is left over"
    finally:
        signal.alarm(0)
        os.close(master)
        os.close(slave)


def test_a_terminal_that_goes_away_mid_run_is_not_an_exception(monkeypatch):
    """Nothing about a terminal disappearing may raise into the display loop.

    Two halves, because they fail differently: a read on a dead terminal, and
    the restore in the caller's ``finally`` afterwards. The restore raising is
    what turned a tidy-up into the exception that ended the run.
    """
    import os

    master, slave = os.openpty()

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return slave

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    keys = Keys()
    keys.__enter__()
    os.close(master)
    assert keys.read() == ""
    os.close(slave)
    # The descriptor is gone now, which is as dead as a terminal gets.
    assert keys.read() == ""
    assert not keys.tty, "and it stops asking"
    keys.restore()  # must not raise


# -- the one-finger trace ----------------------------------------------------
from inspire_hand_driver.velocity_check import (  # noqa: E402
    single_rows,
    velocity_trace,
)


def test_a_flat_trace_is_the_zero_line_and_nothing_else():
    rows = velocity_trace([0.0] * 10, [0.0] * 10, scale=1.0, width=10, height=5)
    assert len(rows) == 5
    body = [r.split("|")[1] for r in rows]
    assert body[2] == "##########", "the driver's own zeros sit on the zero line"
    assert set(body[0] + body[1] + body[3] + body[4]) == {" "}


def test_the_trace_puts_a_positive_rate_above_the_line_and_negative_below():
    high = velocity_trace([1.0] * 4, [], scale=1.0, width=4, height=5)
    assert high[0].split("|")[1] == "####"
    low = velocity_trace([-1.0] * 4, [], scale=1.0, width=4, height=5)
    assert low[4].split("|")[1] == "####"


def test_the_newest_sample_is_on_the_right():
    rows = velocity_trace([0.0, 0.0, 1.0], [], scale=1.0, width=3, height=5)
    assert rows[0].split("|")[1] == "  #"


def test_a_short_history_is_right_aligned_rather_than_stretched():
    rows = velocity_trace([1.0], [], scale=1.0, width=6, height=3)
    assert rows[0].split("|")[1] == "     #"


def test_the_trace_is_clipped_to_its_scale_not_wrapped():
    rows = velocity_trace([99.0], [], scale=1.0, width=1, height=5)
    assert rows[0].split("|")[1] == "#"
    assert len(rows) == 5


def test_more_history_than_width_keeps_the_newest():
    rows = velocity_trace([0.0] * 50 + [1.0], [], scale=1.0, width=5, height=3)
    assert rows[0].split("|")[1] == "    #"


def test_the_reference_is_drawn_but_never_over_the_reported_rate():
    """The reported rate is the subject; the reference is the thing behind it."""
    rows = velocity_trace([1.0], [1.0], scale=1.0, width=1, height=3)
    assert rows[0].split("|")[1] == "#"
    apart = velocity_trace([1.0], [-1.0], scale=1.0, width=1, height=3)
    assert apart[0].split("|")[1] == "#"
    assert apart[2].split("|")[1] == "+"


def test_the_trace_is_labelled_with_the_scale_at_top_middle_and_bottom():
    rows = velocity_trace([0.0], [], scale=1.84, width=1, height=5)
    assert "+1.84" in rows[0]
    assert "+0.00" in rows[2]
    assert "-1.84" in rows[4]


def test_a_zero_scale_does_not_divide_by_it_in_the_trace():
    rows = velocity_trace([1.0, -1.0], [], scale=0.0, width=2, height=3)
    assert rows[1].split("|")[1] == "##"


def test_an_even_height_is_made_odd_so_there_is_a_true_zero_line():
    assert len(velocity_trace([0.0], [], scale=1.0, width=1, height=8)) == 9


def test_the_single_dof_frame_names_the_joint_and_shows_the_count():
    pose = Pose(selected=INDEX)
    pose.apply("h", OPEN)
    history = recording(peak=1.0, moving=20, still=5)
    ratios = list(OPEN)
    ratios[INDEX] = 0.487
    rows = single_rows(
        INDEX, pose, ratios, history, -1.0, 1.0, 1.8,
        central_difference(
            [s.stamp for s in history], [s.position[INDEX] for s in history]
        ),
    )
    printed = "\n".join(rows)
    assert kin.DOFS[INDEX].joint in printed
    assert "count 487" in printed
    assert "err -13" in printed
    assert "cmd 0.50" in printed
    # And the trace is in there, with its zero line.
    assert any("+0.00 |" in row for row in rows)


def test_the_terminal_setup_ignores_sigttou_across_the_call(monkeypatch):
    """The freeze that drew nothing at all, and the reason it was invisible.

    A process outside the terminal's foreground group gets SIGTTOU for
    changing terminal settings, and SIGTTOU's default action is to *stop* the
    process -- no error, no message, just a display that never draws a frame
    until something kills it. That is what happened when this was first run
    through a pipeline, and the only defence is to ignore the signal across
    the call.
    """
    import signal

    seen = []

    def spy(fd, when, mode):
        seen.append((when, signal.getsignal(signal.SIGTTOU)))

    monkeypatch.setattr(termios, "tcsetattr", spy)
    marker = lambda *a: None  # noqa: E731 - a handler to check gets put back
    previous = signal.signal(signal.SIGTTOU, marker)
    try:
        keys = Keys()
        keys.tty = True
        keys._fd = 0
        assert keys._write_mode([0] * 7)
        assert signal.getsignal(signal.SIGTTOU) is marker, "and it is put back"
    finally:
        signal.signal(signal.SIGTTOU, previous)
    when, during = seen[0]
    assert during == signal.SIG_IGN, "SIGTTOU must be ignored during the call"
    # Neither of the draining variants: there is nothing here worth waiting on,
    # and TCSAFLUSH is what blocked under a pty in the first place.
    assert when == termios.TCSANOW


def test_a_terminal_that_refuses_the_setup_becomes_a_watch_only_run(monkeypatch):
    """Refusal is not a reason to fail: the display is still worth having."""
    import os

    master, slave = os.openpty()

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return slave

    monkeypatch.setattr(sys, "stdin", FakeStdin())

    def refuse(*args):
        raise termios.error(22, "Invalid argument")

    monkeypatch.setattr(termios, "tcsetattr", refuse)
    with Keys() as keys:
        assert not keys.tty
        assert keys.read() == ""
    os.close(master)
    os.close(slave)


# -- sizing the trace, and getting the samples out ---------------------------
from inspire_hand_driver.velocity_check import (  # noqa: E402
    LOG_HEADER,
    log_rows,
    trace_size,
    write_log,
)


def test_the_trace_takes_the_window_it_is_given():
    """One column is one sample, so width is resolution in the most direct sense."""
    narrow = trace_size(80, 24)
    wide = trace_size(200, 60)
    assert wide[0] > narrow[0] and wide[1] > narrow[1]
    # 41 rows is the cap: past that a row is a fortieth of the scale and the
    # plot is taller than it is informative.
    assert wide == (189, 41)
    assert trace_size(500, 500) == (400, 41), "both axes are capped"


def test_the_trace_leaves_room_for_the_rest_of_the_frame():
    """A frame as tall as the window scrolls its own top away on every redraw."""
    width, height = trace_size(100, 30)
    assert height + 10 < 30, "the ten other rows of the frame, and a margin"
    assert width + 11 <= 100


def test_a_tiny_window_still_gets_a_usable_trace():
    width, height = trace_size(20, 8)
    assert width >= 20 and height >= 5
    assert height % 2 == 1


def test_the_trace_height_is_always_odd_so_zero_has_its_own_row():
    for lines in range(10, 80):
        assert trace_size(100, lines)[1] % 2 == 1


def test_an_enormous_window_does_not_ask_for_more_history_than_is_kept():
    from inspire_hand_driver.velocity_check import HISTORY

    assert trace_size(10_000, 10_000)[0] <= HISTORY


def test_a_log_row_per_sample_with_the_register_count_in_it():
    samples = recording(peak=1.0, moving=10, still=2)
    rows = log_rows("closing", samples, [INDEX])
    assert len(rows) == len(samples)
    assert all(row.startswith("closing,") for row in rows)
    fields = rows[-1].split(",")
    assert len(fields) == len(LOG_HEADER.split(","))
    assert fields[2] == kin.DOFS[INDEX].channel
    assert fields[3] == kin.DOFS[INDEX].joint
    # The count is the ANGLE register the rate was differenced from, so it has
    # to be a whole number of counts in 0..1000.
    assert 0 <= int(fields[4]) <= 1000


def test_the_logged_reference_is_the_centred_difference_of_the_same_rows():
    samples = recording(peak=1.0, moving=20, still=2)
    rows = log_rows("closing", samples, [INDEX])
    expected = central_difference(
        [s.stamp for s in samples], [s.position[INDEX] for s in samples]
    )
    assert [float(r.split(",")[7]) for r in rows] == pytest.approx(expected, abs=1e-5)


def test_logging_several_dof_keeps_them_in_separate_runs_of_rows():
    samples = recording(peak=1.0, moving=10, still=2)
    rows = log_rows("live", samples, [0, INDEX])
    assert len(rows) == 2 * len(samples)
    channels = [row.split(",")[2] for row in rows]
    assert channels[0] == kin.DOFS[0].channel
    assert channels[-1] == kin.DOFS[INDEX].channel


def test_the_header_is_written_once_and_appending_does_not_repeat_it(tmp_path):
    path = str(tmp_path / "v.csv")
    samples = recording(peak=1.0, moving=10, still=2)
    write_log(path, log_rows("standstill", samples, [INDEX]), header=True)
    write_log(path, log_rows("closing", samples, [INDEX]), header=False)
    lines = open(path).read().strip().split("\n")
    assert lines[0] == LOG_HEADER
    assert sum(1 for line in lines if line == LOG_HEADER) == 1
    assert len(lines) == 1 + 2 * len(samples)


def test_an_empty_log_is_a_header_and_nothing_else(tmp_path):
    path = str(tmp_path / "v.csv")
    write_log(path, [], header=True)
    assert open(path).read() == LOG_HEADER + "\n"
