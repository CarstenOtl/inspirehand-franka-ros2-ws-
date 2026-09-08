"""Recovering a known camera-to-robot delay, and refusing to guess at one.

The failure this guards against is the quiet one: returning a confident number
from a pass where the tag barely moved, or where the true delay is outside the
search range. Both would look like a successful measurement.
"""

import numpy as np
import pytest

from camera_calibration.time_offset import (
    TimeOffsetError,
    estimate_time_offset,
    hermite_interpolate,
)


def tag_path(times):
    """A smooth sweep at roughly 100 mm/s, like the ramps between two poses."""
    times = np.asarray(times, dtype=float)
    return np.column_stack(
        [
            0.50 + 0.12 * np.sin(0.9 * times),
            0.10 * np.sin(0.55 * times + 1.0),
            0.45 + 0.08 * np.cos(0.7 * times),
        ]
    )


def streams(delay_s, robot_rate_hz=30.0, camera_rate_hz=30.0, noise_m=0.001, duration_s=20.0,
            seed=4):
    random = np.random.default_rng(seed)
    robot_times = np.arange(0.0, duration_s, 1.0 / robot_rate_hz)
    camera_times = np.arange(1.0, duration_s - 1.0, 1.0 / camera_rate_hz)
    camera_positions = tag_path(camera_times + delay_s) + random.normal(
        scale=noise_m, size=(len(camera_times), 3)
    )
    return camera_times, camera_positions, robot_times, tag_path(robot_times)


def test_hermite_interpolation_beats_a_straight_line_between_samples():
    times = np.arange(0.0, 5.0, 1.0 / 30.0)
    values = tag_path(times)
    query = times[:-1] + 0.5 / 30.0
    interpolated, inside = hermite_interpolate(times, values, query)
    assert inside.all()
    truth = tag_path(query)
    linear = 0.5 * (values[:-1] + values[1:])
    assert np.abs(interpolated - truth).max() < np.abs(linear - truth).max()


def test_interpolation_marks_queries_outside_the_sampled_span():
    times = np.arange(0.0, 2.0, 0.05)
    _, inside = hermite_interpolate(times, tag_path(times), np.array([-1.0, 1.0, 5.0]))
    np.testing.assert_array_equal(inside, [False, True, False])


def test_interpolation_refuses_unusable_inputs():
    times = np.arange(0.0, 2.0, 0.05)
    with pytest.raises(TimeOffsetError, match="four samples"):
        hermite_interpolate(times[:3], tag_path(times[:3]), times[:1])
    with pytest.raises(TimeOffsetError, match="increasing"):
        hermite_interpolate(times[::-1], tag_path(times), times[:1])


@pytest.mark.parametrize("delay_s", [0.040, -0.025, 0.0, 0.100])
def test_a_known_delay_is_recovered(delay_s):
    result = estimate_time_offset(*streams(delay_s))
    assert result.offset_s == pytest.approx(delay_s, abs=0.002)
    assert result.uncertainty_s < 0.002
    assert result.samples_used > 400


@pytest.mark.parametrize("robot_rate_hz", [30.0, 100.0, 1000.0])
def test_the_joint_state_rate_does_not_bias_the_estimate(robot_rate_hz):
    result = estimate_time_offset(*streams(0.040, robot_rate_hz=robot_rate_hz))
    assert result.offset_s == pytest.approx(0.040, abs=0.002)


def test_a_noisy_pass_widens_the_uncertainty_but_stays_honest():
    quiet = estimate_time_offset(*streams(0.040, noise_m=0.001))
    noisy = estimate_time_offset(*streams(0.040, noise_m=0.004))
    assert noisy.uncertainty_s > quiet.uncertainty_s
    assert noisy.offset_s == pytest.approx(0.040, abs=3.0 * noisy.uncertainty_s)


def test_ignoring_the_delay_costs_what_the_result_says_it_costs():
    result = estimate_time_offset(*streams(0.040))
    assert result.rms_at_zero_m > result.rms_at_offset_m
    # 40 ms at about 100 mm/s is a few millimetres, which is the whole point.
    assert result.rms_at_zero_m > 0.002


def test_a_barely_moving_pass_is_refused():
    times = np.arange(0.0, 20.0, 0.01)
    still = np.column_stack(
        [0.5 + 0.0005 * np.sin(0.3 * times), np.zeros_like(times), 0.45 + np.zeros_like(times)]
    )
    camera_times = np.arange(1.0, 19.0, 1.0 / 30.0)
    camera_positions = np.column_stack(
        [
            0.5 + 0.0005 * np.sin(0.3 * (camera_times + 0.04)),
            np.zeros_like(camera_times),
            0.45 + np.zeros_like(camera_times),
        ]
    )
    with pytest.raises(TimeOffsetError, match="median"):
        estimate_time_offset(camera_times, camera_positions, times, still)


def test_a_delay_outside_the_search_range_is_refused():
    with pytest.raises(TimeOffsetError, match="edge"):
        estimate_time_offset(*streams(0.300))


def test_too_few_observations_are_refused():
    camera_times, camera_positions, robot_times, robot_positions = streams(0.040)
    with pytest.raises(TimeOffsetError, match="at least"):
        estimate_time_offset(
            camera_times[:10], camera_positions[:10], robot_times, robot_positions
        )


def test_mismatched_stream_lengths_are_refused():
    camera_times, camera_positions, robot_times, robot_positions = streams(0.040)
    with pytest.raises(TimeOffsetError, match="counts differ"):
        estimate_time_offset(
            camera_times[:-5], camera_positions, robot_times, robot_positions
        )
