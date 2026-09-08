"""Estimating the delay between camera timestamps and robot timestamps.

Stopping the arm at every pose makes the static calibration immune to timing:
if nothing moves, it does not matter when each measurement was taken.  But a
camera calibration is used while the robot moves, and there the two clocks have
to agree.  The last hand-guided run showed they do not - the recorder kept
reporting ``Lookup would require extrapolation into the future``, meaning image
stamps ran ahead of the newest joint state.

This module measures that delay instead of assuming it away.  With the static
calibration known, the tag's position in the world is predicted twice: once
from the camera and once from the robot's joints.  The delay is the time shift
that makes the two agree, found by scanning it and refining the minimum.  It is
only identifiable while the tag is actually moving, so the estimator checks
that and refuses rather than returning a confident zero.
"""

from dataclasses import dataclass

import numpy as np


class TimeOffsetError(RuntimeError):
    """Raised when the recorded motion cannot identify a delay."""


@dataclass(frozen=True)
class TimeOffsetResult:
    """The fitted delay.

    ``offset_s`` is what has to be *added* to a camera timestamp to land on the
    robot clock: positive means the images are stamped early, so a lookup at an
    image's own stamp reads the robot's past.
    """

    offset_s: float
    uncertainty_s: float
    rms_at_offset_m: float
    rms_at_zero_m: float
    median_speed_m_per_s: float
    samples_used: int
    scan_offsets_s: np.ndarray
    scan_rms_m: np.ndarray

    def as_dict(self) -> dict:
        return {
            "offset_s": self.offset_s,
            "uncertainty_s": self.uncertainty_s,
            "rms_at_offset_m": self.rms_at_offset_m,
            "rms_at_zero_m": self.rms_at_zero_m,
            "median_speed_m_per_s": self.median_speed_m_per_s,
            "samples_used": self.samples_used,
        }


def hermite_interpolate(
    times: np.ndarray, values: np.ndarray, query_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation with central-difference tangents.

    Linear interpolation would bias every query towards the chord between two
    samples, which at a 30 Hz joint-state rate is a systematic error of the same
    order as the delay being measured.  Returns the interpolated values and a
    mask of the queries that fell inside the sampled span.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values):
        raise TimeOffsetError("times must be 1-D and values (N, D) with the same N")
    if len(times) < 4:
        raise TimeOffsetError("need at least four samples to interpolate")
    if np.any(np.diff(times) <= 0.0):
        raise TimeOffsetError("times must be strictly increasing")

    slopes = np.zeros_like(values)
    slopes[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])[:, None]
    slopes[0] = (values[1] - values[0]) / (times[1] - times[0])
    slopes[-1] = (values[-1] - values[-2]) / (times[-1] - times[-2])

    inside = (query_times >= times[0]) & (query_times <= times[-1])
    result = np.full((len(query_times), values.shape[1]), np.nan)
    if not np.any(inside):
        return result, inside

    selected = query_times[inside]
    index = np.clip(np.searchsorted(times, selected, side="right") - 1, 0, len(times) - 2)
    spacing = times[index + 1] - times[index]
    unit = ((selected - times[index]) / spacing)[:, None]
    spacing = spacing[:, None]
    unit_squared = unit * unit
    unit_cubed = unit_squared * unit
    result[inside] = (
        (2.0 * unit_cubed - 3.0 * unit_squared + 1.0) * values[index]
        + (unit_cubed - 2.0 * unit_squared + unit) * spacing * slopes[index]
        + (-2.0 * unit_cubed + 3.0 * unit_squared) * values[index + 1]
        + (unit_cubed - unit_squared) * spacing * slopes[index + 1]
    )
    return result, inside


def _rms(
    offset: float,
    camera_times: np.ndarray,
    camera_positions: np.ndarray,
    robot_times: np.ndarray,
    robot_positions: np.ndarray,
) -> tuple[float, int]:
    predicted, inside = hermite_interpolate(
        robot_times, robot_positions, camera_times + offset
    )
    if not np.any(inside):
        return np.inf, 0
    difference = camera_positions[inside] - predicted[inside]
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))), int(np.count_nonzero(inside))


def estimate_time_offset(
    camera_times_s,
    camera_tag_positions_m,
    robot_times_s,
    robot_tag_positions_m,
    search_range_s: float = 0.15,
    scan_step_s: float = 0.002,
    minimum_samples: int = 30,
    minimum_speed_m_per_s: float = 0.02,
    maximum_uncertainty_s: float = 0.010,
) -> TimeOffsetResult:
    """Fit the camera-to-robot clock delay from a moving pass.

    Both position streams must already be in the same frame - the world frame -
    with the camera stream being ``world_to_camera @ camera_to_tag`` and the
    robot stream ``world_to_hand(t) @ hand_to_tag``.
    """
    camera_times = np.asarray(camera_times_s, dtype=float)
    camera_positions = np.asarray(camera_tag_positions_m, dtype=float).reshape(-1, 3)
    robot_times = np.asarray(robot_times_s, dtype=float)
    robot_positions = np.asarray(robot_tag_positions_m, dtype=float).reshape(-1, 3)
    if len(camera_times) != len(camera_positions):
        raise TimeOffsetError("camera time and position counts differ")
    if len(robot_times) != len(robot_positions):
        raise TimeOffsetError("robot time and position counts differ")
    if len(camera_times) < minimum_samples:
        raise TimeOffsetError(
            f"need at least {minimum_samples} tag observations, have {len(camera_times)}"
        )

    order = np.argsort(robot_times)
    robot_times = robot_times[order]
    robot_positions = robot_positions[order]
    keep = np.concatenate(([True], np.diff(robot_times) > 0.0))
    robot_times = robot_times[keep]
    robot_positions = robot_positions[keep]

    travelled = np.linalg.norm(np.diff(robot_positions, axis=0), axis=1)
    elapsed = np.diff(robot_times)
    median_speed = float(np.median(travelled / elapsed)) if len(elapsed) else 0.0
    if median_speed < minimum_speed_m_per_s:
        raise TimeOffsetError(
            f"the tag moved at a median {median_speed * 1000.0:.1f} mm/s, below the "
            f"{minimum_speed_m_per_s * 1000.0:.0f} mm/s needed to separate a delay from "
            "measurement noise; drive the offset pass faster or over a longer path"
        )

    offsets = np.arange(-search_range_s, search_range_s + 0.5 * scan_step_s, scan_step_s)
    scan = np.empty(len(offsets))
    counts = np.empty(len(offsets), dtype=int)
    for index, offset in enumerate(offsets):
        scan[index], counts[index] = _rms(
            offset, camera_times, camera_positions, robot_times, robot_positions
        )
    best = int(np.argmin(scan))
    if best in (0, len(offsets) - 1):
        raise TimeOffsetError(
            f"the best delay sits at the edge of the +/-{search_range_s * 1000.0:.0f} ms "
            "search range; the two clocks are further apart than that, or the streams "
            "do not overlap"
        )

    # Parabola through the three points around the minimum: its vertex is the
    # sub-step estimate, and its curvature gives the standard error.
    squared = scan[best - 1 : best + 2] ** 2
    curvature = (squared[0] - 2.0 * squared[1] + squared[2]) / (scan_step_s**2)
    if curvature <= 0.0:
        raise TimeOffsetError("the delay scan has no clear minimum")
    slope = (squared[2] - squared[0]) / (2.0 * scan_step_s)
    offset = float(offsets[best] - slope / curvature)

    rms_at_offset, samples_used = _rms(
        offset, camera_times, camera_positions, robot_times, robot_positions
    )
    rms_at_zero, _ = _rms(0.0, camera_times, camera_positions, robot_times, robot_positions)
    # var = sigma^2 / a for a sum of squares S(d) ~ S0 + a (d - d0)^2, with the
    # per-component residual variance sigma^2 taken from the fit itself.
    residual_variance = rms_at_offset**2 / max(1, 3 * samples_used - 1)
    uncertainty = float(np.sqrt(residual_variance / (0.5 * curvature)))
    if uncertainty > maximum_uncertainty_s:
        raise TimeOffsetError(
            f"the delay is only determined to +/-{uncertainty * 1000.0:.1f} ms, worse than "
            f"the {maximum_uncertainty_s * 1000.0:.0f} ms this pass demands; record a longer "
            "or faster offset pass"
        )

    return TimeOffsetResult(
        offset_s=offset,
        uncertainty_s=uncertainty,
        rms_at_offset_m=rms_at_offset,
        rms_at_zero_m=rms_at_zero,
        median_speed_m_per_s=median_speed,
        samples_used=samples_used,
        scan_offsets_s=offsets,
        scan_rms_m=scan,
    )
