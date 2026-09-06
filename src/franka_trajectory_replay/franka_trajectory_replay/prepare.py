# Copyright (c) 2026 Agile Robots SE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turning a policy capture into the dense joint command stream the controller will play.

A policy samples at 15-60 Hz; the arm wants a new position every millisecond that is
continuous in velocity and acceleration. The pipeline is:

1. optional uniform time scaling (slow down),
2. a clamped cubic spline through the capture (zero velocity at both ends, which is also what
   the goto ramp arrives with),
3. sampling at the controller rate with hold segments before and after,
4. optional zero-phase low-pass filtering (Butterworth, ``filtfilt``) of the positions,
5. velocity / acceleration / jerk by finite differences of the *final* positions, so the check
   sees exactly what will be commanded.

``auto_scale`` repeats the pipeline with the slow-down factor the limit check asks for.
"""

from dataclasses import dataclass, field

import numpy as np

from franka_trajectory_replay import limits


@dataclass
class Prepared:
    t: np.ndarray       # (M,) seconds from the start of the command stream
    q: np.ndarray       # (M, 7)
    qd: np.ndarray      # (M, 7)
    qdd: np.ndarray     # (M, 7)
    qddd: np.ndarray    # (M, 7)
    joint_names: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    @property
    def duration(self):
        return float(self.t[-1])

    @property
    def rate(self):
        return float((len(self.t) - 1) / self.t[-1]) if len(self.t) > 1 else 0.0


def _derivatives(t, q):
    dt = float(np.median(np.diff(t)))
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    qddd = np.gradient(qdd, dt, axis=0)
    return qd, qdd, qddd


def _lead(q_end, v_end, rate, min_duration, max_acceleration):
    """Smooth ramp between rest and a moving end state, ending exactly at (q_end, v_end).

    Velocity profile v(tau) = v_end * (3 tau^2 - 2 tau^3): zero acceleration at both ends, peak
    acceleration 1.5 |v_end| / T. T is stretched so that peak stays under max_acceleration. The
    ramp starts at rest at q_end - v_end * T / 2. Returned without its final sample (which is the
    trajectory's own first sample).
    """
    v_end = np.asarray(v_end, dtype=float)
    peak = float(np.abs(v_end).max())
    duration = max(float(min_duration), 1.5 * peak / float(max_acceleration) if peak > 0 else 0.0)
    n = max(2, int(round(duration * rate)))
    tau = np.arange(n) / n
    displacement = duration * (tau ** 3 - 0.5 * tau ** 4)  # integral of the velocity profile
    start = np.asarray(q_end, dtype=float) - v_end * duration / 2.0
    return start + np.outer(displacement, v_end), duration


def resample(trajectory, rate=1000, cutoff_hz=0.0, hold_start=0.5, hold_end=0.5,
             time_scale=1.0, joint_names=None, lead_in=0.5, lead_out=0.5,
             lead_max_acceleration=2.5, interpolation='cubic', blend_time=0.04):
    """Dense, smooth version of ``trajectory`` at ``rate`` Hz.

    Layout of the stream: hold | lead-in | the capture (densified, optionally low-passed) |
    lead-out | hold. ``interpolation='cubic'`` runs a natural cubic spline through the waypoints;
    ``'linear'`` keeps straight lines between them with parabolic corner blends of ``blend_time``. A policy is usually already moving at its first recorded sample, so the stream does
    not force zero velocity there; instead the lead-in accelerates smoothly from rest into the
    capture's initial velocity, and the lead-out brakes from its final velocity. The stream
    therefore starts a little before the capture's first point (by v0 * T_in / 2); that start
    point is what the goto ramp aims at.
    """
    from scipy.interpolate import CubicSpline

    if time_scale < 1.0:
        raise ValueError('time_scale < 1 would speed the trajectory up; use >= 1')
    t_source = np.asarray(trajectory.t, dtype=float) * float(time_scale)
    q_source = np.asarray(trajectory.q, dtype=float)
    if len(t_source) < 2:
        raise ValueError('a trajectory needs at least two samples')

    dt = 1.0 / float(rate)
    n_motion = int(np.floor(t_source[-1] / dt)) + 1
    t_motion = np.arange(n_motion) * dt
    if interpolation == 'cubic':
        spline = CubicSpline(t_source, q_source, axis=0, bc_type='natural')
        q_motion = spline(np.minimum(t_motion, t_source[-1]))
    elif interpolation == 'linear':
        # Straight lines between the policy waypoints, densified to the control rate. A pure
        # polyline has a velocity step at every waypoint, which the FR3 reads as an acceleration
        # of dv / 1 ms and reflexes on; a triangular window of blend_time turns each corner into
        # a parabolic blend (continuous velocity, bounded acceleration) and leaves the straight
        # parts between blends untouched when blend_time is shorter than the waypoint spacing.
        q_motion = np.column_stack([np.interp(t_motion, t_source, q_source[:, j]) for j in range(q_source.shape[1])])
        half = max(1, int(round(0.5 * blend_time * rate)))
        window = np.bartlett(2 * half + 1)
        window /= window.sum()
        padded = np.vstack([np.repeat(q_motion[:1], half, axis=0), q_motion, np.repeat(q_motion[-1:], half, axis=0)])
        q_motion = np.column_stack([np.convolve(padded[:, j], window, mode='valid') for j in range(q_motion.shape[1])])
    else:
        raise ValueError("interpolation must be 'cubic' or 'linear', got %r" % interpolation)

    if cutoff_hz and cutoff_hz > 0.0:
        from scipy.signal import butter, filtfilt

        if cutoff_hz >= rate / 2.0:
            raise ValueError('cutoff_hz must be below the Nyquist rate %.0f Hz' % (rate / 2.0))
        b, a = butter(4, cutoff_hz / (rate / 2.0))
        q_motion = filtfilt(b, a, q_motion, axis=0, padlen=min(3 * max(len(a), len(b)), len(q_motion) - 1))

    v_first = (q_motion[1] - q_motion[0]) / dt if n_motion > 1 else np.zeros(7)
    v_last = (q_motion[-1] - q_motion[-2]) / dt if n_motion > 1 else np.zeros(7)
    ramp_in, t_in = _lead(q_motion[0], v_first, rate, lead_in, lead_max_acceleration)
    ramp_out, t_out = _lead(q_motion[-1], -v_last, rate, lead_out, lead_max_acceleration)
    # Time-reversed lead-in: from (q_last, v_last) down to rest. _lead omits its end sample
    # (which is q_motion[-1] itself), so the reversed ramp continues right after the capture.
    ramp_out = ramp_out[::-1]
    n_start = int(round(hold_start * rate))
    n_end = int(round(hold_end * rate))
    q = np.vstack([
        np.repeat(ramp_in[:1], n_start, axis=0),
        ramp_in,
        q_motion,
        ramp_out,
        np.repeat(ramp_out[-1:], n_end, axis=0),
    ])

    t = np.arange(len(q)) * dt
    qd, qdd, qddd = _derivatives(t, q)
    return Prepared(
        t=t, q=q, qd=qd, qdd=qdd, qddd=qddd,
        joint_names=list(joint_names or []),
        params={
            'rate': float(rate), 'cutoff_hz': float(cutoff_hz or 0.0),
            'hold_start': float(hold_start), 'hold_end': float(hold_end),
            'lead_in': float(t_in), 'lead_out': float(t_out),
            'interpolation': interpolation, 'blend_time': float(blend_time),
            'time_scale': float(time_scale), 'source_samples': int(len(t_source)),
            'source_duration': float(trajectory.t[-1] - trajectory.t[0]),
            'capture_start_index': int(n_start + len(ramp_in)),
            'capture_start_offset': (ramp_in[0] - q_motion[0]).tolist(),
        },
    )


def prepare(trajectory, rate=1000, cutoff_hz=0.0, hold_start=0.5, hold_end=0.5, time_scale=1.0,
            auto_scale=True, velocity_margin=0.8, acceleration_margin=0.5, jerk_margin=0.5,
            joint_names=None, max_iterations=6, lead_in=0.5, lead_out=0.5, lead_max_acceleration=2.5,
            interpolation='cubic', blend_time=0.04):
    """resample + limit check, slowing the trajectory down until it fits if ``auto_scale``."""
    scale = float(time_scale)
    history = []
    for _ in range(max_iterations):
        prepared = resample(trajectory, rate, cutoff_hz, hold_start, hold_end, scale, joint_names,
                            lead_in, lead_out, lead_max_acceleration, interpolation, blend_time)
        report = limits.check(prepared.t, prepared.q, prepared.qd, prepared.qdd, prepared.qddd,
                              velocity_margin, acceleration_margin, jerk_margin)
        history.append({'time_scale': scale, 'ok': report['ok'],
                        'velocity_fraction': max(report['velocity_fraction']),
                        'acceleration_fraction': max(report['acceleration_fraction']),
                        'jerk_fraction': max(report['jerk_fraction'])})
        if report['ok'] or not auto_scale:
            break
        # Slightly more than the analytic factor: the spline re-fit is not exactly self-similar.
        scale *= 1.02 * limits.required_time_scale(report)
    prepared.report = report
    prepared.params['auto_scale_history'] = history
    prepared.params['velocity_margin'] = velocity_margin
    prepared.params['acceleration_margin'] = acceleration_margin
    prepared.params['jerk_margin'] = jerk_margin
    return prepared


def goto_duration(step, max_velocity=0.5, max_acceleration=1.0, min_duration=2.0):
    """Duration the controller picks for a quintic ramp over ``step`` (per-joint, rad)."""
    step = np.abs(np.asarray(step, dtype=float))
    duration = float(min_duration)
    duration = max(duration, float((1.875 * step / max_velocity).max()))
    duration = max(duration, float(np.sqrt(5.7735 * step / max_acceleration).max()))
    return duration


def quintic(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def summarize(prepared, joint_names=None):
    """Short human-readable summary of a prepared trajectory and its limit check."""
    names = joint_names or prepared.joint_names or ['joint%d' % (i + 1) for i in range(7)]
    report = prepared.report
    lines = [
        'prepared: %d samples at %.0f Hz, %.2f s (source %d samples over %.2f s, time scale x%.3f, '
        '%s interpolation, cutoff %s, hold %.2f/%.2f s, lead-in/out %.2f/%.2f s)' % (
            len(prepared.t), prepared.rate, prepared.duration, prepared.params.get('source_samples', 0),
            prepared.params.get('source_duration', 0.0), prepared.params.get('time_scale', 1.0),
            prepared.params.get('interpolation', 'cubic') + (
                ' (%.0f ms blends)' % (1000 * prepared.params['blend_time'])
                if prepared.params.get('interpolation') == 'linear' else ''),
            ('%.0f Hz' % prepared.params['cutoff_hz']) if prepared.params.get('cutoff_hz') else 'off',
            prepared.params.get('hold_start', 0.0), prepared.params.get('hold_end', 0.0),
            prepared.params.get('lead_in', 0.0), prepared.params.get('lead_out', 0.0)),
    ]
    if report:
        lines.append('  %-12s %9s %9s %9s %9s   %6s %6s %6s' % (
            '', 'min', 'max', '|qd|max', '|qdd|max', 'v%', 'a%', 'j%'))
        for j, name in enumerate(names):
            lines.append('  %-12s %+9.4f %+9.4f %9.3f %9.2f   %5.0f%% %5.0f%% %5.0f%%' % (
                name, report['position_min'][j], report['position_max'][j], report['velocity_peak'][j],
                report['acceleration_peak'][j], 100 * report['velocity_fraction'][j],
                100 * report['acceleration_fraction'][j], 100 * report['jerk_fraction'][j]))
        margins = report['margins']
        lines.append('  percentages are of the FR3 limit times the margin (v %.2f, a %.2f, j %.2f)' % (
            margins['velocity'], margins['acceleration'], margins['jerk']))
        if report['violations']:
            lines.append('  VIOLATIONS:')
            lines.extend('    - ' + text for text in report['violations'])
        else:
            lines.append('  within limits')
    return '\n'.join(lines)
