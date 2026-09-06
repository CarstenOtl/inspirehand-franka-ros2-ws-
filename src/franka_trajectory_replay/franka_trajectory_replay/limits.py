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

"""FR3 joint limits, as libfranka's rate_limiting.h and franka_description's joint_limits.yaml.

The velocity limit depends on the joint position (the arm has to be able to brake before the
position limit). The formulas are libfranka's ``computeUpperLimitsJointVelocity`` /
``computeLowerLimitsJointVelocity``; the tolerance term is what libfranka subtracts so that a
few lost packets cannot push a command over the robot's own check.
"""

import numpy as np

POSITION_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
POSITION_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
VELOCITY_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
ACCELERATION_MAX = np.full(7, 10.0)
JERK_MAX = np.full(7, 5000.0)
TORQUE_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

LIMIT_EPS = 1e-3
TOL_NUMBER_PACKETS_LOST = 3.0
VELOCITY_TOLERANCE = LIMIT_EPS + TOL_NUMBER_PACKETS_LOST * 1e-3 * 10.0

# (vmax, offset, gain, bound) per joint for the upper and the lower velocity limit.
_UPPER = np.array([
    [2.62, 0.30, 12.0, 2.75010],
    [2.62, 0.20, 5.17, 1.79180],
    [2.62, 0.20, 7.00, 2.90650],
    [2.62, 0.30, 8.00, -0.1458],
    [5.26, 0.35, 34.0, 2.81010],
    [4.18, 0.35, 11.0, 4.52050],
    [5.26, 0.35, 34.0, 3.01960],
])
_LOWER = np.array([
    [2.62, 0.30, 12.0, 2.750100],
    [2.62, 0.20, 5.17, 1.791800],
    [2.62, 0.20, 7.00, 2.906500],
    [2.62, 0.30, 8.00, 3.048100],
    [5.26, 0.35, 34.0, 2.810100],
    [4.18, 0.35, 11.0, -0.54092],
    [5.26, 0.35, 34.0, 3.019600],
])


def upper_velocity_limits(q):
    """Positive velocity limit for each joint at configuration ``q`` (..., 7)."""
    q = np.asarray(q, dtype=float)
    vmax, offset, gain, bound = _UPPER.T
    inner = np.maximum(0.0, gain * (bound - q))
    return np.minimum(vmax, np.maximum(0.0, -offset + np.sqrt(inner))) - VELOCITY_TOLERANCE


def lower_velocity_limits(q):
    """Negative velocity limit for each joint at configuration ``q`` (..., 7)."""
    q = np.asarray(q, dtype=float)
    vmax, offset, gain, bound = _LOWER.T
    inner = np.maximum(0.0, gain * (bound + q))
    return np.maximum(-vmax, np.minimum(0.0, offset - np.sqrt(inner))) + VELOCITY_TOLERANCE


def check(t, q, qd, qdd, qddd, velocity_margin=1.0, acceleration_margin=1.0, jerk_margin=1.0):
    """Compare a dense trajectory against the limits.

    Returns a dict with per-joint peak values, the peak fraction of each (margin-scaled) limit,
    and a list of human-readable violations. A fraction above 1.0 is a violation.
    """
    q = np.asarray(q)
    upper = upper_velocity_limits(q)
    lower = lower_velocity_limits(q)
    velocity_fraction = np.where(
        qd >= 0.0, qd / (velocity_margin * upper), qd / (velocity_margin * lower)
    )
    velocity_fraction = np.nan_to_num(velocity_fraction, nan=np.inf, posinf=np.inf)
    acceleration_fraction = np.abs(qdd) / (acceleration_margin * ACCELERATION_MAX)
    jerk_fraction = np.abs(qddd) / (jerk_margin * JERK_MAX)

    position_low = (q < POSITION_LOWER)
    position_high = (q > POSITION_UPPER)

    violations = []
    for j in range(7):
        name = 'joint%d' % (j + 1)
        if position_low[:, j].any() or position_high[:, j].any():
            violations.append(
                '%s position leaves [%.4f, %.4f] (min %.4f, max %.4f)'
                % (name, POSITION_LOWER[j], POSITION_UPPER[j], q[:, j].min(), q[:, j].max())
            )
        k = int(np.argmax(velocity_fraction[:, j]))
        if velocity_fraction[k, j] > 1.0:
            violations.append(
                '%s velocity %.3f rad/s at t=%.3f s is %.0f %% of the (margin-scaled) limit'
                % (name, qd[k, j], t[k], 100.0 * velocity_fraction[k, j])
            )
        k = int(np.argmax(acceleration_fraction[:, j]))
        if acceleration_fraction[k, j] > 1.0:
            violations.append(
                '%s acceleration %.2f rad/s^2 at t=%.3f s is %.0f %% of the (margin-scaled) limit'
                % (name, qdd[k, j], t[k], 100.0 * acceleration_fraction[k, j])
            )
        k = int(np.argmax(jerk_fraction[:, j]))
        if jerk_fraction[k, j] > 1.0:
            violations.append(
                '%s jerk %.0f rad/s^3 at t=%.3f s is %.0f %% of the (margin-scaled) limit'
                % (name, qddd[k, j], t[k], 100.0 * jerk_fraction[k, j])
            )

    return {
        'position_min': q.min(axis=0).tolist(),
        'position_max': q.max(axis=0).tolist(),
        'velocity_peak': np.abs(qd).max(axis=0).tolist(),
        'acceleration_peak': np.abs(qdd).max(axis=0).tolist(),
        'jerk_peak': np.abs(qddd).max(axis=0).tolist(),
        'velocity_fraction': velocity_fraction.max(axis=0).tolist(),
        'acceleration_fraction': acceleration_fraction.max(axis=0).tolist(),
        'jerk_fraction': jerk_fraction.max(axis=0).tolist(),
        'margins': {
            'velocity': velocity_margin,
            'acceleration': acceleration_margin,
            'jerk': jerk_margin,
        },
        'violations': violations,
        'ok': not violations,
    }


def required_time_scale(report):
    """Smallest uniform slow-down factor (>= 1) that brings a checked trajectory within limits.

    Slowing time by s scales velocities by 1/s, accelerations by 1/s^2 and jerks by 1/s^3
    while positions - and with them the position-dependent velocity limits - stay the same.
    """
    rv = max(report['velocity_fraction'])
    ra = max(report['acceleration_fraction'])
    rj = max(report['jerk_fraction'])
    return float(max(1.0, rv, np.sqrt(max(ra, 0.0)), np.cbrt(max(rj, 0.0))))
