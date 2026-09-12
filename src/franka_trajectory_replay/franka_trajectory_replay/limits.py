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

"""FR3 joint limits, as franka_description's ``robots/fr3/joint_limits.yaml``.

The velocity limit depends on the joint position (the arm has to be able to brake before the
position limit):

    dq_max(q) = min(vmax, max(0, -velocity_offset + sqrt(2 * deceleration_limit * (q_max - q))))

``velocity_offset`` and ``deceleration_limit`` come straight from the description shipped with
this workspace, and so do ``POSITION_LOWER`` / ``POSITION_UPPER``. That pairing matters: the
offset is only meaningful next to the position limit it was derived for. libfranka's
``computeUpperLimitsJointVelocity`` hard-codes an older generation of FR3 limits (joint 6's
bound 4.52050 encodes the retired 4.5169 rad limit, not today's 4.6216), and mixing those
constants with today's position table shrinks the envelope to zero well inside the usable
range - joint 6 loses its top 0.11 rad, which is a region hand-guided demonstrations reach
routinely. Keep both halves from the same file.

The tolerance term is what libfranka subtracts so that a few lost packets cannot push a
command over the robot's own check. It shrinks the envelope toward zero and never past it:
standing still is legal everywhere inside the position limits.
"""

import numpy as np

POSITION_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
POSITION_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
VELOCITY_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
ACCELERATION_MAX = np.full(7, 10.0)
JERK_MAX = np.full(7, 5000.0)
TORQUE_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# position_based_velocity_limits from the same joint_limits.yaml.
VELOCITY_OFFSET = np.array([0.6520, 0.2500, 0.2005, 0.3542, 0.5738, 0.4885, 0.4592])
DECELERATION_LIMIT = np.array([6.0, 2.585, 3.5, 4.0, 17.0, 5.5, 17.0])

LIMIT_EPS = 1e-3
TOL_NUMBER_PACKETS_LOST = 3.0
VELOCITY_TOLERANCE = LIMIT_EPS + TOL_NUMBER_PACKETS_LOST * 1e-3 * 10.0


def _envelope(distance_to_limit):
    """Braking envelope magnitude for a (..., 7) array of distances to the position limit."""
    braking = np.sqrt(np.maximum(0.0, 2.0 * DECELERATION_LIMIT * distance_to_limit))
    allowed = np.minimum(VELOCITY_MAX, np.maximum(0.0, braking - VELOCITY_OFFSET))
    # Clamped at zero rather than allowed to go negative: an envelope that excludes zero would
    # make holding a pose illegal, which the robot does not ask for.
    return np.maximum(0.0, allowed - VELOCITY_TOLERANCE)


def upper_velocity_limits(q):
    """Positive velocity limit for each joint at configuration ``q`` (..., 7)."""
    return _envelope(POSITION_UPPER - np.asarray(q, dtype=float))


def lower_velocity_limits(q):
    """Negative velocity limit for each joint at configuration ``q`` (..., 7)."""
    return -_envelope(np.asarray(q, dtype=float) - POSITION_LOWER)


def check(t, q, qd, qdd, qddd, velocity_margin=1.0, acceleration_margin=1.0, jerk_margin=1.0,
          max_velocity=None):
    """Compare a dense trajectory against the limits.

    ``max_velocity`` (rad/s, scalar or per joint) is an extra ceiling on top of the FR3's own
    envelope - a house speed limit for replay, well under what the arm would allow. Unlike the
    position-dependent part it is a pure speed limit, so slowing the trajectory down always
    satisfies it and ``required_time_scale`` will quote the factor that does.

    Returns a dict with per-joint peak values, the peak fraction of each (margin-scaled) limit,
    and a list of human-readable violations. A fraction above 1.0 is a violation; an infinite
    one is a violation no amount of slowing down can remove, because the joint is at a position
    where the envelope has already closed to zero and the limit it breaks is a function of
    position rather than of speed.
    """
    q = np.asarray(q)
    upper = velocity_margin * upper_velocity_limits(q)
    lower = velocity_margin * lower_velocity_limits(q)

    # The braking zone is a property of where the arm is, so it is read off the FR3's own
    # envelope before the house limit narrows it - otherwise a low --max-joint-speed would
    # report the arm as cornered against a position limit it is nowhere near.
    braking_zone = (upper <= 0.0) | (lower >= 0.0)
    if max_velocity is not None:
        ceiling = np.abs(np.asarray(max_velocity, dtype=float))
        upper = np.minimum(upper, ceiling)
        lower = np.maximum(lower, -ceiling)

    # Right at a position limit the braking-distance formula closes the envelope [lower, upper]
    # onto zero: the joint may hold its pose, but it may not move. Slowing time maps qd -> qd/s
    # while q, and with it the envelope, is unchanged, so no finite scaling brings a moving
    # sample there back inside. Those count as infinitely out of limits rather than a large
    # finite multiple, which is what keeps `required_time_scale` from quoting a factor that
    # cannot work. A sample that is genuinely at rest is in limits and scores a zero fraction.
    with np.errstate(divide='ignore', invalid='ignore'):
        against_upper = np.divide(qd, upper, out=np.full(np.shape(qd), np.inf), where=upper > 0.0)
        against_lower = np.divide(qd, lower, out=np.full(np.shape(qd), np.inf), where=lower < 0.0)
    velocity_fraction = np.where(qd >= 0.0, against_upper, against_lower)
    velocity_fraction = np.where(qd == 0.0, 0.0, velocity_fraction)
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
        # Only moving inside a closed envelope is a violation. Sitting still there is what the
        # arm is supposed to do at the end of its travel, and the limit check used to reject it.
        stuck = braking_zone[:, j] & ~np.isfinite(velocity_fraction[:, j])
        if stuck.any():
            k = int(np.argmax(np.where(stuck, np.abs(qd[:, j]), -np.inf)))
            if upper[k, j] <= 0.0:
                edge, gap = POSITION_UPPER[j], POSITION_UPPER[j] - q[k, j]
            else:
                edge, gap = POSITION_LOWER[j], q[k, j] - POSITION_LOWER[j]
            violations.append(
                '%s moves while its velocity envelope is closed, for %d of %d samples: at '
                't=%.3f s it is %.4f rad, %.4f rad from the %.4f rad position limit, where the '
                '(margin-scaled) envelope is [%+.3f, %+.3f] rad/s and the trajectory asks for '
                '%+.3f rad/s. No time scaling fixes this - that close to the limit the arm may '
                'hold its pose but not move, whatever the speed.'
                % (name, int(stuck.sum()), stuck.size, t[k], q[k, j], gap, edge,
                   lower[k, j], upper[k, j], qd[k, j])
            )
        # Reported separately from the braking zone, and only where scaling is the answer.
        finite = np.where(np.isfinite(velocity_fraction[:, j]), velocity_fraction[:, j], -np.inf)
        k = int(np.argmax(finite))
        if finite[k] > 1.0:
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
        # Per joint: did it ever move while its envelope was closed? Those samples carry an
        # infinite velocity fraction, so `scalable` is "would slowing down help at all".
        'braking_zone': (braking_zone & ~np.isfinite(velocity_fraction)).any(axis=0).tolist(),
        'scalable': bool(np.isfinite(velocity_fraction).all()),
        'max_velocity': None if max_velocity is None else np.broadcast_to(
            np.abs(np.asarray(max_velocity, dtype=float)), (7,)).tolist(),
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
    That is exactly why this returns ``inf`` for a trajectory that enters a braking zone:
    there the envelope excludes zero, so shrinking velocities toward zero never reaches it.
    Callers must check ``report['scalable']`` (or this value being finite) before quoting a
    factor to the operator, or they will offer a slow-down that cannot work.
    """
    rv = max(report['velocity_fraction'])
    ra = max(report['acceleration_fraction'])
    rj = max(report['jerk_fraction'])
    return float(max(1.0, rv, np.sqrt(max(ra, 0.0)), np.cbrt(max(rj, 0.0))))
