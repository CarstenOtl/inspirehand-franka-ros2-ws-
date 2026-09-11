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

"""Pose streams for the Cartesian impedance replay controller.

The joint-space pipeline (``prepare``) stays the first stage: it is the part validated on
hardware, it is where the FR3 velocity/acceleration/jerk limits and the automatic time
scaling live, and a pose stream produced by forward kinematics of a limit-checked joint
stream describes a motion the arm can make. This module turns such a stream into the
Cartesian reference the controller plays, checks it against libfranka's Cartesian limits,
packs it into the controller's messages, and holds the two frame checks the runner performs
before switching controllers.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from franka_trajectory_replay import kinematics

# libfranka rate_limiting.h, FR3 values (each minus the packet-loss tolerance libfranka
# subtracts). In torque mode the robot does not enforce these on a reference; they are
# sanity bounds on the pose stream.
LIMIT_EPS = 1e-3
TOL_NUMBER_PACKETS_LOST = 3.0
DELTA_T = 1e-3
TRANSLATIONAL_ACCELERATION_MAX = 9.0 - LIMIT_EPS
TRANSLATIONAL_JERK_MAX = 4500.0 - LIMIT_EPS
TRANSLATIONAL_VELOCITY_MAX = 3.0 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * 9.0
ROTATIONAL_ACCELERATION_MAX = 17.0 - LIMIT_EPS
ROTATIONAL_JERK_MAX = 8500.0 - LIMIT_EPS
ROTATIONAL_VELOCITY_MAX = 2.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * 17.0


@dataclass
class PreparedCartesian:
    t: np.ndarray            # (M,) seconds from the start of the command stream
    p: np.ndarray            # (M, 3) position, base frame, metres
    quat: np.ndarray         # (M, 4) orientation x y z w, sign-continuous along the stream
    v: np.ndarray            # (M, 3) linear velocity
    w: np.ndarray            # (M, 3) angular velocity, base frame
    q_null: np.ndarray       # (M, 7) nullspace joint configuration, or None
    joint: object = None     # the Prepared joint stream this was derived from, or None
    tool: np.ndarray = None  # 4x4 flange-to-target transform the stream assumes
    params: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    @property
    def duration(self):
        return float(self.t[-1])

    @property
    def rate(self):
        return float((len(self.t) - 1) / self.t[-1]) if len(self.t) > 1 else 0.0


def make_sign_continuous(quat):
    """Flip quaternion signs so consecutive samples never sit on opposite hemispheres."""
    quat = np.array(quat, dtype=float, copy=True)
    for k in range(1, len(quat)):
        if np.dot(quat[k], quat[k - 1]) < 0.0:
            quat[k] = -quat[k]
    return quat


def angular_velocity(quat, dt):
    """(M, 3) base-frame angular velocity of a quaternion stream, central differences."""
    rotations = Rotation.from_quat(quat)
    if len(quat) < 2:
        return np.zeros((len(quat), 3))
    forward = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt  # over [k, k+1]
    w = np.zeros((len(quat), 3))
    w[0] = forward[0]
    w[-1] = forward[-1]
    w[1:-1] = 0.5 * (forward[:-1] + forward[1:])
    return w


def rotation_angles(quat):
    """(M-1,) angle of the rotation between consecutive samples."""
    rotations = Rotation.from_quat(quat)
    return np.linalg.norm((rotations[1:] * rotations[:-1].inv()).as_rotvec(), axis=1)


def from_joint_stream(prepared, tool=None):
    """Forward kinematics of a prepared joint stream, sample by sample, at its own rate.

    ``tool`` is the fixed flange-to-target transform (identity: the flange itself, which is
    what O_T_EE is when the robot's F_T_EE is identity). The nullspace target is the joint
    stream itself.
    """
    t = np.asarray(prepared.t, dtype=float)
    q = np.asarray(prepared.q, dtype=float)
    if len(t) < 2:
        raise ValueError('a prepared stream needs at least two samples')
    dt = float(np.median(np.diff(t)))
    tool = np.eye(4) if tool is None else np.asarray(tool, dtype=float)
    transforms = kinematics.flange_transforms(q) @ tool
    p = transforms[:, :3, 3].copy()
    quat = make_sign_continuous(Rotation.from_matrix(transforms[:, :3, :3]).as_quat())
    v = np.gradient(p, dt, axis=0)
    w = angular_velocity(quat, dt)
    return PreparedCartesian(
        t=t, p=p, quat=quat, v=v, w=w, q_null=q, joint=prepared, tool=tool,
        params={
            'rate': float(prepared.rate), 'source': 'fk',
            'tool_xyz': tool[:3, 3].tolist(),
            'time_scale': float(prepared.params.get('time_scale', 1.0)),
            'capture_start_index': int(prepared.params.get('capture_start_index', 0)),
        },
    )


def check_cartesian_limits(prepared, velocity_margin=1.0, acceleration_margin=0.5,
                           jerk_margin=0.5):
    """Peak linear/angular velocity, acceleration and jerk against libfranka's limits.

    ``margin`` scales the limit: a fraction above 1.0 of ``limit * margin`` is a violation.
    Sets and returns ``prepared.report``.
    """
    dt = float(np.median(np.diff(prepared.t)))
    a = np.gradient(prepared.v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    alpha = np.gradient(prepared.w, dt, axis=0)
    zeta = np.gradient(alpha, dt, axis=0)
    peaks = {
        'linear_velocity': float(np.linalg.norm(prepared.v, axis=1).max()),
        'linear_acceleration': float(np.linalg.norm(a, axis=1).max()),
        'linear_jerk': float(np.linalg.norm(j, axis=1).max()),
        'angular_velocity': float(np.linalg.norm(prepared.w, axis=1).max()),
        'angular_acceleration': float(np.linalg.norm(alpha, axis=1).max()),
        'angular_jerk': float(np.linalg.norm(zeta, axis=1).max()),
    }
    limits = {
        'linear_velocity': (TRANSLATIONAL_VELOCITY_MAX, velocity_margin, 'm/s'),
        'linear_acceleration': (TRANSLATIONAL_ACCELERATION_MAX, acceleration_margin, 'm/s^2'),
        'linear_jerk': (TRANSLATIONAL_JERK_MAX, jerk_margin, 'm/s^3'),
        'angular_velocity': (ROTATIONAL_VELOCITY_MAX, velocity_margin, 'rad/s'),
        'angular_acceleration': (ROTATIONAL_ACCELERATION_MAX, acceleration_margin, 'rad/s^2'),
        'angular_jerk': (ROTATIONAL_JERK_MAX, jerk_margin, 'rad/s^3'),
    }
    report = {'peaks': peaks, 'fractions': {}, 'violations': [], 'limits': {},
              'margins': {'velocity': velocity_margin, 'acceleration': acceleration_margin,
                          'jerk': jerk_margin}}
    for name, (limit, margin, unit) in limits.items():
        allowed = limit * margin
        fraction = peaks[name] / allowed if allowed > 0 else float('inf')
        report['fractions'][name] = float(fraction)
        report['limits'][name] = float(limit)
        if fraction > 1.0:
            report['violations'].append(
                '%s peak %.3f %s exceeds %.3f %s (%.2f x the limit, margin %.2f)'
                % (name.replace('_', ' '), peaks[name], unit, allowed, unit, fraction, margin))
    report['path_length_m'] = float(np.linalg.norm(np.diff(prepared.p, axis=0), axis=1).sum())
    report['rotation_total_rad'] = float(rotation_angles(prepared.quat).sum())
    report['ok'] = not report['violations']
    prepared.report = report
    return report


def summarize_cartesian(prepared):
    """Short human-readable summary of a pose stream and its Cartesian limit check."""
    lines = [
        'cartesian: %d samples at %.0f Hz, %.2f s, tool offset %s m, path %.3f m, rotation %.2f rad'
        % (len(prepared.t), prepared.rate, prepared.duration,
           np.array2string(np.asarray(prepared.params.get('tool_xyz', [0, 0, 0])), precision=4),
           prepared.report.get('path_length_m', float('nan')),
           prepared.report.get('rotation_total_rad', float('nan'))),
    ]
    report = prepared.report
    if report:
        lines.append('  %-22s %10s %10s %7s' % ('', 'peak', 'limit', '%'))
        for name, peak in report['peaks'].items():
            lines.append('  %-22s %10.3f %10.3f %6.0f%%' % (
                name.replace('_', ' '), peak, report['limits'][name],
                100.0 * report['fractions'][name]))
        margins = report['margins']
        lines.append('  percentages are of the libfranka Cartesian limit times the margin '
                     '(v %.2f, a %.2f, j %.2f)' % (margins['velocity'], margins['acceleration'],
                                                   margins['jerk']))
        if report['violations']:
            lines.append('  VIOLATIONS:')
            lines.extend('    - ' + text for text in report['violations'])
        else:
            lines.append('  within limits')
    return '\n'.join(lines)


# --- frame checks ---------------------------------------------------------------------------

def pose_to_matrix(pose):
    """geometry_msgs Pose (or PoseStamped) to a 4x4 transform."""
    if hasattr(pose, 'pose'):
        pose = pose.pose
    transform = np.eye(4)
    transform[:3, :3] = kinematics.quaternion_to_matrix(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def transform_error(a, b):
    """(position error m, rotation angle rad) between two 4x4 transforms."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    position = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    relative = a[:3, :3].T @ b[:3, :3]
    angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    return position, angle


def check_end_effector_frame(f_t_ee, tool, tolerance_m=1e-4, tolerance_rad=1e-4):
    """The robot's F_T_EE must be the tool the stream was generated for.

    A mismatch would put every waypoint in the wrong place by the offset, so this refuses
    rather than compensates: the fix is ``set_tcp_frame`` (or a matching ``tcp`` block), not
    a silent correction here.
    """
    position, angle = transform_error(f_t_ee, tool)
    if position > tolerance_m or angle > tolerance_rad:
        raise ValueError(
            "the robot's end-effector frame F_T_EE (translation %s m) differs from the tool the "
            "pose stream assumes (%s m) by %.4f m / %.4f rad; set the robot's TCP frame with "
            "service_server/set_tcp_frame or point replay.yaml's tcp block at the configured "
            "end effector" % (np.array2string(np.asarray(f_t_ee)[:3, 3], precision=4),
                              np.array2string(np.asarray(tool)[:3, 3], precision=4),
                              position, angle))
    return position, angle


def check_forward_kinematics(q_measured, o_t_ee, tool, tolerance_m=0.003, tolerance_deg=0.5):
    """FK of the measured joints, through ``tool``, must agree with the robot's O_T_EE.

    Independent of the F_T_EE check: a stale DH table or a wrong tool shows up here.
    """
    predicted = kinematics.flange_transform(q_measured) @ np.asarray(tool, dtype=float)
    position, angle = transform_error(predicted, o_t_ee)
    if position > tolerance_m or angle > np.radians(tolerance_deg):
        raise ValueError(
            'forward kinematics of the measured joints disagrees with the robot\'s O_T_EE by '
            '%.4f m / %.2f deg (limits %.4f m / %.2f deg): the DH model, the tool transform '
            'or the joint state is wrong' % (position, np.degrees(angle), tolerance_m,
                                             tolerance_deg))
    return position, angle


# --- messages -------------------------------------------------------------------------------

def _duration_msg(seconds):
    from builtin_interfaces.msg import Duration

    message = Duration()
    message.sec = int(seconds)
    message.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return message


def _fill_pose(pose, position, quat):
    pose.position.x, pose.position.y, pose.position.z = (float(x) for x in position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
        float(x) for x in quat)


def trajectory_message(prepared, frame_id, send_rate=None, stamp=None):
    """Pack a pose stream into a CartesianTrajectory at ``send_rate`` (default: its own rate)."""
    from franka_trajectory_replay_msgs.msg import CartesianTrajectory, CartesianWaypoint

    send_rate = float(send_rate or prepared.rate)
    stride = max(1, int(round(prepared.rate / send_rate)))
    indices = np.arange(0, len(prepared.t), stride)
    if indices[-1] != len(prepared.t) - 1:
        indices = np.append(indices, len(prepared.t) - 1)
    message = CartesianTrajectory()
    if stamp is not None:
        message.header.stamp = stamp
    message.header.frame_id = frame_id
    points = []
    for k in indices:
        point = CartesianWaypoint()
        _fill_pose(point.pose, prepared.p[k], prepared.quat[k])
        point.twist.linear.x, point.twist.linear.y, point.twist.linear.z = (
            float(x) for x in prepared.v[k])
        point.twist.angular.x, point.twist.angular.y, point.twist.angular.z = (
            float(x) for x in prepared.w[k])
        if prepared.q_null is not None:
            point.nullspace_positions = [float(x) for x in prepared.q_null[k]]
        point.time_from_start = _duration_msg(float(prepared.t[k]))
        points.append(point)
    message.points = points
    return message


def goto_message(position, quat, nullspace=None, duration=0.0):
    from franka_trajectory_replay_msgs.msg import CartesianGoto

    message = CartesianGoto()
    _fill_pose(message.pose, position, quat)
    if nullspace is not None:
        message.nullspace_positions = [float(x) for x in nullspace]
    message.duration = float(duration)
    return message
