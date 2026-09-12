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

"""Forward kinematics of the FR3 flange, plus the pose arithmetic the TCP metrics need.

Two implementations:

* :func:`flange_pose` - closed form from the FR3 modified-DH parameters (Franka's published
  table). Needs only numpy, so it also runs on a laptop without ROS, and it agrees with the
  URDF's ``fr3_link8`` to floating point precision (test/python/test_kinematics.py checks this
  against pinocchio on the real URDF).
* :class:`UrdfKinematics` - pinocchio on the URDF captured from the running robot, when you
  want the exact model that was live (a different end effector, for example).
"""

import numpy as np

# Modified DH (Craig) parameters of the FR3: a_{i-1}, d_i, alpha_{i-1}; the last row is the
# fixed flange transform (fr3_link8 = flange, 0.107 m along z7).
_DH = np.array([
    # a        d       alpha
    [0.0,     0.333,   0.0],
    [0.0,     0.0,    -np.pi / 2],
    [0.0,     0.316,   np.pi / 2],
    [0.0825,  0.0,     np.pi / 2],
    [-0.0825, 0.384,  -np.pi / 2],
    [0.0,     0.0,     np.pi / 2],
    [0.088,   0.0,     np.pi / 2],
    [0.0,     0.107,   0.0],
])

READY_POSE = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])


def _dh_transform(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def flange_transform(q):
    """4x4 transform of the flange (fr3_link8) in the base frame (fr3_link0)."""
    q = np.asarray(q, dtype=float)
    transform = np.eye(4)
    for i in range(7):
        transform = transform @ _dh_transform(_DH[i, 0], _DH[i, 1], _DH[i, 2], q[i])
    transform = transform @ _dh_transform(_DH[7, 0], _DH[7, 1], _DH[7, 2], 0.0)
    return transform


def flange_transforms(q_batch):
    """(N, 4, 4) flange transforms for (N, 7) joint positions; the batched flange_transform."""
    q_batch = np.atleast_2d(np.asarray(q_batch, dtype=float))
    n = len(q_batch)
    transform = np.tile(np.eye(4), (n, 1, 1))
    for i in range(8):
        a, d, alpha = _DH[i]
        theta = q_batch[:, i] if i < 7 else np.zeros(n)
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        step = np.zeros((n, 4, 4))
        step[:, 0, 0] = ct
        step[:, 0, 1] = -st
        step[:, 0, 3] = a
        step[:, 1, 0] = st * ca
        step[:, 1, 1] = ct * ca
        step[:, 1, 2] = -sa
        step[:, 1, 3] = -d * sa
        step[:, 2, 0] = st * sa
        step[:, 2, 1] = ct * sa
        step[:, 2, 2] = ca
        step[:, 2, 3] = d * ca
        step[:, 3, 3] = 1.0
        transform = transform @ step
    return transform


def tool_transform(offset_xyz=(0.0, 0.0, 0.0), offset_rpy=(0.0, 0.0, 0.0)):
    """Fixed flange->TCP transform from a translation and roll/pitch/yaw (URDF convention)."""
    roll, pitch, yaw = offset_rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(offset_xyz, dtype=float)
    return transform


def flange_pose(q, tool=None):
    """(position (3,), quaternion xyzw (4,)) of the flange, or of the TCP if ``tool`` is given."""
    transform = flange_transform(q)
    if tool is not None:
        transform = transform @ tool
    return transform[:3, 3].copy(), matrix_to_quaternion(transform[:3, :3])


def flange_poses(q_batch, tool=None):
    """Batch version: (N, 3) positions and (N, 4) quaternions, sign-continuous along the batch."""
    q_batch = np.asarray(q_batch, dtype=float)
    positions = np.zeros((len(q_batch), 3))
    quaternions = np.zeros((len(q_batch), 4))
    for k, q in enumerate(q_batch):
        positions[k], quaternions[k] = flange_pose(q, tool)
        if k > 0 and np.dot(quaternions[k], quaternions[k - 1]) < 0.0:
            quaternions[k] = -quaternions[k]
    return positions, quaternions


def flange_jacobian(q, tool=None, step=1e-6):
    """Geometric Jacobian (6 x 7: linear then angular, base frame) of the flange / TCP at q.

    Central finite differences of the forward kinematics: exact to ~1e-10 for a 1e-6 step, and
    it needs nothing but the DH model.
    """
    q = np.asarray(q, dtype=float)
    jacobian = np.zeros((6, 7))
    for j in range(7):
        forward = q.copy()
        backward = q.copy()
        forward[j] += step
        backward[j] -= step
        t_f = flange_transform(forward)
        t_b = flange_transform(backward)
        if tool is not None:
            t_f = t_f @ tool
            t_b = t_b @ tool
        jacobian[:3, j] = (t_f[:3, 3] - t_b[:3, 3]) / (2 * step)
        # angular: skew part of dR R^T
        d_rotation = (t_f[:3, :3] - t_b[:3, :3]) / (2 * step) @ t_f[:3, :3].T
        jacobian[3:, j] = [d_rotation[2, 1], d_rotation[0, 2], d_rotation[1, 0]]
    return jacobian


def tcp_error_contributions(q_ref, q_meas, tool=None):
    """Per-joint contribution of the joint tracking error to the TCP error, sample by sample.

    Returns (linear (N, 7, 3), angular (N, 7, 3), error (N, 3), angular_error (N, 3)) where
    linear[k, j] = J_p(q_ref[k])[:, j] * (q_ref[k, j] - q_meas[k, j]). Summing over j gives the
    first-order TCP error; ``error`` is the exact FK difference for comparison.
    """
    q_ref = np.asarray(q_ref, dtype=float)
    q_meas = np.asarray(q_meas, dtype=float)
    n = len(q_ref)
    linear = np.zeros((n, 7, 3))
    angular = np.zeros((n, 7, 3))
    error = np.zeros((n, 3))
    angular_error = np.zeros((n, 3))
    for k in range(n):
        jacobian = flange_jacobian(q_ref[k], tool)
        dq = q_ref[k] - q_meas[k]
        linear[k] = (jacobian[:3] * dq).T
        angular[k] = (jacobian[3:] * dq).T
        t_ref = flange_transform(q_ref[k])
        t_meas = flange_transform(q_meas[k])
        if tool is not None:
            t_ref = t_ref @ tool
            t_meas = t_meas @ tool
        error[k] = t_ref[:3, 3] - t_meas[:3, 3]
        relative = t_ref[:3, :3] @ t_meas[:3, :3].T
        angular_error[k] = [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                            relative[1, 0] - relative[0, 1]]
        angular_error[k] *= 0.5
    return linear, angular, error, angular_error


def matrix_to_quaternion(rotation):
    """Rotation matrix to quaternion (x, y, z, w), Shepperd's method."""
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([x, y, z, w])
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_angle(q_a, q_b):
    """Angle (rad) of the rotation taking q_a to q_b. Batched over the leading axis."""
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    dot = np.abs(np.sum(q_a * q_b, axis=-1))
    dot = np.clip(dot / (np.linalg.norm(q_a, axis=-1) * np.linalg.norm(q_b, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def column_major_to_transform(values):
    """libfranka packs O_T_EE column-major into 16 doubles."""
    return np.asarray(values, dtype=float).reshape(4, 4, order='F')


class UrdfKinematics:
    """Forward kinematics on a URDF via pinocchio (optional dependency)."""

    def __init__(self, urdf_path, joint_names):
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.joint_names = list(joint_names)
        self._q_indices = []
        for name in self.joint_names:
            if not self.model.existJointName(name):
                raise KeyError('joint %r is not in the URDF' % name)
            self._q_indices.append(self.model.joints[self.model.getJointId(name)].idx_q)
        self._q_indices = np.asarray(self._q_indices, dtype=int)
        self._q_neutral = pin.neutral(self.model)

    def frame_exists(self, frame):
        return self.model.existFrame(frame)

    def pose(self, joint_positions, ee_link, base_frame):
        pin = self._pin
        q = self._q_neutral.copy()
        q[self._q_indices] = np.asarray(joint_positions, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        base = self.data.oMf[self.model.getFrameId(base_frame)]
        ee = self.data.oMf[self.model.getFrameId(ee_link)]
        transform = base.actInv(ee)
        return np.array(transform.translation, dtype=float), matrix_to_quaternion(
            np.array(transform.rotation, dtype=float)
        )
