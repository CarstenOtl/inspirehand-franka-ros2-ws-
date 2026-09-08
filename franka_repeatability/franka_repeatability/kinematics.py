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

"""Forward kinematics and the pose statistics the repeatability report is built from."""

import numpy as np


class ForwardKinematics:
    """Forward kinematics on the exact URDF that was used during the run.

    The URDF is captured from the running robot_state_publisher rather than rebuilt from xacro,
    so the model here is the same one move_group solved IK against - including whatever end
    effector and load configuration was active.
    """

    def __init__(self, urdf_path, joint_names):
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.joint_names = list(joint_names)

        # Map our seven joints onto their slots in pinocchio's configuration vector. Doing this
        # by name rather than by position keeps the result correct if the model also carries
        # gripper joints or a mobile base.
        self._q_indices = []
        for name in self.joint_names:
            if not self.model.existJointName(name):
                raise KeyError('joint %r is not in the URDF' % name)
            joint_id = self.model.getJointId(name)
            self._q_indices.append(self.model.joints[joint_id].idx_q)
        self._q_indices = np.asarray(self._q_indices, dtype=int)
        self._q_neutral = pin.neutral(self.model)

    def frame_exists(self, frame):
        return self.model.existFrame(frame)

    def pose(self, joint_positions, ee_link, base_frame):
        """Return (translation, quaternion xyzw) of ``ee_link`` relative to ``base_frame``.

        Both frames are resolved explicitly and composed, so the result does not depend on
        where the URDF root happens to sit (the Franka description roots at ``world``).
        """
        pin = self._pin
        q = self._q_neutral.copy()
        q[self._q_indices] = np.asarray(joint_positions, dtype=float)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        for frame in (ee_link, base_frame):
            if not self.model.existFrame(frame):
                raise KeyError('frame %r is not in the URDF' % frame)

        base_to_world = self.data.oMf[self.model.getFrameId(base_frame)]
        ee_to_world = self.data.oMf[self.model.getFrameId(ee_link)]
        transform = base_to_world.actInv(ee_to_world)

        return np.array(transform.translation, dtype=float), matrix_to_quaternion(
            np.array(transform.rotation, dtype=float)
        )


def canonical_quaternion(quaternion):
    """Normalise a quaternion to one deterministic representative of its rotation.

    ``q`` and ``-q`` are the same rotation, so a convention is needed. ``w >= 0`` is the usual
    one, but the flange-down orientations this package is aimed at sit at ``w == 0`` exactly,
    where that rule decides nothing and the sign flips between otherwise identical readings.
    Falling back to the largest component keeps the printed pose stable.
    """
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    if abs(quaternion[3]) > 1e-6:
        return quaternion if quaternion[3] > 0.0 else -quaternion
    largest = int(np.argmax(np.abs(quaternion)))
    return quaternion if quaternion[largest] > 0.0 else -quaternion


def matrix_to_quaternion(rotation):
    """Rotation matrix to quaternion in (x, y, z, w), with a non-negative scalar part."""
    trace = np.trace(rotation)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (rotation[j, i] + rotation[i, j]) / s
        q[k] = (rotation[k, i] + rotation[i, k]) / s
        w = (rotation[k, j] - rotation[j, k]) / s
        x, y, z = q
    return canonical_quaternion(np.array([x, y, z, w], dtype=float))


def quaternion_mean(quaternions):
    """Average orientation of a set of quaternions (Markley's eigenvector method).

    Componentwise averaging is wrong for rotations and, worse, quietly plausible for the tight
    clusters a repeatability run produces. This returns the rotation that minimises the summed
    squared chordal distance instead.
    """
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    # Sign is arbitrary; align every sample with the first so the accumulation does not cancel.
    signs = np.sign(q @ q[0])
    signs[signs == 0.0] = 1.0
    q = q * signs[:, None]
    accumulator = q.T @ q
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    return canonical_quaternion(eigenvectors[:, int(np.argmax(eigenvalues))])


def quaternion_angle(a, b):
    """Rotation angle in radians between two (x, y, z, w) quaternions."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return 2.0 * np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))


def position_repeatability(positions):
    """ISO 9283 style position repeatability for the visits to one pose.

    ``RP_l = l_bar + 3 * S_l``, where ``l_i`` is the distance of visit ``i`` from the barycentre
    of all visits. Reported alongside the raw spread so a single outlier stays visible instead
    of being absorbed into the standard deviation.
    """
    points = np.asarray(positions, dtype=float)
    barycentre = points.mean(axis=0)
    distances = np.linalg.norm(points - barycentre, axis=1)
    mean_distance = float(distances.mean())
    # Sample standard deviation; ISO 9283 divides by n-1.
    std_distance = float(distances.std(ddof=1)) if len(distances) > 1 else 0.0

    return {
        'samples': int(len(points)),
        'barycentre_m': barycentre.tolist(),
        'repeatability_rp_m': mean_distance + 3.0 * std_distance,
        'mean_distance_m': mean_distance,
        'std_distance_m': std_distance,
        'max_distance_m': float(distances.max()),
        'per_axis_std_m': points.std(axis=0, ddof=1).tolist()
        if len(points) > 1
        else [0.0, 0.0, 0.0],
        'per_axis_range_m': (points.max(axis=0) - points.min(axis=0)).tolist(),
        'distances_m': distances.tolist(),
    }


def orientation_repeatability(quaternions):
    """Angular spread of the visits to one pose, about the mean orientation."""
    quaternions = np.asarray(quaternions, dtype=float)
    mean = quaternion_mean(quaternions)
    angles = np.array([quaternion_angle(q, mean) for q in quaternions])
    mean_angle = float(angles.mean())
    std_angle = float(angles.std(ddof=1)) if len(angles) > 1 else 0.0

    return {
        'mean_quaternion_xyzw': mean.tolist(),
        'repeatability_rad': mean_angle + 3.0 * std_angle,
        'mean_angle_rad': mean_angle,
        'std_angle_rad': std_angle,
        'max_angle_rad': float(angles.max()),
        'angles_rad': angles.tolist(),
    }
