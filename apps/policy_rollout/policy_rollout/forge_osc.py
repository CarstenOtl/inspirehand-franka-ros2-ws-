"""Framework-free port of ForgeUltra's FR3 threading OSC and joint-PD adapter.

Source of truth: ``reference/forgeUltra`` (branch ``franka-chi``):

- ``forge_ultra/tasks/utils/control.py`` ``compute_dof_torque`` / ``get_pose_error``
- ``forge_ultra/tasks/mdp/robot_control.py`` ``_apply_action`` / ``generate_ctrl_signals``
- ``forge_ultra/tasks/utils/grasp_frame.py`` ``hand_grasp_frame`` /
  ``outward_tilted_grasp_z_direction``
- ``distillation/utils/controllers/impedance_joint_pd_controller.py``
  ``target_from_teacher_control`` (the OSC -> joint-PD adapter)
- ``forge_ultra/tasks/forge_franka_threading_v2/config/tasks/vanilla_threading.yaml``

Everything is plain NumPy on one environment. Quaternions are ``(w, x, y, z)``
like Isaac Sim. Angles are radians, lengths metres, world frame is the
training scene's world (FR3 base at ``(1.2, 0, 0)`` yawed by pi, table at z=0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# --- vanilla_threading.yaml ``env.ctrl`` -----------------------------------
POS_ACTION_BOUNDS = np.array([0.05, 0.05, 0.05])
ROT_ACTION_BOUNDS = np.array([1.0, 1.0, 1.0])
POS_ACTION_THRESHOLD = np.array([0.02, 0.02, 0.02])
ROT_ACTION_THRESHOLD = np.array([0.097, 0.097, 0.097])
DEFAULT_TASK_PROP_GAINS = np.array([565.0, 565.0, 565.0, 28.0, 28.0, 28.0])
KP_NULL = 10.0
KD_NULL = 6.3246
DEFAULT_DEAD_ZONE = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
EMA_FACTOR = 0.2
HAND_EMA_FACTOR_SCALE = 3.0
ARM_TORQUE_LIMIT = 100.0
EE_BASE_ORN_EULER = (math.pi, 0.0, 0.0)
HAND_GRASP_FRAME_RESET_Z_DIRECTION = np.array([0.0, 0.0, -1.0])
HAND_GRASP_FRAME_OUTWARD_TILT_DEG = 15.0

# --- FR3 M24 threading task ---------------------------------------------------
FRANKA_ARM_RESET_JOINTS_M24 = np.array(
    [-0.524466740, 0.250818023, 0.130924487, -2.176566846, 1.679049697, 2.057495751, -1.018287386]
)
ROBOT_BASE_POSITION = np.array([1.2, 0.0, 0.0])
BOLT_BASE_POSITION = np.array([0.61, 0.0, 0.05])
M24_BOLT_HEIGHT = 0.035
M24_BOLT_BASE_HEIGHT = 0.01732
M24_NUT_HEIGHT = 0.01732
M24_THREAD_PITCH = 0.003
BOLT_TIP_OFFSET = M24_BOLT_HEIGHT + M24_BOLT_BASE_HEIGHT
BOLT_TIP_POSITION = BOLT_BASE_POSITION + np.array([0.0, 0.0, BOLT_TIP_OFFSET])

# Hand contract (official Inspire profile).
THREADING_HAND_POSTURE = {
    "thumb_joint_0": 1.14,
    "thumb_joint_1": 0.2,
    "index_joint_0": 0.44,
    "middle_joint_0": 1.0999,
    "ring_joint_0": 1.0999,
    "little_joint_0": 1.0999,
}
PINCH_JOINTS = ("thumb_joint_0", "thumb_joint_1", "index_joint_0")
# forge_ultra/tasks/forge_franka_threading_v2/config/reset_pose_m24.yaml: the
# sequential replay writes this reset hand posture into closed_hand_joint_pos,
# so it is the "grasp" that the relative pinch actions perturb. Verified
# exactly against the recorded PD hand targets of the reference episode.
THREADING_GRASP_POSTURE = {
    "thumb_joint_0": 1.185078562,
    "thumb_joint_1": 0.050614548,
    "index_joint_0": 0.214675498,
    "middle_joint_0": 1.0999,
    "ring_joint_0": 1.0999,
    "little_joint_0": 1.0999,
}
PINCH_RANGES = ((0.0, 1.246165), (0.0, 0.3578 / 1.1425), (0.0, 1.333))
MIMIC_JOINT_MAP = {
    "index_joint_1": ("index_joint_0", 1.1169, 0.0),
    "middle_joint_1": ("middle_joint_0", 1.1169, 0.0),
    "ring_joint_1": ("ring_joint_0", 1.1169, 0.0),
    "little_joint_1": ("little_joint_0", 1.1169, 0.0),
    "thumb_joint_2": ("thumb_joint_1", 1.1425, 0.0),
    "thumb_joint_3": ("thumb_joint_2", 0.7508, 0.0),
}

# Hardware joint impedance gains of the deployment adapter.
IMPEDANCE_JOINT_PD_ARM_STIFFNESS = np.array([300.0, 300.0, 160.0, 160.0, 120.0, 90.0, 80.0])
IMPEDANCE_JOINT_PD_ARM_DAMPING = np.array([7.5, 7.5, 6.0, 7.0, 6.0, 4.5, 4.5])


# --- quaternion helpers (w, x, y, z) -----------------------------------------
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    w = q[0]
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_rotate_inverse(q, v):
    return quat_rotate(quat_conjugate(q), v)


def quat_from_matrix(m):
    m = np.asarray(m, dtype=float)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def matrix_from_quat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, dtype=float)
    half = 0.5 * angle
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def quat_from_euler_xyz(roll, pitch, yaw):
    """isaacsim.core.utils.torch.quat_from_euler_xyz (w, x, y, z)."""

    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ]
    )


def get_euler_xyz(q):
    """isaacsim.core.utils.torch.get_euler_xyz: each angle wrapped to [0, 2pi)."""

    qw, qx, qy, qz = q
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    yaw = math.atan2(siny_cosp, cosy_cosp)
    two_pi = 2.0 * math.pi
    return roll % two_pi, pitch % two_pi, yaw % two_pi


def axis_angle_from_quat(q, eps=1.0e-6):
    """isaaclab.utils.math.axis_angle_from_quat (no shortest-path flip)."""

    mag = np.linalg.norm(q[1:])
    half_angle = math.atan2(mag, q[0])
    angle = 2.0 * half_angle
    if abs(angle) <= eps:
        factor = 0.5 - angle * angle / 48.0
    else:
        factor = math.sin(half_angle) / angle
    return q[1:] / factor


def wrap_yaw(angle):
    return angle - 2.0 * math.pi if angle > math.radians(235.0) else angle


def _unit(v, eps=1.0e-6):
    n = np.linalg.norm(v)
    return v / max(n, eps)


# --- grasp frame ---------------------------------------------------------------
def hand_grasp_frame(thumb_pos, index_pos, approach_direction, eps=1.0e-6):
    """Origin at the thumb/index midpoint, X along thumb->index, Z ~ approach."""

    origin = 0.5 * (thumb_pos + index_pos)
    desired_z = np.asarray(approach_direction, dtype=float)
    if np.linalg.norm(desired_z) > eps:
        desired_z = _unit(desired_z, eps)
    else:
        desired_z = np.array([0.0, 0.0, -1.0])
    world_x = np.array([1.0, 0.0, 0.0])
    world_y = np.array([0.0, 1.0, 0.0])

    x_raw = index_pos - thumb_pos
    if np.linalg.norm(x_raw) > eps:
        x_axis = _unit(x_raw, eps)
    else:
        seed = world_x if abs(desired_z[0]) < 0.9 else world_y
        x_axis = _unit(seed - np.dot(seed, desired_z) * desired_z, eps)
    z_raw = desired_z - np.dot(desired_z, x_axis) * x_axis
    if np.linalg.norm(z_raw) > eps:
        z_axis = _unit(z_raw, eps)
    else:
        seed = world_x if abs(x_axis[0]) < 0.9 else world_y
        z_axis = _unit(seed - np.dot(seed, x_axis) * x_axis, eps)
    y_axis = _unit(np.cross(z_axis, x_axis), eps)
    z_axis = np.cross(x_axis, y_axis)
    return origin, np.stack((x_axis, y_axis, z_axis), axis=-1)


def outward_tilted_grasp_z_direction(
    thumb_pos, index_pos, flange_pos, tilt_deg, base_direction=HAND_GRASP_FRAME_RESET_Z_DIRECTION, eps=1.0e-6
):
    origin = 0.5 * (thumb_pos + index_pos)
    x_axis = _unit(index_pos - thumb_pos, eps)
    base_reference = _unit(np.asarray(base_direction, dtype=float), eps)
    base_z = _unit(base_reference - np.dot(base_reference, x_axis) * x_axis, eps)
    outward = origin - flange_pos
    outward = outward - np.dot(outward, x_axis) * x_axis
    outward = outward - np.dot(outward, base_z) * base_z
    if np.linalg.norm(outward) > eps:
        outward = _unit(outward, eps)
    else:
        outward = _unit(np.cross(base_z, x_axis), eps)
    tilt_cosine = math.cos(math.radians(float(tilt_deg)))
    maximum_base_alignment = max(float(np.dot(base_z, base_reference)), eps)
    base_coefficient = float(np.clip(tilt_cosine / maximum_base_alignment, -1.0, 1.0))
    outward_coefficient = math.sqrt(max(1.0 - base_coefficient * base_coefficient, 0.0))
    return base_coefficient * base_z + outward_coefficient * outward


def reset_z_transport(thumb_pos, index_pos, flange_pos, flange_quat, tilt_deg=0.0):
    """Flange-frame vector of the reset grasp Z axis (what the env stores at reset).

    The config declares a 15 deg outward tilt, but the recorded teacher
    episodes reproduce the recorded grasp frame only with ``tilt_deg=0``
    (0.01 deg mean error over a full replay versus 8.4 deg with 15 deg), so
    the sequential replay evidently stored the untilted direction.
    """

    direction = outward_tilted_grasp_z_direction(thumb_pos, index_pos, flange_pos, tilt_deg)
    _, rot = hand_grasp_frame(thumb_pos, index_pos, direction)
    return quat_rotate_inverse(flange_quat, rot[:, 2])


@dataclass
class GraspFrameState:
    pos: np.ndarray
    quat: np.ndarray
    linvel: np.ndarray
    angvel: np.ndarray
    jacobian: np.ndarray  # (6, 7) world-frame geometric Jacobian at the grasp origin


def grasp_frame_state(
    *, thumb_pos, index_pos, flange_pos, flange_quat, z_transport, thumb_linvel, index_linvel, flange_angvel, flange_jacobian
):
    """Mirror ``_compute_intermediate_values`` for the control-relevant fields."""

    approach = quat_rotate(flange_quat, z_transport)
    pos, rot = hand_grasp_frame(thumb_pos, index_pos, approach)
    r = pos - flange_pos
    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    jac_v = flange_jacobian[0:3] - skew @ flange_jacobian[3:6]
    return GraspFrameState(
        pos=pos,
        quat=quat_from_matrix(rot),
        linvel=0.5 * (thumb_linvel + index_linvel),
        angvel=flange_angvel,
        jacobian=np.vstack((jac_v, flange_jacobian[3:6])),
    )


# --- action -> target pose -----------------------------------------------------
@dataclass
class OscTarget:
    pos: np.ndarray
    quat: np.ndarray
    preclipped_pos: np.ndarray
    preclipped_quat: np.ndarray


def decode_action_target(filtered_action, grasp: GraspFrameState, bolt_tip_pos=BOLT_TIP_POSITION, *, clip=True):
    """``_apply_action`` steps (0)-(2): filtered native action -> clipped grasp-frame target."""

    a = np.asarray(filtered_action, dtype=float)
    pos_actions = a[0:3] * POS_ACTION_BOUNDS
    rot_actions = a[3:6] * ROT_ACTION_BOUNDS
    pre_pos = bolt_tip_pos + pos_actions
    # free_ee_orientation = True: roll/pitch stay; yaw mapped into the joint window.
    yaw = math.radians(-180.0) + math.radians(270.0) * (rot_actions[2] + 1.0) / 2.0
    bolt_frame_quat = quat_from_euler_xyz(rot_actions[0], rot_actions[1], yaw)
    base_quat = quat_from_euler_xyz(*EE_BASE_ORN_EULER)
    pre_quat = quat_mul(base_quat, bolt_frame_quat)
    if not clip:
        return OscTarget(pre_pos, pre_quat, pre_pos, pre_quat)

    delta_pos = pre_pos - grasp.pos
    target_pos = grasp.pos + np.clip(delta_pos, -POS_ACTION_THRESHOLD, POS_ACTION_THRESHOLD)

    curr_roll, curr_pitch, curr_yaw = get_euler_xyz(grasp.quat)
    desired_roll, desired_pitch, desired_yaw = get_euler_xyz(pre_quat)
    curr_yaw = wrap_yaw(curr_yaw)
    desired_yaw = wrap_yaw(desired_yaw)
    delta_yaw = desired_yaw - curr_yaw
    yaw_t = curr_yaw + float(np.clip(delta_yaw, -ROT_ACTION_THRESHOLD[2], ROT_ACTION_THRESHOLD[2]))

    if desired_roll < 0.0:
        desired_roll += 2.0 * math.pi
    if desired_pitch < 0.0:
        desired_pitch += 2.0 * math.pi
    delta_roll = desired_roll - curr_roll
    roll_t = curr_roll + float(np.clip(delta_roll, -ROT_ACTION_THRESHOLD[0], ROT_ACTION_THRESHOLD[0]))
    if curr_pitch > math.pi:
        curr_pitch -= 2.0 * math.pi
    if desired_pitch > math.pi:
        desired_pitch -= 2.0 * math.pi
    delta_pitch = desired_pitch - curr_pitch
    pitch_t = curr_pitch + float(np.clip(delta_pitch, -ROT_ACTION_THRESHOLD[1], ROT_ACTION_THRESHOLD[1]))
    target_quat = quat_from_euler_xyz(roll_t, pitch_t, yaw_t)
    return OscTarget(target_pos, target_quat, pre_pos, pre_quat)


# --- operational-space impedance -------------------------------------------------
def get_pose_error(pos, quat, target_pos, target_quat):
    pos_error = target_pos - pos
    if np.dot(target_quat, quat) < 0.0:
        target_quat = -target_quat
    quat_inv = quat_conjugate(quat) / float(np.dot(quat, quat))
    quat_error = quat_mul(target_quat, quat_inv)
    return pos_error, axis_angle_from_quat(quat_error)


def compute_dof_torque(
    *,
    dof_pos_arm,
    dof_vel_arm,
    grasp: GraspFrameState,
    arm_mass_matrix,
    target_pos,
    target_quat,
    task_prop_gains=DEFAULT_TASK_PROP_GAINS,
    rot_deriv_scale=1.0,
    dead_zone_thresholds=None,
    default_dof_pos=FRANKA_ARM_RESET_JOINTS_M24,
):
    """ForgeUltra ``compute_dof_torque`` for one environment.

    Returns ``(arm_torque, task_wrench, pos_error, axis_angle_error)``.
    """

    kp = np.asarray(task_prop_gains, dtype=float)
    kd = 2.0 * np.sqrt(kp)
    kd[3:6] /= rot_deriv_scale
    pos_error, aa_error = get_pose_error(grasp.pos, grasp.quat, target_pos, target_quat)
    wrench = np.zeros(6)
    wrench[0:3] = kp[0:3] * pos_error + kd[0:3] * (0.0 - grasp.linvel)
    wrench[3:6] = kp[3:6] * aa_error + kd[3:6] * (0.0 - grasp.angvel)
    if dead_zone_thresholds is not None:
        dz = np.asarray(dead_zone_thresholds, dtype=float)
        wrench = np.where(np.abs(wrench) < dz, 0.0, np.sign(wrench) * (np.abs(wrench) - dz))
    J = grasp.jacobian
    arm_torque = J.T @ wrench

    M = np.asarray(arm_mass_matrix, dtype=float)
    M_inv = np.linalg.inv(M)
    M_task = np.linalg.inv(J @ M_inv @ J.T)
    j_eef_inv = M_task @ J @ M_inv
    distance = np.asarray(default_dof_pos, dtype=float) - np.asarray(dof_pos_arm, dtype=float)
    distance = (distance + math.pi) % (2.0 * math.pi) - math.pi
    u_null = KD_NULL * -np.asarray(dof_vel_arm, dtype=float) + KP_NULL * distance
    u_null = M @ u_null
    torque_null = (np.eye(7) - J.T @ j_eef_inv) @ u_null
    arm_torque = arm_torque + torque_null
    arm_torque = np.clip(arm_torque, -ARM_TORQUE_LIMIT, ARM_TORQUE_LIMIT)
    return arm_torque, wrench, pos_error, aa_error


# --- hand targets ---------------------------------------------------------------
def pinch_targets(filtered_action, closed_posture=None):
    """``generate_ctrl_signals`` with finger_action_relative_to_grasp=True."""

    closed = closed_posture or THREADING_GRASP_POSTURE
    a = np.asarray(filtered_action, dtype=float)[-3:]
    out = np.zeros(3)
    for k, (name, (lo, hi)) in enumerate(zip(PINCH_JOINTS, PINCH_RANGES)):
        out[k] = float(np.clip(closed[name] + a[k] * (hi - lo) * 0.5, lo, hi))
    return out


def expand_hand_mimic(joint_targets: dict) -> dict:
    """Topologically expand the six policy-level finger targets to all twelve."""

    result = dict(joint_targets)
    for follower, (leader, multiplier, offset) in MIMIC_JOINT_MAP.items():
        result[follower] = result[leader] * multiplier + offset
    return result


# --- the adapter --------------------------------------------------------------------
def joint_pd_command_from_torque(arm_pos, arm_vel, arm_torque, hand_targets):
    """``ImpedanceJointPDController.target_from_teacher_control`` in NumPy.

    ``q_ref = q + tau / Kp`` and ``dq_ref = dq`` make the joint-impedance
    feedback ``Kp(q_ref - q) + Kd(dq_ref - dq)`` reproduce the OSC torque
    exactly at the labelled state. Layout: 7 arm positions, 7 arm velocities,
    3 pinch positions (thumb_joint_0, thumb_joint_1, index_joint_0).
    """

    command = np.zeros(17)
    command[0:7] = np.asarray(arm_pos, dtype=float) + np.asarray(arm_torque, dtype=float) / IMPEDANCE_JOINT_PD_ARM_STIFFNESS
    command[7:14] = np.asarray(arm_vel, dtype=float)
    command[14:17] = np.asarray(hand_targets, dtype=float)
    return command


def joint_pd_torque(q, dq, command, gravity_compensation=None):
    """The low-level joint impedance law the deployment controller runs."""

    tau = IMPEDANCE_JOINT_PD_ARM_STIFFNESS * (command[0:7] - q) + IMPEDANCE_JOINT_PD_ARM_DAMPING * (command[7:14] - dq)
    if gravity_compensation is not None:
        tau = tau + gravity_compensation
    return tau


__all__ = [
    "BOLT_TIP_POSITION",
    "DEFAULT_DEAD_ZONE",
    "FRANKA_ARM_RESET_JOINTS_M24",
    "GraspFrameState",
    "IMPEDANCE_JOINT_PD_ARM_DAMPING",
    "IMPEDANCE_JOINT_PD_ARM_STIFFNESS",
    "MIMIC_JOINT_MAP",
    "OscTarget",
    "PINCH_JOINTS",
    "PINCH_RANGES",
    "THREADING_GRASP_POSTURE",
    "THREADING_HAND_POSTURE",
    "compute_dof_torque",
    "decode_action_target",
    "expand_hand_mimic",
    "grasp_frame_state",
    "joint_pd_command_from_torque",
    "joint_pd_torque",
    "pinch_targets",
    "reset_z_transport",
]
