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

"""The common ``data.npz`` a run directory holds, from a rosbag or from the MuJoCo simulator.

Arrays (M samples at the controller rate unless noted; all joint arrays are (M, 7)):

    t, stamp_ns                       controller_state clock, seconds from the first sample
    q_ref, qd_ref                     reference the control law was applied to
    q, dq, tau                        measured position, velocity, joint torque
    out                               position command after the rate limiter, or the torque
    phase, elapsed                    controller phase (0 idle, 1 goto, 2 trajectory, 3 stop)
                                      and the phase clock
    rs_*                              FrankaRobotState fields (real robot; the simulator fills
                                      the ones it can), on their own clock rs_t / rs_stamp_ns
    js_*                              joint_states, on js_t
    st_*                              controller status samples (rate limiter counters, ids)
    sim_*                             simulator-only: contact count, motion generator violations

``t0_ns`` is the absolute stamp of t = 0, so the step timestamps in run.json can be placed.
"""

import os

import numpy as np

from franka_trajectory_replay.recording import read_messages, stamp_to_ns

ERROR_NAMES = None


def _error_names(errors_msg):
    return [name for name in errors_msg.get_fields_and_field_types()]


def _pose16(pose):
    """PoseStamped -> column-major 4x4 as libfranka packs O_T_EE."""
    from franka_trajectory_replay.kinematics import quaternion_to_matrix

    p = pose.pose.position
    o = pose.pose.orientation
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix([o.x, o.y, o.z, o.w])
    transform[:3, 3] = [p.x, p.y, p.z]
    return transform.reshape(16, order='F')


def _wrench6(w):
    return [w.wrench.force.x, w.wrench.force.y, w.wrench.force.z,
            w.wrench.torque.x, w.wrench.torque.y, w.wrench.torque.z]


def _twist6(t):
    return [t.twist.linear.x, t.twist.linear.y, t.twist.linear.z,
            t.twist.angular.x, t.twist.angular.y, t.twist.angular.z]


def _accel6(a):
    return [a.accel.linear.x, a.accel.linear.y, a.accel.linear.z,
            a.accel.angular.x, a.accel.angular.y, a.accel.angular.z]


def _vec3(v):
    return [v.x, v.y, v.z]


def extract_bag(bag_dir, topics, joint_names, output_path):
    """Read the bag once and write data.npz. ``topics`` maps role -> absolute topic name."""
    cs = {key: [] for key in ('stamp', 'q_ref', 'qd_ref', 'q', 'dq', 'tau', 'out_pos', 'out_eff',
                              'phase', 'elapsed')}
    rs = {}
    js = {key: [] for key in ('stamp', 'q', 'dq', 'tau')}
    st = {key: [] for key in ('stamp', 'phase', 'active', 'completed', 'rate_total', 'rate_last',
                              'rejections')}
    error_names = None
    mode = None
    wanted = {name: role for role, name in topics.items() if name}

    for topic, msg, receive_ns in read_messages(bag_dir, list(wanted)):
        role = wanted[topic]
        if role == 'controller_state':
            order = [msg.joint_names.index(name) for name in joint_names] if msg.joint_names else list(range(7))
            cs['stamp'].append(stamp_to_ns(msg.header.stamp) or receive_ns)
            cs['q_ref'].append([msg.reference.positions[i] for i in order])
            cs['qd_ref'].append([msg.reference.velocities[i] for i in order])
            cs['q'].append([msg.feedback.positions[i] for i in order])
            cs['dq'].append([msg.feedback.velocities[i] for i in order])
            cs['tau'].append([msg.feedback.effort[i] for i in order])
            cs['out_pos'].append([msg.output.positions[i] for i in order] if msg.output.positions else [0.0] * 7)
            cs['out_eff'].append([msg.output.effort[i] for i in order] if msg.output.effort else [0.0] * 7)
            cs['phase'].append(int(round(msg.output.time_from_start.sec + msg.output.time_from_start.nanosec * 1e-9)))
            cs['elapsed'].append(msg.reference.time_from_start.sec + msg.reference.time_from_start.nanosec * 1e-9)
        elif role == 'robot_state':
            if error_names is None:
                error_names = _error_names(msg.current_errors)
                for key in ('stamp', 'q', 'dq', 'tau_J', 'q_d', 'dq_d', 'tau_J_d', 'ddq_d', 'theta',
                            'dtheta', 'dtau_J', 'tau_ext', 'O_F_ext', 'K_F_ext', 'O_T_EE', 'O_T_EE_d',
                            'O_T_EE_c', 'F_T_EE', 'EE_T_K', 'O_dP_EE_d', 'O_dP_EE_c', 'O_ddP_EE_c',
                            'elbow', 'elbow_d', 'elbow_c', 'joint_contact', 'joint_collision',
                            'cart_contact', 'cart_collision', 'success_rate', 'robot_mode', 'time',
                            'errors', 'last_motion_errors', 'm_ee', 'm_load', 'm_total',
                            'F_x_Cee', 'F_x_Cload', 'F_x_Ctotal'):
                    rs[key] = []
            rs['stamp'].append(stamp_to_ns(msg.header.stamp) or receive_ns)
            rs['q'].append(list(msg.measured_joint_state.position))
            rs['dq'].append(list(msg.measured_joint_state.velocity))
            rs['tau_J'].append(list(msg.measured_joint_state.effort))
            rs['q_d'].append(list(msg.desired_joint_state.position))
            rs['dq_d'].append(list(msg.desired_joint_state.velocity))
            rs['tau_J_d'].append(list(msg.desired_joint_state.effort))
            rs['ddq_d'].append(list(msg.ddq_d))
            rs['theta'].append(list(msg.measured_joint_motor_state.position))
            rs['dtheta'].append(list(msg.measured_joint_motor_state.velocity))
            rs['dtau_J'].append(list(msg.dtau_j))
            rs['tau_ext'].append(list(msg.tau_ext_hat_filtered.effort))
            rs['O_F_ext'].append(_wrench6(msg.o_f_ext_hat_k))
            rs['K_F_ext'].append(_wrench6(msg.k_f_ext_hat_k))
            rs['O_T_EE'].append(_pose16(msg.o_t_ee))
            rs['O_T_EE_d'].append(_pose16(msg.o_t_ee_d))
            rs['O_T_EE_c'].append(_pose16(msg.o_t_ee_c))
            rs['F_T_EE'].append(_pose16(msg.f_t_ee))
            rs['EE_T_K'].append(_pose16(msg.ee_t_k))
            rs['O_dP_EE_d'].append(_twist6(msg.o_dp_ee_d))
            rs['O_dP_EE_c'].append(_twist6(msg.o_dp_ee_c))
            rs['O_ddP_EE_c'].append(_accel6(msg.o_ddp_ee_c))
            rs['elbow'].append(list(msg.elbow.position))
            rs['elbow_d'].append(list(msg.elbow.desired_position))
            rs['elbow_c'].append(list(msg.elbow.commanded_position))
            ci = msg.collision_indicators
            rs['joint_contact'].append(list(ci.is_joint_contact))
            rs['joint_collision'].append(list(ci.is_joint_collision))
            rs['cart_contact'].append(_vec3(ci.is_cartesian_linear_contact) + _vec3(ci.is_cartesian_angular_contact))
            rs['cart_collision'].append(_vec3(ci.is_cartesian_linear_collision) + _vec3(ci.is_cartesian_angular_collision))
            rs['success_rate'].append(msg.control_command_success_rate)
            rs['robot_mode'].append(int(msg.robot_mode))
            rs['time'].append(msg.time)
            rs['errors'].append([bool(getattr(msg.current_errors, name)) for name in error_names])
            rs['last_motion_errors'].append([bool(getattr(msg.last_motion_errors, name)) for name in error_names])
            rs['m_ee'].append(msg.inertia_ee.inertia.m)
            rs['m_load'].append(msg.inertia_load.inertia.m)
            rs['m_total'].append(msg.inertia_total.inertia.m)
            rs['F_x_Cee'].append(_vec3(msg.inertia_ee.inertia.com))
            rs['F_x_Cload'].append(_vec3(msg.inertia_load.inertia.com))
            rs['F_x_Ctotal'].append(_vec3(msg.inertia_total.inertia.com))
        elif role == 'joint_states':
            positions = dict(zip(msg.name, msg.position))
            if not all(name in positions for name in joint_names):
                continue
            velocities = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
            efforts = dict(zip(msg.name, msg.effort)) if msg.effort else {}
            js['stamp'].append(stamp_to_ns(msg.header.stamp) or receive_ns)
            js['q'].append([positions[n] for n in joint_names])
            js['dq'].append([velocities.get(n, np.nan) for n in joint_names])
            js['tau'].append([efforts.get(n, np.nan) for n in joint_names])
        elif role == 'status':
            if not msg.status:
                continue
            values = {kv.key: kv.value for kv in msg.status[0].values}
            mode = values.get('command_mode', mode)
            st['stamp'].append(stamp_to_ns(msg.header.stamp) or receive_ns)
            st['phase'].append(int(values.get('phase', 0)))
            st['active'].append(int(values.get('active_command_id', 0)))
            st['completed'].append(int(values.get('completed_command_id', 0)))
            st['rate_total'].append(int(values.get('rate_limit_engaged_total', 0)))
            st['rate_last'].append(int(values.get('rate_limit_engaged_last_command', 0)))
            st['rejections'].append(int(values.get('rejections', 0)))

    if not cs['stamp']:
        raise RuntimeError('the bag holds no controller_state messages (%s)' % topics.get('controller_state'))

    order = np.argsort(np.asarray(cs['stamp']), kind='stable')
    stamp = np.asarray(cs['stamp'], dtype=np.int64)[order]
    t0_ns = int(stamp[0])
    out = {
        't0_ns': np.int64(t0_ns),
        'stamp_ns': stamp,
        't': (stamp - t0_ns) * 1e-9,
        'q_ref': np.asarray(cs['q_ref'])[order], 'qd_ref': np.asarray(cs['qd_ref'])[order],
        'q': np.asarray(cs['q'])[order], 'dq': np.asarray(cs['dq'])[order],
        'tau': np.asarray(cs['tau'])[order],
        'out': (np.asarray(cs['out_eff']) if mode == 'effort' else np.asarray(cs['out_pos']))[order],
        'phase': np.asarray(cs['phase'], dtype=np.int64)[order],
        'elapsed': np.asarray(cs['elapsed'])[order],
        'mode': np.asarray(mode or 'position'),
        'joint_names': np.asarray(joint_names),
        'source': np.asarray('bag'),
    }
    if rs:
        order = np.argsort(np.asarray(rs['stamp']), kind='stable')
        for key, values in rs.items():
            if key == 'stamp':
                continue
            out['rs_' + key] = np.asarray(values)[order]
        out['rs_stamp_ns'] = np.asarray(rs['stamp'], dtype=np.int64)[order]
        out['rs_t'] = (out['rs_stamp_ns'] - t0_ns) * 1e-9
        out['rs_error_names'] = np.asarray(error_names)
    if js['stamp']:
        order = np.argsort(np.asarray(js['stamp']), kind='stable')
        out['js_t'] = (np.asarray(js['stamp'], dtype=np.int64)[order] - t0_ns) * 1e-9
        for key in ('q', 'dq', 'tau'):
            out['js_' + key] = np.asarray(js[key])[order]
    if st['stamp']:
        order = np.argsort(np.asarray(st['stamp']), kind='stable')
        out['st_t'] = (np.asarray(st['stamp'], dtype=np.int64)[order] - t0_ns) * 1e-9
        for key in ('phase', 'active', 'completed', 'rate_total', 'rate_last', 'rejections'):
            out['st_' + key] = np.asarray(st[key], dtype=np.int64)[order]
    np.savez_compressed(output_path, **out)
    return output_path


def load_dataset(run_dir):
    path = os.path.join(str(run_dir), 'data.npz')
    if not os.path.exists(path):
        raise FileNotFoundError('%s has no data.npz - run analyze_replay.py (it extracts the bag) first' % run_dir)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_dataset(run_dir, arrays):
    os.makedirs(str(run_dir), exist_ok=True)
    np.savez_compressed(os.path.join(str(run_dir), 'data.npz'), **arrays)
