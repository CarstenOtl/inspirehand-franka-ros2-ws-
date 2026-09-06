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

"""Offline MuJoCo replay of a prepared trajectory, mirroring what the ROS run does.

The same sequence as replay_trajectory.py - goto the first point, settle, play the trajectory,
settle, return to the first point, settle, goto home - driven through the same rate limiter and
the same emulated robot-internal joint impedance controller as franka_mujoco_hardware. The
output is a run directory with the identical data.npz layout, so analyze_replay.py and
plot_replay.py treat it like a recording from the arm.

What this can rule out: motion-generator limit violations (the arm would reflex), joint limit
excursions, self-collisions and floor contact, gross tracking problems, and mistakes in the
trajectory itself (wrong joint order, degrees, wrong rate). What it cannot tell you: the real
tracking error - the impedance emulation is a plausibility model, not an identified one.
"""

import os
import time

import numpy as np

from franka_trajectory_replay import limits
from franka_trajectory_replay.kinematics import READY_POSE
from franka_trajectory_replay.prepare import goto_duration, quintic

DEFAULT_STIFFNESS = np.array([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])
DEFAULT_DAMPING = np.array([60.0, 60.0, 60.0, 50.0, 40.0, 20.0, 15.0])


def default_model_path():
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory('franka_mujoco_hardware'), 'mujoco', 'scene.xml')


class RateLimiter:
    """Python twin of TrajectoryReplayController::limit_position_rate."""

    def __init__(self, q0, dt=1e-3):
        self.q = np.asarray(q0, dtype=float).copy()
        self.dq = np.zeros(7)
        self.ddq = np.zeros(7)
        self.dt = dt
        self.engaged = 0

    def __call__(self, q_desired):
        dt = self.dt
        upper = limits.upper_velocity_limits(self.q)
        lower = limits.lower_velocity_limits(self.q)
        dq_desired = (q_desired - self.q) / dt
        safe_max_ddq = np.minimum(limits.JERK_MAX * dt + self.ddq, limits.ACCELERATION_MAX)
        safe_min_ddq = np.maximum(-limits.JERK_MAX * dt + self.ddq, -limits.ACCELERATION_MAX)
        dq = np.clip(dq_desired, self.dq + safe_min_ddq * dt, self.dq + safe_max_ddq * dt)
        dq = np.clip(dq, lower, upper)
        if np.any(np.abs(dq - dq_desired) > 1e-9):
            self.engaged += 1
        q = self.q + dq * dt
        self.ddq = (dq - self.dq) / dt
        self.dq = dq
        self.q = q
        return q.copy()


class MotionGeneratorCheck:
    """Python twin of MujocoSystem::check_motion_generator: the libfranka errors a command stream
    would raise."""

    def __init__(self, q0, dt=1e-3):
        self.q_d = np.asarray(q0, dtype=float).copy()
        self.dq_d = np.zeros(7)
        self.ddq_d = np.zeros(7)
        self.dt = dt
        self.started = False
        self.violations = []

    def __call__(self, q_new, t):
        dt = self.dt
        violation = None
        upper = limits.upper_velocity_limits(self.q_d)
        lower = limits.lower_velocity_limits(self.q_d)
        dq = (q_new - self.q_d) / dt
        ddq = (dq - self.dq_d) / dt
        dddq = (ddq - self.ddq_d) / dt
        for j in range(7):
            if q_new[j] < limits.POSITION_LOWER[j] or q_new[j] > limits.POSITION_UPPER[j]:
                violation = ('joint_motion_generator_position_limits_violation', j, q_new[j])
            elif dq[j] > upper[j] + limits.VELOCITY_TOLERANCE or dq[j] < lower[j] - limits.VELOCITY_TOLERANCE:
                violation = ('joint_motion_generator_velocity_limits_violation', j, dq[j])
            elif self.started and abs(ddq[j]) > limits.ACCELERATION_MAX[j] + 1e-6:
                violation = ('joint_motion_generator_velocity_discontinuity', j, ddq[j])
            elif self.started and abs(dddq[j]) > limits.JERK_MAX[j] + 1e-6:
                violation = ('joint_motion_generator_acceleration_discontinuity', j, dddq[j])
            if violation:
                break
        self.q_d, self.dq_d, self.ddq_d = q_new.copy(), dq, ddq
        self.started = True
        if violation:
            self.violations.append({'t': float(t), 'error': violation[0], 'joint': int(violation[1] + 1),
                                    'value': float(violation[2])})
        return violation is None


class ImpedanceSim:
    def __init__(self, model_path=None, stiffness=None, damping=None, initial_q=None, view=False,
                 realtime=False):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path or default_model_path()))
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.stiffness = np.asarray(stiffness if stiffness is not None else DEFAULT_STIFFNESS, dtype=float)
        self.damping = np.asarray(damping if damping is not None else DEFAULT_DAMPING, dtype=float)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'fr3_joint%d' % i)
                          for i in range(1, 8)]
        if min(self.joint_ids) < 0:
            raise KeyError('the MJCF does not name its joints fr3_joint1..7')
        self.qpos_index = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_index = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.actuator_index = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fr3_joint%d' % i) for i in range(1, 8)])
        self.flange_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'flange')
        self.torque_limits = limits.TORQUE_MAX
        self.dt = float(self.model.opt.timestep)
        self.reset(initial_q if initial_q is not None else READY_POSE)
        self.viewer = None
        self.realtime = realtime
        if view:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self, q):
        mujoco = self.mujoco
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_index] = np.asarray(q, dtype=float)
        self.data.qvel[self.dof_index] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self):
        return self.data.qpos[self.qpos_index].copy()

    @property
    def dq(self):
        return self.data.qvel[self.dof_index].copy()

    def flange(self):
        """Flange pose as libfranka packs O_T_EE: 16 values, column-major."""
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[self.flange_site].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[self.flange_site]
        return transform.reshape(16, order='F')

    def contacts(self):
        """Names of geom pairs currently in contact."""
        mujoco = self.mujoco
        pairs = []
        for k in range(self.data.ncon):
            contact = self.data.contact[k]
            a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or str(contact.geom1)
            b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or str(contact.geom2)
            pairs.append('%s<->%s' % (a, b))
        return pairs

    def step(self, q_d, dq_d):
        """One 1 ms cycle of the emulated internal joint impedance controller."""
        bias = self.data.qfrc_bias[self.dof_index]
        tau = self.stiffness * (q_d - self.q) + self.damping * (dq_d - self.dq) + bias
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.actuator_index] = tau
        self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
            if not self.viewer.is_running():
                raise KeyboardInterrupt('viewer closed')
        return tau

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def simulate_run(prepared, home_q=None, model_path=None, stiffness=None, damping=None,
                 settle_seconds=1.0, goto_max_velocity=0.5, goto_max_acceleration=1.0,
                 goto_min_duration=2.0, rate_limit=True, view=False, realtime=False, log=print):
    """Run the replay sequence in MuJoCo. Returns (arrays for data.npz, steps, summary)."""
    home_q = np.asarray(home_q if home_q is not None else READY_POSE, dtype=float)
    sim = ImpedanceSim(model_path, stiffness, damping, home_q, view=view, realtime=realtime)
    dt = sim.dt
    if abs(dt - 1e-3) > 1e-9:
        log('note: model timestep is %.4f s, the control cycle assumes 1 ms' % dt)

    limiter = RateLimiter(home_q, dt)
    checker = MotionGeneratorCheck(home_q, dt)
    rec = {key: [] for key in ('t', 'q_ref', 'qd_ref', 'q', 'dq', 'tau', 'out', 'phase', 'elapsed',
                               'O_T_EE', 'ncon', 'violation')}
    contact_log = []
    steps = []
    clock = [0.0]
    q_cmd_prev = home_q.copy()

    def record(q_ref, phase, elapsed):
        nonlocal q_cmd_prev
        q_out = limiter(q_ref) if rate_limit else q_ref.copy()
        ok = checker(q_out, clock[0])
        qd_ref = (q_ref - q_cmd_prev) / dt
        q_cmd_prev = q_ref.copy()
        q_before, dq_before = sim.q, sim.dq
        tau = sim.step(q_out, limiter.dq if rate_limit else qd_ref)
        rec['t'].append(clock[0]); rec['q_ref'].append(q_ref.copy()); rec['qd_ref'].append(qd_ref)
        rec['q'].append(q_before); rec['dq'].append(dq_before); rec['tau'].append(tau)
        rec['out'].append(q_out); rec['phase'].append(phase); rec['elapsed'].append(elapsed)
        rec['O_T_EE'].append(sim.flange()); rec['ncon'].append(sim.data.ncon)
        rec['violation'].append(0 if ok else 1)
        if sim.data.ncon and (not contact_log or clock[0] - contact_log[-1]['t'] > 0.5):
            contact_log.append({'t': float(clock[0]), 'pairs': sim.contacts()})
        clock[0] += dt
        if realtime:
            time.sleep(dt)

    def goto(target, name):
        start = q_cmd_prev.copy()
        duration = goto_duration(target - start, goto_max_velocity, goto_max_acceleration, goto_min_duration)
        step = {'name': name, 'start_ns': int(clock[0] * 1e9), 'duration': duration,
                'target': target.tolist()}
        n = int(round(duration / dt))
        for k in range(1, n + 1):
            s = k / n
            record(start + (target - start) * quintic(s), 1, k * dt)
        step['end_ns'] = int(clock[0] * 1e9)
        steps.append(step)
        log('  %s: %.2f s ramp, largest step %.3f rad' % (name, duration, np.abs(target - start).max()))

    def settle(seconds):
        for _ in range(int(round(seconds / dt))):
            record(q_cmd_prev.copy(), 0, 0.0)

    log('simulating: home %s' % np.round(home_q, 3).tolist())
    settle(0.5)
    first = prepared.q[0].copy()
    goto(first, 'goto_start')
    settle(settle_seconds)

    step = {'name': 'trajectory', 'start_ns': int(clock[0] * 1e9), 'duration': prepared.duration}
    for k in range(len(prepared.t)):
        record(prepared.q[k].copy(), 2, float(prepared.t[k]))
    step['end_ns'] = int(clock[0] * 1e9)
    steps.append(step)
    log('  trajectory: %.2f s' % prepared.duration)
    settle(settle_seconds)
    goto(first, 'return_to_start')
    settle(settle_seconds)
    goto(home_q, 'goto_home')
    settle(0.5)
    sim.close()

    arrays = {
        't0_ns': np.int64(0),
        't': np.asarray(rec['t']), 'stamp_ns': (np.asarray(rec['t']) * 1e9).astype(np.int64),
        'q_ref': np.asarray(rec['q_ref']), 'qd_ref': np.asarray(rec['qd_ref']),
        'q': np.asarray(rec['q']), 'dq': np.asarray(rec['dq']), 'tau': np.asarray(rec['tau']),
        'out': np.asarray(rec['out']), 'phase': np.asarray(rec['phase'], dtype=np.int64),
        'elapsed': np.asarray(rec['elapsed']), 'mode': np.asarray('position'),
        'joint_names': np.asarray(list(prepared.joint_names) or ['fr3_joint%d' % i for i in range(1, 8)]),
        'source': np.asarray('mujoco'),
        'rs_t': np.asarray(rec['t']), 'rs_q': np.asarray(rec['q']), 'rs_dq': np.asarray(rec['dq']),
        'rs_tau_J': np.asarray(rec['tau']), 'rs_q_d': np.asarray(rec['out']),
        'rs_O_T_EE': np.asarray(rec['O_T_EE']),
        'rs_success_rate': np.ones(len(rec['t'])), 'rs_robot_mode': np.full(len(rec['t']), 2),
        'sim_ncon': np.asarray(rec['ncon'], dtype=np.int64),
        'sim_violation': np.asarray(rec['violation'], dtype=np.int64),
    }
    summary = {
        'rate_limit_engaged': int(limiter.engaged),
        'motion_generator_violations': checker.violations[:50],
        'motion_generator_violation_count': len(checker.violations),
        'contacts': contact_log[:50],
        'contact_samples': int(np.count_nonzero(arrays['sim_ncon'])),
        'stiffness': sim.stiffness.tolist(), 'damping': sim.damping.tolist(),
        'model_path': str(model_path or default_model_path()),
    }
    return arrays, steps, summary
