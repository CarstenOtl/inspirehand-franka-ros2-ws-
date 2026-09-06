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

"""Figures for a run directory. Every figure is a function ``fig_<name>(ctx) -> Figure``."""

import os

import numpy as np

from franka_trajectory_replay import kinematics, limits
from franka_trajectory_replay.analysis import label_segments, segments

# One fixed colour per joint, never cycled, so joint 4 is the same colour in every figure.
JOINT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5cd6']
REF_COLOR = '#333333'
MEAS_COLOR = '#2a78d6'
ROBOT_COLOR = '#eb6834'
PHASE_COLORS = {1: '#dddddd', 2: '#fff1c2', 3: '#f8c9c9'}


class Context:
    def __init__(self, run_dir, data, run_meta, tool=None, joint_names=None):
        self.run_dir = str(run_dir)
        self.data = data
        self.meta = run_meta
        self.tool = tool
        self.joint_names = joint_names or [str(n) for n in data['joint_names']]
        t0_ns = int(data['t0_ns']) if 't0_ns' in data else 0
        self.segments = label_segments(segments(data), run_meta.get('steps'), t0_ns)
        self.t = data['t']


def _plt():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.dpi': 110, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999999',
        'lines.linewidth': 1.2, 'legend.frameon': False, 'font.size': 9,
    })
    return plt


def _shade(ax, ctx):
    for seg in ctx.segments:
        ax.axvspan(seg['t_start'], seg['t_end'], color=PHASE_COLORS.get(seg['phase'], '#eeeeee'),
                   alpha=0.6, lw=0, zorder=0)


def _label_segments(ax, ctx):
    ymax = ax.get_ylim()[1]
    for seg in ctx.segments:
        ax.text(0.5 * (seg['t_start'] + seg['t_end']), ymax, seg['name'], ha='center', va='bottom',
                fontsize=7, color='#666666')


def _joint_grid(plt, title):
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axes = axes.ravel()
    axes[-1].axis('off')
    fig.suptitle(title)
    return fig, axes


def fig_joints(ctx):
    plt = _plt()
    d = ctx.data
    fig, axes = _joint_grid(plt, 'Joint positions: reference vs measured')
    for j in range(7):
        ax = axes[j]
        _shade(ax, ctx)
        ax.plot(ctx.t, d['q_ref'][:, j], color=REF_COLOR, ls='--', label='reference')
        ax.plot(ctx.t, d['q'][:, j], color=JOINT_COLORS[j], label='measured')
        ax.set_ylabel('%s [rad]' % ctx.joint_names[j])
        if j == 0:
            ax.legend(loc='upper right')
            _label_segments(ax, ctx)
    axes[5].set_xlabel('time [s]')
    axes[6].set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def fig_joint_errors(ctx):
    plt = _plt()
    d = ctx.data
    error = 1000 * (d['q_ref'] - d['q'])
    fig, axes = _joint_grid(plt, 'Joint tracking error (reference - measured)')
    for j in range(7):
        ax = axes[j]
        _shade(ax, ctx)
        ax.plot(ctx.t, error[:, j], color=JOINT_COLORS[j])
        ax.axhline(0.0, color='#999999', lw=0.6)
        ax.set_ylabel('%s [mrad]' % ctx.joint_names[j])
        if j == 0:
            _label_segments(ax, ctx)
    axes[5].set_xlabel('time [s]')
    axes[6].set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def fig_velocities(ctx):
    plt = _plt()
    d = ctx.data
    fig, axes = _joint_grid(plt, 'Joint velocities: reference vs measured')
    for j in range(7):
        ax = axes[j]
        _shade(ax, ctx)
        ax.plot(ctx.t, d['qd_ref'][:, j], color=REF_COLOR, ls='--', label='reference')
        ax.plot(ctx.t, d['dq'][:, j], color=JOINT_COLORS[j], label='measured')
        ax.set_ylabel('%s [rad/s]' % ctx.joint_names[j])
        if j == 0:
            ax.legend(loc='upper right')
    axes[5].set_xlabel('time [s]')
    axes[6].set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def fig_torques(ctx):
    plt = _plt()
    d = ctx.data
    fig, axes = _joint_grid(plt, 'Joint torques')
    has_rs = 'rs_t' in d
    for j in range(7):
        ax = axes[j]
        _shade(ax, ctx)
        ax.plot(ctx.t, d['tau'][:, j], color=JOINT_COLORS[j], label='tau_J (measured)')
        if has_rs and 'rs_tau_J_d' in d:
            ax.plot(d['rs_t'], d['rs_tau_J_d'][:, j], color=REF_COLOR, ls='--', lw=0.8, label='tau_J_d (desired)')
        if has_rs and 'rs_tau_ext' in d:
            ax.plot(d['rs_t'], d['rs_tau_ext'][:, j], color=ROBOT_COLOR, lw=0.8, label='tau_ext_hat_filtered')
        if str(d.get('mode', 'position')) == 'effort':
            ax.plot(ctx.t, d['out'][:, j], color='#008300', lw=0.8, label='commanded torque')
        ax.set_ylabel('%s [Nm]' % ctx.joint_names[j])
        if j == 0:
            ax.legend(loc='upper right')
    axes[5].set_xlabel('time [s]')
    axes[6].set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def _tcp(ctx):
    d = ctx.data
    stride = max(1, len(ctx.t) // 6000)
    idx = np.arange(0, len(ctx.t), stride)
    p_ref, r_ref = kinematics.flange_poses(d['q_ref'][idx], ctx.tool)
    p_meas, r_meas = kinematics.flange_poses(d['q'][idx], ctx.tool)
    return idx, p_ref, r_ref, p_meas, r_meas


def fig_tcp_path(ctx):
    plt = _plt()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    idx, p_ref, _, p_meas, _ = _tcp(ctx)
    fig = plt.figure(figsize=(12, 5))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax3.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], color=REF_COLOR, ls='--', label='reference (FK of q_ref)')
    ax3.plot(p_meas[:, 0], p_meas[:, 1], p_meas[:, 2], color=MEAS_COLOR, label='measured (FK of q)')
    if 'rs_O_T_EE' in ctx.data:
        rs = np.asarray(ctx.data['rs_O_T_EE'])[:: max(1, len(ctx.data['rs_t']) // 6000)]
        ax3.plot(rs[:, 12], rs[:, 13], rs[:, 14], color=ROBOT_COLOR, lw=0.8, label='robot O_T_EE')
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.set_title('TCP path in the base frame')
    ax3.legend(loc='upper left', fontsize=7)
    ax = fig.add_subplot(1, 2, 2)
    _shade(ax, ctx)
    for k, axis in enumerate('xyz'):
        ax.plot(ctx.t[idx], p_ref[:, k], color=JOINT_COLORS[k], ls='--')
        ax.plot(ctx.t[idx], p_meas[:, k], color=JOINT_COLORS[k], label=axis)
    ax.set_xlabel('time [s]'); ax.set_ylabel('TCP position [m]')
    ax.set_title('per axis (dashed: reference)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def fig_tcp_error(ctx):
    plt = _plt()
    idx, p_ref, r_ref, p_meas, r_meas = _tcp(ctx)
    diff = 1000 * (p_ref - p_meas)
    angle = np.degrees(kinematics.quaternion_angle(r_ref, r_meas))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('TCP tracking error (FK of reference minus FK of measured)')
    _shade(ax1, ctx); _shade(ax2, ctx)
    for k, axis in enumerate('xyz'):
        ax1.plot(ctx.t[idx], diff[:, k], color=JOINT_COLORS[k], label=axis)
    ax1.plot(ctx.t[idx], np.linalg.norm(diff, axis=1), color=REF_COLOR, label='norm')
    ax1.set_ylabel('position error [mm]')
    ax1.legend(loc='upper right')
    _label_segments(ax1, ctx)
    ax2.plot(ctx.t[idx], angle, color=JOINT_COLORS[3])
    ax2.set_ylabel('orientation error [deg]')
    ax2.set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def fig_external_wrench(ctx):
    d = ctx.data
    if 'rs_O_F_ext' not in d:
        return None
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Estimated external wrench at the EE, base frame (O_F_ext_hat_K)')
    _shade(ax1, ctx); _shade(ax2, ctx)
    f = np.asarray(d['rs_O_F_ext'])
    for k, axis in enumerate('xyz'):
        ax1.plot(d['rs_t'], f[:, k], color=JOINT_COLORS[k], label='F' + axis)
        ax2.plot(d['rs_t'], f[:, 3 + k], color=JOINT_COLORS[k], label='M' + axis)
    ax1.set_ylabel('force [N]'); ax2.set_ylabel('torque [Nm]'); ax2.set_xlabel('time [s]')
    ax1.legend(loc='upper right'); ax2.legend(loc='upper right')
    _label_segments(ax1, ctx)
    fig.tight_layout()
    return fig


def fig_motor_vs_joint(ctx):
    d = ctx.data
    if 'rs_theta' not in d:
        return None
    plt = _plt()
    deflection = 1000 * (np.asarray(d['rs_theta']) - np.asarray(d['rs_q']))
    fig, axes = _joint_grid(plt, 'Motor minus joint position (theta - q): drive elasticity')
    for j in range(7):
        ax = axes[j]
        _shade(ax, ctx)
        ax.plot(d['rs_t'], deflection[:, j], color=JOINT_COLORS[j])
        ax.set_ylabel('%s [mrad]' % ctx.joint_names[j])
    axes[5].set_xlabel('time [s]'); axes[6].set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def fig_robot_health(ctx):
    d = ctx.data
    if 'rs_t' not in d:
        return None
    plt = _plt()
    rows = 4 if 'rs_errors' in d else 3
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.4 * rows + 1), sharex=True)
    fig.suptitle('Robot health')
    ax = axes[0]
    _shade(ax, ctx)
    if 'rs_success_rate' in d:
        ax.plot(d['rs_t'], d['rs_success_rate'], color=MEAS_COLOR)
    ax.set_ylabel('command\nsuccess rate')
    ax.set_ylim(-0.05, 1.05)
    _label_segments(ax, ctx)
    ax = axes[1]
    _shade(ax, ctx)
    if 'rs_robot_mode' in d:
        ax.step(d['rs_t'], d['rs_robot_mode'], color=ROBOT_COLOR, where='post')
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(['other', 'idle', 'move', 'guiding', 'reflex', 'user stop', 'recovery'], fontsize=7)
    ax.set_ylabel('robot mode')
    ax = axes[2]
    _shade(ax, ctx)
    for key, label, color, offset in (('rs_joint_contact', 'joint contact', JOINT_COLORS[0], 0),
                                      ('rs_joint_collision', 'joint collision', JOINT_COLORS[1], 1),
                                      ('rs_cart_contact', 'cartesian contact', JOINT_COLORS[2], 2),
                                      ('rs_cart_collision', 'cartesian collision', JOINT_COLORS[3], 3)):
        if key in d:
            flag = np.asarray(d[key]).any(axis=1).astype(float)
            ax.step(d['rs_t'], flag * 0.8 + offset, color=color, where='post', label=label)
    if 'sim_ncon' in d:
        ax.step(d['t'], (d['sim_ncon'] > 0) * 0.8 + 2, color=JOINT_COLORS[2], where='post', label='sim contact')
        ax.step(d['t'], (d['sim_violation'] > 0) * 0.8 + 3, color=JOINT_COLORS[6], where='post',
                label='sim motion-generator violation')
    ax.set_ylabel('flags')
    ax.set_yticks([])
    ax.legend(loc='upper right', ncol=2)
    if 'rs_errors' in d:
        ax = axes[3]
        _shade(ax, ctx)
        errors = np.asarray(d['rs_errors'])
        names = [str(n) for n in d['rs_error_names']]
        active = np.flatnonzero(errors.any(axis=0))
        for k, column in enumerate(active):
            ax.step(d['rs_t'], errors[:, column] * 0.8 + k, where='post', color=JOINT_COLORS[k % 7],
                    label=names[column])
        ax.set_ylabel('errors')
        ax.set_yticks([])
        if len(active):
            ax.legend(loc='upper right', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'no error flags set', transform=ax.transAxes, ha='center', color='#666666')
    axes[-1].set_xlabel('time [s]')
    fig.tight_layout()
    return fig


def fig_sampling(ctx):
    plt = _plt()
    d = ctx.data
    series = [('controller_state', 't')]
    if 'rs_t' in d and len(d['rs_t']) > 2:
        series.append(('robot_state', 'rs_t'))
    if 'js_t' in d and len(d['js_t']) > 2:
        series.append(('joint_states', 'js_t'))
    fig, axes = plt.subplots(1, len(series), figsize=(4.5 * len(series), 3.5), squeeze=False)
    fig.suptitle('Sample intervals')
    for ax, (name, key) in zip(axes[0], series):
        intervals = 1000 * np.diff(d[key])
        ax.hist(intervals, bins=60, color=MEAS_COLOR)
        ax.set_yscale('log')
        ax.set_title('%s (median %.2f ms, max %.1f ms)' % (name, np.median(intervals), intervals.max()), fontsize=8)
        ax.set_xlabel('interval [ms]')
    fig.tight_layout()
    return fig


def fig_command_limits(ctx):
    plt = _plt()
    d = ctx.data
    t = ctx.t
    q = d['q_ref']
    # Reference derivatives on the controller's own clock where a phase is active (exact), the
    # message stamps elsewhere. Concatenated per sample so the x axis stays the run time.
    clock = np.asarray(d['elapsed'], dtype=float).copy()
    phase = np.asarray(d['phase'])
    base = t.copy()
    for seg in ctx.segments:
        a, b = seg['start'], seg['end']
        base[a:b] = t[a] + clock[a:b] - clock[a]
    base = np.maximum.accumulate(base + np.arange(len(base)) * 1e-9)
    qd = np.gradient(q, base, axis=0)
    qdd = np.gradient(qd, base, axis=0)
    upper = limits.upper_velocity_limits(q)
    lower = limits.lower_velocity_limits(q)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Commanded stream against the FR3 motion generator limits')
    _shade(ax1, ctx); _shade(ax2, ctx)
    fraction = np.where(qd >= 0, qd / upper, qd / lower)
    for j in range(7):
        ax1.plot(t, 100 * fraction[:, j], color=JOINT_COLORS[j], label=ctx.joint_names[j])
        ax2.plot(t, 100 * np.abs(qdd[:, j]) / limits.ACCELERATION_MAX[j], color=JOINT_COLORS[j])
    ax1.axhline(100, color='#c0392b', lw=0.8)
    ax2.axhline(100, color='#c0392b', lw=0.8)
    ax1.set_ylabel('velocity [% of limit at q]')
    ax2.set_ylabel('acceleration [% of limit]')
    ax2.set_xlabel('time [s]')
    ax1.legend(loc='upper right', ncol=4)
    _label_segments(ax1, ctx)
    fig.tight_layout()
    return fig


def fig_tcp_contributions(ctx):
    """Which joint's tracking error the TCP error comes from, over time and at the peaks."""
    from franka_trajectory_replay.analysis import tcp_contributions

    segs = [s for s in ctx.segments if s['phase'] == 2]
    if not segs:
        return None
    plt = _plt()
    seg = segs[0]
    a, b = seg['start'], seg['end']
    stride = max(1, (b - a) // 3000)
    idx = np.arange(a, b, stride)
    linear, _, error, _ = kinematics.tcp_error_contributions(ctx.data['q_ref'][idx], ctx.data['q'][idx], ctx.tool)
    norm = 1000 * np.linalg.norm(error, axis=1)
    direction = error / np.maximum(np.linalg.norm(error, axis=1)[:, None], 1e-12)
    projected = 1000 * np.einsum('kjd,kd->kj', linear, direction)
    t = ctx.t[idx]
    summary = tcp_contributions(ctx.data, seg, ctx.tool)
    peaks = summary['at_peaks']

    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(3, max(1, len(peaks)), height_ratios=[1.3, 1.0, 1.0])
    fig.suptitle('Joint contributions to the TCP position error (Jacobian decomposition, %s)' % seg['name'])

    ax = fig.add_subplot(grid[0, :])
    positive = np.clip(projected, 0, None)
    negative = np.clip(projected, None, 0)
    ax.stackplot(t, positive.T, colors=JOINT_COLORS, labels=ctx.joint_names, lw=0, alpha=0.85)
    ax.stackplot(t, negative.T, colors=JOINT_COLORS, lw=0, alpha=0.85)
    ax.plot(t, norm, color=REF_COLOR, lw=1.0, label='TCP error norm')
    for peak in peaks:
        ax.axvline(peak['t'], color='#c0392b', lw=0.8, ls=':')
    ax.set_ylabel('projection onto the error direction [mm]')
    ax.set_xlabel('time [s]')
    ax.legend(loc='upper right', ncol=4, fontsize=8)
    ax.set_title('stacked: each joint\'s share along the error direction (they sum to the norm); dotted: the peaks below', fontsize=9)

    ax = fig.add_subplot(grid[1, :])
    share = summary['share_rms_mm']
    ax.bar(np.arange(7), share, color=JOINT_COLORS, width=0.6)
    for j, value in enumerate(share):
        ax.text(j, value, '%.3f' % value, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(ctx.joint_names, fontsize=8)
    ax.set_ylabel('RMS contribution [mm]')
    ax.set_title('RMS of each joint\'s contribution vector over the whole segment', fontsize=9)

    for column, peak in enumerate(peaks):
        ax = fig.add_subplot(grid[2, column])
        values = peak['joint_projection_mm']
        ax.bar(np.arange(7), values, color=JOINT_COLORS, width=0.6)
        ax.axhline(0, color='#999999', lw=0.6)
        for j, value in enumerate(values):
            ax.text(j, value, '%+.2f' % value, ha='center', va='bottom' if value >= 0 else 'top', fontsize=7)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(['j%d' % (j + 1) for j in range(7)], fontsize=8)
        ax.set_title('peak %.2f mm at t = %.1f s\njoint errors [mrad]: %s' % (
            peak['error_mm'], peak['t'], ' '.join('%+.1f' % v for v in peak['joint_error_mrad'])), fontsize=8)
        if column == 0:
            ax.set_ylabel('signed share of the peak [mm]')
    fig.tight_layout()
    return fig


FIGURES = {
    'joints': fig_joints,
    'joint_errors': fig_joint_errors,
    'velocities': fig_velocities,
    'torques': fig_torques,
    'tcp_path': fig_tcp_path,
    'tcp_error': fig_tcp_error,
    'tcp_contributions': fig_tcp_contributions,
    'external_wrench': fig_external_wrench,
    'motor_vs_joint': fig_motor_vs_joint,
    'robot_health': fig_robot_health,
    'sampling': fig_sampling,
    'command_limits': fig_command_limits,
}


def fig_compare(contexts, labels):
    """Overlay of the trajectory-segment joint and TCP errors of several runs (sim vs real)."""
    plt = _plt()
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axes = axes.ravel()
    fig.suptitle('Trajectory segment: tracking error per run')
    for ctx, label, color in zip(contexts, labels, [MEAS_COLOR, ROBOT_COLOR, '#1baf7a', '#eda100']):
        segs = [s for s in ctx.segments if s['phase'] == 2]
        if not segs:
            continue
        a, b = segs[0]['start'], segs[0]['end']
        t = ctx.t[a:b] - ctx.t[a]
        error = 1000 * (ctx.data['q_ref'][a:b] - ctx.data['q'][a:b])
        for j in range(7):
            axes[j].plot(t, error[:, j], color=color, label=label)
            axes[j].set_ylabel('%s [mrad]' % ctx.joint_names[j])
        idx = np.arange(a, b, max(1, (b - a) // 3000))
        p_ref, _ = kinematics.flange_poses(ctx.data['q_ref'][idx], ctx.tool)
        p_meas, _ = kinematics.flange_poses(ctx.data['q'][idx], ctx.tool)
        axes[7].plot(ctx.t[idx] - ctx.t[a], 1000 * np.linalg.norm(p_ref - p_meas, axis=1), color=color, label=label)
    axes[7].set_ylabel('TCP error [mm]')
    axes[0].legend(loc='upper right')
    axes[7].legend(loc='upper right')
    axes[6].set_xlabel('trajectory time [s]'); axes[7].set_xlabel('trajectory time [s]')
    fig.tight_layout()
    return fig


def plot_all(ctx, only=None, formats=('png',)):
    plt = _plt()
    out_dir = os.path.join(ctx.run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, function in FIGURES.items():
        if only and name not in only:
            continue
        fig = function(ctx)
        if fig is None:
            continue
        for fmt in formats:
            path = os.path.join(out_dir, '%s.%s' % (name, fmt))
            fig.savefig(path, bbox_inches='tight')
            written.append(path)
        plt.close(fig)
    return written
