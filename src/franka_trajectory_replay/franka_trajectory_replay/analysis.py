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

"""Tracking metrics and the run report, from data.npz."""

import json
import os

import numpy as np

from franka_trajectory_replay import kinematics, limits

PHASE_NAMES = {0: 'idle', 1: 'goto', 2: 'trajectory', 3: 'stopping'}


def segments(data, min_samples=5):
    """Contiguous non-idle runs of the controller phase, split where the phase clock restarts."""
    phase = np.asarray(data['phase'])
    elapsed = np.asarray(data['elapsed'])
    t = np.asarray(data['t'])
    result = []
    start = None
    for k in range(len(phase)):
        active = phase[k] != 0
        new_command = start is not None and (phase[k] != phase[start] or (k > 0 and elapsed[k] < elapsed[k - 1] - 1e-6))
        if active and (start is None or new_command):
            if start is not None and k - start >= min_samples:
                result.append((start, k))
            start = k
        elif not active and start is not None:
            if k - start >= min_samples:
                result.append((start, k))
            start = None
    if start is not None and len(phase) - start >= min_samples:
        result.append((start, len(phase)))
    return [{'phase': int(phase[a]), 'phase_name': PHASE_NAMES.get(int(phase[a]), '?'),
             'start': int(a), 'end': int(b), 't_start': float(t[a]), 't_end': float(t[b - 1])}
            for a, b in result]


def label_segments(segs, steps, t0_ns):
    """Attach the runner's step names (goto_start, trajectory, ...) by time overlap."""
    for seg in segs:
        seg['name'] = seg['phase_name']
        best = 0.0
        for step in steps or []:
            a = (step['start_ns'] - t0_ns) * 1e-9
            b = (step['end_ns'] - t0_ns) * 1e-9
            overlap = min(b, seg['t_end']) - max(a, seg['t_start'])
            if overlap > best:
                best = overlap
                seg['name'] = step['name']
                seg['command_id'] = step.get('command_id')
    return segs


def _resample(t_src, values, t_dst):
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(t_dst, t_src, values)
    return np.column_stack([np.interp(t_dst, t_src, values[:, j]) for j in range(values.shape[1])])


def lag_estimate(t, reference, measured, max_lag=0.15):
    """Delay (s) of ``measured`` behind ``reference`` that minimises the RMS error."""
    if len(t) < 50 or np.ptp(reference) < 1e-4:
        return 0.0
    dt = float(np.median(np.diff(t)))
    lags = np.arange(0.0, max_lag + dt / 2, dt)
    best, best_lag = np.inf, 0.0
    for lag in lags:
        shifted = np.interp(t - lag, t, reference)
        mask = t >= t[0] + lag
        error = np.sqrt(np.mean((shifted[mask] - measured[mask]) ** 2)) if mask.any() else np.inf
        if error < best:
            best, best_lag = error, lag
    return float(best_lag)


def joint_metrics(data, seg, settle_window=0.2):
    a, b = seg['start'], seg['end']
    t = reference_clock(data, seg)
    error = data['q_ref'][a:b] - data['q'][a:b]
    tail = t >= t[-1] - settle_window
    metrics = {'joints': []}
    for j in range(7):
        metrics['joints'].append({
            'rms': float(np.sqrt(np.mean(error[:, j] ** 2))),
            'max_abs': float(np.abs(error[:, j]).max()),
            'mean': float(error[:, j].mean()),
            'final': float(error[tail, j].mean()) if tail.any() else float(error[-1, j]),
            'lag': lag_estimate(t, data['q_ref'][a:b, j], data['q'][a:b, j]),
            'velocity_rms': float(np.sqrt(np.mean((data['qd_ref'][a:b, j] - data['dq'][a:b, j]) ** 2))),
            'reference_range': float(np.ptp(data['q_ref'][a:b, j])),
        })
    metrics['rms_norm'] = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    metrics['max_norm'] = float(np.linalg.norm(error, axis=1).max())
    return metrics


def tcp_metrics(data, seg, tool=None):
    a, b = seg['start'], seg['end']
    t = data['t'][a:b]
    stride = max(1, len(t) // 4000)  # FK on up to ~4000 samples per segment is plenty
    idx = np.arange(a, b, stride)
    p_ref, r_ref = kinematics.flange_poses(data['q_ref'][idx], tool)
    p_meas, r_meas = kinematics.flange_poses(data['q'][idx], tool)
    d = p_ref - p_meas
    norm = np.linalg.norm(d, axis=1)
    angle = kinematics.quaternion_angle(r_ref, r_meas)
    path = float(np.sum(np.linalg.norm(np.diff(p_ref, axis=0), axis=1)))
    result = {
        'position_error_mean_mm': float(1000 * norm.mean()),
        'position_error_rms_mm': float(1000 * np.sqrt(np.mean(norm ** 2))),
        'position_error_max_mm': float(1000 * norm.max()),
        'position_error_final_mm': float(1000 * norm[-max(1, len(norm) // 50):].mean()),
        'axis_rms_mm': (1000 * np.sqrt(np.mean(d ** 2, axis=0))).tolist(),
        'axis_mean_mm': (1000 * d.mean(axis=0)).tolist(),
        'orientation_error_mean_deg': float(np.degrees(angle.mean())),
        'orientation_error_max_deg': float(np.degrees(angle.max())),
        'reference_path_length_m': path,
        'reference_extent_m': np.ptp(p_ref, axis=0).tolist(),
        'fk_samples': int(len(idx)),
    }
    # Cross-check against the robot's own O_T_EE: if this disagrees with FK(q) by more than a
    # few tenths of a millimetre, the kinematic model or the tool offset is the suspect.
    if 'rs_O_T_EE' in data and 'rs_t' in data and len(data['rs_t']):
        rs_idx = np.searchsorted(data['rs_t'], t[::stride])
        rs_idx = np.clip(rs_idx, 0, len(data['rs_t']) - 1)
        robot_p = np.asarray(data['rs_O_T_EE'])[rs_idx][:, 12:15]
        if tool is not None:
            # O_T_EE already includes the EE configured on the robot; compare flanges only
            # when no extra tool offset is requested here.
            result['o_t_ee_note'] = 'O_T_EE compared without the local tool offset'
            p_flange, _ = kinematics.flange_poses(data['q'][idx], None)
            diff = np.linalg.norm(robot_p - p_flange, axis=1)
        else:
            diff = np.linalg.norm(robot_p - p_meas, axis=1)
        result['o_t_ee_vs_fk_mean_mm'] = float(1000 * diff.mean())
        result['o_t_ee_vs_fk_max_mm'] = float(1000 * diff.max())
    return result


def reference_clock(data, seg):
    """Time base of the reference inside a segment.

    The controller advances its phase clock by exactly one cycle per update, so ``elapsed`` is
    the reference's own time - exact even when a 1 kHz message was dropped or the message
    stamps jitter, both of which would turn a finite difference of ``t`` into a fake spike.
    """
    a, b = seg['start'], seg['end']
    elapsed = np.asarray(data['elapsed'][a:b], dtype=float)
    if len(elapsed) > 1 and np.all(np.diff(elapsed) > 0):
        return elapsed
    return np.asarray(data['t'][a:b], dtype=float)


def tcp_contributions(data, seg, tool=None, peaks=3, max_samples=3000):
    """How much each joint's tracking error contributes to the TCP error in a segment.

    ``share_rms`` is each joint's RMS contribution norm over the segment; ``at_peaks`` lists the
    largest TCP deviations with the signed projection of every joint's contribution onto the
    error direction (they sum to the error norm up to the first-order residual).
    """
    a, b = seg['start'], seg['end']
    stride = max(1, (b - a) // max_samples)
    idx = np.arange(a, b, stride)
    linear, angular, error, angular_error = kinematics.tcp_error_contributions(
        data['q_ref'][idx], data['q'][idx], tool)
    norm = np.linalg.norm(error, axis=1)
    direction = error / np.maximum(norm[:, None], 1e-12)
    projected = np.einsum('kjd,kd->kj', linear, direction)  # (N, 7) signed, in metres
    linear_sum = linear.sum(axis=1)
    residual = np.linalg.norm(error - linear_sum, axis=1)
    result = {
        'share_rms_mm': (1000 * np.sqrt(np.mean(np.sum(linear ** 2, axis=2), axis=0))).tolist(),
        'share_of_error_rms_percent': (100 * np.sqrt(np.mean(projected ** 2, axis=0)) /
                                        max(np.sqrt(np.mean(norm ** 2)), 1e-12)).tolist(),
        'angular_share_rms_mdeg': (1000 * np.degrees(np.sqrt(np.mean(np.sum(angular ** 2, axis=2), axis=0)))).tolist(),
        'first_order_residual_max_mm': float(1000 * residual.max()),
        'samples': int(len(idx)),
        'at_peaks': [],
    }
    # the largest deviations, at least 0.5 s apart
    order = np.argsort(-norm)
    chosen = []
    for k in order:
        t_k = float(data['t'][idx[k]])
        if all(abs(t_k - c) > 0.5 for c in chosen):
            chosen.append(t_k)
            result['at_peaks'].append({
                't': t_k,
                'elapsed': float(data['elapsed'][idx[k]]),
                'error_mm': float(1000 * norm[k]),
                'error_vector_mm': (1000 * error[k]).tolist(),
                'joint_projection_mm': (1000 * projected[k]).tolist(),
                'joint_error_mrad': (1000 * (data['q_ref'][idx[k]] - data['q'][idx[k]])).tolist(),
                'dominant_joint': int(np.argmax(np.abs(projected[k])) + 1),
            })
        if len(chosen) >= peaks:
            break
    return result


def command_limits(data, seg):
    """How close the commanded stream came to the FR3 limits during a segment."""
    a, b = seg['start'], seg['end']
    t = reference_clock(data, seg)
    q = data['q_ref'][a:b]
    if len(t) < 5:
        return {}
    qd = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    qddd = np.gradient(qdd, t, axis=0)
    report = limits.check(t, q, qd, qdd, qddd)
    return {'velocity_fraction': report['velocity_fraction'],
            'acceleration_fraction': report['acceleration_fraction'],
            'jerk_fraction': report['jerk_fraction'], 'violations': report['violations']}


def sampling(data):
    result = {}
    if 'elapsed' in data and len(data['elapsed']) > 2:
        # Inside a phase the clock advances one cycle per message, so a jump of more than one
        # cycle is a message the realtime publisher skipped.
        steps = np.diff(np.asarray(data['elapsed'], dtype=float))
        cycle = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1e-3
        result['controller_state_dropped'] = int(np.count_nonzero(steps > 1.5 * cycle))
    for prefix, key in (('controller_state', 't'), ('robot_state', 'rs_t'), ('joint_states', 'js_t')):
        if key not in data or len(data[key]) < 2:
            continue
        intervals = np.diff(data[key])
        result[prefix] = {
            'samples': int(len(data[key])),
            'rate_hz': float(1.0 / np.median(intervals)),
            'max_gap_ms': float(1000 * intervals.max()),
            'gaps_over_5ms': int(np.count_nonzero(intervals > 0.005)),
            'duration_s': float(data[key][-1] - data[key][0]),
        }
    return result


def robot_health(data):
    result = {}
    if 'rs_success_rate' in data:
        result['control_command_success_rate_min'] = float(np.min(data['rs_success_rate']))
    if 'rs_robot_mode' in data:
        modes, counts = np.unique(data['rs_robot_mode'], return_counts=True)
        names = {0: 'other', 1: 'idle', 2: 'move', 3: 'guiding', 4: 'reflex', 5: 'user_stopped',
                 6: 'automatic_error_recovery'}
        result['robot_modes'] = {names.get(int(m), str(m)): int(c) for m, c in zip(modes, counts)}
    if 'rs_errors' in data:
        names = [str(n) for n in data['rs_error_names']]
        active = np.asarray(data['rs_errors']).any(axis=0)
        result['errors_seen'] = [names[k] for k in np.flatnonzero(active)]
        last = np.asarray(data['rs_last_motion_errors'])[-1] if 'rs_last_motion_errors' in data else None
        if last is not None:
            result['last_motion_errors'] = [names[k] for k in np.flatnonzero(last)]
    for key, label in (('rs_joint_contact', 'joint_contact'), ('rs_joint_collision', 'joint_collision'),
                       ('rs_cart_contact', 'cartesian_contact'), ('rs_cart_collision', 'cartesian_collision')):
        if key in data:
            result[label + '_samples'] = int(np.count_nonzero(np.asarray(data[key]).any(axis=1)))
    if 'rs_tau_ext' in data:
        result['tau_ext_max_abs'] = np.abs(data['rs_tau_ext']).max(axis=0).tolist()
    if 'rs_O_F_ext' in data:
        f = np.asarray(data['rs_O_F_ext'])
        result['external_force_max_N'] = float(np.linalg.norm(f[:, :3], axis=1).max())
        result['external_torque_max_Nm'] = float(np.linalg.norm(f[:, 3:], axis=1).max())
    if 'rs_theta' in data and 'rs_q' in data:
        deflection = np.asarray(data['rs_theta']) - np.asarray(data['rs_q'])
        result['motor_joint_deflection_max_mrad'] = (1000 * np.abs(deflection).max(axis=0)).tolist()
    if 'rs_dtau_J' in data:
        result['dtau_J_max_abs'] = np.abs(data['rs_dtau_J']).max(axis=0).tolist()
    if 'rs_tau_J' in data:
        result['tau_J_max_abs'] = np.abs(data['rs_tau_J']).max(axis=0).tolist()
        result['tau_J_fraction_of_limit'] = (np.abs(data['rs_tau_J']).max(axis=0) / limits.TORQUE_MAX).tolist()
    if 'sim_violation' in data:
        result['sim_motion_generator_violations'] = int(np.sum(data['sim_violation']))
        result['sim_contact_samples'] = int(np.count_nonzero(data['sim_ncon']))
    if 'st_rate_total' in data and len(data['st_rate_total']):
        result['rate_limit_engaged_total'] = int(data['st_rate_total'][-1])
        result['rejections'] = int(data['st_rejections'][-1])
    return result


def source_frame_check(run_dir, tool=None):
    """If the capture carried the simulator's own end-effector pose: FK(q_isaac) vs ee_pos."""
    path = os.path.join(str(run_dir), 'prepared.npz')
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as prepared:
        if 'source_ee_pos' not in prepared.files:
            return {}
        q = prepared['source_q']
        ee = prepared['source_ee_pos']
    stride = max(1, len(q) // 500)
    p, _ = kinematics.flange_poses(q[::stride], tool)
    ee = ee[::stride]
    diff = ee - p
    offset = diff.mean(axis=0)
    residual = np.linalg.norm(diff - offset, axis=1)
    return {
        'note': 'source ee_pos minus FK of the source joints; a constant offset is the base pose '
                'of the robot in the source scene, a residual is a tool-offset or kinematics mismatch',
        'mean_offset_m': offset.tolist(),
        'residual_rms_mm': float(1000 * np.sqrt(np.mean(residual ** 2))),
        'residual_max_mm': float(1000 * residual.max()),
    }


def analyze(run_dir, data, run_meta, tool=None):
    t0_ns = int(data['t0_ns']) if 't0_ns' in data else 0
    segs = label_segments(segments(data), run_meta.get('steps'), t0_ns)
    report = {
        'run_dir': str(run_dir),
        'source': str(data['source']) if 'source' in data else 'bag',
        'mode': str(data['mode']) if 'mode' in data else 'position',
        'segments': [],
        'sampling': sampling(data),
        'robot_health': robot_health(data),
        'source_frame_check': source_frame_check(run_dir, tool),
    }
    for seg in segs:
        entry = dict(seg)
        entry['duration'] = seg['t_end'] - seg['t_start']
        entry['joint'] = joint_metrics(data, seg)
        entry['tcp'] = tcp_metrics(data, seg, tool)
        entry['command_limits'] = command_limits(data, seg)
        if seg['phase'] == 2:
            entry['tcp_contributions'] = tcp_contributions(data, seg, tool)
        report['segments'].append(entry)
    trajectory_segments = [s for s in report['segments'] if s['phase'] == 2]
    if trajectory_segments:
        best = trajectory_segments[0]
        report['headline'] = {
            'trajectory_duration_s': best['duration'],
            'joint_rms_error_mrad': [1000 * j['rms'] for j in best['joint']['joints']],
            'joint_max_error_mrad': [1000 * j['max_abs'] for j in best['joint']['joints']],
            'joint_lag_ms': [1000 * j['lag'] for j in best['joint']['joints']],
            'tcp_position_error_mean_mm': best['tcp']['position_error_mean_mm'],
            'tcp_position_error_max_mm': best['tcp']['position_error_max_mm'],
            'tcp_orientation_error_mean_deg': best['tcp']['orientation_error_mean_deg'],
            'tcp_orientation_error_max_deg': best['tcp']['orientation_error_max_deg'],
        }
    return report


def format_report(report, joint_names=None):
    names = joint_names or ['j%d' % (i + 1) for i in range(7)]
    lines = ['# Trajectory replay report', '',
             '- run: `%s`' % report['run_dir'],
             '- source: %s, command mode: %s' % (report['source'], report['mode'])]
    if 'headline' in report:
        h = report['headline']
        lines += ['', '## Headline (first trajectory segment, %.2f s)' % h['trajectory_duration_s'], '',
                  '| | %s |' % ' | '.join(names), '|---|' + '---|' * 7,
                  '| RMS error [mrad] | %s |' % ' | '.join('%.2f' % v for v in h['joint_rms_error_mrad']),
                  '| max error [mrad] | %s |' % ' | '.join('%.2f' % v for v in h['joint_max_error_mrad']),
                  '| lag [ms] | %s |' % ' | '.join('%.0f' % v for v in h['joint_lag_ms']),
                  '',
                  '- TCP position error: mean %.2f mm, max %.2f mm' % (
                      h['tcp_position_error_mean_mm'], h['tcp_position_error_max_mm']),
                  '- TCP orientation error: mean %.3f deg, max %.3f deg' % (
                      h['tcp_orientation_error_mean_deg'], h['tcp_orientation_error_max_deg'])]
    lines += ['', '## Segments', '']
    for seg in report['segments']:
        tcp = seg['tcp']
        lines.append('### %s (%s, %.2f s, t = %.2f .. %.2f s)' % (
            seg['name'], seg['phase_name'], seg['duration'], seg['t_start'], seg['t_end']))
        lines.append('')
        lines.append('| joint | RMS [mrad] | max [mrad] | mean [mrad] | final [mrad] | lag [ms] | ref range [rad] |')
        lines.append('|---|---|---|---|---|---|---|')
        for name, j in zip(names, seg['joint']['joints']):
            lines.append('| %s | %.2f | %.2f | %+.2f | %+.2f | %.0f | %.3f |' % (
                name, 1000 * j['rms'], 1000 * j['max_abs'], 1000 * j['mean'], 1000 * j['final'],
                1000 * j['lag'], j['reference_range']))
        lines.append('')
        lines.append('- TCP: mean %.2f mm, RMS %.2f mm, max %.2f mm, final %.2f mm; per axis RMS %s mm; '
                     'orientation mean %.3f deg, max %.3f deg; reference path %.3f m' % (
                         tcp['position_error_mean_mm'], tcp['position_error_rms_mm'],
                         tcp['position_error_max_mm'], tcp['position_error_final_mm'],
                         ', '.join('%.2f' % v for v in tcp['axis_rms_mm']),
                         tcp['orientation_error_mean_deg'], tcp['orientation_error_max_deg'],
                         tcp['reference_path_length_m']))
        if 'o_t_ee_vs_fk_mean_mm' in tcp:
            lines.append('- robot O_T_EE vs FK of measured q: mean %.3f mm, max %.3f mm' % (
                tcp['o_t_ee_vs_fk_mean_mm'], tcp['o_t_ee_vs_fk_max_mm']))
        contributions = seg.get('tcp_contributions')
        if contributions:
            lines.append('- joint contributions to the TCP error, RMS over the segment [mm]: %s (share of the error %s %%)' % (
                ', '.join('%s %.3f' % (n, v) for n, v in zip(names, contributions['share_rms_mm'])),
                ', '.join('%.0f' % v for v in contributions['share_of_error_rms_percent'])))
            lines.append('- first-order (Jacobian) residual at most %.3f mm' % contributions['first_order_residual_max_mm'])
            for peak in contributions['at_peaks']:
                lines.append('- peak %.2f mm at t = %.2f s (trajectory time %.2f s): projections %s; joint errors [mrad] %s; dominant %s' % (
                    peak['error_mm'], peak['t'], peak['elapsed'],
                    ', '.join('%s %+.3f' % (n, v) for n, v in zip(names, peak['joint_projection_mm'])),
                    ', '.join('%+.2f' % v for v in peak['joint_error_mrad']), names[peak['dominant_joint'] - 1]))
        cl = seg.get('command_limits') or {}
        if cl:
            lines.append('- command stream peaks: velocity %.0f %%, acceleration %.0f %%, jerk %.0f %% of the FR3 limits' % (
                100 * max(cl['velocity_fraction']), 100 * max(cl['acceleration_fraction']),
                100 * max(cl['jerk_fraction'])))
            for text in cl['violations']:
                lines.append('  - VIOLATION: ' + text)
        lines.append('')
    lines += ['## Sampling', '']
    for topic, s in report['sampling'].items():
        if not isinstance(s, dict):
            lines.append('- %s: %s' % (topic, s))
            continue
        lines.append('- %s: %d samples, %.0f Hz, largest gap %.1f ms, %d gaps over 5 ms' % (
            topic, s['samples'], s['rate_hz'], s['max_gap_ms'], s['gaps_over_5ms']))
    health = report['robot_health']
    if health:
        lines += ['', '## Robot health', '']
        for key, value in health.items():
            if isinstance(value, list):
                value = '[' + ', '.join('%.3g' % v if isinstance(v, float) else str(v) for v in value) + ']'
            lines.append('- %s: %s' % (key, value))
    check = report.get('source_frame_check')
    if check:
        lines += ['', '## Source frame check', '',
                  '- %s' % check['note'],
                  '- mean offset %s m, residual RMS %.2f mm, max %.2f mm' % (
                      ['%.4f' % v for v in check['mean_offset_m']], check['residual_rms_mm'],
                      check['residual_max_mm'])]
    return '\n'.join(lines) + '\n'


def write_report(run_dir, report, joint_names=None):
    with open(os.path.join(str(run_dir), 'report.json'), 'w') as handle:
        json.dump(report, handle, indent=2)
    text = format_report(report, joint_names)
    with open(os.path.join(str(run_dir), 'report.md'), 'w') as handle:
        handle.write(text)
    return text
