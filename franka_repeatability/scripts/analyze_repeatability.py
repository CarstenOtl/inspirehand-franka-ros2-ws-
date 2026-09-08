#!/usr/bin/env python3
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

"""Turn a recorded run into TCP repeatability and joint tracking numbers.

Reads the run directory written by run_repeatability.py and produces report.md, report.json and
CSV exports. Runs entirely offline - no robot, no move_group.
"""

import argparse
import bisect
import csv
import json
import os
import sys

import numpy as np

from franka_repeatability.bag_io import read_messages, stamp_to_ns
from franka_repeatability.kinematics import (
    ForwardKinematics,
    orientation_repeatability,
    position_repeatability,
    quaternion_mean,
)

NS_PER_S = 1e9


def pose_to_arrays(pose):
    position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    quaternion = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=float,
    )
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return position, quaternion


def load_run(run_dir):
    with open(os.path.join(run_dir, 'run.json'), 'r') as handle:
        return json.load(handle)


def collect(run_dir, metadata):
    """Pull the two recorded streams out of the bag, keyed by measurement window."""
    config = metadata['config']

    def find_topic(suffix):
        matches = [topic for topic in metadata['topics'] if topic.endswith(suffix)]
        if not matches:
            raise RuntimeError(
                'run.json records no topic ending in %r; recorded: %s' % (suffix, metadata['topics'])
            )
        return matches[0]

    controller_topic = find_topic('controller_state')
    state_topic = find_topic('robot_state')

    windows = sorted(metadata['windows'], key=lambda window: window['window_start_ns'])
    settle_ns = int(config['timing']['settle_seconds'] * NS_PER_S)
    # Windows never overlap and are sorted, so a bisect on their start times finds the only
    # candidate for a sample instead of scanning every window per message.
    commanded_starts = [window['commanded_ns'] for window in windows]

    # Per window: the raw arm measurements inside the averaging window.
    measured = {
        index: {'t': [], 'q': [], 'ee_p': [], 'ee_q': [], 'success': [], 'mode': []}
        for index in range(len(windows))
    }
    # Per window: the controller reference against the measurement, split by phase.
    tracking = {
        index: {phase: {'t': [], 'ref': [], 'fb': [], 'err': [], 'tau': []}
                for phase in ('motion', 'dwell')}
        for index in range(len(windows))
    }

    def window_for(timestamp_ns):
        """Which window a sample belongs to, and in which phase."""
        index = bisect.bisect_right(commanded_starts, timestamp_ns) - 1
        if index < 0:
            return None, None
        window = windows[index]
        if timestamp_ns > window['window_end_ns']:
            return None, None  # between the end of a dwell and the next command
        if timestamp_ns >= window['window_start_ns']:
            return index, 'dwell'
        if timestamp_ns < window['window_start_ns'] - settle_ns:
            return index, 'motion'
        return index, 'settle'

    counts = {state_topic: 0, controller_topic: 0}
    for topic, message, _ in read_messages(run_dir_bag(run_dir), topics=set(metadata['topics'])):
        counts[topic] = counts.get(topic, 0) + 1
        timestamp_ns = stamp_to_ns(message.header.stamp)
        index, phase = window_for(timestamp_ns)
        if index is None:
            continue

        if topic == state_topic:
            if phase != 'dwell':
                continue
            entry = measured[index]
            entry['t'].append(timestamp_ns)
            entry['q'].append(list(message.measured_joint_state.position))
            position, quaternion = pose_to_arrays(message.o_t_ee.pose)
            entry['ee_p'].append(position)
            entry['ee_q'].append(quaternion)
            # Without a PREEMPT_RT kernel the 1 kHz deadline is missed occasionally; libfranka
            # reports that here. Measurements taken while this dips are not trustworthy.
            entry['success'].append(float(message.control_command_success_rate))
            entry['mode'].append(int(message.robot_mode))
        elif topic == controller_topic and phase in ('motion', 'dwell'):
            entry = tracking[index][phase]
            entry['t'].append(timestamp_ns)
            entry['ref'].append(list(message.reference.positions))
            entry['fb'].append(list(message.feedback.positions))
            entry['err'].append(list(message.error.positions))
            entry['tau'].append(list(message.output.effort))

    return windows, measured, tracking, counts


def run_dir_bag(run_dir):
    return os.path.join(run_dir, 'bag')


def summarise_window(window, index, measured, tracking, forward_kinematics, config):
    samples = measured[index]
    if not samples['q']:
        raise RuntimeError(
            'no robot_state samples inside the window for %s cycle %d. The bag and run.json '
            'disagree on time - was the recording started late?' % (window['pose'], window['cycle'])
        )

    q = np.asarray(samples['q'], dtype=float)
    q_mean = q.mean(axis=0)
    q_std = q.std(axis=0, ddof=1) if len(q) > 1 else np.zeros(7)

    fk_position, fk_quaternion = forward_kinematics.pose(
        q_mean, config['ee_link'], config['base_frame']
    )
    ee_position = np.asarray(samples['ee_p'], dtype=float).mean(axis=0)
    ee_quaternion = quaternion_mean(np.asarray(samples['ee_q'], dtype=float))

    result = {
        'pose': window['pose'],
        'cycle': window['cycle'],
        'samples': int(len(q)),
        'duration_s': float((samples['t'][-1] - samples['t'][0]) / NS_PER_S),
        'joint_mean_rad': q_mean.tolist(),
        'joint_std_rad': q_std.tolist(),
        'fk_position_m': fk_position.tolist(),
        'fk_quaternion_xyzw': fk_quaternion.tolist(),
        'o_t_ee_position_m': ee_position.tolist(),
        'o_t_ee_quaternion_xyzw': ee_quaternion.tolist(),
        # Cross-check: our forward kinematics against the pose libfranka reports for the same
        # configuration. A large value means ee_link is not the frame the robot calls O_T_EE.
        'fk_vs_o_t_ee_mm': float(np.linalg.norm(fk_position - ee_position) * 1000.0),
        'control_command_success_rate_min': float(np.min(samples['success'])),
        'control_command_success_rate_mean': float(np.mean(samples['success'])),
        # 2 == ROBOT_MODE_MOVE. Anything else during a measurement window means the arm was not
        # under our control for part of it.
        'robot_modes': sorted(set(samples['mode'])),
    }

    dwell = tracking[index]['dwell']
    if dwell['err']:
        error = np.asarray(dwell['err'], dtype=float)
        reference = np.asarray(dwell['ref'], dtype=float)
        feedback = np.asarray(dwell['fb'], dtype=float)
        torque = np.asarray(dwell['tau'], dtype=float)

        reference_position, _ = forward_kinematics.pose(
            reference.mean(axis=0), config['ee_link'], config['base_frame']
        )
        feedback_position, _ = forward_kinematics.pose(
            feedback.mean(axis=0), config['ee_link'], config['base_frame']
        )
        result['tracking_dwell'] = {
            'samples': int(len(error)),
            'mean_error_rad': error.mean(axis=0).tolist(),
            'rms_error_rad': np.sqrt((error ** 2).mean(axis=0)).tolist(),
            'max_abs_error_rad': np.abs(error).max(axis=0).tolist(),
            'mean_torque_nm': torque.mean(axis=0).tolist(),
            # The steady-state joint error expressed where it matters: at the tool.
            'cartesian_offset_m': (reference_position - feedback_position).tolist(),
            'cartesian_offset_mm': float(
                np.linalg.norm(reference_position - feedback_position) * 1000.0
            ),
        }

    motion = tracking[index]['motion']
    if motion['err']:
        error = np.asarray(motion['err'], dtype=float)
        torque = np.asarray(motion['tau'], dtype=float)
        result['tracking_motion'] = {
            'samples': int(len(error)),
            'max_abs_error_rad': np.abs(error).max(axis=0).tolist(),
            'rms_error_rad': np.sqrt((error ** 2).mean(axis=0)).tolist(),
            'max_abs_torque_nm': np.abs(torque).max(axis=0).tolist(),
            # Largest commanded torque step between consecutive control cycles. The only
            # difference between this controller and the upstream example is the rate limiter,
            # so if this stays below torque_rate_limit the limiter never engaged and the applied
            # torque was exactly what the example would have applied.
            'max_torque_step_nm': float(np.abs(np.diff(torque, axis=0)).max())
            if len(torque) > 1
            else 0.0,
        }

    return result


def aggregate(summaries, config):
    """Repeatability per pose, from the per-visit averages."""
    poses = {}
    for pose in [entry['name'] for entry in config['poses']]:
        visits = [summary for summary in summaries if summary['pose'] == pose]
        if len(visits) < 2:
            continue

        fk_positions = np.array([visit['fk_position_m'] for visit in visits])
        fk_quaternions = np.array([visit['fk_quaternion_xyzw'] for visit in visits])
        ee_positions = np.array([visit['o_t_ee_position_m'] for visit in visits])

        entry = {
            'visits': len(visits),
            'cycles': [visit['cycle'] for visit in visits],
            'position_fk': position_repeatability(fk_positions),
            'orientation_fk': orientation_repeatability(fk_quaternions),
            # The same statistic on the pose the robot reports itself, as an independent check
            # that the number is the arm and not our kinematic model.
            'position_o_t_ee': position_repeatability(ee_positions),
            'joint_std_rad': np.array([visit['joint_std_rad'] for visit in visits])
            .mean(axis=0)
            .tolist(),
        }

        dwell = [visit['tracking_dwell'] for visit in visits if 'tracking_dwell' in visit]
        if dwell:
            entry['tracking_dwell'] = {
                'mean_error_rad': np.array([d['mean_error_rad'] for d in dwell]).mean(axis=0).tolist(),
                'max_abs_error_rad': np.array([d['max_abs_error_rad'] for d in dwell]).max(axis=0).tolist(),
                'mean_cartesian_offset_mm': float(
                    np.mean([d['cartesian_offset_mm'] for d in dwell])
                ),
            }
        motion = [visit['tracking_motion'] for visit in visits if 'tracking_motion' in visit]
        if motion:
            entry['tracking_motion'] = {
                'max_abs_error_rad': np.array([m['max_abs_error_rad'] for m in motion]).max(axis=0).tolist(),
                'max_abs_torque_nm': np.array([m['max_abs_torque_nm'] for m in motion]).max(axis=0).tolist(),
            }

        poses[pose] = entry
    return poses


def write_csv(run_dir, summaries):
    path = os.path.join(run_dir, 'visits.csv')
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['pose', 'cycle', 'samples']
            + ['q%d_mean_rad' % i for i in range(1, 8)]
            + ['q%d_std_rad' % i for i in range(1, 8)]
            + ['fk_x_m', 'fk_y_m', 'fk_z_m', 'fk_qx', 'fk_qy', 'fk_qz', 'fk_qw']
            + ['o_t_ee_x_m', 'o_t_ee_y_m', 'o_t_ee_z_m']
            + ['fk_vs_o_t_ee_mm', 'dwell_cartesian_offset_mm']
            + ['dwell_mean_err%d_rad' % i for i in range(1, 8)]
            + ['motion_max_err%d_rad' % i for i in range(1, 8)]
        )
        for summary in summaries:
            dwell = summary.get('tracking_dwell', {})
            motion = summary.get('tracking_motion', {})
            writer.writerow(
                [summary['pose'], summary['cycle'], summary['samples']]
                + ['%.9f' % value for value in summary['joint_mean_rad']]
                + ['%.9f' % value for value in summary['joint_std_rad']]
                + ['%.9f' % value for value in summary['fk_position_m']]
                + ['%.9f' % value for value in summary['fk_quaternion_xyzw']]
                + ['%.9f' % value for value in summary['o_t_ee_position_m']]
                + ['%.6f' % summary['fk_vs_o_t_ee_mm'],
                   '%.6f' % dwell.get('cartesian_offset_mm', float('nan'))]
                + ['%.9f' % value for value in dwell.get('mean_error_rad', [float('nan')] * 7)]
                + ['%.9f' % value for value in motion.get('max_abs_error_rad', [float('nan')] * 7)]
            )
    return path


def format_report(metadata, summaries, poses):
    config = metadata['config']
    lines = []
    lines.append('# FR3 TCP repeatability and joint tracking\n')
    lines.append('Run started %s, finished %s.\n' % (metadata['started_at'], metadata['finished_at']))
    lines.append(
        '- controller: `%s` (%s)\n' % (metadata['controller_node'], config['ik_mode'])
    )
    lines.append('- frame measured: `%s` relative to `%s`\n' % (config['ee_link'], config['base_frame']))
    lines.append(
        '- %d poses x %d cycles, %.1f s averaged per visit\n'
        % (len(config['poses']), config['cycles'], config['timing']['dwell_seconds'])
    )
    if config['ik_mode'] == 'per_visit':
        lines.append(
            '\n> Recorded in `per_visit` mode: IK was re-solved on every visit from a slightly '
            'different seed, so the numbers below include solver variation on top of the arm.\n'
        )

    lines.append('\n## Position repeatability\n')
    lines.append('Forward kinematics of the averaged joint positions, per ISO 9283 '
                 '(RP = mean distance from the barycentre + 3 sigma).\n')
    lines.append('\n| pose | visits | RP (mm) | mean dist (mm) | max dist (mm) | sigma x/y/z (mm) |\n')
    lines.append('| --- | --- | --- | --- | --- | --- |\n')
    for name, entry in poses.items():
        position = entry['position_fk']
        lines.append(
            '| %s | %d | %.3f | %.3f | %.3f | %.3f / %.3f / %.3f |\n'
            % (
                name,
                entry['visits'],
                position['repeatability_rp_m'] * 1000.0,
                position['mean_distance_m'] * 1000.0,
                position['max_distance_m'] * 1000.0,
                position['per_axis_std_m'][0] * 1000.0,
                position['per_axis_std_m'][1] * 1000.0,
                position['per_axis_std_m'][2] * 1000.0,
            )
        )

    lines.append('\n## Orientation repeatability\n')
    lines.append('\n| pose | RP (mdeg) | mean (mdeg) | max (mdeg) |\n| --- | --- | --- | --- |\n')
    for name, entry in poses.items():
        orientation = entry['orientation_fk']
        to_mdeg = 1000.0 * 180.0 / np.pi
        lines.append(
            '| %s | %.1f | %.1f | %.1f |\n'
            % (
                name,
                orientation['repeatability_rad'] * to_mdeg,
                orientation['mean_angle_rad'] * to_mdeg,
                orientation['max_angle_rad'] * to_mdeg,
            )
        )

    lines.append('\n## Cross-check against the robot\'s own O_T_EE\n')
    lines.append('\n| pose | RP from FK (mm) | RP from O_T_EE (mm) | mean FK vs O_T_EE offset (mm) |\n')
    lines.append('| --- | --- | --- | --- |\n')
    for name, entry in poses.items():
        visits = [summary for summary in summaries if summary['pose'] == name]
        lines.append(
            '| %s | %.3f | %.3f | %.3f |\n'
            % (
                name,
                entry['position_fk']['repeatability_rp_m'] * 1000.0,
                entry['position_o_t_ee']['repeatability_rp_m'] * 1000.0,
                float(np.mean([visit['fk_vs_o_t_ee_mm'] for visit in visits])),
            )
        )

    lines.append('\n## Tracking: commanded reference vs measured position\n')
    lines.append(
        '\nThe reference is the joint position the impedance law was actually applied to, '
        'published by the controller at the update rate. A joint impedance law is a spring: a '
        'steady-state offset under gravity load is expected, not a fault.\n'
    )
    lines.append('\n### Steady state, during the averaging window\n')
    lines.append('\n| pose | mean joint error (mrad) J1..J7 | at the tool (mm) |\n| --- | --- | --- |\n')
    for name, entry in poses.items():
        dwell = entry.get('tracking_dwell')
        if not dwell:
            continue
        lines.append(
            '| %s | %s | %.3f |\n'
            % (
                name,
                ' '.join('%+.2f' % (value * 1000.0) for value in dwell['mean_error_rad']),
                dwell['mean_cartesian_offset_mm'],
            )
        )

    lines.append('\n### Peak, during the ramp between poses\n')
    lines.append('\n| pose | max abs joint error (mrad) J1..J7 | max abs torque (Nm) |\n| --- | --- | --- |\n')
    for name, entry in poses.items():
        motion = entry.get('tracking_motion')
        if not motion:
            continue
        lines.append(
            '| %s | %s | %s |\n'
            % (
                name,
                ' '.join('%.1f' % (value * 1000.0) for value in motion['max_abs_error_rad']),
                ' '.join('%.1f' % value for value in motion['max_abs_torque_nm']),
            )
        )

    limit = metadata.get('controller_parameters', {}).get('torque_rate_limit')
    steps = [
        summary['tracking_motion']['max_torque_step_nm']
        for summary in summaries
        if 'tracking_motion' in summary
    ]
    if steps:
        largest = max(steps)
        lines.append('\n## Torque rate limiter\n')
        lines.append(
            '\nLargest commanded torque step between consecutive cycles: %.4f Nm, against a '
            'torque_rate_limit of %s Nm. %s\n'
            % (
                largest,
                limit,
                'The limiter never engaged, so the applied torque was exactly what the upstream '
                'joint_impedance_with_ik control law produces.'
                if limit and largest < 0.98 * float(limit)
                else 'The limiter clipped the command, so the applied torque differs from the '
                'unlimited upstream example. Set torque_rate_limit to 0.0 to disable it.',
            )
        )

    lines.append('\n## Control loop health\n')
    worst = min(summary['control_command_success_rate_min'] for summary in summaries)
    mean_rate = float(np.mean([s['control_command_success_rate_mean'] for s in summaries]))
    modes = sorted({mode for summary in summaries for mode in summary['robot_modes']})
    lines.append(
        '\n`control_command_success_rate` mean %.4f, worst %.4f across all windows. Robot modes '
        'seen: %s (2 = MOVE). This workstation has no PREEMPT_RT kernel, so libfranka runs with '
        '`RealtimeConfig::kIgnore` and the 1 kHz deadline can slip; a rate meaningfully below '
        '1.0 means missed control cycles, and the tracking numbers above should be read as an '
        'upper bound on the arm\'s real performance rather than a measurement of it.\n'
        % (mean_rate, worst, modes)
    )

    lines.append('\n## Sampling\n')
    counts = [summary['samples'] for summary in summaries]
    rates = [
        summary['samples'] / summary['duration_s']
        for summary in summaries
        if summary['duration_s'] > 0.0
    ]
    lines.append(
        '\n%d windows, %d-%d samples each over %.1f s, i.e. %.0f-%.0f Hz. The broadcaster runs '
        'at the controller update rate, so a rate well below that means the recording dropped '
        'messages and these averages rest on less data than intended.\n'
        % (
            len(counts),
            min(counts),
            max(counts),
            config['timing']['dwell_seconds'],
            min(rates) if rates else 0.0,
            max(rates) if rates else 0.0,
        )
    )
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='run directory written by run_repeatability.py')
    parser.add_argument('--urdf', default=None, help='override the captured URDF')
    args = parser.parse_args()

    run_dir = os.path.expanduser(args.run)
    metadata = load_run(run_dir)
    config = metadata['config']

    urdf_path = args.urdf or os.path.join(run_dir, 'robot.urdf')
    forward_kinematics = ForwardKinematics(urdf_path, config['joint_names'])
    for frame in (config['ee_link'], config['base_frame']):
        if not forward_kinematics.frame_exists(frame):
            raise SystemExit('frame %r is not in %s' % (frame, urdf_path))

    print('reading %s ...' % run_dir_bag(run_dir))
    windows, measured, tracking, counts = collect(run_dir, metadata)
    print('  %s' % ', '.join('%s: %d messages' % item for item in counts.items()))

    summaries = [
        summarise_window(window, index, measured, tracking, forward_kinematics, config)
        for index, window in enumerate(windows)
    ]
    poses = aggregate(summaries, config)

    report = {
        'run': run_dir,
        'started_at': metadata['started_at'],
        'ik_mode': config['ik_mode'],
        'ee_link': config['ee_link'],
        'base_frame': config['base_frame'],
        'per_pose': poses,
        'per_visit': summaries,
    }
    with open(os.path.join(run_dir, 'report.json'), 'w') as handle:
        json.dump(report, handle, indent=2)

    text = format_report(metadata, summaries, poses)
    with open(os.path.join(run_dir, 'report.md'), 'w') as handle:
        handle.write(text)
    csv_path = write_csv(run_dir, summaries)

    print(text)
    print('written: report.md, report.json, %s' % os.path.basename(csv_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
