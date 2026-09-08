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

"""Load an Isaac Sim capture, resample it to the control rate, check it against the FR3 limits.

Writes prepared.npz (the exact stream the controller will play) and a preview figure. The same
loading and preparation runs inside replay_trajectory.py; use this to look at a capture first.
"""

import argparse
import os
import sys

import numpy as np

from franka_trajectory_replay.prepare import prepare, summarize
from franka_trajectory_replay.runconfig import load_config
from franka_trajectory_replay.trajectory_io import load_trajectory, save_prepared


def add_trajectory_arguments(parser):
    parser.add_argument('trajectory', help='.npy or .npz capture from Isaac Sim')
    parser.add_argument('--rate', type=float, default=None,
                        help='capture rate in Hz when the file carries no time base')
    parser.add_argument('--time-column', type=int, default=None, help='column holding time (npy)')
    parser.add_argument('--joint-map', type=int, nargs=7, default=None,
                        help='7 column indices in FR3 joint order, if the names are not Franka-like')
    parser.add_argument('--assume-order', action='store_true',
                        help='accept non-Franka joint names and take the first 7 columns as joints 1..7')
    parser.add_argument('--degrees', action='store_true', help='capture is in degrees')
    parser.add_argument('--env', type=int, default=0, help='environment index in a multi-env npz')
    parser.add_argument('--key', default=None,
                        help='npz array to replay (default: joint_pos; e.g. joint_pos_target for the policy targets)')
    parser.add_argument('--csv-prefix', default='pos', help='csv columns to replay: pos (measured) or cmd (target)')
    parser.add_argument('--config', default=None, help='replay.yaml (defaults: the packaged one)')
    parser.add_argument('--control-rate', type=float, default=None, help='override prepare.rate')
    parser.add_argument('--cutoff', type=float, default=None, help='override prepare.cutoff_hz (0 = off)')
    parser.add_argument('--interpolation', choices=('cubic', 'linear'), default=None,
                        help='cubic spline (default) or straight lines between waypoints with corner blends')
    parser.add_argument('--blend-time', type=float, default=None, help='linear mode: corner blend duration [s]')
    parser.add_argument('--time-scale', type=float, default=None, help='override prepare.time_scale (>= 1 slows down)')
    parser.add_argument('--no-auto-scale', action='store_true', help='do not slow down to satisfy the limits')
    parser.add_argument('--hold-start', type=float, default=None)
    parser.add_argument('--hold-end', type=float, default=None)


def prepare_from_args(args, config):
    settings = dict(config['prepare'])
    if args.control_rate:
        settings['rate'] = args.control_rate
    if args.cutoff is not None:
        settings['cutoff_hz'] = args.cutoff
    if args.interpolation:
        settings['interpolation'] = args.interpolation
    if args.blend_time is not None:
        settings['blend_time'] = args.blend_time
    if args.time_scale is not None:
        settings['time_scale'] = args.time_scale
    if args.no_auto_scale:
        settings['auto_scale'] = False
    if args.hold_start is not None:
        settings['hold_start'] = args.hold_start
    if args.hold_end is not None:
        settings['hold_end'] = args.hold_end
    source = load_trajectory(args.trajectory, rate=args.rate, time_column=args.time_column,
                             joint_map=args.joint_map, assume_order=args.assume_order,
                             degrees=args.degrees, env=args.env, key=args.key, csv_prefix=args.csv_prefix)
    prepared = prepare(
        source, rate=settings['rate'], cutoff_hz=settings['cutoff_hz'],
        hold_start=settings['hold_start'], hold_end=settings['hold_end'],
        time_scale=settings['time_scale'], auto_scale=settings['auto_scale'],
        velocity_margin=settings['velocity_margin'], acceleration_margin=settings['acceleration_margin'],
        jerk_margin=settings['jerk_margin'], joint_names=config['joint_names'],
        lead_in=settings['lead_in'], lead_out=settings['lead_out'],
        lead_max_acceleration=settings['lead_max_acceleration'],
        interpolation=settings['interpolation'], blend_time=settings['blend_time'])
    return source, prepared


def preview_figure(source, prepared, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from franka_trajectory_replay.plots import JOINT_COLORS

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    scale = prepared.params.get('time_scale', 1.0)
    for j in range(7):
        axes[0].plot(source.t * scale + prepared.params['hold_start'], source.q[:, j], 'o', ms=2,
                     color=JOINT_COLORS[j], alpha=0.5)
        axes[0].plot(prepared.t, prepared.q[:, j], color=JOINT_COLORS[j], label=prepared.joint_names[j] if prepared.joint_names else 'j%d' % (j + 1))
        axes[1].plot(prepared.t, prepared.qd[:, j], color=JOINT_COLORS[j])
        axes[2].plot(prepared.t, prepared.qdd[:, j], color=JOINT_COLORS[j])
    axes[0].set_ylabel('position [rad]'); axes[0].legend(ncol=7, fontsize=7, loc='upper right')
    axes[0].set_title('capture samples (dots) and the prepared stream (lines)')
    axes[1].set_ylabel('velocity [rad/s]')
    axes[2].set_ylabel('acceleration [rad/s^2]'); axes[2].set_xlabel('time [s]')
    for ax in axes:
        ax.grid(color='#e6e6e6')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_trajectory_arguments(parser)
    parser.add_argument('--output', default=None, help='prepared.npz path (default: next to the capture)')
    parser.add_argument('--plot', default=None, help='preview figure path (default: next to the output)')
    args = parser.parse_args()

    config = load_config(args.config)
    source, prepared = prepare_from_args(args, config)
    print('source: %s' % source.source)
    print('  %d samples, %.2f s, %.1f Hz, q[0] = %s' % (
        len(source.t), source.duration, source.rate, np.round(source.q[0], 4).tolist()))
    for key, value in source.meta.items():
        print('  %s: %s' % (key, str(value)[:120]))
    print(summarize(prepared, config['joint_names']))

    output = args.output or os.path.splitext(os.path.expanduser(args.trajectory))[0] + '_prepared.npz'
    save_prepared(output, prepared, source, {'params': prepared.params, 'report': prepared.report,
                                             'source_meta': source.meta})
    print('wrote %s' % output)
    plot = args.plot or os.path.splitext(output)[0] + '.png'
    preview_figure(source, prepared, plot)
    print('wrote %s' % plot)
    return 0 if prepared.report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
