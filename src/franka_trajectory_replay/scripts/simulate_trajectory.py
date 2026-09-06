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

"""Replay a capture in MuJoCo, offline, and analyse it exactly like a run on the arm.

Produces a run directory (default ~/franka_replay_runs/sim_<stamp>) with prepared.npz,
data.npz, run.json, report.md and plots/. Add --view for the MuJoCo viewer (needs a display)
and --realtime to pace it at wall-clock speed.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

from franka_trajectory_replay.analysis import analyze, write_report
from franka_trajectory_replay.dataset import save_dataset
from franka_trajectory_replay.kinematics import READY_POSE, tool_transform
from franka_trajectory_replay.mujoco_sim import simulate_run
from franka_trajectory_replay.plots import Context, plot_all
from franka_trajectory_replay.prepare import summarize
from franka_trajectory_replay.runconfig import load_config
from franka_trajectory_replay.trajectory_io import save_prepared

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_trajectory import add_trajectory_arguments, prepare_from_args  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_trajectory_arguments(parser)
    parser.add_argument('--output', default=None, help='run directory (default: output_dir/sim_<stamp>)')
    parser.add_argument('--model', default=None, help='MJCF (default: franka_mujoco_hardware/mujoco/scene.xml)')
    parser.add_argument('--home', type=float, nargs=7, default=None, help='home configuration (default: FR3 ready pose)')
    parser.add_argument('--stiffness', type=float, nargs=7, default=None)
    parser.add_argument('--damping', type=float, nargs=7, default=None)
    parser.add_argument('--view', action='store_true', help='open the MuJoCo viewer')
    parser.add_argument('--realtime', action='store_true', help='pace the simulation at wall-clock speed')
    parser.add_argument('--force', action='store_true', help='simulate even if the limit check failed')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    source, prepared = prepare_from_args(args, config)
    print(summarize(prepared, config['joint_names']))
    if not prepared.report['ok'] and not args.force:
        print('\nthe prepared trajectory violates the limits; fix it (cutoff / time scale) or pass --force')
        return 1

    run_dir = args.output or os.path.join(config['output_dir'], datetime.now().strftime('sim_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    save_prepared(os.path.join(run_dir, 'prepared.npz'), prepared, source,
                  {'params': prepared.params, 'report': prepared.report, 'source_meta': source.meta})

    home = np.asarray(args.home if args.home else READY_POSE)
    arrays, steps, summary = simulate_run(
        prepared, home_q=home, model_path=args.model, stiffness=args.stiffness, damping=args.damping,
        settle_seconds=config['timing']['settle_seconds'], view=args.view, realtime=args.realtime)
    save_dataset(run_dir, arrays)
    run_meta = {
        'kind': 'mujoco', 'started_at': datetime.now().isoformat(timespec='seconds'),
        'source': source.source, 'source_meta': source.meta, 'prepare': prepared.params,
        'limit_report': prepared.report, 'home': home.tolist(), 'steps': steps, 'sim': summary,
        'config': {k: v for k, v in config.items() if k != 'config_path'},
    }
    with open(os.path.join(run_dir, 'run.json'), 'w') as handle:
        json.dump(run_meta, handle, indent=2, default=str)

    print('\nsimulation: rate limiter engaged %d times, %d motion-generator violations, %d samples in contact' % (
        summary['rate_limit_engaged'], summary['motion_generator_violation_count'], summary['contact_samples']))
    for violation in summary['motion_generator_violations'][:5]:
        print('  VIOLATION t=%.3f s: %s joint %d (%.3f)' % (
            violation['t'], violation['error'], violation['joint'], violation['value']))
    for contact in summary['contacts'][:5]:
        print('  CONTACT t=%.3f s: %s' % (contact['t'], ', '.join(contact['pairs'])))

    tcp = config['tcp']
    tool = tool_transform(tcp['offset_xyz'], tcp['offset_rpy']) if any(tcp['offset_xyz']) or any(tcp['offset_rpy']) else None
    report = analyze(run_dir, arrays, run_meta, tool)
    text = write_report(run_dir, report, config['joint_names'])
    print('\n' + text)
    if not args.no_plots:
        written = plot_all(Context(run_dir, arrays, run_meta, tool, config['joint_names']))
        print('wrote %d figures to %s/plots' % (len(written), run_dir))
    print('run directory: %s' % run_dir)
    bad = summary['motion_generator_violation_count'] > 0 or summary['contact_samples'] > 0
    return 2 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
