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

"""Replay a trajectory on the arm (or the simulated stack) and record everything.

The sequence, each step gated by Enter unless --yes:

  0. load + prepare the capture, check it against the FR3 limits, connect, capture HOME
     (wherever the arm is right now)
  1. [Enter]  goto the first point of the trajectory
  2. [Enter]  make sure the replay controller is the active one, start recording, play the
              trajectory, then return to its first point
  3. [Enter]  goto HOME
  4. analyse and plot (unless --no-analyze)

Ctrl-C at any time sends an abort to the controller, which decelerates and holds.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from franka_trajectory_replay.prepare import goto_duration, summarize
from franka_trajectory_replay.recording import BagRecorder
from franka_trajectory_replay.replay_client import Rejected, ReplayClient
from franka_trajectory_replay.runconfig import load_config, namespaced
from franka_trajectory_replay.trajectory_io import load_prepared, save_prepared

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_trajectory import add_trajectory_arguments, prepare_from_args  # noqa: E402


def fmt(values):
    return '[' + ' '.join('%+.4f' % v for v in values) + ']'


class Runner:
    def __init__(self, node, config, args):
        self.node = node
        self.config = config
        self.args = args
        self.steps = []
        self.recorder = None

    def log(self, text):
        print(text, flush=True)

    def gate(self, prompt):
        if self.args.yes:
            self.log('>> %s (auto)' % prompt)
            return
        try:
            input('>> %s  [Enter to continue, Ctrl-C to abort] ' % prompt)
        except EOFError:
            print('\nstdin closed - treating that as an abort (use --yes for non-interactive runs)')
            raise KeyboardInterrupt()

    def goto(self, name, target, controller_params):
        current = np.asarray(self.node.current_joint_positions())
        step = np.asarray(target) - current
        duration = goto_duration(step, controller_params['goto_max_velocity'],
                                 controller_params['goto_max_acceleration'], controller_params['goto_min_duration'])
        self.log('%s: largest joint step %.3f rad, quintic ramp of about %.1f s' % (name, np.abs(step).max(), duration))
        if np.abs(step).max() > controller_params['max_joint_step']:
            raise RuntimeError('%s is %.3f rad away, more than the controller\'s max_joint_step (%.2f)' % (
                name, np.abs(step).max(), controller_params['max_joint_step']))
        result = self.node.goto(target)
        result['name'] = name
        result['target'] = [float(v) for v in target]
        self.steps.append(result)
        time.sleep(self.config['timing']['settle_seconds'])
        arrived = np.asarray(self.node.current_joint_positions())
        self.log('   arrived, residual %s rad' % fmt(arrived - np.asarray(target)))
        return result

    def start_recording(self, run_dir):
        namespace = self.config['namespace']
        recording = self.config['recording']
        available = {name for name, _ in self.node.get_topic_names_and_types()}
        topics = {}
        for entry in recording['topics']:
            topic = namespaced(namespace, entry)
            role = ('controller_state' if entry.endswith('controller_state') else
                    'status' if entry.endswith('status') else
                    'robot_state' if entry.endswith('robot_state') else
                    'joint_states' if entry.endswith('joint_states') else entry)
            if topic in available:
                topics[role] = topic
            elif entry in recording['required']:
                raise RuntimeError('required topic %s is not published - is the controller active?' % topic)
            else:
                self.log('   note: %s is not published (fake/simulated hardware?), skipping it' % topic)
        self.recorder = BagRecorder(os.path.join(run_dir, 'bag'), topics.values(), recording['storage_id'],
                                    self.node.get_logger())
        self.recorder.start()
        return topics

    def stop_recording(self):
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_trajectory_arguments(parser)
    parser.add_argument('--prepared', action='store_true',
                        help='the trajectory argument is a prepared.npz (skip preparation)')
    parser.add_argument('--namespace', default=None, help='override the namespace from the config')
    parser.add_argument('--output', default=None, help='override output_dir')
    parser.add_argument('--cycles', type=int, default=None, help='play the trajectory this many times')
    parser.add_argument('--send-rate', type=float, default=None, help='points per second sent to the controller')
    parser.add_argument('--yes', '-y', action='store_true', help='do not wait for Enter (simulation)')
    parser.add_argument('--dry-run', action='store_true', help='prepare, check and connect, but do not move')
    parser.add_argument('--no-record', action='store_true')
    parser.add_argument('--no-analyze', action='store_true')
    parser.add_argument('--force', action='store_true', help='replay even if the limit check failed')
    parser.add_argument('--home', type=float, nargs=7, default=None,
                        help='HOME configuration to return to (default: where the arm is when the script starts)')
    parser.add_argument('--home-from', default=None, metavar='RUN_DIR',
                        help='take HOME from an earlier run\'s run.json (after an interrupted run, for example)')
    args = parser.parse_args()
    if args.home_from:
        with open(os.path.join(os.path.expanduser(args.home_from), 'run.json')) as handle:
            args.home = json.load(handle)['home']

    config = load_config(args.config)
    if args.namespace is not None:
        config['namespace'] = args.namespace
    if args.output:
        config['output_dir'] = os.path.expanduser(args.output)
    cycles = args.cycles or config['timing']['cycles']

    # --- the trajectory ---------------------------------------------------------------------
    if args.prepared:
        prepared, extras, meta = load_prepared(args.trajectory)
        source = None
        print('loaded prepared stream %s' % args.trajectory)
    else:
        source, prepared = prepare_from_args(args, config)
        meta = {'params': prepared.params, 'report': prepared.report, 'source_meta': source.meta}
    print(summarize(prepared, config['joint_names']))
    if prepared.report and not prepared.report['ok'] and not args.force:
        print('\nthe prepared trajectory violates the limits; adjust cutoff / time scale, or pass --force '
              '(the controller will still reject a stream over the velocity or acceleration limit)')
        return 1

    # --- connect ----------------------------------------------------------------------------
    rclpy.init()
    node = ReplayClient(config)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    runner = Runner(node, config, args)
    run_dir = None
    exit_code = 0
    try:
        print('\nconnecting to %s ...' % node.manager_ns)
        node.ensure_active(runner.log)
        controller_params = node.controller_parameters()
        print('controller: %s mode, rate limiter %s, goto <= %.2f rad/s / %.2f rad/s^2, k_gains %s' % (
            controller_params['command_interface'], 'on' if controller_params['rate_limit'] else 'off',
            controller_params['goto_max_velocity'], controller_params['goto_max_acceleration'],
            controller_params['k_gains']))
        current = np.asarray(node.current_joint_positions())
        home = np.asarray(args.home, dtype=float) if args.home else current
        first = prepared.q[0]
        print('current configuration        : %s' % fmt(current))
        print('HOME (%s): %s' % ('given' if args.home else 'current configuration', fmt(home)))
        print('first trajectory point       : %s' % fmt(first))
        print('largest joint step to it     : %.3f rad' % np.abs(first - home).max())
        if args.dry_run:
            print('\ndry run - not moving.')
            return 0

        run_dir = os.path.join(config['output_dir'], datetime.now().strftime('run_%Y%m%d_%H%M%S'))
        os.makedirs(run_dir, exist_ok=True)
        save_prepared(os.path.join(run_dir, 'prepared.npz'), prepared, source, meta)
        urdf = node.robot_description()
        if urdf:
            with open(os.path.join(run_dir, 'robot.urdf'), 'w') as handle:
                handle.write(urdf)
        run_meta = {
            'kind': 'robot', 'started_at': datetime.now().isoformat(timespec='seconds'),
            'run_dir': run_dir, 'source': getattr(source, 'source', args.trajectory),
            'source_meta': getattr(source, 'meta', {}), 'prepare': prepared.params,
            'limit_report': prepared.report, 'home': home.tolist(), 'joint_names': config['joint_names'],
            'controller': node.controller_ns, 'controller_parameters': controller_params,
            'config': {k: v for k, v in config.items() if k != 'config_path'}, 'cycles': cycles,
        }

        # --- 1. goto the first point ----------------------------------------------------------
        runner.gate('The arm will move to the first point of the trajectory (%.1f s ramp).' % goto_duration(
            first - home, controller_params['goto_max_velocity'], controller_params['goto_max_acceleration'],
            controller_params['goto_min_duration']))
        # Recording covers the whole session from here to HOME, so the approach and return
        # ramps are in the bag too (they are cheap and useful when something looks off).
        topics = {}
        if not args.no_record:
            topics = runner.start_recording(run_dir)
        run_meta['topics'] = topics
        runner.goto('goto_start', first, controller_params)

        # --- 2. replay -------------------------------------------------------------------------
        runner.gate('The arm will replay the trajectory (%.1f s%s) and return to its first point.' % (
            prepared.duration, '' if cycles == 1 else ', %d cycles' % cycles))
        node.ensure_active(runner.log)
        for cycle in range(cycles):
            result = node.send_trajectory(prepared, args.send_rate or config['prepare']['send_rate'])
            result['name'] = 'trajectory'
            result['cycle'] = cycle
            runner.steps.append(result)
            runner.log('trajectory %d/%d done' % (cycle + 1, cycles))
            time.sleep(config['timing']['settle_seconds'])
            runner.goto('return_to_start', first, controller_params)

        # --- 3. home ---------------------------------------------------------------------------
        runner.gate('The arm will move back to HOME %s.' % fmt(home))
        runner.goto('goto_home', home, controller_params)
        runner.stop_recording()

    except Rejected as error:
        exit_code = 3
        print('\nthe controller rejected the command: %s' % error)
    except KeyboardInterrupt:
        exit_code = 130
        node.abort()
        print('\ninterrupted - abort sent, the controller decelerates and holds')
    except Exception as error:  # noqa: BLE001 - report, then still try to save what we have
        exit_code = 1
        node.abort()
        print('\nERROR: %s' % error)
        import traceback

        traceback.print_exc()
    finally:
        runner.stop_recording()
        if run_dir is not None:
            run_meta['finished_at'] = datetime.now().isoformat(timespec='seconds')
            run_meta['steps'] = runner.steps
            run_meta['exit_code'] = exit_code
            with open(os.path.join(run_dir, 'run.json'), 'w') as handle:
                json.dump(run_meta, handle, indent=2, default=str)
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()

    if run_dir is not None and not args.no_record and not args.no_analyze and exit_code in (0, 130, 3):
        print('\nanalysing %s ...' % run_dir)
        try:
            from analyze_replay import load_run, tool_from
            from franka_trajectory_replay.analysis import analyze, write_report
            from franka_trajectory_replay.plots import Context, plot_all

            run_dir, run_meta, data = load_run(run_dir)
            report = analyze(run_dir, data, run_meta, tool_from(config))
            print(write_report(run_dir, report, config['joint_names']))
            written = plot_all(Context(run_dir, data, run_meta, tool_from(config), config['joint_names']))
            print('wrote %d figures to %s/plots' % (len(written), run_dir))
        except Exception as error:  # noqa: BLE001
            print('analysis failed: %s (re-run analyze_replay.py %s)' % (error, run_dir))
    if run_dir is not None:
        print('run directory: %s' % run_dir)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
