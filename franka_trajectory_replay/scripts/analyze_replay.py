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

"""Extract a run's bag into data.npz (once) and write report.md / report.json."""

import argparse
import json
import os
import sys

from franka_trajectory_replay.analysis import analyze, write_report
from franka_trajectory_replay.dataset import extract_bag, load_dataset
from franka_trajectory_replay.kinematics import tool_transform
from franka_trajectory_replay.runconfig import load_config


def load_run(run_dir, reextract=False):
    run_dir = os.path.expanduser(run_dir)
    with open(os.path.join(run_dir, 'run.json'), 'r') as handle:
        run_meta = json.load(handle)
    data_path = os.path.join(run_dir, 'data.npz')
    if (reextract or not os.path.exists(data_path)) and run_meta.get('kind') == 'robot':
        print('extracting %s/bag ...' % run_dir)
        extract_bag(os.path.join(run_dir, 'bag'), run_meta['topics'], run_meta['joint_names'], data_path)
    return run_dir, run_meta, load_dataset(run_dir)


def tool_from(config):
    tcp = config['tcp']
    if any(tcp['offset_xyz']) or any(tcp['offset_rpy']):
        return tool_transform(tcp['offset_xyz'], tcp['offset_rpy'])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--config', default=None)
    parser.add_argument('--reextract', action='store_true', help='read the bag again even if data.npz exists')
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir, run_meta, data = load_run(args.run_dir, args.reextract)
    report = analyze(run_dir, data, run_meta, tool_from(config))
    print(write_report(run_dir, report, run_meta.get('joint_names') or config['joint_names']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
