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

"""Write the figures for a run directory into <run>/plots. --compare overlays other runs."""

import argparse
import os
import sys

from franka_trajectory_replay.plots import FIGURES, Context, fig_compare, plot_all
from franka_trajectory_replay.runconfig import load_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_replay import load_run, tool_from  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--config', default=None)
    parser.add_argument('--only', nargs='+', choices=sorted(FIGURES), default=None)
    parser.add_argument('--format', nargs='+', default=['png'])
    parser.add_argument('--compare', nargs='+', default=None,
                        help='other run directories to overlay (e.g. the MuJoCo run of the same trajectory)')
    args = parser.parse_args()
    config = load_config(args.config)
    tool = tool_from(config)

    run_dir, run_meta, data = load_run(args.run_dir)
    ctx = Context(run_dir, data, run_meta, tool, run_meta.get('joint_names') or config['joint_names'])
    written = plot_all(ctx, args.only, args.format)
    if args.compare:
        contexts = [ctx]
        labels = [os.path.basename(run_dir.rstrip('/'))]
        for other in args.compare:
            other_dir, other_meta, other_data = load_run(other)
            contexts.append(Context(other_dir, other_data, other_meta, tool, ctx.joint_names))
            labels.append(os.path.basename(other_dir.rstrip('/')))
        fig = fig_compare(contexts, labels)
        for fmt in args.format:
            path = os.path.join(run_dir, 'plots', 'compare.%s' % fmt)
            fig.savefig(path, bbox_inches='tight')
            written.append(path)
    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
