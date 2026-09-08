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

"""Write a synthetic FR3 trajectory in the Isaac Sim capture layout, for testing the pipeline.

The motion is a sum of slow sinusoids around the ready pose, sampled at a policy-like rate,
with the end-effector pose from forward kinematics stored alongside as Isaac would. Use
``--kind step`` for a deliberately jerky trajectory that violates the limits (to see the
checks fire) and ``--kind still`` for a stationary one.
"""

import argparse
import sys

import numpy as np

from franka_trajectory_replay.kinematics import READY_POSE, flange_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', help='.npz to write')
    parser.add_argument('--rate', type=float, default=15.0, help='capture rate [Hz]')
    parser.add_argument('--duration', type=float, default=8.0, help='seconds')
    parser.add_argument('--amplitude', type=float, default=0.35, help='peak excursion [rad]')
    parser.add_argument('--kind', choices=('smooth', 'step', 'still'), default='smooth')
    parser.add_argument('--start-offset', type=float, default=0.15,
                        help='offset of the first sample from the ready pose [rad], so goto has work to do')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(round(args.duration * args.rate)) + 1
    t = np.arange(n) / args.rate
    q = np.tile(READY_POSE, (n, 1))
    if args.kind == 'smooth':
        for j in range(7):
            f1, f2 = 0.15 + 0.05 * j, 0.35 + 0.07 * j
            phase = rng.uniform(0, 2 * np.pi)
            envelope = np.sin(np.pi * t / args.duration) ** 2  # zero velocity at both ends
            q[:, j] += args.amplitude * (0.7 if j != 3 else 0.4) * envelope * (
                0.7 * np.sin(2 * np.pi * f1 * t + phase) + 0.3 * np.sin(2 * np.pi * f2 * t))
        q[:, 0] += args.start_offset
    elif args.kind == 'step':
        q[:, 0] += np.where(t > args.duration / 2, args.amplitude, 0.0)
        q[:, 3] -= np.where(t > args.duration / 3, args.amplitude, 0.0)
    ee_pos, ee_quat = flange_poses(q)
    np.savez(
        args.output,
        joint_pos_arm=q.astype(np.float32),
        arm_joint_names=np.array(['fr3_joint%d' % i for i in range(1, 8)]),
        dt=np.float64(1.0 / args.rate),
        ee_pos=ee_pos.astype(np.float32),
        ee_quat=ee_quat.astype(np.float32),
        task_id=np.array('synthetic-%s' % args.kind),
    )
    print('wrote %s: %d samples at %.0f Hz, %.1f s, kind %s' % (args.output, n, args.rate, args.duration, args.kind))
    return 0


if __name__ == '__main__':
    sys.exit(main())
