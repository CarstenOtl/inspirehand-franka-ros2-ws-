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

"""Move the arm to a single target through the measurement controller.

Goes through the same path as a measurement run, so the quintic ramp, the max_joint_step guard
and the torque rate limiter all apply. Useful for positioning the arm before a run, for checking
a pose by eye, and for driving to a pose you want to capture.
"""

import argparse
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from franka_repeatability.runconfig import default_config_path, load_config
from franka_repeatability.target_client import TargetClient

# The FR3 ready configuration, the same one move_to_start_example_controller drives to.
READY = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='run configuration YAML')

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--name', help='a pose from the config, by name')
    target.add_argument(
        '--position',
        nargs=3,
        type=float,
        metavar=('X', 'Y', 'Z'),
        help='Cartesian target in metres, in the config base_frame',
    )
    target.add_argument(
        '--joints', nargs=7, type=float, metavar='Q', help='joint target in radians'
    )
    target.add_argument(
        '--ready', action='store_true', help='the FR3 ready configuration, as a joint target'
    )

    parser.add_argument(
        '--orientation',
        nargs=4,
        type=float,
        default=None,
        metavar=('X', 'Y', 'Z', 'W'),
        help='quaternion for --position; defaults to the first configured pose\'s orientation',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='resolve a Cartesian target through compute_ik and print it, without moving',
    )
    args = parser.parse_args()

    config = load_config(args.config or default_config_path())

    pose = None
    joints = None
    if args.ready:
        joints = list(READY)
    elif args.joints:
        joints = list(args.joints)
    elif args.name:
        matches = [entry for entry in config['poses'] if entry['name'] == args.name]
        if not matches:
            raise SystemExit(
                'no pose named %r in the config; available: %s'
                % (args.name, [entry['name'] for entry in config['poses']])
            )
        pose = matches[0]
    else:
        orientation = args.orientation or config['poses'][0]['orientation']
        pose = {'name': 'cli', 'position': list(args.position), 'orientation': list(orientation)}
        if args.orientation is None:
            print(
                'no --orientation given, using %s from pose %r'
                % (orientation, config['poses'][0]['name'])
            )

    rclpy.init()
    node = TargetClient(config, node_name='goto_pose')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if args.dry_run:
            if pose is None:
                print('joint target: %s' % ' '.join('%+.4f' % value for value in joints))
                return 0
            seed = node.current_joint_positions()
            solution = node.solve_ik(pose, seed)
            print('IK solution       : %s' % ' '.join('%+.4f' % value for value in solution))
            print(
                'largest joint step: %.4f rad'
                % max(abs(a - b) for a, b in zip(solution, seed))
            )
            return 0

        print('commanding the target; the controller ramps over its motion_duration ...')
        _, accepted = node.send_target(pose=pose, joint_positions=joints)
        if accepted:
            print('arrived at: %s' % ' '.join('%+.4f' % value for value in accepted))
        return 0
    except Exception as error:  # noqa: BLE001 - a rejected target is a normal outcome here
        print('failed: %s' % error, file=sys.stderr)
        return 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
