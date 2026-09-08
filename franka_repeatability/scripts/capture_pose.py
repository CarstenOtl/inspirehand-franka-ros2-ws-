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

"""Read the arm's current TCP pose and print it as a repeatability.yaml poses block.

Drive the arm wherever you want it - the RViz interactive marker under fake hardware, or by
hand with the gravity compensation controller on the real robot - then capture. The pose is
read from the same frame pair the measurement uses, so what you capture is what gets measured.
"""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from franka_repeatability.kinematics import (
    canonical_quaternion,
    quaternion_angle,
    quaternion_mean,
)
from franka_repeatability.runconfig import default_config_path, load_config, namespaced


class PoseCapture(Node):
    def __init__(self, config, source):
        # TF is namespaced along with everything else, so this node has to sit in the same
        # namespace as the robot to see /<ns>/tf at all.
        super().__init__('capture_pose', namespace=config['namespace'])
        self.config = config
        self.source = source

        if source == 'tf':
            from tf2_ros import Buffer, TransformListener

            self._buffer = Buffer()
            self._listener = TransformListener(self._buffer, self)
        else:
            from franka_msgs.msg import FrankaRobotState

            self._latest = None
            self._lock = threading.Lock()
            self.create_subscription(
                FrankaRobotState,
                namespaced(config['namespace'], 'franka_robot_state_broadcaster', 'robot_state'),
                self._on_robot_state,
                10,
            )

    def _on_robot_state(self, msg):
        pose = msg.o_t_ee.pose
        with self._lock:
            self._latest = (
                np.array([pose.position.x, pose.position.y, pose.position.z]),
                np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                ),
            )

    def read_once(self):
        if self.source == 'tf':
            transform = self._buffer.lookup_transform(
                self.config['base_frame'], self.config['ee_link'], rclpy.time.Time()
            ).transform
            position = np.array(
                [transform.translation.x, transform.translation.y, transform.translation.z]
            )
            quaternion = np.array(
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
            )
        else:
            with self._lock:
                if self._latest is None:
                    raise RuntimeError('no robot_state received yet')
                position, quaternion = self._latest
        return position, canonical_quaternion(quaternion)

    def capture(self, samples, duration=1.0):
        """Average a burst of readings, and report how much the arm moved during it."""
        positions, quaternions = [], []
        deadline = time.monotonic() + duration
        last_error = None
        while len(positions) < samples and time.monotonic() < deadline:
            try:
                position, quaternion = self.read_once()
                positions.append(position)
                quaternions.append(quaternion)
            except Exception as error:  # noqa: BLE001 - TF may not have the transform yet
                last_error = error
            time.sleep(duration / samples)

        if not positions:
            raise RuntimeError(
                'could not read %s -> %s: %s'
                % (self.config['base_frame'], self.config['ee_link'], last_error)
            )

        positions = np.asarray(positions)
        mean_position = positions.mean(axis=0)
        mean_quaternion = quaternion_mean(np.asarray(quaternions))
        spread_mm = float(np.linalg.norm(positions - mean_position, axis=1).max() * 1000.0)
        spread_mdeg = max(
            quaternion_angle(q, mean_quaternion) for q in quaternions
        ) * 1000.0 * 180.0 / np.pi

        return mean_position, mean_quaternion, len(positions), spread_mm, spread_mdeg


def format_pose(name, position, quaternion):
    return (
        '  - name: %s\n'
        '    position: [%.6f, %.6f, %.6f]\n'
        '    orientation: [%.6f, %.6f, %.6f, %.6f]   # x, y, z, w\n'
        % (name, position[0], position[1], position[2], *quaternion)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='run configuration YAML')
    parser.add_argument(
        '--names', nargs='+', default=['P1', 'P2', 'P3'], help='names for the captured poses'
    )
    parser.add_argument(
        '--source',
        choices=('tf', 'robot_state'),
        default='tf',
        help='tf reads base_frame -> ee_link; robot_state reads the arm\'s own O_T_EE',
    )
    parser.add_argument(
        '--namespace',
        default=None,
        help=(
            'override the namespace from the config. franka_fr3_moveit_config/moveit.launch.py '
            'defaults to no namespace, so pass --namespace "" when teaching poses with it.'
        ),
    )
    parser.add_argument('--base-frame', default=None, help='override base_frame from the config')
    parser.add_argument('--ee-link', default=None, help='override ee_link from the config')
    parser.add_argument('--samples', type=int, default=50, help='readings to average per pose')
    parser.add_argument(
        '--no-prompt', action='store_true', help='capture immediately without waiting for Enter'
    )
    parser.add_argument('--output', default=None, help='also write the block to this file')
    args = parser.parse_args()

    config = load_config(args.config or default_config_path())
    if args.namespace is not None:
        config['namespace'] = args.namespace
    if args.base_frame:
        config['base_frame'] = args.base_frame
    if args.ee_link:
        config['ee_link'] = args.ee_link

    rclpy.init()
    node = PoseCapture(config, args.source)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print(
        '\nCapturing %s relative to %s via %s.\n'
        % (config['ee_link'], config['base_frame'], args.source)
    )
    blocks = []
    try:
        for name in args.names:
            if not args.no_prompt:
                try:
                    input('Move the arm to %s, then press Enter (Ctrl-C to stop): ' % name)
                except EOFError:
                    break

            position, quaternion, count, spread_mm, spread_mdeg = node.capture(args.samples)
            print(
                '  %s: %d samples, moved %.3f mm / %.1f mdeg during the capture'
                % (name, count, spread_mm, spread_mdeg)
            )
            if spread_mm > 0.5:
                print('  the arm was still moving - hold it steadier and recapture if this matters')
            blocks.append(format_pose(name, position, quaternion))
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()

    if not blocks:
        return 1

    text = 'poses:\n' + ''.join(blocks)
    print('\n# paste into %s\n%s' % (args.config or 'repeatability.yaml', text))
    if args.output:
        with open(args.output, 'w') as handle:
            handle.write(text)
        print('written to %s' % args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
