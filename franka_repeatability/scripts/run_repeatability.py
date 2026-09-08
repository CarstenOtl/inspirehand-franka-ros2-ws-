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

"""Drive the arm through the configured poses and record the raw 1 kHz state.

Produces a run directory containing the rosbag2 recording, the URDF that was live at the time,
and run.json with the measurement windows. Feed that directory to analyze_repeatability.py.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import rclpy
from rclpy.executors import MultiThreadedExecutor

from franka_repeatability.runconfig import default_config_path, load_config, namespaced
from franka_repeatability.target_client import TargetClient

class RepeatabilityRunner(TargetClient):
    """The measurement run: drives the configured poses and marks the averaging windows."""

    def __init__(self, config):
        super().__init__(config, node_name='repeatability_runner')

    def visit(self, pose, cycle, joint_target=None):
        timing = self.config['timing']
        commanded_ns, accepted = self.send_target(pose=pose, joint_positions=joint_target)

        time.sleep(timing['settle_seconds'])
        window_start_ns = self.get_clock().now().nanoseconds
        time.sleep(timing['dwell_seconds'])
        window_end_ns = self.get_clock().now().nanoseconds

        self.get_logger().info(
            'cycle %d, pose %s: measured %.1f s window' % (cycle, pose['name'], timing['dwell_seconds'])
        )
        return {
            'pose': pose['name'],
            'cycle': cycle,
            'commanded_ns': commanded_ns,
            'window_start_ns': window_start_ns,
            'window_end_ns': window_end_ns,
            'joint_target': accepted,
        }


class BagRecorder:
    """Wraps `ros2 bag record`, which does the 1 kHz capture that Python cannot keep up with."""

    def __init__(self, bag_dir, topics, storage_id, logger):
        self.bag_dir = str(bag_dir)
        self.topics = list(topics)
        self.storage_id = storage_id
        self.logger = logger
        self.process = None

    def start(self, timeout=20.0):
        command = ['ros2', 'bag', 'record', '-o', self.bag_dir, '-s', self.storage_id]
        command += self.topics
        self.logger.info('recording: %s' % ' '.join(command))
        self.process = subprocess.Popen(command, preexec_fn=os.setsid)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('ros2 bag record exited immediately with code %d' % self.process.returncode)
            if os.path.isdir(self.bag_dir) and any(
                name.endswith('.db3') or name.endswith('.mcap')
                for name in os.listdir(self.bag_dir)
            ):
                time.sleep(1.0)  # let the subscriptions finish matching before we move
                return
            time.sleep(0.2)
        raise TimeoutError('ros2 bag record did not start writing within %.0f s' % timeout)

    def stop(self, timeout=30.0):
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warning('ros2 bag record did not stop on SIGINT, terminating')
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=10.0)


def dry_run(node, config):
    """Resolve every pose without moving the arm, and check the frames line up."""
    print('\nDry run - the arm is not commanded.\n')
    seed = node.current_joint_positions()
    print('current joint positions: %s' % ' '.join('%+.4f' % value for value in seed))

    forward_kinematics = None
    handle, urdf_path = tempfile.mkstemp(prefix='franka_repeatability_', suffix='.urdf')
    os.close(handle)
    try:
        save_urdf(node, config, urdf_path)
        from franka_repeatability.kinematics import ForwardKinematics

        forward_kinematics = ForwardKinematics(urdf_path, config['joint_names'])
    except Exception as error:  # noqa: BLE001 - diagnostics only, never fatal here
        print('could not build forward kinematics for the cross-check: %s' % error)

    failures = 0
    for pose in config['poses']:
        print('\n--- %s ---' % pose['name'])
        print('requested position  : %s' % ' '.join('%+.4f' % v for v in pose['position']))
        try:
            solution = node.solve_ik(pose, seed)
        except Exception as error:  # noqa: BLE001 - report and continue to the next pose
            print('IK FAILED: %s' % error)
            failures += 1
            continue

        largest_step = max(abs(a - b) for a, b in zip(solution, seed))
        print('IK solution         : %s' % ' '.join('%+.4f' % value for value in solution))
        print('largest joint step  : %.4f rad from the current configuration' % largest_step)

        if forward_kinematics is not None:
            position, quaternion = forward_kinematics.pose(
                solution, config['ee_link'], config['base_frame']
            )
            error_mm = 1000.0 * float(
                sum((a - b) ** 2 for a, b in zip(position, pose['position'])) ** 0.5
            )
            print('FK of that solution : %s' % ' '.join('%+.4f' % value for value in position))
            print('FK vs requested     : %.3f mm' % error_mm)
            if error_mm > 1.0:
                print(
                    'WARNING: the forward kinematics of %s does not land on the requested pose. '
                    'ee_link in the run config and ik_link_name in the controller config are '
                    'probably different frames.' % config['ee_link']
                )
                failures += 1

    os.unlink(urdf_path)
    if failures:
        print('\n%d pose(s) need attention.\n' % failures)
        return 1
    print('\nAll poses resolve and land where the config says they should.\n')
    return 0


def save_urdf(node, config, path):
    parameters = node.remote_parameters(
        namespaced(config['namespace'], 'robot_state_publisher'), ['robot_description']
    )
    urdf = parameters.get('robot_description')
    if not urdf:
        raise RuntimeError('robot_state_publisher returned an empty robot_description')
    with open(path, 'w') as handle:
        handle.write(urdf)
    return path


def execute_run(node, config, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    urdf_path = save_urdf(node, config, os.path.join(run_dir, 'robot.urdf'))
    node.get_logger().info('captured the live URDF to %s' % urdf_path)

    controller_node = namespaced(config['namespace'], config['controller_name'])
    controller_parameters = node.remote_parameters(
        controller_node,
        ['k_gains', 'd_gains', 'motion_duration', 'max_joint_step', 'ik_link_name',
         'torque_rate_limit'],
    )
    if abs(
        float(controller_parameters.get('motion_duration') or 0.0)
        - config['timing']['motion_duration']
    ) > 1e-6:
        node.get_logger().warning(
            'timing.motion_duration (%.2f s) differs from the controller parameter (%s s); the '
            'controller wins, the config value only sets timeouts.'
            % (config['timing']['motion_duration'], controller_parameters.get('motion_duration'))
        )

    topics = [namespaced(config['namespace'], topic) for topic in config['recording']['topics']]
    available = {name for name, _ in node.get_topic_names_and_types()}
    missing = [topic for topic in topics if topic not in available]
    if missing:
        raise RuntimeError(
            'these topics are not being published: %s. Is the controller active and the robot '
            'on real hardware (franka_robot_state_broadcaster is skipped for fake hardware)?'
            % missing
        )

    recorder = BagRecorder(
        os.path.join(run_dir, 'bag'), topics, config['recording']['storage_id'], node.get_logger()
    )

    windows = []
    joint_targets = {}
    started_at = datetime.now().isoformat(timespec='seconds')

    recorder.start()
    try:
        if config['ik_mode'] == 'cached':
            node.get_logger().info('warm-up: resolving each pose once, then replaying the targets')
            for pose in config['poses']:
                _, accepted = node.send_target(pose=pose)
                if accepted is None:
                    raise RuntimeError('the controller did not report a target for %s' % pose['name'])
                joint_targets[pose['name']] = accepted
                node.get_logger().info(
                    '%s -> %s' % (pose['name'], ' '.join('%+.4f' % v for v in accepted))
                )

        for cycle in range(config['cycles']):
            for pose in config['poses']:
                target = joint_targets.get(pose['name']) if config['ik_mode'] == 'cached' else None
                windows.append(node.visit(pose, cycle, joint_target=target))
    finally:
        recorder.stop()

    metadata = {
        'started_at': started_at,
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'config': {key: value for key, value in config.items()},
        'controller_parameters': controller_parameters,
        'controller_node': controller_node,
        'joint_targets': joint_targets,
        'topics': topics,
        'windows': windows,
    }
    with open(os.path.join(run_dir, 'run.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None, help='run configuration YAML')
    parser.add_argument(
        '--namespace',
        default=None,
        help='override the namespace from the config (fake-hardware stacks run without one)',
    )
    parser.add_argument('--output', default=None, help='override output_dir from the config')
    parser.add_argument('--cycles', type=int, default=None, help='override the cycle count')
    parser.add_argument(
        '--ik-mode', choices=('cached', 'per_visit'), default=None, help='override ik_mode'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='resolve the poses and check the frames without commanding the arm',
    )
    args = parser.parse_args()

    config = load_config(args.config or default_config_path())
    if args.namespace is not None:
        config['namespace'] = args.namespace
    if args.output:
        config['output_dir'] = os.path.expanduser(args.output)
    if args.cycles:
        config['cycles'] = args.cycles
    if args.ik_mode:
        config['ik_mode'] = args.ik_mode

    rclpy.init()
    node = RepeatabilityRunner(config)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if args.dry_run:
            return dry_run(node, config)

        run_dir = os.path.join(
            config['output_dir'], datetime.now().strftime('run_%Y%m%d_%H%M%S')
        )
        metadata = execute_run(node, config, run_dir)
        print('\nRecorded %d measurement windows to %s' % (len(metadata['windows']), run_dir))
        print('Analyse with:\n  ros2 run franka_repeatability analyze_repeatability.py --run %s\n' % run_dir)
        return 0
    except KeyboardInterrupt:
        node.get_logger().warning('interrupted - the controller holds its last commanded target')
        return 130
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
