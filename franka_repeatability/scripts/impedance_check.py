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

"""Verify the impedance control law is actually running, by measuring its stiffness.

Subscribes only - it never commands the arm. The controller publishes the reference it applied,
the measured position and the torque it sent, so the control law can be identified from the
data instead of taken on trust:

    tau = K * (q_ref - q) - D * dq + coriolis

A least squares fit of tau against error and velocity recovers K and D, which should come back
as the k_gains and d_gains in controllers.yaml. If they do, impedance is live. If tau is flat at
zero while error is not, it is not.

The fit needs the arm to be excited - a pure hold at equilibrium has error and torque both near
zero and identifies nothing. Either run this during a goto_pose.py move (safe, recommended) or
push the arm gently by hand. Push gently: a hard push trips a reflex, and franka_hardware's
recovery path takes the whole ros2_control_node down with it.
"""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from franka_repeatability.runconfig import default_config_path, load_config, namespaced
from franka_repeatability.target_client import parameter_value
from rcl_interfaces.srv import GetParameters


class ImpedanceCheck(Node):
    def __init__(self, config):
        super().__init__('impedance_check', namespace=config['namespace'])
        self.config = config
        self._lock = threading.Lock()
        self._rows = []
        topic = namespaced(
            config['namespace'], config['controller_name'], 'controller_state'
        )
        self._topic = topic
        self.create_subscription(JointTrajectoryControllerState, topic, self._on_state, 50)

    def _on_state(self, msg):
        with self._lock:
            self._rows.append(
                (
                    list(msg.error.positions),
                    list(msg.feedback.velocities),
                    list(msg.output.effort),
                )
            )

    def collect(self, seconds):
        with self._lock:
            self._rows = []
        deadline = time.monotonic() + seconds
        last = 0
        while time.monotonic() < deadline:
            time.sleep(0.25)
            with self._lock:
                count = len(self._rows)
            if count != last:
                remaining = deadline - time.monotonic()
                print('\r  %6d samples, %4.1f s left ' % (count, max(remaining, 0)), end='')
                sys.stdout.flush()
                last = count
        print()
        with self._lock:
            return list(self._rows)

    def gains(self):
        node = namespaced(self.config['namespace'], self.config['controller_name'])
        client = self.create_client(GetParameters, node + '/get_parameters')
        if not client.wait_for_service(timeout_sec=5.0):
            return None, None
        future = client.call_async(GetParameters.Request(names=['k_gains', 'd_gains']))
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            return None, None
        values = [parameter_value(v) for v in future.result().values]
        return values[0], values[1]


def identify(rows):
    """Least squares fit of tau = K*error - D*velocity, per joint."""
    error = np.array([r[0] for r in rows], dtype=float)
    velocity = np.array([r[1] for r in rows], dtype=float)
    torque = np.array([r[2] for r in rows], dtype=float)

    results = []
    for j in range(error.shape[1]):
        design = np.column_stack([error[:, j], -velocity[:, j]])
        # Excitation check: without spread in the error there is nothing to fit.
        spread = float(error[:, j].max() - error[:, j].min())
        if spread < 1e-5:
            results.append((None, None, None, spread))
            continue
        solution, residuals, _, _ = np.linalg.lstsq(design, torque[:, j], rcond=None)
        predicted = design @ solution
        denominator = float(((torque[:, j] - torque[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float(((torque[:, j] - predicted) ** 2).sum()) / denominator if denominator > 0 else 0.0
        results.append((float(solution[0]), float(solution[1]), r2, spread))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--namespace', default=None)
    parser.add_argument('--seconds', type=float, default=10.0, help='how long to collect')
    args = parser.parse_args()

    config = load_config(args.config or default_config_path())
    if args.namespace is not None:
        config['namespace'] = args.namespace

    rclpy.init()
    node = ImpedanceCheck(config)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        k_gains, d_gains = node.gains()
        print('\nListening on %s' % node._topic)
        print(
            'Excite the arm now: run a goto_pose.py move from another terminal, or push it\n'
            'GENTLY by hand. A hard push trips a reflex and kills ros2_control_node.\n'
        )
        rows = node.collect(args.seconds)

        if len(rows) < 50:
            print(
                '\nOnly %d samples. The controller is not publishing - it is not active, or the\n'
                'namespace is wrong. Check: ros2 control list_controllers -c %s\n'
                % (len(rows), namespaced(config['namespace'], 'controller_manager'))
            )
            return 1

        results = identify(rows)
        print('\n%d samples\n' % len(rows))
        print('  joint  measured K   configured K   measured D   configured D    fit R^2  excitation')
        verdict = []
        for j, (k, d, r2, spread) in enumerate(results):
            k_ref = k_gains[j] if k_gains else float('nan')
            d_ref = d_gains[j] if d_gains else float('nan')
            if k is None:
                print('  J%d     %-12s %-14.1f %-12s %-14.1f %-8s %.4f mrad  (too still to identify)'
                      % (j + 1, '-', k_ref, '-', d_ref, '-', spread * 1000))
                continue
            ratio = k / k_ref if k_ref else float('nan')
            verdict.append(ratio)
            print('  J%d     %-12.1f %-14.1f %-12.1f %-14.1f %-8.3f %.4f mrad'
                  % (j + 1, k, k_ref, d, d_ref, r2, spread * 1000))

        print()
        if not verdict:
            print('The arm never moved enough to identify anything. Re-run while commanding a move.')
            return 1
        median = float(np.median(verdict))
        if 0.8 <= median <= 1.25:
            print('Impedance control is ACTIVE: measured stiffness is %.0f%% of configured.' % (median * 100))
            return 0
        print(
            'Measured stiffness is %.0f%% of configured. That is not the configured law - check\n'
            'that the gains in controllers.yaml are the ones the controller actually loaded.'
            % (median * 100)
        )
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
