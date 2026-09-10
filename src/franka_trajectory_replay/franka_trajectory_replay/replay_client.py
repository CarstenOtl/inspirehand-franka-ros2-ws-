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

"""Talking to the replay controller: activation, goto, trajectory, and waiting for completion."""

import threading
import time

import numpy as np
from controller_manager_msgs.srv import (ConfigureController, ListControllers, LoadController,
                                         SwitchController)
from diagnostic_msgs.msg import DiagnosticArray
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_trajectory_replay.runconfig import namespaced

BROADCASTER_TYPES = ('broadcaster',)


class Rejected(RuntimeError):
    pass


def parameter_value(value):
    lookup = {1: 'bool_value', 2: 'integer_value', 3: 'double_value', 4: 'string_value',
              5: 'byte_array_value', 6: 'bool_array_value', 7: 'integer_array_value',
              8: 'double_array_value', 9: 'string_array_value'}
    if value.type not in lookup:
        return None
    result = getattr(value, lookup[value.type])
    return list(result) if value.type >= 5 else result


class ReplayClient(Node):
    def __init__(self, config, node_name='trajectory_replay_client', **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self.config = config
        namespace = config['namespace']
        self.controller = config['controller_name']
        self.controller_ns = namespaced(namespace, self.controller)
        self.manager_ns = namespaced(namespace, config['controller_manager'])

        self._goto_publisher = self.create_publisher(JointState, self.controller_ns + '/goto', 1)
        self._trajectory_publisher = self.create_publisher(
            JointTrajectory, self.controller_ns + '/trajectory', 1)
        self._pause_publisher = self.create_publisher(Empty, self.controller_ns + '/pause', 1)
        self._resume_publisher = self.create_publisher(Empty, self.controller_ns + '/resume', 1)
        self._abort_publisher = self.create_publisher(Empty, self.controller_ns + '/abort', 1)

        self._lock = threading.Lock()
        self._status = None
        self._status_stamp = 0.0
        self._joint_state = None
        self._service_clients = {}
        self.create_subscription(DiagnosticArray, self.controller_ns + '/status', self._on_status, 10)
        self.create_subscription(
            JointState, namespaced(namespace, config['joint_state_topic']), self._on_joint_state, 10)

    # --- subscriptions ----------------------------------------------------------------------
    def _on_status(self, msg):
        if not msg.status:
            return
        values = {kv.key: kv.value for kv in msg.status[0].values}
        with self._lock:
            self._status = values
            self._status_stamp = time.monotonic()

    def _on_joint_state(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.config['joint_names']):
            with self._lock:
                self._joint_state = [positions[name] for name in self.config['joint_names']]

    # --- helpers ----------------------------------------------------------------------------
    def status(self):
        with self._lock:
            return dict(self._status) if self._status else None

    def current_joint_positions(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_state is not None:
                    return list(self._joint_state)
            time.sleep(0.05)
        raise TimeoutError('no joint states on %s - is the robot brought up?'
                           % namespaced(self.config['namespace'], self.config['joint_state_topic']))

    def wait_for_status(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.status() is not None:
                return self.status()
            time.sleep(0.05)
        raise TimeoutError('no status from %s/status - is the controller active?' % self.controller_ns)

    def wait_until(self, predicate, timeout, description, progress=None):
        deadline = time.monotonic() + timeout
        last_report = 0.0
        while time.monotonic() < deadline:
            if predicate():
                return
            if progress is not None and time.monotonic() - last_report > 1.0:
                progress()
                last_report = time.monotonic()
            time.sleep(0.02)
        raise TimeoutError('timed out after %.1f s waiting for %s' % (timeout, description))

    def wait_for_future(self, future, timeout, description):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            future.cancel()
            raise TimeoutError('timed out after %.1f s waiting for %s' % (timeout, description))
        return future.result()

    def call(self, service_type, name, request, timeout=10.0):
        key = (service_type, name)
        client = self._service_clients.get(key)
        if client is None:
            client = self.create_client(service_type, name)
            self._service_clients[key] = client
        if not client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError('service %s is not available' % name)
        return self.wait_for_future(client.call_async(request), timeout, name)

    def remote_parameters(self, node_name, names, timeout=10.0):
        response = self.call(GetParameters, node_name.rstrip('/') + '/get_parameters',
                             GetParameters.Request(names=list(names)), timeout)
        return {name: parameter_value(value) for name, value in zip(names, response.values)}

    def controller_parameters(self):
        return self.remote_parameters(self.controller_ns, [
            'command_interface', 'rate_limit', 'k_gains', 'd_gains', 'goto_max_velocity',
            'goto_max_acceleration', 'goto_min_duration', 'max_joint_step',
            'max_trajectory_start_error', 'coriolis_compensation', 'set_collision_behavior',
            'torque_rate_limit', 'pause_ramp_duration', 'stiffness_scale',
            'gain_ramp_duration'])

    def robot_description(self):
        return self.remote_parameters(
            namespaced(self.config['namespace'], 'robot_state_publisher'), ['robot_description']
        ).get('robot_description')

    # --- controller manager -----------------------------------------------------------------
    def list_controllers(self):
        response = self.call(ListControllers, self.manager_ns + '/list_controllers',
                             ListControllers.Request(), 15.0)
        return {c.name: c for c in response.controller}

    def ensure_active(self, log=print):
        """Make the replay controller the one driving the arm.

        Loads and configures it if the launch file did not, deactivates whatever other
        controller claims the arm's command interfaces, then activates it. Broadcasters are
        left alone.
        """
        controllers = self.list_controllers()
        if self.controller not in controllers:
            log('controller %s is not loaded - loading it' % self.controller)
            response = self.call(LoadController, self.manager_ns + '/load_controller',
                                 LoadController.Request(name=self.controller), 30.0)
            if not response.ok:
                raise RuntimeError('could not load %s' % self.controller)
            controllers = self.list_controllers()
        state = controllers[self.controller].state
        if state == 'unconfigured':
            log('configuring %s' % self.controller)
            response = self.call(ConfigureController, self.manager_ns + '/configure_controller',
                                 ConfigureController.Request(name=self.controller), 60.0)
            if not response.ok:
                raise RuntimeError('could not configure %s (see the controller log)' % self.controller)
            controllers = self.list_controllers()
            state = controllers[self.controller].state
        if state == 'active':
            log('controller %s is active' % self.controller)
            return
        joints = set(self.config['joint_names'])
        to_stop = []
        for name, info in controllers.items():
            if name == self.controller or info.state != 'active':
                continue
            if 'broadcaster' in info.type.lower():
                continue
            claimed = {interface.split('/')[0] for interface in info.claimed_interfaces}
            if claimed & joints:
                to_stop.append(name)
        if to_stop:
            log('deactivating %s (they hold the arm command interfaces)' % ', '.join(to_stop))
        request = SwitchController.Request()
        request.activate_controllers = [self.controller]
        request.deactivate_controllers = to_stop
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        from rclpy.duration import Duration
        request.timeout = Duration(seconds=10.0).to_msg()
        response = self.call(SwitchController, self.manager_ns + '/switch_controller', request, 30.0)
        if not response.ok:
            raise RuntimeError('switch_controller refused to activate %s' % self.controller)
        self.wait_until(lambda: self.list_controllers()[self.controller].state == 'active', 10.0,
                        '%s to become active' % self.controller)
        log('activated %s' % self.controller)

    # --- commanding -------------------------------------------------------------------------
    def _snapshot_ids(self):
        status = self.wait_for_status()
        return int(status['active_command_id']), int(status['completed_command_id']), int(status['rejections'])

    def _wait_command(self, active_before, rejections_before, timeout, description,
                      accept_timeout=5.0, on_accept=None, allow_pauses=False):
        def accepted():
            status = self.status()
            return status is not None and (int(status['active_command_id']) > active_before
                                           or int(status['rejections']) > rejections_before)
        self.wait_until(accepted, accept_timeout, 'the controller to acknowledge the %s' % description)
        status = self.status()
        if int(status['rejections']) > rejections_before:
            raise Rejected(status.get('last_rejection', 'rejected'))
        command_id = int(status['active_command_id'])
        if on_accept is not None:
            on_accept()

        def finished():
            status = self.status()
            return status is not None and int(status['completed_command_id']) >= command_id \
                and status['phase_name'] == 'idle'

        # A deliberate interactive pause can last indefinitely. Count only
        # unpaused wall time against the normal watchdog while continuing to
        # require fresh controller status through status().
        deadline = time.monotonic() + timeout
        previous = time.monotonic()
        last_report = 0.0
        while True:
            if finished():
                break
            now = time.monotonic()
            status = self.status()
            if allow_pauses and status and status.get('pause_requested') == 'true':
                deadline += now - previous
            if now >= deadline:
                raise TimeoutError('timed out after %.1f active seconds waiting for the %s to finish'
                                   % (timeout, description))
            if status and now - last_report > 1.0:
                state = 'paused' if status.get('paused') == 'true' else status['phase_name']
                print('  %s: %.1f / %.1f s' % (state, float(status['elapsed']),
                                               float(status['duration'])), flush=True)
                last_report = now
            previous = now
            time.sleep(0.02)
        return command_id

    def goto(self, positions, timeout=None):
        """Ramp to a joint target and block until the controller reports idle again."""
        active_before, _, rejections_before = self._snapshot_ids()
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.config['joint_names'])
        message.position = [float(value) for value in positions]
        started_ns = self.get_clock().now().nanoseconds
        self._goto_publisher.publish(message)
        command_id = self._wait_command(active_before, rejections_before,
                                        timeout or self.config['timing']['goto_timeout'], 'goto')
        return {'command_id': command_id, 'start_ns': started_ns,
                'end_ns': self.get_clock().now().nanoseconds}

    def send_trajectory(self, prepared, send_rate=None, timeout_margin=15.0,
                        on_accept=None, allow_pauses=False):
        """Send a prepared trajectory (positions and velocities) and block until it is done."""
        send_rate = float(send_rate or prepared.rate)
        stride = max(1, int(round(prepared.rate / send_rate)))
        indices = np.arange(0, len(prepared.t), stride)
        if indices[-1] != len(prepared.t) - 1:
            indices = np.append(indices, len(prepared.t) - 1)

        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = list(self.config['joint_names'])
        points = []
        for k in indices:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in prepared.q[k]]
            point.velocities = [float(v) for v in prepared.qd[k]]
            seconds = float(prepared.t[k])
            point.time_from_start.sec = int(seconds)
            point.time_from_start.nanosec = int(round((seconds - int(seconds)) * 1e9))
            points.append(point)
        message.points = points

        active_before, _, rejections_before = self._snapshot_ids()
        started_ns = self.get_clock().now().nanoseconds
        self._trajectory_publisher.publish(message)
        command_id = self._wait_command(active_before, rejections_before,
                                        prepared.duration + timeout_margin, 'trajectory',
                                        accept_timeout=15.0, on_accept=on_accept,
                                        allow_pauses=allow_pauses)
        return {'command_id': command_id, 'start_ns': started_ns,
                'end_ns': self.get_clock().now().nanoseconds, 'points_sent': int(len(points))}

    def _set_paused(self, requested, timeout=2.0):
        publisher = self._pause_publisher if requested else self._resume_publisher
        publisher.publish(Empty())
        expected = 'true' if requested else 'false'
        self.wait_until(
            lambda: (self.status() or {}).get('pause_requested') == expected,
            timeout,
            'the controller to acknowledge %s' % ('pause' if requested else 'resume'),
        )

    def pause(self):
        self._set_paused(True)
        self.wait_until(
            lambda: (self.status() or {}).get('paused') == 'true',
            3.0,
            'the trajectory clock to stop',
        )

    def resume(self):
        self._set_paused(False)

    def abort(self):
        self._abort_publisher.publish(Empty())
