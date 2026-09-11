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

"""Talking to the Cartesian impedance replay controller.

Everything the joint client does that is not the command type is inherited: activation and
the STRICT swap away from whatever controller holds the arm, the command-id handshake on the
status stream, pause/resume/abort, the wall-clock watchdog that stops counting while paused.
"""

import threading
import time

import numpy as np
from franka_msgs.msg import FrankaRobotState
from std_msgs.msg import Empty

from franka_trajectory_replay import cartesian
from franka_trajectory_replay.replay_client import Rejected, ReplayClient
from franka_trajectory_replay.runconfig import namespaced

CONTROLLER_TYPE = 'franka_trajectory_replay/CartesianTrajectoryReplayController'


class CartesianReplayClient(ReplayClient):
    def __init__(self, config, node_name='cartesian_replay_client', **node_kwargs):
        settings = config['cartesian']
        merged = dict(config)
        merged['controller_name'] = settings['controller_name']
        super().__init__(merged, node_name=node_name, **node_kwargs)
        self.settings = settings
        self.base_frame = settings['base_frame']
        self._robot_state_lock = threading.Lock()
        self._robot_state = None
        self._peak_errors = {'position_m': 0.0, 'orientation_rad': 0.0}
        self.create_subscription(
            FrankaRobotState,
            namespaced(config['namespace'], settings['robot_state_topic']),
            self._on_robot_state, 10)

    def _create_command_publishers(self):
        from franka_trajectory_replay_msgs.msg import CartesianGoto, CartesianTrajectory

        return (self.create_publisher(CartesianGoto, self.controller_ns + '/goto', 1),
                self.create_publisher(CartesianTrajectory, self.controller_ns + '/trajectory', 1))

    # --- subscriptions ----------------------------------------------------------------------
    def _on_status(self, msg):
        super()._on_status(msg)
        # Peak tracking error while a command is running, reported when it finishes.
        status = self.status() or {}
        if status.get('phase_name') in ('goto', 'trajectory'):
            with self._lock:
                self._peak_errors['position_m'] = max(
                    self._peak_errors['position_m'], float(status.get('position_error_m', 0.0)))
                self._peak_errors['orientation_rad'] = max(
                    self._peak_errors['orientation_rad'],
                    float(status.get('orientation_error_rad', 0.0)))

    def _reset_peak_errors(self):
        with self._lock:
            self._peak_errors = {'position_m': 0.0, 'orientation_rad': 0.0}

    def peak_errors(self):
        with self._lock:
            return dict(self._peak_errors)

    def _report_peak_errors(self, description):
        peaks = self.peak_errors()
        print('  peak tracking error during the %s: %.1f mm / %.2f deg'
              % (description, 1000.0 * peaks['position_m'],
                 np.degrees(peaks['orientation_rad'])), flush=True)

    def _on_robot_state(self, msg):
        with self._robot_state_lock:
            self._robot_state = msg

    def robot_state_once(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._robot_state_lock:
                if self._robot_state is not None:
                    return self._robot_state
            time.sleep(0.05)
        raise TimeoutError('no FrankaRobotState on %s - is franka_robot_state_broadcaster running?'
                           % namespaced(self.config['namespace'], self.settings['robot_state_topic']))

    def status(self):
        # A deactivated controller stops publishing: never treat old 'idle' feedback as
        # successful completion after a hardware reflex.
        with self._lock:
            if self._status is not None and time.monotonic() - self._status_stamp > 1.0:
                raise Rejected('Cartesian replay controller feedback stopped; check the hardware log')
            return dict(self._status) if self._status else None

    # --- controller manager -----------------------------------------------------------------
    def controller_parameters(self):
        return self.remote_parameters(self.controller_ns, [
            'translational_stiffness', 'rotational_stiffness', 'nullspace_stiffness',
            'stiffness_scale', 'target_filter', 'nullspace_target', 'coriolis_compensation',
            'torque_rate_limit', 'max_trajectory_start_error_m', 'max_trajectory_start_error_rad',
            'goto_max_velocity', 'goto_max_angular_velocity', 'pause_ramp_duration',
            'max_position_error', 'max_orientation_error', 'base_frame',
            'set_collision_behavior', 'model_source', 'tool_offset_xyz', 'tool_offset_rpy'])

    def ensure_active(self, log=print):
        controllers = self.list_controllers()
        controller = controllers.get(self.controller)
        if controller is None or controller.type != CONTROLLER_TYPE:
            raise Rejected(
                'Cartesian replay requires %s of type %s; restart replay.launch.py with '
                'arm_controller:=cartesian-impedance' % (self.controller, CONTROLLER_TYPE))
        parameters = self.controller_parameters()
        if parameters.get('base_frame') != self.base_frame:
            raise Rejected('controller base_frame %r does not match the configured %r'
                           % (parameters.get('base_frame'), self.base_frame))
        super().ensure_active(log)
        self.wait_until(
            lambda: all(publisher.get_subscription_count() > 0 for publisher in (
                self._goto_publisher, self._trajectory_publisher,
                self._pause_publisher, self._resume_publisher, self._abort_publisher)),
            10.0, "the Cartesian controller's command subscriptions")
        status = self.wait_for_status()
        if status.get('command_mode') != 'cartesian_impedance':
            raise Rejected('controller %s is not publishing Cartesian impedance status' % self.controller)
        log('Cartesian replay uses the example Cartesian impedance law: translational %g N/m, '
            'rotational %g Nm/rad, nullspace %g (scale %g), target filter %g, nullspace target %s, '
            '%s model'
            % (parameters['translational_stiffness'], parameters['rotational_stiffness'],
               parameters['nullspace_stiffness'], parameters.get('stiffness_scale', 1.0),
               parameters['target_filter'], parameters['nullspace_target'],
               parameters.get('model_source', 'franka')))
        return parameters

    def check_tool(self, tool, log=print):
        """The controlled point: the controller's tool offset must be the stream's."""
        parameters = self.controller_parameters()
        cartesian.check_controller_tool(
            parameters.get('tool_offset_xyz'), parameters.get('tool_offset_rpy'), tool)
        offset = np.asarray(parameters['tool_offset_xyz'], dtype=float)
        distance = float(np.linalg.norm(offset))
        log('controlled point: %s in the flange frame, %.1f mm from it'
            % ('the flange itself' if distance == 0.0
               else np.array2string(offset, precision=4) + ' m', 1000.0 * distance))
        return parameters

    def uses_dh_model(self):
        """True when the controller computes its own pose and Jacobian (simulation)."""
        return self.controller_parameters().get('model_source', 'franka') == 'dh'

    def set_parameter_remote(self, name, value):
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        request = SetParameters.Request()
        request.parameters = [Parameter(
            name=name,
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))]
        response = self.call(SetParameters, self.controller_ns + '/set_parameters', request)
        result = response.results[0]
        if not result.successful:
            raise Rejected('%s refused %s=%g: %s' % (self.controller, name, value, result.reason))

    def set_stiffness_scale(self, scale):
        """Live stiffness_scale on the controller; blended in through the example's filter."""
        self.set_parameter_remote('stiffness_scale', scale)
        self.get_logger().info('stiffness_scale set to %g' % scale)

    # --- commanding -------------------------------------------------------------------------
    def _check_fault(self):
        status = self.status() or {}
        if status.get('tracking_fault') == 'true':
            raise Rejected('the Cartesian controller stopped on a tracking fault: %s'
                           % status.get('last_fault', 'unknown'))

    def goto(self, position, quat, nullspace=None, duration=0.0, timeout=None):
        """Ramp the pose (and nullspace) reference to a target and block until idle again."""
        active_before, _, rejections_before = self._snapshot_ids()
        started_ns = self.get_clock().now().nanoseconds
        self._reset_peak_errors()
        self._goto_publisher.publish(cartesian.goto_message(position, quat, nullspace, duration))
        command_id = self._wait_command(active_before, rejections_before,
                                        timeout or self.config['timing']['goto_timeout'], 'goto')
        self._report_peak_errors('goto')
        self._check_fault()
        return {'command_id': command_id, 'start_ns': started_ns,
                'end_ns': self.get_clock().now().nanoseconds}

    def send_trajectory(self, prepared, send_rate=None, timeout_margin=15.0,
                        on_accept=None, allow_pauses=False):
        """Send a prepared pose stream and block until it is done."""
        message = cartesian.trajectory_message(
            prepared, self.base_frame, send_rate or self.settings['send_rate'],
            stamp=self.get_clock().now().to_msg())
        active_before, _, rejections_before = self._snapshot_ids()
        started_ns = self.get_clock().now().nanoseconds
        self._reset_peak_errors()
        self._trajectory_publisher.publish(message)
        command_id = self._wait_command(active_before, rejections_before,
                                        prepared.duration + timeout_margin, 'trajectory',
                                        accept_timeout=15.0, on_accept=on_accept,
                                        allow_pauses=allow_pauses)
        self._report_peak_errors('trajectory')
        self._check_fault()
        return {'command_id': command_id, 'start_ns': started_ns,
                'end_ns': self.get_clock().now().nanoseconds,
                'points_sent': int(len(message.points))}

    def abort(self):
        with self._lock:
            before = int((self._status or {}).get('processed_command_id', 0))
        self._abort_publisher.publish(Empty())
        try:
            def stopped():
                status = self.status() or {}
                processed = int(status.get('processed_command_id', 0))
                return (processed > before
                        and int(status.get('completed_command_id', 0)) >= processed
                        and status.get('phase_name') == 'idle')

            self.wait_until(stopped, 2.0, 'the Cartesian reference to stop')
        except (Rejected, TimeoutError) as exc:
            self.get_logger().warning('could not confirm arm hold: %s' % exc)

    # --- preflight --------------------------------------------------------------------------
    def preflight(self, q_measured, log=print):
        """Refuse to switch unless the robot still reports the bare flange.

        The controller applies the tool offset itself, so an F_T_EE set in Desk would be
        applied twice. ``check_tool`` covers the other half: that the controller's tool is the
        one the pose stream was generated for.
        """
        state = self.robot_state_once()
        f_t_ee = cartesian.pose_to_matrix(state.f_t_ee)
        o_t_ee = cartesian.pose_to_matrix(state.o_t_ee)
        cartesian.check_end_effector_frame(
            f_t_ee, np.eye(4), self.settings['tool_tolerance_m'],
            self.settings['tool_tolerance_rad'])
        position, angle = cartesian.check_forward_kinematics(
            q_measured, o_t_ee, np.eye(4), self.settings['fk_tolerance_m'],
            self.settings['fk_tolerance_deg'])
        log('preflight: the robot reports the bare flange (F_T_EE identity); FK agrees with '
            'O_T_EE to %.2f mm / %.3f deg' % (1000.0 * position, np.degrees(angle)))
        return o_t_ee
