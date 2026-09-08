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

"""Commanding the measurement controller: one target at a time, and waiting for it to arrive.

Shared by the measurement run and by goto_pose.py, so a manual jog goes through exactly the
same path - and the same max_joint_step guard - as the run itself.
"""

import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from franka_repeatability.runconfig import namespaced

TRANSIENT_QOS = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


def parameter_value(value):
    """Unpack an rcl_interfaces ParameterValue into a plain Python value."""
    lookup = {
        1: 'bool_value',
        2: 'integer_value',
        3: 'double_value',
        4: 'string_value',
        5: 'byte_array_value',
        6: 'bool_array_value',
        7: 'integer_array_value',
        8: 'double_array_value',
        9: 'string_array_value',
    }
    if value.type not in lookup:
        return None
    result = getattr(value, lookup[value.type])
    return list(result) if value.type >= 5 else result


class TargetClient(Node):
    """Publishes targets to the measurement controller and waits for the ramp to finish.

    Every target goes through the controller, so the quintic ramp, the max_joint_step guard and
    the torque rate limiter apply to a manual jog exactly as they do to a measurement run.
    """

    def __init__(self, config, node_name='franka_repeatability_client'):
        super().__init__(node_name)
        self.config = config
        namespace = config['namespace']
        controller = config['controller_name']

        self._controller_ns = namespaced(namespace, controller)
        self._pose_publisher = self.create_publisher(
            PoseStamped, self._controller_ns + '/target_pose', 1
        )
        self._joint_publisher = self.create_publisher(
            JointState, self._controller_ns + '/target_joint_positions', 1
        )

        self._motion_active = None
        self._goal_count = 0
        self._last_goal = None
        self._joint_state = None
        self._lock = threading.Lock()

        self.create_subscription(
            Bool, self._controller_ns + '/motion_active', self._on_motion_active, TRANSIENT_QOS
        )
        self.create_subscription(
            JointState, self._controller_ns + '/active_goal', self._on_active_goal, TRANSIENT_QOS
        )
        self.create_subscription(
            JointState, namespaced(namespace, 'joint_states'), self._on_joint_state, 10
        )

        self._ik_client = self.create_client(GetPositionIK, namespaced(namespace, 'compute_ik'))

    # --- subscriptions ----------------------------------------------------------------------
    def _on_motion_active(self, msg):
        with self._lock:
            self._motion_active = bool(msg.data)

    def _on_active_goal(self, msg):
        with self._lock:
            self._goal_count += 1
            self._last_goal = list(msg.position)

    def _on_joint_state(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in self.config['joint_names']):
            with self._lock:
                self._joint_state = [positions[name] for name in self.config['joint_names']]

    # --- helpers ----------------------------------------------------------------------------
    def current_joint_positions(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_state is not None:
                    return list(self._joint_state)
            time.sleep(0.05)
        raise TimeoutError(
            'no joint states on %s - is the robot brought up?'
            % namespaced(self.config['namespace'], 'joint_states')
        )

    def wait_until(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
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

    def remote_parameters(self, node_name, names, timeout=10.0):
        client = self.create_client(GetParameters, node_name.rstrip('/') + '/get_parameters')
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise TimeoutError('%s does not expose get_parameters' % node_name)
            response = self.wait_for_future(
                client.call_async(GetParameters.Request(names=list(names))),
                timeout,
                'parameters from %s' % node_name,
            )
            return {
                name: parameter_value(value) for name, value in zip(names, response.values)
            }
        finally:
            self.destroy_client(client)

    def solve_ik(self, pose, seed, timeout=10.0):
        """Resolve a pose through the same service the controller uses."""
        if not self._ik_client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(
                'compute_ik is not available on %s - is move_group running?'
                % namespaced(self.config['namespace'], 'compute_ik')
            )

        request = GetPositionIK.Request()
        prefix = self.config['arm_prefix'] + '_' if self.config['arm_prefix'] else ''
        request.ik_request.group_name = '%s%s_arm' % (prefix, self.config['robot_type'])
        request.ik_request.avoid_collisions = True
        request.ik_request.pose_stamped = self.make_pose_message(pose)
        request.ik_request.robot_state.joint_state.name = list(self.config['joint_names'])
        request.ik_request.robot_state.joint_state.position = list(seed)

        response = self.wait_for_future(
            self._ik_client.call_async(request), timeout, 'compute_ik for %s' % pose['name']
        )
        if response.error_code.val != response.error_code.SUCCESS:
            raise RuntimeError(
                'IK failed for pose %s with MoveIt error code %d'
                % (pose['name'], response.error_code.val)
            )

        solution = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        return [solution[name] for name in self.config['joint_names']]

    def make_pose_message(self, pose):
        message = PoseStamped()
        message.header.frame_id = self.config['base_frame']
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(value) for value in pose['position']
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = (float(value) for value in pose['orientation'])
        return message

    def make_joint_message(self, positions):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.config['joint_names'])
        message.position = [float(value) for value in positions]
        return message

    # --- the run ----------------------------------------------------------------------------
    def send_target(self, pose=None, joint_positions=None, ack_timeout=15.0):
        """Command one target and block until the controller reports the ramp finished."""
        with self._lock:
            goals_before = self._goal_count

        commanded_ns = self.get_clock().now().nanoseconds
        if joint_positions is not None:
            self._joint_publisher.publish(self.make_joint_message(joint_positions))
        else:
            self._pose_publisher.publish(self.make_pose_message(pose))

        self.wait_until(
            lambda: self._goal_count > goals_before,
            ack_timeout,
            'the controller to accept the target (check its log: IK failure, or a step larger '
            'than max_joint_step)',
        )
        motion_duration = self.config['timing']['motion_duration']
        self.wait_until(lambda: self._motion_active is True, 5.0, 'the ramp to start')
        self.wait_until(
            lambda: self._motion_active is False,
            2.0 * motion_duration + 10.0,
            'the ramp to finish',
        )

        with self._lock:
            accepted = list(self._last_goal) if self._last_goal else None
        return commanded_ns, accepted

