"""Action client for the stock ROS 2 joint trajectory controller.

The controller claims the FR3 position interfaces.  That makes franka_hardware
start libfranka joint-position control with ControllerMode::kJointImpedance; no
impedance or torque law is implemented here.
"""

import threading
import time

import numpy as np
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ConfigureController, ListControllers, LoadController, SwitchController
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from franka_trajectory_replay.prepare import goto_duration
from franka_trajectory_replay.replay_client import Rejected
from franka_trajectory_replay.runconfig import namespaced


class JointTrajectoryClient(Node):
    """Activate and command a standard JointTrajectoryController by action."""

    def __init__(self, config, node_name="inspire_franka_trajectory_replay", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self.config = config
        namespace = config["namespace"]
        self.controller = config["controller_name"]
        self.controller_ns = namespaced(namespace, self.controller)
        self.manager_ns = namespaced(namespace, config["controller_manager"])
        self._action = ActionClient(
            self, FollowJointTrajectory, self.controller_ns + "/follow_joint_trajectory"
        )
        self._lock = threading.Lock()
        self._joint_state = None
        self._goal_handle = None
        self._service_clients = {}
        self.create_subscription(
            JointState,
            namespaced(namespace, config["joint_state_topic"]),
            self._on_joint_state,
            10,
        )

    def _on_joint_state(self, message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in self.config["joint_names"]):
            with self._lock:
                self._joint_state = [positions[name] for name in self.config["joint_names"]]

    def wait_until(self, predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise TimeoutError(f"timed out after {timeout:.1f} s waiting for {description}")

    def wait_for_future(self, future, timeout, description):
        self.wait_until(future.done, timeout, description)
        return future.result()

    def current_joint_positions(self, timeout=10.0):
        def available():
            with self._lock:
                return self._joint_state is not None

        self.wait_until(available, timeout, "joint states")
        with self._lock:
            return list(self._joint_state)

    def call(self, service_type, name, request, timeout=10.0):
        key = (service_type, name)
        client = self._service_clients.get(key)
        if client is None:
            client = self.create_client(service_type, name)
            self._service_clients[key] = client
        if not client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"service {name} is not available")
        return self.wait_for_future(client.call_async(request), timeout, name)

    def list_controllers(self):
        response = self.call(
            ListControllers,
            self.manager_ns + "/list_controllers",
            ListControllers.Request(),
            15.0,
        )
        return {controller.name: controller for controller in response.controller}

    def ensure_active(self, log=print):
        controllers = self.list_controllers()
        if self.controller not in controllers:
            log(f"controller {self.controller} is not loaded - loading it")
            response = self.call(
                LoadController,
                self.manager_ns + "/load_controller",
                LoadController.Request(name=self.controller),
                30.0,
            )
            if not response.ok:
                raise RuntimeError(f"could not load {self.controller}")
            controllers = self.list_controllers()
        state = controllers[self.controller].state
        if state == "unconfigured":
            log(f"configuring {self.controller}")
            response = self.call(
                ConfigureController,
                self.manager_ns + "/configure_controller",
                ConfigureController.Request(name=self.controller),
                60.0,
            )
            if not response.ok:
                raise RuntimeError(f"could not configure {self.controller}")
            controllers = self.list_controllers()
            state = controllers[self.controller].state
        if state != "active":
            joints = set(self.config["joint_names"])
            to_stop = []
            for name, info in controllers.items():
                if name == self.controller or info.state != "active" or "broadcaster" in info.type.lower():
                    continue
                claimed = {interface.split("/")[0] for interface in info.claimed_interfaces}
                if claimed & joints:
                    to_stop.append(name)
            request = SwitchController.Request()
            request.activate_controllers = [self.controller]
            request.deactivate_controllers = to_stop
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout = Duration(seconds=10.0).to_msg()
            response = self.call(
                SwitchController,
                self.manager_ns + "/switch_controller",
                request,
                30.0,
            )
            if not response.ok:
                raise RuntimeError(f"switch_controller refused to activate {self.controller}")
            self.wait_until(
                lambda: self.list_controllers()[self.controller].state == "active",
                10.0,
                f"{self.controller} to become active",
            )
        if not self._action.wait_for_server(timeout_sec=10.0):
            raise TimeoutError(
                f"action {self.controller_ns}/follow_joint_trajectory is not available"
            )
        log(f"controller {self.controller} is active (stock JointTrajectoryController)")

    @staticmethod
    def _point(positions, velocities, seconds, accelerations=None):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        if velocities is not None:
            point.velocities = [float(value) for value in velocities]
        if accelerations is not None:
            point.accelerations = [float(value) for value in accelerations]
        point.time_from_start = Duration(seconds=float(seconds)).to_msg()
        return point

    def _execute(self, points, timeout, description, on_accept=None):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.config["joint_names"])
        goal.trajectory.points = points
        started_ns = self.get_clock().now().nanoseconds
        goal_handle = self.wait_for_future(
            self._action.send_goal_async(goal), 15.0, f"{description} goal acceptance"
        )
        if not goal_handle.accepted:
            raise Rejected(f"JointTrajectoryController rejected the {description} goal")
        with self._lock:
            self._goal_handle = goal_handle
        if on_accept is not None:
            on_accept()
        wrapped = self.wait_for_future(
            goal_handle.get_result_async(), timeout, f"the {description} to finish"
        )
        with self._lock:
            self._goal_handle = None
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or result.error_code != result.SUCCESSFUL:
            detail = result.error_string or f"error code {result.error_code}"
            raise Rejected(f"JointTrajectoryController did not complete {description}: {detail}")
        return {
            "goal_id": bytes(goal_handle.goal_id.uuid).hex(),
            "start_ns": started_ns,
            "end_ns": self.get_clock().now().nanoseconds,
        }

    def goto(self, positions, timeout=None):
        settings = self.config.get("goto", {})
        current = np.asarray(self.current_joint_positions(), dtype=float)
        target = np.asarray(positions, dtype=float)
        step = target - current
        max_step = float(settings.get("max_joint_step", 3.0))
        if np.max(np.abs(step)) > max_step:
            raise Rejected(
                f"goto target is {np.max(np.abs(step)):.3f} rad from the measured pose, "
                f"over the {max_step:.3f} rad guard"
            )
        duration = goto_duration(
            step,
            float(settings.get("max_velocity", 0.5)),
            float(settings.get("max_acceleration", 1.0)),
            float(settings.get("min_duration", 5.0)),
        )
        # Send only the rest-to-rest endpoint.  With spline interpolation and
        # position, velocity, and acceleration specified, the stock JTC builds
        # the quintic from its last commanded state.  Sampling this ramp in the
        # client from measured positions is unsafe on a compliant arm: measured
        # q can differ slightly from libfranka's q_d, turning that offset into a
        # one-cycle velocity/acceleration discontinuity when the goal starts.
        zeros = np.zeros_like(target)
        points = [self._point(target, zeros, duration, accelerations=zeros)]
        result = self._execute(
            points,
            timeout or float(self.config["timing"]["goto_timeout"]),
            "goto",
        )
        result["duration"] = duration
        result["points_sent"] = 1
        return result

    def send_trajectory(self, prepared, send_rate=None, timeout_margin=15.0, on_accept=None):
        send_rate = float(send_rate or prepared.rate)
        stride = max(1, int(round(prepared.rate / send_rate)))
        indices = np.arange(0, len(prepared.t), stride)
        if indices[-1] != len(prepared.t) - 1:
            indices = np.append(indices, len(prepared.t) - 1)
        points = [
            self._point(prepared.q[index], prepared.qd[index], prepared.t[index])
            for index in indices
        ]
        result = self._execute(
            points,
            prepared.duration + timeout_margin,
            "trajectory",
            on_accept=on_accept,
        )
        result["points_sent"] = len(points)
        return result

    def abort(self):
        with self._lock:
            goal_handle = self._goal_handle
        if goal_handle is not None:
            goal_handle.cancel_goal_async()
