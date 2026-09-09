#!/usr/bin/env python3
"""Move an already-running FR3 to its standard valid home configuration."""

from __future__ import annotations

import argparse
import sys
import time

from operation_common import (
    ARM_HOME,
    arm_joint_names,
    conflicting_controllers,
    maximum_error,
    namespaced,
    positions_by_name,
    validate_arm_target,
)

try:
    import rclpy
    from controller_manager_msgs.srv import (
        ConfigureController,
        ListControllers,
        LoadController,
        SwitchController,
        UnloadController,
    )
    from rcl_interfaces.srv import GetParameters, SetParameters
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState
except ModuleNotFoundError as exc:  # lets --dry-run work outside the ROS container
    ROS_IMPORT_ERROR = exc
    rclpy = None
    Node = object


CONTROLLER = "move_to_start_example_controller"


class ArmHomeNode(Node):
    def __init__(self, namespace: str, robot_type: str, arm_prefix: str) -> None:
        super().__init__("home_fr3")
        self.namespace = namespace
        self.robot_type = robot_type
        self.arm_prefix = arm_prefix
        self.manager = namespaced(namespace, "controller_manager")
        self.controller_node = namespaced(namespace, CONTROLLER)
        self.joint_names = arm_joint_names(robot_type, arm_prefix)
        self.positions = None
        self.motion_may_be_active = False
        self.create_subscription(
            JointState,
            namespaced(namespace, "joint_states"),
            self._on_joint_state,
            10,
        )
        self._clients = {}

    def _on_joint_state(self, message: JointState) -> None:
        self.positions = positions_by_name(message.name, message.position)

    def call(self, service_type, path: str, request, timeout: float = 30.0):
        key = (service_type, path)
        client = self._clients.get(key)
        if client is None:
            client = self.create_client(service_type, path)
            self._clients[key] = client
        if not client.wait_for_service(timeout_sec=min(timeout, 10.0)):
            raise TimeoutError(f"service {path} is unavailable")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            future.cancel()
            raise TimeoutError(f"service {path} timed out")
        if future.exception() is not None:
            raise RuntimeError(f"service {path} failed: {future.exception()}")
        return future.result()

    def list_controllers(self):
        response = self.call(
            ListControllers,
            self.manager + "/list_controllers",
            ListControllers.Request(),
        )
        return response.controller

    def switch(self, activate=(), deactivate=(), strict=True):
        request = SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = request.STRICT if strict else request.BEST_EFFORT
        request.activate_asap = True
        request.timeout = Duration(seconds=10.0).to_msg()
        response = self.call(
            SwitchController, self.manager + "/switch_controller", request, timeout=30.0
        )
        if not response.ok:
            raise RuntimeError("controller manager refused the controller switch")

    def set_controller_parameters(self, target) -> None:
        parameters = [
            Parameter("process_finished", value=False),
            Parameter("robot_type", value=self.robot_type),
            Parameter("arm_prefix", value=self.arm_prefix),
            Parameter("start_joint_configuration", value=list(target)),
        ]
        request = SetParameters.Request(
            parameters=[parameter.to_parameter_msg() for parameter in parameters]
        )
        response = self.call(
            SetParameters, self.controller_node + "/set_parameters", request, timeout=10.0
        )
        rejected = [
            result.reason or "unknown reason"
            for result in response.results
            if not result.successful
        ]
        if rejected:
            raise RuntimeError("controller rejected its parameters: " + "; ".join(rejected))

    def process_finished(self) -> bool:
        request = GetParameters.Request(names=["process_finished"])
        response = self.call(
            GetParameters, self.controller_node + "/get_parameters", request, timeout=2.0
        )
        return bool(response.values and response.values[0].bool_value)

    def prepare_and_activate(self, target) -> list[str]:
        controllers = {controller.name: controller for controller in self.list_controllers()}
        if CONTROLLER in controllers and controllers[CONTROLLER].state == "active":
            self.switch(deactivate=[CONTROLLER])
        if CONTROLLER in controllers:
            response = self.call(
                UnloadController,
                self.manager + "/unload_controller",
                UnloadController.Request(name=CONTROLLER),
            )
            if not response.ok:
                raise RuntimeError(f"could not unload {CONTROLLER} before reconfiguration")

        print(f"Loading {CONTROLLER}")
        response = self.call(
            LoadController,
            self.manager + "/load_controller",
            LoadController.Request(name=CONTROLLER),
        )
        if not response.ok:
            raise RuntimeError(
                f"could not load {CONTROLLER}; the normal inspire_franka_bringup "
                "controller configuration is required"
            )

        self.set_controller_parameters(target)
        response = self.call(
            ConfigureController,
            self.manager + "/configure_controller",
            ConfigureController.Request(name=CONTROLLER),
            timeout=60.0,
        )
        if not response.ok:
            raise RuntimeError(f"could not configure {CONTROLLER}; check the controller log")

        active = self.list_controllers()
        to_stop = conflicting_controllers(active, self.joint_names, excluded=[CONTROLLER])
        if to_stop:
            print("Deactivating arm controller(s): " + ", ".join(to_stop))
        # Set this before the service call: Ctrl-C can arrive after the manager
        # has switched controllers but before the response reaches this node.
        self.motion_may_be_active = True
        self.switch(activate=[CONTROLLER], deactivate=to_stop)
        return to_stop

    def stop_motion(self) -> None:
        try:
            controllers = {controller.name: controller for controller in self.list_controllers()}
            if CONTROLLER in controllers and controllers[CONTROLLER].state == "active":
                self.switch(deactivate=[CONTROLLER], strict=False)
                self.motion_may_be_active = False
                print(f"Emergency stop: deactivated {CONTROLLER}", file=sys.stderr)
        except Exception as exc:  # best-effort cleanup while handling an earlier failure
            print(f"WARNING: could not deactivate {CONTROLLER}: {exc}", file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--namespace", default="", help="FR3 ROS namespace")
    result.add_argument("--robot-type", default="fr3")
    result.add_argument("--arm-prefix", default="")
    result.add_argument(
        "--target",
        type=float,
        nargs=7,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        default=ARM_HOME,
        help="optional seven-joint target in radians (default: Franka ready pose)",
    )
    result.add_argument("--timeout", type=float, default=45.0)
    result.add_argument("--tolerance", type=float, default=0.05)
    result.add_argument("--yes", "-y", action="store_true", help="skip the motion prompt")
    result.add_argument(
        "--dry-run", action="store_true", help="validate and print the target only"
    )
    return result


def main(argv=None) -> int:
    raw = sys.argv if argv is None else [sys.argv[0], *argv]
    command_parser = parser()
    application_args = raw[1:]
    ros_args = [raw[0]]
    if "--ros-args" in application_args:
        ros_index = application_args.index("--ros-args")
        ros_args += application_args[ros_index:]
        application_args = application_args[:ros_index]
    args = command_parser.parse_args(application_args)
    if args.timeout <= 0:
        command_parser.error("--timeout must be positive")
    if args.tolerance < 0:
        command_parser.error("--tolerance must not be negative")
    try:
        target = validate_arm_target(args.target)
    except ValueError as exc:
        command_parser.error(str(exc))

    print("FR3 home target [rad]: " + " ".join(f"{value:+.6f}" for value in target))
    print("Note: literal all-zero is invalid for FR3 joints 4 and 6.")
    if args.dry_run:
        print("dry run: no ROS command sent")
        return 0
    if rclpy is None:
        print(f"ERROR: ROS 2 Python modules are unavailable: {ROS_IMPORT_ERROR}", file=sys.stderr)
        return 1
    if not args.yes:
        try:
            input("Clear the arm's path, then press Enter to move (Ctrl-C aborts): ")
        except (EOFError, KeyboardInterrupt):
            print("Aborted before motion", file=sys.stderr)
            return 130

    rclpy.init(args=ros_args)
    node = ArmHomeNode(args.namespace, args.robot_type, args.arm_prefix)
    try:
        node.prepare_and_activate(target)
        print(f"Moving with {CONTROLLER}; Ctrl-C deactivates it")
        deadline = time.monotonic() + args.timeout
        last_report = 0.0
        last_finished_poll = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            error = (
                maximum_error(node.positions, node.joint_names, target)
                if node.positions is not None
                else None
            )
            now = time.monotonic()
            if error is not None and now - last_report >= 1.0:
                print(f"  maximum joint error: {error:.3f} rad")
                last_report = now
            finished = False
            if now - last_finished_poll >= 0.2:
                finished = node.process_finished()
                last_finished_poll = now
            if finished:
                if error is None:
                    raise RuntimeError(
                        f"controller finished but no complete joint state arrived on "
                        f"{namespaced(args.namespace, 'joint_states')}"
                    )
                if error > args.tolerance:
                    raise RuntimeError(
                        f"controller finished {error:.3f} rad from target, over the "
                        f"{args.tolerance:g} rad tolerance"
                    )
                print(f"FR3 home complete (maximum joint error {error:.3f} rad)")
                print(f"{CONTROLLER} remains active at zero commanded torque")
                return 0
        raise TimeoutError(f"arm did not reach home within {args.timeout:g} s")
    except KeyboardInterrupt:
        if node.motion_may_be_active:
            node.stop_motion()
        print("Interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, TimeoutError) as exc:
        if node.motion_may_be_active:
            node.stop_motion()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
