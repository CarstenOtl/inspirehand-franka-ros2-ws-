#!/usr/bin/env python3
"""Put the FR3 under the policy-profile Cartesian impedance controller and hold.

This is the bench step between "the law is right in simulation" and "run the
policy on hardware": it activates the controller the policy rollout would use,
verifies which law that controller is actually running, and then holds the arm
where it stands so the behaviour can be watched and measured with no policy in
the loop.

Activation commands no motion. The controller snapshots the measured pose,
orientation and joint configuration on its first update and holds them, so it
becomes an impedance hold about wherever the arm already is. ``--nudge``
optionally asks for one bounded Cartesian step from there, which is the
smallest useful excitation of the law.

Like the other operations scripts this never launches hardware. Bring the cell
up first, with the profile that loads the Cartesian controller:

    ros2 launch inspire_franka_trajectory_replay replay.launch.py \\
      arm_controller:=policy robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0

then home the arm (``apps/operations/home_arm.py``) and run this from a second
sourced shell.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    WORKSPACE_ROOT / "src/inspire_franka_trajectory_replay/config/replay.yaml"
)

# The policy profile's law, as controllers_policy.yaml sets it. These are the
# parameters that make the controller reproduce ForgeUltra's compute_dof_torque
# rather than franka_example_controllers' variant of it: the full-angle rotation
# error, the exact mass-weighted nullspace projection, and no Coriolis term. A
# bench session that silently ran the example law would read as a test of this
# one, so refuse unless the operator asks for that on purpose.
FORGE_LAW_ROTATION_ERROR = "axis_angle"
FORGE_LAW_NULLSPACE_LAMBDA = 0.0
FORGE_LAW_MASS_WEIGHTED_NULLSPACE = True
FORGE_LAW_CORIOLIS_COMPENSATION = False

# Bounded so a typo cannot ask for a large unplanned move. The controller
# clamps again on its own side (goto_max_velocity, max_goto_step).
MAX_NUDGE_M = 0.05
MAX_NUDGE_DEG = 10.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--nudge",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="after activating, command one Cartesian step of this size in metres, "
        "in the base frame, relative to the pose the controller is holding. Each "
        f"component must be <= {MAX_NUDGE_M} m.",
    )
    parser.add_argument(
        "--nudge-yaw",
        type=float,
        default=0.0,
        metavar="DEGREES",
        help=f"yaw component of --nudge about the base z axis (<= {MAX_NUDGE_DEG:.0f} deg).",
    )
    parser.add_argument(
        "--nudge-duration",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="how long the commanded step should take (default 3 s; 0 lets the "
        "controller pick from its velocity limits).",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="hold for this long after activating, then hand the arm back. "
        "0 (the default) holds until Ctrl-C.",
    )
    parser.add_argument(
        "--report-hz",
        type=float,
        default=2.0,
        metavar="HZ",
        help="how often to print the controller's tracking status (default 2 Hz).",
    )
    parser.add_argument(
        "--allow-example-law",
        action="store_true",
        help="proceed even if the controller is not running ForgeUltra's law. "
        "Use only to characterise the example law on purpose.",
    )
    parser.add_argument(
        "--keep-active",
        action="store_true",
        help="leave the Cartesian controller active on exit instead of handing "
        "the arm back to whatever held it before.",
    )
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the arguments and print the plan without connecting to ROS.",
    )
    args = parser.parse_args(argv)

    if args.nudge is not None:
        for axis, value in zip("xyz", args.nudge):
            if not math.isfinite(value) or abs(value) > MAX_NUDGE_M:
                parser.error(
                    f"--nudge {axis} component {value} exceeds the {MAX_NUDGE_M} m bound"
                )
    if not math.isfinite(args.nudge_yaw) or abs(args.nudge_yaw) > MAX_NUDGE_DEG:
        parser.error(f"--nudge-yaw {args.nudge_yaw} exceeds {MAX_NUDGE_DEG:.0f} deg")
    if args.nudge is None and args.nudge_yaw:
        parser.error("--nudge-yaw needs --nudge")
    if args.report_hz <= 0:
        parser.error("--report-hz must be positive")
    if args.hold < 0:
        parser.error("--hold must not be negative")
    return args


def describe_plan(args) -> str:
    steps = [
        "check the Cartesian controller is loaded and read its parameters",
        "verify the law is ForgeUltra's"
        + (" (overridden by --allow-example-law)" if args.allow_example_law else ""),
        "deactivate whatever holds the arm command interfaces and activate it"
        " (it holds its activation pose; this commands no motion)",
    ]
    if args.nudge is not None:
        steps.append(
            "command one step of dx=%.3f dy=%.3f dz=%.3f m, yaw %+.1f deg over %.1f s"
            % (*args.nudge, args.nudge_yaw, args.nudge_duration)
        )
    steps.append(
        "hold %s, printing tracking status at %.1f Hz"
        % ("until Ctrl-C" if args.hold == 0 else "for %.1f s" % args.hold, args.report_hz)
    )
    steps.append(
        "leave the controller active (--keep-active)"
        if args.keep_active
        else "hand the arm back to the controller that held it before"
    )
    return "plan:\n" + "\n".join(
        "  %d. %s" % (i, step) for i, step in enumerate(steps, start=1)
    )


def check_law(parameters, *, allow_example_law, log=print) -> None:
    """Report the active law; refuse the example one unless asked for."""

    log("controller law:")
    log(
        "  translational %s N/m, rotational %s Nm/rad, nullspace %s"
        % (
            parameters.get("translational_stiffness"),
            parameters.get("rotational_stiffness"),
            parameters.get("nullspace_stiffness"),
        )
    )
    log(
        "  rotation_error=%r, nullspace_damping_lambda=%r, mass_weighted_nullspace=%r, "
        "coriolis_compensation=%r, model_source=%r"
        % (
            parameters.get("rotation_error"),
            parameters.get("nullspace_damping_lambda"),
            parameters.get("mass_weighted_nullspace"),
            parameters.get("coriolis_compensation"),
            parameters.get("model_source"),
        )
    )
    wrong = []
    if parameters.get("mass_weighted_nullspace") is not FORGE_LAW_MASS_WEIGHTED_NULLSPACE:
        wrong.append(
            "mass_weighted_nullspace=%r (ForgeUltra weights the nullspace joint PD by the "
            "arm mass matrix)" % (parameters.get("mass_weighted_nullspace"),)
        )
    if parameters.get("coriolis_compensation") is not FORGE_LAW_CORIOLIS_COMPENSATION:
        wrong.append(
            "coriolis_compensation=%r (compute_dof_torque has no Coriolis term)"
            % (parameters.get("coriolis_compensation"),)
        )
    if str(parameters.get("rotation_error")) != FORGE_LAW_ROTATION_ERROR:
        wrong.append(
            "rotation_error=%r (ForgeUltra uses %r; the quaternion-vector form "
            "halves the effective rotational stiffness)"
            % (parameters.get("rotation_error"), FORGE_LAW_ROTATION_ERROR)
        )
    lam = parameters.get("nullspace_damping_lambda")
    if lam is None or abs(float(lam) - FORGE_LAW_NULLSPACE_LAMBDA) > 1e-9:
        wrong.append(
            "nullspace_damping_lambda=%r (ForgeUltra projects exactly; the "
            "example's 0.2 leaks the joint spring onto the tool)" % (lam,)
        )
    if not wrong:
        log("  -> ForgeUltra's law")
        return
    message = "controller is not running ForgeUltra's law: " + "; ".join(wrong)
    if not allow_example_law:
        raise SystemExit(
            message
            + "\nBring the cell up with arm_controller:=policy, or pass "
            "--allow-example-law to characterise this law on purpose."
        )
    log("  -> NOT ForgeUltra's law; continuing because --allow-example-law")


def arm_holding_controllers(controllers, joint_names):
    """Active, non-broadcaster controllers that claim an arm command interface.

    The same rule ``ReplayClient.ensure_active`` uses to decide what to stop,
    applied before the switch so the arm can be handed back afterwards.
    """

    joints = set(joint_names)
    holding = []
    for name, info in controllers.items():
        if info.state != "active" or "broadcaster" in info.type.lower():
            continue
        claimed = {interface.split("/")[0] for interface in info.claimed_interfaces}
        if claimed & joints:
            holding.append(name)
    return holding


def parse_reference_pose(status):
    """The pose the controller is holding, as (position, quaternion xyzw).

    ``reference_pose`` is the controller's own filtered reference in the base
    frame, already at the tool the profile configures, so a step relative to it
    needs no separate tool bookkeeping.
    """

    raw = (status or {}).get("reference_pose")
    if not raw:
        raise RuntimeError("controller status carries no reference_pose")
    values = [float(v) for v in str(raw).split()]
    if len(values) != 7:
        raise RuntimeError("reference_pose should hold 7 values, got %r" % (raw,))
    return values[0:3], values[3:7]


def yaw_about_base_z(quat_xyzw, yaw_rad):
    """Pre-multiply an xyzw quaternion by a rotation about the base z axis."""

    x, y, z, w = quat_xyzw
    half = 0.5 * yaw_rad
    cz, sz = math.cos(half), math.sin(half)
    return (
        cz * x - sz * y,
        cz * y + sz * x,
        cz * z + sz * w,
        cz * w - sz * z,
    )


def confirm(parameters, args) -> bool:
    print()
    print("About to put the FR3 under the Cartesian impedance controller.")
    print(
        "It becomes compliant about its activation pose at %s N/m. With the exact "
        "nullspace projection the elbow is genuinely free in the redundant "
        "direction, so it may settle visibly further than it does under the "
        "example law." % parameters.get("translational_stiffness")
    )
    if args.nudge is not None:
        print(
            "It will then be commanded %.3f %.3f %.3f m, yaw %+.1f deg. Clear the "
            "swept path." % (*args.nudge, args.nudge_yaw)
        )
    print("Keep the enabling device in reach.")
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def report(status, log=print) -> None:
    if status is None:
        log("  (no controller status yet)")
        return
    log(
        "  phase=%s pos_err=%s m rot_err=%s rad joint_margin=%s rad fault=%s"
        % (
            status.get("phase_name"),
            status.get("position_error_m"),
            status.get("orientation_error_rad"),
            status.get("joint_limit_margin_rad"),
            status.get("tracking_fault"),
        )
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    print(describe_plan(args))
    if args.dry_run:
        print("\n--dry-run: nothing was contacted.")
        return 0

    import threading

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from franka_trajectory_replay.cartesian_replay_client import CartesianReplayClient
    from franka_trajectory_replay.replay_client import Rejected, ReplayClient
    from franka_trajectory_replay.runconfig import load_config

    config = load_config(args.config)
    rclpy.init(args=None)
    client = CartesianReplayClient(config, node_name="activate_policy_controller")
    # The client blocks on service futures and on status messages, so something
    # has to spin it. Several of its waits overlap, hence a multi-threaded
    # executor, as the rollout uses for the same client.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(client)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    activated = False
    previously_holding = []
    try:
        client.require_loaded()
        parameters = client.controller_parameters()
        check_law(parameters, allow_example_law=args.allow_example_law)

        previously_holding = arm_holding_controllers(
            client.list_controllers(), config["joint_names"]
        )
        if previously_holding:
            print("currently holding the arm: %s" % ", ".join(previously_holding))

        if not args.yes and not confirm(parameters, args):
            print("aborted before activating; nothing was switched.")
            return 1

        client.ensure_active()
        activated = True
        print("controller active; holding its activation pose.")

        if args.nudge is not None:
            position, quaternion = parse_reference_pose(client.wait_for_status())
            target = [position[i] + args.nudge[i] for i in range(3)]
            if args.nudge_yaw:
                quaternion = yaw_about_base_z(
                    quaternion, math.radians(args.nudge_yaw)
                )
            print(
                "commanding %.3f %.3f %.3f m, yaw %+.1f deg over %.1f s"
                % (*args.nudge, args.nudge_yaw, args.nudge_duration)
            )
            client.goto(target, quaternion, duration=args.nudge_duration)
            print("step complete; peak errors %s" % (client.peak_errors(),))

        period = 1.0 / args.report_hz
        started = time.monotonic()
        print("holding (Ctrl-C to hand the arm back):")
        while args.hold == 0 or time.monotonic() - started < args.hold:
            report(client.status())
            time.sleep(period)
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Rejected as exc:
        print("controller refused the request: %s" % exc, file=sys.stderr)
        return 1
    finally:
        if activated and not args.keep_active and previously_holding:
            # Best effort, as home_arm.py does: put back whichever controller held
            # the arm before, so the cell is left as it was found. One atomic
            # switch_controller, so the arm is never left unclaimed.
            handback = previously_holding[0]
            try:
                ReplayClient.ensure_active(client, controller=handback)
                print("handed the arm back to %s" % handback)
            except Exception as exc:  # noqa: BLE001 - never mask the original failure
                print(
                    "could not hand the arm back to %s: %s\n"
                    "The Cartesian controller is still active. Restart the launch, "
                    "or switch manually with `ros2 control switch_controllers`."
                    % (handback, exc),
                    file=sys.stderr,
                )
        try:
            executor.shutdown(timeout_sec=2.0)
            spin_thread.join(timeout=5.0)
            client.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
