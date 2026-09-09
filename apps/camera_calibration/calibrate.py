#!/usr/bin/env python3
"""User-facing command line entry point for camera calibration."""

from __future__ import annotations

import argparse
import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


TAG_FAMILIES = ("tag16h5", "tag25h9", "tag36h10", "tag36h11")
REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SERVICE_NAME = "/camera_calibration/capture"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description=(
            "Calibrate the fixed RealSense D415 RGB pose from an AprilTag on the back "
            "of the Inspire Hand. The result is written to the ROS log."
        ),
    )
    parser.add_argument("--tag-family", choices=TAG_FAMILIES, default="tag36h11")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=0.040,
        help="measured outer edge of the tag's black square (default: 0.040)",
    )
    parser.add_argument("--world-frame", default="fr3_link0")
    parser.add_argument(
        "--hand-frame",
        default="fr3_link8",
        help=(
            "moving TF frame rigidly carrying the hand/tag; fr3_link8 is used "
            "because the real hand description is published separately"
        ),
    )
    parser.add_argument("--camera-mount-frame", default="camera_link")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "logs"),
        help="parent directory for timestamped sample/result folders",
    )
    parser.add_argument(
        "--camera-optical-frame",
        default="",
        help="leave empty to obtain the optical frame from CameraInfo",
    )
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--camera-info-topic", default="/camera/camera/color/camera_info"
    )
    parser.add_argument("--minimum-samples", type=int, default=12)
    parser.add_argument(
        "--waypoint-duration-s",
        type=float,
        default=5.0,
        help="seconds for each automatic waypoint move (default: 5.0)",
    )
    parser.add_argument(
        "--trajectory-action",
        default="/fr3_arm_controller/follow_joint_trajectory",
        help="FollowJointTrajectory action used by --auto",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "hand-guided mode: stop at each pose and press Enter to capture one "
            "RGB/FK pair (this is the default)"
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "move the FR3 through 12 visibility-checked waypoints and capture one "
            "sample at each; physical motion starts only after Enter is pressed"
        ),
    )
    parser.add_argument(
        "--triggered-capture",
        action="store_true",
        help="collect a sample only after /camera_calibration/capture is called",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="do not launch the D415; use an already-running camera node",
    )
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be greater than zero")
    if args.minimum_samples < 4:
        parser.error("--minimum-samples must be at least 4")
    if args.waypoint_duration_s <= 0.0:
        parser.error("--waypoint-duration-s must be greater than zero")
    selected_modes = int(args.manual) + int(args.auto) + int(args.triggered_capture)
    if selected_modes > 1:
        parser.error("--manual, --auto, and --triggered-capture are mutually exclusive")
    if args.auto and args.minimum_samples != 12:
        parser.error("--auto uses exactly 12 waypoints; --minimum-samples must be 12")


def _launch_command(args: argparse.Namespace) -> list[str]:
    # The user-facing manual workflow arms one capture at a time by calling the
    # existing Trigger service after Enter is pressed.  Keep the direct
    # --triggered-capture mode for users who want to call that service from a
    # different terminal or program.
    capture_mode = "auto" if args.auto else "triggered"
    command = [
        "ros2",
        "launch",
        "camera_calibration",
        "calibrate.launch.py",
        f"start_camera:={'false' if args.no_camera else 'true'}",
        f"tag_family:={args.tag_family}",
        f"tag_id:={args.tag_id}",
        f"tag_size_m:={args.tag_size_m}",
        f"world_frame:={args.world_frame}",
        f"hand_frame:={args.hand_frame}",
        f"camera_mount_frame:={args.camera_mount_frame}",
        f"output_root:={args.output_root}",
        f"image_topic:={args.image_topic}",
        f"camera_info_topic:={args.camera_info_topic}",
        f"capture_mode:={capture_mode}",
        f"auto_motion_authorized:={'true' if args.auto else 'false'}",
        f"auto_waypoint_duration_s:={args.waypoint_duration_s}",
        f"trajectory_action:={args.trajectory_action}",
        f"minimum_samples:={args.minimum_samples}",
    ]
    if args.camera_optical_frame:
        command.append(f"camera_optical_frame:={args.camera_optical_frame}")
    return command


class _PersistentCaptureClient:
    """One ROS client reused for every manual capture request."""

    def __init__(self) -> None:
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.signals import SignalHandlerOptions
            from std_srvs.srv import Trigger
        except ImportError as error:
            raise RuntimeError(
                "Python cannot import rclpy/std_srvs; source the ROS 2 installation "
                "and this workspace before starting calibration"
            ) from error

        self._rclpy = rclpy
        self._trigger_type = Trigger
        self._context = Context()
        rclpy.init(
            args=[],
            context=self._context,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self._node = rclpy.create_node(
            "camera_calibration_manual_control", context=self._context
        )
        # rclpy's implicit global executor belongs to the default context.  This
        # client intentionally has its own context (and no signal handlers), so
        # it must also have an executor bound to that same context.
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._client = self._node.create_client(Trigger, CAPTURE_SERVICE_NAME)

    def wait_until_ready(self, launch_process: subprocess.Popen) -> bool:
        """Wait once for node startup, stopping if the launch exits."""
        print(
            f"Waiting for {CAPTURE_SERVICE_NAME} to become ready...",
            flush=True,
        )
        while launch_process.poll() is None:
            if self._client.wait_for_service(timeout_sec=0.25):
                print("Manual capture service is ready.", flush=True)
                return True
        return False

    def request_capture(self) -> tuple[bool, str]:
        future = self._client.call_async(self._trigger_type.Request())
        self._executor.spin_until_future_complete(future, timeout_sec=5.0)
        if not future.done():
            future.cancel()
            return False, "capture service did not respond within 5 seconds"
        try:
            response = future.result()
        except Exception as error:
            return False, f"capture service failed: {error}"
        return bool(response.success), str(response.message)

    def close(self) -> None:
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        self._executor.shutdown()
        if self._context.ok():
            self._context.shutdown()


def _stop_launch(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=3.0)


def _run_enter_capture(command: Sequence[str], client_factory=None) -> int:
    """Run calibration and arm exactly one sample for each Enter press."""
    try:
        # This wrapper owns the terminal: ros2 launch must not compete with
        # input() for keystrokes or receive terminal-generated signals directly.
        # We forward shutdown explicitly from _stop_launch instead.
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        print(
            "error: ros2 was not found; source the ROS 2 installation and workspace",
            file=sys.stderr,
        )
        return 127

    capture_client = None
    try:
        factory = client_factory or _PersistentCaptureClient
        capture_client = factory()
        if not capture_client.wait_until_ready(process):
            print(
                "Calibration launch exited before its capture service became ready.",
                file=sys.stderr,
                flush=True,
            )
            return process.returncode if process.returncode is not None else 1

        requested = 0
        print(
            "\nManual capture control is active. At each pose, hold the robot still, "
            "wait for a sharp full-tag view, and press Enter. Type q then Enter to stop.",
            flush=True,
        )
        while process.poll() is None:
            try:
                response = input(
                    f"\nPose {requested + 1}: press Enter to capture (q to stop): "
                )
            except EOFError:
                print("\nInput closed; stopping calibration.", flush=True)
                break
            if response.strip().lower() in ("q", "quit"):
                break
            if response.strip():
                print("Press Enter with no text to capture, or q to stop.", flush=True)
                continue
            if process.poll() is not None:
                break
            try:
                success, message = capture_client.request_capture()
            except Exception as error:
                print(
                    f"Capture request raised an unexpected error: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not success:
                print(
                    f"Capture request failed: {message}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            requested += 1
            print(
                f"{message} Wait for the 'Accepted valid sample' log before moving.",
                flush=True,
            )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("\nStopping calibration...", flush=True)
    finally:
        if capture_client is not None:
            capture_client.close()
        _stop_launch(process)
    return process.returncode if process.returncode is not None else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    command = _launch_command(args)
    interactive_manual = not args.auto and not args.triggered_capture
    if args.auto:
        print(
            "AUTO MODE WILL MOVE THE PHYSICAL FR3 through 12 programmed joint poses.\n"
            "Use the fr3_arm_controller trajectory controller, keep the workcell clear, "
            "stay at the emergency stop, and verify the camera faces the tag.\n"
            "Press Enter to authorize motion, or Ctrl-C to cancel.",
            flush=True,
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("Auto calibration cancelled; no motion command was sent.", flush=True)
            return 130
        mode_message = (
            "Starting automatic calibration. The FR3 will move slowly to 12 waypoints "
            "and capture one settled RGB/FK sample at each."
        )
        motion_message = "Physical robot motion is enabled."
    elif args.triggered_capture:
        mode_message = (
            "Starting service-triggered calibration. Move the robot by hand, then call "
            "/camera_calibration/capture for each pose."
        )
        motion_message = "No robot motion commands are sent."
    else:
        mode_message = (
            "Starting manual hand-guided calibration. Each synchronized OpenCV tag "
            "pose and robot FK sample is armed by pressing Enter."
        )
        motion_message = "No robot motion commands are sent."
    print(
        f"{mode_message}\n{motion_message} Keep the full tag visible; "
        f"every accepted sample is reported. Calibration runs automatically after "
        f"{args.minimum_samples} valid samples; no separate solve command is needed.",
        flush=True,
    )
    print(f"Starting: {shlex.join(command)}", flush=True)
    if interactive_manual:
        return _run_enter_capture(command)
    try:
        # Replace this wrapper so Ctrl-C and exit codes belong directly to ros2 launch.
        os.execvp(command[0], command)
    except FileNotFoundError:
        print(
            "error: ros2 was not found; source the ROS 2 installation and workspace",
            file=sys.stderr,
        )
        return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
