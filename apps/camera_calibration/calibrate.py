#!/usr/bin/env python3
"""User-facing command line entry point for camera calibration."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from typing import Optional, Sequence


TAG_FAMILIES = ("tag16h5", "tag25h9", "tag36h10", "tag36h11")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description=(
            "Calibrate the fixed RealSense D435 RGB pose from an AprilTag on the back "
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
    parser.add_argument("--world-frame", default="world")
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
            "hand-guided mode: you move the robot and every valid, sufficiently "
            "different RGB/FK pair is collected (this is the default)"
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
        help="do not launch the D435; use an already-running camera node",
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
        f"image_topic:={args.image_topic}",
        f"camera_info_topic:={args.camera_info_topic}",
        f"capture_mode:={'auto' if args.auto else 'triggered' if args.triggered_capture else 'manual'}",
        f"auto_motion_authorized:={'true' if args.auto else 'false'}",
        f"auto_waypoint_duration_s:={args.waypoint_duration_s}",
        f"trajectory_action:={args.trajectory_action}",
        f"minimum_samples:={args.minimum_samples}",
    ]
    if args.camera_optical_frame:
        command.append(f"camera_optical_frame:={args.camera_optical_frame}")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    command = _launch_command(args)
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
            "Starting manual hand-guided calibration. The app automatically accepts "
            "synchronized OpenCV tag poses and robot FK."
        )
        motion_message = "No robot motion commands are sent."
    print(
        f"{mode_message}\n{motion_message} Keep the full tag visible; "
        f"every accepted sample is reported. Calibration runs automatically after "
        f"{args.minimum_samples} valid samples; no separate solve command is needed.",
        flush=True,
    )
    print(f"Starting: {shlex.join(command)}", flush=True)
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
