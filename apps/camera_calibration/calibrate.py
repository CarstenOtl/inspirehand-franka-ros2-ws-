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
            "Calibrate the fixed RealSense pose from an AprilTag on the back "
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
        "--manual",
        action="store_true",
        help="capture only when /camera_calibration/capture is called",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="do not launch a D415; use an already-running camera node",
    )
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be greater than zero")
    if args.minimum_samples < 4:
        parser.error("--minimum-samples must be at least 4")


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
        f"auto_capture:={'false' if args.manual else 'true'}",
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
    print(
        "Starting passive calibration recording. No robot motion commands are sent.\n"
        "Guide the FR3 by hand while keeping the tag visible. When finished, run:\n"
        "  ros2 service call /camera_calibration/solve std_srvs/srv/Trigger '{}'",
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
