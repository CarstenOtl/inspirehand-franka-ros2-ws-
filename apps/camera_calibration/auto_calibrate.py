#!/usr/bin/env python3
"""User-facing command line entry point for automated camera calibration.

The passive recorder in ``calibrate.py`` never commands the robot: a human
guides the hand and the recorder pairs each image with a robot pose looked up at
that image's timestamp. That pairing is what went wrong on the first real run -
the camera and the robot do not share a clock, and while the arm moves an
unknown delay of tens of milliseconds is tens of millimetres of error that
biases the solve instead of averaging out.

This entry point drives the arm itself and stops at every pose, so no delay can
affect the static result, and then measures the delay deliberately in a final
moving pass. It only forwards arguments; everything happens in
``camera_calibration.auto_calibration``.
"""

from __future__ import annotations

import os
import shlex
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = ["ros2", "run", "camera_calibration", "auto_calibrate", *arguments]
    if not any(argument in ("--help", "-h") for argument in arguments):
        print(
            "Automated calibration. The arm will move under the replay controller's\n"
            "guarded goto; Ctrl-C aborts it and holds position. A realsense2_camera\n"
            "node and the replay controller must already be running.\n",
            flush=True,
        )
    print(f"Starting: {shlex.join(command)}", flush=True)
    try:
        # Replace this wrapper so Ctrl-C and exit codes belong directly to the run.
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
