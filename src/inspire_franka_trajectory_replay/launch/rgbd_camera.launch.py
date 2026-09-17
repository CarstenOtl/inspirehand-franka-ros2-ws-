"""The D415 at full colour resolution, for ``replay_trajectory --record-rgbd``.

    ros2 launch inspire_franka_trajectory_replay rgbd_camera.launch.py

The description itself lives in
:mod:`inspire_franka_trajectory_replay.launch_files.rgbd_camera`, so that its
profiles and fixed arguments can be imported and tested; this file is the
launch-system entry point.
"""

from inspire_franka_trajectory_replay.launch_files.rgbd_camera import (
    generate_launch_description,
)

__all__ = ["generate_launch_description"]
