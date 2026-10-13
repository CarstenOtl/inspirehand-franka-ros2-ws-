"""The D415 as the policy rollout needs it.

    ros2 launch inspire_franka_trajectory_replay policy_camera.launch.py

The description itself lives in
:mod:`inspire_franka_trajectory_replay.launch_files.policy_camera`, so that its
profiles and fixed arguments can be imported and tested; this file is the
launch-system entry point.
"""

from inspire_franka_trajectory_replay.launch_files.policy_camera import (
    generate_launch_description,
)

__all__ = ["generate_launch_description"]
