#!/usr/bin/env bash
# Container entrypoint for inspire_franka.
#
# Its one job is to trim src/franka_ros2 down to the packages this image can
# actually build. `vcs import` has no way to take a subset of a repo, so the
# whole of franka_ros2 lands in src/ and the rest is marked COLCON_IGNORE here.
#
# COLCON_IGNORE rather than `colcon build --packages-ignore` in the `rg2` helper,
# so that a plain `colcon build` typed by hand behaves the same way. The files
# land inside a vcs checkout and show up as untracked there; that is expected,
# not a stray edit.
#
# Set FRANKA_ROS2_BUILD_ALL=1 in docker-compose.yml to opt back in. You will
# then have to install those dependencies yourself.
set -e

WS_SRC=/root/develop_ws/src
FRANKA_SRC="$WS_SRC/franka_ros2"

# Package directories to skip, relative to src/franka_ros2, and why:
#
#   franka_gazebo                       needs gz_ros2_control and a Gazebo install
#   franka_fr3_moveit_config            needs the full MoveIt stack (we install
#                                       only moveit_core/moveit_msgs, which
#                                       franka_example_controllers links against)
#   franka_mobile_fr3_duo_moveit_config likewise
#   franka_mobile                       mobile base, not this robot
#   mobile_fr3_duo_trajectory_controller likewise
#   franka_description_extensions       RealSense / SICK / ZED / Robotiq hardware
#   franka_ros2                         meta-package; depends on all of the above
#   realtime_tools                      a vendored copy of an upstream package.
#                                       Jazzy's apt realtime_tools is new enough
#                                       for franka_ros2 v3.5.x, and building the
#                                       vendored one would shadow it for every
#                                       other package in the workspace.
FRANKA_SKIP=(
  franka_gazebo
  franka_fr3_moveit_config
  franka_mobile_fr3_duo_moveit_config
  franka_mobile
  mobile_fr3_duo_trajectory_controller
  franka_description_extensions
  franka_ros2
  realtime_tools
)

if [ -d "$FRANKA_SRC" ]; then
  for pkg in "${FRANKA_SKIP[@]}"; do
    [ -d "$FRANKA_SRC/$pkg" ] || continue
    if [ "${FRANKA_ROS2_BUILD_ALL:-0}" = "1" ]; then
      rm -f "$FRANKA_SRC/$pkg/COLCON_IGNORE"
    else
      : > "$FRANKA_SRC/$pkg/COLCON_IGNORE"
    fi
  done
else
  echo "entrypoint: note - src/franka_ros2 is missing, so there is no real-arm support." >&2
  echo "            Run: vcs import src < workspace.repos" >&2
fi

if [ ! -d "$WS_SRC/franka_description" ]; then
  echo "entrypoint: note - src/franka_description is missing; the descriptions will not build." >&2
  echo "            Run: vcs import src < workspace.repos" >&2
fi

exec "$@"
