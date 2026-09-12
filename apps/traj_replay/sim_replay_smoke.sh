#!/usr/bin/env bash
# Unattended MuJoCo smoke test of the torque-controlled replay stack.
#
#   ./apps/traj_replay/sim_replay_smoke.sh [trajectory_dir] [homing.yaml] [runner flags...]
#
# Launches sim_replay.launch.py headless with ARM_CONTROLLER (default
# cartesian-impedance; joint-impedance also works), waits for the controllers,
# replays the trajectory with --yes, prints the runner's output, and shuts the
# launch down again. Only the launch it started is ever signalled.
#
# It refuses to start while any /controller_manager is on the graph: ROS 2 has
# one name for that service, so a simulator started next to a hardware bringup
# (or a sim that was never stopped) would hand controller requests to the wrong
# one. Stop the other session first.
set -euo pipefail

ARM_CONTROLLER=${ARM_CONTROLLER:-cartesian-impedance}
HEADLESS=${HEADLESS:-true}
TIME_SCALE=${TIME_SCALE:-5}
WORKSPACE=${WORKSPACE:-/root/develop_ws}

TRAJ=${1:-apps/traj_replay/demo_trajs/traj_2_cycle3}
HOME_YAML=${2:-$TRAJ/homing.yaml}
if [ $# -ge 2 ]; then shift 2; elif [ $# -eq 1 ]; then shift 1; fi

cd "$WORKSPACE"
if [ -f install/setup.bash ]; then
  # shellcheck disable=SC1091
  source install/setup.bash
fi

if timeout 5 ros2 service list 2>/dev/null | grep -q '^/controller_manager/list_controllers$'; then
  echo "refusing to start: a /controller_manager is already on the graph (a hardware bringup" >&2
  echo "or a simulator that was not shut down). Stop it, then run this again." >&2
  exit 2
fi

LOG=${LOG:-/tmp/sim_replay_smoke_launch.log}
ros2 launch inspire_franka_trajectory_replay sim_replay.launch.py \
  "arm_controller:=$ARM_CONTROLLER" "headless:=$HEADLESS" > "$LOG" 2>&1 &
LAUNCH=$!
cleanup() {
  # ros2 launch turns SIGINT into an orderly shutdown of everything it started.
  kill -INT "$LAUNCH" 2>/dev/null || true
  wait "$LAUNCH" 2>/dev/null || true
}
trap cleanup EXIT

case "$ARM_CONTROLLER" in
  cartesian-impedance) WAIT_FOR=cartesian_trajectory_replay_controller ;;
  *) WAIT_FOR=trajectory_replay_controller ;;
esac
for _ in $(seq 1 60); do
  if timeout 5 ros2 control list_controllers 2>/dev/null | grep -q "$WAIT_FOR"; then
    break
  fi
  sleep 2
done
echo "=== controllers ==="
timeout 10 ros2 control list_controllers || true

echo "=== replay ($ARM_CONTROLLER, time scale $TIME_SCALE) ==="
set +e
ros2 run inspire_franka_trajectory_replay replay_trajectory "$TRAJ" \
  --home "$HOME_YAML" --arm-controller "$ARM_CONTROLLER" \
  --time-scale "$TIME_SCALE" --yes --timeout-margin 90 "$@"
STATUS=$?
set -e
echo "runner exit code: $STATUS"
echo "launch log: $LOG"
exit "$STATUS"
