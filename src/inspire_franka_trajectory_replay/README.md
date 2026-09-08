# inspire_franka_trajectory_replay

Coordinated hardware replay for the FR3 and Inspire RH56. The arm uses ROS 2's
stock `joint_trajectory_controller/JointTrajectoryController` over the seven
position command interfaces. `franka_hardware` consequently starts libfranka's
joint-position motion generator with `ControllerMode::kJointImpedance`: the
impedance loop and gains live on the robot, and this package implements no
torque or impedance controller. The hand stays on its independent 50 Hz RS485
driver and receives synchronized position commands on `/inspire_hand/command`.

The application still prepares and validates the recorded waypoints before it
sends a standard `FollowJointTrajectory` goal. That preparation is where the
FR3 position, velocity, acceleration, and jerk margins are enforced.

## Hardware checkpoint and RT-machine TODO

The stock position `JointTrajectoryController` path is **validated in MuJoCo
but not yet validated on the FR3**. On the current workstation it accepts the
home action and begins moving, then the robot stops on
`joint_motion_generator_acceleration_discontinuity`. The latest attempt reached
about 0.4 s before the reflex. Starting each goal from the previous desired
command removed the accompanying velocity discontinuity, and enabling
libfranka's joint-position rate limiter in a discarded experiment did not
eliminate the remaining one. Both Franka submodules remain unchanged from
their pinned upstream revisions.

This workstation is not a valid final FCI test host: it runs a generic
`PREEMPT_DYNAMIC` kernel, `/sys/kernel/realtime` is absent, and its CPU governor
was `powersave`. Do not interpret another result from that setup as trajectory
validation.

TODO on the PREEMPT_RT workstation:

- Verify `/sys/kernel/realtime` contains `1`, the controller process receives
  FIFO priority, the CPU governor is `performance`, and the dedicated robot NIC
  has stable 1 kHz latency before enabling motion.
- Build and source the pinned upstream `franka_hardware` and
  `inspire_franka_trajectory_replay`; do not patch `franka_ros2` or
  `franka_description` for this test.
- Run the `threading_5x_flange180` dry-run, then test arm-only homing before
  connecting the hand. Record the complete controller-manager log and confirm
  that homing and the 49.65 s prepared trajectory finish without an FCI reflex.
- If the discontinuity remains under PREEMPT_RT, stop testing the stock
  position JTC. Replace only its trajectory-sampling role with a minimal
  position adapter paced from the FR3 `robot_time` state. It must continue to
  claim the position interfaces so libfranka uses the robot's internal joint
  impedance controller; do not reintroduce the custom effort/PD/IK controller.
- Once hardware succeeds, document the result here and add a hardware-tested
  launch profile.

Everything in this package is in **radians** -- the Forge trajectories, the
homing YAMLs, the tracking comparison against `joint_states`. The driver
commands in **open ratios** (`1.0` fully open), running the opposite way.
`CoordinatedReplayClient.command_hand` is the single place the two meet; the
YAMLs stay in radians because they also carry the FR3's seven joints.

The runner performs this sequence:

1. load and validate the homing YAML and trajectory;
2. densify and check the arm stream against the FR3 position, velocity,
   acceleration, and jerk limits;
3. activate the replay controller and move both devices to the homing pose;
4. move the arm to the smooth stream's lead-in point;
5. start the guarded arm stream and mapped six-DOF hand stream together.

It accepts either a trajectory directory containing `metadata.json` and
`replay_data.npz`, or a coordinated NPZ containing `joint_pos_arm` and optional
`joint_pos_hand`. Raw Forge `(time, environment, 19 joints)` recordings are
mapped by joint name; passive hand joints are never commanded.

A Forge file is one file per *run*, not per episode: at a task reset the
simulator teleports the arm back to its start pose between two consecutive
samples, and a spline drawn through that teleport asks the FR3 for accelerations
in the thousands of percent. One continuous episode therefore has to be selected
before anything is prepared, by whichever of these the recording supports:

- **`--cycle N`** when the file carries a `cycle` field, as the threading
  captures do.
- **`--segment N`** otherwise. The boundaries are found rather than read: a step
  no joint could make even at the FR3's own velocity limit did not happen, so it
  is a reset. The margin is wide — real steps stay under 0.07 rad where resets
  are over 0.8 rad, against a 0.17 rad bound — and a recording with no reset in
  it needs no selection at all. Running without `--segment` lists what is on
  offer:

  ```
  this recording holds 3 episodes separated by a reset the FR3 cannot follow;
  select one continuous run with --segment N
  [0: 81 samples / 5.3 s, 1: 674 samples / 44.9 s, 2: 596 samples / 39.7 s]
  ```

The check runs after `--cycle` has had its say as well, so a reset left inside a
selected cycle is caught here rather than discovered as a limit violation two
steps later.

### The homing YAML has to match the recording

Replay homes to the YAML and then moves to the trajectory's first point. When
the two agree, that second move is nothing. When they do not, the arm makes an
unplanned trip the moment replay starts — and, more to the point, the scene the
policy was recorded against is not the scene it is being replayed into. The
runner refuses a gap over `--max-home-delta` (0.1 rad) and names the joints:

```
the homing pose in .../pickup.yaml is 2.620 rad from the trajectory's first
point, over the 0.1 rad limit: fr3_joint7 home -2.620279 vs trajectory
+0.000000 (2.620 rad); ...
```

`threading.yaml` matches its capture to the last digit, so a gap this size means
the YAML belongs to a different task configuration, not that the tolerance is
too tight.

Build and validate without motion:

```bash
cd ~/develop_ws
rg2
source install/setup.bash

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_1 --cycle 1 --dry-run
```

Inspect the coordinated recording in MuJoCo with Chi's source-compatible replay
viewer. This is the only MuJoCo process needed for trajectory inspection; do
not start a ROS simulation launch alongside it:

```bash
python3 apps/traj_replay/tests/test_mujoco_traj_replay.py \
  --trajectory apps/traj_replay/demo_trajs/traj_1
```

Add `--headless` to validate every sample and print the TCP-fit report without
opening a window. The viewer deliberately uses the recording's original
12-DoF training-hand geometry. It does not publish ROS commands and cannot move
the physical robot.

Bring up the real replay stack (this replaces the ordinary
`inspire_franka_bringup` launch for a replay session):

```bash
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0
```

Then, in a second sourced shell:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_1 \
  --cycle 1 \
  --home apps/traj_replay/demo_trajs/homing/threading.yaml
```

The runner prompts before homing and again before replay. `--yes` disables the
prompts, `--no-hand` keeps the original arm-only behavior, and `--no-arm` is
the mirror of it.

## One device at a time

On hardware the arm and the hand are two independent stacks, so either can be
left out of both the launch and the run. **The launch argument and the runner
flag have to agree**: the runner reaches the arm through the controller
manager, so a hand-only launch plus a coordinated run blocks waiting for a
controller that was never started.

```bash
# --- hand only: no FCI, no arm, no FR3 limit check ------------------------
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  hand_port:=/dev/ttyUSB0 arm:=false

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_1 --cycle 1 --no-arm \
  --home apps/traj_replay/demo_trajs/homing/threading.yaml

# --- arm only -------------------------------------------------------------
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  robot_ip:=172.16.0.2 hand:=false

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_1 --cycle 1 --no-hand
```

`--no-arm` **replays the hand on the recording's own clock**, not on the arm's.
That is the one behavioural difference worth knowing about, and it is
deliberate. The coordinated path resamples the hand onto the arm's *prepared*
clock, which the FR3 limit check time-scales (roughly 1.6x on the current
file); with no arm there is nothing to scale for, so the hand runs at the
timing the policy actually produced. `--hand-time-scale` stretches or
compresses that clock and is rejected unless `--no-arm` is given, because the
coordinated stream does not own its timing.

Two consequences follow from the decoupling:

- **No cycle is gated by the FR3.** Cycle 21 is rejected outright in a
  coordinated run, and replays fine on the hand alone. That is correct — the
  guard protects the arm — but it means a hand-only pass proves nothing about
  whether the arm can follow the same cycle.
- **Hand-only timing is not coordinated timing.** A cycle that tracks well at
  15 Hz native will see the same targets ~1.6x slower in a coordinated run.

Hand-only replay is verified on the bench RH56 (see the workspace README's
status section for the measured tracking).

The checked-in recording contains cycles 1 through 22. Cycles 1–20 and 22 are
made FR3-safe by the normal automatic time scaling (roughly 1.6x on the current
file). Cycle 21 approaches the position-dependent velocity boundary at
`fr3_joint5` and is deliberately rejected; do not bypass that guard on real
hardware. `--max-prepared-duration` can raise the default 120 s ceiling for a
legitimately long recording, but it does not disable any FR3 limit check.

## The two captures in this repo

| capture | task | selection | homing YAML |
|---|---|---|---|
| `demo_trajs/traj_1` | `Isaac-Forge-Franka-Threading-M24-v0` | `--cycle 1`..`22` | `homing/threading.yaml` |
| `demo_trajs/pickup_1` | `Isaac-Forge-Franka-Pickplace-Multi-v0` | `--env 0`..`2` and `--segment` | `homing/pickup_multi.yaml` |

Both were recorded at 15 Hz and are densified to the controller's 1 kHz by a
clamped cubic spline through the waypoints (`prepare.rate`,
`prepare.interpolation` in `config/replay.yaml`), then time-scaled until the
FR3's velocity, acceleration and jerk limits are satisfied.

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/pickup_1 --env 0 --segment 1 \
  --home apps/traj_replay/demo_trajs/homing/pickup_multi.yaml --dry-run
```

The pickplace capture holds three environments and resets twice in each, giving
seven usable episodes plus two 3-sample tails that are listed and refused. All
seven prepare within the FR3's limits at time scales between 1.08x and 1.64x.

`homing/pickup_multi.yaml` is derived from the capture itself — every one of its
six episode starts agrees on the same reset pose — because `homing/pickup.yaml`
cites a single-object variant of the task and its **arm** block is up to 2.62 rad
away from where this capture begins. Its hand block matches exactly, which is
what makes the arm block look like a stale copy rather than a calibration. Worth
settling with whoever exported the capture before a hardware run.

## Running several cycles, and ending at home

`--cycle N` replays exactly one cycle, which is the right unit to validate but
usually not the run you want on hardware. The threading capture holds its cycles
back to back with continuous seams, so consecutive ones can simply be taken
whole:

```bash
ros2 run inspire_franka_trajectory_replay make_cycle_trajectory \
  apps/traj_replay/demo_trajs/traj_1 --cycles 5 \
  --home apps/traj_replay/demo_trajs/homing/threading.yaml \
  --output apps/traj_replay/demo_trajs/threading_5x
```

What the capture does *not* do is come back: each cycle's `return_to_reset`
phase stops about 0.11 rad from the homing pose and the next rollout starts from
there, so playing it to the end leaves the arm short of where it began. The
generator appends the missing piece — a cubic Hermite ramp onto the exact homing
pose, matching the recording's arrival velocity (the last sample is still moving
at ~0.2 rad/s, already toward home) and arriving at rest. Starting that ramp
from rest instead would put a 0.2 rad/s step in the middle of the stream.

It refuses a selection that is not one continuous run, checked against the same
FR3 velocity bound the reader uses, and it refuses the pickplace captures
outright — they have no `cycle` field and their episodes are separated by a reset
teleport, so there is nothing to concatenate.

The result is written in the **coordinated** NPZ form, so it is one continuous
run by construction and needs neither `--cycle` nor `--segment`:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_5x \
  --home apps/traj_replay/demo_trajs/homing/threading.yaml --dry-run
```

For cycles 1–5 that is 451 samples over 30.0 s, densified to 49649 samples at
1 kHz and time-scaled 1.588x to stay inside the FR3's limits (joint 6
acceleration binds at 97 %, the same as any single threading cycle). It starts
at the homing pose and its last commanded sample is 3e-4 rad from it, at rest.

One consequence of the coordinated form: `test_mujoco_traj_replay.py` reads the
Forge schema and cannot open the generated file. Preview the cycles in the
source capture instead — the only thing the generated file adds is the closing
ramp.

The hand mapping from the Forge model is:

| Forge joint | Inspire driven joint |
|---|---|
| `little_joint_0` | `pinky_proximal_joint` |
| `ring_joint_0` | `ring_proximal_joint` |
| `middle_joint_0` | `middle_proximal_joint` |
| `index_joint_0` | `index_proximal_joint` |
| `thumb_joint_1` | `thumb_proximal_pitch_joint` |
| `thumb_joint_0` | `thumb_proximal_yaw_joint` |
