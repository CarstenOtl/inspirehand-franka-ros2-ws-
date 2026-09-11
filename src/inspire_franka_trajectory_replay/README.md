# inspire_franka_trajectory_replay

Coordinated hardware replay for the FR3 and Inspire RH56. Hardware replay uses
the working `JointImpedanceExampleController`'s torque law and gains:

```text
dq_filtered = 0.01 * dq_filtered + 0.99 * dq_measured
torque = K * (q_reference - q_measured) - D * dq_filtered
K = [24, 24, 24, 24, 10, 6, 2]
D = [2, 2, 2, 1, 1, 1, 0.5]
```

The `franka_trajectory_replay/TrajectoryReplayController` effort branch supplies
`q_reference` from a smooth homing ramp and interpolated recorded waypoints,
replacing the example's sinusoidal reference. Its `example_joint_impedance`
function reproduces the simple example's PD law; Coriolis compensation and
the extra controller-side torque limiter are disabled in
`config/controllers_joint_impedance.yaml`. The hardware wrapper's existing
torque rate limiter and the robot's gravity compensation remain in the path.
The Franka submodules are unchanged. This profile does not change collision
thresholds or use the much stiffer IK example's gains.

The controller claims seven **effort** interfaces, so `franka_hardware` starts
`startTorqueControl()`. References advance using the controller update period,
as in the simple example. Python prepares the trajectory before submitting it
over `~/goto` and `~/trajectory`; it does not send torques from a Python loop.
Completion and abort use the controller's status and `~/abort` topics. The hand
starts after the arm's acceptance is observed on the 50 Hz status stream; this
is coordination with status-message latency, not cycle-exact synchronization.

The hand stays on its independent RS485 driver and receives 50 Hz position
commands on `/inspire_hand/command`. Trajectory preparation still checks FR3
position, velocity, acceleration, and jerk margins. The controller also checks
waypoints and their start reference before accepting them.

## Hardware status and validation

The user has confirmed that the unmodified simple joint-impedance example runs
on the FR3 and `/sys/kernel/realtime` is `1`. The previous position-JTC replay
failed with motion-generator velocity/acceleration discontinuity reflexes.
Those runs do not establish the precise cause of the received position-sample
discontinuity. On 2026-09-08, the user confirmed that the current simple
joint-impedance waypoint replay works well on the arm. This is a qualitative
hardware check; quantitative waypoint tracking accuracy has not been measured.

The example gains are deliberately compliant. Completion means the **reference**
has reached its end, not that measured joints are exactly on the waypoint.
Inspect measured tracking before using the replay for contact tasks. Homing
starts from the measured pose; subsequent ramps start from the last reference.

### Change joint stiffness while replay is running

In the default joint-impedance mode, `stiffness_scale`, `k_gains`, and
`d_gains` are live ROS parameters. The simplest control is
`stiffness_scale`, which multiplies all seven configured `K` values without
changing `D`. From another sourced terminal, while the controller is active:

```bash
# Inspect the current target scale (1.0 is the checked-in example profile).
ros2 param get /trajectory_replay_controller stiffness_scale

# Try a modest increase. Effective K becomes
# [30, 30, 30, 30, 12.5, 7.5, 2.5] Nm/rad.
ros2 param set /trajectory_replay_controller stiffness_scale 1.25

# Return to the baseline at any time.
ros2 param set /trajectory_replay_controller stiffness_scale 1.0
```

Every accepted change is blended in the 1 kHz control loop with a zero-slope
quintic transition; the default transition is one second. Set its duration
before the next gain change if a slower adjustment is wanted:

```bash
ros2 param set /trajectory_replay_controller gain_ramp_duration 2.0
```

Per-joint tuning is also live:

```bash
ros2 param set /trajectory_replay_controller k_gains \
  "[30.0, 30.0, 30.0, 30.0, 12.0, 7.0, 2.5]"
ros2 param set /trajectory_replay_controller d_gains \
  "[2.5, 2.5, 2.5, 1.2, 1.2, 1.2, 0.6]"
```

The status topic exposes `stiffness_scale_target`, `k_gains_applied`, and
`d_gains_applied`; the applied values move during the ramp. Invalid sizes,
negative values, NaNs, and live changes in position mode are rejected.

The ramp prevents an instantaneous gain jump, but it does not make an
arbitrarily high gain safe. Increase in small steps, watch tracking and
oscillation, and keep the normal Franka collision/reflex protections and lab
procedure in force. Changing `stiffness_scale` does not automatically retune
damping; use `d_gains` if the higher stiffness becomes underdamped.

After stopping existing bringup, build in the ROS container:

```bash
cd /root/develop_ws
colcon build --symlink-install --packages-select \
  franka_trajectory_replay inspire_franka_trajectory_replay
source install/setup.bash
colcon test --packages-select franka_trajectory_replay inspire_franka_trajectory_replay
colcon test-result --verbose
```

## Joint-7 orientation policy and threading baselines

`traj_2` defines the convention for new trajectories. It and all later policy
captures are developed with the current hardware joint orientation, so
`fr3_joint7` must be replayed exactly as recorded with `offset_rad: 0.0`.
Do not apply the legacy `+90 deg`/`+pi/2` retargeting step or create a
`*_flange180` derivative from these captures. Their matching homes use the
same unmodified joint-7 convention.

The rotation remains part of the historical `traj_1` artifacts only. The de
facto legacy hardware baseline is cycle 1 retargeted for the physical
180-degree flange mount. It adds exactly `+pi/2 rad` to `fr3_joint7` in both
the trajectory and its matching home, and applies the hand-only support-finger
override at replay time. This exact configuration was confirmed on the real
FR3 + RH56 on 2026-09-09: tool orientation was correct and replay was good.

Do not substitute raw `traj_1` or `homing/threading.yaml` for that historical
baseline. They use the legacy source convention (`fr3_joint7=-1.846102 rad` at
home rather than the baseline's `-0.275306 rad`). Conversely, do not apply the
historical offset to `traj_2` or future captures. Do not bypass a mismatch with
`--max-home-delta`; that only authorizes the wrong move.

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers --dry-run

# Terminal 1: launch holds the current pose; it does not begin replay.
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0

# Terminal 2: prompts before homing and before replay.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers
```

The current-orientation five-cycle `traj_2` candidate needs no joint-7
retargeting and no `--cycle` selection because it is already one continuous
artifact. Its 5x stream is 159 seconds, so raise only the duration allocation
guard—not any motion limit:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_2_5x \
  --home apps/traj_replay/demo_trajs/traj_2_5x/homing.yaml \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 300 \
  --dry-run
```

This candidate passes dry-run preparation but is not labelled physically
validated until a complete hardware run is confirmed. Add
`--close-support-fingers` only when deliberately replacing its recorded support
finger values with the fully closed limits.

The default runner refuses a position controller or the Coriolis-enabled IK
profile.

The old MuJoCo launch still runs position JTC and requires the explicit runner
option `--arm-controller position-jtc`:

```bash
ros2 launch inspire_franka_trajectory_replay sim_replay.launch.py
# Another sourced shell:
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers \
  --arm-controller position-jtc
```

That simulation validates the retained position path, not the new effort law's
closed-loop performance. `controllers_internal_impedance.yaml` is retained only
for explicitly selected position-JTC sessions; it is no longer the default.

To inspect a trajectory that the FR3 safety preparation rejects, the MuJoCo
position-JTC path has an explicit simulation-only override. The runner refuses
this option with the hardware `joint-impedance` controller:

```bash
ros2 launch inspire_franka_trajectory_replay sim_replay.launch.py headless:=false

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_3 \
  --home apps/traj_replay/demo_trajs/traj_3/homing.yaml \
  --arm-controller position-jtc \
  --time-scale 5 \
  --allow-unsafe-simulation
```

This override is for visual diagnosis only. It does not make the trajectory
hardware-safe, and interactive pause is unavailable on the position-JTC path.

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

The raw `homing/threading.yaml` matches raw `traj_1` to the last digit, but both
are legacy source material. The historical `traj_1`-derived 180-degree baseline
must use its colocated retargeted `homing.yaml`. In contrast, `traj_2` and its
derived artifacts use their colocated zero-offset `homing.yaml` and keep joint 7
unchanged. A 1.571 rad joint-7 mismatch is the signature of mixing conventions,
not a tolerance that should be raised.

Build and validate without motion:

```bash
cd ~/develop_ws
rg2
source install/setup.bash

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers --dry-run
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
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers
```

`--close-support-fingers` changes only the hand: pinky, ring, and middle are
held at their fully closed `1.47 rad` limits in the homing command and at every
replay waypoint. It never changes or retargets an FR3 joint. Omit the flag when
the trajectory's recorded support-finger motion should be preserved. Historical
`traj_1` derivatives already contain their joint-7 compensation in the artifact;
`traj_2` and future trajectories contain no such compensation.

The runner prompts before homing and again before replay. `--yes` disables the
prompts, `--no-hand` keeps the original arm-only behavior, and `--no-arm` is
the mirror of it.

Scale only the index- and thumb-MCP flexion waypoints at replay time with
`--finger-flexion-scale`. The scale is applied to the loaded trajectory before
interpolation; the YAML homing pose, all seven arm joints, the three support
fingers, and thumb yaw remain unchanged. `1.3` means 30% more flexion than the
stored waypoints, while `1.5` means 50% more. The runner rejects a scale that
would exceed an Inspire joint limit. Use an original, unscaled artifact such as
`traj_2_6x` so the factor is relative to the original recording:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_2_6x \
  --home apps/traj_replay/demo_trajs/traj_2_6x/homing.yaml \
  --finger-flexion-scale 1.3 \
  --time-scale 5 \
  --max-prepared-duration 300 \
  --dry-run
```

At open ratio `0.0` the thumb swings past the palm plane, so the bottom of its
commanded range is unusable. The hand driver therefore treats `0.25` as the
thumb's zero and universally rescales `thumb_proximal_yaw_joint` commands onto
`[0.25, 1.0]` in its existing open-ratio convention (`1.0` is open):

```
physical = 0.25 + 0.75 * commanded
```

That is equivalent to contracting the yaw angle onto `[0, 0.981]` rad with the
current kinematics — `0.981` rad is 56.2°, against 74.9° at a raw `0.0`. Because
the map is a rescale rather than a floor, it stays monotonic and every commanded
value remains distinct; a floor would collapse the bottom quarter of the range
onto one pose. Note that the whole range contracts, so mid-travel commands close
the thumb further than they used to: `0.5` now lands at 28.1° rather than 37.5°.

Replay publishes its original commands without pre-scaling thumb abduction; only
its feedback expectation uses the driver's shared setting so homing agrees with
the physical target. Thus a raw ROS thumb-abduction command of `0.0` reaches the
driver as `0.0`, and the driver sends `0.25` to the hand. Arm motion, thumb
flexion, all other hand joints, and the source NPZ/YAML files remain unchanged.

### Pause, adjust the scene, and continue

The hardware waypoint controller can pause a coordinated replay without losing
its place. Add `--interactive-pause`; during the trajectory, **SPACE** smoothly
ramps the arm's trajectory clock to zero and holds both the arm and hand. Wait
until the runner prints `PAUSED` before approaching the setup. Press **SPACE**
again to resume from that trajectory position, or **q** to abort and hold.

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/pickup_1 --env 0 --segment 1 \
  --home apps/traj_replay/demo_trajs/homing/pickup_multi.yaml \
  --interactive-pause
```

The hand is keyed to the controller's reported trajectory time, not wall time,
so an arbitrarily long pause does not advance either device. Interactive pause
requires the default `joint-impedance` hardware controller; it is intentionally
unavailable for `--no-arm` and the legacy `position-jtc` simulation path.

The arm remains actively torque-controlled while paused. `PAUSED` means the
reference clock is stopped, not that power is removed or that the workspace is
safe to enter; follow the lab's hardware access procedure when repositioning
objects.

## Cartesian impedance replay

`--arm-controller cartesian-impedance` replays the same captures with the torque
law of Franka's `CartesianImpedanceExampleController` (franka_ros2 v3.5.3):
task-space stiffness and damping through the Jacobian transpose, a
damped-pseudo-inverse nullspace term, and coriolis compensation, with the
example's own first-order target filter. The law is lifted verbatim into
`franka_trajectory_replay/cartesian_impedance.hpp` and unit-tested against a
transcription of the upstream `update()`; the surrounding replay machinery
(goto, trajectory, pause/resume/abort, status) is the joint controller's design
in pose space. The full design, its settled decisions and the hardware
validation ladder are in `docs/cartesian_replay_blueprint.md`.

**Status: built and tested in the container, dry-run validated, and the full
flow has run unattended in MuJoCo up to and including trajectory execution;
not yet run on the arm.** Follow the blueprint's ladder (hold test, goto test,
arm-only replay, coordinated replay) before treating it as a working mode.

### Simulation first

`sim_replay.launch.py arm_controller:=cartesian-impedance` runs both hardware
replay controllers over MuJoCo's effort interfaces in a gravity-free copy of
the flange scene (libfranka compensates gravity underneath a torque controller
on the real arm; mujoco_ros2_control does not), with the Cartesian controller
on its built-in DH model (`model_source: dh`) because the simulator has no
`robot_model` or `cartesian_pose_state` interfaces. The runner sees that model
source, prints that it is skipping the robot-frame preflight, and otherwise
does exactly what it does on hardware: homes with the joint controller, swaps,
settles onto the first pose, streams. One controller manager at a time: stop
any hardware bringup before starting the simulator, and the simulator before
bringing up the arm.

```bash
# Unattended: launch, wait, replay traj_2_cycle3 at 5x, print, shut down.
# Refuses to start while any /controller_manager is on the graph.
./apps/traj_replay/sim_replay_smoke.sh

# Or by hand. Terminal 1 (headless:=false opens the MuJoCo viewer):
ros2 launch inspire_franka_trajectory_replay sim_replay.launch.py \
  arm_controller:=cartesian-impedance headless:=false
# Terminal 2:
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_2_cycle3 \
  --home apps/traj_replay/demo_trajs/traj_2_cycle3/homing.yaml \
  --arm-controller cartesian-impedance --time-scale 5 --interactive-pause
```

What the simulation shows is the *flow* against MuJoCo's dynamics, not the
FR3's: the Menagerie FR3 model carries 1.137 Nm of joint friction with nothing
compensating it below the controller, so the example's 10 Nm/rad rotational
stiffness cannot hold the wrist there and the first headless run at 2x tripped
the 0.35 rad tracking fault after 8 s. `controllers_sim_impedance.yaml`
therefore runs the simulation at `stiffness_scale: 4.0`; the hardware profile
keeps the example's gains. The runner prints the peak position and orientation
tracking error after every goto and trajectory, in the sim and on the arm.

Everything that is not the arm's command type is unchanged: capture loading,
`--cycle`/`--segment`, homing and `--max-home-delta`, `--time-scale`,
`--finger-flexion-scale`, `--close-support-fingers`, the 50 Hz hand stream
keyed on the controller's trajectory clock, `--interactive-pause`, `--dry-run`.

```bash
# Terminal 1: both replay controllers are loaded; the joint controller comes
# up active and holding, the Cartesian one inactive.
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  arm_controller:=cartesian-impedance

# Terminal 2: dry run first. Prints the joint-space summary, then the pose
# stream's libfranka Cartesian limit check.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_2_cycle3 \
  --home apps/traj_replay/demo_trajs/traj_2_cycle3/homing.yaml \
  --arm-controller cartesian-impedance \
  --time-scale 5 --interactive-pause --dry-run
```

Live, the runner: homes the arm with the validated joint-impedance controller
(the policy's home is a joint configuration, which a Cartesian controller with
20 Nm/rad of nullspace stiffness cannot promise to reach); reads one
`FrankaRobotState` and refuses to continue unless the robot's `F_T_EE` is the
identity tool the pose stream assumes and forward kinematics of the measured
joints agrees with the robot's `O_T_EE`; swaps to
`cartesian_trajectory_replay_controller` (both claim the effort interfaces, so
franka_hardware stays in torque control); ramps the Cartesian reference onto
the stream's first pose and waits for the example's filter to settle; then
sends the pose stream. Pause, resume and abort work as in joint mode, and the
hand follows the same trajectory clock.

The pose stream is forward kinematics of the *prepared* joint stream, so the
FR3 velocity/acceleration/jerk check and the automatic time scaling apply
unchanged and the joint stream itself is the nullspace target. The recorded
`tcp_pos` in the Forge captures is the fingertip midpoint and is not used.

Live tuning, while the Cartesian controller is active:

```bash
ros2 param set /cartesian_trajectory_replay_controller stiffness_scale 1.25
ros2 param set /cartesian_trajectory_replay_controller rotational_stiffness 20.0
ros2 param set /cartesian_trajectory_replay_controller nullspace_stiffness 30.0
# 1.0 bypasses the example's 200 ms target filter; the replay reference is
# already smooth, so compare both on the same artifact.
ros2 param set /cartesian_trajectory_replay_controller target_filter 1.0
```

`--stiffness-scale N` sets the first of these before replay. Changes take
effect through the example's own 0.005 filter. The status topic reports the
applied gains, the reference pose, the position and orientation error, the
smallest joint-limit margin, and `tracking_fault`: when the measured pose
drifts more than `max_position_error` (8 cm) or `max_orientation_error`
(0.35 rad) from the reference the controller ramps its clock to zero, holds,
and the runner reports the fault instead of success.

`~/cartesian_state` publishes, every cycle, the unfiltered target, the filtered
reference, the measured pose, the law's six-vector error and the task,
nullspace, coriolis and commanded torques separately, so a tracking problem can
be attributed without re-deriving forward kinematics.

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
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --no-hand
```

`--no-arm` **replays the hand on the recording's own clock**, not on the arm's.
That is the one behavioural difference worth knowing about, and it is
deliberate. The coordinated path resamples the hand onto the arm's *prepared*
clock, which the FR3 limit check time-scales (roughly 1.6x on the current
file); with no arm there is nothing to scale for, so the hand runs at the
timing the policy actually produced. `--time-scale` stretches or compresses
that hand-only clock too. The older `--hand-time-scale` spelling remains an
alias for hand-only runs and is rejected unless `--no-arm` is given.

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

## Trajectory status

| artifact | status | selection / matching home |
|---|---|---|
| `demo_trajs/traj_2` | source capture using the **current hardware joint orientation**; joint-7 offset is zero | `--cycle 1`..`6`; use its colocated `homing.yaml` |
| `demo_trajs/traj_2_6x` | all six original cycles plus return home; supports runtime MCP scaling | one continuous run; use its colocated `homing.yaml` and optionally `--finger-flexion-scale N` |
| `demo_trajs/traj_2_cycle3` | current-orientation single-cycle candidate; dry-run validated, physical validation pending | one continuous cycle plus return home; use its colocated `homing.yaml` |
| `demo_trajs/traj_2_5x` | current-orientation five-cycle candidate; dry-run validated, physical validation pending | cycles 1–5 plus return home; use its colocated `homing.yaml`, `--time-scale 5`, `--max-prepared-duration 300`, and optionally `--interactive-pause` |
| `demo_trajs/traj_3` | V2-policy source capture; raw arm replay is rejected because joint 5 reaches its position-dependent velocity boundary | 449-sample continuous rollout (the two-sample reset tail is ignored automatically); matching home is colocated, but do not bypass the FR3 safety guard |
| `demo_trajs/traj_3_joint5_cap_2p8` | minimally retargeted `traj_3` hardware candidate; saturated joint-5 waypoints capped at 2.800 rad; physical validation pending | use its colocated `homing.yaml` and `--time-scale 5` |
| `demo_trajs/traj_3_multi` | raw ten-cycle V2 source capture; cycles 1, 6, 7, and 8 reach the joint-5 braking boundary | preserve as source material; individual cycles can be inspected with `--cycle N`, but use the derived artifact for the complete run |
| `demo_trajs/traj_3_multi_joint5_cap_2p8` | all ten V2 cycles plus return home; 171 saturated joint-5 waypoints capped at 2.800 rad; dry-run validated at 5x, physical validation pending | one continuous run; use its colocated `homing.yaml`, `--time-scale 5`, `--interactive-pause`, and `--max-prepared-duration 400` |
| `demo_trajs/threading_cycle1_flange180` | **legacy-source hardware baseline**; validated 2026-09-09 with its historical joint-7 compensation | one continuous cycle; use its colocated `homing.yaml` |
| `demo_trajs/traj_1` | source capture; **outdated for direct arm replay** because it uses the legacy mount convention | `--cycle 1`..`22`; raw `homing/threading.yaml`; safe for viewing, conversion, or `--no-arm` |
| `demo_trajs/threading_5x` | generated intermediate; **outdated for current-mount arm replay** | raw `homing/threading.yaml` |
| `demo_trajs/threading_5x_flange180` | **validated legacy-source extended run**; confirmed on hardware at 5x slowdown with interactive pause | five continuous cycles; use its colocated `homing.yaml`, `--time-scale 5`, and `--interactive-pause` |
| `demo_trajs/pickup_1` | separate task; not part of the threading baseline | `--env 0`..`2`, `--segment`, and `homing/pickup_multi.yaml` |

The source captures `traj_1`, `traj_2`, `traj_3`, `traj_3_multi`, and `pickup_1`
were recorded at 15 Hz.
Selected or derived runs are densified to the controller's 1 kHz by a clamped
cubic spline through the waypoints (`prepare.rate`, `prepare.interpolation` in
`config/replay.yaml`), then time-scaled until the FR3's velocity, acceleration,
and jerk limits are satisfied.

Use `--time-scale N` to deliberately stretch that waypoint clock in addition
to the automatic safety scaling. For example, `--time-scale 5` makes each
15 Hz waypoint interval and all of its interpolated samples take five times as
long; the coordinated hand follows the same stretched clock:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/pickup_1 --env 0 --segment 1 \
  --home apps/traj_replay/demo_trajs/homing/pickup_multi.yaml \
  --time-scale 5
```

Values of 5 or 10 therefore mean 5x or 10x slower, not faster. The dry-run
summary prints the final scale, which can be slightly larger if satisfying the
FR3 limits requires further automatic slow-down. The same option works for
arm-only and hand-only replay; `--hand-time-scale` remains as a deprecated
hand-only alias.

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

## Deriving additional current-orientation cycles

`--cycle N` replays exactly one cycle, which is the right unit to validate but
usually not the run you want on hardware. The threading capture holds its cycles
back to back with continuous seams, so consecutive ones can simply be taken
whole:

```bash
ros2 run inspire_franka_trajectory_replay make_cycle_trajectory \
  apps/traj_replay/demo_trajs/traj_2 --first 1 --cycles 5 \
  --home apps/traj_replay/demo_trajs/traj_2/homing.yaml \
  --output /tmp/traj_2_5x
```

If a reviewed source capture reaches the known joint-5 braking boundary, the
generator can preserve the source and apply the same narrowly scoped cap used
by the checked-in V2 candidates:

```bash
ros2 run inspire_franka_trajectory_replay make_cycle_trajectory \
  apps/traj_replay/demo_trajs/traj_3_multi --first 1 --cycles 10 \
  --home apps/traj_replay/demo_trajs/traj_3_multi/homing.yaml \
  --joint5-cap 2.8 \
  --output /tmp/traj_3_multi_joint5_cap_2p8
```

The output metadata records the cap, source maximum, number of affected
samples, and maximum change. This option is an explicit waypoint retarget, not
a way to bypass preparation: the generated artifact must still pass the normal
FR3 limit check.

That output is ready for preparation without another orientation step. New
captures use the hardware joint convention and must not be passed through
`retarget_flange_mount.py`. The retargeting tool remains only for intentional
conversion of legacy `traj_1`-convention sources; its `+90 deg` operation is not
part of the current workflow.

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
  /tmp/traj_2_5x \
  --home apps/traj_replay/demo_trajs/traj_2/homing.yaml \
  --time-scale 5 \
  --max-prepared-duration 300 \
  --dry-run
```

For `traj_2` cycles 1–5, the checked-in `traj_2_5x` artifact contains 472 source
samples over 31.4 seconds, including its return home. At `--time-scale 5` it is
densified to 159000 samples at 1 kHz over 159 seconds and passes the configured
FR3 limits with joint 7 unchanged. It is dry-run validated; physical validation
must still be recorded separately. The historical `threading_5x_flange180`
artifact remains the already validated five-cycle run derived from legacy
`traj_1`.

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
