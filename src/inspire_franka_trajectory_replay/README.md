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
| `demo_trajs/traj_2_cycle3` | current-orientation single-cycle candidate; dry-run validated, physical validation pending | one continuous cycle plus return home; use its colocated `homing.yaml` |
| `demo_trajs/traj_2_5x` | current-orientation five-cycle candidate; dry-run validated, physical validation pending | cycles 1–5 plus return home; use its colocated `homing.yaml`, `--time-scale 5`, `--max-prepared-duration 300`, and optionally `--interactive-pause` |
| `demo_trajs/threading_cycle1_flange180` | **legacy-source hardware baseline**; validated 2026-09-09 with its historical joint-7 compensation | one continuous cycle; use its colocated `homing.yaml` |
| `demo_trajs/traj_1` | source capture; **outdated for direct arm replay** because it uses the legacy mount convention | `--cycle 1`..`22`; raw `homing/threading.yaml`; safe for viewing, conversion, or `--no-arm` |
| `demo_trajs/threading_5x` | generated intermediate; **outdated for current-mount arm replay** | raw `homing/threading.yaml` |
| `demo_trajs/threading_5x_flange180` | **validated legacy-source extended run**; confirmed on hardware at 5x slowdown with interactive pause | five continuous cycles; use its colocated `homing.yaml`, `--time-scale 5`, and `--interactive-pause` |
| `demo_trajs/pickup_1` | separate task; not part of the threading baseline | `--env 0`..`2`, `--segment`, and `homing/pickup_multi.yaml` |

The source captures `traj_1`, `traj_2`, and `pickup_1` were recorded at 15 Hz.
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
