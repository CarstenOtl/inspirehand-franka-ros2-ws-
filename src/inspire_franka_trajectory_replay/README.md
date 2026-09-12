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
opening a window. The viewer retains the recording's original 12 joint names,
but uses the official TienKung 2 Pro hand geometry and physical settings. It
does not publish ROS commands and cannot move the physical robot.

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

### Hand-guided interventions: safe DAgger on the real rig

`--intervene` lets the operator take the arm mid-rollout, do the task step by
hand, and give it back. It is the pause above plus three things: the arm is
handed to gravity compensation so it can be moved freely, the poses the
operator marks are recorded, and the run then rejoins the trajectory at the
interrupted cycle's **release point** rather than where it was interrupted.

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8 \
  --home apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8/homing.yaml \
  --time-scale 5 --max-prepared-duration 400 \
  --intervene --note "nut jammed on cycle 3"
```

One intervention, key by key:

| key | what happens |
|---|---|
| **Enter** | steps in, in one press. The trajectory clock ramps to zero, and once the controller reports it stopped, `gravity_compensation_example_controller` takes the arm's effort interfaces. **Be holding the arm before you press it**, and wait for `FLOATING`. |
| preset / jog keys | command the hand, from `config/hand_presets.yaml` exactly as in `capture_demo` — `o` open, `r` preshape, `p` pinch, `-`/`=` and `[`/`]` to jog. This is how the grip is tuned on the object actually in front of you. |
| **Enter** (inside the stage) | records this pose: the arm's and the hand's measured joints at that instant. |
| **g** | ends the supervision and continues the run. |
| **q** | ends the run here; the arm stiffens and holds. |
| **SPACE** | still the plain pause: the clock stops and the arm keeps holding stiffly, nothing goes limp. `i` then steps in from there, for when you want to look before deciding. |

`g` hands back in two steps, both printed with their distance and put behind a
prompt first:

1. The replay controller is reactivated. It initialises its reference from the
   measured joints, so it holds the pose the arm was left in and there is no
   step to ramp out.
2. The arm ramps onto the current cycle's release point and the rest of the
   trajectory is replayed from there. The gap is refused above
   `--max-release-delta` (0.6 rad), naming the joints.

The trajectory then carries on into the next cycle, and Enter works again.

**Nothing you did by hand is re-executed.** The arm goes from where you left it
straight to the release point; the recorded poses are a record, not a motion to
replay. That is deliberate for this task: once a nut has been threaded by hand,
driving the arm back through the turn with the nut already on the bolt is not a
correction, it is a collision. To replay a session's poses as a motion of their
own — to check they are reachable, or to build training material — run
`extract_waypoints` on the session afterwards and replay the artifact it writes.

Because nothing is replayed from them, recording no poses at all is valid: if
you only repositioned the workpiece, press `g` and the run carries on. It says
so, since a session whose purpose is the record is worth a word when it has
none.

**Nothing moves the arm while a person is touching it.** The recording node
this uses is the one `capture_demo` uses, and it holds no arm publisher and no
arm command interface at all; during an intervention the only thing published
is the hand command. Every motion happens after the handback, through the same
`goto`/`trajectory` path and the same FR3 preparation as an ordinary replay.
`FLOATING` means zero commanded torque with libfranka compensating gravity
underneath — it does not mean the arm is safe to let go of, and it does not
remove the lab's hardware access procedure.

#### Where the release point comes from

Each cycle of a Forge threading rollout is `policy`, then `follow_waypoints`,
then `return_to_reset`. The middle phase is the release: the index finger goes
from 0.70 rad of flexion to 0.06 rad over sixteen samples with the arm still at
the bolt, and the source metadata says `threading_release_motion: manual`. That
is the point worth rejoining, because an intervention has already done by hand
what the rest of the `policy` phase was going to attempt; what still has to
happen is letting go, retreating, and the next cycle.

`make_cycle_trajectory` resolves those per-sample phases into one sample index
per cycle and writes them into the artifact's `metadata.json`:

```json
"release_phase": "follow_waypoints",
"cycle_index": [
  {"cycle": 1, "start_sample": 0, "end_sample": 102,
   "release_sample": 56, "release_time_s": 3.733},
  ...
]
```

The flags are in the **artifact's** own 15 Hz samples. `--intervene` prints
where each one lands on the controller's prepared clock, which is where the
automatic time scaling and `--time-scale` have had their say:

```
intervention: 10 cycles, each releasing in its 'follow_waypoints' phase
  cycle 1: samples 0-102, release at 56 (3.73 s)  ->  prepared stream 19.67 s
  cycle 2: samples 103-201, release at 155 (10.33 s)  ->  prepared stream 52.67 s
  ...
```

`traj_3_multi_joint5_cap_2p8` carries these flags. An artifact that does not is
refused rather than guessed at, with the command that regenerates it. Because
the flags are numbered against the artifact's samples, `--intervene` also
refuses `--cycle`/`--segment`: use a derived continuous artifact instead.

The remainder after a release point is re-prepared from the source slice rather
than cut out of the dense stream that was running. That is what puts a lead-in
ramp in front of it, accelerating from rest into the velocity the recording
actually has at that sample; a dense stream cut mid-motion would ask the arm to
be already moving the instant the trajectory is accepted.

#### A pose the arm cannot hold is refused at capture, not at handback

The FR3's velocity limit follows joint position: close enough to a stop, the
envelope does not contain zero, and the arm is *required* to still be moving
away from the limit. Standing still there is itself a limit violation, and no
amount of slowing down fixes it. This task already runs joint 5 to 2.80 rad
against a 2.8763 rad limit, so it is the realistic way for a hand-guided pose
to be unusable.

Such a pose is therefore rejected when **Enter** is pressed, while the operator
still has hold of a floating arm and can move that joint out:

```
waypoint 3 NOT recorded: the arm cannot hold this pose.
  fr3_joint5 is at +2.8500 rad, 0.0263 rad from its upper limit +2.8763 rad and
  inside the braking zone, where the FR3 velocity envelope is [-4.183, -0.025]
  rad/s and so does not contain nought: the arm cannot be commanded to hold
  still here at any speed
Move that joint away from its limit and press Enter again.
```

The keypress still appears in the event log as `pose_capture` followed by
`pose_capture_refused`; it just does not join the correction. Every one of the
1112 samples of `traj_3_multi_joint5_cap_2p8` passes this check.

#### What a session writes

`logs/dagger/<UTC stamp>[_<note>]/`, with `--session-root` to put it elsewhere:

- `events.jsonl` — the rollout's own marks (`rollout_start`, `intervention_open`,
  `arm_floating`, `arm_stiff`, `intervention_release`, `handback`, `rollout_end`)
  beside the operator's `pose_capture` and `hand_command` events. `arm_stiff`
  carries the arm and hand joints the supervision was left at, which is where
  the goto onto the release point starts from. Flushed and fsynced per line, so
  a session that ends on a Ctrl-C still describes everything it did.

  What it does **not** hold is the continuous path your hand took: only the
  poses you marked. Marking is deliberate rather than incidental — but if the
  guided motion itself is wanted as training data (a thread turn is continuous,
  and a handful of marks is a thin description of one), that wants a bag
  recorded across the stage, which this does not yet do.
- `manifest.json` — one entry per intervention: where it paused, which cycle,
  which release sample it rejoined at, how many waypoints, and how it ended.

`pose_capture` is written in the shape `capture_demo` writes it, so the session
is a capture session as far as the rest of the toolchain is concerned, and the
corrections can be turned into a replayable artifact of their own with no bag
read at all:

```bash
ros2 run inspire_franka_trajectory_replay extract_waypoints logs/dagger/<stamp>
```

To build one composite artifact that keeps the original prefix and suffix but
replaces the interrupted interval with a smooth path through the marked poses,
use `splice_intervention` instead:

```bash
ros2 run inspire_franka_trajectory_replay splice_intervention \
  logs/dagger/<stamp>

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  logs/dagger/<stamp>/composite \
  --home logs/dagger/<stamp>/composite/homing.yaml \
  --time-scale 5 --max-prepared-duration 500 --dry-run
```

The splice begins with the original artifact through `paused_sample`, visits
the intervention's captured poses on zero-velocity/zero-acceleration quintic
segments, eases the hand onto each recorded posture while the arm dwells, and
then bridges to `rejoin_sample` before retaining the rest of the original
artifact. It also renumbers the artifact's `cycle_index` release markers, so a
later `--intervene` run still refers to the composite's own samples.

This is an approximation of the correction, not the path the operator's hand
actually took: intervention sessions record only deliberate `Enter` captures,
not continuous arm state. Capture more intermediate poses when the shape of the
replacement matters. The generated metadata states this explicitly, and the
result must pass the ordinary `replay_trajectory --dry-run` before motion.

**Status: built, unit-tested, and dry-run validated; not yet run on the arm.**
The pieces it is assembled from are hardware-validated — the joint-impedance
replay controller, its pause, `capture_demo`'s gravity-compensation hand
guiding — but the switch between them mid-rollout, and the handback, have not
been performed on the FR3. Walk it up: step in, record one pose without moving
the arm, and press `g`, so the only motion is the goto onto the release point.
Then one where the arm is guided somewhere first.

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

### The controlled point: the hand's grasp centre

The impedance acts about the Inspire hand's grasp centre, `[-0.0874, -0.0327,
0.1453]` m in `fr3_link8`, 172.7 mm from the flange. That is the thumb/index
fingertip midpoint at the threading grip, and it is the point the policy itself
controlled: the Forge captures' recorded `tcp_pos` is the same midpoint on the
training hand, matching the model to 1-3 mm of scatter. The value is measured on
the physical RH56, where the URDF and the MuJoCo model agree on it to 0.1 mm.
Its scatter across the 423 grip samples of `traj_3` is 1 to 2 mm, so a fixed
frame describes it well; an open hand sits about 25 mm away, and the older
threading and pickup captures grip 10 to 25 mm differently.

The controller applies that offset itself, to its own copy of the measured pose
and of the Jacobian, rather than through the robot's `F_T_EE`. That keeps
hardware and simulation identical, leaves no state on the robot, and is why the
preflight insists the robot still reports the bare flange: an offset set in Desk
would be applied twice. Three places have to agree and the preflight checks
them: `tcp` in `config/replay.yaml` (which generates the pose stream and the TCP
tracking metrics), and `tool_offset_xyz` / `tool_offset_rpy` in the hardware and
simulation controller profiles. Zero all three to control the flange again.

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
joints agrees with the robot's `O_T_EE` (both checks are about the flange, since
the controller carries the tool itself); swaps to
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
| `demo_trajs/traj_3_multi_joint5_cap_2p8` | all ten V2 cycles plus return home; 171 saturated joint-5 waypoints capped at 2.800 rad; carries the per-cycle **release flags** `--intervene` needs; dry-run validated at 5x, physical validation pending | one continuous run; use its colocated `homing.yaml`, `--time-scale 5`, `--interactive-pause` or `--intervene`, and `--max-prepared-duration 400` |
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

The output also carries the **release flag** of every cycle it took, resolved
from the source's per-sample `replay_phase` into the artifact's own sample
numbering as `cycle_index`. That is what `--intervene` rejoins at; see
[Hand-guided interventions](#hand-guided-interventions-safe-dagger-on-the-real-rig).
A source with no `replay_phase` field produces no flags, and the generator says
so rather than guessing at one.

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

## Hand-guided demonstration capture

`capture_demo` records a kinesthetic teaching session beside a
`gravity_compensation:=true` bringup, and `extract_demo` turns it into an
artifact this package's own `replay_trajectory` consumes unmodified. The
end-to-end workflow, the preflight, the key map and the filter rationale are in
the repo README's "Hand-guided demonstration capture" section; what follows is
the part specific to this package.

**Why here.** A hand-guided capture is the one job that touches both halves of
this workspace at once: the FR3's 1 kHz state and limit machinery in
`franka_trajectory_replay`, and the RH56's DOF table, coupling and command
overlay in `inspire_hand_driver`. This package is the only one that already
depends on both, and it already owns the artifact schema a capture has to
produce. Putting the tools anywhere else would mean a second package growing
the same two dependencies and a second opinion about that schema.

**Modules.**

| module | holds |
|---|---|
| `hand_presets.py` | loads and validates `config/hand_presets.yaml` — postures, the pinned-DOF table and the jog bindings; every joint name and limit resolved through `inspire_hand_driver.kinematics`, never restated |
| `capture.py` | the recorded topic list and why each one is there, the fail-closed preflight, the keyboard session, the event log and the manifest |
| `extract.py` | bag to artifact: resampling, the zero-phase low-pass, the 19-joint layout, the TCP conversion, and the artifact writer |
| `waypoints.py` | marks to artifact: the operator's `c` presses as a point-to-point trajectory, without reading the bag |
| `launch_files/sim_capture.py` | the MuJoCo rehearsal stack, as an importable module so its four load-bearing choices are testable; `launch/sim_capture.launch.py` is a shim over it |

**What the artifact says about itself.** `metadata.json` carries
`"replay": "hand_guided"`, the session directory it came from, the filter's
kind, cutoff and what it was applied to, and `offset_rad: 0.0` under
`hardware_orientation`. `homing.yaml` is read off the first extracted waypoint,
so a home and its trajectory cannot disagree.

**The two conversions that are silent when wrong**, and are therefore tested:
libfranka packs `O_T_EE` column-major, and
`franka_trajectory_replay.kinematics.matrix_to_quaternion` returns XYZW while
every artifact's `tcp_quat` is WXYZ. Likewise the driver's twelve joint names
map onto the artifact's twelve; `test_extract.py` asserts the driven six against
`trajectory.FORGE_HAND_JOINTS`, which is the table the replay loader itself
reads, so a permuted finger cannot pass.

**Pinned DOF and jogging.** `fixed_open_ratio` in the preset file pins a DOF
for the whole session — currently `thumb_proximal_yaw_joint: 0.0`, the thumb held
in opposition. It is enforced three times over, because a silently unpinned joint
would be invisible in the data: a preset that disagrees with it is a load error,
a jog control that names it is a load error, and `PresetTable.apply_fixed` puts
the pinned value back on every command that leaves the tool. The `jog` block
binds two keys per DOF (`close_key` lowers the open ratio, `open_key` raises it)
and the loader refuses a binding that collides with a preset, with another jog
control, or with the tool's own reserved keys.

**The two-stage loop.** Confirming a demonstration is worth keeping is a
question about the poses the operator *meant* — the ones they stopped at and
pressed `c` on — not about the continuous path their hand happened to take.
Those two questions have very different costs, so they are two tools:

```sh
# 1. capture (lean by default: no 1 kHz FrankaRobotState)
ros2 run inspire_franka_trajectory_replay capture_demo --note "pick the red block"

# 2. the marked poses only - events.jsonl, no bag read at all
ros2 run inspire_franka_trajectory_replay extract_waypoints \
    logs/demo_capture/<stamp>_<name>
ros2 run inspire_franka_trajectory_replay replay_trajectory \
    logs/demo_capture/<stamp>_<name>/waypoints \
    --home logs/demo_capture/<stamp>_<name>/waypoints/homing.yaml --dry-run

# 3. once the waypoints are confirmed, re-capture the motion as training data
ros2 run inspire_franka_trajectory_replay capture_demo --full-state --note "..."
ros2 run inspire_franka_trajectory_replay extract_demo logs/demo_capture/<stamp>_<name>
```

`extract_waypoints` writes the same three files in the same schema
`extract_demo` does, so stage 2 adds no replay code and no new motion path: the
artifact is a dense 15 Hz trajectory like any other and goes through
`prepare()`, the limit guards and the same client. Its `metadata.json` says
`hand_guided_waypoints`, never `hand_guided` — it visits the marked poses by
the shortest joint-space route between them, which is not what was
demonstrated, and nothing downstream may confuse the two.

The motion is quintic ease from waypoint to waypoint (zero velocity and
acceleration at each end) at a deliberately slow 0.35 rad/s peak, a `--dwell`
hold on arrival, and the hand commanded to that waypoint's recorded posture
over the first half of the dwell — so the fingers move once the arm has
stopped, not during transit. `--speed` scales the peak, `--dwell` the hold.

**Lean by default, `--full-state` for a keeper.** `FrankaRobotState` is 3.7 kB
per message at 1 kHz: on a 176 s session that is 623 MB of a 1.4 GB bag, and
about 90% of what `extract_demo` then spends its time on — measured, ~405 s to
deserialize the robot_state alone against ~7 s for the whole lean set. A lean
capture drops it and keeps `/franka/joint_states` at the same 1 kHz, which is
every number the 15 Hz artifact is built from.

What a lean bag gives up, permanently: `tau_ext`, both external wrenches,
`O_T_EE`/`F_T_EE`/`EE_T_K`, the elbow, the collision and contact indicators,
the error flags and the load model. Those matter for training augmentation and
for after-the-fact contact analysis and nothing recovers them later, so a take
you intend to keep wants `--full-state`. `manifest.json` records which kind it
was, and `extract_demo` reads either without being told.

**Two profiles, one of which is not data.** `--profile hardware` is the only
profile whose output is a demonstration. `--profile sim` targets `sim_capture.launch.py`: MuJoCo has no
`FrankaRobotState` at all, so that profile records `/joint_states`, derives the
TCP by forward kinematics, runs on `/clock` so the event log and the bag share
one clock, and asks the arm-safety question differently — on hardware "is
gravity compensation active", in simulation "is the zero-effort controller
active". Not "is the arm unclaimed": the MuJoCo hardware holds unclaimed joints
on their last desired position, so a bare arm is rigid, and treating that as
free was the first version's bug. Both the manifest and the artifact say
`hand_guided_sim`, and `metadata.limitations` lists what is missing.

**What a demonstration does not contain.** There is no arm action. The arm block
of `joint_pos_target` is `joint_pos` itself; the hand block is the preset in
force at that instant, from the operator's keypresses in `events.jsonl`, with
`hand_command_active` marking the samples before the first press. Nothing is
filled in to complete a schema.
