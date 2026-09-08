# franka_repeatability

Measures the TCP repeatability of an FR3 and the joint tracking behaviour of joint impedance
control with inverse kinematics.

The arm is driven to three Cartesian poses in a cycle. At each visit it dwells, the raw 1 kHz
state is averaged over the dwell, and forward kinematics turns the averaged joint positions into
a TCP pose. The spread of those poses across cycles is the repeatability. Alongside that, the
joint position the impedance law was actually applied to is recorded next to the measured
position, which is the tracking error.

## Why this ships its own controller

`franka_example_controllers/JointImpedanceWithIKExampleController` cannot be used for this, for
two reasons that have nothing to do with the control law:

- It has no target input. `compute_new_position()` hard-codes a 0.1 m sinusoid relative to the
  pose the arm happened to be at when the controller activated.
- It never publishes its IK output. The joint positions the impedance law tracks exist only as a
  local variable. They cannot be recovered from `FrankaRobotState.desired_joint_state` either -
  that is libfranka's `q_d` from the motion generator, which is not running in torque mode.

`CartesianTargetImpedanceIKController` keeps the upstream control law byte for byte - the same
stiffness/damping form with coriolis compensation, and the same gains from
`franka_bringup/config/controllers.yaml` - and adds a target input, a quintic ramp towards it, a
torque rate limiter, and the instrumented output.

## Differences from the upstream example, exhaustively

The torque law is token-for-token identical to
`JointImpedanceWithIKExampleController::compute_torque_command`, and `controllers.yaml` carries
the same `k_gains` and `d_gains` as `franka_bringup`. What differs:

| | upstream example | here |
| --- | --- | --- |
| reference fed to the law | raw IK solution, re-solved every 1 kHz cycle | quintic ramp toward a cached target |
| torque rate limit | none | 1.0 Nm/ms, `torque_rate_limit: 0.0` disables it |
| collision thresholds | set on configure | same values, exposed as parameters |
| target input | none, hard-coded sinusoid | `~/target_pose`, `~/target_joint_positions` |
| reference published | no | `~/controller_state` at the update rate |

The reference difference is the point of the fork, and it means the tracking numbers here are
better than upstream's would be: you are measuring the arm following a smooth setpoint, not the
arm chasing KDL's cycle-to-cycle jitter. `ik_mode: per_visit` puts IK back into the loop.

The rate limiter is the only thing that changes the *applied* torque. The analysis reports the
largest torque step actually seen against the limit, so each run states whether it engaged
rather than leaving you to assume.

## Reflexes and collision thresholds

`cartesian_reflex` aborts control when the external force estimate crosses the threshold. The
impedance law's own restoring torque counts towards that, so with low thresholds simply leaning
on the arm - or a fast enough commanded transient - trips it, and franka_hardware's recovery
path takes `ros2_control_node` down with it.

The controller therefore raises the thresholds on configure, using the identical values the
upstream example installs. They are parameters here rather than hard-coded, so if the arm still
reflexes under a deliberate push, raise `upper_force_thresholds_nominal` - understanding that
you are raising the force at which the robot stops protecting itself. `set_collision_behavior:
false` leaves whatever Desk last set.

## Why IK is not in the measurement loop

The default `ik_mode: cached` solves IK once per pose during a warm-up pass and then commands
that exact joint target on every visit. If IK were re-solved on each visit it would be seeded
from a slightly different current configuration each cycle, so the joint target itself would
drift and the run would report the KDL solver's variation on top of the arm's. Set
`ik_mode: per_visit` only when characterising the pose-in path as a whole.

## Bringing it up

Bring up the robot, move_group and the controller. The controller holds position on activation;
nothing moves until the run script commands something:

```bash
ros2 launch franka_repeatability repeatability.launch.py \
    robot_config_file:=/ros2_ws/src/franka_bringup/config/tekken.config.yaml
```

The three poses in `config/repeatability.yaml` are placeholders around the FR3 ready pose. They
assume an empty workspace. Replace them with poses that mean something in your cell - see
*Teaching the poses* below.

## Teaching the poses

Rather than typing coordinates, drive the arm where you want it and capture what it reports.
`capture_pose.py` reads the same frame pair the measurement uses, so what you capture is what
gets measured.

### Just typing the numbers

Nothing requires teaching. `config/repeatability.yaml` takes plain coordinates:

```yaml
poses:
  - name: P1
    position: [0.35, -0.15, 0.50]          # metres, in base_frame (fr3_link0)
    orientation: [0.92388, -0.38268, 0.0, 0.0]   # quaternion x, y, z, w, of ee_link (fr3_link8)
```

`--dry-run` is what makes this safe: it resolves each pose through `compute_ik`, reports the
joint step from where the arm is now, and checks that forward kinematics of the solution lands
back on the pose you asked for. A typo shows up there rather than as arm motion.

For reference, the FR3 ready pose puts `fr3_link8` at `[0.3069, 0.0000, 0.5903]` with
orientation `[0.92388, -0.38268, 0, 0]` - flange pointing down. That is a good anchor to offset
from when picking numbers by hand.

### In simulation, with RViz

`franka_fr3_moveit_config/moveit.launch.py` brings up its own ros2_control stack, move_group and
an RViz with the MotionPlanning display. Under fake hardware the joint states are a loopback of
the commands, so the kinematics are exact even though nothing physical happens:

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
    use_fake_hardware:=true load_gripper:=false ee_id:=none robot_ip:=dont-care
```

**`load_gripper:=false` is not optional here.** That launch file defaults to `true`, and
`load_gripper` feeds both the URDF and the SRDF, which moves the planning group's tip from
`fr3_link8` to `fr3_hand_tcp` - about 10 cm further out. Teach against the wrong tip and every
pose is silently offset by the length of a hand the robot is not wearing.

Drag the orange interactive marker, hit *Plan & Execute*, then capture. `moveit.launch.py` runs
without a namespace, unlike the measurement bringup:

```bash
ros2 run franka_repeatability capture_pose.py --namespace "" --names P1 P2 P3
```

It prompts before each pose, averages a burst of readings, and prints a `poses:` block to paste
into `config/repeatability.yaml`.

What this does and does not tell you: the pose is kinematically reachable and the joint solution
is real. It says nothing about whether your actual cell has a fixture in that spot, and nothing
about the dynamics - fake hardware is a command-to-state loopback, not a physics simulation.

### On the real arm, by hand

More representative for a real cell, and the reason `--source robot_state` exists. Run the
gravity compensation controller, which commands zero torque so libfranka's own gravity
compensation leaves the arm free to push around, while the state broadcaster keeps publishing:

```bash
ros2 launch franka_bringup example.launch.py \
    controller_names:=gravity_compensation_example_controller \
    robot_config_file:=/ros2_ws/src/franka_bringup/config/tekken.config.yaml

ros2 run franka_repeatability capture_pose.py --names P1 P2 P3
```

Push the arm to each pose and press Enter. The tool reports how far it drifted during the
capture, so a steady hold is distinguishable from a wobble. `--source robot_state` reads the
arm's own `O_T_EE` instead of TF if you would rather trust the robot's number than the URDF's.

Either way, run `--dry-run` afterwards: captured poses are reachable by construction, but the
dry run confirms they still resolve through `compute_ik` from wherever the arm currently is, and
that the joint step stays inside `max_joint_step`.

## Moving the arm

Once `repeatability.launch.py` is up, `goto_pose.py` drives the arm to a single target through
the measurement controller - same quintic ramp, same `max_joint_step` guard, same torque rate
limiter as a real run:

```bash
ros2 run franka_repeatability goto_pose.py --ready                    # FR3 ready configuration
ros2 run franka_repeatability goto_pose.py --name P1                  # a pose from the config
ros2 run franka_repeatability goto_pose.py --position 0.35 -0.15 0.50 # a Cartesian target
ros2 run franka_repeatability goto_pose.py --joints 0 -0.785 0 -2.356 0 1.571 0.785
```

Add `--dry-run` to resolve a Cartesian target and print the joint solution and step size without
moving. It blocks until the ramp finishes and prints the joint configuration it arrived at.

The controller takes targets on plain topics, so `ros2 topic pub` works too if you would rather
not use the wrapper:

```bash
ros2 topic pub --once /NS_1/repeatability_ik_controller/target_joint_positions \
    sensor_msgs/msg/JointState '{position: [0, -0.785, 0, -2.356, 0, 1.571, 0.785]}'
```

Two things that will not work:

- **MoveIt's *Plan & Execute* against the real arm while this controller is active.** Execution
  goes through `fr3_arm_controller`, a JointTrajectoryController, and two controllers cannot
  claim the same joint command interfaces at once. Use `goto_pose.py`, or stop one controller
  before starting the other. This is why the RViz teaching flow above runs on fake hardware in
  a separate stack.
- **A target far from where the arm is.** `max_joint_step` rejects anything beyond 1.5 rad per
  joint and says so in the controller log. Get to a sane configuration first with
  `--ready`, or with `franka_bringup`'s `move_to_start_example_controller`.

## Before launching: preflight

`franka_hardware` does not catch libfranka's `ControlException`. If the robot will not accept
control, the exception escapes the control thread and takes the whole `ros2_control_node` down
with a SIGABRT, while the launch stalls on spawners that will not terminate - so the symptom is
a hung launch and a motionless arm, not a readable error. The one line that matters is buried in
the log:

```
libfranka: Move command rejected: command not possible in the current mode ("User stopped")!
terminate called after throwing an instance of 'franka::ControlException'
```

Check first. This is read-only, needs no authentication, and takes no control token:

```bash
ros2 run franka_repeatability preflight.py --host robot.example
```

It verifies operating mode, user stop, guiding button, brakes, FCI activation, power and pending
recoveries, and names the fix for whatever fails. The one that bites is **operating mode**: it
must be `Execution`. `Programming` or `Teach` - which is what holding the enabling device gives
you - produces exactly the "User stopped" rejection above.

## The bringup does not move the arm

Worth stating plainly, because a healthy bringup and a crashed one look similar from the
terminal: `repeatability.launch.py` activates the controller, which then *holds position*. It
will sit there indefinitely. Nothing moves until `goto_pose.py` or `run_repeatability.py` is run
from a second terminal.

To tell a healthy bringup from a dead one:

```bash
ros2 control list_controllers -c /NS_1/controller_manager
```

`repeatability_ik_controller` must be `active`. If the node died it will not answer at all.

## Running the measurement

Check the poses before the first run. `--dry-run` resolves each one through the same
`compute_ik` service the controller uses, reports the joint step from where the arm is now, and
cross-checks the forward kinematics against the requested pose - all without commanding the arm:

```bash
ros2 run franka_repeatability run_repeatability.py --dry-run
```

Then run and analyse:

```bash
ros2 run franka_repeatability run_repeatability.py
ros2 run franka_repeatability analyze_repeatability.py --run ~/franka_repeatability_runs/run_<stamp>
ros2 run franka_repeatability plot_repeatability.py ~/franka_repeatability_runs/run_<stamp>
```

A run of 3 poses x 10 cycles takes about 5 minutes at the default timings.

## What a run produces

```
run_<timestamp>/
├── bag/           rosbag2 of robot_state and controller_state, both at the controller rate
├── robot.urdf     the URDF that was live during the run, captured from robot_state_publisher
├── run.json       poses, joint targets, controller parameters, measurement window timestamps
├── report.md      repeatability and tracking tables
├── report.json    the same numbers, machine readable
├── visits.csv     one row per visit: averaged joints, FK pose, tracking error
└── plots/         figures, once plot_repeatability.py has been run
```

`capture_pose.py` is also useful after the fact to check where the arm actually ended up.

Recording goes through `ros2 bag record` rather than a Python subscriber: at 1 kHz rclpy drops
messages, which would silently bias the very averages the run exists to produce. The analysis
reports the observed sample rate per window so a shortfall is visible rather than assumed away.

## What the report contains

- **Position repeatability** per ISO 9283: `RP = mean distance from the barycentre + 3 sigma`,
  plus per-axis spread and the worst single visit.
- **Orientation repeatability**: angular spread about the mean orientation, where the mean is
  Markley's eigenvector average rather than a componentwise one.
- **A cross-check** of the same statistic computed from the robot's own `O_T_EE`. If the two
  columns disagree, the kinematic model is the suspect, not the arm.
- **Tracking**, split into the steady state during the dwell and the peak during the ramp. A
  joint impedance law is a spring, so a steady-state offset under gravity load is expected. The
  report also expresses that offset at the tool, in millimetres.

## Plotting

`plot_repeatability.py RUN_DIR` writes six figures into `RUN_DIR/plots`:

| figure | what it shows |
| --- | --- |
| `spread` | box plot of the visits per pose: distance from the barycentre, and the same visits resolved onto each axis |
| `drift` | deviation against elapsed time, with the fitted trend - separates a slow walk from real scatter |
| `repeatability` | RP as reported against RP with the trend removed, against the datasheet figure |
| `joint_contribution` | each joint's spread mapped through the Jacobian, so the error budget is split by joint |
| `joint_deviation` | seven subplots: per-joint visit-to-visit deviation of the dwell mean, with sigma and drift per pose |
| `joint_tracking` | seven subplots: the controller's tracking error through the move and the dwell, every visit overlaid |

Only `joint_tracking` needs the bag - it is the one figure whose data is not in the summary
files - and the extraction is cached as an npz, so re-plotting is fast. `--no-bag` skips it,
`--only NAME ...` picks individual figures, `--format png pdf` writes both, and
`--tracking-mode absolute` plots reference against measured instead of the error between them.

A joint whose scatter in `joint_deviation` is dominated by its trend line is drifting rather
than repeating badly; the same distinction is what `drift` and `repeatability` make in
Cartesian space.

## Frames

`ee_link` in `config/repeatability.yaml` must name the same frame as `ik_link_name` in
`config/controllers.yaml`. An empty `ik_link_name` resolves to the MoveIt group tip, which is
`fr3_link8` without a gripper and `fr3_hand_tcp` with one. The dry run checks this and says so
explicitly if the forward kinematics does not land on the requested pose.

## Configuration

`config/controllers.yaml` is passed to `franka.launch.py` through its `controllers_yaml`
argument, so `franka_bringup`'s own copy stays untouched. It has to declare every controller the
bringup spawns, which is why `joint_state_broadcaster` and `franka_robot_state_broadcaster`
appear there too.

Safety-relevant parameters:

- `motion_duration` (5.0 s): the quintic ramp towards a new target. Zero velocity and
  acceleration at both ends, so a distant target is not a torque step.
- `max_joint_step` (1.5 rad): targets further than this from the current configuration are
  rejected. Move the arm to a sensible start with `move_to_start_example_controller` rather than
  raising this.
- `torque_rate_limit` (1.0 Nm/ms): caps how fast the commanded torque may change.
