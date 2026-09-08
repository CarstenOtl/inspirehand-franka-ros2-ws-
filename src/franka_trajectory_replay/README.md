# franka_trajectory_replay

Replays joint-space policy trajectories captured in Isaac Sim on an FR3, under the robot's
default joint impedance control, and measures how well the arm followed them: joint tracking
(commanded vs measured, per joint, with lag), TCP accuracy (forward kinematics of both, and
the robot's own `O_T_EE`), and every proprioceptive signal libfranka offers, recorded at 1 kHz
so a bad run can be debugged down to motor-side encoders and reflex flags.

Everything the script does on the arm can first be done in MuJoCo - both offline (a pure
Python replay through the same rate limiter and the same emulated internal controller) and
as the *identical ROS stack* with `franka_mujoco_hardware` standing in for `franka_hardware`.

## The sequence

```
ros2 run franka_trajectory_replay replay_trajectory.py ~/trajs/policy.npz
```

1. loads the capture, resamples it to 1 kHz, checks it against the FR3 limits (slowing it
   down if it has to), connects, and records **HOME** = wherever the arm is right now;
2. **Enter** -> starts `ros2 bag record`, quintic ramp to the first point of the trajectory;
3. **Enter** -> makes sure the replay controller is the active one, plays the trajectory,
   returns to its first point;
4. **Enter** -> ramp back to HOME, recording stops;
5. extracts the bag, writes `report.md` / `report.json` and the figures.

`--yes` skips the prompts (simulation), `--dry-run` stops after step 1, `--cycles N` plays the
trajectory N times, `--no-record` / `--no-analyze` do what they say. Ctrl-C sends an abort:
the controller decelerates over 0.5 s and holds.

## Control modes

| `command_interface` | what runs the joints | when |
|---|---|---|
| `position` (default, `config/controllers.yaml`) | the position command interface -> libfranka's joint position motion generator with the **robot-internal joint impedance controller** (`ControllerMode::kJointImpedance`). The commanded vs measured comparison is then the robot's `q_d` vs `q`, and the arm reports it itself in `desired_joint_state`. | the "default joint impedance mode" of the arm; what a Desk-programmed motion uses |
| `effort` (`config/controllers_effort.yaml`) | the effort interface with the joint impedance law of `JointImpedanceWithIKExampleController` (`k (q_d - q) - d dq + coriolis`, gains 600/.../50), plus a torque rate limiter | when you want the ROS-side impedance law instead of the robot's, or to compare the two |

Pick the mode with `controllers_yaml:=` on the launch; the controller reports which one it is
in on its status topic and the report records it.

### What the controller guards

`franka_hardware` passes position commands to libfranka **unfiltered and unlimited** (both
flags default off in `robot.hpp`), and the FR3 reflexes on a velocity, acceleration or jerk
discontinuity. So:

- a trajectory is **rejected before anything moves** if any segment exceeds the position,
  velocity (position-dependent, the libfranka formula) or acceleration limits, or if it does
  not start where the arm's command currently is (`max_trajectory_start_error`, 0.05 rad);
- the accepted stream is interpolated with cubic Hermite splines (C1) between the points the
  script sends, and then passed through a re-implementation of libfranka's `limitRate` for
  joint positions (`rate_limit: true`). It should never engage on a prepared stream; the status
  topic counts every cycle it did and the report prints the count;
- point-to-point moves (to the first point, back home) are one synchronous quintic ramp on all
  seven joints, at least `goto_min_duration` (5 s) long and stretched so that the peak velocity
  stays under `goto_max_velocity` (0.5 rad/s) and the peak acceleration under
  `goto_max_acceleration` (1 rad/s^2): 1.5 rad takes 5.6 s, 3 rad 11 s. The arm never jumps.
  `max_joint_step` (3 rad) is a sanity guard against a trajectory from another part of the
  workspace, not a speed limit.

## Trajectory files

`prepare_trajectory.py` (and the replay script, which does the same thing) accept:

- the Isaac `replay_data.npz` export: `joint_pos` of shape (steps, environments, dofs) plus
  `tcp_pos` / `tcp_quat`; `--env N` picks the environment (default 0). The `metadata.json`
  next to it supplies `dt`, `joint_names` and `arm_joint_ids`, so the seven FR3 columns are
  picked out of the 19-dof arm+hand vector by name. `--key joint_pos_target` replays the policy
  targets instead of the measured positions;
- the per-episode `joint_trajectories/envN_segmentM.csv` exports (`time_s`, `pos_<joint>`,
  `vel_<joint>`, `cmd_<joint>`); `--csv-prefix cmd` replays the targets;
- `.npz` with `joint_pos_arm` / `q` / `joint_pos` / `qpos` (N x 7) and a time base: `dt`,
  `rate`, or per-sample `t`/`time`; optional `arm_joint_names`, `ee_pos`, `ee_quat`, `qd`;
- `.npy` holding an (N x 7) array (pass `--rate`), an (N x 8) array with a time column, or a
  pickled dict with the keys above.

**Replay one episode, not the whole record.** `replay_data.npz` concatenates the episodes of an
environment; at every reset the joints jump (1.6 rad in `~/franka_trajectory_01`, samples 80
and 754), and the preparation would slow the whole thing down by more than 10x to make that
jump feasible. The `envN_segmentM.csv` files are the episodes.

Joint names are matched to the FR3 by their index (`fr3_joint3`, `panda_joint3`, `joint_3`
all work; a hand joint such as `thumb_joint_1` cannot steal an arm slot). Names that do not
look like a Franka's are **refused** - the capture in `~/trajs/forge_tg2_pickplace` is a 7-dof
humanoid arm, and replaying it on the FR3 would be a mistake the loader should catch - unless
you pass `--joint-map` or `--assume-order`. Values above 2 pi are refused as "not radians"
unless `--degrees`.

### From 15 Hz waypoints to the 1 kHz command stream

The FCI cycle is 1 kHz: libfranka exchanges one state packet and one command per millisecond,
`franka_hardware` blocks on that packet, and the controller manager runs at 1000 Hz. A 15 Hz
policy therefore needs about 66 command samples between two of its waypoints, and the FR3
checks every consecutive pair of commands against its acceleration (10 rad/s^2) and jerk
(5000 rad/s^3) limits. A plain straight line between waypoints has a velocity step at each
waypoint, which the arm reads as an acceleration of dv / 1 ms and reflexes on.

Preparation (`config/replay.yaml`, section `prepare`) builds the stream as
`hold | lead-in | capture | lead-out | hold`:

- `interpolation: cubic` (default): a natural cubic spline through the waypoints;
- `interpolation: linear`: straight lines between the waypoints, each corner rounded into a
  parabolic blend of `blend_time` (40 ms default; the straight parts between the blends stay
  exactly on the line while `blend_time` is below the 66 ms waypoint spacing). Being straight
  between waypoints costs acceleration at the corners, so this mode needs a stronger slow-down:
  for `env0_segment1` cubic needs x1.13, linear x3.3 with 40 ms blends and x2.1 with 66 ms;
- the policy is normally already moving at its first recorded sample, so the lead-in
  accelerates smoothly from rest into that velocity (and the lead-out brakes from the last
  one). The stream starts a little before the capture's first point; that start point is what
  the goto ramp aims at;
- optional zero-phase low-pass (`cutoff_hz`), then velocity / acceleration / jerk by finite
  differences of the final stream against the FR3 limits with margins (80 % of the velocity,
  50 % of the acceleration and jerk limits). `auto_scale` slows the trajectory down uniformly
  until it fits; the summary printed before every run says by how much.

```
ros2 run franka_trajectory_replay prepare_trajectory.py \
    ~/franka_trajectory_01/data/traj_1/joint_trajectories/env0_segment1.csv [--interpolation linear]
```

writes `env0_segment1_prepared.npz` and a preview figure without touching a robot.

## Simulation first

### Offline, in MuJoCo

```
ros2 run franka_trajectory_replay simulate_trajectory.py \
    ~/franka_trajectory_01/data/traj_1/joint_trajectories/env0_segment1.csv [--view] [--realtime]
```

Runs the full sequence (ramp in, trajectory, return, ramp home) through the same rate limiter
and the same emulated internal joint impedance controller as `franka_mujoco_hardware`, and
writes a run directory with exactly the layout of a real run, so the analysis and plots are
the same code. The exit code is non-zero if the command stream would have raised a libfranka
motion generator error or if any geom of the arm touched anything (self-collision, floor).

### The ROS stack, with MuJoCo as the robot

```
ros2 launch franka_trajectory_replay sim.launch.py            # instead of replay.launch.py
ros2 run franka_trajectory_replay replay_trajectory.py \
    ~/franka_trajectory_01/data/traj_1/joint_trajectories/env0_segment1.csv --yes
```

Same namespace (`NS_1`), same controller, same topics, same script. `franka_mujoco_hardware`
checks every position command against the motion generator limits and, with
`reflex_on_violation:=true` (default), stops the hardware exactly where the real arm would
reflex. What is not there: `franka_robot_state_broadcaster` (no `robot_state` interface), so
the `rs_*` signals are absent and the report's health section is empty.

Plain fake hardware (a loopback, no physics) also works, for testing the ROS side alone:

```
ros2 launch franka_trajectory_replay replay.launch.py \
    robot_config_file:=/ros2_ws/src/franka_trajectory_replay/config/fake.config.yaml
```

### What simulation can and cannot rule out

Can: limit violations that would reflex the arm, joint-limit excursions, self-collisions and
floor contact, a trajectory in the wrong joint order / unit / rate, mistakes in the run logic
(the script, controller switching, recording). Cannot: the real tracking error - the internal
controller emulation (`K = 3000 3000 3000 2500 2500 2000 2000`, libfranka's default joint
impedance, with hand-picked damping) is a plausibility model, not an identified one.

## On the arm

```
ros2 run franka_trajectory_replay preflight.py --host 172.16.0.2  # mode, brakes, FCI, user stop

ros2 launch franka_trajectory_replay replay.launch.py \
    robot_config_file:=/ros2_ws/src/franka_bringup/config/tekken.config.yaml

ros2 run franka_trajectory_replay replay_trajectory.py ~/trajs/policy.npz
```

The launch brings up `franka.launch.py` with this package's `controllers.yaml`, spawns the
broadcasters and the replay controller. **The controller holds position on activation; nothing
moves until the script commands it.** If the arm is somewhere silly, jog it first with
`move_to_start_example_controller` (declared in the same yaml); the script deactivates it
when it activates the replay controller.

Note that `set_collision_behavior: true` raises the collision thresholds to the values the
upstream examples use, because the impedance law's own restoring torque can otherwise trip a
`cartesian_reflex`.

## What a run produces

```
run_<stamp>/
├── prepared.npz   the exact command stream, plus the source samples and the source ee pose
├── robot.urdf     the URDF that was live (real runs)
├── bag/           rosbag2: controller_state + status (1 kHz), robot_state (1 kHz), joint_states
├── data.npz       everything above extracted into arrays (see dataset.py for the layout)
├── run.json       config, controller parameters, HOME, every step with its timestamps
├── report.md/json metrics
└── plots/
```

### Metrics (`report.md`)

Per segment (ramp in, trajectory, return, ramp home): per joint RMS / max / mean / final
error and the lag (the delay of the measurement behind the reference that minimises the RMS
error); TCP position error mean / RMS / max / final and per axis, orientation error mean /
max, both from forward kinematics of the reference and of the measurement; a cross-check of
that FK against the robot's `O_T_EE`; and how close the command stream came to the limits.
Then sampling statistics (observed rates, largest gap), and the robot health section:
`control_command_success_rate`, robot modes seen, every error flag that was ever set,
contact / collision indicator counts, peak external torque and wrench, peak `tau_J` as a
fraction of the limit, motor-vs-joint deflection (`theta - q`), and the rate-limiter count.

The TCP frame is the flange (`fr3_link8`); `tcp.offset_xyz/rpy` in `replay.yaml` adds a tool.
If the capture carried Isaac's own `ee_pos`, the report also compares it with FK of the
source joints: a constant offset is the base pose of the robot in the Isaac scene, a residual
means a different tool frame or kinematics.

### Figures (`plot_replay.py RUN_DIR`, `--only`, `--format png pdf`)

| figure | content |
|---|---|
| `joints` | reference vs measured position, seven panels, phases shaded |
| `joint_errors` | tracking error per joint [mrad] |
| `velocities` | reference vs measured velocity |
| `torques` | `tau_J`, `tau_J_d`, `tau_ext_hat_filtered`, commanded torque in effort mode |
| `tcp_path` | 3-D path of reference, measurement and `O_T_EE`, plus per axis |
| `tcp_error` | position error per axis and norm, orientation error |
| `tcp_contributions` | Jacobian decomposition of the TCP error into the seven joints: stacked shares over time, RMS per joint, and signed bars at the largest deviations |
| `external_wrench` | `O_F_ext_hat_K` |
| `motor_vs_joint` | `theta - q`, the drive-side elasticity |
| `robot_health` | success rate, robot mode, contact / collision flags, error flags timeline |
| `sampling` | sample interval histograms of every recorded topic |
| `command_limits` | commanded velocity and acceleration as a percentage of the FR3 limits |

`plot_replay.py real_run --compare sim_run` overlays the trajectory-segment errors of several
runs in `compare.png`.

## Topics of the controller

| topic | type | direction |
|---|---|---|
| `~/goto` | `sensor_msgs/JointState` | in: ramp to a joint target |
| `~/trajectory` | `trajectory_msgs/JointTrajectory` | in: the stream (positions, velocities, `time_from_start`) |
| `~/abort` | `std_msgs/Empty` | in: decelerate and hold |
| `~/controller_state` | `control_msgs/JointTrajectoryControllerState` | out, every update: reference, feedback, error, output; `reference.time_from_start` is the phase clock, `output.time_from_start` the phase id |
| `~/status` | `diagnostic_msgs/DiagnosticArray` | out, 50 Hz: phase, command ids, progress, rate-limiter counts, last rejection |

## Tests

```
colcon build --packages-select franka_trajectory_replay franka_mujoco_hardware --cmake-args -DBUILD_TESTING=ON
./build/franka_trajectory_replay/franka_trajectory_replay_test_load      # loads the plugin, checks the math
python3 -m pytest src/franka_trajectory_replay/test/python              # loader, preparation, DH vs URDF, analysis, figures
```
