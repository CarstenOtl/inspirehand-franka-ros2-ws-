# Cartesian impedance trajectory replay: blueprint

Status: implemented 2026-09-11 (section 14); built and tested in the
container, dry-run validated, **not yet run on the arm**. Written 2026-09-11;
the four open decisions were settled the same day (section 13). This is the plan for
replaying 6-DOF Cartesian waypoint trajectories on the FR3 with the torque law
of Franka's own `CartesianImpedanceExampleController` (franka_ros2 v3.5.3, the
pinned submodule), with every feature the joint-space replay already has:
homing, 15 Hz waypoint densification, automatic and explicit time scaling,
interactive pause/resume/abort, coordinated hand streaming, live gain tuning,
dry runs, and the same status handshake the runner and hand thread key on.

The pattern is the one that already worked for joint impedance: the upstream
example's control law is lifted verbatim into a header, the sinusoidal demo
reference is replaced by a waypoint sampler, and the surrounding replay
machinery (goto, trajectory, pause, abort, status) is the same design as
`TrajectoryReplayController`, expressed in pose space.

## 1. The upstream controller, and why it cannot be used as-is

`franka_example_controllers/CartesianImpedanceExampleController`
(`src/franka_ros2/franka_example_controllers/src/fr3/cartesian_impedance_example_controller.cpp`)
claims the seven effort interfaces and computes, every cycle:

```text
error[0:3] = p - p_d                                   (base frame)
q_c        = q_measured, sign-flipped if q_d . q_c < 0 (same hemisphere)
e_q        = q_c^-1 * q_d
error[3:6] = -R * [e_q.x, e_q.y, e_q.z]                (R = measured rotation)

tau_task   = J^T * (-K * error - D * (J * dq))
J^T+       = damped pseudo-inverse of J^T   (SVD, lambda = 0.2)
tau_null   = (I - J^T * J^T+) * (k_n * (q_null - q) - 2 sqrt(k_n) * dq)
tau_d      = tau_task + tau_null + coriolis

K = diag(t_k, t_k, t_k, r_k, r_k, r_k),  D = 2 sqrt(K)   (critically damped)
defaults: t_k = 150 N/m, r_k = 10 Nm/rad, k_n = 20

after commanding, first-order filters with alpha = 0.005 (200 ms time constant at 1 kHz):
K, D, k_n <- 0.005 * target + 0.995 * previous
p_d       <- 0.005 * p_target + 0.995 * p_d
q_d       <- slerp(q_d, q_target, 0.005), hemisphere-corrected, normalised
```

`J` is `getZeroJacobian(kEndEffector)`, `p`/`q` come from the
`cartesian_pose_state` interfaces (`O_T_EE`), coriolis from the robot model.
Gravity is compensated by the robot; there is no torque rate saturation in the
example (franka_hardware has its own).

Three things stop it being used directly:

1. **It always runs its demo motion.** `update()` calls `updateMotionTarget()`
   every cycle, which overwrites `target_pose_buffer_` with a cosine arc around
   the activation pose. The `~/equilibrium_pose` subscription is effectively
   dead in this release; a published pose is used for at most one cycle.
2. **It has no replay machinery.** No goto, no trajectory, no pause/resume,
   no abort, no status, no command-id handshake, no state publishing. The
   runner, the hand thread and the interactive pause all key on those.
3. **Its nullspace target is frozen at activation.** For threading the arm's
   configuration matters as much as the tool pose; the recorded joint path is
   the nullspace reference we actually want.

The Franka submodule stays unchanged, exactly as it did for joint impedance.

## 2. Architecture

```text
 replay_trajectory --arm-controller cartesian-impedance
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ load capture ─> hand overrides ─> joint-space prepare (existing, FR3      │
 │ limit check + auto time scale) ─> FK at 1 kHz ─> pose stream + q_null    │
 │ ─> Cartesian limit check ─> dry-run summary                              │
 │                                                                          │
 │ live: home in JOINT mode (validated path) ─> F_T_EE / O_T_EE preflight   │
 │ ─> STRICT switch to the Cartesian controller ─> Cartesian goto to        │
 │ pose[0] ─> CartesianTrajectory ─> hand thread on the controller's        │
 │ trajectory clock ─> SPACE / q interactive pause (unchanged)              │
 └───────────┬──────────────────────────────────────────────────────────────┘
             │ franka_trajectory_replay_msgs (CartesianGoto, CartesianTrajectory)
             │ std_msgs/Empty (pause, resume, abort)   diagnostic_msgs (status)
             ▼
 CartesianTrajectoryReplayController  (new plugin in franka_trajectory_replay)
   phases idle / goto / trajectory / stopping, clock-rate ramps for pause+abort
   reference sampler: Hermite on position, slerp on orientation, Hermite on q_null
   law: cartesian_impedance.hpp  ==  the example's update() equations
             │ 7 effort interfaces; state: joint pos/vel, cartesian_pose_state,
             │ robot_model, robot_state
             ▼
 franka_hardware (torque control mode, gravity compensation, torque rate limiter)
```

## 3. Frames: what a "6-DOF waypoint" is

The pose the law measures is libfranka's `O_T_EE`: the end effector frame in
the base frame `fr3_link0`, where the end effector is defined by `F_T_EE`, the
flange-to-EE transform the robot holds (set in Desk, or over the
`service_server/set_tcp_frame` service). The Inspire hand is not a Franka end
effector, so on this arm `F_T_EE` is expected to be identity and `O_T_EE` is
the flange `fr3_link8`. That is not guaranteed by anything in this repository;
it is whatever Desk last set, which is why the runner has to verify it (section
7.3).

Waypoint sources, in order of preference:

| source | frame | status |
|---|---|---|
| **FK of the recorded joint waypoints** (`kinematics.flange_transform`, tested against pinocchio on the URDF) | flange in base, `F_T_EE = I` | **default.** Consistent with the joint replay already validated on hardware; also supplies the nullspace target. |
| explicit `ee_pos` / `ee_quat` arrays in a capture (`trajectory_io` already reads them) | must be base-frame, `xyzw` | phase 2, for policies that emit pose targets. No joint data means no nullspace path and no FR3 joint-limit check; needs its own opt-in flag. |
| Forge `tcp_pos` / `tcp_quat` in `replay_data.npz` | environment-relative, TCP = thumb/index fingertip midpoint, `wxyz` | **not a rigid tool point.** Expressed in the flange frame it wanders by 12 mm (`traj_3`) to 57 mm (`traj_2`) and 27 to 64 degrees across one capture, because the midpoint moves with the fingers. It is the recording's task marker, not a controller target. Keep for cross-checking the FK stream, nothing more. |

A `tcp` block already exists in `replay.yaml` (`frame`, `offset_xyz`,
`offset_rpy`); the Cartesian path uses it as the flange-to-target transform the
stream is generated for, and requires the robot's `F_T_EE` to match it.

## 4. The controller: `CartesianTrajectoryReplayController`

New plugin `franka_trajectory_replay/CartesianTrajectoryReplayController` in
`src/franka_trajectory_replay`, alongside the joint controller, not a mode of
it. The two share the phase machine design and the status contract; the
reference type, guards and law are different enough that a second class is
clearer than a third `command_interface` branch.

### 4.1 Interfaces

- **Command:** `<prefix>fr3_joint{1..7}/effort`. franka_hardware therefore
  runs `initializeTorqueInterface()`, as for the joint controller.
- **State:** `fr3_joint{i}/position`, `/velocity`, `/effort` (blocks of seven,
  as the joint controller reads them), the 16 `cartesian_pose_state`
  interfaces through `FrankaCartesianPoseInterface` (state only; its command
  side is never claimed), and `fr3/robot_model` + `fr3/robot_state` through
  `FrankaRobotModel` for the Jacobian and coriolis vector.

Because both replay controllers claim the same seven effort interfaces, a
STRICT `switch_controller` from one to the other keeps franka_hardware in
torque mode: `perform_command_mode_switch` sees `desired == active_mode_`
and returns without `stopRobot()`. That is what makes "home in joint mode,
replay in Cartesian mode" a single session (section 7.2).

### 4.2 The law as a pure function

`include/franka_trajectory_replay/cartesian_impedance.hpp`, header-only and
unit-tested like `joint_impedance.hpp`:

```cpp
struct CartesianImpedanceTerms {  // published, so tracking can be attributed
  Vector7d tau_task, tau_nullspace, tau_coriolis, tau_command;
  Vector6d error;
};

// CartesianImpedanceExampleController::update(), with p_d / q_d / q_null supplied
// by the caller instead of the demo arc and the activation pose. Same error
// construction, same damped pseudo-inverse (lambda 0.2), same nullspace damping.
CartesianImpedanceTerms example_cartesian_impedance(
    const Eigen::Vector3d& p, const Eigen::Quaterniond& q,           // measured O_T_EE
    const Matrix6x7d& jacobian, const Vector7d& coriolis,
    const Vector7d& q_joint, const Vector7d& dq_joint,
    const Eigen::Vector3d& p_d, const Eigen::Quaterniond& q_d,       // filtered reference
    const Vector7d& q_null,
    const Matrix6d& stiffness, const Matrix6d& damping, double nullspace_stiffness);

// The example's post-command filters, also lifted verbatim.
void filter_reference(double alpha, const Eigen::Vector3d& p_target,
                      const Eigen::Quaterniond& q_target,
                      Eigen::Vector3d& p_d, Eigen::Quaterniond& q_d);
```

The RT loop calls exactly these, so the controller's torque is the example's
torque by construction, and the test can replicate the upstream equations
independently and compare over a long sequence (section 9).

### 4.3 Phases and the realtime loop

Same four phases as the joint controller: `idle`, `goto`, `trajectory`,
`stopping`, with the same atomics (`phase`, `active_command_id`,
`processed_command_id`, `completed_command_id`, `elapsed`, `duration`,
`pause_requested`, `paused`, `playback_rate`) so `ReplayClient._wait_command`,
the hand thread's `trajectory_clock()` and `_InteractivePause` need no change.

```text
update(period):
  read q, dq, O_T_EE, J, coriolis
  first cycle after activation: p_target = p_d = p_measured, q_target = q_d = q_measured,
                                q_null = q_measured, tau_prev = 0      (as the example's on_activate)
  apply live gain revision (filtered with target_filter, as the example)
  take a new Command from the RT buffer, if any (goto / trajectory / abort)

  switch phase:
    goto:        s = elapsed / duration; p_target = p0 + (p1 - p0) * quintic(s)
                 q_target = slerp(q0, q1, quintic(s)); q_null = n0 + (n1 - n0) * quintic(s)
                 at s = 1: hold the target until the *filtered* reference is within
                 goto_settle_tolerance (bounded by goto_settle_timeout), then finished
    trajectory:  playback-rate ramp exactly as the joint controller (quintic over
                 pause_ramp_duration, trapezoidal clock integration)
                 sample_trajectory(t): Hermite on p (velocities from the message),
                 slerp between the two bracketing quaternions, Hermite/linear on q_null
                 finished when the clock reaches the end
    stopping:    playback-rate ramp to zero over abort_stop_duration, then hold
                 (the joint controller integrates its own stop; ramping the clock is
                 dimension-agnostic and keeps the reference on the planned path)

  p_d, q_d <- filter_reference(target_filter, p_target, q_target, p_d, q_d)
  terms = example_cartesian_impedance(...)
  tau = saturate_torque_rate(terms.tau_command, tau_prev)   // no-op at torque_rate_limit 0
  write effort interfaces; snapshots; publish state
```

`sample_trajectory` reuses the joint controller's segment-hint search; the
Hermite basis is shared. Orientation between 1 ms samples is slerp; at that
spacing linear interpolation plus normalisation would be indistinguishable,
but slerp costs nothing and needs no argument.

### 4.4 Non-realtime inputs

| topic / service | type | behaviour |
|---|---|---|
| `~/goto` | `franka_trajectory_replay_msgs/CartesianGoto` | pose + nullspace configuration + optional duration. Rejected while busy, outside the workspace box, further than `max_goto_step` (translation or rotation) from the current reference, or non-finite. Duration is at least `goto_min_duration` and stretched so the quintic peaks stay under `goto_max_velocity` (m/s) and `goto_max_angular_velocity` (rad/s). |
| `~/trajectory` | `franka_trajectory_replay_msgs/CartesianTrajectory` | at least two points, strictly increasing times, finite, inside the workspace box, first pose within `max_trajectory_start_error` (`_m` and `_rad`) of the current reference, per-segment linear and angular velocity under `trajectory_velocity_scale` times libfranka's Cartesian limits. `header.frame_id` must equal `base_frame`. |
| `~/pause`, `~/resume`, `~/abort` | `std_msgs/Empty` | identical to the joint controller. |
| `~/set_cartesian_stiffness` | `franka_msgs/srv/SetCartesianStiffness` | kept because the example has it: six diagonal stiffnesses, damping rebuilt as `2 sqrt(k)`. |
| live parameters | `translational_stiffness`, `rotational_stiffness`, `nullspace_stiffness`, `stiffness_scale`, `target_filter` | validated (finite, non-negative), then take effect through the example's own 0.005 filter rather than the joint controller's quintic ramp; the ramp already exists in the law. `stiffness_scale` multiplies the translational and rotational stiffness, not the nullspace. |

### 4.5 Outputs

- `~/status` (`diagnostic_msgs/DiagnosticArray`, `status_rate` Hz): every key
  the joint controller publishes with the same meaning (`phase`, `phase_name`,
  `command_mode` = `cartesian_impedance`, ids, `elapsed`, `duration`,
  `pause_requested`, `paused`, `playback_rate`, `rejections`,
  `last_rejection`, `stiffness_scale_target`), plus
  `translational_stiffness_applied`, `rotational_stiffness_applied`,
  `nullspace_stiffness_applied`, `reference_pose` (`x y z qx qy qz qw`),
  `position_error_m`, `orientation_error_rad`, `tracking_fault`.
- `~/controller_state` (`control_msgs/JointTrajectoryControllerState`, every
  cycle): feedback `q`, `dq`, `tau`; output effort; reference positions =
  nullspace target; the phase and clock in the same `time_from_start` slots
  the joint controller uses, so `dataset.extract_bag` keeps working unchanged.
- `~/cartesian_state` (`franka_trajectory_replay_msgs/CartesianReplayState`,
  every cycle, realtime publisher): unfiltered target, filtered reference,
  measured pose, the six-vector error the law used, the nullspace reference,
  and `tau_task`, `tau_nullspace`, `tau_coriolis`, `tau_command` separately.
  This is what makes a tracking problem attributable without re-deriving FK.

### 4.6 Guards

Acceptance-time (reject, nothing moves): everything in the table above.

Runtime: if `|p - p_d|` exceeds `max_position_error` or the orientation error
angle exceeds `max_orientation_error` during `goto` or `trajectory`, the
controller ramps the clock to zero (the abort path), sets `tracking_fault`
in the status and stays holding. With the example's soft gains a 5 cm error is
only 7.5 N, so this is not a torque guard; it is the backstop for a reference
in the wrong frame, which the preflight in section 7.3 is meant to catch
first.

What a Cartesian controller does not guard: joint position limits. The
nullspace term pulls toward the recorded configuration, and the recorded
configuration passed the FR3 joint limit check, but nothing in the law stops a
joint drifting into its limit if tracking is poor. The status publishes the
smallest joint-limit margin (`joint_limit_margin_rad`) so it is visible; the
robot's own limit reflexes remain the hard stop.

### 4.7 Parameters

| parameter | default | note |
|---|---|---|
| `arm_id`, `arm_prefix` | `fr3`, `""` | as the example |
| `base_frame` | `fr3_link0` | required `frame_id` on trajectories |
| `translational_stiffness`, `rotational_stiffness`, `nullspace_stiffness` | 150, 10, 20 | the example's defaults; live |
| `stiffness_scale` | 1.0 | live, multiplies translational and rotational |
| `target_filter` | 0.005 | the example's `filter_params_`, a first-order low-pass on the target pose with a 200 ms time constant; 1.0 bypasses it (the replay reference is already C1 at 1 kHz). Live, so it can be compared without a relaunch |
| `nullspace_target` | `trajectory` | `trajectory`: follow the waypoints' configuration; `fixed`: freeze at activation, exactly the example |
| `coriolis_compensation` | true | the example adds coriolis |
| `torque_rate_limit` | 0.0 | as the validated joint profile; franka_hardware's limiter stays |
| `goto_max_velocity`, `goto_max_angular_velocity`, `goto_min_duration` | 0.10 m/s, 0.50 rad/s, 3.0 s | |
| `max_goto_step_m`, `max_goto_step_rad` | 0.30, 1.00 | sanity guard, not a speed limit |
| `max_trajectory_start_error_m`, `_rad` | 0.002, 0.01 | the joint profile uses 0.001 rad |
| `goto_settle_tolerance_m`, `_rad`, `goto_settle_timeout` | 0.0005, 0.002, 2.0 s | filtered reference convergence before `idle` |
| `workspace_min`, `workspace_max` | `[-0.855, -0.855, -0.36]`, `[0.855, 0.855, 1.19]` | base-frame box; FR3 reach, tighten per setup |
| `trajectory_velocity_scale` | 1.0 | fraction of libfranka's Cartesian velocity limits allowed between points |
| `max_position_error`, `max_orientation_error` | 0.08 m, 0.35 rad | runtime tracking fault |
| `pause_ramp_duration`, `abort_stop_duration`, `status_rate` | 0.5, 0.5, 50 | as the joint controller |
| `set_collision_behavior` + thresholds | false | as the validated joint profile (the upstream example launch would set them) |

## 5. Messages: `franka_trajectory_replay_msgs`

A small ament_cmake interface package. Standard messages were considered and
rejected: `trajectory_msgs/MultiDOFJointTrajectory` carries transforms and
twists but has nowhere to put the nullspace configuration, and
`geometry_msgs/PoseStamped` for goto has neither. Keeping the controller
library free of `rosidl` generation is the ROS convention and keeps the C++
package's build as it is.

```text
CartesianWaypoint.msg
  geometry_msgs/Pose pose                  # O_T_EE target in base_frame
  geometry_msgs/Twist twist                # base-frame linear/angular velocity of the reference;
                                           # used for Hermite interpolation of position only,
                                           # the law has no feedforward
  float64[] nullspace_positions            # 7 joint values, or empty to hold the previous target
  builtin_interfaces/Duration time_from_start

CartesianTrajectory.msg
  std_msgs/Header header                   # frame_id == controller base_frame
  CartesianWaypoint[] points

CartesianGoto.msg
  geometry_msgs/Pose pose
  float64[] nullspace_positions            # 7 or empty
  float64 duration                         # 0 => derived from the goto limits

CartesianReplayState.msg
  std_msgs/Header header
  int32 phase
  float64 trajectory_time
  geometry_msgs/Pose target                # sampler output, unfiltered
  geometry_msgs/Pose reference             # after the example's filter; what the law used
  geometry_msgs/Pose measured              # O_T_EE
  float64[6] error                         # [p - p_d ; -R * vec(q_c^-1 q_d)]
  float64[7] nullspace_reference
  float64[7] tau_task
  float64[7] tau_nullspace
  float64[7] tau_coriolis
  float64[7] tau_command
```

## 6. Python: preparing a pose stream

New module `franka_trajectory_replay/cartesian.py`. The joint-space pipeline is
kept as the first stage on purpose: it is the part validated on hardware, it is
where the FR3 velocity/acceleration/jerk limits and the automatic time scaling
live, and a pose stream produced by forward kinematics of a limit-checked joint
stream describes a motion the arm can make.

```text
prepare(joint capture)             existing: spline, holds, lead-in/out, FR3 limits, auto scale
   -> Prepared (t, q, qd, ...) at 1 kHz
from_joint_stream(Prepared, tool)  NEW: batched FK of every sample, sign-continuous quaternions,
   -> PreparedCartesian             linear velocity by finite differences, angular velocity from
                                    quaternion differences, q_null = q
check_cartesian_limits(...)        NEW: libfranka FR3 Cartesian limits with margins
to_message(...)                    NEW: CartesianTrajectory at send_rate, twists filled
```

```python
@dataclass
class PreparedCartesian:
    t: np.ndarray          # (M,)
    p: np.ndarray          # (M, 3) metres, base frame
    quat: np.ndarray       # (M, 4) xyzw, sign-continuous along the stream
    v: np.ndarray          # (M, 3) linear velocity
    w: np.ndarray          # (M, 3) angular velocity, base frame
    q_null: np.ndarray     # (M, 7) nullspace configuration, or None
    joint: Prepared        # the joint stream it was derived from, or None
    tool: np.ndarray       # 4x4 flange-to-target transform the stream assumes
    params: dict
    report: dict
```

Cartesian limits, from `/opt/libfranka/include/franka/rate_limiting.h` in the
container: translation 3.0 m/s, 9.0 m/s^2, 4500 m/s^3; rotation 2.5 rad/s,
17 rad/s^2, 8500 rad/s^3 (each minus libfranka's packet-loss tolerance).
In torque mode the robot does not enforce these on a reference; they are
sanity bounds, and the joint-space check already bounds the same motion more
tightly (joint 7 alone may turn the flange at 4 rad/s under the joint limits),
so the velocity margin defaults to 1.0, the datasheet limit itself, with 0.5
on acceleration and jerk in `replay.yaml`.

FK is `kinematics.flange_transform` composed with the `tcp` transform, batched
over the stream so a 160 000-sample dry run stays under a few seconds.
`flange_poses` already returns sign-continuous quaternions.

Phase 2, `from_pose_capture(t, p, quat, q_null=None)`: cubic spline on
position, `scipy.spatial.transform.RotationSpline` on orientation, the same
holds and lead-ins, Cartesian limit check only. Requires an explicit
`--pose-source` flag because without joint data there is no FR3 joint-limit
check and no nullspace path.

## 7. Runner integration

### 7.1 What is reused unchanged

`--arm-controller cartesian-impedance` is a third choice next to
`joint-impedance` and `position-jtc` in `replay.py`. Everything that is not
the arm's command type stays as it is: capture loading, `--cycle`/`--segment`,
`--env`, the homing YAML and `--max-home-delta`, `--time-scale`,
`--max-prepared-duration`, `--finger-flexion-scale`,
`--close-support-fingers`, hand validation and the 50 Hz hand thread keyed on
the controller's `elapsed`, `--interactive-pause` (SPACE/q) through the same
`pause`/`resume`/`abort` topics and status keys, `--dry-run`, the two motion
prompts and `--yes`, abort on Ctrl-C.

### 7.2 Sequence

1. Load and prepare in joint space exactly as today; derive the pose stream;
   check Cartesian limits; print both summaries. `--dry-run` stops here.
2. Activate `trajectory_replay_controller` (the joint controller, validated
   profile), home the arm with its `goto`, home the hand. Homing stays in
   joint space because the policy's home is a joint configuration and a
   Cartesian controller with 20 Nm/rad of nullspace stiffness cannot promise
   to arrive in it.
3. Preflight (7.3).
4. STRICT switch: deactivate the joint controller, activate
   `cartesian_trajectory_replay_controller`. No libfranka mode change; the
   Cartesian controller's first cycle holds the measured pose, and its
   torque at rest is the coriolis term, i.e. zero.
5. Cartesian `goto` to the stream's first pose with the stream's first
   nullspace configuration. This is a millimetre-scale move that absorbs the
   at-rest joint tracking offset of the compliant joint controller; the
   trajectory's start-error check then passes by construction. The goto
   waits for the filtered reference to settle before reporting idle.
6. Send the `CartesianTrajectory`, start the hand thread on
   `on_accept`, run the interactive pause loop if requested.
7. On completion or abort the arm holds its last reference in Cartesian mode.
   The session's next replay repeats from step 2; the switch back to the joint
   controller is the same STRICT swap.

### 7.3 Preflight, before the switch

- **`F_T_EE` matches the tool the stream was generated for.** Read one
  `franka_robot_state_broadcaster/robot_state` message; `f_t_ee` must equal
  the configured `tcp` transform within 1e-4 m / 1e-4 rad (identity for the
  flange). Otherwise refuse with the measured transform in the message and
  point at `set_tcp_frame`. A mismatch here would put every waypoint in the
  wrong place by the offset; the controller's tracking fault is the second
  line of defence, not the first.
- **FK agrees with the robot.** At the home pose, `flange_transform(q_measured)
  @ tool` versus `o_t_ee`: refuse above 3 mm or 0.5 deg. This catches a stale
  DH table or a wrong `tool`, independently of the first check.
- **The Cartesian controller is loaded, its parameters are the expected
  profile** (`command_mode` in status, `nullspace_target`, `target_filter`),
  mirroring `CoordinatedReplayClient.ensure_active`'s refusal of the wrong
  controller type.

### 7.4 Client

`franka_trajectory_replay/cartesian_replay_client.py`: `CartesianReplayClient`
subclasses `ReplayClient`, replaces `goto` and `send_trajectory` with the
Cartesian message types, keeps `_wait_command`, `pause`, `resume`, `abort`,
`ensure_active`, `list_controllers`, and adds `switch_from(joint_controller)`
and `robot_state_once()` for the preflight. In `replay.py`,
`CartesianCoordinatedReplayClient(HandReplayMixin, CartesianReplayClient)`
composes the hand link exactly as `CoordinatedReplayClient` does, and the
runner holds one joint client for homing and one Cartesian client for replay
on the same executor.

### 7.5 CLI additions

| flag | meaning |
|---|---|
| `--arm-controller cartesian-impedance` | select the path |
| `--cartesian-velocity-margin`, `--cartesian-acceleration-margin`, `--cartesian-jerk-margin` | override `replay.yaml` margins for the Cartesian check |
| `--stiffness-scale N` | request a live `stiffness_scale` on the Cartesian controller before replay (existing `ros2 param set` keeps working) |
| `--pose-source {fk,capture}` | phase 2: take poses from `ee_pos`/`ee_quat` instead of FK |

Rejected combinations: `--allow-unsafe-simulation` (hardware controller),
`--no-arm` with `--interactive-pause` (as today).

## 8. Launch and configuration

- `config/controllers_cartesian_impedance.yaml` in
  `inspire_franka_trajectory_replay`: declares `joint_state_broadcaster`,
  `franka_robot_state_broadcaster`, `trajectory_replay_controller` with the
  validated joint-impedance parameters copied verbatim, and
  `cartesian_trajectory_replay_controller` with section 4.7. Both are effort
  controllers on the same interfaces; only one is active at a time.
- `replay.launch.py` gains `arm_controller:=joint-impedance|cartesian-impedance`
  (default unchanged). With `cartesian-impedance` it selects the yaml above and
  adds a second spawner, `cartesian_trajectory_replay_controller --inactive`,
  so the joint controller still comes up active and holding, as today.
- `replay.yaml` gains a `cartesian:` block: margins, `send_rate`, and the
  preflight tolerances. The `tcp` block is the tool transform.
- `recording.topics` gains `cartesian_trajectory_replay_controller/cartesian_state`.

## 9. Tests

C++ (`ament_add_gmock`):

- `test_cartesian_impedance.cpp`: (a) zero task and nullspace torque when the
  reference equals the measurement; (b) equivalence: re-implement the upstream
  `update()` equations inline in the test, including the SVD damped
  pseudo-inverse, the hemisphere flip and the post-command filters, and
  compare against `example_cartesian_impedance` + `filter_reference` over
  5000 cycles of a synthetic Jacobian, moving targets and joint states, to
  1e-12; (c) hemisphere invariance, `q` and `-q` give the same error;
  (d) sign: a small rotation error about base z produces a torque that
  reduces it.
- `test_load_controller.cpp`: load the second plugin through the controller
  manager.

Python (`ament_add_pytest_test`):

- `test_cartesian.py`: FK stream continuity (no quaternion sign flips, angular
  velocity by finite differences agrees with `flange_jacobian @ qd`); the
  Cartesian limit check flags a synthetic violent stream and passes a gentle
  one; message packing round-trips; the `F_T_EE` preflight refuses a
  non-identity tool when the stream assumed the flange.
- `test_replay.py` additions: the client type check for the Cartesian
  controller, CLI validation, the prepare-then-FK path produces a
  `PreparedCartesian` with the joint stream attached.

## 10. Hardware validation ladder

Each step is a session on the real arm with the lab's access procedure in
force; do not skip a rung.

0. Build both packages, `colcon test`, then a dry run of a current-orientation
   artifact: `replay_trajectory apps/traj_replay/demo_trajs/traj_2_cycle3
   --home .../homing.yaml --arm-controller cartesian-impedance --time-scale 5
   --dry-run` prints both summaries and the tool transform assumed.
1. Launch with `arm_controller:=cartesian-impedance`. Both controllers listed,
   joint active, Cartesian inactive. `ros2 param get` the Cartesian gains.
2. **Hold test.** Runner `--hold-only` (or the switch by hand): the arm must
   not move on the swap; status idle; `position_error_m` under 1 mm. Push the
   flange gently: 150 N/m is about 1.5 N per centimetre, so it should give and
   return. This is also the `F_T_EE` and FK preflight's first real run.
3. **Goto test.** `~/goto` +2 cm in base z with the current nullspace
   configuration; watch settle time (about 1 s from the 200 ms filter) and
   overshoot in `cartesian_state`.
4. **Arm-only replay** of `traj_2_cycle3` at `--time-scale 5 --no-hand`.
   Record a bag; compare `cartesian_state` reference versus measured, and the
   joint-space `controller_state` versus the joint-impedance run of the same
   artifact.
5. **Coordinated replay** at 5x with `--interactive-pause`: SPACE holds both
   devices, resume continues from the same clock, q aborts to a hold.
6. Reduce `--time-scale`, and only then raise gains with `stiffness_scale`
   in small steps, as the joint README describes. Expect the example's 10
   Nm/rad rotational and 20 nullspace stiffness to be too soft for threading;
   the live parameters exist so that tuning does not need a relaunch.

## 11. Out of scope for the first iteration

- **MuJoCo** was out of scope in the plan and is now supported (section 14):
  neither `inspire_franka_sim`'s ros2_control block nor Franka's own
  `franka_mujoco_hardware` export `robot_model`, `robot_state` or
  `cartesian_pose_state`, so the controller gained `model_source: dh` (pose
  and Jacobian from its built-in FR3 DH model, no coriolis) and the sim a
  gravity-free scene in place of libfranka's gravity compensation.
- **Analysis of `cartesian_state`.** `dataset.extract_bag` gets a
  `cartesian_state` role and `analysis` a direct pose-error metric in a later
  step; until then `tcp_metrics` (FK of `controller_state`) and
  `rs_O_T_EE` cover the same ground.
- **Pose-only captures** (`--pose-source capture`), section 6.
- **Elbow.** `k_elbow_activated_` stays false as in the example.
- **Torque-level feedforward.** The example has none; adding inertia-shaped
  feedforward would make it a different controller.

## 12. File plan

| path | change |
|---|---|
| `src/franka_trajectory_replay_msgs/` | new: `package.xml`, `CMakeLists.txt`, the four messages in section 5 |
| `src/franka_trajectory_replay/include/franka_trajectory_replay/cartesian_impedance.hpp` | new: the law and filters, header-only |
| `src/franka_trajectory_replay/include/franka_trajectory_replay/cartesian_trajectory_replay_controller.hpp` | new |
| `src/franka_trajectory_replay/src/cartesian_trajectory_replay_controller.cpp` | new |
| `src/franka_trajectory_replay/franka_trajectory_replay.xml` | add the plugin |
| `src/franka_trajectory_replay/CMakeLists.txt`, `package.xml` | add sources, `franka_trajectory_replay_msgs`, `geometry_msgs`, tests |
| `src/franka_trajectory_replay/test/test_cartesian_impedance.cpp` | new |
| `src/franka_trajectory_replay/test/test_load_controller.cpp` | add a load case |
| `src/franka_trajectory_replay/franka_trajectory_replay/cartesian.py` | new: `PreparedCartesian`, `from_joint_stream`, `check_cartesian_limits`, `to_message`, Cartesian limits |
| `src/franka_trajectory_replay/franka_trajectory_replay/cartesian_replay_client.py` | new |
| `src/franka_trajectory_replay/franka_trajectory_replay/runconfig.py` | `cartesian:` defaults |
| `src/franka_trajectory_replay/test/python/test_cartesian.py` | new |
| `src/inspire_franka_trajectory_replay/config/controllers_cartesian_impedance.yaml` | new |
| `src/inspire_franka_trajectory_replay/config/replay.yaml` | `cartesian:` block, recording topic |
| `src/inspire_franka_trajectory_replay/launch/replay.launch.py` | `arm_controller` argument, second spawner |
| `src/inspire_franka_trajectory_replay/inspire_franka_trajectory_replay/replay.py` | third `--arm-controller`, `CartesianCoordinatedReplayClient`, the sequence in 7.2, preflight |
| `src/inspire_franka_trajectory_replay/test/test_replay.py` | additions |
| `src/inspire_franka_trajectory_replay/README.md`, `README.md` | usage and status |

Suggested order: msgs package and the header with its equivalence test first (the
law is testable with no robot), then the controller and its load test, then
`cartesian.py` with its tests and the dry-run path in the runner, then the
launch/config and the live sequence, then the hardware ladder.

## 13. Decisions (settled 2026-09-11)

1. **Target frame: flange, `F_T_EE = I`.** Confirmed. The stream is generated
   for the flange and the preflight refuses a robot whose `F_T_EE` is not
   identity. If a fingertip TCP is ever wanted, `tcp` in `replay.yaml` changes
   and `set_tcp_frame` is called at launch; the preflight then enforces that
   instead.
2. **Nullspace target: `trajectory`.** The nullspace term pulls toward the
   recorded joint configuration at every sample, so the elbow follows the
   capture. `fixed` (freeze at activation, exactly the example) stays
   available as a parameter. Not to be confused with the recorded `tcp_pos`
   data, which is the fingertip midpoint and plays no part in either option
   (section 3).
3. **Keep the example's target filter for the first hardware runs.** The
   example low-passes its target pose with alpha 0.005 every cycle, a 200 ms
   time constant. Keeping it means the first runs use the controller exactly
   as Franka ships it; the cost is that the arm follows the waypoint stream
   about 200 ms late, which at a 5x time scale is small. Because the reference
   the sampler produces is already smooth, the filter can later be bypassed
   by setting the live parameter `target_filter` to 1.0 and the same artifact
   replayed for comparison, without a relaunch.
4. **Separate `franka_trajectory_replay_msgs` package.** Confirmed.

## 14. Implementation notes (2026-09-11)

Everything in sections 4 to 9 is implemented as written, with these
deviations and additions:

- **Nullspace reference is filtered too.** The controller applies the example's
  `target_filter` to the nullspace configuration as well as to the pose, so the
  two references move together instead of the nullspace leading by 200 ms.
- **`goto_max_nullspace_velocity`** (0.5 rad/s) was added; a goto's duration
  also accounts for the nullspace step, not only the pose step.
- **Twists in `CartesianWaypoint`:** when every point's twist is exactly zero the
  trajectory is taken to carry no velocity information and positions are
  interpolated linearly; otherwise the linear part drives cubic Hermite.
- **Cartesian velocity margin defaults to 1.0**, acceleration and jerk to 0.5
  (section 6). The synthetic test stream showed that a 0.5 velocity margin
  would reject motions the joint-space check accepts.
- **`--stiffness-scale`** goes through the controller's `set_parameters`
  service; the margin overrides are `--cartesian-velocity-margin`,
  `--cartesian-acceleration-margin`, `--cartesian-jerk-margin`.
- **Two clients on one executor.** `CoordinatedReplayClient` (joint controller
  plus hand) homes and drives the hand; `CartesianReplayClient` takes the arm
  over after the preflight. The joint client's goto/trajectory publisher
  creation became an overridable hook so the Cartesian client can carry its
  own message types; no other change to the validated joint client.
- **Collision-behavior parameters** moved into `collision_behavior.hpp`, used
  by the Cartesian controller; the joint controller keeps its own copy for
  now.
- **Tests.** C++: the law against the upstream transcription (6 cases), plugin
  load and sampler math (5), joint law (2). Python: 32 in
  `franka_trajectory_replay` (batched FK, pose stream, limits, frame checks,
  message packing) and 12 new runner tests in
  `inspire_franka_trajectory_replay` (pose stream, margins, yaml profile,
  client refusals, a full dry run through `main`, CLI refusals).
- **Dry run.** `traj_2_cycle3` at `--time-scale 5`: 41 s stream, path 0.78 m,
  2.89 rad of rotation, peak angular velocity 0.22 rad/s (9 % of the limit).

- **MuJoCo simulation mode.** `model_source: dh` on the controller
  (`fr3_kinematics.hpp`, tested against the Python DH model and as the exact
  derivative of its own forward kinematics), the gravity-free
  `inspire_franka_flange_torque_scene.xml`, `controllers_sim_impedance.yaml`
  (both hardware profiles verbatim, plus the model source), and
  `sim_replay.launch.py arm_controller:=joint-impedance|cartesian-impedance`.
  The runner skips the robot-frame preflight when the controller reports the
  DH model and says so. `apps/traj_replay/sim_replay_smoke.sh` runs the whole
  thing unattended and refuses to start next to a live controller manager.
- **First unattended sim run** (`traj_2_cycle3`, 2x): joint homing over a
  1.80 rad step, the swap, a 1.5 cm / 0.05 rad settle goto, trajectory
  accepted with zero start error, then the controller stopped itself on the
  orientation tracking fault after 8 s. Cause: the Menagerie FR3 carries
  1.137 Nm of joint friction with nothing compensating it below the controller,
  which the example's 10 Nm/rad rotational stiffness cannot hold. The sim
  profile therefore runs at `stiffness_scale: 4.0` (hardware keeps 1.0); the
  client now prints the peak tracking error after every goto and trajectory.
  A completed run at that scale is still to be confirmed.

Not done: the hardware ladder (section 10), `cartesian_state` in the bag
analysis, pose-only captures.
