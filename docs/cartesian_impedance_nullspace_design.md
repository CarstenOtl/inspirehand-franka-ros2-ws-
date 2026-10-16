# The nullspace term: why the example impedance law fails the distilled policy

Companion to [cartesian_replay_blueprint.md](cartesian_replay_blueprint.md), which designed
`CartesianImpedanceController` around Franka's
`CartesianImpedanceExampleController` law lifted verbatim. That was the right call for replaying
recorded waypoints. It is the wrong law for running a policy distilled in Isaac Lab, and this
document explains why, in enough detail to argue with.

## Status, stated plainly

Fixed and committed (branch `controller/forge-osc-law`):

- the nullspace projector is now exact rather than damped (`nullspace_damping_lambda: 0.0`);
- the rotation error carries the full angle rather than its half-angle sine
  (`rotation_error: axis_angle`);
- the nullspace term is mass-weighted and dynamically consistent (`mass_weighted_nullspace:
  true`, A1b, 2026-10-01);
- 2026-10-06: the Coriolis term is off (`coriolis_compensation: false`), the summed torque is
  clamped at Forge's +-100 Nm (`torque_limit: 100.0`), and the policy interface changed so the
  controller does what `_apply_action` does at every physics substep: it takes the **preclipped
  goal** and the **live grasp frame on the flange** (`~/policy_goal`, `PolicyGoal`), re-clips
  the goal against the measured grasp pose every 1 kHz cycle (`policy_clip_position_m`,
  `policy_clip_orientation_rad`, in the training world frame `policy_clip_frame_yaw`), and runs
  the law at that live frame instead of the frozen midpoint (sections 9.1 to 9.4 below, items
  A3 and A4 in the todo).

All are controller parameters that **default to the example's behaviour**, so trajectory replay
and every non-policy profile are untouched. Only the two policy profiles opt in, and
`run_hardware_rollout` refuses a controller that does not.

What the controller still cannot match, stated in section 9: the arm armature (unreadable
here), the hand's mass and geometry (one RH56, two derivations; a bench question), the
finger-velocity part of the damping term, and the 1 kHz vs 120 Hz discretisation.

And the task is not solved. The nullspace and rotation fixes moved the ros-sim rollout from
"never leaves the `policy` phase" to "reaches `release_started`, then fails `return_to_reset`".
That is the same phase the MuJoCo-only loop fails in on the training plant, so the controller is
no longer the thing in the way — but nothing here makes a nut get threaded ten times. The
2026-10-06 changes are verified in unit tests and by driving the rebuilt controller with
hand-made goals in ros-sim (tool latched, target leading by exactly one clip, 66 mm closed to
within 2 mm, hold after the stream stopped, bad goals rejected); no policy rollout has run on
them yet.

## 1. The two laws

Training, `forge_ultra/tasks/utils/control.py::compute_dof_torque`, ported verbatim to
[forge_osc.py:360](../apps/policy_rollout/policy_rollout/forge_osc.py#L360):

```text
e        = [ p_d - p ; axis_angle(q_d q^-1) ]           # world frame, full angle
F        = K e - D v                                     # v = absolute grasp-frame twist
tau_task = J^T F
M_task   = (J M^-1 J^T)^-1
Jbar     = M_task J M^-1
u_null   = kp_null (q_0 - q) - kd_null qdot
tau_null = (I - J^T Jbar) M u_null                       # M-weighted, dynamically consistent
tau      = clamp(tau_task + tau_null, +-100)
```

The example law, [cartesian_impedance.hpp:126](../src/franka_forge_cartesian_impedance/include/franka_forge_cartesian_impedance/cartesian_impedance.hpp#L126):

```text
e        = [ p - p_d ; -R vec(q^-1 q_d) ]                # sin(theta/2) * axis
tau_task = J^T (-K e - D J qdot)
tau_null = (I - J^T pinv_lambda(J^T)) (k_n (q_0 - q) - 2 sqrt(k_n) qdot)
tau      = tau_task + tau_null + tau_coriolis
```

with `pinv_lambda` the SVD pseudo-inverse regularised by `lambda = 0.2`, no mass weighting, a
Coriolis term training never had, and no clamp (a torque **rate** limit instead). Upstream of
the law, training also re-decodes and re-clips the target every substep at the live grasp frame
(section 9); the ROS stack used to clip once per tick in Python and anchor the law at a frozen
midpoint.

Gains, nullspace target and nullspace gains were already identical: 565 N/m, 28 Nm/rad,
`D = 2 sqrt(K)`, `k_n = 10`, `kd_null = 2 sqrt(10) = 6.3246`, target = the M24 reset joints
(`hardware.py` publishes them as `nullspace_positions` every tick). So this was never a tuning
difference. Two structural terms differed, and both are below.

## 2. Why the nullspace term is where the laws can differ at all

The FR3 has seven joints; a 6-DOF pose task constrains six directions. For any commanded pose
there is a one-parameter family of joint torques that realise it, and a torque controller must
pick one. The pose command does not determine the choice — which is exactly why "same inputs,
same outputs, same impedance structure" does not imply "same controller".

Both laws spend that freedom the same way: a joint-space spring pulling the arm toward a
reference posture. The term exists so the elbow does not wander, and it is supposed to be
**invisible to the tool**. Whether it actually is depends entirely on the projection, and the
projection is what differs.

## 3. The free parameter is not small

A full 20 mm clipped policy step commands `565 N/m * 0.02 m = 11.3 N` through the task spring.
That is the entire authority the policy has over the tool in one tick. At the posture the arm
reaches descending to the nut (joint 5 +1.1 rad, joint 7 -0.9 rad from the nullspace target) the
example law's nullspace term pushes **12.5 N and 8.0 Nm** on the grasp frame.

So the term that is supposed to be invisible outweighs the policy's whole command. Not a
correction — a competing controller.

## 4. Why `lambda = 0.2` stops being a projection

### 4.1 It is the damping, not the missing mass weighting

`pinv_lambda` replaces each `1/sigma` with `sigma / (sigma^2 + lambda^2)`. For
`sigma >> lambda` those agree; for `sigma ~ lambda` the damped version is roughly half, and the
"projector" `I - J^T pinv_lambda(J^T)` stops annihilating `range(J^T)`.

The FR3 at the threading posture has three small singular values, and `lambda = 0.2` sits right
on top of them:

| singular value of `J^T` | fraction of that direction removed | fraction left in `tau_null` |
|---|---|---|
| 1.971 | 99.0% | 1.0% |
| 1.887 | 98.9% | 1.1% |
| 1.212 | 97.4% | 2.6% |
| 0.306 | 70.0% | **30.0%** |
| 0.291 | 68.0% | **32.0%** |
| 0.213 | 53.1% | **46.9%** |

The stiff directions are projected out properly. In the three compliant wrist directions, a
third to a half of the joint spring survives as a tool wrench. Those are the directions the
hand descends and twists in.

### 4.2 Two different senses of "does not disturb the task"

This is the part worth getting right, because the obvious fix is not the training one.

- **Static leak**: the component of `tau_null` lying in `range(J^T)` — the part some external
  wrench could have produced. Killed by the *exact orthogonal* projector
  `I - J^T pinv(J^T)`, which is what `lambda = 0` gives.
- **Dynamic leak**: the task-space acceleration the torque actually causes, `J M^-1 tau_null`,
  equivalently the dynamically consistent wrench `Jbar tau_null`. Killed only by the
  **M-weighted** projection, because
  `J M^-1 (I - J^T Jbar) = J M^-1 - (J M^-1 J^T)(J M^-1 J^T)^-1 J M^-1 = 0` identically.

They are not the same invariant, and each projector kills one:

| projector | norm of `tau_null` | static leak | dynamic leak |
|---|---|---|---|
| example, `lambda = 0.2`, unweighted | 4.24 Nm | 4.20 Nm | 12.5 N / 8.0 Nm |
| exact orthogonal, unweighted (**committed**) | 0.58 Nm | **0.00 Nm** | 0.40 N |
| Forge, M-weighted dynamically consistent | 0.07 Nm | 0.03 Nm | **0.00 N** |

(joint 5 +1.1, joint 7 -0.9 rad off the nullspace target, training scene, M24 reset)

An arm accelerates, so the invariant that matters for the tool's motion is the dynamic one.
Orthogonality to `range(J^T)` is the wrong quantity to zero — it just happens to be very nearly
right here, which is why `lambda = 0` alone recovers 97% of the fix. The last 0.40 N is the mass
weighting, and that is A1b.

Note the units: `Jbar tau` carries force in its first three components and torque in its last
three. Do not take a norm across all six.

## 5. Why a leak is fatal for *this* policy specifically

A 12 N uncommanded wrench would be a nuisance for a servo with integral action. For this
checkpoint it is disqualifying, for three compounding reasons.

**It produces a standing pose offset.** Statically, the tool settles where the task spring
balances the leak:

| wrist offset | leak | deflection |
|---|---|---|
| joint 5 +0.5 rad | 3.19 N / 1.99 Nm | 5.6 mm / 4.1 deg |
| joint 5 +1.1, joint 7 -0.9 rad | 12.49 N / 7.98 Nm | **22.1 mm / 16.3 deg** |

(a static estimate — the real equilibrium involves the coupled dynamics — but the scale is the
point: one full policy step is 20 mm and 0.097 rad)

**Nothing corrects it.** The student is a progress- and phase-conditioned trajectory player. It
does observe proprioception (`q`, `qdot`, the previous filtered action), so it is not strictly
open loop, but it has no integral term and — more importantly — it was distilled on a plant
where this offset does not exist. A 22 mm standing error is off its training distribution, so
there is no reason to expect a corrective response, and the recorded rollouts show none.

**It grows with the motion.** The leak scales with the posture error, and descending to the nut
*is* the posture excursion. So it is not a bias that could be trimmed at the reset pose: it ramps
up precisely during the phase that was failing. This also explains why zeroing MuJoCo joint
friction did not help — friction is 1.14 / 0.76 Nm at those joints, the same order as the leak,
so removing it could not overcome a term that grows faster.

## 6. The second defect: `sin(theta/2)` instead of `theta`

The example's orientation error is `-R vec(q^-1 q_d)`. Since
`R vec(q^-1 q_d) = vec(q_d q^-1)` and `vec` of a unit quaternion is `sin(theta/2) * axis`, the
rotational spring sees the **same axis** but the half-angle sine where training uses the full
angle. Effective rotational stiffness is therefore 14 Nm/rad, not 28, while the damping stays
`2 sqrt(28)` on the full angular velocity: orientation is twice as soft and over-damped.

Yaw is what threads the nut, so this is not cosmetic. It is also the larger of the two errors at
the reset pose, where the nullspace error is zero by construction and the rotation term is the
*entire* 2.87 Nm discrepancy.

`RotationErrorForm::kAxisAngle`
([cartesian_impedance.hpp:71](../src/franka_forge_cartesian_impedance/include/franka_forge_cartesian_impedance/cartesian_impedance.hpp#L71))
builds the rotation block from `Eigen::AngleAxisd(q^-1 q_d)` instead, matching
`forge_osc.get_pose_error(rot_error_type="axis_angle")`.

## 7. What the two laws already shared — do not "fix" these

Verified identical; changing any of them moves away from training:

- task-space gains 565 / 28, damping `2 sqrt(K)`, no dead zone in nominal replay;
- the damping acts on the **absolute** grasp-frame twist, not on an error velocity — the
  example's `-D (J qdot)` is already what Forge computes;
- nullspace gains and target, as above;
- **no gravity term.** The Isaac asset sets `disable_gravity=True`, and libfranka compensates
  gravity internally on hardware. These agree by accident of construction, not by design, but
  they agree.
- the implicit-actuator joint PD layer is provably an identity and must **not** be ported:
  `q_ref = q + tau/Kp`, `dq_ref = qdot`, so `Kp(q_ref - q) + Kd(0) = tau` at the same state. It
  exists because the distillation action space is a joint PD command, not because it changes any
  dynamics.
- the `+-100 Nm` clamp never binds: the FR3's own limits are 87 Nm (joints 1-4) and 12 Nm
  (joints 5-7).

## 8. What the change bought

Residual against `forge_osc.compute_dof_torque`, same state and target, on the training scene:

| wrist offset | norm of `tau_forge` | example law | `lambda = 0` | + axis-angle |
|---|---|---|---|---|
| at the reset pose | 8.66 Nm | 2.87 | 2.87 | **0.00** |
| joint 5 +0.5 rad | 13.72 Nm | 3.11 | 2.85 | **0.35** |
| joint 5 +1.1, joint 7 -0.9 rad | 18.20 Nm | 5.29 | 3.02 | **0.52** |

Exact at the reset pose. Within 2.9% at the descent posture, the remainder being the mass
weighting.

In the loop (`ros_sim-20260930-143512-493634`): `release_started` at a 63.3 deg grasp-turn proxy,
321 steps at 15 Hz, 0 missed policy deadlines, 1 clipped target, no watchdog stop — then
`return_failed`. The first ros-sim run to leave the `policy` phase at all. The release-proxy sign
fix was needed in the same breath: with the proxy inverted the cycle could never advance, so no
controller change could have produced a readable result.

## 9. What was still different, and what remains

Done since the first version of this document:

1. **A1b, the mass weighting** (2026-10-01). `forge_nullspace_torque` projects with
   `I - J^T Jbar^T` and weights the joint PD by `M`. On hardware `M` is
   `FrankaRobotModel::getMassMatrix()`; on the `dh` path it is a pinocchio `crba` over the robot
   description with the hand locked (`arm_mass_model.hpp`), which matched MuJoCo's arm block to
   8e-04 kg m^2. The **arm armature** is still open: `robot.py` sets `armature` for the hand
   joints only, the arm keeps whatever `fr3_no_hand.usd` carries, and that file is a git-lfs
   pointer in both checkouts on this machine (no `git-lfs` installed to pull it). Both profiles
   run `arm_armature: [0 x7]`; MuJoCo uses 0.195 (joints 1-4) and 0.074 (joints 5-7). Read it on
   the training machine.
2. **Coriolis** (2026-10-06). `control.py` has no Coriolis term; the hardware profile used to add
   one (0.08 Nm mean at training's 0.17 rad/s, 0.69 / 1.9 Nm at the 0.5 rad/s of release and
   return). Both profiles now set `coriolis_compensation: false`, the runner refuses `true`, and
   `activate_policy_controller.py` reports it. Gravity stays libfranka's, as it was Isaac's
   `disable_gravity`.
3. **The compliance centre** (2026-10-06, A3). Every `PolicyGoal` carries `tool_in_flange`, the
   live grasp frame on the flange (`hardware.grasp_in_flange`, from the training hand
   kinematics and the reset z transport), and the controller measures its pose, clips and
   differentiates the law at that frame. The static `tool_offset_xyz` is now only the
   activation hold. `retarget_grasp_pose_to_controlled_pose` is gone; the goal is the grasp
   goal. `report.json`'s `grasp_controlled_offset_m` is the largest distance between the
   controller's measured point and the live grasp over the run, which should be sampling skew.
4. **The decode rate** (2026-10-06, A4). The runner sends `_apply_action` steps (0) and (1), the
   preclipped `bolt_tip + a*0.05` goal (`hardware.policy_goal_base`); the controller runs step
   (2), the component-wise 20 mm position clip and the per-Euler-angle 0.097 rad clip with
   Isaac's yaw wrap (`forge_clip_target`, `isaac_euler_xyz`, `isaac_quat_from_euler_xyz`),
   against the measured grasp pose on every cycle, in the training world frame
   (`policy_clip_frame_yaw: pi`, since the Euler clip is not frame-invariant in general). The
   port is pinned against `forge_osc.decode_action_target` on six random cases to 1e-12
   (`ForgeDecode.clip_matches_decode_action_target`). The `max_policy_step` guard does not
   apply to goals: the clip bounds the executed step from the *measured* pose every cycle,
   which is the stronger property; the workspace box bounds the goal.
5. **The +-100 Nm clamp** (2026-10-06). `torque_limit: 100.0`, `clamp_torque`, before the rate
   limiter. It never binds on an FR3; it is there so the law reads line for line.

Still different, and why each is left:

- **`torque_rate_limit: 1.0`** (1 Nm per 1 ms cycle) is libfranka's own `kMaxTorqueRate`
  (1000 Nm/s), which `franka_hardware` enforces on the commanded torque anyway. The training
  plant had no such bound. Keep it; with an exact OSC it is the most likely thing left to
  distort a step response, so look at `tau_command` against `tau_task + tau_nullspace` in
  `cartesian_state` when a step looks slow.
- **The damping velocity.** Training damps `0.5 (v_thumb_tip + v_index_tip)`, PhysX body
  velocities that include the finger joints' own motion; the controller damps `J qdot` at the
  grasp point, the arm-only part. The difference is the finger contribution, up to
  `2 sqrt(565) = 47.5 Ns/m` times the midpoint's finger-driven speed. Matching it needs hand
  joint velocities (C1), which the driver does not publish.
- **The rate.** Training re-decodes at 120 Hz in lockstep with the 15 Hz policy; the controller
  re-clips at 1 kHz against commands that arrive asynchronously. Not a law difference.
- **The hand's mass and geometry** (D1) and **the end-effector load in Desk** (section 10):
  bench questions, not code.

## 10. Hardware notes

- `getMassMatrix()` reflects the load configured in **Franka Desk's active end-effector
  profile**; the launch never calls `setLoad`
  ([inspire_franka.launch.py:20](../src/inspire_franka_bringup/launch/inspire_franka.launch.py#L20)).
  Training's `M` carries the Inspire hand. If Desk is not configured with the hand as the load,
  the matrix fed to the projector is not the one training used. This is a Desk setting, not code.
- Expect the elbow to settle visibly further than under the example law. The redundant direction
  is now genuinely free; that is the change working, not a fault.
- `apps/operations/activate_policy_controller.py` puts the arm under this controller with no
  policy in the loop, refuses to run unless the law is the ForgeUltra one, holds the activation
  pose, and hands the arm back on exit. Use it before trusting the student with the arm. It
  holds about the static `tool_offset_xyz`; the live grasp frame only arrives with policy goals.
- After a policy goal the controlled frame stays where that goal put it, through a watchdog
  stop, an abort and the idle hold, until deactivation. That is deliberate: switching back to
  the static tool at the moment the stream stops would move the measured point and make the
  hold jump.

## 11. Reproducing every number here

All measurements are on the training scene at the M24 reset, inside the `inspire_franka`
container with `MUJOCO_GL=egl`, through `policy_rollout.mujoco_threading_env.ThreadingScene` and
`policy_rollout.forge_osc`. The scripts that produced the tables are session scratch
(`nullspace.py`, `nullspace_decompose.py`, `law_residual.py`, `coriolis.py`, `design_numbers.py`);
each is a few dozen lines and rebuilding one from the formulas above is faster than recovering it.

Unit tests pin the two behaviours:

- `ForgeCartesianImpedance.axis_angle_error_carries_the_full_angle`
  ([test_cartesian_impedance.cpp:301](../src/franka_forge_cartesian_impedance/test/test_cartesian_impedance.cpp#L301))
  checks `|rot| == theta` against the example's `sin(theta/2)` at five angles;
- `ForgeCartesianImpedance.lambda_zero_keeps_the_nullspace_torque_out_of_the_task_space`
  ([test_cartesian_impedance.cpp:319](../src/franka_forge_cartesian_impedance/test/test_cartesian_impedance.cpp#L319))
  checks the static leak is below 1e-9 at `lambda = 0` and at least 20x larger at 0.2;
- `ExampleCartesianImpedance.matches_upstream_update_over_a_moving_sequence` still passes, which
  is what guarantees the defaults are byte-identical to upstream.

## 12. File map

| what | where |
|---|---|
| the law, both variants | [cartesian_impedance.hpp](../src/franka_forge_cartesian_impedance/include/franka_forge_cartesian_impedance/cartesian_impedance.hpp) — rotation error, `forge_nullspace_torque`, `clamp_torque`, and the decode (`forge_clip_target` and the Isaac Euler helpers) |
| the policy goal interface | `PolicyGoal.msg` in `franka_forge_cartesian_impedance_msgs`; `policy_goal_callback` and the `kPolicy` sampler branch in [cartesian_impedance_controller.cpp](../src/franka_forge_cartesian_impedance/src/cartesian_impedance_controller.cpp) |
| parameter declaration | `on_init` in the same file |
| status fields | `publish_status` in the same file (`nullspace_damping_lambda`, `rotation_error`, `coriolis_compensation`, `torque_limit`, `policy_clip_*`, `tool_in_flange`, `policy_goal_active`) |
| hardware profile | [controllers_policy.yaml](../src/inspire_franka_trajectory_replay/config/controllers_policy.yaml) |
| ros-sim profile | [controllers_sim_policy.yaml](../src/inspire_franka_trajectory_replay/config/controllers_sim_policy.yaml) |
| the rollout's parameter check | `run_hardware_rollout` in [hardware.py](../apps/policy_rollout/policy_rollout/hardware.py) |
| training law, ported | [forge_osc.py:360](../apps/policy_rollout/policy_rollout/forge_osc.py#L360) |
| training law, original | `forgeUltra/forge_ultra/tasks/utils/control.py` |
| bench activation | [activate_policy_controller.py](../apps/operations/activate_policy_controller.py) |
| remaining work | [policy_rollout_training_gap_todo.md](policy_rollout_training_gap_todo.md) |

## 13. Symbols

| symbol | meaning |
|---|---|
| `J` | geometric Jacobian, base frame, at the compliance point (6x7) |
| `M` | arm mass matrix (7x7), the arm block of the full articulation |
| `M_task` | `(J M^-1 J^T)^-1`, the task-space mass |
| `Jbar` | `M_task J M^-1`, the dynamically consistent inverse |
| `K`, `D` | task stiffness and damping, `diag(565,565,565,28,28,28)` and `2 sqrt(K)` |
| `k_n`, `kd_null` | nullspace stiffness 10 and damping 6.3246 |
| `q_0` | nullspace target: the M24 reset joints |
| `lambda` | the example's pseudo-inverse regularisation, 0.2 upstream, 0.0 in the policy profiles |
