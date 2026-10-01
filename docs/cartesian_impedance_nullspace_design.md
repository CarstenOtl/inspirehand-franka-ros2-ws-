# The nullspace term: why the example impedance law fails the distilled policy

Companion to [cartesian_replay_blueprint.md](cartesian_replay_blueprint.md), which designed
`CartesianTrajectoryReplayController` around Franka's
`CartesianImpedanceExampleController` law lifted verbatim. That was the right call for replaying
recorded waypoints. It is the wrong law for running a policy distilled in Isaac Lab, and this
document explains why, in enough detail to argue with.

## Status, stated plainly

Fixed and committed (branch `controller/forge-osc-law`):

- the nullspace projector is now exact rather than damped (`nullspace_damping_lambda: 0.0`);
- the rotation error carries the full angle rather than its half-angle sine
  (`rotation_error: axis_angle`).

Both are controller parameters that **default to the example's behaviour**, so trajectory replay
and every non-policy profile are untouched. Only the two policy profiles opt in.

Not yet done, and worth being precise about: the committed projector is the **exact orthogonal**
one, not ForgeUltra's **mass-weighted dynamically consistent** one. Those are different (section
4.2), and the gap between them is 0.40 N at the descent posture. That is item A1b in
[policy_rollout_training_gap_todo.md](policy_rollout_training_gap_todo.md).

And the task is not solved. The change moved the ros-sim rollout from "never leaves the `policy`
phase" to "reaches `release_started` at a 63.3 deg turn, then fails `return_to_reset`". That is
the same phase the MuJoCo-only loop fails in on the training plant, so the controller is no
longer the thing in the way — but nothing here makes a nut get threaded ten times.

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

The example law, [cartesian_impedance.hpp:126](../src/franka_trajectory_replay/include/franka_trajectory_replay/cartesian_impedance.hpp#L126):

```text
e        = [ p - p_d ; -R vec(q^-1 q_d) ]                # sin(theta/2) * axis
tau_task = J^T (-K e - D J qdot)
tau_null = (I - J^T pinv_lambda(J^T)) (k_n (q_0 - q) - 2 sqrt(k_n) qdot)
tau      = tau_task + tau_null + tau_coriolis
```

with `pinv_lambda` the SVD pseudo-inverse regularised by `lambda = 0.2`, no mass weighting, and
no clamp (a torque **rate** limit instead).

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
([cartesian_impedance.hpp:71](../src/franka_trajectory_replay/include/franka_trajectory_replay/cartesian_impedance.hpp#L71))
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

## 9. Still different from training

In the order worth doing them:

1. **A1b, the mass weighting.** On hardware the matrix already exists:
   `franka_semantic_components::FrankaRobotModel::getMassMatrix()`. For the `dh` path that
   ros-sim uses there is no mass model; pinocchio is available in the container (C++ config under
   `/opt/ros/jazzy`) and `get_robot_description()` hands the controller the URDF at configure
   time, so a `pinocchio::Model` built once plus `crba` per cycle is allocation-free and
   realtime-safe. Use the **full tree**, not a KDL chain: Isaac's `arm_mass_matrix` is the arm
   block of the whole articulation and so carries the hand's inertia through the finger joints.
   Open question, recorded rather than guessed: whether the arm armature belongs on the diagonal.
   `assets/fr3_inspirehand/robot.py` sets `armature` for hand joints only, so the arm keeps
   whatever the USD carries, and the USD is a git-lfs pointer in this checkout. MuJoCo uses 0.195
   (joints 1-4) and 0.074 (joints 5-7).
2. **Coriolis.** `control.py` has no Coriolis term; the hardware policy profile sets
   `coriolis_compensation: true`, adding one training never had. Measured on the training scene:
   0.08 Nm mean / 0.21 Nm peak at training's 0.17 rad/s, but 0.69 / 1.9 Nm at 0.5 rad/s, which is
   the release and return phases. One line.
3. **The compliance centre.** Training applies the wrench at the live thumb/index midpoint;
   hardware uses that midpoint frozen at the threading grip, 3-23 mm off. The cheap fix is to
   send the live midpoint — `hardware.py` already computes it — in the policy command and let the
   controller shift the Jacobian to it instead of using the static `tool_offset_xyz`.
4. **The decode rate.** Python decodes `bolt_tip + a*0.05` and clips once per 15 Hz tick; Isaac
   re-decodes and re-clips against the *live* grasp frame every 120 Hz substep, 20 mm / 0.097 rad
   each, so the target creeps across the tick instead of arriving at once. Matching this means
   sending the 9-D action and the z-transport instead of a pose, and giving the controller the
   bolt tip and the grasp-frame construction. It is the only item here that changes the interface
   between the policy runtime and the controller.
5. **`torque_rate_limit: 1.0`** (1 Nm per 1 ms cycle) is a ROS-side safety the training plant
   never had. Keep it, but log when it binds — with an exact OSC it is the most likely thing left
   to distort a step response.

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
  pose, and hands the arm back on exit. Use it before trusting the student with the arm.

## 11. Reproducing every number here

All measurements are on the training scene at the M24 reset, inside the `inspire_franka`
container with `MUJOCO_GL=egl`, through `policy_rollout.mujoco_threading_env.ThreadingScene` and
`policy_rollout.forge_osc`. The scripts that produced the tables are session scratch
(`nullspace.py`, `nullspace_decompose.py`, `law_residual.py`, `coriolis.py`, `design_numbers.py`);
each is a few dozen lines and rebuilding one from the formulas above is faster than recovering it.

Unit tests pin the two behaviours:

- `ForgeCartesianImpedance.axis_angle_error_carries_the_full_angle`
  ([test_cartesian_impedance.cpp:301](../src/franka_trajectory_replay/test/test_cartesian_impedance.cpp#L301))
  checks `|rot| == theta` against the example's `sin(theta/2)` at five angles;
- `ForgeCartesianImpedance.lambda_zero_keeps_the_nullspace_torque_out_of_the_task_space`
  ([test_cartesian_impedance.cpp:319](../src/franka_trajectory_replay/test/test_cartesian_impedance.cpp#L319))
  checks the static leak is below 1e-9 at `lambda = 0` and at least 20x larger at 0.2;
- `ExampleCartesianImpedance.matches_upstream_update_over_a_moving_sequence` still passes, which
  is what guarantees the defaults are byte-identical to upstream.

## 12. File map

| what | where |
|---|---|
| the law, both variants | [cartesian_impedance.hpp](../src/franka_trajectory_replay/include/franka_trajectory_replay/cartesian_impedance.hpp) — rotation error L71, nullspace L113-130 |
| parameter declaration | [cartesian_trajectory_replay_controller.cpp:1064](../src/franka_trajectory_replay/src/cartesian_trajectory_replay_controller.cpp#L1064) |
| status fields | same file, L680 (`nullspace_damping_lambda`, `rotation_error`) |
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
