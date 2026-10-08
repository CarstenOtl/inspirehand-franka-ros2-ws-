# Closing the training gap: policy rollout vs Isaac Lab

Checklist of changes that bring the ROS rollout stack (`run_policy_rollout.py hardware` and
`ros-sim`) in line with the setup the student was trained and evaluated in
(`forgeUltra/distillation`, Isaac Lab), so that the only remaining difference is physics that
has not been measured yet.

Scope: checkpoint `sequential_threading_cycle10_hybrid_teacher_d415_20ep`, task
`vanilla_threading.yaml`, cyclic-student-owned lifecycle. Measurements are from the
2026-09-30 crosscheck (see "Evidence sources" at the end). Each item states what was measured,
what to change, and a test that decides when it is done. Tick the box when the "done when"
test passes; append a dated note under the item.

Priority tags: **blocking** = the stack cannot reproduce training without it;
**should** = measurable effect, do after the blockers; **minor** = small or sim-only;
**bench** = needs a physical measurement first.

Working rules for agents:

- Do not change the checkpoint, the DP3 camera contract, or the 29-D proprio layout. Those are
  verified identical (last section).
- One item per commit where possible. Cite the "done when" evidence in the commit message.
- Runs happen in the `inspire_franka` container (`/root/develop_ws` is this workspace) with
  `MUJOCO_GL=egl`. Check `ros2 node list` before starting anything; the graph is shared.

---

## Status

2026-09-30, after A1a + A2 + B1 (`ros_sim-20260930-143512-493634`): the rollout reached
`release_started` at a 63.3 deg grasp-turn proxy, the first ros-sim run ever to leave the
`policy` phase. 321 steps at 15 Hz, 0 missed policy deadlines, 1 clipped target, no controller
watchdog. It then ended in `return_failed` with 0 completed cycles.

So the open frontier moved from "cannot descend" to the return-to-reset phase -- which is where
the MuJoCo-only loop on the training plant also failed on 2026-09-30 (0.24 deg short on the
orientation check). Both stacks now fail at the same place, so the next thing to look at is the
return itself (B4, C2), not the controller law.

2026-10-01, re-reading that recording. Two things qualify the summary above.

- The release fired at step 23, 1.53 s in, at `trajectory_progress` 0.024. The MuJoCo-only loop
  releases at 2.8 s, about 43 % into a ~6.5 s cycle; this was 24 %. 23 policy steps is not
  enough to descend onto the nut and turn it 55 deg, so "the arm descends and turns the nut"
  is more than that run shows. The grasp-frame yaw proxy cannot tell a nut turn from the wrist
  yaw of the descent, and training does not use a proxy -- it reads the thread twist joint.
  Worth settling before reading anything else into release timing (see E1).
- The phase machine itself ran exactly to spec: 44 steps of `follow_waypoints` (2.93 s, 4 x
  0.7 s) then 254 of `return_to_reset` (16.9 s, 1.0 s + the 16 s timeout). Joint 5 ended at
  2.807 rad against the 2.8065 MJCF limit, so the arm sat on a joint limit for the whole
  return. Note the MuJoCo-only loop also reaches 2.81 rad by progress 0.05, so the limit is
  the training posture, not by itself the fault.

The return check is NOT suspect: `build_hybrid_transition_sources` passes
`use_live_policy_reset_pose=(mode == CYCLIC_THREADING)`, so training's evaluator also measures
against the live grasp pose captured at reset, with the same 5 mm / 5 deg / 0.15 rad and the same
16 s timeout. `GripCycleCoordinator._returned` is faithful; do not re-audit it.

The recording cannot say which of the three gates failed, or by how much. `hardware.py` passes
only `completed_cycles` and `watchdog_stop` as task signals, so
`threading_turn_progress_rad` -- the one signal the release trigger runs on -- is NaN in every
ros-sim and hardware recording (`data_collection.py` defaults it to NaN; only
`mujoco_student_rollout.py` fills it), and no grasp pose is recorded at all. Isaac stores and
logs all three residuals (`_distillation_cyclic_return_{position,orientation,hand}_error*`).
Recording the proxy, the grasp pose and the return residuals is the cheapest next step on the
return, and it is a prerequisite for diagnosing it rather than guessing. All three are pure
functions of the recorded joints, so they can be rebuilt offline (session scratch
`replay_proxy.py` / `return_residuals.py`: flange from arm FK on the training scene, tips from
`TrainingHandKinematics`, then `grasp_frame_from_tips` and the coordinator's own arithmetic).
The reconstruction reproduces the proxy the rollout printed at its events to 0.1 deg, so it is
trustworthy -- but it should not have to be redone by hand every run.

2026-10-05, partly closed by E1. `threading_turn_progress_rad` is now filled in ros-sim
recordings with the simulated nut's own twist, and `report.json` carries `turn_source` and
`final_nut_twist_deg`. Still missing everywhere: the grasp pose and the three return residuals,
which is what would say *which* gate failed without the offline rebuild. On hardware the turn
signal is still NaN, because there is nothing there to measure it with.

2026-10-06, Layer A closed as far as the code can close it. A3 and A4 are done through a new
controller interface (`~/policy_goal`, `PolicyGoal`): the runner sends the preclipped goal and
the live grasp frame on the flange, the controller clips at 1 kHz against the measured grasp
pose and runs the law there. The Coriolis term is off and the +-100 Nm clamp is in. What the
controller still cannot copy from training is listed in
[cartesian_impedance_nullspace_design.md](cartesian_impedance_nullspace_design.md) section 9:
the arm armature (USD unreadable here), the finger-velocity part of the damping, the hand's
mass (D1). **Verified in gtest and by driving the controller with scripted goals in ros-sim
(no policy in the loop); no ros-sim or hardware policy rollout has run on this interface yet**,
so the "done when" tests of A3 and A4 below are still owed.

2026-10-08, that rollout happened, and the descent is solved. `ros_sim-20261008-080700-477622`
is the first policy rollout on `~/policy_goal` (both 10-07 attempts predate ac1db2b): 1500
steps, 100.2 s of simulated time at 15 Hz, median period 66 ms, 24 missed deadlines, live
preflight 20 ms against the 60 ms budget, no controller fault and no watchdog. A3 and A4 are
ticked on its evidence; the paragraph above is superseded on that point.

**The arm now reaches the nut and holds the policy's goal.** The grasp descends from z 0.264 m
to 0.110 m within 8.3 s (bolt tip 0.102 m) and stays there, `|goal - grasp|` falls from 36 mm
to 5-9 mm against the README's "about 1 cm in a healthy stack", and the wrist turns to 47.7 deg
by 8.3 s. The 09-14 descent stall is closed, and so is the suspicion over the controller law.

**The nut still does not turn, and the hand geometry is why.** Measured nut twist over the whole
run is 0.016 deg, in the loosening direction, so the 55 deg release gate never fires: the run
stays in `policy` for all 1500 steps and completes 0 cycles. At the hand postures this run
actually commanded, the thumb-to-index tip separation is

| model | min | median | max | sampled ticks <= 41.6 mm |
|---|---|---|---|---|
| training asset (what the student was distilled on) | 18.4 mm | 38.3 mm | 71.0 mm | 124 of 150 |
| workspace URDF (what ros-sim simulates), 2.36x | 60.4 mm | 90.3 mm | 123.3 mm | 0 of 150 |

An M24 nut is 36 mm across flats and 41.6 mm across corners. The training hand would be closed
on it in 83 % of the sampled ticks; the workspace hand never comes within 19 mm of touching it,
at any tick of the run. The index joint reached E1's measured contact threshold of 0.565 rad in
3 of 1500 ticks (0.2 %). So D1 is no longer a fidelity item to schedule behind the others -- it
is the binding constraint on ros-sim completing a cycle, and it needs the ruler.

One caveat when reading this recording: `trajectory_progress` saturates at 1.0 at t 64.6 s
(step 966) and no completed cycle ever rebased it, so the last 35 s are outside the trained
horizon and are not policy behaviour.

**2026-10-08, later: the paragraph above is wrong about the cause, and C1 is done.** Fingertip
separation was the wrong test. The pads that reach the nut are the finger *segments*, not the tip
frames -- E1's own contact measurement says so (14.6 N across `thumb_distal` and
`index_intermediate`), and this morning's reasoning ignored it. Two rollouts after C1 turned the
nut, which a hand that cannot touch it could not do:

| run | hand velocities | nut turn over the run |
|---|---|---|
| `20261008-080700` | zeros | +0.016 deg |
| `20261008-125849` | published | -37.8 deg |
| `20261008-130221` | published | -43.6 to -61.4 deg |

So the hand geometry is not what stops a cycle, and D1 goes back to being a fidelity item
rather than the blocker. Three things that are now the open questions instead.

- **The policy unscrews the nut in ros-sim.** `turn_progress_rad` is negative throughout, and
  the axial coordinate rises from 7.66 to 8.02 mm, which is the loosening direction on E1's own
  convention (it measured a tightening turn as negative twist, nut descending). The release gate
  wants +55 deg, so it can never fire while the nut backs out. The sign handling in the code is
  self-consistent -- `THREADING_DIRECTION_SIGN` is applied in `ThreadPairClient` -- so this is
  the motion, not a bookkeeping error.
- **Do not attribute the turn to C1 yet.** The commanded hand postures of the 08:07 and 12:58
  runs are nearly identical (index mean 0.404 vs 0.396 rad, thumb yaw 1.213 vs 1.207), so the
  velocity signal did not visibly change what the hand was told to do. With one run on each side
  and a plant the status section above already calls non-reproducible, contact luck is not ruled
  out.
- **`/reset_thread` does not actually reset the nut between runs.** The service answers success
  and the plugin's `reset_thread` writes both qpos values, but `20261008-130221` began at -43.6
  deg, exactly where its predecessor stopped. Most likely the hand is still gripping the nut when
  the reset lands, so contact drags it straight back. Until that is settled, any ros-sim
  comparison needs a fresh `sim_policy.launch.py`, not a second `run_policy_rollout.py`.

### C1. Publish hand joint velocities from the Inspire driver  [should]

Done 2026-10-08, in `inspire_hand_driver`. The hand has no speed sensor -- its register map
exposes POS_ACT, ANGLE_ACT, FORCE_ACT, CURRENT, ERROR, STATUS and TEMP, and SPEED_SET is a
commanded limit -- so the driver differences the ANGLE readings it already takes at 50 Hz and
fills `JointState.velocity` for all twelve joints, followers at their coupling ratio and zero
wherever a coupling clamps. Nothing downstream changed: `hand_joint_state_to_arrays` already
read the field with a zero fallback.

Why a backward difference. Checked against the reference episode's own recorded PhysX
velocities: differencing the positions recovers them at r 0.96 (thumb yaw) and 0.97 (index)
with a central difference, and essentially all of the backward difference's apparent error is
its half-sample lag -- correcting for that lag lifts it to 0.978 and 0.975. That lag is 10 ms at
50 Hz, so the causal form costs little and avoids holding a sample back. Thumb pitch recovers
poorly (r 0.48) because its true velocity is mostly contact chatter that no position difference
can see; it also carries the least signal.

Two guards, both from measurement rather than taste. The rate is clamped to what the hand was
told it may travel, via the manual's own calibration (2.4.8: speed 1000 crosses the full range
in 800 ms unloaded), which bounds the non-physical spikes a raw difference throws at slew steps.
And a one-pole 10 Hz filter is on by default, not for the size of the quantisation error but its
shape: one count is 1.5e-3 rad on a finger, so a slow creep flips a count only every few samples
and the raw difference comes out as bursts separated by exact zeros. At 0.05 rad/s that is zero
in a third of samples with a 0.034 spread; filtered it is a steady 0.050 with 0.010, unbiased at
every rate tested, for 16 ms of lag.

Done when, met. `joint_velocity[:, 7:10]` in `20261008-125849` is non-zero with mean magnitudes
0.074 / 0.041 / 0.037 rad/s against training's 0.126 / 0.040 / 0.317. Thumb pitch matches;
thumb yaw and index read low because that run moved the fingers less than the training episode
did, not because the estimate is attenuated.

Also added, for the bench: the driver reads SPEED_SET and DEFAULT_SPEED_SET at startup and logs
both with the resulting clamp, and `inspire_hand_probe` dumps them too. That is how to find the
speed the hand is really running at -- nothing in this stack writes SPEED_SET unless asked
(`startup_speed` defaults to 0), so the flash DEFAULT_SPEED_SET at register 1032 is what governs
finger rate, and it was previously unknown. Both are SET registers and write-only on some
firmware, so a zero or an error there is a firmware trait; a zero is never used as a clamp.

---

## Status, 2026-10-01: three ros-sim runs with A1b

| run | steps | release | return outcome |
|---|---|---|---|
| `20260930-143512` (A1a only) | 321 | step 22, 1.47 s, proxy jumped 68.7 deg to 63.3 | `return_failed` |
| `20261001-165420` (A1a+A1b) | 321 | step 22, 1.47 s, proxy jumped 99.8 deg to 94.7 | `return_failed` |
| `20261001-170332` (A1a+A1b, viewer) | 363 | step 64, 4.27 s, smooth ~3 deg/step to 55.5 | `return_failed` |

**The return fails on orientation, and only on orientation.** Reconstructed gates over the 254
`return_to_reset` steps of `170332`:

| gate | best reached | limit | |
|---|---|---|---|
| position | 0.4 mm | 5.0 mm | passes |
| hand | 0.060 rad | 0.15 rad | passes |
| orientation | 10.8 deg | 5.0 deg | **fails** |

The near-miss is step ~150, t 10.0 s: 3.2 mm / 12.9 deg / 0.127 rad, two gates in and
orientation 7.9 deg over. After that it drifts to ~162 mm / ~51 deg and sits there, flat, for
the last nine seconds of the 16 s wait -- the student has played past the return part of its
trajectory while the coordinator is still testing the check. The orientation residual tracks the
turn proxy almost exactly (51 vs 49 deg at the end, 12.9 vs 11.6 deg at the near-miss), so what
fails is specifically that **the student only partly unwinds the turn it made**. The MuJoCo-only
loop fails the same gate by 0.24 deg; we fail it by 10.8 deg, 40x worse but the same failure.

**A degenerate grasp frame can fire the release trigger, in some runs.** The grasp frame is built
from the fingertip midpoint and the thumb->index direction (`fo.hand_grasp_frame`). ros-sim's
`m24_nut` is `contype="0" conaffinity="0"` with zero joints -- a fixed visual, as
`make_ros_sim_scene.py` says in its own docstring -- so when the student closes the pinch there
is nothing between the fingers and the tips converge: 57.8 mm at reset to 4.5 mm by step 22.
At that separation the thumb->index direction is meaningless, the frame tips over
(`grasp_z . world_z` from -0.99 to -0.27 in two steps) and the proxy jumps ~100 deg in one 67 ms
step. In `143512` and `165420` that jump crossed the 55 deg gate and fired `release_started`; in
`170332` the same jump happened (37.2 deg at step 24, tips at 8.3 mm) but peaked near 49.5 deg,
under the gate, and release then fired later on a genuine smooth wrist turn. So it is a real
hazard that fired in two runs out of three, not a deterministic one.

2026-10-05, E1 removes this in ros-sim two ways over: the nut is now collidable, so the
fingertips cannot converge past it, and the release no longer reads the proxy at all -- it reads
the nut's own twist. **Hardware is unchanged and still exposed**, because a narrow pinch can
collapse the pads there too even though the real nut stops them; the proxy still needs a guard
that refuses to fire on a degenerate fingertip separation. `GripCycleCoordinator` now records
the proxy alongside whatever it gates on, so the size of that error is measurable from any
ros-sim recording.

**ros-sim runs are not reproducible, so do not compare single runs.** `170332` differed from
`165420` only in having the viewer open: preflight 0.048 s against 0.020 s, one missed deadline.
Camera frames arrive on wall clock while the runner paces the policy on `/clock`, so the policy
saw different frames and the trajectories separated -- different release step, different
posture, 363 steps against 321. Any claim of the form "change X moved metric Y" needs repeated
runs. In particular, steps with `|joint5| > 2.80` went 240 (A1a) / 24 (A1b) / 178 (A1b), so
A1b's effect on the joint-5 excursion is NOT established; the spread is run-to-run.

---

## Layer A: arm controller law

Training executes Forge's operational-space controller
(`forge_ultra/tasks/utils/control.py`, `compute_dof_torque`) at 120 Hz. The ROS stack executes
the franka_ros2 example Cartesian impedance law
(`src/franka_forge_cartesian_impedance/include/franka_forge_cartesian_impedance/cartesian_impedance.hpp`) at
1 kHz. The gains are the same numbers (565 / 28, damping 2*sqrt(K), nullspace 10); the law
behind them is not.

### A1. Replace the example nullspace term with Forge's mass-weighted, dynamically consistent projection  [blocking]

- [x] A1a. Exact projector (`nullspace_damping_lambda: 0.0`) -- done 2026-09-30
- [x] A1b. Mass weighting -- done 2026-10-01; fidelity now capped by the hand model, see D1

A1a is the part that mattered and it needed no mass matrix. `nullspace_damping_lambda` is now a
controller parameter (`cartesian_impedance.hpp`, default 0.2 = the example); both policy profiles
set 0.0, and `run_hardware_rollout` refuses a controller that does not. A1b was worth 0.52 Nm of
18.2 (see the residual table below), i.e. fidelity work rather than a blocker.

A1b, done 2026-10-01. `forge_nullspace_torque` in `cartesian_impedance.hpp` is Forge's term as
written: `M_task = (J M^-1 J^T)^-1`, `Jbar^T = M_task J M^-1`,
`tau_null = (I - J^T Jbar^T) M (kp wrap(q0 - q) - kd qdot)`, including the wrap to [-pi, pi]
the ROS law did not have. `example_cartesian_impedance` takes an optional `arm_mass_matrix` and
uses it instead of the damped-pseudo-inverse term; both policy profiles set
`mass_weighted_nullspace: true` and `run_hardware_rollout` refuses a controller that does not.

The mass matrix (`arm_mass_model.hpp` / `src/arm_mass_model.cpp`, pimpl so pinocchio stays out
of the impedance header). On the `dh` path it is a pinocchio `crba` over the description the
controller gets from `get_robot_description()`, with every non-arm joint locked by
`buildReducedModel` -- a 7-DOF model whose mass matrix *is* the arm block, so there is no index
juggling and nothing to allocate in `update()`. On hardware it is
`franka_robot_model_->getMassMatrix()`, which carries the configured end-effector load, so the
Inspire hand has to be set as the load or it is the mass matrix of a bare flange.

Verified 2026-10-01, all in the `inspire_franka` container:

| check | result |
|---|---|
| `forge_nullspace_torque` vs a transcription of `compute_dof_torque` | 1e-12 (gtest) |
| `Jbar^T tau_null` over six postures (dynamic consistency) | < 1e-9 N, i.e. no leak to the tool |
| Forge term at `M = I` vs the A1a exact-pinv term | agree, so lambda 0 really is the exact projector |
| reduced (hand-locked) model vs the full tree's arm block | 1.4e-17 kg m^2 |
| `ArmMassModel` on the real description vs MuJoCo `mj_fullM` arm block | 8.1e-04 kg m^2 (0.05 %) |
| controller configure in ros-sim | builds the model from the description, no fault |

The 8.1e-04 residual is the frozen hand posture, not an error in the model: sweeping every
finger joint from limit to limit moves the arm block by at most 1.2e-03 kg m^2 (0.06 %). The
controller has no hand state interfaces, so locking the hand is also the only option it has.

What A1b does NOT fix, and this is the part that matters. The arm mass matrix carries the hand's
inertia, and the three stacks do not agree on the hand:

| model | mass distal to `fr3_link7` | dominated by |
|---|---|---|
| `assets/fr3_inspirehand/fr3_inspirehand_replay.xml` (training mirror) | 0.5392 kg | `palm`, 0.3876 kg |
| `src/inspire_franka_sim/mjcf/inspire_franka_policy_scene.xml` (ros-sim) | 0.1918 kg | `hand_base_link`, 0.1414 kg |
| `inspire_franka.urdf.xacro hand_mount:=flange` (what the controller models) | 0.1918 kg | matches ros-sim to 9.1e-07 |

The seven FR3 link masses are identical in all three, and MuJoCo's `mj_fullM` was confirmed to
include armature (the difference is exactly `dof_armature`), so that 2.8x hand is the whole
remaining gap: 0.125 kg m^2, 6.5 % of the arm block at the reset pose. The ROS description and
the ros-sim plant agree with each other and both disagree with training. Which one is right is
D1's question -- weigh the hand -- and until it is settled the mass weighting is exact against
our own plant and 6.5 % off training's. Note this is the same root cause as the grasp-frame
mismatch: one RH56, two independent derivations.

Open, needs the training machine. The arm armature. `arm_armature` is a controller parameter and
both profiles set it to zeros (rigid-body only), because `assets/fr3_inspirehand/robot.py` sets
armature for the hand joints only and `fr3_no_hand.usd` is still a git-lfs pointer in this
checkout, so the arm's USD value could not be read. Both MJCFs carry Menagerie's 0.195 (joints
1-4) / 0.074 (joints 5-7), which are simulation-stability values rather than Franka data. Fetch
the USD (`git lfs pull`) or read it on the training machine, then set the parameter and say so
here. If PhysX reports armature in `get_generalized_mass_matrices()` and the USD carries any,
zeros are wrong.

Evidence. Forge: `tau_null = (I - J^T Jbar^T) M (10 (q0 - q) - 6.32 qdot)` with
`Jbar = M_task J M^-1`. ROS: `tau_null = (I - J^T pinv_lambda(J^T)) (10 (q0 - q) - 6.32 qdot)`
with a damped pseudo-inverse (lambda 0.2) that leaks into task space and no mass weighting.
Measured on the training scene at the M24 reset pose, for the wrist excursion the MuJoCo-only
loop uses to descend to the nut:

| wrist offset from home | Isaac projected torque | ROS projected torque | ROS on joints 5 / 7 |
|---|---|---|---|
| joint 5 +0.5 rad | 0.01 Nm | 1.17 Nm | -0.86 / +0.52 Nm |
| joint 5 +1.1, joint 7 -0.9 rad | 0.07 Nm | 4.24 Nm | -2.97 / +2.56 Nm |

Why a nullspace difference reaches the tool at all. A nullspace torque is by definition the
part of the command the tool should not feel. Forge's projector is dynamically consistent, so
it does not: mapping `tau_null` back through `Jbar` gives 0.00 N at the grasp frame. The
example's damped pseudo-inverse is not a projector at all for lambda 0.2, because lambda sits
right on top of the three smallest singular values of `J^T` (0.21 / 0.29 / 0.31 at this
posture), so it stops annihilating `range(J^T)` and the joint spring pushes on the tool:

| wrist offset | Forge: M-weighted + dyn. consistent | ROS: unweighted + damped pinv | unweighted + exact pinv (lambda 0) |
|---|---|---|---|
| joint 5 +0.5 rad | 0.00 N / 0.00 Nm | 3.19 N / 1.99 Nm | 0.22 N / 0.08 Nm |
| joint 5 +1.1, joint 7 -0.9 rad | 0.00 N / 0.00 Nm | 12.49 N / 7.98 Nm | 0.38 N / 0.12 Nm |

For scale, a full 20 mm clipped policy step commands 11.3 N through the 565 N/m task spring.
So at the descent posture the uncommanded nullspace force exceeds the policy's entire
authority, and it settles the tool about 22 mm away from the commanded pose. The student is a
progress-conditioned trajectory player with no integral action, so that offset is never
corrected, and it grows with the wrist excursion rather than staying a trimmable bias. Note
the decomposition: lambda, not the missing mass weighting, is what leaks (0.38 N vs 12.49 N),
so dropping lambda is the cheap fix and the M weighting is what makes the term match training
bit for bit.

That restoring torque is the same order as the joint friction removed in the uncommitted
`fr3.xml` experiment (1.14 / 0.76 Nm), which is why zeroing friction did not restore the
descent. Joints 5 to 7 saturate at 12 Nm. In the 2026-09-30 ros-sim run joint 5 reached only
1.86 rad where the MuJoCo-only loop reaches 2.81 rad by progress 0.05.

Change.
- `cartesian_impedance.hpp`: add a Forge-law variant that takes the arm mass matrix and
  projects with `Jbar`.
- Mass matrix source: `franka_robot_model_->getMass()` on hardware (`model_source: franka`).
  The DH path (`model_source: dh`, used by ros-sim) has no mass model: add a rigid-body model
  to the DH path, or let ros-sim read `M` from MuJoCo.
- A1b, the mass matrix. On hardware it already exists: `franka_robot_model_->getMass()`
  (`model_source: franka`). The `dh` path has no mass model, and ros-sim runs on it, so add one.
  Pinocchio is already available in the container (C++ config at
  `/opt/ros/jazzy/lib/x86_64-linux-gnu/cmake/pinocchio`, python 4.0.0) and
  `controller_interface::ControllerInterfaceBase::get_robot_description()` hands the controller
  the URDF at configure time. Build a `pinocchio::Model` from it once in `on_configure`, keep a
  `pinocchio::Data`, and call `crba(model, data, q)` in `update()` -- allocation-free, so
  realtime-safe -- then take the 7x7 arm block. The full tree (not a KDL chain) is what matches
  training, because Isaac's `arm_mass_matrix` is the arm block of the whole articulation and so
  carries the hand's inertia through the finger joints. Expand the URDF with
  `xacro src/inspire_franka_description/urdf/inspire_franka.urdf.xacro hand_mount:=flange`
  (21 inertials, 19 revolute joints) and check the result against MuJoCo's `mj_fullM` on the
  arm DOFs before wiring it into the law. `hand_mount` defaults to `bench`, which parents the
  hand to `world` and leaves the arm block with no hand inertia at all (10 to 14 % low); only
  `sim.launch.py`'s `flange` is the articulation training models. Open question: whether to add the arm armature to the diagonal --
  `assets/fr3_inspirehand/robot.py` sets `armature` for hand joints only, so the arm keeps
  whatever the USD carries, and the USD is a git-lfs pointer here. MuJoCo uses 0.195 (joints 1-4)
  and 0.074 (joints 5-7).
- Quick A/B before the real fix: `nullspace_stiffness: 0.0` in
  `src/inspire_franka_trajectory_replay/config/controllers_policy.yaml` and
  `controllers_sim_policy.yaml`, plus the `expected` dict in
  `apps/policy_rollout/policy_rollout/hardware.py` (`run_hardware_rollout`).

Done when. `src/franka_forge_cartesian_impedance/test/test_cartesian_impedance.cpp` reproduces
`forge_osc.compute_dof_torque` for the same state and target to numerical precision, and the
projected nullspace torque at a 1 rad wrist offset is below 0.1 Nm.

Residual after A1a and A2, against `forge_osc.compute_dof_torque` on the training scene with
the same state and target (`|tau_ros - tau_forge|`, Nm):

| wrist offset | `|tau_forge|` | example law | + lambda 0 | + axis-angle |
|---|---|---|---|---|
| at the reset pose | 8.66 | 2.87 | 2.87 | **0.00** |
| joint 5 +0.5 rad | 13.72 | 3.11 | 2.85 | **0.35** |
| joint 5 +1.1, joint 7 -0.9 rad | 18.20 | 5.29 | 3.02 | **0.52** |

At the reset pose the nullspace error is zero, so the whole 2.87 Nm discrepancy there is the
rotation error (A2) and the patched law is exact. The 0.35 / 0.52 Nm that survive at a wrist
excursion are the missing mass weighting, i.e. A1b.

Nullspace target and gain already match training: `hardware.py` publishes `home_arm` as
`nullspace_positions` and the controller runs `nullspace_stiffness: 10.0`, against Forge's
`kp_null: 10.0` / `kd_null: 6.3246` toward `default_dof_pos_tensor`. Only the projection differs.

Reference scripts: session scratch `nullspace.py` and `nullspace_decompose.py` (ThreadingScene at
the reset pose; all four weighting x projector combinations, plus the task-wrench equivalent
`Jbar tau_null`).

### A2. Use the axis-angle rotation error, not the quaternion vector part  [blocking]

- [x] done 2026-09-30

`example_cartesian_error` takes a `RotationErrorForm`; `kAxisAngle` builds the rotation block
from `Eigen::AngleAxisd(q_c^-1 q_d)`, which is the same axis with the full angle. The controller
parameter is `rotation_error` (`quaternion_vector` | `axis_angle`), both policy profiles set
`axis_angle`, and `ForgeCartesianImpedance.axis_angle_error_carries_the_full_angle` pins
`|rot| == theta` against the example's `sin(theta/2)` at five angles. The example law is
byte-identical at the defaults --
`ExampleCartesianImpedance.matches_upstream_update_over_a_moving_sequence` still passes -- so
trajectory replay and the hardware bringup profiles are untouched.

Evidence. ROS `example_cartesian_error` uses `-R vec(q_c^-1 q_d)` = `sin(theta/2) axis`.
Forge `get_pose_error` uses `axis_angle_from_quat(q_d q_c^-1)` = `theta axis`. The
proportional term is half of training's (ratio 0.500 at 0.05 rad, 0.493 at 0.6 rad):
14 Nm/rad effective instead of 28, while damping stays 2*sqrt(28) on the full angular
velocity. Orientation is soft and over-damped; yaw is what threads the nut.

Change. In `example_cartesian_error`, after the hemisphere flip, replace the vector part with
`2 atan2(|v|, w) v / |v|` (same convention as `forge_osc.get_pose_error`).

Done when. A 0.3 rad yaw error produces 8.4 Nm of task torque about the tool z axis at K = 28
in `test_cartesian_impedance.cpp`.

### A3. Anchor the compliance at the live fingertip midpoint instead of a fixed tool offset  [should]

- [x] done 2026-10-08

Evidence. Training applies the wrench at the live thumb/index midpoint, which travels about
20 mm over a threading cycle. The policy profile froze it at the threading grip
(`POLICY_TOOL_OFFSET_XYZ = (-0.059067, -0.028773, 0.173311)`); the commanded pose was
retargeted, the spring anchor was not. Distance to the live midpoint over the reference
episode: mean 11.5 mm, range 3.2 to 23.0 mm.

Done 2026-10-06. `PolicyGoal.tool_in_flange` carries the live grasp frame on the flange
(`hardware.grasp_in_flange`: the training hand kinematics' tips and the reset z transport, so
it depends on the hand joints only); the controller latches it before measuring, so the
measured pose, the clip and the Jacobian all refer to the live midpoint from that cycle on
(`rt_tool_translation_` / `rt_tool_rotation_`). The static `tool_offset_xyz` is now the
activation hold only. `retarget_grasp_pose_to_controlled_pose` and the runner's
`controller_target` are deleted; the goal IS the grasp goal. The latched frame survives a
watchdog stop and an abort so the hold does not jump; deactivation resets it.

Done when. `grasp_controlled_offset_m` in `report.json` (now the largest distance between the
controller's measured point and the live grasp over the run) stays within sampling skew, a few
mm at most, through a full cycle.

2026-10-08, passed. `ros_sim-20261008-080700-477622` reports `grasp_controlled_offset_m`
9.4 mm. That is one tick of hand motion, not a tracking error: the grasp midpoint in the flange
frame (a function of the three hand joints alone) moves up to 8.05 mm per 67 ms tick during the
initial pinch closure (step 19, t 1.27 s), median 0.63 mm, p90 1.62 mm, and at most 4.18 mm
after step 150. The controller latches the frame once per goal, so the offset is bounded by that
per-tick travel plus sample skew, and steady-state agreement is sub-millimetre.

### A4. Re-clip the 20 mm / 0.097 rad target step at the controller rate, from the live grasp pose  [minor]

- [x] done 2026-10-08

Evidence. Isaac decodes `bolt_tip + a * 0.05` and clips it against the current grasp pose at
every 120 Hz substep, so the target keeps leading the hand by up to 20 mm as it moves. The ROS
path clipped once per policy tick and the controller held that target for the whole period.

Done 2026-10-06. The runner sends `_apply_action` steps (0)-(1) (`hardware.policy_goal_base`,
the preclipped goal, constant over the tick); the controller runs step (2) every cycle:
`forge_clip_target` in `cartesian_impedance.hpp`, a line-for-line port of the position clip
and the Euler clip with Isaac's `get_euler_xyz` / `quat_from_euler_xyz` / `wrap_yaw`, in the
training world frame (`policy_clip_frame_yaw: pi`; the Euler clip is not frame-invariant in
general). Pinned against `forge_osc.decode_action_target` on six random cases near the reset
to 1e-12. `limit_cartesian_step` and `max_policy_step_*` no longer apply to goals: the clip
bounds the executed step from the measured pose every cycle, the workspace box bounds the goal,
and `max_position_error` still faults. `report.json` counts `clipped_policy_ticks` from the
controller's own `position_clipped` / `orientation_clipped` flags.

Done when. `clipped_policy_ticks` is a large fraction of the steps while the hand is moving,
as the clip is in `mujoco_threading_env.ThreadingScene.control_tick`, and the
`cartesian_state` target sits at the clip limit from the measured pose during the descent.

2026-10-08, passed. `ros_sim-20261008-080700-477622`: `clipped_policy_ticks` 1499 of 1500, i.e.
the controller's own clip is active on essentially every tick, as it is in
`ThreadingScene.control_tick`.

---

## Layer B: run lifecycle (rate, cycles, timing)

The student is progress-conditioned and phase-conditioned. If the clock or the cycle state
machine diverges from training, the policy plays the right trajectory against the wrong
state.

### B1. Fix the sign of the release-trigger proxy (grasp-frame yaw)  [blocking]

- [x] done 2026-09-30

`turn_progress_rad` no longer negates. Because the reset grasp z points down
(`grasp z . world z = -0.993`), a tightening turn already reads positive in that frame; the
`-1.0` was the bug. Verified on the training reset pose: a -56 deg world-z (tightening) turn now
reads +55.5 deg and fires `release_started`, a +56 deg (loosening) turn reads negative and fires
nothing. Both coordinator tests were rebuilt on the real z-down reset frame -- with the identity
quaternion they used before, either sign passes -- and
`test_cycle_coordinator_ignores_a_loosening_turn` now pins the direction.

Evidence. `GripCycleCoordinator.turn_progress_rad` in `hardware.py` takes the yaw of
`R_reset^T R_now`, a rotation about the grasp frame's own z axis, which points world-down. A
clockwise hand turn viewed from above (training's positive
`threading_directional_turn_progress`) therefore reads -60 deg and release never fires. The
unit test `test_cycle_coordinator_enters_release_after_clockwise_turn` passes only because it
uses an identity reset quaternion, where z points up. In the 2026-09-30 ros-sim run all 427
steps stayed in the `policy` phase.

Numeric check (numpy, no ROS): z-down reset frame from `grasp_frame_from_tips`, hand rotated
-60 deg about world +z -> proxy = -60 deg, no event; +60 deg -> +60 deg, `release_started`.

Change. Measure yaw about world +z (`fr3_link0` z, same as the training world) and keep
`progress = -delta_yaw`; rewrite the test in `apps/policy_rollout/tests/test_hardware.py`
with a z-down reset frame.

Done when. The numeric check above returns +60 deg and `release_started` for the clockwise
case; a ros-sim recording contains `follow_waypoints` rows.

### B2. Run the policy at 15 Hz, in simulated time for ros-sim  [blocking]

- [x] done 2026-09-30, ticked 2026-10-01

The `ros_sim-20260930-143512-493634` report already meets the test below:
`requested_period_s: 0.06666666666666667` and `missed_policy_deadlines: 0`. The evidence
paragraph describes the earlier `114721` run; the box was simply never ticked.

Evidence. The checkpoint is a 15 Hz policy (decimation 8 at 120 Hz, 64.6 s horizon). The
2026-09-30 `114721` ros-sim run requested a 0.1333 s period (`--rate 7.5`) and recorded a median
sim-time step of 0.134 s. The temporal ensemble then blends actions meant for 66 ms later
against a 133 ms tick, and the progress clock advances twice as far per policy step.

Change. Launch `sim_policy.launch.py sim_speed:=0.5 headless:=true camera_view:=false` and
run `ros-sim --rate 15`; the runner already paces on `/clock`. On hardware keep the 90 %
preflight budget at 66 ms.

Done when. `report.json` shows `requested_period_s: 0.0667` and `missed_policy_deadlines`
near zero.

### B3. Give ros-sim a longer policy-command watchdog  [minor]

- [ ] done

Evidence. The controller drops to hold after `policy_command_timeout: 0.5` s without a
command. On the shared 4-core CPU five missed deadlines ended the 2026-09-30 run with
`controller_watchdog` before anything task-related happened. Isaac has no equivalent.

Change. `controllers_sim_policy.yaml`: raise `policy_command_timeout` for rehearsal only, and
relax the matching check in `hardware.py` when `sim` is true. Keep 0.5 s on hardware.

Done when. A rehearsal ends on the step budget or a cycle limit, never on
`controller_watchdog`.

2026-10-01. The change was never made, but the symptom has not recurred: `143512` ran 321 steps
with `missed_policy_deadlines: 0` and `watchdog_stop: false`. Leave the item open -- the margin
is a shared-CPU accident, not a fix.

### B4. Document the progress-clock origin  [minor]

- [ ] done

Evidence. Training episodes start at the first policy row (no reset rows,
`sample_time_s[0] = 0`). Isaac's evaluator (`TrajectoryProgressClock.prime_before_reset`)
primes its clock before the 0.25 s reset settle, so its progress runs about +0.005 ahead of
the data. The local loops start at the first tick, which matches the data.

Change. Comment in `hardware.py` (`start = node.now_s()`) and
`utils/mujoco_student_rollout.py` so nobody "aligns" it with the evaluator.

Done when. Documented.

---

## Layer C: what the policy sees

The 29-D proprioception, the previous-action semantics, the phase one-hot and the DP3 camera
contract are verified identical to training. Three inputs still differ.

### C1. Publish hand joint velocities from the Inspire driver  [should]

- [x] done 2026-10-08; see the status section above for the evidence

Evidence. `src/inspire_hand_driver/inspire_hand_driver/driver_node.py` (around line 701)
fills `JointState.position` only, so the three hand-velocity slots of the proprio vector are
exactly zero on hardware and in ros-sim. In the reference training episode the hand velocity
averages 0.16 rad/s and reaches 0.97 rad/s at the first row; the MuJoCo-only loop averages
0.13 rad/s.

Change. Finite-difference the 50 Hz register reads in the driver (light low-pass) and publish
`velocity` for the six driven joints; the mock transport can differentiate its slew.
`hand_joint_state_to_arrays` already consumes it and `physical_hand_state_to_policy` already
rescales thumb yaw by 1/0.75.

Done when. `joint_velocity[:, 7:10]` in a rollout recording is non-zero and of the same
magnitude as the training episode.

### C2. Start the policy from the training start state, not from a settled grasp posture  [should]

- [ ] done

Evidence. Training row 0 has the hand at thumb yaw 1.085, thumb pitch 0.001, index 0.220 rad
with the index closing at 0.97 rad/s: episodes begin while the hand is still moving from a
more open posture to the grasp. Both local loops begin at the settled grasp posture
(1.185 / 0.051 / 0.215, zero velocity). With the same image, that changes the first commanded
z by 1.4 cm and yaw by 0.18 (filtered units). `apps/traj_replay/demo_trajs/traj_2/homing.yaml`
(thumb yaw 1.086, pitch 0.0002) is almost exactly the training row-0 posture.

Change. In `run_hardware_rollout`, home the hand to the row-0 posture, command the grasp
posture, and take the first policy sample on the first hand-state message after that command
instead of after `wait_for_hand`. Same for the reset in `utils/mujoco_student_rollout.py`.

Done when. The first recorded proprio row matches the training row-0 hand state to about
0.05 rad and shows a non-zero index velocity.

### C3. Bring the ros-sim scene appearance closer to the Isaac render  [minor, sim-only]

- [ ] done

Evidence. This checkpoint barely tracks the nut (its training set has one nut pose), but
appearance still moves the first action: with identical proprio the Isaac frame gives a
filtered z of 2.8, the ros-sim relay frame 2.5, the MuJoCo-only render 1.55 (a 6 cm spread).
The real D415 is the training camera, so hardware is unaffected.

Change. `apps/policy_rollout/utils/make_ros_sim_scene.py`: walnut table top, Isaac's floor
and dome lighting, no marker geometry; regenerate `inspire_franka_policy_scene.xml`. Verify
with `checkpoints/reference_episode/scratch/vision_head_crosscheck.py`.

Done when. Latent cosine to the Isaac frame above 0.9 at the reset view and the step-0 z
action within 0.3 of the Isaac-frame value.

### C4. Use 16 flow integration steps when the CPU budget allows  [minor]

- [ ] done

Evidence. Isaac samples with 16 Euler steps; the hardware and ros-sim default is 2.
Teacher-forced first-action MAE on training frames (rows 0..119, unified units): 0.0072 with
2 steps, 0.0063 with 4, 0.0052 with 16, i.e. 2.5 mm vs 1.1 mm in z. Not a cause of the
failures, but a free gain once the loop fits at 15 Hz.

Change. `--integration-steps 16`, or 4 as the compromise; the preflight rejects what does not
fit.

Done when. Preflight passes at 15 Hz with the chosen value.

---

## Layer D: hand posture, model, actuation

The policy's three hand coordinates are logical training-model joints. What the physical
fingers do with them is a separate question that needs a ruler.

### D1. Measure which hand model matches the real fingertips, then make the other follow  [should, bench]

- [ ] done

Evidence. Two independent RH56 descriptions: training (Tiangong URDF, tip frames added by
forgeUltra) and this workspace (dex-urdf). At the same joint command the tip separation is
57.8 mm vs 113.6 mm, the midpoint 14.8 mm apart, the grasp frame 20.9 deg apart. The rollout
now computes the policy's frame with the training kinematics, so the control loop is
consistent with training, but whether the physical pads land on the nut depends on which
model is right.

Change. At the threading grasp posture, measure the real thumb-to-index distance and the pad
midpoint relative to the flange. If the training model wins, update
`src/inspire_hand_description`; if the workspace model wins, retrain or add a joint remap in
`forge_osc.pinch_targets`.

Done when. One model, used by TF, MuJoCo and `TrainingHandKinematics`, within 3 mm of the
measured tips.

2026-10-01, the two models also disagree on mass, which A1b made load-bearing. Everything distal
to `fr3_link7` weighs 0.5392 kg in the training mirror (a 0.3876 kg `palm`) and 0.1918 kg here (a
0.1414 kg `hand_base_link`) -- 2.8x. That is 6.5 % of the arm mass matrix, which the nullspace
term is now weighted by, so the hand model is no longer only a geometry question. Weigh the hand
and its palm shell on the bench along with the tip measurement; a scale settles this one
outright, unlike the tip frames.

### D2. Decide the thumb-yaw overlay for policy commands  [should, bench]

- [ ] done

Evidence. The driver remaps thumb abduction into the top 75 % of its travel
(`command_overlays.THUMB_ABDUCTION_ZERO_OPEN_RATIO = 0.25`). A logical 1.185 rad is executed
as 0.889 rad physically, and the rollout inverts the overlay on feedback so the policy never
sees the difference. Training has no such remap: it commanded and observed the same joint.

Change. Check on the bench whether the training posture (1.185 rad thumb yaw in the training
model) is physically reachable. If it is, bypass the overlay for policy commands in
`publish_hand_target` and drop the inversion in `physical_hand_state_to_policy`; if not, the
retrain in D1 has to include the reachable range.

Done when. Commanded and reported thumb yaw agree without an inversion step.

### D3. Match the simulated hand response to training's PD drive  [minor, sim-only]

- [ ] done

Evidence. Training: PhysX PD, stiffness 30, damping 3, effort 20 Nm on the pinch joints,
targets refreshed at 120 Hz. ros-sim: mock driver quantises to 0..1000 registers and slews at
1200 counts/s in wall time, then `mujoco_ros2_control` PID (p 20, d 0.4, +-1 Nm) on the
dex-urdf hand. Only matters once the sim hand has contact geometry (E1).

Change. Slew the mock in sim time (`use_sim_time` in `MockTransport`); raise the hand PID
clamp toward the training effort in `src/inspire_franka_sim/config/pids.yaml` once contact
exists.

Done when. A 1 rad step on the index joint settles in about the same time in ros-sim as in
the MuJoCo-only loop.

---

## Layer E: ros-sim plant fidelity (sim-only)

These make the rehearsal predictive of hardware; they do not change the hardware stack.

### E1. Give ros-sim the dynamic thread pair, and read the real turn  [should]

- [x] ported 2026-10-05; the "done when" below is not yet demonstrated

**The original evidence here was wrong on one point: the ros-sim hand DOES have collision
geometry.** Measured on the compiled scene: 20 collidable group-3 geoms on the hand (plus 8 on
the arm), which `make_hand_mjcf.py` has always emitted. What the scene lacked was anything for
them to touch -- the nut was `contype="0" conaffinity="0"` with no joints, a fixed visual -- and
training's pad friction. The generator docstring, `sim_policy.launch.py` and this item all
claimed "no collision geometry"; all three are corrected.

Ported 2026-10-05.

- `make_ros_sim_scene.py` emits ForgeUltra's thread pair as static MJCF (`thread_pair_xml` /
  `thread_equality_xml`): `nut_carrier` at the bolt tip with the `nut_axial` slide (armature
  5 kg, so MuJoCo's equality is as stiff as Isaac's 200 kN/m drive), the `m24_nut` body with the
  `nut_twist` hinge (0.002 Nm Coulomb, 0.0002 viscous, armature 0.001), a collidable nut geom at
  0.05 kg, the `thread_coupling` joint equality (`axial = 0.00766 + 4.77e-4 * twist`) and an
  inactive `thread_hold`. The keyframe now carries the nut's start pose.
- `make_hand_mjcf.py` sets the pads to training's 0.75 (`CONTACT_FRICTION`). This matters because
  MuJoCo takes the element-wise MAXIMUM of the two geoms' friction, so lowering the nut alone
  would have left contacts at the hand's default 1.0.
- The bolt stays non-collidable -- `contype=0` in the training asset too. The equality already
  constrains the nut to the bolt axis; meshing the two convex hulls would fight it.
- `inspire_franka_sim`'s thread pair plugin (new C++ pluginlib plugin for
  `mujoco_ros2_control_plugins`) publishes `/thread_state` (JointState, 100 Hz sim time),
  clamps on `/thread_hold` and resets on `/reset_thread`. The nut is not part of the robot, so
  no ros2_control state interface can reach it. Loaded via the `mujoco_plugins` parameter in
  `controllers_sim_policy.yaml`.
- `hardware.py`'s `ThreadPairClient` feeds the measured turn into `GripCycleCoordinator.update`,
  holds the thread for the release/return transition and rebases on a completed cycle. The proxy
  is still advanced and recorded, so the gap between the two is now measurable, and
  `threading_turn_progress_rad` is no longer NaN in ros-sim recordings. `turn_source` and
  `final_nut_twist_deg` are in `report.json`.

Verified (container, `MUJOCO_GL=egl`):

| check | result |
|---|---|
| thread pair parameters vs `ThreadingScene` (joints, equalities, nut geom, masses) | 27/27 identical |
| coupling: d(axial)/d(twist) against pitch/2pi | 0.00047678 vs 0.00047746 (0.14 %) |
| mechanism vs training under the same loads | -0.02 Nm: -479 vs -486 deg; 20 N push: -202 vs -208 deg |
| unloaded drift | 0.000 deg (training -0.01 deg, its gravity on the 50 g nut) |
| pinch closes onto the nut | contact at index 0.565 rad, 14.6 N across thumb_distal + index_intermediate |
| wrist yaw with the nut gripped | twist -47.8 deg, axial -0.437 mm, i.e. it tightens and descends |
| `thread_hold` clamps | 0.00 deg under a torque that otherwise spins it > 360 deg |
| plugin in a live ros-sim | loads, publishes at 100 Hz sim time, service answers |

Gotchas found on the way, both now in the code as comments: `mujoco_vendor`'s exported cmake
target carries a RELATIVE library path, so `ament_target_dependencies(mujoco_vendor)` fails to
link and `mujoco::mujoco` from its extras file must be used instead; and `mujoco_plugins` is a
NESTED parameter (`mujoco_plugins.<name>.type`) -- passing it as a flat list of plugin names
crashes `MujocoSystemInterface` with `basic_string::substr: __pos (which is 15) > this->size()`
and takes the whole hardware component down.

Two things this does NOT settle.

- **Whether the policy's pinch lands on the nut is D1's question, not E1's.** At `policy_home`
  the ros-sim fingertips are 114.9 mm apart against training's 57.8, and the hand sits 154 mm
  above the nut (the student descends onto it). Contact needed index at 0.565 rad; the policy
  reached 0.66 rad in the `170332` run, so the range does overlap -- but where the pads land is
  the 2x geometry mismatch, and that is decided by a ruler.
- **Gravity is off in ros-sim** (libfranka compensates it on the real arm) while training has it
  on with `gravcomp=1` on the robot bodies only, so training's nut carries its own 0.49 N. On
  this thread that is worth 2.3e-4 Nm against the hinge's 2e-3 Nm Coulomb, i.e. it cannot turn
  the nut; measured drift over a second is 0.084 mm. Left as-is and noted rather than changed.

A side effect worth having: with the nut collidable the fingertips can no longer collapse to
4.5 mm, so the degenerate-grasp-frame release artefact described in the 2026-10-01 status
section is both unreachable (the nut is between the pads) and no longer what the release reads.

Done when. A ros-sim run completes a cycle on the physical turn, not the proxy. **Not yet
shown -- the blocker is CPU, not the port.** Five attempts on 2026-10-05, all in
`ROS_DOMAIN_ID=73` beside the long-lived simulator in the default domain, on a box at load
16-25:

| attempt | setting | outcome |
|---|---|---|
| 1 | 15 Hz, sim_speed 0.5 | homed, controller switched, thread pair live, then `controller_watchdog`; steady policy pass 0.354 s against the 0.060 s budget |
| 2 | 5 Hz, sim_speed 0.2 | `sim_speed` 0.2 is out of range: `Pid is called with negative dt` deactivates `InspireFrankaSystem` during write, and no controller can then activate. 0.5 is the usable floor |
| 3, 4 | 5 Hz, sim_speed 0.5 | leftover nodes from the earlier attempts were still publishing `/clock` in the same domain; two clocks give the PID a negative dt, same failure. Clearing the domain by `ROS_DOMAIN_ID` in each process's environment fixed it |
| 5 | 5 Hz, sim_speed 0.5, camera 15 Hz | reached the policy loop again, `controller_watchdog`; policy pass 0.672 s (worse -- the camera rate it needs costs the CPU the loop needs) |

So everything up to and including the first policy tick is exercised: homing, the controller
switch, `/thread_state` arriving, the reset service answering, and the release switching to the
measured turn. What is unproven is the only thing that needs a loop fast enough to hold a
period: the cycle itself. Re-run on a quiet machine (close the viewer simulator first), or do
B3 and raise `policy_command_timeout` for rehearsal.

Two bugs the attempts found, both fixed:

- the hold publisher was VOLATILE against the plugin's TRANSIENT_LOCAL subscription. DDS reports
  `requesting incompatible QoS ... DURABILITY` once and then silently delivers nothing, which
  presents as a thread that refuses to clamp. `ThreadPairClient` now publishes TRANSIENT_LOCAL.
- `mujoco_plugins` is a nested parameter, not a list (see the gotchas above).

### E2. Pin the FR3 joint dynamics to the training asset and commit the choice  [minor]

- [ ] done

Evidence. The working tree has an uncommitted experiment in
`src/inspire_franka_sim/mjcf/fr3.xml` that zeroes damping and friction but keeps Menagerie's
armature (0.195 / 0.074). The Isaac USD (`forgeUltra/assets/fr3_inspirehand/fr3_no_hand.usd`)
is a git-lfs pointer in the local checkout, so its joint armature and friction could not be
read; `robot.py`'s ArticulationCfg leaves the arm values at the USD defaults. The MuJoCo-only
loop zeroes damping and friction and keeps armature too.

Change. Read the arm joint properties from the USD on the training machine, set `fr3.xml`
and `assets/fr3_inspirehand/fr3_inspirehand_replay.xml` to the same values, and commit with
the numbers in the comment.

Done when. `git status` is clean and both MJCFs cite the USD values.

---

## Leave as the sim-to-real gap

Physics that training approximated and nobody has measured. Once every item above is done,
these are the only things that can still explain a hardware difference. Measure them; do not
tune around them.

| Gap | Training assumption | Reality / how to measure |
|---|---|---|
| FR3 joint friction and armature | PhysX asset values (not readable locally) | Unknown; libfranka compensates gravity, not friction. Slow joint-space sine sweep. |
| Hand response and RS485 timing | Ideal 120 Hz PD | About 0.17 s settling, bus latency, register quantisation. |
| Finger-pad and nut friction | 0.75 static/dynamic on nut and robot, 0.25 to 1.25 on the bolt | Real pad rubber on a steel nut, unmeasured. |
| Thread engagement | Coaxial slide with a 200 kN/m spring, 0.002 Nm Coulomb, releases after one turn | A real M24 binds, cross-threads, has backlash. |
| Nut spawn | One pose, yaw 30 deg, 7.66 mm axial start, bit-exact in all 20 episodes | Operator-placed. |
| D415 depth | Ideal pinhole depth | Holes, edge noise, 6 px principal-point offset, camera-pose calibration RMSE 27 mm / 3.5 deg. |
| Coriolis and model mismatch | Isaac OSC has no coriolis term, PhysX's arm feels its own | Since 2026-10-06 the hardware profile compensates none either (`coriolis_compensation: false`); what differs is the real arm's Coriolis vs PhysX's, i.e. the mass model. |
| Damping velocity | `0.5 (v_thumb_tip + v_index_tip)`, finger motion included | `J qdot` at the grasp point, arm only. Needs hand joint velocities (C1) to close. |
| Arm armature in `M` | whatever `fr3_no_hand.usd` carries (lfs pointer here, not readable) | `arm_armature: 0`; MuJoCo's 0.195 / 0.074 are Menagerie's, not Franka's. Read the USD on the training machine. |

---

## Already identical, no work needed

- Unified action -> filtered native action: Isaac inverts the EMA
  (`forge_raw_native_action_for_filtered_target`), the local `OscActionFilter` multiplies by
  the scale `[3, 4, 16, 3, 3, 3, 2, 2, 2]`. Same applied target.
- Previous action in the proprio vector is the previous step's filtered native action:
  bit-exact in the training data (`prev[t] == osc_filtered_action[t-1]`, set by
  `rewards/threading.py` post-step), in Isaac (`head_rgbd_env.py` uses `self.prev_actions`),
  and locally.
- Nominal replay in Isaac (`disable_domain_randomization`) zeroes gain noise, threshold noise
  and the dead zone, and fixes the EMA at 0.0625; the local stacks use the same constants and
  no dead zone.
- Phase one-hot schema and ordering; temporal-ensemble weighting (0.5 per age); decoder
  clipping math; 20 mm / 0.097 rad thresholds; 565 / 28 / 10 gains; 2*sqrt(K) damping;
  torque clamp 100 Nm then 87 / 12 Nm effort limits.
- Frames: training world = `fr3_link0` yawed by pi at (1.2, 0, 0); bolt tip, nut pose and
  camera pose agree to 0.1 mm. Nut yaw differs by a multiple of the hex symmetry only.
- Grasp approach axis: the untilted reset z transported with the flange, captured after
  homing, exactly as `randomize_initial_state` stores `hand_grasp_reset_z_transport`.
- DP3 intrinsics, crop, point-cloud box, RGB scaling and depth validity handling.
- Release/return timing: 55 deg trigger, four 0.7 s waypoint phases, 1.0 s return, 5 mm /
  5 deg / 0.15 rad return check, 16 s timeout.

---

## Evidence sources

- Training scene compiled from `assets/fr3_inspirehand/fr3_inspirehand_replay.xml` at the
  M24 reset pose (`policy_rollout.mujoco_threading_env.ThreadingScene`).
- Reference training episode
  `apps/policy_rollout/checkpoints/reference_episode/episode_000_sequential_threading.npz`.
- Recordings `logs/policy_rollout/ros_sim-20260930-114721-942722` (7.5 Hz, 2 flow steps,
  427 steps, controller watchdog) and
  `apps/policy_rollout/checkpoints/reference_episode/student_rollout` (MuJoCo-only, 10 cycles
  on the 2026-09-11 scene) / `student_rollout_20260930` (release at 2.8 s, return check
  missed by 0.24 deg).
- Isaac side: `forgeUltra/distillation/tasks/ablation_offline_flow_matching/evaluate_flow.py`,
  `distillation/utils/evaluation/cyclic_threading.py`, `distillation/utils/teachers/forge_transitions.py`,
  `forge_ultra/tasks/mdp/robot_control.py`, `forge_ultra/tasks/utils/control.py`,
  `forge_ultra/tasks/forge_franka_threading/forge_franka_env.py`, branch `franka-chi`.
- Checkpoint inference on CPU in the `inspire_franka` container (`torch 2.14.0+cpu`,
  `mujoco 3.12.0`).
