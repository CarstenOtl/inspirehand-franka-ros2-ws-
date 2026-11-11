# OSC characterization on the real FR3: test design (2026-11-11)

Status:
- **Part A** (S static holds, D dead-zone probe, wrist spin) is built:
  `apps/policy_rollout/utils/osc_characterization.py` and `osc_characterization_analysis.py`.
  - **Duration:** about 3.7 min per friction session with both payload levels; 146 s
    in ros-sim with one level.
  - **Sim check:** with the worst-case dead zone [5 N, 1 N m], the probe measures
    breakaway 5.2-5.4 N and 1.1-1.17 N m, about 4-8 % / 10-15 % high. Without the dead
    zone it measures 0-0.15 N / 0-0.05 N m. The result is the same at 5 and 6 mm/s.
  - **Hardware:** not run yet.
  - **Defaults:** the probe runs at one pose (o0), ±15 mm at 6 mm/s and ±8° at 2°/s.
    The static holds cover all three poses plus 5 wrist-spin stops, with 2 s holds.
    The second payload level walks the same stations in reverse.
- **Part B** (the M repeatability matrix: 3 clip levels × payload × 5/10/50 % ×
  orientations) is not built.

## Part A on hardware

1. **Desk:** activate a copy of the Inspire-hand end-effector profile with mass, centre of
   mass and inertia set to 0, and the flange-to-EE transform left at identity. The script
   refuses to start if the EE mass is not 0 or the transform is not identity.
2. **Hand:** leave it open, as at the start of a rollout. The fingers are part of the
   payload, and the script does not command them.
3. **Launch one friction level** (the friction-on session uses the default profile):

   ```bash
   # session 1: friction compensation off
   ros2 launch inspire_franka_trajectory_replay replay.launch.py arm_controller:=policy \
     robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0 \
     controllers_yaml:=$(ros2 pkg prefix inspire_franka_trajectory_replay)/share/inspire_franka_trajectory_replay/config/controllers_policy_no_friction_comp.yaml
   # session 2: friction compensation on (0.3 N m)
   ros2 launch inspire_franka_trajectory_replay replay.launch.py arm_controller:=policy \
     robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0
   ```

4. **Plan check:**

   ```bash
   cd /root/develop_ws
   python3 apps/policy_rollout/utils/osc_characterization.py hardware --plan-only
   ```

   It prints the three orientations (seed 20261111), their MuJoCo clearance, and the
   time estimate.
5. **Run:**

   ```bash
   python3 apps/policy_rollout/utils/osc_characterization.py hardware --payload modelled none
   ```

   The script:
   - asks before the first move and before the first `set_load`;
   - records its own bag to `debug/osc_characterization/hardware-<stamp>/bag`;
   - finishes with the modelled payload applied, and reminds you to switch Desk back
     to the normal profile.
6. **Analysis:**

   ```bash
   python3 apps/policy_rollout/utils/osc_characterization_analysis.py debug/osc_characterization/hardware-<stamp>
   ```

   The output:
   - **per pose and axis:** breakaway (static dead zone, N / N m), bias, and creep minus
     damping;
   - **payload:** a fit per level, the `none − modelled` difference with its sign
     convention, and the identified hand against Desk and the training URDF.

   Everything is also written to `analysis.json`.

Notes:
- **First real `set_load`:** this is the first time `set_load` runs with no controller
  active. Keep a hand on the E-stop for the first switch.
- **Abort:** the probe stops if the tracking error exceeds 45 mm or 20°.
- **Spin posture:** the wrist spin holds the flange pointing straight out over the table at
  0.78 m. Check by eye that the head camera rig is clear before the first run.

## What the test answers

1. **EE payload model.** Do the mass, centre of mass and inertia we give Franka for the
   mounted Inspire hand match the real hand? We cross-check them against Franka's
   external-wrench estimate (`tau_ext_hat_filtered`, `O_F_ext_hat_K`, `K_F_ext_hat_K`).
2. **Controller dead zones.** How big is the band of task-space error that the deployment
   OSC cannot correct, because of joint friction and the torque-sensor loop? This is the
   model gap behind the 11-13 mm / 20-25 deg return residual (docs/hardware_rollout_2026-11-10.md).
   The result is given in N and N m, so it compares directly with ForgeUltra's
   `task_wrench_dead_zone` (training default [5, 5, 5 N, 1, 1, 1 N m]; at 565 N/m and
   28 N m/rad that is 8.8 mm and 2.0 deg).

## Factors

| Factor | Levels | How it is switched |
|---|---|---|
| Controller | **clipped**: `~/policy_goal` with the 20 mm / 0.097 rad clip (deployment, policy phase); **transition clip**: `~/policy_goal` with the per-goal clip set to 0.05 m / 0.25 rad and fault limits 0.1 m / 0.5 rad (what the training team proposes for `follow_waypoints`/`return_to_reset`); **unclipped**: `~/policy_command`, same law, gains, nullspace and friction feedforward, no clip, profile fault limits 0.08 m / 0.35 rad | Inside one run. The clip levels use the new per-goal `PolicyGoal` fields (uncommitted, 2026-11-11). The profile ceilings are 0.05 m / 0.25 rad. |
| Friction comp | off (`controllers_policy_no_friction_comp.yaml`), on (`controllers_policy.yaml`, 0.3 N m on all 7 joints) | **One launch per level.** `friction_compensation_nm` is only read in `on_configure`, and the runtime parameter callback does not take it. Switching profiles needs no change to the real-time controller. |
| EE payload model | **modelled**: today's Desk values (m 0.579 kg, COM (0.0036, -0.0041, 0.0981) m, I diag 0.00041/0.00054/0.00020); **none**: 0 kg. The hand stays mounted in both. | Chosen: `service_server/set_load` from the script, on top of a zero-mass Desk EE profile (see "Payload switching" below). |
| Speed | 5 %, 10 %, 50 % of the FR3 Cartesian limits (libfranka `rate_limiting.h`: 3.0 m/s, 9.0 m/s², 2.5 rad/s, 17 rad/s²) | Per block |
| Orientation | 3 random, seeded, rejection-sampled (below) | Per block |

### Payload switching

`setLoad` is libfranka's *payload* on top of the Desk end-effector. It does not edit Desk.
The robot uses EE + load (`inertia_total`) for its gravity compensation and for
`tau_ext`/`F_ext`. It cannot remove mass, so the "none" level needs a Desk EE with zero mass:

1. In Desk, copy the active Inspire-hand EE profile and set mass, COM and inertia to 0.
   **Keep its flange-to-EE transform as it is (identity).** The controller reads `O_T_EE`
   and the `kEndEffector` Jacobian and adds the tool offset itself, assuming the robot's
   F_T_EE is identity. Activate the copy for the test.
2. The script sets the payload with no arm controller active: the hand's Desk values, or 0.
   It then reads `inertia_ee`, `inertia_load`, `inertia_total` and `f_t_ee` back from
   `robot_state`, and stops on any mismatch. It does this at the start of every payload
   level and after any reconnect, because the load is not assumed to survive an FCI
   reconnect.
3. At the end, the script sets the load back to 0. **Switch Desk back to the normal hand
   profile before any policy rollout.** Otherwise the deployment runs without the hand's
   mass: the runner's `configured load` line would show 0.000 kg EE mass.

The speed level scales velocity and acceleration together. The goal moves along a quintic
blend sized to the slower of the two limits:

| Level | v (m/s) | a (m/s²) | ω (rad/s) | α (rad/s²) | Peak reached on a 20 cm stroke | Note |
|---|---|---|---|---|---|---|
| 5 % | 0.15 | 0.45 | 0.125 | 0.85 | 0.15 m/s (2.5 s move) | below the clip ceiling |
| 10 % | 0.30 | 0.90 | 0.25 | 1.7 | 0.30 m/s (1.25 s) | just above the clip ceiling |
| 50 % | 1.50 | 4.5 | 1.25 | 8.5 | **0.74 m/s** (0.51 s, limited by acceleration) | 1.5 m/s needs an 0.8 m stroke; a 30 cm stroke gives 0.9 m/s |

The clipped controller cannot follow faster than about 0.24 m/s per axis or 0.26 rad/s
(clip ÷ damping: 565 · 0.02 / (2√565), 28 · 0.097 / (2√28)), which is about 8 % of the
translational limit. Its 10 % and 50 % blocks are expected to give the same motion. They
stay in the plan as a check, and they show what the policy's clip does to a fast command.

The unclipped controller damps absolute velocity and has no velocity feedforward. It
therefore trails a moving goal by about 2v/√K = 0.084 s · v: 13 mm at 5 %, 25 mm at 10 %,
and about 60 mm at the 50 % peak (a spring force of about 35 N). That lag is part of the
measurement, and it also enters the collision margin below.

## Poses

- Positions A and B are 20 cm apart (30 cm if the 50 % level should get closer to its target
  speed). They default to the repeatability script's (0.30, 0.20, 0.30) → (0.30, 0.40, 0.30)
  in the base frame, unless the collision check rejects them.
- **Orientations:** 3 draws from a fixed seed, written into the report. Each is flange-down
  tilted by up to 35° about a random horizontal axis, with yaw up to ±60°. A draw is rejected
  if any of these fail along the whole path (transits, A↔B lines, the 5 cm overshoot
  waypoints):
  - IK with a joint margin of at least 0.10 rad and a Jacobian σ_min of at least 0.03;
  - no crossing of the clip's yaw seam;
  - joint speed under 50 % of the limit at the 50 % level;
  - **MuJoCo clearance** in `inspire_franka_policy_scene_windback45_tilt30.xml` (FR3 v2
    links and hand against the table, holder, bolt, nut and floor): `mj_geomDistance` of at
    least 5 cm at 5/10 %. At 50 % the minimum is 5 cm plus the predicted lag and overshoot
    (about 10 cm), checked with the path swept back along the direction of motion.
    MuJoCo is 3.12 in the container. The scene has no back table, shelf, camera rig or
    cables. The operator confirms those by eye on the dry run, before the first live block.
- One fixed elbow posture, reached by the joint controller and done in several short steps.
  The 2026-11-10 attempts failed on exactly this: a 3.03 rad goto step was refused, and the
  start pose was 0.71 rad off.

## Phases of one session (friction comp fixed by the launch)

Warm up first: 10 min of slow motion. Joint friction depends on temperature.

```
preflight: controller profile, friction comp, Desk F_T_EE and EE mass, tool check, plan + collision check
reference block R0: clipped, payload modelled, 5 %, orientation o0, 2 repeats   (drift check)
for payload in [modelled, none]:            # set_load with no controller active, then read back
    S  static holds: 6 poses (o0-o2 × A, B), 5 s each, plus the wrist spin
    D  dead-zone probe at o0-o2 / A          (clipped is enough: the ramp stays inside the clip)
    for controller in [clipped, transition clip, unclipped]:
        for speed in [5, 10, 50]:            # 50 % unclipped only after 10 % unclipped passed its lag check
            M  repeatability block: 3 orientations × A, B × both approach sides × 3 repeats = 36 arrivals
reference block R1: same as R0              (drift check)
```

The order of the M blocks inside a payload level comes from a shuffle with the session
seed, so that drift does not line up with a single factor.

### M: accuracy and repeatability (reuses `osc_pose_repeatability.py`)

Each target is reached from both sides along A-B, which splits the rest error into:
- **hysteresis**: the difference between the two approach directions. This is friction or
  the dead zone.
- **bias**: the mean of the two directions. This is gravity or payload model error, plus
  any pose-dependent offset.

The script also records: ISO 9283 repeatability, time to the 5 mm / 5° gate,
overshoot, the tracking lag profile during the move, the peak speed reached, the fraction
of time clipped, and `tau_task`/`wrench`/`tau_command` at rest.

### S: static payload check

At rest the joint torque sensor is link-side, so gearbox friction is not in `tau_J`. That
makes the rest value of `tau_ext_hat_filtered` a friction-free payload signal:
- payload **modelled**: `tau_ext` ≈ payload model error + sensor offset + arm model error;
- payload **none**: `tau_ext` ≈ the whole hand + sensor offset + arm model error.

Fit: across poses, `tau_ext(q) = J(q)^T [m g ; (m c) × g] + offset_j`. That is 4 payload
unknowns plus 7 per-joint offsets, fitted on the payload-none data, which gives the
identified m and c. On the payload-modelled data, a residual near zero means the Desk model
is right. The fit is also run with the training URDF hand (+40 g cable) for comparison.
The report gives the fit's condition number and per-joint residuals.

**Wrist spin (included).** Point the flange axis horizontal, then turn j7 slowly through
about ±150° with the joint controller. The arm posture j1-j5 stays fixed, so arm model
errors stay constant. The hand's gravity torque on j6/j7 is then a sinusoid in the j7
angle: its amplitude and phase give c_xy, and the j6 term gives c_z.
Recommended, because the COM difference in question is small: Desk vs training is 20 mm,
which is about 0.11 N m. That is the same size as the pose-dependent arm model error seen on
2026-11-10 (0.4-0.7 N m on j3/j5/j7), which three tilted orientations alone may not separate.

### D: dead-zone probe

At a static pose, the goal is moved slowly along one task axis at a time, in a triangle:
- translation: 0 → +15 mm → −15 mm → 0 at 2 mm/s;
- rotation: 0 → +5° → −5° → 0 at 0.5°/s.

All 6 axes are probed, which takes about 3.5 min per pose. Because 15 mm is under the 20 mm
clip, clipped and unclipped behave the same here. From the 1 kHz robot state:
- **task-space dead zone**: the spring wrench (K × error) at which the EE starts to move,
  per axis and for each direction (N, N m). This is the number to compare with
  `task_wrench_dead_zone`;
- **per-joint breakaway**: `tau_J_d` (desired, without gravity) when |dq| first rises above
  a threshold;
- **torque-loop gap**: `tau_J_d` against `tau_J − g` (= `tau_ext` at rest) while stuck. This
  is the commanded torque that never reaches the link;
- **elasticity**: motor angle θ minus link angle q, against torque;
- with friction comp on: the dead zone left over, and whether it limit-cycles.

The moving parts of M and D also give a **Coulomb + viscous fit per joint**: the
torque-loop gap against dq. That is a measured calibration that can replace the uniform
0.3 N m friction comp.

### Dynamic payload cross-check

On the unclipped 10 % and 50 % moves, the predicted payload wrench
m(a − g) + Iα + ω × Iω uses accelerations from the encoders (q, dq, then a
Savitzky-Golay ddq) through FK. It is compared with the change in `K_F_ext_hat_K` during the
move, once with the Desk values and once with the identified values.

Mass and COM show up: 0.58 kg × 4.5 m/s² ≈ 2.6 N. **Rotational inertia will not show up**:
I ≤ 0.0012 kg m² × 8.5 rad/s² ≈ 0.01 N m, which is below the torque noise and offsets
(about 0.1-0.3 N m). The test can therefore only bound how much the hand inertia matters for
the OSC at these speeds; it cannot validate the inertia values. Mass and COM are what the
test can actually check.

## Recording

- **rosbag2 (MCAP, zstd)**, one bag per session:
  - `franka_robot_state_broadcaster/robot_state`: full `FrankaRobotState` at 1 kHz: q, dq,
    θ, dθ, tau_J, tau_J_d, dtau_J, tau_ext_hat_filtered, O/K F_ext, inertia_ee/load/total,
    O_T_EE, robot_mode and errors;
  - `/joint_states`;
  - the controller's `cartesian_state_fast` at 1 kHz (unclipped goal, clipped target,
    measured pose, tau_task, tau_nullspace, tau_command, wrench, clip flags, clip and fault
    limits in effect) and `controller_state`;
  - both goal topics;
  - `/osc_characterization/events`: JSON markers for each block, move, arrival and probe
    segment, with factor labels.

  Expect about 2-3 GB per hour compressed. Writing a bag does not depend on the test
  process surviving: the 2026-11-10 run lost its data when the process died during the
  save.
- A JSON report per session: seed, sampled orientations, plan checks, controller
  parameters, Desk F_T_EE and inertia readback, `set_load` calls and their readback, and
  block status.
- Offline analysis reads the bag and the report, and writes per-condition tables and plots
  to `debug/osc_characterization/<session>/`.

## Safety gates

- An operator confirms each new orientation set, and the first block of each speed level.
  Keep the E-stop in hand.
- Abort on: a tracking error above 2× the predicted lag plus 10 mm; the controller leaving
  the policy phase; any rejection or reflex; `robot_mode` other than move; a joint inside
  0.10 rad of a limit.
- The 50 % unclipped block runs only after the 10 % unclipped block of the same session
  stayed under its predicted lag. Its first repeat runs at 25 % first.
- The collision behaviour (Desk thresholds) is left as it is.

## Run order and time

1. **ros-sim dry run** of the whole script. `set_load` is skipped. Friction comp comes from
   `arm_friction_compensation`, and `arm_wrench_dead_zone:=worst` is a known-answer test for
   the probe: it should recover 5 N / 1 N m. The orientation draws are checked by eye in
   the viewer.
2. Hardware session 1: friction comp off. Session 2: friction comp on. Each takes about
   110-120 min, made up of:
   - 18 M blocks at about 3.5-5 min each;
   - 2 × (S + D) at about 12 min;
   - 2 reference blocks.

   For a short version, about 45 min per session: 2 repeats instead of 3, the D probe only
   with payload modelled, and the clipped 50 % block dropped.

## What the dead-zone probe means for the clip

The clip limits the target against the *measured* pose on every tick, so it caps the spring
wrench. Per axis that cap is 565 N/m × 0.02 m = 11.3 N and 28 N m/rad × 0.097 rad = 2.7 N m;
the transition clip raises it to 28 N and 7 N m. A clip only gets the arm out of a stall if
that cap is above the breakaway wrench. The D probe measures the breakaway wrench, which
gives a lower bound for the transition clip on this arm:

    clip_min = breakaway / K

## Decisions (2026-11-11)

1. Payload: `set_load` on a zero-mass copy of the Desk profile.
2. The 50 % unclipped level stays, gated behind the 10 % unclipped block.
3. The wrist spin is included.
