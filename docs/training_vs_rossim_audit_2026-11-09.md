# Training (Isaac Lab) vs ros-sim (MuJoCo) audit, 2026-11-09

The goal: ros-sim should differ from the Isaac Lab environment the exp19 student
was trained in only by the physics engine and by the policy's actions reaching
the plant through ROS drivers. This document lists every difference found, what
each one does to the rollout, and the fix.

Reference student: `m24-dp3-exp19r01-windback45tilt30-clock-force-nohv-da-s0-20261109-105240-553825`
(official mimic hand, wound/tilted reset, progress clock, 35-D proprio). Its
training chain is `m24_threading_fr3v2_joint_limit_guard_windback45_tilt30_obsnoise_sequential_collection_dr.yaml`
in ForgeUltra b968687, followed through `extends:` down to
`forge_franka_env_cfg.py`. Paths: `FU` = `~/code/forgeUltra`, `WS` = this workspace.

Method: four read-only audits (arm + controller, hand + force sensing, scene +
thread + camera, policy loop + coordinator), each citing file and line on both
sides. The items marked **verified** below were re-read in the source by hand;
the rest carry the audit's citation. ForgeUltra's USD assets are Git LFS pointers
on this machine, so anything baked into a USD could only be read from the script
that authored it.

## 1. The mismatches, ranked by expected effect

| # | Layer | Training | ros-sim | What it changes | Fix |
|---|---|---|---|---|---|
| 1 | Hand drives | Pinch drives (thumb_joint_0/1, index_joint_0) capped at **0.5 N m** (`hand_joint_effort_limits_nm`, FU `config/tasks/m24_threading_force_limited.yaml:14-17`, applied after the 20 N m ceiling in `forge_franka_env.py:837-846`). **Verified.** | **20 N m** (`frozen_hand_effort` only: WS `make_hand_mjcf.py:178-180`, `inspire_hand_right_isaac.xml` forcerange ±20; `mujoco_threading_env.py:73 PINCH_EFFORT = 20.0`) | 40x the squeeze authority. Training's drive saturates at 17 mrad of position error (k = 30), so the grip is a torque-limited, compliant squeeze of ~15-20 N per pad (training comment: 12/7 N at 0.3 N m, 27/21 N at 1.0 N m). ros-sim tracks the pinch stiffly; exp19 squeezed 33-45 N. The nut pushes the saturated training thumb back ~0.1 rad (ISAAC_ENTRY_PINCH 1.0717 vs commanded 1.185), so the hand angles the student observes at grasp are also different. Closing speed into the nut differs by the same factor. | Cap the three pinch actuators at 0.5 N m in `make_hand_mjcf.py` (ISAAC_DRIVES effort for the leaders), regenerate the hand MJCF and both policy scenes; same in `mujoco_threading_env.PINCH_EFFORT`. Must be done together with #4. |
| 2 | Contact | Pad-nut friction = PhysX average of pads (1.5, collection U[1.2, 1.8]) and nut (`held_asset_friction_override` 1.0, collection U[0.8, 1.2]) = **1.25** nominal, 1.0-1.5 (FU `forge_franka_env_cfg.py:473`, `m24_threading_force_curriculum_grip.yaml:17`, `robot_control.py:70-98`, `forge_franka_env.py:1955`). **Verified.** | **0.75**: pads 0.75 (`make_hand_mjcf.py:113`), nut 0.75 (`mujoco_threading_env.py:61`, scene xml); MuJoCo takes the element-wise max. | The friction cone is 40 % narrower than anything the student trained in. The same shear/normal ratio that held in Isaac slips here. The `make_ros_sim_scene.py:148` comment "training's nut friction" quotes the YAML material that the override replaces at startup. | Pads 1.5 and nut 1.0 in MuJoCo give max = 1.5, not the 1.25 average. Set both pad geoms and the nut to **1.25** (MuJoCo max of equal values) in `make_hand_mjcf.CONTACT_FRICTION` and `mujoco_threading_env.CONTACT_FRICTION`; regenerate. |
| 3 | Thread | One-way: a prismatic position drive (200 kN/m, 200 N s/m, 1000 N cap) follows a target computed from the measured nut rotation every physics step; axial force deflects the drive ~0.1 mm at 20 N and produces no twist torque (FU `thread_pair.py:10-12, 196-207`, `forge_franka_env.py:1446-1522`). **Verified.** Twist joint: armature 0, friction 0; passive Coulomb 0.002 N m (x U[0.5, 2] log-uniform in collection) and viscous 0.0002 applied as a smoothed effort capped at the stopping torque. | Holonomic joint equality `nut_axial = c0 + 4.77e-4 * nut_twist` (WS `make_ros_sim_scene.py:265-271`, `mujoco_threading_env.py` `_add_thread_pair`): axial force F produces twist torque 4.77e-4 F, so 20 N along the bolt is 0.0095 N m, five times the 0.002 N m frictionloss. Twist armature 0.001 kg m^2 (~75x the nut's own I_zz, inference). | Pressing the nut turns it, lifting it unturns it, in ros-sim only. The exp19 recording shows 646 deg forward and 431 deg backward in the policy phases, 380/246 of them on ticks with zero pad force. The release gate reads this motion as real turn. The scene docstring ("this is Isaac's thread, not a port defect") is wrong; `thread_pair.py` says the opposite. | Replace the equality with Isaac's mechanism in `thread_pair_plugin.cpp`: each step read `nut_twist`, set the axial target `c0 - pitch * sign * theta / 2pi`, drive `nut_axial` with a 200 kN/m, 200 N s/m position actuator clamped at 1000 N, and keep the twist hinge passive with frictionloss 0.002 / damping 0.0002 and armature 0. Keep the limits and the hold. Apply to both policy scenes and the MuJoCo-only loop. |
| 4 | Hand joints | `ImplicitActuatorCfg.friction` 0.132 (thumb_0/1), 0.07456 (finger_0, thumb_2), 0.01276 (finger_1, thumb_3) (FU `assets/fr3_inspirehand/robot.py:221-234`). Under Isaac Sim 5.1 the arm code states the same API writes **constant friction torques in N m** (`robot_control.py:223-228`). **Unverified for the hand**: Isaac Lab is not installed here. | `IsaacHandPlugin` treats the numbers as PhysX 4 coefficients: `frictionloss = coeff * (|F| + |T|)` of the transmitted spatial force each step (WS `isaac_hand_plugin.cpp:186-202`). | If training's value is a constant 0.132 N m, the deployment thumb has ~0 friction unloaded and ~1.3 N m at 10 N of pad load. With #1 applied (0.5 N m drive), a load-dependent 1.3 N m would lock the fingers; with 20 N m it is negligible, which is why it has not shown. | Decide the semantics on the training machine: print `robot.root_physx_view.get_dof_friction_coefficients()` / the static-dynamic friction tensors for the hand joints under Isaac Sim 5.1. Then either make the plugin write constant `frictionloss` = the N m value, or keep the coefficient model. Do this before or with #1. |
| 5 | Runner | Episode start: fast reset writes the hand to the grasp posture (1.185078562 / 0.050614548 / 0.214675498) and the arm to the reset joints at step 0; no open-then-close (`reset_settle_time_s 0.0`, FU `robot_control.py:948-963`, `forge_franka_env.py:2853-2856`). **Verified.** | `--hand-start release` (default) opens thumb/index to `RELEASE_START_PINCH` (1.093, 0.0001, 0.03), then commands the grasp and takes the first sample while the hand is still closing (WS `hardware.py:2356-2400`). | The first observations show a hand 0.09/0.05/0.18 rad from the grasp with zero pad force; exp19's cycle 1 never starts that way (cycles 2-10 do start from the open waypoint-1 posture, so the student has seen closing hands, but not at progress 0). | Run exp19 with `--hand-start grasp`, or make `grasp` the default for profiles whose recipe has `reset_settle_time_s 0`. |
| 6 | Runner | 10-cycle episodes take ~1782 steps (the clock duration 118.77 s is the dataset median); Isaac eval ran `--max_steps 4500`. | `--max-steps 1500` (100 s) | A run at training pace is cut at progress 0.84 before cycle 10, and the student never sees late-episode clock values. The exp19 run was interrupted by hand; the next would hit the budget. | `--max-steps 4500` for exp19, and make the default follow the checkpoint (`horizon_steps` x 2.5). |
| 7 | Runner | Return gate: collection 5 mm / 5 deg / 0.15 rad; the Isaac eval that scored this checkpoint used **10 mm / 10 deg** (FU `evaluate_flow.py:105-127`, `evaluation_report.json`). | `--return-tolerance-mm 5 --return-tolerance-deg 5` defaults; the 2026-11-09 run used 12 / 7. | A return Isaac counts can wait out 16 s and fail here. The 2026-11-09 returns measured 3.3-4.2 mm / 6.3-6.9 deg against the wired reset grasp. | Use 10 / 10 to compare against the Isaac report. |
| 8 | Controller | Collection ran with task DR **on** (`m24_dr_recipe`, `task_domain_randomization=True`, FU `tools/dataset_builder/collect_m24_success_term_dataset.py:99-109, 147, 174-188`). **Verified.** Per episode: wrench dead zone U(0,1) x [5,5,5,1,1,1] N / N m, task gains x[0.71, 1.41], clip thresholds x[0.78, 1.29], EMA U[0.025, 0.1], bolt action-frame offset N(0, 2 mm)/axis, arm friction x[0.8, 2.0], armature/damping x[0.7, 1.3], camera pose +-11 mm / +-1.1 deg, nut yaw +-30 deg, spawn depth over the top 20 mm, thread resistance x[0.5, 2], nut mass +-5 g, table pose +-5/10 cm, +-2.5 deg. | Fixed nominal everywhere: dead zone 0, gains 565/28, thresholds 0.02/0.097, no frame offset, friction x1, nominal spawn, calibrated camera. | Every deployment value is inside the training range, so these narrow the distribution rather than shift it. Listed because the dead zone (0 to 5 N removed from an 11.3 N full-clip command) is the one controller term with no deployment counterpart at all, and because an optional DR mode in ros-sim would let us reproduce the data distribution. | No change needed for parity at the nominal point. Optional later: a `--randomize` mode in the runner/launch for the dead zone, gains and spawn. |
| 9 | Scene | Table (mu 0.8/0.6) and bolt holder (convex decomposition) are colliders; nut-bolt contact pair filtered, bolt SDF collider otherwise active (FU `author_lab_bench_table.py:86`, `forge_franka_env_cfg.py:675`, `forge_franka_env.py:1332-1360`). | Table, holder and bolt all `contype 0` (visual only). | Fingers that miss the nut pass through the bolt shaft, holder and table in ros-sim; in training they are stopped. Normal cycles are unaffected. | Give the bolt its collider with a `<contact><exclude>` for the nut pair; holder and table colliders as boxes/hulls (the holder's hex pocket is filled by a hull, so use the plate only). |
| 10 | Camera | Isaac renders 320x180 directly with the scaled K; collection jitters the pose and randomizes textures, dome lighting (1000-4000), background replacement; offline training adds the `realistic` depth corruption (exp19's `run_provenance.json`: `depth_augmentation` true, `depth_noise_model` realistic, `dextrah_rgb_augmentation` false). | Renders 640x480, crops rows 60-420, nearest-neighbour subsample to 320x180 (WS `camera_calibration/calibration.py:458-500`); clean depth quantized to 1 mm; fixed walnut/grey/black appearance, headlight + one directional light. | Pose and K match exactly. Edge aliasing differs; the fixed look is plausibly inside the training distribution but is not a sample of it. | Low priority. Render at 320x180 in MuJoCo with the scaled K (removes the subsampling), optionally add the depth noise model at inference for A/B only. |
| 11 | Arm | PhysX clamps joint velocity at 2.62 (j1-4), 5.26, 4.18, 5.26 rad/s (FU `robot.py:256-264`). | No velocity clamp. | Only if the student commands fast wrist motion. | Optional: a per-joint velocity clamp in the arm MJCF is not available; leave. |
| 12 | Arm | Controller mass matrix: PhysX generalized mass matrix of the live arm+hand (includes 0.195 armature if PhysX does; unverified). | Pinocchio CRBA with the hand locked, `arm_armature` all zeros (`controllers_sim_policy.yaml:115`). | Affects only the nullspace term (kp 10). | Set `arm_armature` to 0.195 x 7 once the PhysX question is answered. |
| 13 | Arm | Inner loop 120 Hz (decode + clip + law per substep, 8 per policy step). | 1 kHz clip and law; torque rate limit 1 N m per cycle. | Same algorithm on a finer grid; contact transients shaped differently. | Leave. |
| 14 | Arm | Damping velocity: mean of thumb/index pad body linvels (includes finger motion), flange angvel. | `J dq` at the tool point with the fingers locked. | Transient difference while the pinch moves. | Leave, or feed the finger joint velocities into the tool-point velocity. |
| 15 | Obs | Arm joint torque = PhysX projected joint-reaction force, normalised by (87,87,87,87,12,12,12), with exp19 noise N(0, 0.01), gain U[0.8, 1.2], bias U[-0.02, 0.02] (FU `franka_joint_torque.py:66-72`, `..._adr80_obsnoise.yaml`). | `commanded` = `tau_task + tau_nullspace` in ros-sim; `external` on hardware (WS `hardware.py:54-116`). | 7 of 35 proprio values. The two codebases agree that no candidate has been shown equal to PhysX's quantity. | Record PhysX projected joint forces and the applied effort target in one Isaac episode and compare; `utils/compare_torque_candidates.py` is the deployment half. |

## 2. What matches (checked, no work needed)

Arm: Menagerie FR3 v2 link masses/CoM/inertia, joint ranges, nominal friction
0.2 N m, armature 0.195, damping 0.003, effort limits 87/12, gravity off on the
robot, Kp 565/28 with Kd = 2 sqrt(Kp), axis-angle rotation error, mass-weighted
nullspace with q0 = the wound reset joints and kd 6.3246, no Coriolis, clamp
+-100 N m, action bounds 0.05 / 1.0, thresholds 0.02 / 0.097, the yaw window,
`ee_base_orn_euler` (pi, 0, pi/4), yaw seam 280 deg, grasp-frame construction
(pad midpoint, thumb->index x, world-down z transported with the flange), the
Jacobian shift, joint-limit guard 0.15 rad.

Hand: URDF geometry and 12 joint ranges, drive gains 30/3 on the six leaders,
4.76/0.21 inactive, follower gains, armature per joint, velocity limit 1 rad/s,
mimic ratios and target-level coupling, inactive fingers at 1.333 (training's
`inactive_hand_at_upper_limit` overrides the reset file's 1.0999), grasp posture
values, pinch ranges and the relative decode, q10 order, hand velocities
omitted, fingertip force = net pad force projected on the tip -X, clamped >= 0,
/20, no low-pass, tip-frame yaws +150 / -15 deg.

Scene: robot base, table position and size, surface z 0.0065, bolt base
(0.61, 0, 0.05), M24 geometry, nut mass 0.05, nut spawn 0.00766 and yaw 75 deg,
thread limits -0.02614..0.01216, direction sign -1, thread hold during
release/return, camera pose and intrinsics, 15 Hz camera, DP3 crop box and 4096
points, RGB 0-1 / depth metres with invalid = 0.

Loop: 35-D layout and scaling, unified -> native scale (EMA bypassed on both
sides for the unified codec), clip to [-1, 1], ensembler horizon 8 / decay 0.5
/ one action per inference, 16 integration steps on CUDA (2 on CPU), progress
clock from sim time with duration 118.77 s, process-phase one-hot timing, phase
clock 11 / 15 / 240 steps, release at target - 5 deg from the nut twist, 60 deg
grid with the 55-65 clamp, previous-action seeding at reset and at each cycle
boundary, the student (not a script) acting through release and return.

## 3. Evidence from the 2026-11-09 ros-sim run

`logs/policy_rollout/ros_sim-20261109-124519-871171` (exp19, interrupted after 3
cycles in 1016 steps): policy phases of 172-194 steps vs run09's ~40; the nut
reaches 35-43 deg, runs backward to -18 / -76 / -71 deg, then a second attempt
passes the gate; pad forces 33-45 N (run09 8-13 N); fingertip gap at grip 18-21 mm
(run09 11-13); arm at the 20 mm clip ceiling on 56 % of policy ticks (median
target-measured 21 mm; run09 11 mm). #1 (no effort cap) explains the forces,
#2 (friction) the slip at those forces, #3 (two-way thread) the backward motion on
ticks with zero pad force. Isaac's own nominal eval of this checkpoint reached 10
cycles in 3/48 environments, so parity will not make exp19 reliable; it makes
ros-sim a valid test of it.

## 4. Open items that need the training machine

- Hand joint friction semantics under Isaac Sim 5.1 (#4). Decides how #1 is applied.
- Whether PhysX's generalized mass matrix includes armature (#12).
- Arm joint torque observation equivalence (#15).
- The actual pad forces and settled pinch angles in an exp19 episode (would
  confirm the 0.5 N m effect on the observed hand angles and the entry pinch).

## 5. Proposed order

1. #2 friction and #1 effort cap, with #4 resolved or the plugin's friction set
   to the constant-torque reading; regenerate `inspire_hand_right_isaac.xml`,
   `inspire_hand_on_flange_isaac.xml`, both policy scenes.
2. #3 one-way thread in `thread_pair_plugin.cpp` and `mujoco_threading_env.py`.
3. Runner: `--hand-start grasp`, `--max-steps 4500`, `--return-tolerance 10/10`
   for exp19; make the defaults follow the checkpoint.
4. #9 colliders for bolt (nut pair excluded), holder plate, table.
5. Re-run run09 and exp19 in the same ros-sim so the baseline moves with the plant.
6. #10, #12, #15 after the training-machine probes.

## 6. Status (2026-11-09, uncommitted)

Done:

- **#2 friction.** The rollout hands (`--isaac`, `--rh56`) carry 1.25 on the pad
  links (`thumb_distal`, `index_intermediate`) and 0.875 on the rest, the nut
  0.875: MuJoCo's maximum then gives training's pad-nut 1.25 and link-nut 0.875.
  The plain replay hand is unchanged.
- **#3 thread.** `thread_coupling` names only `nut_axial`, and the thread pair
  plugin (`ThreadingScene._apply_thread_drive` in the MuJoCo-only loop) writes the
  target for the measured twist every physics step. Twist armature 0, as in
  `thread_pair.py`. The first attempt kept `joint2` with a zero slope; MuJoCo's
  constraint regularization then includes the twist dof's inverse inertia and the
  axial stiffness fell to about 4 kN/m, so the joint has to be absent. The coupling
  has no velocity feedforward, unlike Isaac's drive, and lags by about 8 ms of
  axial travel (0.3 mm at a free spin of 85 rad/s, microns at threading speed).
- **#5-#7 runner.** `--hand-start auto` is `grasp` in ros-sim for fast-reset
  profiles (`TaskProfile.fast_reset`). `--max-steps` and the return tolerances
  default to 4500 and 10 mm / 10 deg in ros-sim. Hardware defaults are unchanged
  (`release`, 1500, 5 mm / 5 deg).
- **#9 colliders**, ros-sim scenes only: bolt shank cylinder (nut pair excluded),
  holder plate box and collar cylinder, table. The bolt also now has training's
  105 deg yaw, which seats its head in the holder's pocket.
- **#1 effort cap**: the three pinch drives are capped at 0.5 N m in the Isaac
  hand and the MuJoCo-only loop (every old-hand recipe inherits
  `m24_threading_force_limited.yaml`; RH56 keeps its own drives). Applied before
  #4 was settled, so the plugin still uses the coefficient friction model. Exp19
  headless, ros_sim-20261109-135944-434757: 3/3 cycles in 569 steps, policy
  phases 96/72/70, 5.7 deg backward; ros_sim-20261109-140153-673673 (10 cycles
  requested): 3 cycles, 2.5 deg backward, stopped on fr3_joint5 2.745 rad in
  cycle 4 with j5 peaks rising 1.69, 2.2, 2.35, 2.72 rad. Thumb peak force was
  0.5-1.5 N in cycles 2 and 4 of that run against index 8.5-9.9 N.

Left on purpose:

- #8: nominal deployment values sit inside every training range; the Isaac eval
  this is compared with also runs nominal.
- #10 pixel sampling: taking every second pixel of a 640-wide render is the same
  pinhole sampling as rendering 320 wide, apart from a quarter-pixel offset.
  Appearance waits for Test 5's reference frame.
- #11: the arm peaked at 24 % of Isaac's joint velocity limits in the recorded runs.
- #13, #14: these belong to the ROS controller path, the permitted difference.

Waiting on the training-side probes: #4 (hand joint friction), #12, #15, the
entry pinch.
