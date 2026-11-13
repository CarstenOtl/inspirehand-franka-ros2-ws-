# Checkpoints

Everything in this directory except this file and `.gitignore` is ignored by
git (see `.gitignore` here and the `*.pt` rule in the repository root). The
files are large binaries; copy them in by hand from wherever the training run
exported them.

## Expected layout

```
checkpoints/
  m24-dp3-run09fr3v2-nominal-cycles-noclock-force-nohv-da-s0-20261102-173148-415770/
    nn/checkpoint.pt                   the deployed student (policy_rollout.DEFAULT_CHECKPOINT)
    nn/flow_config.json, nn/run_provenance.json, evaluation/, summaries/   training-run side files
  m24-dp3-nominal-holder-da-force-nohv-s0-20261030-173636-450280/
    nn/checkpoint.pt                   the same student on the FR3 v1 arm, with the progress clock
  m24-dp3-nominal-calib1029-da-s0-20261030-112229-116599/
    nn/checkpoint.pt                   the same student without the bolt holder
  m24-dp3-nominal-da-force-nohv-s0-20261029-113811-885629/
    nn/checkpoint.pt                   the same student trained on the 2026-09-10 camera pose
  fr3_offline_flow_matching_force_obs_v1_25102026/
    nn/checkpoint.pt                   the 38-D RGB-D student that observes forces
  fr3_offline_flow_matching_force_obs_25102026/
    nn/checkpoint.pt                   the 29-D RGB-D student of the force-obs teacher
  fr3_offline_flow_matching_parallel_vision_13102026/
    m24-dr-disturbed-dp3-no-finger-velocity-seed0-20261013-135143-648368/
      nn/checkpoint.pt                 the previous 26-D DP3 student
  sequential_threading_cycle10_hybrid_teacher_d415_20ep/
    checkpoint.pt                      an earlier DP3 student, paired with reference_episode/
  fr3/
    vision_head-fr3_d415_nut_sweep_20260910T152223_calibration/
      nn/best.pt                       DP3 vision-head pretraining checkpoint (provenance only)
```

## `m24-dp3-exp20r01-proprio-friction-returndz7.5-tclip-smooth-clock-force-nohv-da-s0-20261111-173541-799500-*/.../nn/checkpoint.pt`

The exp_20 proprio-friction student, recollected for the return: task-wrench
dead zones, return poses randomized up to 15 mm / 10 deg, a 2.5 s settle, the
transition clip and a smoother return trajectory. Network and observations as
the proprio-friction student below; only the progress horizon differs. Resolves
to `mimic_windback45_tilt30`. Not `DEFAULT_CHECKPOINT`.

| field | value |
|---|---|
| sha256 | `55d8723f26b3f1b129fdd162f578673de4e4d16f2748953a26e5253afde53f01` |
| size | 101 MB |
| conditioning | progress 98.2 s / 1473 steps, cyclic phase |
| code | ForgeUltra `ab34bc7` (clean) |
| training data | 36 `sequential-threading-dual-supervision-20261111-163042-486505-rebuilt-20261111-171623-536975` episodes |
| Isaac closed loop | 3 episodes x 16 environments, return gate 15 mm / 10 deg (`returndz7.5-ret15x10`), no DR: 477 cycles, 47/48 completed 10, 0 transition failures |

## `m24-dp3-exp20r01-{extended,proprio-friction}-rollback3-clock-force-nohv-da-s0-*/.../nn/checkpoint.pt`

Two DP3 students of the exp_20 teacher below, on the hardware-fitted hand. The
network layout and observations match the exp19 students; only the progress
horizon differs. Both resolve to the `mimic_windback45_tilt30` task profile. Not
`DEFAULT_CHECKPOINT`.

| | extended | proprio-friction |
|---|---|---|
| run | `...-20261110-113938-776156` | `...-20261110-120354-491945` |
| sha256 | `4d0b498411c01eb8933336a02f9f1419adb747134af26a195feaee14e29e2fac` | `223d8e7e847231f828abb8b1d17a15f9efd76427cb0c3b7684907884a2527bdb` |
| size | 106 MB | 106 MB |
| conditioning | progress 129.5 s / 1943 steps, cyclic phase | progress 120.3 s / 1804 steps, cyclic phase |
| collection recipe | `..._windback45_tilt30_obsnoise_hwfit_sequential_collection_dr.yaml` | `..._windback45_tilt30_obsnoise_hwfit_sequential_collection_proprio_friction.yaml` |
| training data | 23 `sequential-threading-dual-supervision-20261110-103754-033968` episodes | 33 `sequential-threading-dual-supervision-20261110-103805-743944` episodes |
| Isaac closed loop | 3 episodes x 16 environments, return gate 10 mm / 10 deg (`extended-ret10`): 36 cycles, 0/48 completed 10, 19 transition failures | none shipped with the checkpoint |

## `m24-fr3v2-exp20-run01-hwfit-extended-2026-11-09_23-43-16-20261110T112616Z-1-001/m24-fr3v2-exp20-run01-hwfit-2026-11-09_23-43-16/nn/last_ForgeTg2_ep_200_rew_517.838.pth`

The exp_20 RL teacher on the hardware-fitted hand, not a distilled student: an
rl_games LSTM actor over 46 state observations, without vision. The runner
recognises the `.pth` and uses `policy_rollout/teacher.py` with the
`mimic_windback45_tilt30` profile.

- ros-sim only: the actor observes the nut pose, which the workcell does not
  measure, so the hardware backend refuses it;
- one turn per run: it was trained on approach, grip and turn and has no release
  or return, so `--cycles` is forced to 1 and the run ends on the task's success
  (12 ticks within 2.5 deg of a 60 deg turn with both pads loaded).

| field | value |
|---|---|
| sha256 | `fbaa7b69a70d73f15d8a54934b1d0ec32ce6afbda74cccfb4cf958592d1c799d` |
| size | 215 MB |
| training | recipe `m24_threading_fr3v2_joint_limit_guard_windback45_tilt30_adr80_obsnoise_hwfit.yaml` (committed afterwards as ForgeUltra ac8d948), 200 epochs, threading success 0.87 |

## `m24-dp3-exp19r01-proprio-friction-rollback3-clock-force-nohv-da-s0-20261109-143521-315193-20261109T170730Z-1-001/m24-dp3-exp19r01-.../nn/checkpoint.pt`

The exp19 teacher below, collected with a reduced randomization set. The network
layout and observations match the student below; only the progress horizon
differs. Same `mimic_windback45_tilt30` task profile, resolved automatically.
Not `DEFAULT_CHECKPOINT`.

- randomized: proprioception noise (EE pose, fingertip force, FR3 joint torque),
  FR3 joint friction (x0.8 to x2.0), hand pad friction, head-camera pose and
  visual textures;
- nominal: nut spawn yaw and depth, contact and thread physics, FR3 armature and
  damping, controller gains and dead zone.

Because it trained against arm friction, `controllers_policy_no_friction_comp.yaml`
is the matching hardware controller config.

| field | value |
|---|---|
| sha256 | `bf3cda5fac37a62baa65661336999db6840f8ef3e7df56ff20aeb0aef42717b9` |
| size | 106 MB |
| conditioning | trajectory progress (114.7 s / 1720 steps) and cyclic process phase |
| training | ForgeUltra b968687 (clean), collection recipe `m24_threading_fr3v2_joint_limit_guard_windback45_tilt30_obsnoise_sequential_collection_proprio_friction.yaml` |
| training data | 40 `sequential-threading-dual-supervision-20261109-133804-877920` episodes |
| Isaac closed loop, nominal | 3 episodes x 16 environments, return gate 10 mm / 10 deg: 356 cycles, 31/48 completed 10, 1 transition failure |
| Isaac closed loop, proprio-friction | same: 269 cycles, 23/48 completed 10, 0 transition failures |

## `m24-dp3-exp19r01-windback45tilt30-clock-force-nohv-da-s0-20261109-105240-553825-20261109T123227Z-1-001/m24-dp3-exp19r01-.../nn/checkpoint.pt`

exp_17's wound, tilted reset on the old hand of run09 and run10 (official
Inspire hand, linear map, mimic followers), still in the export zip's nested
folder. Not `DEFAULT_CHECKPOINT`. The runner resolves it to the
`mimic_windback45_tilt30` task profile (`forge_osc.CHECKPOINT_TASK_PROFILES`).
In ros-sim, launch `sim_policy.launch.py task_profile:=mimic_windback45_tilt30`,
keeping the default `hand_model:=mimic`; the MuJoCo-only loop refuses it.

- reset pose wound 45 deg about the bolt and tilted 30 deg wrist-up
  (`config/policy_home_windback45_tilt30.yaml`), nut spawn yaw 75 deg;
- orientation-action frame `ee_base_orn_euler` yaw +45 deg, yaw seam 280 deg;
- pinch actions are joint angles about the grasp over the URDF's ranges, as for
  run09. The entry pinch is the grasp posture, inferred from the fast reset and
  not measured against a reference episode.

| field | value |
|---|---|
| sha256 | `9d8249daa1bb4eae738d46b1edb1ecdce01075f692ae2757f1beb820b4bd06e5` |
| size | 106 MB |
| proprio, control, vision | as run09 below |
| conditioning | trajectory progress (118.77 s / 1782 steps) and cyclic process phase |
| training | ForgeUltra `forgeUltra_force_obs` 6c9b02a (dirty), task `Isaac-Forge-Franka-Threading-V2-M24-FR3v2-v0`, collection recipe `m24_threading_fr3v2_joint_limit_guard_windback45_tilt30_obsnoise_sequential_collection_dr.yaml` (untracked there) |
| training data | 36 `sequential-threading-dual-supervision-20261109-093524-438290` episodes |
| Isaac closed loop | 3 episodes x 16 environments, nominal, return gate 10 mm / 10 deg: 117 cycles, 3/48 completed 10, 1 transition failure |

## `m24-dp3-run09fr3v2-nominal-cycles-noclock-force-nohv-da-s0-20261102-173148-415770/nn/checkpoint.pt`

The deployed student: the recipe of the one below, trained in Isaac on the
FR3 v2 arm (task `Isaac-Forge-Franka-Threading-V2-M24-FR3v2-v0`, the model both
MuJoCo backends use) with the bolt holder, and without trajectory-progress
conditioning ("noclock"): it sees only the cyclic process phase, so no backend
feeds it a clock. Otherwise its `flow_config.json` matches the one below, so
the 35-D layout is unchanged. Copied in from the `...-20261103T125536Z-1-001.zip`
export.

| field | value |
|---|---|
| sha256 | `ccf44fa0c8a575edf61f0a039ffcef99c2d6874e3507fd88bee46ed61c5ba9d8` |
| size | 105 MB |
| epoch | 100 (EMA decay 0.999), final flow MSE 0.0117 |
| proprio, control, vision | as below |
| conditioning | cyclic process phase only (no trajectory progress) |
| training | ForgeUltra `forgeUltra_force_obs` e870374 (dirty), depth augmentation `realistic`, no RGB augmentation |
| training data | 68 `sequential-threading-dual-supervision-20261102-145157-329200-completed-cycles-20261102-171410-489253` episodes |
| Isaac closed loop | 3 episodes x 16 environments, nominal: 45 cycles, 1/48 completed 10, 2 transition failures |

## `m24-dp3-nominal-holder-da-force-nohv-s0-20261030-173636-450280/nn/checkpoint.pt`

The previous deployed student: the recipe of the one below, retrained on data collected
with the hardware's M24 bolt holder in the scene (the bolt stands in its hex
pocket, the tabletop raised to the holder's foot; both simulators reproduce
it, see `policy_rollout/mujoco_scene.py`). Its `flow_config.json` differs from
the one below only in the progress horizon, so the 35-D layout is unchanged.
Copied in from the `...-20261031T144813Z-1-001.zip` export.

| field | value |
|---|---|
| sha256 | `1202527878cfc6b28238b8e6f68063d1563eca89bfa741afc010379b43f8648d` |
| size | 101 MB |
| epoch | 100 (EMA decay 0.999), final flow MSE 0.0108 |
| proprio, control, vision | as below |
| conditioning | trajectory progress (72.3 s / 1084 steps) and cyclic process phase |
| training | ForgeUltra `forgeUltra_force_obs` ee3ca0e (dirty, not pushed), depth augmentation `realistic`, no RGB augmentation |
| training data | 40 `sequential-threading-dual-supervision-20261030-110819-178662` episodes |
| Isaac closed loop | 3 episodes x 16 environments: nominal 428 cycles, 41/48 completed 10; randomized (seed 7) 466 cycles, 46/48; 0 transition failures |

## `m24-dp3-nominal-calib1029-da-s0-20261030-112229-116599/nn/checkpoint.pt`

The previous deployed student: the recipe of the one below, retrained on images
rendered from the 2026-10-29 camera calibration
(`assets/camera/calibration/20261029T173220_626741Z`, 25.5 mm / 3.55 deg from
the 2026-09-10 pose). Its `flow_config.json` differs from the one below only
in the progress horizon, and its first proprio layer has the same column
fingerprint, so the 35-D layout is unchanged. The checkpoint does not record
the camera pose; the run name and the training side do.

| field | value |
|---|---|
| sha256 | `495618add3f72c224810cea083d30beb54ed068922076d40f66e452423cbff23` |
| size | 106 MB |
| epoch | 100 (EMA decay 0.999), final flow MSE 0.0113 |
| proprio, control, vision | as below |
| conditioning | trajectory progress (72.17 s / 1082 steps) and cyclic process phase |
| training | ForgeUltra `forgeUltra_force_obs` ee3ca0e (dirty, not pushed), depth augmentation `realistic`, no RGB augmentation |
| training data | 40 `sequential-threading-dual-supervision-20261029-191033-137205` episodes |
| Isaac closed loop | 3 episodes x 16 environments: 461 cycles, 44/48 completed 10, 0 transition failures |

## `m24-dp3-nominal-da-force-nohv-s0-20261029-113811-885629/nn/checkpoint.pt`

The previous deployed student, trained on the 2026-09-10 camera pose. To run
it, put that pose back into `utils/camera_calibration/fr3_realsense_dp3.yaml`
(the file's header lists it) and regenerate the ros-sim scene. It observes
forces and drops the finger velocities.

| field | value |
|---|---|
| sha256 | `387e5899331e11f6edf313056e7d4dddcd8f4d88797d2d369bdcce72ac44ae1b` |
| size | 106 MB |
| epoch | 100 (EMA decay 0.999; `ema_model` is loaded by default) |
| control | 9-D OSC, `unified` action representation, native scale `[3,4,16,3,3,3,2,2,2]` |
| proprio | 35-D: `q10 + arm dq7 + previous filtered native action + [thumb, index] pad force (N / 20) + 7 FR3 joint torques / effort limits (clipped to 1)` |
| action horizon | 8 |
| backbone | trajectory Transformer, 4 layers, 8 heads |
| conditioning | trajectory progress (72.2 s / 1083 steps) and cyclic process phase (`policy`, `follow_waypoints`, `return_to_reset`) |
| vision | RGB point-cloud DP3 encoder, 320x180, 4096 points, same crop box as the other DP3 students |
| training | ForgeUltra `forgeUltra_force_obs` 0e575f7 (dirty, not pushed), depth augmentation `realistic`, no RGB augmentation |
| training data | 40 `sequential-threading-dual-supervision-20261027-173748-484021` episodes |
| Isaac closed loop | 3 episodes x 16 environments, all completed 10 cycles (480 cycles, 0 transition failures) |

The checkpoint records no proprio layout; the training side confirmed the one
above (student.py `build_native_osc_student_proprio`). The order matches the force
channels appended after the previous action, as in the teacher actor and the
v1 student. The first proprio layer confirms it: the previous action's column
fingerprint sits at columns 17-25, as in the 26-D DP3 student.

## `fr3_offline_flow_matching_force_obs_25102026/nn/checkpoint.pt`

Distilled from the force-observation teacher. The force
and torque signals are teacher inputs only; the student does not observe them.

| field | value |
|---|---|
| sha256 | `6f68b30b20c897208c4b6368163c819638ee63b40bc43a2d239499101c9b1967` |
| size | 159 MB |
| epoch | 100 (EMA decay 0.999; `ema_model` is loaded by default) |
| control | 9-D OSC, `unified` action representation, native scale `[3,4,16,3,3,3,2,2,2]` |
| proprio | 29-D (`q10 + dq10 + previous filtered native action`), hand velocities included |
| action horizon | 8 |
| backbone | trajectory Transformer, 4 layers, 8 heads |
| conditioning | trajectory progress (81.2 s / 1218 steps) and cyclic process phase (`policy`, `follow_waypoints`, `return_to_reset`) |
| vision | two-channel RGB-D encoder (ResNet-18 stage-3 stems, 2 bidirectional fusion layers, 8x14 tokens), 320x180, 64-D output |
| training data | 20 `sequential-threading-dual-supervision-20261025-100347-893694` episodes, 24573 samples |
| Isaac closed loop | 15/16 environments completed 10 cycles (159 cycles, 1 transition failure) |

Its camera matrix is the same 320x180 policy view as the DP3 students'.

## `sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt`

An earlier ForgeUltra offline-flow student for the ten-cycle sequential
threading task, kept because the diagnostics that replay
`reference_episode/` default to it.

| field | value |
|---|---|
| sha256 | `8d9abc1101d06d8ff5d03b0c87e769429ee2ddc0b1998bc159c47c63a2a9313f` |
| size | 106 MB |
| epoch | 300 (EMA decay 0.999; `ema_model` is loaded by default) |
| control | 9-D OSC, `unified` action representation, native scale `[3,4,16,3,3,3,2,2,2]` |
| proprio | 29-D (`q10 + dq10 + previous filtered native action`) |
| action horizon | 8 |
| backbone | trajectory Transformer, 4 layers, 8 heads |
| conditioning | trajectory progress (64.6 s / 969 steps) and cyclic process phase (`policy`, `follow_waypoints`, `return_to_reset`) |
| vision | RGB point-cloud DP3 encoder, 320x180 input, 4096 points |
| training data | 20 `sequential-threading-dual-supervision-20260910-181107-771583` episodes |

Its DP3 camera contract (intrinsics, crop box, and point-cloud normalisation)
is what `../utils/camera_calibration/fr3_realsense_dp3.yaml` encodes. The
intrinsics come from this workcell's own D415 calibration run
`logs/20260910T152223_150165Z`: the 640x480 colour calibration, cropped to the
central 640x360 rows and halved to 320x180.

## `fr3/vision_head-.../nn/best.pt`

The DP3 encoder pretraining checkpoint ("vision head") that the student was
initialised from. It is kept for provenance, not loaded at runtime:

- its sha256 `2d4075bc34453b6b12cb65127111c08deafb5f14424acf3699484679ea3e94b9`
  is exactly the `vision_initialization.sha256` recorded inside
  `checkpoint.pt`;
- the student's `fusion.visual_encoder.*` weights started from its `encoder`
  and were then fine-tuned end to end (`encoder_trainable: true`), so they no
  longer equal the pretraining weights;
- its pose decoder (`decoder.*`, position plus 6-D rotation head) is not part
  of the student. The student carries its own 3-D `nut_position_head`.

Pretraining diagnostics stored in the file: mean nut position error 5.9 mm,
mean rotation error 46 deg after 1000 steps on 16000 frames.

## Loading contract

The app accepts a ForgeUltra checkpoint dictionary containing `config`,
`model`, and optionally `ema_model`; EMA is selected by default. It rejects
anything except the 9-D OSC contract with 29-D (or, without hand velocities,
26-D) proprio, plus 9 when it observes forces, and either the RGB point-cloud DP3 or the two-channel RGB-D
vision head. It refuses a checkpoint whose camera matrix, or, for DP3, whose
crop box, does not match the camera profile YAML.
