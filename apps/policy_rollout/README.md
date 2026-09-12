# FR3 DP3 distilled-policy rollout

This app is the portable inference boundary for the outcome recorded on
`forgeUltra/franka-chi` at commit `3fa5a545`: an RGB point-cloud DP3 vision
head fused with 29-D native-OSC proprioception and a distilled conditional-flow
student. It does not import Isaac Lab.

The upstream Flow Matching 1.0.10 runtime package is vendored at
`third_party/flow_matching`, together with its CC BY-NC 4.0 license and package
metadata. The rollout adapter verifies that it imported this copy and uses the
same fixed-step Euler ODE solver as ForgeUltra.

## What is implemented

- strict loading of ForgeUltra offline-flow checkpoints (`ema_model` by
  default, with strict state-dict matching);
- the checkpoint-compatible RGB+point-cloud DP3 encoder;
- DEXTRAH image-first late fusion, trajectory Transformer, conditional-flow
  action chunks, 16-step Euler sampling, and temporal chunk ensembling;
- exact `q10 + dq10 + previous_filtered_native_OSC_action` observations;
- checkpoint-schema global-progress and cyclic-process-phase conditioning;
- native-action EMA and unified-to-native OSC scaling;
- calibrated RGB-D validation, 1280x720 to 320x180 preparation, millimetre to
  metre depth conversion, and DP3 camera/crop contract checks;
- inspection and a synthetic one-step dry run that never imports ROS or sends
  commands.
- reusable policy-rate data collection, headless plotting, rollout health
  metrics, and time-aligned comparison against a reference recording.

All camera topics, intrinsics, training pose, DP3 crop, and physical
calibration placeholders live in
`utils/camera_calibration/fr3_realsense_dp3.yaml`. Do not duplicate those
values in a runtime adapter.

## Environment

The runtime needs CPU PyTorch and `torchdiffeq`; both are installed by the
workspace Docker image (see `docker/Dockerfile`). This workcell has no NVIDIA
GPU, so use `--device cpu`. One policy step (1280x720 frame preparation plus a
16-step flow sample) takes about 0.13 s on the container's CPU, comfortably
inside the 15 Hz policy period.

`rg2test` inside the container runs this app's tests together with the rest
of the workspace's hardware-free tests.

## Current evidence and deliberate placeholders

The student checkpoint for the ten-cycle sequential threading task lives at
`checkpoints/sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt`
(ignored by git; see `checkpoints/README.md` for its provenance and the
vision-head pretraining file that sits next to it). `inspect` and `dry-run`
load it strictly and pass, and the camera profile YAML is verified against its
DP3 contract by the test suite.

The camera profile now encodes this workcell's own D415 calibration
(`logs/20260910T152223_150165Z`), which is the calibration the training scene
was built from. Its measured world-to-camera pose is recorded with status
`measured`, not `validated`, because that run's residuals (27 mm, 3.5 deg RMSE)
exceed the profile's 10 mm / 2 deg tolerance for a physical rollout.

`run` still fails closed. The following must be completed before it may
publish anything:

1. multi-seed replay-qualify the student on a simulation that matches the
   current FR3/Inspire mount;
2. record the physical D415 serial number and re-calibrate to within the
   profile's tolerances, then set the measured pose status to `validated`;
3. run the RealSense node with aligned depth enabled (today it publishes
   1280x720 colour and depth without `align_depth.enable`);
4. implement live FR3 FK/Jacobian and the 1 kHz OSC controller/command bridge;
5. implement the student-owned release/return phase coordinator;
6. validate the official Forge hand coordinates against this workspace's RH56
   driver, then add limits, stale-data watchdogs, operator gates, and stop paths.

The three Forge hand coordinates map semantically onto this driver as follows,
but that mapping alone is not hardware qualification:

| Forge coordinate | Workspace RH56 joint |
|---|---|
| `thumb_joint_0` | `thumb_proximal_yaw_joint` |
| `thumb_joint_1` | `thumb_proximal_pitch_joint` |
| `index_joint_0` | `index_proximal_joint` |

## Commands

Install the vendored solver's runtime dependency (`torchdiffeq`) in the rollout
environment, then inspect a checkpoint:

```bash
CK=apps/policy_rollout/checkpoints/sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt
python3 apps/policy_rollout/run_policy_rollout.py inspect $CK --device cpu
```

Run exactly one synthetic observation through DP3 and flow sampling:

```bash
python3 apps/policy_rollout/run_policy_rollout.py dry-run $CK --device cpu
```

Show every outstanding physical-integration blocker:

```bash
python3 apps/policy_rollout/run_policy_rollout.py camera-check
python3 apps/policy_rollout/run_policy_rollout.py run $CK --device cpu
```

The last command exits nonzero and sends no robot command by design.

## MuJoCo smoke test against a training episode

`smoke_test_mujoco.py` poses the replay MJCF from frames of a recorded
training episode (`checkpoints/reference_episode/`, ignored by git), renders
aligned RGB-D from the calibrated camera, and runs one policy step per frame
on that render and on the episode's own recorded image:

```bash
MUJOCO_GL=glfw python3 apps/policy_rollout/smoke_test_mujoco.py
```

It writes `report.json` and a `side_by_side.png` next to the episode. With
the ten-cycle student the actions from MuJoCo renders match the episode
labels to about 0.005 mean absolute error on the [-1, 1] action scale on
every frame after the first, and differ from the recorded-image actions by
about the same amount. The camera is attached at the training scene's robot
root (table height, joint 1), not at the MJCF base plate; the script's
docstring explains the evidence.

This is an open-loop check of the perception and inference chain only. A
closed-loop MuJoCo rollout additionally needs ForgeUltra's OSC controller
(native action to joint command), which is not in this workspace.

## Data collection, plotting, and evaluation

`PolicyRolloutSession` accepts an optional `RolloutDataCollector`. Every policy
step then records aligned 10-D joint position and velocity, 29-D proprioception,
9-D policy and filtered native OSC actions, clipping, progress, and process
phase. Prepared DP3 RGB-D input is opt-in because it makes recordings much
larger. A future ROS adapter should also pass its monotonic sample timestamp and
the supported task signals (`pickup_success`, `threading_entered`,
`completed_cycles`, `threading_turn_progress_rad`, termination, truncation, and
watchdog state).

The dry run can exercise the complete recording path without robot access:

```bash
python3 apps/policy_rollout/run_policy_rollout.py dry-run $CK --device cpu \
  --recording-dir /tmp/fr3-dp3-dry-run --record-rgbd
```

This writes `data/rollout_data.npz` and `data/metadata.json`. NPZ loading always
uses `allow_pickle=False`. Generate plots or an evaluation report with:

```bash
python3 apps/policy_rollout/run_policy_rollout.py plot \
  /path/to/rollout

python3 apps/policy_rollout/run_policy_rollout.py evaluate \
  /path/to/rollout --reference /path/to/nominal-rollout
```

Plots and `evaluation.json` are written under the rollout's `analysis/`
directory unless `--output-dir` is supplied. Reference state/action errors are
linearly aligned on overlapping elapsed time and stored in
`reference_error_trace.npz`. The default full-task threshold is six completed
threading cycles. If the runtime adapter does not supply the task signals, the
report says `not_evaluable`; it never converts smooth motion or low tracking
error into a task-success claim. Physical rollout remains disabled regardless
of an evaluation report.

## Camera input contract

DP3 consumes aligned colour/depth at 320x180. The checkpoint's intrinsics are
the workcell's calibrated 640x480 D415 colour stream with the central 640x360
rows kept and then halved (`policy_input.source_crop_px` in the YAML). The
loader recomputes the policy matrix from the 640x480 calibration through that
crop and refuses to start if it does not equal the checkpoint's matrix.

`prepare_rgbd` accepts three frame shapes and converts the first two to the
policy view by cropping with the profile's rectangle and resizing:

| frame | crop | result |
|---|---|---|
| 640x480 (training source) | rows 60..420 | halve to 320x180 |
| 1280x720 (`physical_camera.color_stream`) | x 160..1120, y 90..630 | third to 320x180 |
| 320x180 | none | passthrough |

The 1280x720 mapping holds because the D415's 640x480 mode is the central
960x720 of the 1280x720 sensor stream scaled by two thirds; the live
`CameraInfo` observed on this workcell maps onto the checkpoint matrix through
that crop to better than 0.001 px. A live adapter must call
`CameraCalibrationProfile.assert_live_camera_info` with the stream's
`CameraInfo` before its first policy step, and it must subscribe to depth that
is aligned to colour (`align_depth.enable:=true`).

Depth may enter `prepare_rgbd` as `16UC1` millimetres or `32FC1` metres. Invalid
and non-positive values become zero and are excluded by the DP3 valid mask.
