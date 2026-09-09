# FR3 DP3 distilled-policy rollout

This app is the portable inference boundary for the outcome recorded on
`forgeUltra/franka-chi` at commit `3fa5a545`: an RGB point-cloud DP3 vision
head fused with 29-D native-OSC proprioception and a distilled conditional-flow
student. It does not import Isaac Lab.

The complete upstream Flow Matching 1.0.10 source tree is vendored at
`third_party/flow_matching`, including its CC BY-NC 4.0 license. The rollout
adapter verifies that it imported this copy and uses the same fixed-step Euler
ODE solver as ForgeUltra.

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

## Current evidence and deliberate placeholders

There is no checkpoint in this workspace that is ready for physical rollout.
The retained DP3 sequential student completed three cycles in one matched
simulation before losing grip on cycle four; it did not satisfy the six-cycle
requirement. The later FR3 RealSense D415 experiment pretrained the DP3 visual
head only.

For those reasons, `run` always fails closed. The following must be completed
before it may publish anything:

1. train and multi-seed replay-qualify a student with the current corrected
   FR3/Inspire mount;
2. record and verify the physical D415 serial number and RGB-D calibration;
3. record the physical `world -> camera_color_optical_frame` transform in the
   camera YAML and validate it against the training pose;
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
python3 apps/policy_rollout/run_policy_rollout.py inspect \
  /absolute/path/to/checkpoint.pt --device cuda
```

Run exactly one synthetic observation through DP3 and flow sampling:

```bash
python3 apps/policy_rollout/run_policy_rollout.py dry-run \
  /absolute/path/to/checkpoint.pt --device cpu
```

Show every outstanding physical-integration blocker:

```bash
python3 apps/policy_rollout/run_policy_rollout.py camera-check
python3 apps/policy_rollout/run_policy_rollout.py run \
  /absolute/path/to/checkpoint.pt --device cuda
```

The last command exits nonzero and sends no robot command by design.

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
python3 apps/policy_rollout/run_policy_rollout.py dry-run \
  /absolute/path/to/checkpoint.pt --device cpu \
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

DP3 consumes aligned color/depth at 320x180. The checked-in training intrinsics
are exactly the 1280x720 color calibration divided by four. A future live
adapter must therefore request synchronized 1280x720 color plus aligned depth,
or produce a geometrically equivalent rectified 16:9 stream whose `CameraInfo`
matches the checkpoint. The existing 640x480 workspace camera command is not
compatible: cropping it to 16:9 changes the principal point and cannot be
treated as a simple resize.

Depth may enter `prepare_rgbd` as `16UC1` millimetres or `32FC1` metres. Invalid
and non-positive values become zero and are excluded by the DP3 valid mask.
