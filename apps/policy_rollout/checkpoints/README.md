# Checkpoints

Everything in this directory except this file and `.gitignore` is ignored by
git (see `.gitignore` here and the `*.pt` rule in the repository root). The
files are large binaries; copy them in by hand from wherever the training run
exported them.

## Expected layout

```
checkpoints/
  sequential_threading_cycle10_hybrid_teacher_d415_20ep/
    checkpoint.pt                      the student policy that the rollout app loads
  fr3/
    vision_head-fr3_d415_nut_sweep_20260910T152223_calibration/
      nn/best.pt                       DP3 vision-head pretraining checkpoint (provenance only)
```

## `sequential_threading_cycle10_hybrid_teacher_d415_20ep/checkpoint.pt`

The ForgeUltra offline-flow student for the ten-cycle sequential threading
task. This is the only file the rollout app reads.

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
anything except the 9-D OSC, 29-D proprio, RGB point-cloud DP3 contract, and
it refuses a checkpoint whose camera matrix or DP3 crop does not match the
camera profile YAML.
