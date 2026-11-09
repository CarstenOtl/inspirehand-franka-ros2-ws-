# RH56 hand map: one angle ↔ register function for sim and hardware

For the deployment side (inspirehand-franka-ros2-ws). Copy `rh56_hand_map.py` and `rh56_hand_map.json`
verbatim. They need only numpy and json.

## Why

The driver currently converts policy angles linearly:

    open_ratio = clamp(1 - (rad - lower) / (upper - lower), 0, 1)

The real hand is not linear in that angle. One register channel drives a four-bar (fingers) or six-bar
(thumb bend) linkage, so the joint the policy commands sits elsewhere than the linear map assumes.
On the index the error is about 0.13 rad mid-stroke: a reading of 0.59 rad is really 0.72 rad.

The sim's RH56 hand (task `Isaac-Forge-Franka-Threading-V2-M24-FR3v2-RH56-v0`) drives every register
through the measured linkage at nominal geometry. This package is that same function. With it on the
robot, a commanded or reported angle means the same register in sim and on hardware.

## Use

```python
from rh56_hand_map import RH56HandMap

hand_map = RH56HandMap()  # loads rh56_hand_map.json next to it

# command: policy leader angle -> register (0..1000) or open_ratio (0..1)
register = hand_map.register_from_rad("index_joint_0", rad)
open_ratio = hand_map.open_ratio_from_rad("index_joint_0", rad)

# reading: register the hand reports -> leader angle for the policy's hand-q observation
rad = hand_map.rad_from_register("index_joint_0", register_actual)
```

- Keys are a channel (`little`, `ring`, `middle`, `index`, `thumb_bend`, `thumb_rotation`) or its leader
  joint (`little_joint_0` … `index_joint_0`, `thumb_joint_1`, `thumb_joint_0`).
- Register 1000 is the open stop, 0 the closed stop. Angles are the sim's leader joint angles, with 0
  meaning open. Values outside the stroke clamp to the stops.
- Use it for the commands and for the hand-q observation. Both directions must go through the map.

## Which checkpoints

| Checkpoint trained on | Map |
|---|---|
| RH56 hand (`hand_asset_profile: rh56` in the run's `params/env.yaml`) | this map |
| mimic hand (`official` or `calibrated`, e.g. the run09 FR3 v2 student) | the existing linear map |

The joint ranges differ too. On RH56 the policy's pinch ranges are the full linkage strokes:
- thumb_joint_0: 0–1.308;
- thumb_joint_1: 0–0.716;
- index_joint_0: 0–1.470.

The URDF limits do not apply to RH56 policies.

## Not covered

- The force stop (FORCE_SET) is untouched. The sim stops on rod force too, with the threshold
  randomized.
- Distal joint angles are not reported by the hand. In the sim they follow the linkage.

## Joint table: every hand joint per register

`rh56_joint_table.json` (numpy + json, same nominal geometry and grid as the map) gives, per channel,
the angle of EVERY joint of that finger at each register, distal followers included. Use it where a
simulator (e.g. the MuJoCo rollout) drives all hand joints and should follow the linkage instead of the
linear URDF mimic ratios:

```python
import json, numpy as np
table = json.load(open("rh56_joint_table.json"))["channels"]

def joint_targets(channel, register):
    """{joint: rad} for one channel at register 0..1000 (1000 = open)."""
    ch = table[channel]
    reg = np.asarray(ch["register"])[::-1]               # arrays run 1000 -> 0; interp needs ascending
    return {j: float(np.interp(register, reg, np.asarray(q)[::-1])) for j, q in ch["joint_rad"].items()}

joint_targets("thumb_bend", 400.0)  # {'thumb_joint_1': ..., 'thumb_joint_2': ..., 'thumb_joint_3': ...}
```

- Joint names are the forgeUltra names; `official_joint_name` maps them to the official Tiangong2Pro
  URDF names (`index_joint_0` -> `right_index_1_joint`, ...). Angles: rad, 0 = open, + = flexion.
- Ranges are the full linkage strokes and exceed the URDF limits (e.g. thumb_joint_1 0.716 vs 0.48,
  thumb_joint_2 0.787 vs 0.358); widen the joint ranges in the MJCF accordingly.
- Register is linear in rod length: a rod speed v [mm/s] is `1000 * v / stroke_mm` register/s
  (training: 20 mm/s nominal, 16-24 randomized).
- Nominal geometry only. The training sim also randomizes the linkage over 32 variants
  (`tables/rh56_tables.npz`); export those if the rollout should randomize too.
