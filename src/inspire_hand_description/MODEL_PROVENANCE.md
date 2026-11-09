Model provenance

`urdf/inspire_hand_{right,left}.macro.xacro` and everything under `meshes/` were
vendored from the training URDF. The thumb and index limits and couplings have
since been fitted to the real hand; see the hardware fit below.

| | |
|---|---|
| Model | Inspire Robotics RH56 dexterous hand, as the right hand of the Tiangong 2.0 Pro humanoid |
| Source | forgeUltra's `assets/fr3_inspirehand/urdf/inspirehand_right.urdf` (forgeUltra commit `58046e0`, "update hand asset to be official supported") |
| Upstream of that | Open-X-Humanoid's official `TienKung_URDF`, `tiangong2pro_urdf`, commit `5c221783fb92fcc4af891ef1dc0502963caf2266`; see forgeUltra's `assets/fr3_inspirehand/OFFICIAL_HAND_SOURCE.md` |
| Licence | No licence or notice file is present in the upstream package |
| Vendored on | 2026-10-16 |

## Hardware fit (2026-11-09)

The thumb and index limits and the follower couplings in the macros are not the
training URDF's. They were fitted to the real hand by a 1080p camera sweep of
the registers, so a joint angle means the real joint angle, and the driver's
`kinematics.py` is the single table they come from:

| Follower | Leader | Multiplier | Offset |
|---|---|---|---|
| `index_intermediate_joint` | `index_proximal_joint` | 0.8867 | 0.0275 |
| `thumb_intermediate_joint` | `thumb_proximal_pitch_joint` | 1.2092 | 0.0252 |
| `thumb_distal_joint` | `thumb_intermediate_joint` | 0.5592 | 0.0062 |

The middle, ring and pinky fingers were not swept and keep 1.1169. Geometry
and masses are the training URDF's. Thumb yaw commands are scaled onto registers
250 to 1000, the range the fitted yaw limits describe.

## Why this model

It is the hand the student policy was trained on. forgeUltra extracted the
`R_base_link` subtree of the official Tiangong URDF, renamed its links and
joints, and added five tip frames that the task code builds its grasp frame
from. Isaac Lab trained against exactly that file, so the fingertips this
description puts in TF are the policy's own, to within 1e-7 m
(`apps/policy_rollout/tests/test_hardware.py`).

Until 2026-10-16 this package carried dex-urdf's independent derivation of the
same hand, from Inspire's STEP files. It disagreed with the training hand in
three ways that mattered:

- **Fingertips.** At the same joint command the pads were 114 mm apart instead
  of 58 mm, the midpoint 15 mm away and the grasp frame 21 degrees off.
- **Mass.** 0.19 kg distal to the flange instead of 0.54 kg. Retailers list the
  RH56DFX at 540 g.
- **Joint ranges.** The driver maps the hand's 0-1 open ratio linearly onto
  each driven joint's range, so the range decides where a radian command puts
  the physical finger. dex-urdf's were 1.47 (fingers), 0.6 (thumb pitch) and
  1.308 (thumb yaw); training's are 1.333, 0.48 and 1.246.

Which of the two describes the physical hand is still a bench measurement
(D1 in `docs/policy_rollout_training_gap_todo.md`). Adopting the training model
makes the whole stack consistent with the policy; it does not by itself prove
the model right.

## What the generator changes

1. **Names.** Links and joints take the names the workspace already used
   (`hand_base_link`, `index_proximal_joint`, `thumb_proximal_yaw_joint`, ...),
   so the driver, controllers, `config/pids.yaml` and the sim bridge address the
   same joints as before. The map is `LINKS`/`JOINTS` in the generator.
2. **Mount.** A massless `hand_mount` link sits above the palm with an identity
   joint. forgeUltra mounts the palm 10 mm out from `fr3_link8`, yawed by pi;
   the arm description puts `hand_mount` exactly there.
3. **Effort and velocity limits** are the workspace's placeholders (1 N m,
   0.5 rad/s), not training's PhysX drive values (50 N m, 1 rad/s). Nothing on
   the real hand reads them.
4. **Meshes.** The 13 official STL files are copied byte-for-byte, renamed after
   their link, under `meshes/right/`. Each is both the visual and the collision
   geometry, as in the source.
5. **Left hand.** There is no official left hand. The left macro is the exact
   mirror of the right across the `hand_mount` XZ plane: frames conjugated by the
   reflection so they stay right-handed, joint axes negated so a positive angle
   still closes the finger, inertias reflected, mesh vertices reflected with
   their winding flipped. Ranges and couplings are the right hand's.

## Twelve joints, six actuators

The hand has six actuators. Each finger's `*_intermediate` joint follows its
`*_proximal` joint through a linkage, and the thumb has two followers in a
chain. The source records this as `<mimic>`:

| follower | follows | multiplier | offset |
|---|---|---|---|
| `*_intermediate_joint` (four fingers) | `*_proximal_joint` | 1.1169 | 0 |
| `thumb_intermediate_joint` | `thumb_proximal_pitch_joint` | 1.1425 | 0 |
| `thumb_distal_joint` | `thumb_intermediate_joint` | 0.7508 | 0 |

A `<mimic>` is a statement, not a mechanism: nothing in ROS propagates it at
runtime. So the same numbers appear in `inspire_hand_driver.kinematics`, which
computes the follower angles the driver publishes, and in the MuJoCo equality
constraints that `inspire_franka_sim/scripts/make_hand_mjcf.py` emits.
`test/test_mimic_matches_driver.py` fails if the copies, or the driver's
ranges and the URDF limits, drift apart.

Two details worth knowing before trusting the numbers:

- **The finger followers stop short.** At the closed pose (1.333 rad) the
  follower reaches 1.489 rad, about 2.2 degrees inside its own 1.527 rad limit.
- **The thumb intermediate clamps.** 1.1425 x 0.48 = 0.548 would overshoot its
  0.3578 rad limit, so it stops at about two thirds of the pitch's travel
  (0.313 rad) and the distal stops with it, at 0.7508 x 0.3578 = 0.269 rad.
  The driver clamps to the limits, so TF and MuJoCo agree about where the thumb
  is. The policy's own mimic expansion (`forge_osc.expand_hand_mimic`) does not
  clamp, so above 0.313 rad of pitch its thumb tip and this description's part
  ways; the threading grasp holds the pitch at 0.05 rad.

## Known gaps

- **No joint dynamics.** The description declares no damping or friction. The
  simulation supplies its own; see `inspire_franka_sim`.
- **Collision meshes are not convex.** MuJoCo collides each one as its convex
  hull, which for the palm fills the hollow the thumb's proximal segment sits
  in; `make_hand_mjcf.py` excludes that pair. A planner using the raw meshes
  sees the true shells.
