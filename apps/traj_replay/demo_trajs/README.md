# Trajectory replay artifacts

## `fr3_joint7` orientation contract

`traj_2` establishes the convention for all new captures: policies are developed
with the current physical FR3 + Inspire hand orientation, so `fr3_joint7` is
replayed exactly as recorded. The matching home must use the same recorded
value. Do **not** apply `+90 deg`, `+pi/2`, `retarget_flange_mount.py`, or a
`*_flange180` conversion to `traj_2` or any later trajectory.

The rotation remains relevant only to legacy `traj_1` material. `traj_1` was
recorded under the old convention, and its retained hardware derivatives
`threading_cycle1_flange180` and `threading_5x_flange180` contain the historical
`+90 deg` compensation. Those artifacts remain valid as already generated and
validated; do not add or remove the offset from them.

Always keep a trajectory with its matching `homing.yaml`. A difference of about
`1.571 rad` on joint 7 means conventions were mixed; never hide that with
`--max-home-delta`.

## Current-orientation policy artifacts

- `traj_2`: six-cycle 15 Hz source capture; select one with `--cycle N` and use
  its `homing.yaml`.
- `traj_2_cycle3`: one continuous cycle-3 candidate plus a smooth return home.
- `traj_2_5x`: cycles 1–5 as one continuous candidate plus a smooth return home.
- `traj_3`: one 449-sample continuous rollout from the V2 policy. The final
  two-sample reset fragment is ignored automatically. Its matching home is
  recorded, but raw arm replay is rejected because `fr3_joint5` reaches the
  position-dependent velocity boundary; do not bypass the safety guard.
- `traj_3_joint5_cap_2p8`: hardware candidate derived from `traj_3`. Only
  saturated `fr3_joint5` waypoints are capped at `2.800 rad`; the maximum
  change is `0.0065 rad` and all other waypoints and timing are unchanged.
- `traj_3_multi`: raw ten-cycle V2 capture. Cycles 1, 6, 7, and 8 touch the
  same joint-5 braking boundary, so keep it as source material rather than
  replaying the complete raw capture on hardware.
- `traj_3_multi_joint5_cap_2p8`: all ten recorded cycles as one continuous
  hardware candidate, plus a smooth return home. It caps only saturated
  `fr3_joint5` waypoints at `2.800 rad` (171 samples across cycles 1, 6, 7,
  and 8; maximum change `0.006507 rad`) and passes replay preparation at 5x
  slowdown.

These current-orientation artifacts keep the recorded joint-7 values unchanged
(`offset_rad: 0.0`). The derived candidates pass the replay dry-run checks but
remain distinct from a physically validated baseline until a hardware run is
confirmed.

Replay the prepared ten-cycle candidate directly; it needs no `--cycle` or
`--segment` selection. At 5x slowdown it prepares to about 372 seconds, so the
duration guard must be raised:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8 \
  --home apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8/homing.yaml \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 400 \
  --dry-run
```

Remove `--dry-run` only after the normal hardware checks and replay launch.

Validate the raw `traj_3` waypoints with the default joint-impedance controller:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_3 \
  --home apps/traj_replay/demo_trajs/traj_3/homing.yaml \
  --dry-run
```

This currently reports the joint-5 safety violation. Do not remove `--dry-run`
or raise the duration guard for raw arm replay. A hardware-safe policy capture
or an explicitly retargeted derivative is required first.

Five continuous cycles at 5x slowdown prepare to 159 seconds, so the duration
guard must be raised explicitly:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_2_5x \
  --home apps/traj_replay/demo_trajs/traj_2_5x/homing.yaml \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 300 \
  --dry-run
```

Remove `--dry-run` only after the normal hardware checks and replay launch. The
runner overrides the recorded pinky, ring, and middle positions with their fully
closed limits by default; pass `--no-close-support-fingers` to replay the
recorded ones.

## Sequential multi-nut threading: `traj_4_m30` and `traj_4_m36`

Ten-cycle sequential threading captures from the
`Isaac-Forge-Franka-Threading-V2-Multi-v0` policy, one per nut. Each cycle is
the policy turn (`policy`), the scripted release and retreat
(`follow_waypoints`), and the return to the start pose (`return_to_reset`),
recorded back to back at 15 Hz. `hardware_trajectory_focused/` holds the same
motion as CSV with its manifest.

Each `replay_data.npz` is a batched capture of three simulated nuts, and the
focused nut is not environment 0:

| Folder | Nut | `--env` | Cycles | Source | 5x replay |
| --- | --- | --- | --- | --- | --- |
| `traj_4_m30` | M30, 3.5 mm pitch | `1` | 10 | 1286 samples, 85.7 s | 430 s |
| `traj_4_m36` | M36, 4.0 mm pitch | `2` | 10 | 1139 samples, 75.9 s | 381 s |

Always pass `--env`: the default environment 0 is the M24 nut, whose joint 7
comes within 0.04 rad of its limit. Neither folder carries a `homing.yaml`;
both start exactly at the `traj_3` home (arm delta 0.000 rad), so use
`traj_3/homing.yaml`. The recorded joint-7 values use the current orientation
(no offset). Ignore `trajectory_name: "traj_1"` and the humanoid
`impedance_joint_pd_command_names` in `metadata.json`; they are exporter
leftovers that the replay does not read.

Replay a complete recording as it is with `--all-cycles`. The phases stay
embedded: `--intervene` reads each cycle's release point from the capture's
own `cycle` and `replay_phase` fields. The run ends at the last recorded
sample, about 0.01 rad short of home.

```bash
D=apps/traj_replay/demo_trajs

# M30, joint impedance (default controller)
ros2 run inspire_franka_trajectory_replay replay_trajectory $D/traj_4_m30 \
  --env 1 --all-cycles \
  --home $D/traj_3/homing.yaml \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 500 \
  --dry-run

# M36, joint impedance (default controller)
ros2 run inspire_franka_trajectory_replay replay_trajectory $D/traj_4_m36 \
  --env 2 --all-cycles \
  --home $D/traj_3/homing.yaml \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 450 \
  --dry-run
```

For Cartesian impedance, launch the bringup with the Cartesian controller
loaded, then add `--arm-controller cartesian-impedance` to either command. The
arm homes with the joint controller and switches for the trajectory; the runner
refuses before homing if the launch did not load it. `--stiffness-scale` scales
the Cartesian stiffness. `--intervene` is joint-impedance only.

```bash
ros2 launch inspire_franka_trajectory_replay replay.launch.py arm_controller:=cartesian-impedance

ros2 run inspire_franka_trajectory_replay replay_trajectory $D/traj_4_m30 \
  --env 1 --all-cycles \
  --home $D/traj_3/homing.yaml \
  --arm-controller cartesian-impedance \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 500 \
  --dry-run
```

To validate or run one cycle, replace `--all-cycles` with `--cycle N` (1–10)
and drop `--time-scale` and `--max-prepared-duration`; the runner picks a
2.4–3.7x slowdown per cycle. For a hand-only check, replace
`--time-scale 5 --interactive-pause --max-prepared-duration ...` with
`--no-arm --hand-time-scale 1`.

All ten cycles of both nuts, and both complete runs in either controller, pass
replay preparation. What to expect and watch:

- Joint 5 reaches 2.81 rad, as in `traj_3`; this is within the corrected FR3
  limits.
- The release waypoints move further than `traj_3`: joint 7 to −2.81 rad (M30)
  and joint 6 to 2.54 rad (M36, 73 % of the house speed limit at 5x).
- The hand motion is small: pinky, ring, and middle stay closed; the index
  bends to 0.62 rad and the thumb yaw swings between 1.09 and 0.74 rad.
- In simulation the proper grip holds for under half of each policy phase, and
  M36 cycle 10 never registers a grasp. Stop at the pause before it if needed.

Remove `--dry-run` only after the normal hardware checks and replay launch.

## Legacy `traj_1` hardware baselines

The de facto one-cycle baseline remains `threading_cycle1_flange180`. It was
confirmed on hardware with the historical joint-7 compensation and the
support-finger override:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --dry-run
```

`threading_5x_flange180` is the validated five-cycle legacy run. Its complete
replay was confirmed on hardware at 5x slowdown with interactive pause:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_5x_flange180 \
  --home apps/traj_replay/demo_trajs/threading_5x_flange180/homing.yaml \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 300
```

Raw `traj_1`, `homing/threading.yaml`, and `threading_5x` remain legacy source
material rather than direct hardware-arm artifacts. Raw `traj_1` is still valid
with `--no-arm`, because hand-only replay never commands `fr3_joint7`.

For future multi-cycle captures, use `make_cycle_trajectory` with the capture's
matching zero-offset home and replay the generated result directly. The
retargeting script is retained only for deriving artifacts from legacy
`traj_1`-convention sources.
