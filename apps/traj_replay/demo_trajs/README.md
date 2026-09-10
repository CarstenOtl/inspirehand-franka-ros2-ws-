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

## Current-orientation `traj_2` artifacts

- `traj_2`: six-cycle 15 Hz source capture; select one with `--cycle N` and use
  its `homing.yaml`.
- `traj_2_cycle3`: one continuous cycle-3 candidate plus a smooth return home.
- `traj_2_5x`: cycles 1–5 as one continuous candidate plus a smooth return home.

All three keep the recorded joint-7 values unchanged (`offset_rad: 0.0`). The
derived candidates pass the replay dry-run checks but remain distinct from a
physically validated baseline until a hardware run is confirmed.

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

Remove `--dry-run` only after the normal hardware checks and replay launch. Add
`--close-support-fingers` only when deliberately overriding the recorded pinky,
ring, and middle positions with their fully closed limits.

## Legacy `traj_1` hardware baselines

The de facto one-cycle baseline remains `threading_cycle1_flange180`. It was
confirmed on hardware with the historical joint-7 compensation and the
support-finger override:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers --dry-run
```

`threading_5x_flange180` is the validated five-cycle legacy run. Its complete
replay was confirmed on hardware at 5x slowdown with interactive pause:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_5x_flange180 \
  --home apps/traj_replay/demo_trajs/threading_5x_flange180/homing.yaml \
  --close-support-fingers \
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
