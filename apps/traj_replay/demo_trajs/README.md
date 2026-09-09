# Trajectory replay artifacts

## De facto threading hardware baseline

Use `threading_cycle1_flange180` for coordinated or arm-only threading replay
on the current physical FR3 + Inspire RH56 setup. It is the configuration that
was confirmed on hardware with the correct tool orientation and a good replay:

- physical 180-degree flange mount;
- `+90 deg` (`+pi/2 rad`) applied to every `fr3_joint7` waypoint;
- the matching retargeted `homing.yaml` colocated with the trajectory;
- `--close-support-fingers` at replay time.

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
  --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
  --close-support-fingers --dry-run
```

Remove `--dry-run` only after the normal hardware checks and replay launch.
Never raise `--max-home-delta` to bridge a 1.571 rad joint-7 mismatch: that is
the known signature of mixing the legacy and current mount conventions.

## Outdated threading arm configurations

The following are retained for provenance, visualization, hand-only testing,
or deriving new candidates. They are not current hardware-arm baselines:

- `traj_1`: original Forge source using the legacy mount convention;
- `homing/threading.yaml`: matching legacy-mount source home;
- `threading_5x`: unretargeted five-cycle intermediate.

`threading_5x_flange180` is not outdated: it is the validated extended run for
the same mount convention. Its complete five-cycle replay was confirmed on
hardware at 5x slowdown with interactive pause. The smaller
`threading_cycle1_flange180` remains the canonical baseline.

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/threading_5x_flange180 \
  --home apps/traj_replay/demo_trajs/threading_5x_flange180/homing.yaml \
  --close-support-fingers \
  --time-scale 5 \
  --interactive-pause \
  --max-prepared-duration 300
```

Raw `traj_1` is still valid with `--no-arm`, because hand-only replay never
commands `fr3_joint7`. The pickup artifacts belong to a separate task and are
not classified by this threading decision.

When deriving another threading cycle or multi-cycle candidate, first use
`make_cycle_trajectory`, then apply `retarget_flange_mount.py` with
`--joint7-offset-deg 90`, and keep the generated trajectory paired with its
generated homing YAML. A derived candidate does not replace or extend the
baseline until it is independently hardware-validated.
