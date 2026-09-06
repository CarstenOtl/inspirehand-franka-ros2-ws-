# inspire_franka_trajectory_replay

Coordinated hardware replay for the FR3 and Inspire RH56. The arm uses the
guarded `franka_trajectory_replay/TrajectoryReplayController`; position commands
therefore run through libfranka's internal joint-impedance mode. The hand stays
on its independent 50 Hz RS485 driver and receives synchronized position
commands on `/inspire_hand/command`.

The runner performs this sequence:

1. load and validate the homing YAML and trajectory;
2. densify and check the arm stream against the FR3 position, velocity,
   acceleration, and jerk limits;
3. activate the replay controller and move both devices to the homing pose;
4. move the arm to the smooth stream's lead-in point;
5. start the guarded arm stream and mapped six-DOF hand stream together.

It accepts either a trajectory directory containing `metadata.json` and
`replay_data.npz`, or a coordinated NPZ containing `joint_pos_arm` and optional
`joint_pos_hand`. Raw Forge `(time, environment, 19 joints)` recordings are
mapped by joint name; passive hand joints are never commanded. A Forge file can
contain several independent rollouts. In that case `--cycle N` is mandatory so
a reset between rollouts can never be mistaken for arm motion.

Build and validate without motion:

```bash
cd ~/develop_ws
rg2
source install/setup.bash

ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_1 --cycle 1 --dry-run
```

Inspect the coordinated recording in MuJoCo with Chi's source-compatible replay
viewer. This is the only MuJoCo process needed for trajectory inspection; do
not start a ROS simulation launch alongside it:

```bash
python3 apps/traj_replay/tests/test_mujoco_traj_replay.py \
  --trajectory apps/traj_replay/demo_trajs/traj_1
```

Add `--headless` to validate every sample and print the TCP-fit report without
opening a window. The viewer deliberately uses the recording's original
12-DoF training-hand geometry. It does not publish ROS commands and cannot move
the physical robot.

Bring up the real replay stack (this replaces the ordinary
`inspire_franka_bringup` launch for a replay session):

```bash
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  robot_ip:=10.7.7.7 hand_port:=/dev/ttyUSB0
```

Then, in a second sourced shell:

```bash
ros2 run inspire_franka_trajectory_replay replay_trajectory \
  apps/traj_replay/demo_trajs/traj_1 \
  --cycle 1 \
  --home apps/traj_replay/demo_trajs/homing/threading.yaml
```

The runner prompts before homing and again before replay. `--yes` disables the
prompts, and `--no-hand` keeps the original arm-only behavior.

The checked-in recording contains cycles 1 through 22. Cycles 1–20 and 22 are
made FR3-safe by the normal automatic time scaling (roughly 1.6x on the current
file). Cycle 21 approaches the position-dependent velocity boundary at
`fr3_joint5` and is deliberately rejected; do not bypass that guard on real
hardware. `--max-prepared-duration` can raise the default 120 s ceiling for a
legitimately long recording, but it does not disable any FR3 limit check.

The hand mapping from the Forge model is:

| Forge joint | Inspire driven joint |
|---|---|
| `little_joint_0` | `pinky_proximal_joint` |
| `ring_joint_0` | `ring_proximal_joint` |
| `middle_joint_0` | `middle_proximal_joint` |
| `index_joint_0` | `index_proximal_joint` |
| `thumb_joint_1` | `thumb_proximal_pitch_joint` |
| `thumb_joint_0` | `thumb_proximal_yaw_joint` |
