# inspire_franka_sim

MuJoCo simulation of the Franka FR3 and the Inspire RH56 hand, driven through
the same `ros2_control` interfaces the rest of the workspace uses.

The package also contains a separate **passive viewer**. It mirrors one ROS
joint-state topic or planned trajectories into `mjData.qpos`, but has no command
publisher, controller manager, or hardware interface. Use it to inspect a
recording, preview a planned trajectory, or watch one live state stream in the
MJCF scene without starting a simulated plant. It is a one-way display, not a
robot-control GUI.

This is the **only** place the two assets share a controller_manager. On real
hardware they are two independent stacks (see the repo README); here they are
one hardware component and one world, driven by separate controllers.

## Quick start

Inside the container, after `rg2`:

```bash
# Arm + flange-mounted hand, both under position control, headless
ros2 launch inspire_franka_sim sim.launch.py

ros2 launch inspire_franka_sim sim.launch.py headless:=false     # MuJoCo viewer
ros2 launch inspire_franka_sim sim.launch.py start_rviz:=true    # RViz
ros2 launch inspire_franka_sim sim.launch.py hand:=false         # bare arm
ros2 launch inspire_franka_sim sim.launch.py arm:=false          # hand only
ros2 launch inspire_franka_sim sim.launch.py arm_command_interface:=effort
```

Both GUIs need `xhost +local:root` on the host.

## Passive visualization and replay

### What it does—and what it does not do

Start the subscriber-only viewer:

```bash
ros2 launch inspire_franka_sim passive_viewer.launch.py
```

It subscribes to `/joint_states` and `/mujoco_sim/joint_trajectory`.  Exact
joint names map automatically.  `JointState.position` and `.velocity` are
written using each joint's `jnt_qposadr` and `jnt_dofadr`, so MuJoCo's internal
kinematic order is never assumed.  Partial messages are supported; unknown
joints are reported once and ignored.  Hand follower joints are derived from
the MJCF equality polynomials when a message only contains the six driven hand
joints.

The viewer does not know whether a `JointState` came from simulation, a rosbag,
mock hardware, or a physical robot. Consequently, it *can* show live physical
measurements when its selected topic is published by a hardware driver, but it
never opens or commands a hardware connection itself.

The combined real-hardware bringup publishes two independent streams:

| topic | source |
|---|---|
| `/joint_states` | physical FR3 (when the arm namespace is empty, as it is by default) |
| `/inspire_hand/joint_states` | physical Inspire hand |

The default passive-viewer invocation therefore follows the real arm. To follow
the real hand instead:

```bash
ros2 launch inspire_franka_sim passive_viewer.launch.py \
  joint_state_topic:=/inspire_hand/joint_states
```

That leaves the arm in the combined MJCF at its initial pose. A viewer instance
currently accepts only one `joint_state_topic`; it does not combine the arm and
hand topics from real-hardware bringup. A namespaced arm similarly requires its
namespaced topic, for example `joint_state_topic:=/cell_1/joint_states`.

The sliders shown in the passive viewer's side panel are MuJoCo actuator-control
sliders, not joint-position or ROS controls. They are intentionally inert in
this node: it disables MuJoCo actuation, does not step physics by default, and
clears `mjData.ctrl` and applied forces before every optional physics step.
Setting `subscribe_joint_states:=false` only stops state updates; it does not
enable the sliders. This node publishes no command topic, so the panel cannot
configure or move the physical robot or a separately running `ros2_control`
simulation.

Both panels are hidden by default so the passive display does not present inert
controls. Press `Tab` to toggle the left panel and `Shift+Tab` to toggle the
right, or select the initial layout explicitly:

```bash
ros2 launch inspire_franka_sim passive_viewer.launch.py \
  show_left_ui:=true show_right_ui:=true
```

Mouse camera navigation continues to work with the panels hidden.

The supported way to pose the passive display is to publish a `JointState`, as
in the example below, or a `JointTrajectory`. The current
`mujoco_ros2_control` window opened by `sim.launch.py` does not provide a slider
panel; `arm_command_interface:=none` and `hand_command_interface:=none` only
leave joints unclaimed. Drive that full simulation through its ROS controllers.

To check it without any robot or controller, publish a state repeatedly:

```bash
ros2 topic pub --rate 20 /joint_states sensor_msgs/msg/JointState \
  "{name: [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6, fr3_joint7],
    position: [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.7],
    velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

Or replay a planned trajectory.  Disable joint states for this mode if a
controller is publishing them continuously; by default a measured state
preempts a plan because it is the more truthful visualization:

```bash
ros2 launch inspire_franka_sim passive_viewer.launch.py \
  subscribe_joint_states:=false

ros2 topic pub --once /mujoco_sim/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6, fr3_joint7],
    points: [
      {positions: [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.7], time_from_start: {sec: 1}},
      {positions: [0.4, -0.3, 0.2, -1.9, 0.1, 1.4, 0.9], time_from_start: {sec: 4}}
    ]}"
```

Trajectory timing starts when the message arrives and uses a monotonic local
clock, so replay remains usable without `/clock` or a network.  Segments are
linear when velocities are omitted and cubic Hermite when both endpoint
velocities are supplied.  Rendering and ROS callbacks run on separate threads.

For ROS/MJCF name mismatches, pass a YAML alias table.  See
`config/joint_map.example.yaml`:

```bash
ros2 launch inspire_franka_sim passive_viewer.launch.py \
  joint_map_file:=/absolute/path/to/joint_map.yaml
```

An offline `ros2_control` pipeline can feed the same viewer.  Run the existing
MuJoCo or mock launch headless in one terminal, then the passive viewer in
another; `/joint_states` is enough:

```bash
ros2 launch inspire_franka_sim sim.launch.py headless:=true
ros2 launch inspire_franka_sim passive_viewer.launch.py \
  subscribe_trajectory:=false
```

Do not launch real-hardware bringup merely to use this offline workflow. The
viewer itself cannot command hardware. Its default is stricter still:
`step_physics:=false` only writes kinematic state and calls `mj_forward`.
`step_physics:=true` is an explicit opt-in for passive, unactuated stepping of
unreported degrees of freedom.

Curl the fingers:

```bash
ros2 action send_goal /hand_joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names:
  [index_proximal_joint, middle_proximal_joint, ring_proximal_joint,
  pinky_proximal_joint], points: [{positions: [1.2, 1.2, 1.2, 1.2],
  time_from_start: {sec: 2}}]}}"
```

Move the arm:

```bash
ros2 action send_goal /fr3_joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names:
  [fr3_joint1, fr3_joint2, fr3_joint3, fr3_joint4, fr3_joint5, fr3_joint6,
  fr3_joint7], points: [{positions: [0.5,-0.4,0.3,-2.0,0.2,1.4,0.9],
  time_from_start: {sec: 3}}]}}"
```

Both at once is just both commands — the controllers claim disjoint joints and
never interact.

## How it fits together

Two descriptions are maintained and both must agree:

| | file | authoritative for |
|---|---|---|
| URDF/xacro | `inspire_franka_description/urdf/inspire_franka.urdf.xacro` | TF, RViz, `<ros2_control>` interface declarations |
| MJCF | `mjcf/*_scene.xml` | physics, actuators, contacts |

`mujoco_ros2_control` reads the `<ros2_control>` tags from the URDF and
instantiates `MujocoSystemInterface`, which owns the MuJoCo Simulate app. The
`mujoco_model` hardware parameter points it at the MJCF. **Joint names are the
only join between the two**, so `test/test_mjcf.py` checks everything else that
has to line up. See `mjcf/MJCF_PROVENANCE.md`.

One set of launch arguments picks both the URDF variant and the MJCF scene, so
the two cannot drift apart by accident.

### One hardware component, two controllers

`mujoco_ros2_control` builds **one MuJoCo simulation per hardware component**, so
a second `<ros2_control>` block would give the hand its own separate world where
it could never touch anything the arm is holding. Arm and hand therefore share a
single component, and are driven by separate controllers over disjoint joints.
That is what makes "together" and "individually" both work.

### Which hand joints are controllable

| joints | | interfaces |
|---|---|---|
| 6 | `{index,middle,ring,pinky}_proximal`, `thumb_proximal_{pitch,yaw}` | position + effort command, position/velocity/effort state |
| 6 | `*_intermediate`, `thumb_distal` | **state only** — followers, realised as MuJoCo equality constraints |

This mirrors the real driver exactly: six commanded, twelve published. The
followers are declared even though nothing drives them, because otherwise
nothing publishes their positions and `robot_state_publisher` drops every
fingertip frame from TF.

## Measured behaviour

Everything below is from this model at its 2 ms timestep, with the shipped
gains and the actuators' ±1 Nm clamp.

| | |
|---|---|
| Contacts at the rest pose (all scenes) | 0 |
| Finger step 0 → 1.0 rad | 0.21 s to 90 %, settles within **0.17 mrad** |
| Four-finger curl to 1.2 rad | worst joint error **0.18 mrad** |
| Coupled joint tracking its driver | **0.03 mrad** free, **1.4 mrad** in contact |
| Return to the open pose | reaches the limit, `max|qvel|` → 0 |

Driving all six DOF to their limits at once leaves the index finger ~0.69 rad
short — the thumb swings across the palm and they genuinely collide. That is
physics, not a tuning failure.

## Known rough edges

- **The flange mount has 180-degree clocking.** `hand_mount` is attached to
  `fr3_link8` with zero translation and 180 degrees of yaw. The equivalent transforms
  in the description Xacro and `mjcf/inspire_hand_on_flange.xml` must remain
  synchronized.
- **The shipped scenes are right-handed.** `mjcf/inspire_hand_left.xml` is
  generated but no scene binds it; `sim.launch.py` refuses `hand_side:=left`
  under MuJoCo rather than silently simulating the wrong hand.
- **The arm's joint ranges are tighter in MJCF than in the URDF**, by
  Menagerie's safety margin. A goal accepted by a controller can be clamped by
  MuJoCo.
- **The hand's joint dynamics are not identified.** Damping, armature and
  friction are stabilisers chosen to make the model behave, not measurements.
- **Soft joint limits.** Driving a joint hard into its limit overshoots it by
  around 10 mrad, which is MuJoCo's default limit softness rather than a bug.

## Regenerating the assets

```bash
python3 scripts/make_hand_mjcf.py --side right
python3 scripts/make_hand_mjcf.py --side left
```

Then re-run `colcon test --packages-select inspire_franka_sim`.
