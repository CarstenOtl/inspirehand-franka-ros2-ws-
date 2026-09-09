# inspire_franka_sim

MuJoCo simulation of the Franka FR3 and the Inspire RH56 hand, driven through
the same `ros2_control` interfaces the rest of the workspace uses.

This is the **only** place the two assets share a controller_manager. On real
hardware they are two independent stacks (see the repo README); here they are
one hardware component and one world, driven by separate controllers.

## Quick start

Inside the container, after `rg2`:

```bash
# Arm + bench hand, both under position control, headless
ros2 launch inspire_franka_sim sim.launch.py

ros2 launch inspire_franka_sim sim.launch.py headless:=false     # MuJoCo viewer
ros2 launch inspire_franka_sim sim.launch.py start_rviz:=true    # RViz
ros2 launch inspire_franka_sim sim.launch.py hand:=false         # bare arm
ros2 launch inspire_franka_sim sim.launch.py arm:=false          # hand only
ros2 launch inspire_franka_sim sim.launch.py hand_mount:=flange  # hand on the flange
ros2 launch inspire_franka_sim sim.launch.py arm_command_interface:=effort
```

Both GUIs need `xhost +local:root` on the host.

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

- **The flange mount follows the current physical installation.** The hand is
  clocked 90 degrees from the legacy forgeUltra/franka-chi convention. A black,
  10 mm-thick adapter flange offsets the palm along `fr3_link8`'s +z axis; the
  composed rotation is approximately
  `(w, x, y, z) = (0.7071, -0.7071, 0, 0)`. The equivalent
  transforms in the description Xacro and `mjcf/inspire_hand_on_flange.xml`
  must remain synchronized.
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
