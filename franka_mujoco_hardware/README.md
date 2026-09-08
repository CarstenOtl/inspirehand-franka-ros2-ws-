# franka_mujoco_hardware

A ros2_control `SystemInterface` that stands in for `franka_hardware` with a MuJoCo model of
the FR3, so the same controllers, launch files and scripts run against a simulated arm.

What it reproduces of libfranka:

- the joint interfaces of the real driver: position / velocity / effort command, position /
  velocity / effort state, one command type claimed at a time (mode switch like
  `franka_hardware`), the first command initialised to the current state;
- **position commands** drive an emulation of the robot-internal joint impedance controller,
  `tau = K (q_d - q) + D (dq_d - dq) + bias(q, dq)`, with libfranka's default joint impedance
  stiffness (`3000 3000 3000 2500 2500 2000 2000`) and hand-picked damping. Parameters
  `stiffness`, `damping`;
- the **motion generator checks**: every position command is compared with the previous one
  against the FR3 position limits, the position-dependent velocity limits, 10 rad/s^2 and
  5000 rad/s^3. A violation is logged with the name of the `franka::Errors` flag the real
  robot raises and, with `reflex_on_violation` (default true), `write()` returns ERROR so the
  controller manager stops the controllers - the same outcome as a reflex on the arm;
- **effort commands** are applied on top of gravity compensation, as libfranka does.

Not reproduced: the `robot_state` / `robot_model` interfaces (so no
`franka_robot_state_broadcaster` and no `FrankaRobotModel` in controllers), Cartesian
interfaces, the gripper, external-force estimation, and any identified dynamics of the real
controller. This is a plausibility model.

The MJCF in `mujoco/` is the Menagerie FR3 (Apache-2.0) with torque actuators, see
`mujoco/README.md`. `urdf/fr3_mujoco.urdf.xacro` wraps `franka_description`'s FR3 with the
ros2_control block for this plugin.

## Building

MuJoCo is found either from the `mujoco` Python wheel (`pip install mujoco` - it ships the
headers and `libmujoco.so`) or from a release tarball via `-DMUJOCO_DIR=/path/to/mujoco-3.x`.

## Extra state interfaces

`<system name>/sim_time` and `<system name>/motion_generator_violations` (a running count).
