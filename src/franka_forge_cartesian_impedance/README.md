# franka_forge_cartesian_impedance

A `ros2_control` Cartesian impedance controller for the FR3, split out of
`franka_trajectory_replay` so that the controller used for policy rollouts is its own package.

Plugin: `franka_forge_cartesian_impedance/CartesianImpedanceController`.

## The two laws

Both live in
[cartesian_impedance.hpp](include/franka_forge_cartesian_impedance/cartesian_impedance.hpp) and
the controller's parameters pick between them:

- **Franka's example law**, lifted from `CartesianImpedanceExampleController`. The default, and
  the one used for replaying recorded Cartesian waypoints. Designed in
  [cartesian_replay_blueprint.md](../../docs/cartesian_replay_blueprint.md).
- **The ForgeUltra operational-space law** the Isaac Lab policies are trained against: stiffness
  and damping on the pose error of the controlled frame, a mass-weighted dynamically consistent
  nullspace term, an axis-angle rotation error, no Coriolis term and a torque clamp. Why the
  example law is the wrong one for a distilled policy is argued in
  [cartesian_impedance_nullspace_design.md](../../docs/cartesian_impedance_nullspace_design.md).

## Command interfaces

| topic | type | what it is for |
|---|---|---|
| `~/goto` | `CartesianGoto` | quintic ramp to a pose and nullspace configuration |
| `~/trajectory` | `CartesianTrajectory` | replay a recorded Cartesian trajectory |
| `~/policy_command` | `CartesianGoto` | bounded live targets from a non-realtime policy |
| `~/policy_goal` | `PolicyGoal` | a policy's preclipped goal plus the live controlled frame; the controller runs ForgeUltra's per-cycle clip and the law at that frame |
| `~/pause`, `~/resume`, `~/abort` | `std_msgs/Empty` | clock-rate ramps |

Outputs are `~/status` (`DiagnosticArray`), `~/controller_state` and `~/cartesian_state`. The
messages live in `franka_forge_cartesian_impedance_msgs`. The full parameter list and the phase
machine are documented in the class comment in
[cartesian_impedance_controller.hpp](include/franka_forge_cartesian_impedance/cartesian_impedance_controller.hpp).

## Where it is configured

This package ships the controller only. The parameter profiles that load it live with the robot
bringup, in `inspire_franka_trajectory_replay/config/`:
`controllers_cartesian_impedance.yaml`, `controllers_sim_impedance.yaml`,
`controllers_policy.yaml` and `controllers_sim_policy.yaml`. The Python client that drives it is
`franka_trajectory_replay.cartesian_replay_client`.

The controller instance is named `cartesian_trajectory_replay_controller` in those profiles, so
its topics and parameters keep that prefix.

## Build and test

```bash
colcon build --packages-up-to franka_forge_cartesian_impedance
colcon test --packages-select franka_forge_cartesian_impedance
colcon test-result --verbose
```
