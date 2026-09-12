# FR3 DP3 policy rollout

This app runs the distilled threading policy through one launcher and two
backends:

- `hardware`: the physical FR3, Inspire RH56, and RealSense D415;
- `mujoco`: the closed-loop MuJoCo threading scene.

The policy runtime is independent of Isaac Lab. It implements the checkpoint's
RGB point-cloud DP3 encoder, 29-D native-OSC proprioception, conditional-flow
action sampling, action filtering, and cyclic process-phase conditioning. The
Flow Matching runtime is vendored under `third_party/flow_matching`.

## Run on hardware

Build and source the ROS workspace, then bring up the interfaces used by
`traj_replay`, selecting the policy controller profile:

```bash
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  arm_controller:=policy robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0
```

Start the D415 separately with aligned depth enabled. Its topics must match the
camera calibration profile:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

In another sourced shell, run:

```bash
python3 apps/policy_rollout/run_policy_rollout.py hardware --device cpu
```

The hardware backend deliberately has two operator gates unless `--yes` is
given: one before homing and one before policy commands. It first homes through
the hardware-proven `trajectory_replay_controller`, then switches the FR3 to
the Cartesian impedance controller. That controller owns the seven effort
interfaces, reads the real Franka model/Jacobian, accepts bounded live
setpoints, faults on excessive tracking error, and holds when its command
watchdog expires. The hand uses the same `/inspire_hand/command` radians-to-open
ratio conversion and `/inspire_hand/joint_states` feedback as `traj_replay`.

The launch also attaches the hand's root frame to `fr3_link8`, allowing the
runtime to derive the policy grasp frame from the measured thumb/index tip TFs.
RGB, aligned depth, arm state, hand state, and controller state are checked for
availability and freshness before and during motion. A recording is always
written under `logs/policy_rollout` (override with `--recording-root`).

Physical commands remain fail-closed until
`utils/camera_calibration/fr3_realsense_dp3.yaml` contains the actual D415
serial and a camera pose whose status is `validated`. The checked-in profile
still says `PLACEHOLDER` / `measured`, because its calibration residuals exceed
the stated physical-rollout limits. Update those fields only after a qualifying
calibration; the launcher reports the exact blockers without sending a robot
command.

Use `python3 apps/policy_rollout/run_policy_rollout.py hardware --help` for the
hardware safety and topic options.

## Run in MuJoCo

```bash
MUJOCO_GL=glfw python3 apps/policy_rollout/run_policy_rollout.py mujoco \
  --device cpu
```

The simulator runs the same student policy at 15 Hz and executes native OSC
actions through the ported Forge control path at 120 Hz. It models the cyclic
release/return coordinator and records both policy data and simulation state.
Use `python3 apps/policy_rollout/run_policy_rollout.py mujoco --help` for output,
preview, video, cycle, and step-limit options.

## Camera contract

DP3 consumes aligned colour/depth at 320x180. The profile converts either the
training 640x480 stream or the physical 1280x720 stream to that view and checks
the live `CameraInfo` against the checkpoint intrinsics. Depth may be `16UC1`
millimetres or `32FC1` metres. Non-positive and invalid values are excluded by
the DP3 valid mask.

The policy's three hand coordinates map to the physical hand as follows:

| Policy coordinate | RH56 joint |
|---|---|
| `thumb_joint_0` | `thumb_proximal_yaw_joint` |
| `thumb_joint_1` | `thumb_proximal_pitch_joint` |
| `index_joint_0` | `index_proximal_joint` |

The physical workcell has no measured nut-angle topic. Consequently, the
hardware cyclic coordinator uses measured grasp-frame yaw as its release
threshold proxy and records completed return cycles, but does not claim
threading success from that proxy alone.
