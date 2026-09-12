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

Start the workcell D415 separately with the checkpoint's calibrated 640x480
mode, synchronized colour/depth, and depth registered into the colour grid:

```bash
ros2 launch realsense2_camera rs_launch.py \
  device_type:=d415 \
  camera_namespace:=camera camera_name:=camera \
  enable_color:=true enable_depth:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  enable_sync:=true align_depth.enable:=true enable_rgbd:=true
```

In another sourced shell, run:

```bash
python3 apps/policy_rollout/run_policy_rollout.py hardware --device cpu --viewer
```

`--viewer` opens a side-by-side window using the exact RGB and aligned-depth
topics from the camera profile. It attaches to the existing D415, never starts
a second camera driver, and closes automatically when rollout exits. Its
hardware default is 5 Hz so rendering does not compete heavily with policy
inference. Use `--viewer-depth-max METRES` or `--viewer-hz HZ` to change the
display.

The CPU hardware backend uses two flow integration steps per action. On this
workcell that benchmarks at about 34 Hz before ROS/control load; an offline
teacher-forced check gives 0.0060 unified-action MAE versus 0.0043 with four
steps. Before enabling motion, the runner measures four steady real-frame
passes and requires the slowest to fit within 90% of the requested policy
period. Use `--integration-steps N` to override it; a setting that cannot meet
the requested rate is rejected before physical execution.

The camera path can be checked against real frames and the student checkpoint
without bringing up or commanding the arm or hand:

```bash
python3 apps/policy_rollout/tests/smoke_test_camera.py --device cpu
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
The Cartesian impedance itself acts at a fixed flange-relative point. At each
policy step the runtime measures the live transform from the fingertip midpoint
to that fixed point and converts the desired grasp pose into an equivalent
controller pose. This prevents hand-posture changes and the hardware thumb-yaw
overlay from turning a fingertip target into a centimetre-scale TCP offset.
The policy consumes the RealSense driver's atomic `rgbd` message, so RGB and
the manufacturer-aligned depth come from one synchronized frameset. RGB-D,
arm state, hand state, and controller state are checked for availability and
freshness before and during motion. A recording is always
written under `logs/policy_rollout` (override with `--recording-root`).

The controller still computes impedance and its watchdog at 1 kHz. Its ROS
state snapshots are published at 50 Hz, more than three times the 15 Hz policy
rate, to avoid spending policy CPU time serializing an unnecessary 1 kHz state
feed.

Physical commands remain fail-closed unless the camera pose in
`utils/camera_calibration/fr3_realsense_dp3.yaml` has status `validated`. The
checked-in profile records the actual D415 serial, the 640x480 stream geometry,
and the operator-accepted
`assets/camera/calibration/20260910T152223_150165Z/calibrated_tf.yaml` artifact.
The launcher reports any future mismatch without sending a robot command.

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

DP3 consumes aligned colour/depth at 320x180. The profile crops the live
640x480 stream to the training view and checks the live `CameraInfo` against
the checkpoint intrinsics. Depth may be `16UC1`
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
