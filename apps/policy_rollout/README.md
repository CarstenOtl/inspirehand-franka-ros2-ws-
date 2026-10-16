# FR3 DP3 policy rollout

This app runs the distilled threading policy through one launcher and three
backends:

- `hardware`: the physical FR3, Inspire RH56, and RealSense D415;
- `ros-sim`: the hardware backend against `sim_policy.launch.py` (MuJoCo behind ROS);
- `mujoco`: the closed-loop MuJoCo threading scene.

The policy runtime is independent of Isaac Lab. It implements the checkpoint's
RGB point-cloud DP3 encoder, native-OSC proprioception, conditional-flow
action sampling, action filtering, and cyclic process-phase conditioning.
Proprioception is `q10 + dq + the previous action`, so it is 29 values wide for
a checkpoint shown the three hand velocities and 26 for one trained without
them; the checkpoint's own `hand_velocity_observation` picks the layout and the
runner refuses a checkpoint whose width disagrees with it. The
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
python3 apps/policy_rollout/run_policy_rollout.py hardware --viewer
```

`--viewer` opens a side-by-side window using the exact RGB and aligned-depth
topics from the camera profile. It attaches to the existing D415, never starts
a second camera driver, and closes automatically when rollout exits. Its
hardware default is 5 Hz so rendering does not compete heavily with policy
inference. Use `--viewer-depth-max METRES` or `--viewer-hz HZ` to change the
display.

The runner picks its device on its own: CUDA when the container's torch build
sees a GPU, otherwise CPU, and `--device` overrides that. The whole pass moves
with it, point-cloud construction included. The flow-step default follows the
device. On CUDA the policy runs the 16 Euler steps it was distilled with (see
"GPU inference" below for the measured cost). On CPU it runs two, which
benchmarks at about 34 Hz on the workcell before ROS/control load; an offline
teacher-forced check gives 0.0060 unified-action MAE versus 0.0043 with four
steps. Before enabling motion, the runner measures four steady real-frame
passes and requires the slowest to fit within 90% of the requested policy
period. Use `--integration-steps N` to override either default; a setting
that cannot meet the requested rate is rejected before physical execution.
Each recording stores `policy_device` and `flow_integration_steps` in its
metadata. A seeded run on CUDA does not reproduce a seeded CPU run: the flow
noise comes from a device-bound generator. Point sampling is identical on
both, since evaluation uses a fixed priority buffer.

### GPU inference

A host with an NVIDIA GPU runs the GPU variant of the image; the top-level
README ("CPU or GPU") covers the choice and the one-time host setup. With it,
`--device auto` resolves to CUDA and the default becomes 16 flow steps.
Nothing else in this app changes between the two variants. Outside Docker,
`requirements-cpu.txt` and `requirements-gpu.txt` install the matching torch.

Measured on the development laptop (i7-12700H, RTX 3060 Laptop 6 GB) with
`tests/benchmark_policy_inference.py`, on the deployed 26-D checkpoint with
320x180 frames, on 2026-10-15:

| flow steps | CUDA mean / max | CPU mean / max |
|---|---|---|
| 2 | 3.0 / 3.3 ms | 16 / 18 ms |
| 16 | 11.7 / 12.0 ms | 92 / 225 ms |
| 32 | 21.9 / 22.3 ms | - |

Sixteen steps on the GPU use a fifth of the 60 ms preflight budget at 15 Hz;
on this CPU they fail it. The per-step cost is about 0.6 ms on CUDA against
3 ms on CPU, and the fixed vision cost drops from about 10 ms to 2 ms. Compute
is no longer what bounds `--rate`: the checkpoint was distilled at 15 Hz, so
its action horizon, temporal ensemble and progress clock assume that period,
and the D415 profile delivers 30 frames per second. Raising the rate is a
training-side change, not a deployment flag.

Through the regular compose setup, the same benchmark gives 13.5 ms for 16
steps. A 1500-step `ros-sim` rollout on 2026-10-16 missed no 15 Hz deadline
on CUDA with 16 steps. On the same laptop's CPU with 2 steps, it missed 27.

The camera path can be checked against real frames and the student checkpoint
without bringing up or commanding the arm or hand:

```bash
python3 apps/policy_rollout/tests/smoke_test_camera.py
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
written under `logs/policy_rollout` (override with `--recording-root`), with a
`report.json` holding the final status, including the error that stopped the
run.

Homing uses `config/policy_home.yaml`, which holds the M24 reset joints and the
training grasp posture. The grasp frame's approach axis is captured once after
homing, as a vector in the flange frame, and then rotates with the flange,
exactly as in training. Before 2026-09-14 the runner used the flange's -Z axis
instead. That axis is about 107 degrees from the training axis on this hand
mount, so every orientation target was decoded in the wrong frame.

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

## Rehearse the hardware stack in MuJoCo

`ros-sim` runs the hardware backend unchanged against a MuJoCo plant behind
the same ROS interfaces. That means the same homing, controller swap, Cartesian
policy commands, mock hand driver and `/camera/camera/rgbd`.

```bash
ros2 launch inspire_franka_trajectory_replay sim_policy.launch.py
python3 apps/policy_rollout/run_policy_rollout.py ros-sim --yes
```

The launch opens two windows. One is the MuJoCo viewer, which starts on a free
camera; press `]` in it to switch to the calibrated `policy_d415` view. The
other is `rqt_image_view`, showing the relayed colour stream the policy
consumes. Use `headless:=true` and `camera_view:=false` to skip them.

The launch loads `inspire_franka_policy_scene.xml`. That scene is the flange
torque scene placed in the training environment:

- bench top at `fr3_link0` z = 0;
- the M24 bolt and nut;
- the calibrated D415 with its exact pinhole matrix, generated by
  `utils/make_ros_sim_scene.py`.

`policy_camera_relay` republishes the rendered camera with the profile's
matrix and frame. The controller runs `controllers_sim_policy.yaml`, which is
the policy profile on the DH model. Recordings go to
`logs/policy_rollout/ros_sim-*`.

The simulator shares the policy's CPU, a 4-core i7-8550U. So the launch runs
MuJoCo at half real time (`sim_speed:=0.5`), and `ros-sim` paces its default
15 Hz loop on `/clock`. The policy period, input freshness, trajectory progress
and recorded timestamps are all in simulated time. With both windows open a
policy pass costs about 100 ms of wall time, which is 50 ms of simulated time.
`sim_speed:=1.0` gives real time; that fits 15 Hz only headless, at a pass of
about 45 ms. The mock hand driver still slews in wall time.

The simulation does not model:

- contact: the hand has no collision geometry and the nut is fixed;
- FR3 dynamics;
- RealSense noise;
- RS485 timing.

It does model frames, units, rates and the camera contract.

`utils/analyze_rollout_motion.py RUN_DIR [--plot out.png]` rebuilds the grasp
frame of any hardware or ros-sim recording. It compares that frame with the
goal the policy was steering to. In a healthy stack the distance falls to
about 1 cm.

## The grasp frame

The policy's actions are decoded relative to, and executed at, the thumb/index
fingertip midpoint. That frame is taken from **ForgeUltra's training hand
geometry**, not from this workspace's URDF, because the two are independent
derivations of the same Inspire RH56 and they disagree:

| at the training grasp posture | training | workspace URDF |
|---|---|---|
| tip separation | 57.8 mm | 113.6 mm |
| midpoint in the flange frame | (-0.0591, -0.0288, 0.1733) | (-0.0638, -0.0422, 0.1773) |
| grasp-frame orientation | reference | 14.7 deg away |

The tips hang off parent links that themselves differ by 26 mm / 16 deg
(thumb) and 12 mm / 179 deg (index), posed by different mimic ratios, so no
fixed tip offset reconciles them at every posture. `TrainingHandKinematics` in
`policy_rollout/hardware.py` therefore evaluates the training asset's own
kinematics on the measured joints; TF still supplies the flange, where the two
descriptions agree. The hand is rigid on the flange, so this depends only on
the three policy hand joints.

This puts the policy's observation and command frame in its training
distribution. Which of the two models describes the *physical* hand is still
an open bench measurement: put the hand at the training grasp posture and read
the real thumb-to-index distance, then make the losing model follow the
winner.

### The controlled point

The Cartesian controller shifts its measured pose *and its Jacobian* onto
`tool_offset_xyz`, so that offset is the compliance centre: where the spring
and damper are anchored. Training anchored its operational-space wrench at the
live fingertip midpoint, which travels about 20 mm over a threading cycle. A
fixed offset cannot follow that, so the policy profiles use the midpoint frozen
at the threading grip:

    tool_offset_xyz: [-0.059067, -0.028773, 0.173311]

That is `POLICY_TOOL_OFFSET_XYZ` in `policy_rollout/hardware.py`, mirrored in
`controllers_policy.yaml` and `controllers_sim_policy.yaml` and checked at
startup. It deliberately differs from `replay.yaml`'s `tcp.offset_xyz`
(-0.0874, -0.0327, 0.1453), which is a *closed* replay pinch 40 mm away;
trajectory replay keeps that one, because its recorded Cartesian poses are
expressed at it. Both are marked in `inspire_hand_on_flange.xml` as the
`grasp_tcp` and `policy_grasp_tcp` sites (group 4, hidden by default).

Distance from the live grasp midpoint over the reference episode's 328
policy rows:

| tool offset | mean | min | max |
|---|---|---|---|
| old (replay pinch) | 37.8 mm | 20.6 mm | 52.8 mm |
| new (threading grip) | 11.5 mm | 3.2 mm | 23.0 mm |

The remaining 3 to 23 mm is the live midpoint's own travel, which a fixed
offset cannot express. `retarget_grasp_pose_to_controlled_pose` still
compensates the commanded pose either way; what this changes is only where the
impedance acts.

## Run in MuJoCo

```bash
MUJOCO_GL=glfw python3 apps/policy_rollout/run_policy_rollout.py mujoco
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

No depth post-processing is enabled: the driver's spatial, temporal,
hole-filling and decimation filters are all off by default and none of the
launch commands turn them on. To see what they would change, attach a live
comparison to the running camera (it never opens the device):

```bash
python3 apps/policy_rollout/utils/compare_depth_filters.py
```

It shows the raw sensor depth, the current policy input, and both again after
librealsense's own filters, which it runs on the driver's raw frames in the
driver's order. Each panel shows fill rate, temporal noise, plane-fit RMS in
a dragged rectangle and the DP3 in-box pixel count; the status bar shows the
per-frame filter cost and the driver launch arguments for the current toggles.
Keys: `s`/`t`/`h`/`d` toggle spatial,
temporal, hole filling and disparity domain; `x` shows filtered minus current;
`p` saves arrays and a screenshot under `artifacts/depth_filter_comparison/`.
The policy panels are exact only with the camera in the 640x480 mode above.

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
