# FR3 student policy rollout

This app runs the distilled threading policy through one launcher and three
backends:

- `hardware`: the physical FR3, Inspire RH56, and RealSense D415;
- `ros-sim`: the hardware backend against `sim_policy.launch.py` (MuJoCo behind ROS);
- `mujoco`: the closed-loop MuJoCo threading scene.

The policy runtime is independent of Isaac Lab. It implements the checkpoint's
vision head (the RGB point-cloud DP3 encoder, or the two-channel RGB-D
encoder), native-OSC proprioception, conditional-flow action sampling, action
filtering, and cyclic process-phase conditioning. The checkpoint's own
`vision_encoder_config` picks the head. Proprioception is
`q10 + dq + the previous action`, so it is 29 values wide for a checkpoint
shown the three hand velocities and 26 for one trained without them; the
checkpoint's own `hand_velocity_observation` picks the layout and the runner
refuses a checkpoint whose width disagrees with it. The Flow Matching runtime
is vendored under `third_party/flow_matching`.

Every backend and diagnostic tool defaults to `policy_rollout.DEFAULT_CHECKPOINT`;
pass `--checkpoint` to run another.

## The deployed checkpoint

`checkpoints/m24-dp3-run09fr3v2-nominal-cycles-noclock-force-nohv-da-s0-20261102-173148-415770/nn/checkpoint.pt`
(2026-11-02, epoch 100, EMA weights; details in `checkpoints/README.md`),
trained on the FR3 v2 arm without the trajectory-progress clock (cyclic phase only),
trained with the hardware's M24 bolt holder in the scene and on images from
the 2026-10-29 camera calibration, which
`utils/camera_calibration/fr3_realsense_dp3.yaml` and both simulators now use. DP3
point-cloud vision, no hand velocities, and force inputs: proprioception is
35-D, `q10 + arm dq7 + previous action + [thumb, index] pad force / 20 N +
seven FR3 torques / [87, 87, 87, 87, 12, 12, 12] N m` clipped to 1. Its
training scene is threading_v2's table in front of the robot (forgeUltra
35373e0) with the bolt standing in the bolt holder and the top raised to the
holder's foot, which both simulators now reproduce.

Where the force inputs come from:

| backend | pad force | arm torque (default) |
|---|---|---|
| `mujoco` | contact forces on the pad links, projected like ForgeUltra | applied actuator torque |
| `ros-sim` | `/fingertip_force` (`FingertipForcePlugin`) | controller `tau_task + tau_nullspace` |
| `hardware` | RH56 `/inspire_hand/grip_force`, grams to N, zeroed with the hand open | `tau_ext_hat_filtered`, low-passed 0.02 s |

Training's torque is PhysX's projected joint-reaction torque with the robot's
gravity off. Which FR3 signal matches it best is still being measured, so the
runner records every candidate on every step, and `--arm-torque-source` picks
the one the policy sees:

| candidate | source | contains |
|---|---|---|
| `external` | `/franka_robot_state_broadcaster/external_joint_torques` | libfranka's estimate: `tau_J` minus the model's gravity and dynamics |
| `measured` | `~/cartesian_state` `tau_measured_filtered - tau_gravity` | `tau_J` minus the model's gravity, so it keeps the motion torques PhysX's transmitted torque also has |
| `desired` | `/franka_robot_state_broadcaster/desired_joint_states` | libfranka's `tau_J_d`, the command without gravity |
| `commanded` | `~/cartesian_state` `tau_task + tau_nullspace` | the law's own torque, without the coriolis feed-forward |

ros-sim has `measured` (MuJoCo's applied actuator torque, gravity off) and
`commanded` only. The recordings hold them as `arm_torque_<name>_nm`, plus
`arm_torque_gravity_nm`. The controller low-passes `tau_J` at 1 kHz
(`measured_torque_lowpass_s`, 0.02 s), and the runner low-passes the
broadcaster topics with the same time constant (`--arm-torque-lowpass-s`).
`measured` needs the rebuilt `franka_forge_cartesian_impedance(_msgs)`.

`external`, `measured` and `desired` depend on the end-effector mass configured
in Desk, so run hardware with `--expected-load-mass-kg`. With the hand open at
the start, the runner prints every candidate at rest, and records them as
`arm_torque_at_rest_nm` in `report.json`. Training reads zero there, so a large
value is a payload or model error. Compare runs, and optionally a training
episode with force columns:

```bash
python3 apps/policy_rollout/utils/compare_torque_candidates.py \
  logs/policy_rollout/hardware-<stamp> [...] [--reference EPISODE.npz]
```

To see what the DP3 encoder got in a recorded run (`--record-rgbd`), next to
where training's nut sits in the same camera view and, through the
calibration, in the base frame:

```bash
python3 apps/policy_rollout/utils/render_policy_view.py logs/policy_rollout/hardware-<stamp> --times 0,4,10
```

## The 2026-10-25 RGB-D student (history)

`checkpoints/fr3_offline_flow_matching_force_obs_25102026/nn/checkpoint.pt`
(2026-10-25, epoch 100, EMA weights). It was distilled from the force-observation
teacher on twenty 10-cycle sequential-threading episodes
(`sequential-threading-dual-supervision-20261025-100347-893694`). Isaac's
closed-loop check completed all 10 cycles in 15 of 16 environments.

- **No force or torque input.** The *teacher* observes the two fingertip pad
  forces and the seven FR3 joint torques. ForgeUltra has not yet added them to
  the student vectors (`docs/distillation_observation_action_reference.md`
  there), and the weights confirm it: proprioception is 29-D
  (`q10 + dq10 + previous action`) and the condition is 128 proprio + 64 vision
  features. The rollout therefore reads no force or torque signal.
- **Vision: the two-channel RGB-D encoder** (`policy_rollout/rgbd_encoder.py`),
  not DP3. ResNet-18 stems on RGB (ImageNet-normalised) and on log depth plus
  validity, RGB<->depth cross-attention, and a learned pooling query. It sees
  the whole 320x180 view, not a cropped point cloud, so the rendered or real
  background now reaches the policy. Same camera matrix and crop as before.
  The trunk is rebuilt without torchvision. On the checkpoint's own weights it
  matches ForgeUltra's implementation to 4e-7.
- **Hand velocities are inputs again** (29-D, `hand_velocity_observation:
  included`). The previous 26-D checkpoint left them out.
- **Release rule.** Each cycle aims at the next 60 deg step of the nut's total
  turn and releases 5 deg before it, which is 55 deg on the first cycle
  (`forge_osc.next_cycle_target_turn_rad`, from ForgeUltra 304d4d9). ros-sim
  measures the total turn; hardware's grasp-yaw proxy cannot, so hardware
  keeps 55 deg for every cycle.
- **Cost.** On the laptop CPU with 2 flow steps, almost all of it the encoder:
  47 ms per action in `tests/benchmark_policy_inference.py`, and 85-120 ms of
  wall time in the live preflight with the simulator sharing the CPU. That
  does not fit the 60 ms preflight at 15 Hz in real time, so on CPU run ros-sim
  at `sim_speed:=0.4`. GPU cost is not measured yet (the host's NVIDIA driver
  was down on 2026-10-25).

First ros-sim runs (2026-10-25, CPU, `sim_speed:=0.4`, seed 0): the student
closes the index on the nut, turns it 56-60 deg and releases. Earlier students
never turned the nut in ros-sim. It then returns to a pose 7.5-8 mm / 7-8 deg
from the trained reset grasp, outside the 5 mm / 5 deg gate, and the run ends
`return_failed` after the 16 s timeout. The arm tracks its goal to 1-3 mm; the
goal itself is off. The MuJoCo-only backend shows the same: 11.3 mm / 6.5 deg.

2026-10-26 follow-up. The thumb was not the problem. Isaac's cyclic teacher
returns to the pinch it captured live at threading entry, not to the closed
grasp posture. Its own return command (unified pinch -0.091 / -0.083 / 0 in
the training episode) puts that entry pinch at about
`forge_osc.ISAAC_ENTRY_PINCH` = [1.072, 0.025, 0.215], and the student returns
the hand there. The return gate and the homing now use that pinch. The
student's position and rotation command is still off, by the same amount in
every run: 9.3-9.5 mm (-6 mm x, -7 mm z in the base frame) from the teacher's
return goal. The controller tracks it to 1-3 mm / 1-2 deg. Our bolt frame
and decoder reproduce the teacher's entry pose to 0.4 mm / 0.8 deg, so the
offset is in the student's raw output. Offline input swaps leave it unchanged
for arm joints, joint velocities, hand joints, progress and an earlier camera
frame; only the previous-action input moves it, closing about half the gap.

That input also started wrong. Isaac's reset seeds `prev_actions` from the
live pose: the grasp relative to the bolt tip over the 5 cm bounds (native z
about 3.3 at the entry grasp), the wrist angles, and a pinch of 0. The dataset
builder takes each sample's previous action from the filtered stream that
starts there, so every training episode opens with that seed. The rollout
started cycle 1 from zeros, and seeded the pinch from the live hand at cycle
boundaries. Both now follow Isaac (`seed_action_history`);
`--prev-action-seed zeros` restores the old cycle-1 start for comparison.

The camera, measured on a training episode
(`checkpoints/reference_episode/`, copied from the training machine). Isaac's
return is a 15-tick ramp and an 8-tick hold at the entry command. On Isaac's
own hold inputs the student holds the teacher's command (x -0.176 vs -0.177);
on ros-sim's it holds -0.144. Swapping one input at a time puts about 60 % of
that on the camera frame and 40 % on the previous action, which feeds the
student's own output back in; joints, velocities and progress move it by
nothing. The ros-sim scene did not look like the training scene:
forgeUltra's cyclic-threading environment (threading_v2) slides the bench
under the FR3 base, so the bolt stands 6 cm inside its far edge with floor
beyond, and renders it walnut on light metal legs; the hand is the training
URDF's white; and the camera clips depth at (0.01, 5.0) m, writing zero
beyond. `make_ros_sim_scene.py`, `make_hand_mjcf.py --isaac` and
`prepare_rgbd` now do the same. Rendered offline at the failed return, those
changes move the held command from -0.145 to -0.156. The rest of the image
effect is training's visual randomization (HDRI lighting, random table, floor
and nut/bolt textures, image backgrounds beyond 1.5 m), which a fixed scene
cannot match. The depth clip applies on hardware too: training never saw a
valid depth beyond 5 m.

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
`tests/benchmark_policy_inference.py`, on the then-deployed 26-D DP3 checkpoint
with 320x180 frames, on 2026-10-15 (the RGB-D student's encoder is much
heavier; see "The deployed checkpoint"):

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
and the
`assets/camera/calibration/20261029T173220_626741Z/calibrated_tf.yaml` artifact
the deployed student was trained on.
The launcher reports any future mismatch without sending a robot command.

The workcell has no nut-angle sensor, so on hardware the scripted release
fires on `--release-signal finger-sweep` (the default there): a loaded pinch
with the fingertips within 10 mm, then the index closing to 0.82 rad in the same
grasp. The deployed student turns the nut with its fingers, not the wrist, and
ends each turn with the fingertips ~16 mm apart, where the older grasp-yaw
proxy (`--release-signal grasp-yaw`) is vetoed as a collapsed pinch. In ros-sim
the default gates on the simulated nut's twist; pass `--release-signal
finger-sweep` there to rehearse the hardware signal against it. It fired at
39-70 deg of twist on the nominal arm and earlier, at 22-65 deg, with
`arm_friction_scale:=2`, where the nut lags the fingers.

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

Without a GPU the RGB-D student needs a slower simulator so that its CPU pass
(85-120 ms of wall time next to the simulator) fits the 15 Hz period in
simulated time. Headless:

```bash
ros2 launch inspire_franka_trajectory_replay sim_policy.launch.py \
  headless:=true camera_view:=false sim_speed:=0.4
python3 apps/policy_rollout/run_policy_rollout.py ros-sim --yes --record-rgbd
```

The launch opens two windows. One is the MuJoCo viewer, which starts on a free
camera; press `]` in it to switch to the calibrated `policy_d415` view. The
other is `rqt_image_view`, showing the relayed colour stream the policy
consumes. Use `headless:=true` and `camera_view:=false` to skip them.

The launch loads `inspire_franka_policy_scene.xml`. That scene is the flange
torque scene placed in the training environment:

- the M24 bolt and nut, the bolt standing in the hardware's printed bolt
  holder (`Bolt-Holder-M24-v2lprt.STEP`, visual only);
- bench top at `fr3_link0` z = 0.0065, the holder's foot;
- the calibrated D415 with its exact pinhole matrix, generated by
  `utils/make_ros_sim_scene.py`.

The hand is the training hand as Isaac Lab simulates it. Geometry, masses and
limits come from forgeUltra's official URDF, and the drives, armature and
effort limits from its `robot.py` and threading env (pinch 30 / 3 / 20 N m,
inactive fingers 4.76 / 0.21 / 1.864, followers 0.9 or 4.76). The
`IsaacHandPlugin` in `inspire_franka_sim` adds three things: Isaac's
target-level mimic (followers are free joints driven to multiplier times their
leader's target), the 1 rad/s joint velocity limit, and PhysX's load-dependent
joint friction. With the default `hand_chain:=isaac`, the policy's pinch
targets go straight to the joint drives, and the runner observes the simulated
joints, so a finger that stops on the nut reads as stopped. `hand_chain:=driver`
puts the mock RS485 driver in between instead. It slews in wall time and
reports its own state, not the simulated fingers'. Not reproduced: PhysX's
convex decomposition of the finger meshes (MuJoCo collides one hull per mesh)
and its contact solver.

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

    tool_offset_xyz: [-0.058208, -0.028713, 0.173779]

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
release/return coordinator and records both policy data and simulation state
under `logs/policy_rollout/mujoco-<stamp>/`. It defaults to the deployed
checkpoint and spawns the nut at the trained 30 deg yaw; `--episode` spawns it
from a reference episode instead.
Use `python3 apps/policy_rollout/run_policy_rollout.py mujoco --help` for output,
preview, video, cycle, and step-limit options.

`--arm-friction-scale` sets the arm's joint dry friction and damping as a
multiple of MuJoCo Menagerie's FR3 values (default 0, the frictionless
training-asset reading of E2 in `docs/policy_rollout_training_gap_todo.md`).
With the deployed checkpoint, 0 reproduces the ros-sim behaviour (index held
open, lift at 3 s) and 1.0 the physical arm's (index closes on the nut at
3-4 s, a 55 deg turn, then a return that holds short of the 5 mm gate), so it
is the knob for reproducing either stack's failure offline; see the 2026-10-18
status in that document.

## Camera contract

Both vision heads consume aligned colour/depth at 320x180. The profile crops
the live 640x480 stream to the training view and checks the live `CameraInfo`
against the checkpoint intrinsics. Depth may be `16UC1`
millimetres or `32FC1` metres. Non-finite values and depths outside the
training camera's clipping range, (0.01, 5.0) m, are excluded by the valid
mask, as Isaac zeroed them; the RGB-D encoder also feeds that mask to the
network as a channel.

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
