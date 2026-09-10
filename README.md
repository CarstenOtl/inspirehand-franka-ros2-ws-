# inspire_franka

ROS 2 **Jazzy** workspace combining a **Franka FR3** arm, an **Inspire
Robotics RH56** dexterous hand, and an **Intel RealSense D415**, in one Docker
image — plus a MuJoCo simulation of the arm and hand.

This repo *is* the workspace: it owns the dev image, the glue packages, the hand
driver and the vendored hand description. The Franka and RealSense stacks are
tracked as pinned Git submodules.

## Layout

```
docker/                          dev image, compose, entrypoint
docs/hand.md                     RS485 wiring, bring-up, and the driver's interface
docs/network.md                  network layout and FCI access
apps/camera_calibration/         calibration entry scripts, utilities, and hardware tests
apps/operations/                 one-shot hand-zero and FR3-home commands
.gitmodules                     external ROS repositories
src/
  inspire_hand_msgs/             service definitions for the hand
  inspire_hand_driver/           the RS485 driver (rclpy) and its read-only probe
  inspire_hand_description/      RH56 URDF/xacro and meshes, vendored + generated
  inspire_franka_description/    FR3 + hand composed into one description
  inspire_franka_sim/            MJCF models, controller config, MuJoCo launch
  inspire_franka_bringup/        real hardware: arm, hand, or both
  franka_trajectory_replay/      guarded FR3 replay controller and preparation
  inspire_franka_trajectory_replay/ coordinated FR3 + Inspire replay runner
  camera_calibration/            D415 eye-to-hand calibration: automated runner, passive recorder
  franka_ros2/                   submodule - Franka's stack
  franka_description/            submodule - Franka's descriptions
  realsense_d415/                submodule - RealSense ROS wrapper
```

## Getting started

```bash
git clone --recurse-submodules <this repo> inspire_franka
cd inspire_franka

cd docker
docker compose build                  # only after editing the Dockerfile or a *_REF
docker compose up -d
docker exec -it inspire_franka bash
```

For an existing clone, initialize the newly added submodules once:

```bash
git submodule update --init --recursive
```

Inside the container:

```bash
rg2        # colcon build --symlink-install, Release, then source install/setup.bash
rg2test    # the tests that need neither hardware nor a simulator
```

Then pick one:

```bash
# --- Simulation (no hardware needed) ---------------------------------------
ros2 launch inspire_franka_sim sim.launch.py                      # arm + flange hand
ros2 launch inspire_franka_sim sim.launch.py headless:=false      # MuJoCo viewer
ros2 launch inspire_franka_sim sim.launch.py arm:=false           # hand only

# Read-only MuJoCo visualization/replay; this is not a simulator control GUI.
# With default real-hardware bringup it mirrors the FR3 on /joint_states only.
ros2 launch inspire_franka_sim passive_viewer.launch.py

# --- Real hardware ----------------------------------------------------------
ros2 run inspire_franka_bringup fci_check 172.16.0.2               # is the FCI up?
ros2 run inspire_hand_driver inspire_hand_probe /dev/ttyUSB0      # is the hand up?

ros2 launch inspire_franka_bringup inspire_franka.launch.py       # both

ros2 launch inspire_franka_bringup arm.launch.py                  # arm alone
ros2 launch inspire_franka_bringup arm.launch.py \
    gravity_compensation:=true                                    # arm, hand-guiding/float mode
ros2 launch inspire_franka_bringup hand.launch.py                 # hand alone

# In another sourced shell, zero/open the hand or home the arm. The arm command
# asks for confirmation and uses the valid Franka ready pose (not seven zeros).
./apps/operations/zero_hand.py
./apps/operations/home_arm.py
# Options and safety behavior: apps/operations/README.md

# Coordinated replay: example joint-impedance effort law + recorded waypoints.
# This launch replaces ordinary bringup for the session.
ros2 launch inspire_franka_trajectory_replay replay.launch.py

# Either device alone. The runner flag must match the launch argument:
# arm:=false with --no-arm, hand:=false with --no-hand.
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
    arm:=false

# In another sourced shell: run the hardware-validated legacy traj_1 baseline.
# This already-generated artifact contains the historical +90-degree joint-7
# retarget. Do not add or remove that offset from this retained baseline.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
    apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
    --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
    --close-support-fingers

# traj_2 and every later policy capture use the current hardware orientation:
# replay joint 7 exactly as recorded and never apply the legacy +90-degree step.
# This five-cycle candidate has passed dry-run preparation; remove --dry-run
# only for an intentional hardware validation run.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
    apps/traj_replay/demo_trajs/traj_2_5x \
    --home apps/traj_replay/demo_trajs/traj_2_5x/homing.yaml \
    --time-scale 5 --interactive-pause \
    --max-prepared-duration 300 --dry-run

# Pickup rollout with live scene-adjustment pauses. During replay, SPACE
# smoothly pauses and holds both devices; SPACE resumes and q aborts.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
    apps/traj_replay/demo_trajs/pickup_1 --env 0 --segment 1 \
    --home apps/traj_replay/demo_trajs/homing/pickup_multi.yaml \
    --interactive-pause

# Slow every recorded waypoint interval (and the interpolated stream) by 5x.
# The coordinated hand follows the same scaled trajectory clock.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
    apps/traj_replay/demo_trajs/threading_cycle1_flange180 \
    --home apps/traj_replay/demo_trajs/threading_cycle1_flange180/homing.yaml \
    --close-support-fingers \
    --time-scale 5

# Hand only. Raw traj_1 remains valid here because --no-arm never commands
# joint 7. It is otherwise a source/legacy-mount artifact, not a hardware-arm
# baseline. Hand-only replay uses the recording's own clock.
ros2 run inspire_franka_trajectory_replay replay_trajectory \
    apps/traj_replay/demo_trajs/traj_1 --cycle 1 --no-arm \
    --home apps/traj_replay/demo_trajs/homing/threading.yaml

# Inspect the source trajectory in Chi's MuJoCo replay viewer. Do not run a
# second ROS/MuJoCo launcher alongside it.
python3 apps/traj_replay/tests/test_mujoco_traj_replay.py \
    --trajectory apps/traj_replay/demo_trajs/traj_1

# RealSense D415 (publishes under /camera/camera by default):
ros2 launch realsense2_camera rs_launch.py device_type:=d415

# Calibrate the fixed D415 against the Franka base frame with a hand-mounted
# AprilTag. The arm drives itself through generated poses and stops at each one;
# teach the seed poses once under gravity compensation first. The frame and
# namespace flags match franka.launch.py's own bringup - see the app's README.
./apps/camera_calibration/auto_calibrate.py --teach \
    --namespace "" --world-frame base --base-frame fr3_link0
./apps/camera_calibration/auto_calibrate.py \
    --namespace "" --world-frame base --base-frame fr3_link0 \
    --tag-id 0 --tag-size-m 0.040

# The passive hand-guided recorder, when no controller can be brought up:
./apps/camera_calibration/calibrate.py \
    --tag-id 0 --tag-size-m 0.040

# Start all three, require live telemetry, and print a per-device report:
./apps/traj_replay/tests/system_check.py --robot-ip 172.16.0.2

# Neither device present, everything else identical:
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
    use_fake_hardware:=true hand_mock:=true start_rviz:=true
```

The real-arm launch files default `robot_ip` to the fixed Control C2 address
`172.16.0.2`, so it does not need to be repeated. Pass `robot_ip:=...` only when
temporarily targeting a different control box. `fci_check` is a standalone
executable rather than a launch file, so its IP remains a positional argument.

### Simulation GUI versus passive viewer

`sim.launch.py` and `passive_viewer.launch.py` deliberately open different
MuJoCo integrations:

| launch | purpose | where motion comes from |
|---|---|---|
| `sim.launch.py headless:=false` | simulated robot plant, with physics and `ros2_control` | simulated controllers and ROS command/action topics |
| `sim.launch.py start_rviz:=true` | RViz view of the simulated robot | TF and `/joint_states` |
| `passive_viewer.launch.py` | read-only display of a state topic or planned trajectory | `/joint_states` or `/mujoco_sim/joint_trajectory` |

The passive viewer is useful for inspecting a recording or previewing a
trajectory in the MJCF scene without starting a controller manager. It can also
display live measurements, but it does not connect to hardware itself: it only
subscribes to ROS messages already published by a driver. With the default
real-hardware bringup, `/joint_states` contains the physical FR3 while
`/inspire_hand/joint_states` contains the physical hand. One viewer instance
accepts one joint-state topic, so the default invocation displays the arm only;
use `joint_state_topic:=/inspire_hand/joint_states` to display the hand instead.
It does not currently merge the two real-hardware topics into one combined
model.

The passive viewer's side-panel sliders are **not functional pose controls** in
this node. They edit MuJoCo actuator controls, while the node deliberately
disables actuation, does not step physics by default, and clears those controls
before every optional step. Disabling the joint-state subscriber does not change
that. The viewer publishes no ROS commands and cannot configure or move either
the physical robot or the `ros2_control` simulator.

The passive viewer now hides both MuJoCo sidebars by default. Press `Tab` for the
left panel or `Shift+Tab` for the right panel, or launch with
`show_left_ui:=true` and/or `show_right_ui:=true`. Camera navigation remains
available with the mouse while the panels are hidden.

The current `mujoco_ros2_control` window opened by `sim.launch.py` has no slider
panel either; setting a command interface to `none` merely leaves those joints
unclaimed. Set a passive-viewer pose by publishing a `JointState` or
`JointTrajectory`, and drive the full simulation through the ROS trajectory or
forward-command controllers documented in `src/inspire_franka_sim/README.md`.

If a previous simulation or hardware launch was not fully stopped, its
`/controller_manager` can capture controller requests from the next launch. Stop
old launch terminals with Ctrl-C. When it is unclear which ROS processes remain,
reset the container from the host:

```bash
docker restart inspire_franka
docker exec -it inspire_franka bash
```

After restarting, `ros2 control list_controllers` should fail until a new launch
starts; that confirms there is no stale controller manager.

GUIs need `xhost +local:root` on the host.

## RealSense D415: connection, live view, and recording

The D415 connects directly to a USB 3 host port and is passed into the
container as raw USB (`/dev/bus/usb`) and V4L2 (`/dev/video*`) devices. The
workspace is bind-mounted at `/root/develop_ws`; the ROS wrapper source lives
in `src/realsense_d415` and its built package in
`install/realsense2_camera`.

Before launching, confirm the host sees the actual D415 (not another webcam)
and that it has a USB 3 link:

```bash
lsusb -d 8086:0ad3       # Intel RealSense D415
lsusb -t                 # expect 5000M or higher; 480M is USB 2
```

Only one `realsense2_camera` node may own the physical device. Start the
recommended 640x480 at 30 FPS color/depth stream from a sourced container
shell:

```bash
ros2 launch realsense2_camera rs_launch.py \
  device_type:=d415 \
  camera_namespace:=camera camera_name:=camera \
  enable_color:=true enable_depth:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  enable_sync:=true align_depth.enable:=true
```

It runs as `/camera/camera` and publishes, among others:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/aligned_depth_to_color/image_raw  # when alignment is enabled
```

From a second sourced shell, inspect the stream and its actual rate:

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/depth/image_rect_raw
ros2 run rqt_image_view rqt_image_view
# Side-by-side RGB + metric depth; topics are selected automatically:
ros2 run camera_calibration rgbd_view
```

In `rqt_image_view`, select either image topic above. The calibration viewers
are also available; because the driver is already running, they must attach
with `--no-launch`:

```bash
./apps/camera_calibration/tests/test_camera.py --no-launch
./apps/camera_calibration/tests/test_april_tag.py --no-launch \
  --tag-family tag36h11 --tag-id 0 --tag-size 0.040
```

The camera node publishes live data only; to retain footage, record the ROS
topics as a bag:

```bash
ros2 bag record -o d415_check \
  /camera/camera/color/image_raw \
  /camera/camera/color/camera_info \
  /camera/camera/depth/image_rect_raw
```

If a D415 falls back to USB 2, use a USB 3 cable/port rather than expecting
reliable synchronized 30 FPS color and depth. The extended calibration and
camera troubleshooting guide is in
[apps/camera_calibration/README.md](apps/camera_calibration/README.md) -
including why the automated run exists, what the first hand-guided run on real
hardware got wrong, and how to read the measured camera-to-robot clock delay.

After a camera crash or `Depth stream start failure`, stop the existing camera
node before starting another one. A one-time device reset is available from
either viewer with `--initial-reset`; do not use it while another process owns
the D415.

### Manual eye-to-hand camera calibration

Build and source the calibration package after changing it:

```bash
colcon build --symlink-install --packages-select camera_calibration
source install/setup.bash
```

Start the FR3 in gravity-compensation mode in one sourced shell so the
timestamped `fr3_link0 -> fr3_link8` transform remains available while the arm
is moved by hand:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
  gravity_compensation:=true
```

In a second sourced shell, start manual calibration. By default this command
starts the D415 itself; add `--no-camera` only when one camera node is already
publishing `/camera/camera/color/image_raw` and `camera_info`.

```bash
./apps/camera_calibration/calibrate.py --manual \
  --tag-family tag36h11 --tag-id 0 --tag-size-m 0.040
```

At each pose, hold the arm completely still with the whole tag sharp and
visible, then press Enter once. Wait for `Accepted valid sample` before moving.
Use varied image positions, distances, roll, pitch, and yaw. The solver runs
after 12 accepted samples and writes the annotated captures plus
`calibration_result.json` under `logs/<UTC timestamp>/`.

Lower final translation/rotation RMSE is better; also check how many samples
were rejected. A small tag reprojection error only confirms image detection,
so it does not replace the final cross-pose RMSE or a visual check. Inspect the
result in MuJoCo from inside the development container:

```bash
./apps/camera_calibration/tests/visualize_calibrated_camera.py \
  /root/develop_ws/logs/<UTC timestamp>/calibration_result.json --live
```

With the robot bringup and one RGB camera producer still running, this opens a
side-by-side real/MuJoCo comparison driven by the FR3 angles on
`/joint_states`. Confirm that the camera viewpoints agree across several arm
poses. Omit `--live` for the offline passive viewer, where `C` or `2` selects
the calibrated RGB point of view and `F` or `1` selects the external overview.
The complete calibration and troubleshooting guide is in
[apps/camera_calibration/README.md](apps/camera_calibration/README.md).

## The one structural thing to understand

**On real hardware the arm and the hand are two independent stacks. In
simulation they are one.**

That asymmetry is deliberate, and it is the thing that explains most of the
layout:

| | real hardware | simulation |
|---|---|---|
| arm | `franka_hardware` ros2_control component, 1 kHz over the FCI | one `<ros2_control>` component… |
| hand | `inspire_hand_driver`, a plain rclpy node, ~50 Hz Modbus RTU over RS485 | …shared with the arm |
| shared | ROS graph, TF tree | controller_manager, clock, MuJoCo world |

The reason is the hand's link. RS485 is half-duplex and the driver's cycle is
three register round-trips; a read takes milliseconds and can time out. Put that
inside the arm's 1 kHz `read()`/`write()` and you stall the FCI loop, which ends
the connection. Keeping the hand as its own node means its worst case costs the
arm nothing.

The hand is physically bolted to the flange, but "control both together" on
hardware still means publishing to both independent drivers. There is no
combined trajectory action and no shared clock. In simulation, where there is no
serial link, they share a controller_manager.

Coordinated *planning* still needs a MoveIt config over the combined
description. Whenever both assets are enabled, that description attaches the
hand to the flange automatically.

### What that means on the ROS graph

Running both:

```
/joint_states                  the arm, from franka's joint_state_publisher
/robot_description             the arm's URDF
/inspire_hand/joint_states     the hand, twelve joints in radians
/inspire_hand/state            the hand, six channels as open ratios
/inspire_hand/command          command the hand, open ratios (1.0 = open)
/hand/robot_description        the hand's URDF, namespaced so it does not
                               collide with the arm's
/tf                            both, merged
```

The hand's `robot_state_publisher` is namespaced only when the arm is also
running: two latched publishers on `/robot_description` means RViz shows
whichever it happened to hear last. `inspire_franka_bringup`'s RViz config has
one RobotModel display per description topic.

## Six actuators, twelve joints

The RH56 has six motors. Each finger's `*_intermediate` joint follows its
`*_proximal` joint through a four-bar linkage, and the thumb has two such
followers, so **commands address six joints and state reports twelve**.

Publishing only the driven six would leave `robot_state_publisher` unable to
place a single fingertip, so the driver computes the followers from the
coupling. Commanding a follower is rejected rather than silently redirected —
the hardware cannot do it.

The same six multiplier/offset pairs necessarily appear in three places: the
URDF's `<mimic>` tags, `inspire_hand_driver.kinematics`, and the MuJoCo equality
constraints. They are duplicated because the consumers are a xacro file, a
Python module and an XML generator, with no natural shared format — so two tests
fail if the copies ever disagree
(`inspire_hand_description/test/test_mimic_matches_driver.py`,
`inspire_franka_sim/test/test_mjcf.py`).

Details and the derivation are in
[`src/inspire_hand_description/MODEL_PROVENANCE.md`](src/inspire_hand_description/MODEL_PROVENANCE.md).

## Versions, and where each piece lives

| | version | where | why |
|---|---|---|---|
| ROS 2 | Jazzy | base image | |
| `franka_ros2` | `v3.5.3` | `src/franka_ros2` submodule | ordinary ROS packages you may want to patch |
| `franka_description` | `2.9.0` | `src/franka_description` submodule | what v3.5.3's own `dependency.repos` pins |
| `libfranka` | `0.20.5` | `/opt/libfranka`, built by the image | plain CMake, built separately with its own nested submodule |
| `mujoco_ros2_control` | `0.1.1` | apt | released for Jazzy; no source build needed |
| `mujoco_vendor` | `0.1.0` (MuJoCo 3.12.0) | apt | the committed MJCFs were generated with exactly this MuJoCo |
| `realsense-ros` | `4.58.3` | `src/realsense_d415` submodule | ROS 2 wrapper, pinned to the Jazzy-compatible release |
| `librealsense2` | `2.58.x` (at least 2.58.0) | apt | native SDK required by `realsense-ros` 4.58.3 |
| Inspire RH56 model | vendored + generated | `src/inspire_hand_description` | see its `MODEL_PROVENANCE.md` |

Two things worth knowing before bumping anything:

- **Do not import `src/franka_ros2/dependency.repos`**, as
  upstream's README says to. It would clone a second `franka_description` at the
  same tag, a `libfranka` that will not build here, and source checkouts of
  `ros2_control`, `gz_ros2_control`, MoveIt and RealSense/ZED/Robotiq
  descriptions — none of which this image needs, because Jazzy's apt
  `ros2_control` is new enough for v3.5.x.
- **Only the real-arm core of `franka_ros2` is built.** `docker/entrypoint.sh`
  drops a `COLCON_IGNORE` into the packages needing Gazebo, the full MoveIt
  stack, a mobile base, or RealSense/SICK/ZED hardware, plus the vendored
  `realtime_tools` copy (Jazzy's apt one is newer and building the vendored one
  would shadow it workspace-wide). Set `FRANKA_ROS2_BUILD_ALL=1` in
  `docker-compose.yml` to opt back in, and install those dependencies yourself.

Bump `LIBFRANKA_REF` in `docker/Dockerfile` and the `src/franka_ros2` submodule
commit together. `franka_hardware` does a versioned `find_package`, so
a too-old libfranka fails at build time — but a too-new one does not, and the
real constraint on hardware is the arm's own system version. Check Franka's
compatibility matrix.

## Tests

Everything that can be checked without hardware or a simulator is a test, and
`rg2test` runs the lot:

| package | covers |
|---|---|
| `inspire_hand_driver` | Modbus/legacy framing and CRCs, open-ratio conversions, the channel↔joint mapping and its coupling |
| `inspire_hand_description` | the shipped URDF's `<mimic>` values still match the driver's table |
| `inspire_franka_description` | every launch variant expands to one root link with no dangling joints; `ros2_control` names only joints that exist; followers expose no command interface; unsupported combinations fail loudly |
| `inspire_franka_sim` | every MJCF scene compiles; joint sets per scene; actuators only on driven joints; couplings equal the URDF's; zero contacts at rest; the flange hand matches the combined URDF |
| `camera_calibration` | synthetic eye-to-hand recovery, SE(3) conventions, outlier rejection and degenerate-motion detection; AprilTag corner ordering; generated pose programs against joint limits, field of view and rotation spread; standstill detection; clock-offset recovery; and the whole automated run against a simulated arm and camera |
| `apps/operations` | valid FR3 home target and limits, namespace/prefix handling, feedback matching, and safe controller-conflict selection |

The two that matter most are the cross-checks. `ros2_control` matches the
description to the simulator **by joint name and nothing else**, so a joint that
exists in both but sits somewhere different, or is coupled differently, produces
no error at all — just a robot that quietly does the wrong thing.

## Status

- **The image builds with librealsense, and all three active RealSense source
  packages compile on ROS 2 Jazzy.** The full Franka/Inspire workspace was last
  verified before the camera wrapper was added; camera streaming still requires
  a connected D415.
- **153 tests pass in the container**, 0 failures, including the 48 driver
  tests (29 originally, plus the endianness regression and the command
  unit/range tests below). Add `inspire_franka_trajectory_replay` to
  `rg2test`'s selection to pick up its 9 as well; the stock selection misses
  them. Note that `rg2test` selects
  `camera_calibration`, whose package has no `test/` directory - its tests live
  in `apps/camera_calibration/tests/` - so pytest collects nothing there and
  colcon reports the package as failed with exit code 5. That is a stale
  selection in `rg2test`, not a broken test. Run those 53 directly:

  ```bash
  python3 -m pytest apps/camera_calibration/tests
  ```
- **Camera calibration — automated, not yet run on the arm.** The first
  hand-guided run on real hardware solved to 85.0 mm / 20.47 deg with a
  flange-to-tag offset of 1.39 m, which is meaningless: its samples were taken
  while the arm moved, and the log shows the image stamps running ahead of the
  newest joint state. `auto_calibrate.py` replaces that with generated poses the
  arm drives itself, a standstill at every pose proved by the tag's corner
  scatter, and a final moving pass that measures the camera-to-robot clock delay
  instead of assuming it away. Against a simulated arm and camera the whole run
  recovers a known camera pose to 0.2 mm and a known 40 ms delay to 0.1 ms
  (`apps/camera_calibration/tests/test_auto_calibration.py`). **Still to do on
  hardware:** teach the seed poses, run it, and confirm the residual and the
  measured delay against the numbers above.
- **Trajectory replay — example joint-impedance effort mode is now the hardware
  default.** The user confirmed the simple upstream example works on the FR3
  and the host has a real-time kernel. Replay now uses that example's PD torque
  law and gains with a waypoint reference, through the existing replay plugin.
  The previous position-JTC path reflexed during homing; its passing MuJoCo
  run does not validate the new effort path. On 2026-09-08, the user confirmed
  the current simple joint-impedance replay works well on the arm; quantitative
  tracking accuracy has not been measured. See `src/inspire_franka_trajectory_replay/README.md`
  for build and arm-only test commands. The retained MuJoCo replay launch needs
  `--arm-controller position-jtc` on the runner.
  On 2026-09-09, coordinated threading cycle 1 was confirmed on hardware with
  the correct tool orientation and a good replay using the legacy-source
  180-degree flange baseline: a `+90 deg` retarget of `fr3_joint7`, its matching
  retargeted home, and `--close-support-fingers`. The checked-in
  `demo_trajs/threading_cycle1_flange180` artifact is the default hardware
  baseline. Raw `traj_1`, raw `homing/threading.yaml`, and the unretargeted
  `threading_5x` artifact are outdated for arm replay with the current mount.
  This compensation is historical, not the current capture workflow. `traj_2`
  and all later policy trajectories are developed with the hardware joint
  orientation already correct: joint 7 is replayed unchanged with a zero
  offset, and `retarget_flange_mount.py` must not be applied. The checked-in
  `traj_2_cycle3` and `traj_2_5x` candidates follow that rule and pass dry-run
  preparation; physical validation is still recorded separately.
- **Simulation — verified end to end.** MuJoCo runs headless with
  `MujocoSystemInterface` active at 1 kHz, all three controllers active, sim
  clock advancing. Arm and hand trajectories sent *simultaneously* both report
  `SUCCEEDED`:

  | | commanded | reached |
  |---|---|---|
  | `fr3_joint1` | +0.5 | +0.5162 |
  | `fr3_joint4` | −2.0 | −2.0046 |
  | `index_proximal_joint` | +1.2 | +1.2001 |
  | coupled `index_intermediate_joint` | — | 0.021 mrad from its coupling |

  The arm's ~16 mrad steady-state error on `joint1` is the inherited
  Menagerie-derived PID gains under gravity, not a plumbing problem; the hand,
  which was tuned here, lands within 0.1 mrad. Tune `config/pids.yaml` if you
  need the arm tighter.
- **Hand driver — verified against this hand.** Read, write and a full
  trajectory replay all confirmed on the bench RH56 over RS485 at 50 Hz.

  Getting there required a real fix. The Modbus path encoded and decoded
  register *values* little-endian; Modbus RTU is big-endian, and the register
  addresses in the same frames were already correct. Nothing caught it, because
  `test_protocol.py` built its fake replies with the driver's own byte order —
  the test and the code shared one assumption, so 29 tests passed against a
  driver that could not talk to the hardware. It read `HAND_ID` 1 as 256 and a
  fully-open angle of 1000 as 59395, and the same swap on the write path turned
  every commanded angle into an out-of-range value. `test_protocol.py` now
  asserts both directions against bytes captured from the hand.

  A second problem sat in the node above the transport, and inferring the unit
  was the root of it. `_apply` decided radians-or-ratios from the *naming* --
  joint names meant radians, channel ids meant ratios -- which put two
  conventions on one `position` field. `1.5` as a channel id clamped to a
  fully open hand; the same `1.5` as a joint name clamped to a fully closed
  one. One number, opposite ends of travel, nothing logged either way. It also
  made `SetAngles` misread its own field: called with joint names, the field
  literally named `open_ratio` was read as radians, so `0.0` -- fully closed --
  produced a fully **open** hand.

  **Commands are now open ratios everywhere: `1.0` fully open, `0.0` fully
  closed.** Names address DOF and nothing else, so channel ids and joint names
  may be mixed. Out-of-range targets are **rejected, not clamped** -- the whole
  message fails and the log names the offending entries, because clamping is
  what turned a typo into a full-travel move.

  Two consequences worth knowing:

  - **Commands and `joint_states` deliberately run opposite ways.**
    `joint_states` stays in radians, where `0.0` is the *open* pose, because
    that is what the URDF, `robot_state_publisher` and TF need. So a rising
    `joint_states` value means a closing hand, and a rising commanded ratio
    means an opening one.
  - **Callers holding radians convert at the boundary.**
    `inspire_franka_trajectory_replay` does exactly that in `command_hand`;
    its trajectories and homing YAMLs stay in radians, because those files
    also carry the FR3's seven joints and a mixed-unit YAML would be worse
    than a conversion. That runner also now derives the hand's joint limits
    from `inspire_hand_driver.kinematics` instead of keeping its own copy --
    the copy is what the conversion divides by, so a drifted one would have
    silently mis-scaled every hand command rather than failing a comparison.

  Nothing covered the node's command paths at all before this;
  `test_command_units.py` now pins the unit and the range on both paths, and
  `test_replay.py` pins the radian-to-ratio conversion at both ends of travel.

  Measured tracking, cycle 1 of the checked-in threading recording, hand-only
  at the recording's native 15 Hz resampled to a 50 Hz command stream (jitter
  1.4 ms sd, worst interval 25 ms):

  | joint | commanded range | RMSE | worst error | lag |
  |---|---|---|---|---|
  | `index_proximal_joint` | 0.657 | 0.027 | 0.083 | ~185 ms |
  | `thumb_proximal_yaw_joint` | 0.300 | 0.035 | 0.048 | ~165 ms |
  | `thumb_proximal_pitch_joint` | 0.305 | 0.009 | 0.023 | ~180 ms |
  | three held fingers | 0.022 | 0.004 | 0.012 | not resolvable |

  All radians, error quoted after shifting by the lag. **The ~0.17 s lag is the
  hand's own closed-loop response, and it is the number that matters for
  coordinated replay**: the RH56 takes a position target, not a trajectory, so
  the hand trails the arm by about that much regardless of the command rate.
  The held fingers move 0.022 rad in total, which is too little to fit a lag
  to — their figure is curve-fitting noise, not a measurement.
- **Arm — not yet run over the FCI from this workspace.** `franka_ros2` v3.5.3
  is a major version newer than the v2.6.0 line used previously on this machine,
  so treat the first FCI connection as unproven. `fci_check` is the cheapest
  first step.
- **Flange mounting.** When arm and hand are enabled together, both Xacro and
  MuJoCo attach `hand_mount` to the flange with zero translation and a 180-degree
  clocking angle. Their transforms must remain synchronized. On real hardware,
  Franka Desk's active end-effector profile is
  the sole source of hand mass, centre of mass, and inertia; ROS does not add a
  runtime payload.

### A note on linters

The three ament style linters (`copyright`, `flake8`, `pep257`) are switched off
in each `CMakeLists.txt`, with the reason recorded there. `xmllint` and
`lint_cmake` are kept and pass — they catch real mistakes in the xacro and CMake
files. ament's flake8 config prefers single quotes; this workspace, like the
others on this machine, is written with double quotes, and conforming would be
~500 edits that say nothing about correctness.
