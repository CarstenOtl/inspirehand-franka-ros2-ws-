# inspire_franka

ROS 2 **Jazzy** workspace combining a **Franka FR3** arm, an **Inspire
Robotics RH56** dexterous hand, and an **Intel RealSense D415**, in one Docker
image — plus a MuJoCo simulation of the arm and hand.

This repo *is* the workspace: it owns the dev image, the glue packages, the hand
driver and the vendored hand description. The Franka and RealSense stacks are
pulled in with `vcs`.

## Layout

```
docker/                          dev image, compose, entrypoint
docs/hand.md                     RS485 wiring, bring-up, and the driver's interface
docs/network.md                  network layout and FCI access
apps/camera_calibration/         calibration entry script, utilities, and hardware tests
workspace.repos                  pins the repos that vcs imports
src/
  inspire_hand_msgs/             service definitions for the hand
  inspire_hand_driver/           the RS485 driver (rclpy) and its read-only probe
  inspire_hand_description/      RH56 URDF/xacro and meshes, vendored + generated
  inspire_franka_description/    FR3 + hand composed into one description
  inspire_franka_sim/            MJCF models, controller config, MuJoCo launch
  inspire_franka_bringup/        real hardware: arm, hand, or both
  camera_calibration/            ROS node and launch files for D415 eye-to-hand calibration
  franka_ros2/                   vcs import  - Franka's stack (gitignored here)
  franka_description/            vcs import  - Franka's descriptions (gitignored here)
  realsense_d415/                vcs import  - RealSense ROS wrapper (gitignored here)
```

## Getting started

```bash
git clone <this repo> inspire_franka
cd inspire_franka

vcs import src < workspace.repos      # pulls Franka and RealSense sources

cd docker
docker compose build                  # only after editing the Dockerfile or a *_REF
docker compose up -d
docker exec -it inspire_franka bash
```

Inside the container:

```bash
rg2        # colcon build --symlink-install, Release, then source install/setup.bash
rg2test    # the tests that need neither hardware nor a simulator
```

Then pick one:

```bash
# --- Simulation (no hardware needed) ---------------------------------------
ros2 launch inspire_franka_sim sim.launch.py                      # arm + bench hand
ros2 launch inspire_franka_sim sim.launch.py headless:=false      # MuJoCo viewer
ros2 launch inspire_franka_sim sim.launch.py arm:=false           # hand only

# --- Real hardware ----------------------------------------------------------
ros2 run inspire_franka_bringup fci_check 10.7.7.7                # is the FCI up?
ros2 run inspire_hand_driver inspire_hand_probe /dev/ttyUSB0      # is the hand up?

ros2 launch inspire_franka_bringup inspire_franka.launch.py \
    robot_ip:=10.7.7.7 hand_port:=/dev/ttyUSB0                    # both

ros2 launch inspire_franka_bringup arm.launch.py robot_ip:=10.7.7.7   # arm alone
ros2 launch inspire_franka_bringup hand.launch.py port:=/dev/ttyUSB0  # hand alone

# RealSense D415 (publishes under /camera/camera by default):
ros2 launch realsense2_camera rs_launch.py device_type:=d415

# Calibrate the fixed D415 in the Franka world frame with a hand-mounted AprilTag:
./apps/camera_calibration/calibrate.py \
    --tag-id 0 --tag-size-m 0.040

# Start all three, require live telemetry, and print a per-device report:
./apps/traj_replay/tests/system_check.py --robot-ip 10.7.7.7

# Neither device present, everything else identical:
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
    use_fake_hardware:=true hand_mock:=true start_rviz:=true
```

GUIs need `xhost +local:root` on the host.

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

So "control both together" on hardware means publishing to both — which is all
it *can* mean for two mechanically independent devices bolted to the same bench.
There is no combined trajectory action and no shared clock. In simulation, where
there is no serial link, they share a controller_manager and you get exactly
that.

If you later bolt the hand to the flange and want coordinated *planning* rather
than coordinated commanding, the missing piece is a MoveIt config over the
combined description — the description already supports `hand_mount:=flange`.

### What that means on the ROS graph

Running both:

```
/joint_states                  the arm, from franka's joint_state_publisher
/robot_description             the arm's URDF
/inspire_hand/joint_states     the hand, twelve joints in radians
/inspire_hand/state            the hand, six channels as open ratios
/inspire_hand/command          command the hand
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
| `franka_ros2` | `v3.5.3` | `src/`, via `workspace.repos` | ordinary ROS packages you may want to patch |
| `franka_description` | `2.9.0` | `src/`, via `workspace.repos` | what v3.5.3's own `dependency.repos` pins |
| `libfranka` | `0.20.5` | `/opt/libfranka`, built by the image | plain CMake, and its `libfranka-common` submodule is not something `vcs import` initialises |
| `mujoco_ros2_control` | `0.1.1` | apt | released for Jazzy; no source build needed |
| `mujoco_vendor` | `0.1.0` (MuJoCo 3.12.0) | apt | the committed MJCFs were generated with exactly this MuJoCo |
| `realsense-ros` | `4.58.3` | `src/realsense_d415`, via `workspace.repos` | ROS 2 wrapper, pinned to the Jazzy-compatible release |
| `librealsense2` | `2.58.x` (at least 2.58.0) | apt | native SDK required by `realsense-ros` 4.58.3 |
| Inspire RH56 model | vendored + generated | `src/inspire_hand_description` | see its `MODEL_PROVENANCE.md` |

Two things worth knowing before bumping anything:

- **Do not run `vcs import src < src/franka_ros2/dependency.repos`**, as
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

Bump `LIBFRANKA_REF` in `docker/Dockerfile` and the `franka_ros2` pin in
`workspace.repos` together. `franka_hardware` does a versioned `find_package`, so
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
| `inspire_franka_sim` | every MJCF scene compiles; joint sets per scene; actuators only on driven joints; couplings equal the URDF's; zero contacts at rest; the bench hand sits where the URDF puts it |
| `camera_calibration` | synthetic eye-to-hand recovery, SE(3) conventions, outlier rejection and degenerate-motion detection |

The two that matter most are the cross-checks. `ros2_control` matches the
description to the simulator **by joint name and nothing else**, so a joint that
exists in both but sits somewhere different, or is coupled differently, produces
no error at all — just a robot that quietly does the wrong thing.

## Status

- **The image builds with librealsense, and all three active RealSense source
  packages compile on ROS 2 Jazzy.** The full Franka/Inspire workspace was last
  verified before the camera wrapper was added; camera streaming still requires
  a connected D415.
- **94 tests pass in the container**, 0 failures: 29 driver, 20 description, 16
  MJCF, 4 mimic cross-check, plus xmllint and lint_cmake.
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
- **Hand driver — unit-tested, not yet run against this hand.** The transport
  and register map come from a driver already working against an RH56 on a
  bench; the joint-state and coupling layer above it is new here and has only
  been exercised in mock and simulation.
- **Arm — not yet run over the FCI from this workspace.** `franka_ros2` v3.5.3
  is a major version newer than the v2.6.0 line used previously on this machine,
  so treat the first FCI connection as unproven. `fci_check` is the cheapest
  first step.
- **Flange mounting.** The default is a bench hand, which is how the hardware
  actually sits. `hand_mount:=flange` uses forgeUltra's franka-chi mounting
  convention: zero flange-to-palm translation and quaternion
  `(w, x, y, z) = (0.5, -0.5, -0.5, 0.5)`. The equivalent transforms in the
  Xacro and MJCF wrapper must remain synchronized.

### A note on linters

The three ament style linters (`copyright`, `flake8`, `pep257`) are switched off
in each `CMakeLists.txt`, with the reason recorded there. `xmllint` and
`lint_cmake` are kept and pass — they catch real mistakes in the xacro and CMake
files. ament's flake8 config prefers single quotes; this workspace, like the
others on this machine, is written with double quotes, and conforming would be
~500 edits that say nothing about correctness.
