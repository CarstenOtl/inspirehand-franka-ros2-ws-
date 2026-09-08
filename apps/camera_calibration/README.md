# RealSense D415 eye-to-hand calibration

This app estimates a fixed table camera's pose in the Franka `world` frame.
An AprilTag is stuck rigidly to the back of the Inspire Hand. Its placement
does not need to be measured: the solver estimates both `world -> camera` and
the unknown carrier-to-tag transform from the motion.

There are two entry points, and they are not interchangeable:

| | `auto_calibrate.py` | `calibrate.py` |
|---|---|---|
| who moves the arm | the script, through the replay controller's guarded `goto` | you, by hand |
| when samples are taken | at a standstill, averaged over a burst of frames | in passing, one frame each |
| pose selection | generated from the camera's point of view | whatever you happened to do |
| clock skew | cannot affect the static result; measured separately | enters every sample |
| use it | **normally** | when no controller can be brought up |

Use `auto_calibrate.py`. The passive recorder is kept because it needs nothing
but gravity compensation, but its numbers are only as good as the clock
agreement between the camera and the robot - see below for what that cost.

## Before running

1. Print a supported AprilTag (default `tag36h11`, ID 0), keep it flat, and
   attach it to the back of the palm—not to a moving finger.
2. Measure the outer edge of the black square, excluding the white paper, in
   metres. Pose scale depends directly on this value.
3. Start the robot so `world -> fr3_link8` is available: under the replay
   controller for the automated run, or with `gravity_compensation:=true` for
   hand guiding and for teaching seed poses.
4. Build and source the workspace (`rg2` in this workspace).

## Camera connection, ROS graph, and live views

The D415 is a USB 3 camera. Connect it directly to a USB 3 (or faster) host
port with a USB 3 cable; avoid hubs while bringing it up. On the host,
`lsusb -t` should show its link at `5000M` or higher. `480M` means USB 2,
which cannot reliably carry synchronized D415 color and depth at 30 FPS.
Confirm that the device itself is present with `lsusb -d 8086:0ad3`; do not
mistake another UVC webcam for the D415.

`docker-compose.yml` passes both `/dev/bus/usb` (used by librealsense control
transfers) and `/dev/video*` (the V4L2 stream nodes) into the container. The
workspace itself is bind-mounted: the host checkout is
`<checkout>/`, and the same files appear in the container at
`/root/develop_ws/`. The relevant software is:

- `/root/develop_ws/src/realsense_d415/`: the `realsense-ros` source checkout
  imported by `vcs import src < workspace.repos`.
- `/root/develop_ws/install/realsense2_camera/`: the built ROS 2 overlay.
- `/root/develop_ws/apps/camera_calibration/tests/test_camera.py`: RGB/depth
  viewer.
- `/root/develop_ws/apps/camera_calibration/tests/test_april_tag.py`: annotated
  RGB and aligned-depth AprilTag viewer.

One `realsense2_camera` node must be the sole owner of the physical D415. It
normally runs as `/camera/camera`, publishes raw color and depth under
`/camera/camera`, and publishes the D415 sensor transforms. The AprilTag
viewer additionally requires `align_depth.enable:=true`, which provides
`/camera/camera/aligned_depth_to_color/image_raw` in color-camera pixels.
The camera never writes footage by itself; ROS topics are live in memory.

Start one camera producer (the recommended 640x480 at 30 FPS D415 profile):

```bash
ros2 launch realsense2_camera rs_launch.py \
  device_type:=d415 \
  camera_namespace:=camera camera_name:=camera \
  enable_color:=true enable_depth:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Then inspect its actual output from a second sourced shell:

```bash
ros2 node list
ros2 topic list
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/depth/image_rect_raw
ros2 run rqt_image_view rqt_image_view
```

In `rqt_image_view`, select `/camera/camera/color/image_raw` or
`/camera/camera/depth/image_rect_raw`. This is the best basic live-image check;
the Python viewers are diagnostic tools and may render more slowly than the
incoming ROS stream. To save actual footage for later replay, record a rosbag:

```bash
ros2 bag record -o d415_check \
  /camera/camera/color/image_raw \
  /camera/camera/color/camera_info \
  /camera/camera/depth/image_rect_raw
```

Run either viewer as the camera producer, or use `--no-launch` if the command
above is already running. Do not run `test_camera.py` and `test_april_tag.py`
without `--no-launch` at the same time: that starts two drivers against one USB
device and causes `Device or resource busy` / depth-stream failures. The
viewers now detect the usual `/camera/camera` collision and ask you to use
`--no-launch`.

If the D415 reports `Depth stream start failure` after a crash, stop the active
camera node and retry one viewer with `--initial-reset`. This resets the
physical camera, so it must not be used while another process owns the device.

The hardware viewers are useful preflight checks. Close each before starting
calibration because each launches its own camera by default. Both request
640x480x30 by default and redraw at up to 30 Hz; use `--no-launch` when a
camera node is already running:

```bash
./apps/camera_calibration/tests/test_camera.py
./apps/camera_calibration/tests/test_april_tag.py \
  --tag-family tag36h11 --tag-id 0 --tag-size 0.040
```

## Why the automated run exists

The first hand-guided run on real hardware collected 20 samples with clean tag
detections - reprojection errors of 0.04 to 0.42 px - and solved to:

```
19 samples used, 1 rejected, RMSE 85.0 mm / 20.47 deg
estimated fr3_link8 -> apriltag_0 translation: (0.017, -0.049, -1.393) m
```

The tag is glued to the palm, so that last number has to be a few centimetres.
It came out at 1.39 m, and the camera pose was fitted to compensate. The result
is not slightly wrong, it is meaningless - and nothing in the per-frame
diagnostics said so.

The cause is in the same log, repeated throughout:

```
No base -> fr3_link8 TF at the image time: Lookup would require extrapolation
into the future. Requested time 1788719916.493552 but the latest data is at
time 1788719916.486465
```

The images are stamped ahead of the newest joint state. The recorder pairs each
image with the robot pose at that image's stamp, so every sample is a picture of
the tag in one place matched against a robot pose from a different moment. At
hand-guiding speed a few tens of milliseconds is tens of millimetres, and
because the error follows the motion it biases the solution instead of averaging
out. `joint_state_rate` also defaults to 30 in `arm.launch.py`, which quantises
the robot side to 33 ms on top of that.

Two secondary contributors: a 40 mm tag at about a metre constrains its own
out-of-plane rotation weakly no matter how precisely the corners are found, and
one frame per sample carries the detector's full pixel noise.

The automated run answers all three. It stops, so no delay can matter. It
averages a burst at each standstill. It picks poses that use the whole image and
rotate about three axes. Against synthetic per-pose noise of 0.35 deg and 0.8 mm
- what an averaged burst of a 40 mm tag actually gives - the same solver recovers
the camera to under a millimetre.

## Automated calibration

### What has to be running

The script commands no hardware itself; it needs a camera producer and the
replay controller, both up before it starts.

```bash
# 1. one camera node, as above
ros2 launch realsense2_camera rs_launch.py \
  device_type:=d415 camera_namespace:=camera camera_name:=camera \
  enable_color:=true rgb_camera.color_profile:=640x480x30

# 2. the arm under the guarded replay controller
ros2 run franka_trajectory_replay preflight.py --host 172.16.0.2
ros2 launch franka_trajectory_replay replay.launch.py \
  robot_config_file:=/root/develop_ws/src/inspire_franka_trajectory_replay/config/robot.config.yaml
```

That config is the one to use on this rig: `robot_ip: 172.16.0.2`, no Franka
gripper, and `namespace: ""`. The namespace matters. `replay.yaml` defaults to
`NS_1`, so without `--namespace ""` the run looks for the controller and the
joint states under `/NS_1/...` and simply waits for a status that never comes.
The first line it prints says which namespace and joint-state topic it settled
on; check it against `ros2 topic list` if anything hangs.

Two frame arguments also have to match this bringup. `franka.launch.py`
publishes the arm under its own root, `base`, and there is no `world` frame -
which is why the earlier hand-guided run was started with `world_frame:=base`.
`base -> fr3_link0` is identity, so:

```bash
--world-frame base --base-frame fr3_link0
```

Every motion the run makes is a `goto` on that controller: one synchronous
quintic ramp on all seven joints, at least 5 s long, stretched to keep the peak
velocity under 0.5 rad/s and the acceleration under 1 rad/s², rejected outright
if it would exceed a joint limit. **The controller holds position on activation;
nothing moves until the script asks.** Ctrl-C sends its abort, which decelerates
over 0.5 s and holds.

### Teaching the seed poses, once

Generating poses needs a rough idea of where the camera is, and nothing in the
system knows that yet. The first run gets it from a handful of hand-taught
configurations - joint values only, no calibration is computed while you teach:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
  robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0 gravity_compensation:=true

./apps/camera_calibration/auto_calibrate.py --teach \
  --namespace "" --world-frame base --base-frame fr3_link0
```

Guide the hand somewhere the whole tag is in view, let go, press Enter. Six is
the minimum; eight to ten spread across the camera's field of view, with clearly
different tilts, makes the coarse solve safe. The poses are written to
`~/camera_calibration_runs/seed_poses.yaml`, and the run refuses to record a
pose where the tag is not currently being detected.

Restart the arm under the replay controller afterwards - teaching needs gravity
compensation, driving needs the replay controller, and they are different
controllers.

### The run

```bash
./apps/camera_calibration/auto_calibrate.py \
  --namespace "" --world-frame base --base-frame fr3_link0 \
  --tag-family tag36h11 --tag-id 0 --tag-size-m 0.040
```

Four passes, each gated by Enter unless `--yes`:

1. **Coarse pass.** Drives the taught seed poses, stopping at each, and solves
   for a rough `world -> camera`. Its flange-to-tag offset is printed as a
   sanity check: if it is not within a few centimetres, stop and fix the tag or
   its measured edge length before going further.
2. **Pose program.** Samples where the tag should sit in the image, over a grid
   of nine cells, at 0.35-0.85 m, tilted up to 35° about two axes and rolled up
   to 60°; turns each into the flange pose that puts it there; and solves
   inverse kinematics for it. Candidates are discarded for leaving the image,
   grazing the tag beyond 55°, missing a joint limit margin, dropping the flange
   below 0.15 m, duplicating an orientation already chosen, or sitting more than
   one ramp away from the previous pose. The set is then ordered to minimise
   travel and checked for three-axis rotation spread before anything moves.
3. **The program.** Every pose: ramp, settle, then a burst of frames. A burst is
   accepted only if the tag's corners scatter by less than 0.35 px across it and
   the joints spread by less than 200 µrad - that is the evidence that the arm
   really stopped, which the hand-guided run never had. The corners are averaged
   and PnP is run once on the average. The forward kinematics of the measured
   joints are cross-checked against TF at every pose and the disagreement
   reported, which is what licenses using FK for the next pass.
4. **Offset pass.** Ramps through six of the poses twice, recording continuously
   while the arm moves, then fits the time shift that makes the camera's view of
   the tag agree with the robot's. This is the number that makes the calibration
   usable in motion.

Useful options - `--help` lists all of them:

| | |
|---|---|
| `--dry-run` | generate and save the program, move nothing |
| `--seed-result <result.json>` | skip the coarse pass; a repeat calibration is then one command |
| `--program <program.yaml>` | drive a stored program; skips the coarse pass and the generation entirely |
| `--poses N` | how many calibration poses (default 28) |
| `--no-offset-pass` | static calibration only |
| `--yes` | no prompts |

### What it writes

A run directory under `~/camera_calibration_runs/run_<stamp>/`:

| file | content |
|---|---|
| `result.json` | the calibration: `world -> camera_link`, the same transform for the optical frame, the estimated flange-to-tag offset, intrinsics, the measured delay, and quality figures |
| `poses.json` | per pose: both transforms, the burst's frame count and corner scatter, reprojection error, distance, view angle, FK-vs-TF disagreement |
| `program.yaml` | the pose program, replayable with `--program` |
| `offset_scan.json` | residual against assumed delay, the whole curve |

The same document also goes to stdout as `CALIBRATION_RESULT`, as the passive
recorder does. Nothing is published as TF and nothing is installed: the result
is a measurement, and wiring it into the description is a separate decision.

What a healthy run looks like: residual of a few millimetres and well under half
a degree, corner scatter around 0.1 px, flange-to-tag offset matching where the
tag physically sits, and FK-vs-TF disagreement near zero.

### Reading the delay

```json
"time_offset": {"offset_s": 0.0402, "uncertainty_s": 0.0009,
                "rms_at_offset_m": 0.0021, "rms_at_zero_m": 0.0045}
```

`offset_s` is what to **add to a camera timestamp** to reach the robot clock. A
positive value means images are stamped early, so looking up a robot pose at an
image's own stamp reads the robot's past - the failure mode the first run hit.
`rms_at_zero_m` is what ignoring the delay costs at the speed of that pass, and
`rms_at_offset_m` what is left after correcting it; the gap between them is the
delay's whole contribution.

The estimate is refused rather than guessed when the pass cannot support it: if
the tag moved slower than 20 mm/s in median, if the best fit sits at the edge of
the ±150 ms search range, or if the curvature of the residual leaves it worse
than ±10 ms.

`joint_state_rate` is 30 in every robot config here, which quantises the robot
side of the fit to 33 ms. The cubic interpolation absorbs most of that on a
smooth ramp, but raising the value in the robot config file tightens the
estimate; `replay.launch.py` reads it from that file and offers no override.

## Hand-guided recording (fallback)

This is the passive path. It sends no motion commands, so it needs only gravity
compensation - but read "Why the automated run exists" first: every sample it
takes is paired with a robot pose looked up at the image's timestamp, and the
two clocks do not agree.

Start the FR3/hand bringup with Franka's zero-effort gravity-compensation
controller, then run the recording entry point:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
  robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0 \
  gravity_compensation:=true

./apps/camera_calibration/calibrate.py \
  --tag-family tag36h11 --tag-id 0 --tag-size-m 0.040
```

Use `./apps/camera_calibration/calibrate.py --help` for every option.
The underlying `calibrate.launch.py` remains available for ROS launch
composition.

Defaults:

- image: `/camera/camera/color/image_raw`
- intrinsics: `/camera/camera/color/camera_info`
- moving pose: `world -> fr3_link8`
- calibrated pose: `world -> camera_link`

The tag is physically on the Inspire Hand, but the real-hardware description
currently publishes the hand as a separate TF tree. Because the hand is rigidly
mounted to the FR3, the recorder uses the moving `fr3_link8` pose and estimates
the complete unknown offset from that frame to the tag. This requires no mount
measurement and avoids treating the standalone `hand_base_link` as moving.

Move the hand across the image, vary its distance, and rotate it about at least
two axes. Aim for 20–40 distinct poses. The terminal logs every accepted
sample. Use `tests/test_april_tag.py` for an annotated view; the calibration node
itself publishes no images or transforms.

When enough samples have accumulated, solve:

```bash
ros2 service call /camera_calibration/solve std_srvs/srv/Trigger '{}'
```

On success the node writes a machine-readable `CALIBRATION_RESULT` JSON object
to its ROS log. It contains the `world -> camera_link` transform, the estimated
carrier-to-tag transform, full 4x4 matrices, and translation and
rotation RMSE. Nothing is published or saved.

To clear the samples and try again:

```bash
ros2 service call /camera_calibration/reset std_srvs/srv/Trigger '{}'
```

For deliberate one-pose-at-a-time collection, add `--manual` to the app
command, move to a pose, stop, and call:

```bash
ros2 service call /camera_calibration/capture std_srvs/srv/Trigger '{}'
```

## Common failure modes

Either entry point:

- `No world -> fr3_link8 TF`: the FR3's
  `robot_state_publisher` is not running, its prefix differs, or the camera and
  robot clocks disagree.
- `cannot look up camera-internal TF`: override `camera_mount_frame` or
  `camera_optical_frame` to match the D415's actual frame names.
- `not enough rotational excitation`: add clear roll, pitch, and yaw changes;
  translation alone cannot determine the unknown hand-to-tag offset.
- Large error: confirm the exact black-square size, avoid motion blur, keep the
  complete border visible, use more of the image, and keep the sticker flat.
- A flange-to-tag offset far larger than where the tag physically sits is the
  signature of a badly conditioned solve, not of a badly placed tag. Check it
  before trusting the camera pose.

The automated run:

- `no status from .../status - is the controller active?`: the replay controller
  is not up. `auto_calibrate.py` never loads it for you; bring up
  `franka_trajectory_replay/replay.launch.py` first.
- `only N of M poses were usable (...)`: the coarse camera pose is wrong, or the
  camera cannot see the workspace at the configured distances. The rejection
  counts in the message say which check did the discarding. Re-teach the seed
  poses, or widen `--distance-range-m`.
- `the arm had not settled`, either from corner scatter or joint spread: raise
  `--settle-seconds`, or look for a mechanical cause - a loose tag, a flexing
  mount, or a table the arm shakes.
- `only N frames carried a clean tag detection`: the tag left the image or the
  detection is marginal at that pose. A few of these per run are normal and are
  simply dropped; many mean the tag is too small for the configured distances.
- `the tag moved at a median X mm/s, below ...`: the offset pass waypoints were
  too close together. Raise `--offset-pass-poses` spacing by generating more
  poses, or accept `--no-offset-pass` and correct the delay another way.
- `the best delay sits at the edge of the +/-150 ms search range`: the clocks are
  further apart than that. Check whether the camera and the robot bringup are
  even using the same clock source before treating it as a latency.
