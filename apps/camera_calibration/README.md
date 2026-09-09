# RealSense D415 RGB eye-to-hand calibration

This app estimates a fixed table camera's pose in the Franka `world` frame.
An AprilTag is stuck rigidly to the back of the Inspire Hand. Its placement
is the measured fixed pose represented by the current robot asset. The solver
uses that known mount and estimates only `fr3_link0 -> camera`.

The entry point is a passive recorder. It starts the camera and calibration
node, but never starts a controller or sends a robot/hand command. Start the
combined bringup with `gravity_compensation:=true`, then move the hand slowly
while keeping the complete tag visible.

## Before running

1. Print a supported AprilTag (default `tag36h11`, ID 0), keep it flat, and
   attach it to the back of the palm—not to a moving finger.
2. Measure the outer edge of the black square, excluding the white paper, in
   metres. Pose scale depends directly on this value.
3. Start the combined robot bringup so `fr3_link0 -> fr3_link8` is available.
   The camera and robot timestamps must use the same clock.
4. Build and source the workspace (`rg2` in this workspace).

## Camera connection, ROS graph, and live views

The D415 is a USB 3 camera. Connect it directly to a USB 3 (or faster) host
port with a USB 3 cable; avoid hubs while bringing it up. On the host,
`lsusb -t` should show its link at `5000M` or higher. `480M` means USB 2,
which cannot reliably carry full-resolution RGB at 30 FPS. Confirm the D415
serial number with `rs-enumerate-devices` rather than selecting a generic UVC
webcam.

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

One `realsense2_camera` node must be the sole owner of the physical D415. The
calibration launch enables only the RGB channel and publishes it under
`/camera/camera/color`. The camera never writes footage by itself; ROS topics
are live in memory.

Start one camera producer (the D415 full-resolution RGB calibration profile):

```bash
ros2 launch realsense2_camera rs_launch.py \
  device_type:=d415 \
  camera_namespace:=camera camera_name:=camera \
  enable_color:=true enable_depth:=false \
  rgb_camera.color_profile:=1920x1080x30
```

Then inspect its actual output from a second sourced shell:

```bash
ros2 node list
ros2 topic list
ros2 topic hz /camera/camera/color/image_raw
ros2 run rqt_image_view rqt_image_view
```

In `rqt_image_view`, select `/camera/camera/color/image_raw`. This is the best
basic live-image check;
the Python viewers are diagnostic tools and may render more slowly than the
incoming ROS stream.

For an automatic side-by-side RGB and metric-depth view, use the workspace's
rqt plugin. It subscribes to both standard D415 topics without a dropdown:

```bash
ros2 run camera_calibration rgbd_view
```

Use `--camera-namespace`, `--camera-name`, or `--depth-max` when the camera does
not use the defaults. The same plugin is available from rqt's
**Visualization > RealSense RGB + Depth** menu.

To save actual footage for later replay, record a rosbag:

```bash
ros2 bag record -o d415_rgb_check \
  /camera/camera/color/image_raw \
  /camera/camera/color/camera_info
```

Run either viewer as the camera producer, or use `--no-launch` if the command
above is already running. Do not run `test_camera.py` and `test_april_tag.py`
without `--no-launch` at the same time: that starts two drivers against one USB
device and causes `Device or resource busy` / depth-stream failures. The
viewers now detect the usual `/camera/camera` collision and ask you to use
`--no-launch`.

If the D415 reports a stream-start failure after a crash, stop the active
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

## Record the calibration

Start the FR3/hand bringup with Franka's zero-effort gravity-compensation
controller, then run the recording entry point:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
  gravity_compensation:=true

./apps/camera_calibration/calibrate.py \
  --manual --tag-family tag36h11 --tag-id 0 --tag-size-m 0.040
```

Use `./apps/camera_calibration/calibrate.py --help` for every option.
The underlying `calibrate.launch.py` remains available for ROS launch
composition.

Defaults:

- image: `/camera/camera/color/image_raw`
- intrinsics: `/camera/camera/color/camera_info`
- moving pose: `fr3_link0 -> fr3_link8`
- calibrated pose: `fr3_link0 -> camera_link`

The tag is physically fixed on the Inspire Hand. The recorder uses timestamped
`fr3_link0 -> fr3_link8` FK generated from the arm's `/joint_states`, then applies
the measured fixed `fr3_link8 -> apriltag_0` transform from the current physical
asset, `assets/fr3_inspirehand/fr3_inspirehand.xml`. That transform includes the
180-degree Z clocking and 10 mm adapter between the flange and palm. The tag
mount is not estimated; only the fixed camera pose is calibrated.

`--manual` means hand-guided collection: move the hand across the image, vary
its distance and orientation, and keep the complete tag visible. At each pose,
hold the robot still and press Enter; the next valid OpenCV tag-pose/FK pair is
accepted and reported in the terminal. Wait for the `Accepted valid sample`
message before moving to the next pose. While waiting for Enter, the calibration
node leaves the image stream idle; pressing Enter arms detection and receives an
immediate service acknowledgement even if a camera frame is being processed.
Pressing Enter again while that capture is still armed cannot queue an extra
sample. The wrapper establishes one persistent ROS service connection before
showing the first pose prompt and reuses it for the entire run, so Enter does not
incur repeated DDS discovery. Aim for 20–40 distinct poses. Use
`tests/test_april_tag.py` for an annotated view; the calibration node itself
publishes no images or transforms.

Each run creates `logs/<UTC timestamp>/`. Every accepted sample writes an
annotated `sample_NNN.png` and a matching `sample_NNN.json` containing the
image timestamp, camera-to-tag pose, timestamped world-to-hand TF, fixed
hand-to-tag transform, and that sample's camera-pose candidate. The completed
solve is written to `calibration_result.json` in the same directory. Use
`--output-root PATH` to choose a different parent directory.

### Automatic 12-waypoint mode

Auto mode requires the real FR3 trajectory controller. Start the MoveIt FR3
bringup instead of the gravity-compensation bringup (never start both against
the same arm):

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 load_gripper:=false ee_id:=none use_rviz:=false

./apps/camera_calibration/calibrate.py --auto
```

The app displays a physical-motion warning and sends no command until Enter is
pressed. It then visits 12 slow, joint-limited poses that keep the complete tag
inside the simulated camera view, waits for each trajectory goal to succeed, and
accepts exactly one settled RGB/FK sample at each pose. Keep the real workcell
clear and remain at the emergency stop: the programmed joint trajectory cannot
model untracked objects in the real workspace. Calibration solves automatically
after the twelfth sample.

As soon as the configured minimum number of valid samples has accumulated, the
node solves automatically and prints the `CALIBRATION_RESULT`. No second command
is required. The solve service remains available if an explicit re-solve is
needed:

```bash
ros2 service call /camera_calibration/solve std_srvs/srv/Trigger '{}'
```

On success the node writes a machine-readable `CALIBRATION_RESULT` JSON object
to its ROS log. It contains the `fr3_link0 -> camera_link` mount transform, the
exact `fr3_link0 -> camera_color_optical_frame` pose used by the image solver, the
camera intrinsics/image size, the fixed carrier-to-tag transform, full 4x4
matrices, and translation and rotation RMSE. The same object is saved as
`calibration_result.json`; the transform is not automatically published.

### Confirm the calibrated camera in MuJoCo

Capture the calibration output with `tee` (or copy just the JSON object after
`CALIBRATION_RESULT` into a file), then open the read-only test viewer:

```bash
./apps/camera_calibration/calibrate.py --manual 2>&1 | tee calibration.log

./apps/camera_calibration/tests/visualize_calibrated_camera.py calibration.log --live
```

The test uses the same combined FR3/Inspire MJCF as the calibration simulation.
It adds a non-colliding camera housing, RGB optical axes, and a cyan view
frustum at the calibrated pose. With the robot bringup and one RealSense RGB
producer still running, the live view compares the real image with the MuJoCo
camera while updating the simulated arm from `/joint_states`. It subscribes
only: it does not publish commands or step physics.

- Press `C` or `2` for the calibrated RGB point of view.
- Press `F` or `1` to return to the free overview and inspect the camera's
  location and viewing direction.
- Use `--start-in-pov` to open directly in the RGB view.
- Use `--headless` to validate that the result and decorated MJCF compile
  without opening a window.

This opens one side-by-side view: the latest real RGB frame on the left and a
MuJoCo render through the calibrated camera on the right. The MuJoCo arm is
updated from `fr3_joint1` through `fr3_joint7` on `/joint_states`; it is purely
kinematic, creates no application publisher, and never sends robot commands.
The viewer only subscribes to the existing camera producer, so it cannot
contend for the D415 USB device. Use `--joint-state-topic` or `--image-topic`
for non-default ROS names, and `--render-width` or `--max-fps` to tune display
cost. Press `Q` or Esc to close it.

The two panels use the most recently received image and joint state rather than
hardware-triggered synchronization. Their title bars show message age so stale
input is visible. The simulated view intentionally contains only geometry from
the MuJoCo scene; unmodelled real workcell objects will appear only on the left.

Omit `--live` to use the offline passive MuJoCo viewer instead. That mode does
not start ROS; press `C` or `2` for the calibrated camera and `F` or `1` for the
external overview.

The POV uses the camera's vertical focal length and image height; keep the
MuJoCo window at the printed RGB aspect ratio for matching horizontal coverage.
The passive interactive viewport does not reproduce lens distortion or a small
off-centre principal point, so compare scene coverage and orientation rather
than distorted edge pixels. Calibration files created before the optical
transform was added must be regenerated; the mount pose alone is insufficient
to reconstruct the exact RGB viewpoint.

To clear the samples and try again:

```bash
ros2 service call /camera_calibration/reset std_srvs/srv/Trigger '{}'
```

For deliberate service-triggered one-pose-at-a-time collection, use
`--triggered-capture`, move to a pose, stop, and call:

```bash
ros2 service call /camera_calibration/capture std_srvs/srv/Trigger '{}'
```

## Common failure modes

- `No fr3_link0 -> fr3_link8 TF`: the FR3's
  `robot_state_publisher` is not running, its prefix differs, or the camera and
  robot clocks disagree.
- `cannot look up camera-internal TF`: override `camera_mount_frame` or
  `camera_optical_frame` to match the D415's actual frame names.
- Large error: confirm the exact black-square size, avoid motion blur, keep the
  complete border visible, use more of the image, and keep the sticker flat.
