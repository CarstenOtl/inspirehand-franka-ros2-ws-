# RealSense D415 eye-to-hand calibration

This app estimates a fixed table camera's pose in the Franka `world` frame.
An AprilTag is stuck rigidly to the back of the Inspire Hand. Its placement
does not need to be measured: the solver estimates both `world -> camera` and
the unknown carrier-to-tag transform from the motion.

The entry point is a passive recorder. It starts the camera and calibration
node, but never starts a controller or sends a robot/hand command. Start the
combined bringup with `gravity_compensation:=true`, then move the hand slowly
while keeping the complete tag visible.

## Before running

1. Print a supported AprilTag (default `tag36h11`, ID 0), keep it flat, and
   attach it to the back of the palm—not to a moving finger.
2. Measure the outer edge of the black square, excluding the white paper, in
   metres. Pose scale depends directly on this value.
3. Start the combined robot bringup so `world -> fr3_link8` is available.
   The camera and robot timestamps must use the same clock.
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

## Record the calibration

Start the FR3/hand bringup with Franka's zero-effort gravity-compensation
controller, then run the recording entry point:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
  robot_ip:=10.7.7.7 hand_port:=/dev/ttyUSB0 \
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

- `No world -> fr3_link8 TF`: the FR3's
  `robot_state_publisher` is not running, its prefix differs, or the camera and
  robot clocks disagree.
- `cannot look up camera-internal TF`: override `camera_mount_frame` or
  `camera_optical_frame` to match the D415's actual frame names.
- `not enough rotational excitation`: add clear roll, pitch, and yaw changes;
  translation alone cannot determine the unknown hand-to-tag offset.
- Large error: confirm the exact black-square size, avoid motion blur, keep the
  complete border visible, use more of the image, and keep the sticker flat.
