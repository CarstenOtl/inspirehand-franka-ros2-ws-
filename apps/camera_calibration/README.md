# RealSense D415 eye-to-hand calibration

This app estimates a fixed table camera's pose in the Franka `world` frame.
An AprilTag is stuck rigidly to the back of the Inspire Hand. Its placement
does not need to be measured: the solver estimates both `world -> camera` and
the unknown carrier-to-tag transform from the motion.

The entry point is a passive recorder. It starts the camera and calibration
node, but never starts a controller or sends a robot/hand command. Put the FR3
into its built-in hand-guiding mode and move the hand slowly while keeping the
complete tag visible.

## Before running

1. Print a supported AprilTag (default `tag36h11`, ID 0), keep it flat, and
   attach it to the back of the palm—not to a moving finger.
2. Measure the outer edge of the black square, excluding the white paper, in
   metres. Pose scale depends directly on this value.
3. Start the combined robot bringup so `world -> fr3_link8` is available.
   The camera and robot timestamps must use the same clock.
4. Build and source the workspace (`rg2` in this workspace).

The hardware viewers are useful preflight checks. Close each before starting
calibration because each launches its own camera by default:

```bash
./apps/camera_calibration/tests/test_camera.py
./apps/camera_calibration/tests/test_april_tag.py \
  --tag-family tag36h11 --tag-id 0 --tag-size 0.040
```

## Record the calibration

Start the normal FR3/hand bringup separately, enable the FR3's hand-guiding
feature, then run the recording entry point:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py \
  robot_ip:=10.7.7.7 hand_port:=/dev/ttyUSB0

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
