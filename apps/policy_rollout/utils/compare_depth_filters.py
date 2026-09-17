#!/usr/bin/env python3
"""Live D415 depth, side by side: raw, what the policy gets, and SDK-filtered.

    python3 apps/policy_rollout/utils/compare_depth_filters.py

Attaches to the running ``realsense2_camera`` node; it never opens the device,
so it can run next to a rollout or a recording. The window has four panels on
one colour scale:

    raw sensor depth           | SDK-filtered sensor depth
    (depth/image_rect_raw)     | (same frame through the librealsense filters)
    ---------------------------+----------------------------------------------
    current policy input       | SDK-filtered policy input
    (driver's aligned depth    | (filtered depth, aligned to colour by
     through prepare_rgbd)     |  librealsense, through prepare_rgbd)

The filters are librealsense's own processing blocks, fed the raw frames
through a software device and run in the order ``realsense2_camera`` runs them
(depth->disparity, spatial, temporal, hole filling, disparity->depth, then
align). They are host-side blocks, so the right-hand column is what the driver
would publish with the same ``*_filter.enable`` parameters. At start-up the
unfiltered software alignment of one frame is compared pixel for pixel with
the driver's aligned depth, and the result is printed and shown in the status
bar; a mismatch there means the right-hand column is not trustworthy.

Each panel reports:

- ``fill``: fraction of pixels with a depth;
- ``tnoise``: median per-pixel standard deviation over the last ``--window``
  frames, over pixels valid in all of them. Only meaningful for a static scene;
- ``roi``: RMS of a plane fit inside the rectangle dragged with the mouse. Drag
  it over something flat (the table). The rectangle is in each panel's own
  normalized coordinates, so it covers the same pixels within a row and nearly
  the same scene across rows (the depth and colour fields of view differ);
- ``dp3``: policy panels only, the pixels that survive DP3's depth range and
  3D crop box, i.e. the candidate points the encoder samples from.

Keys: ``s`` spatial, ``t`` temporal, ``h`` hole filling, ``d`` disparity
domain, ``x`` show filtered minus current (mm) instead of the filtered policy
panel, ``[`` / ``]`` far end of the colour scale, ``r`` clear the ROI, ``p``
save a snapshot (arrays + screenshot), ``q`` / Esc quit. Toggling a filter
restarts the temporal filter and the noise windows. The equivalent driver
launch arguments for the current toggles are shown in the status bar.

The status bar shows the median filter and align cost per frame. The driver
runs the same blocks in its frame callback, so that cost is also what enabling
the filters would add to the camera node. If processing falls behind the
camera, frames are dropped and counted; the temporal filter then sees a lower
rate than it would inside the driver. The spatial filter alone takes about
34 ms on a 1280x720 depth frame on this workcell, so the 1280x720 recording
mode cannot hold 30 Hz with it; the policy's 640x480 mode has a quarter of the
pixels.

The policy panels are exact (``prepare_rgbd``, the profile's crop) only when
the camera runs the profile's colour mode (640x480). In another mode they are
a centre crop to the policy aspect and a nearest resize, flagged in the status
bar.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import signal
import threading
import time

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.camera_calibration import (  # noqa: E402
    CameraIntrinsics,
    CropRectangle,
    load_camera_calibration,
    prepare_rgbd,
)


DEPTH_UNITS_M = 0.001
PANEL_SIZE = (640, 360)
STATUS_HEIGHT = 118


# ---------------------------------------------------------------------------
# Pure numpy pieces (tested without a camera or librealsense)
# ---------------------------------------------------------------------------


def fill_fraction(depth: np.ndarray) -> float:
    return float(np.count_nonzero(depth > 0)) / depth.size


def temporal_noise_mm(frames) -> float | None:
    """Median per-pixel standard deviation over pixels valid in every frame."""

    if len(frames) < 3:
        return None
    stack = np.stack(frames).astype(np.float32)
    valid = np.all(stack > 0, axis=0)
    if np.count_nonzero(valid) < 100:
        return None
    return float(np.median(stack[:, valid].std(axis=0)) * 1000.0)


def roi_slices(shape, roi):
    height, width = shape
    x0, y0, x1, y1 = roi
    return (
        slice(int(round(min(y0, y1) * height)), max(int(round(max(y0, y1) * height)), 1)),
        slice(int(round(min(x0, x1) * width)), max(int(round(max(x0, x1) * width)), 1)),
    )


def plane_rms_mm(depth_m: np.ndarray, roi) -> float | None:
    """RMS residual of a least-squares plane z = a*u + b*v + c inside ``roi``."""

    if roi is None:
        return None
    rows, cols = roi_slices(depth_m.shape, roi)
    patch = depth_m[rows, cols]
    v, u = np.nonzero(patch > 0)
    if u.size < 50:
        return None
    z = patch[v, u].astype(np.float64)
    design = np.column_stack((u, v, np.ones_like(u))).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ coefficients
    return float(np.sqrt(np.mean(residual**2)) * 1000.0)


def center_crop_for_aspect(width: int, height: int, aspect_w: int, aspect_h: int):
    if width * aspect_h == height * aspect_w:
        return None
    if width * aspect_h > height * aspect_w:
        crop_width = height * aspect_w // aspect_h
        return CropRectangle((width - crop_width) // 2, 0, crop_width, height)
    crop_height = width * aspect_h // aspect_w
    return CropRectangle(0, (height - crop_height) // 2, width, crop_height)


def nearest_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    rows = np.minimum(
        ((np.arange(height) + 0.5) * source_height / height).astype(int),
        source_height - 1,
    )
    columns = np.minimum(
        ((np.arange(width) + 0.5) * source_width / width).astype(int),
        source_width - 1,
    )
    return image[rows[:, None], columns[None, :]]


@dataclass
class PolicyView:
    """Maps an aligned full-colour-resolution depth onto the policy input."""

    profile: object
    color_intrinsics: CameraIntrinsics

    def __post_init__(self) -> None:
        self.policy_height, self.policy_width = self.profile.policy_shape
        shape = (self.color_intrinsics.height, self.color_intrinsics.width)
        try:
            crop = self.profile.crop_for_frame(shape)
            self.exact = True
        except ValueError:
            # Not a stream shape the checkpoint profile knows (e.g. the 1080p
            # recording mode): centre-crop to the policy aspect and resize the
            # same way, so the comparison still happens at policy resolution.
            crop = center_crop_for_aspect(
                shape[1], shape[0], self.policy_width, self.policy_height
            )
            self.exact = False
        self.crop = crop
        self.intrinsics = self.color_intrinsics.policy_view(
            crop, self.policy_width, self.policy_height
        )
        self._blank_rgb = np.zeros(shape + (3,), dtype=np.uint8)
        cloud = self.profile.dp3_point_cloud
        self.depth_min_m = float(cloud["depth_min_m"])
        self.depth_max_m = float(cloud["depth_max_m"])
        self.crop_min_m = np.asarray(cloud.get("crop_min_m", (-np.inf,) * 3), float)
        self.crop_max_m = np.asarray(cloud.get("crop_max_m", (np.inf,) * 3), float)
        k = self.intrinsics.camera_matrix
        v, u = np.meshgrid(
            np.arange(self.policy_height, dtype=np.float32),
            np.arange(self.policy_width, dtype=np.float32),
            indexing="ij",
        )
        self._x_per_z = (u - k[2]) / k[0]
        self._y_per_z = (v - k[5]) / k[4]

    def depth_m(self, aligned_mm: np.ndarray) -> np.ndarray:
        if self.exact:
            prepared = prepare_rgbd(
                self._blank_rgb, aligned_mm, self.profile, depth_units="millimetres"
            )
            return prepared.depth[0]
        image = aligned_mm if self.crop is None else self.crop.apply(aligned_mm)
        resized = nearest_resize(image, self.policy_height, self.policy_width)
        return resized.astype(np.float32) * DEPTH_UNITS_M

    def dp3_points(self, depth_m: np.ndarray) -> int:
        z = depth_m
        valid = (z > 0) & (z >= self.depth_min_m) & (z <= self.depth_max_m)
        xyz = np.stack((self._x_per_z * z, self._y_per_z * z, z), axis=-1)
        valid &= np.all(xyz >= self.crop_min_m, axis=-1)
        valid &= np.all(xyz <= self.crop_max_m, axis=-1)
        return int(np.count_nonzero(valid))


def driver_launch_arguments(state: dict) -> str:
    names = {
        "spatial": "spatial_filter.enable",
        "temporal": "temporal_filter.enable",
        "hole_filling": "hole_filling_filter.enable",
        "disparity": "disparity_filter.enable",
    }
    enabled = [f"{names[key]}:=true" for key in names if state[key]]
    return " ".join(enabled) if enabled else "(no filter arguments: the current setup)"


# ---------------------------------------------------------------------------
# librealsense software device
# ---------------------------------------------------------------------------


def _rs_intrinsics(rs, info, model):
    intrinsics = rs.intrinsics()
    intrinsics.width, intrinsics.height = int(info.width), int(info.height)
    intrinsics.fx, intrinsics.fy = float(info.k[0]), float(info.k[4])
    intrinsics.ppx, intrinsics.ppy = float(info.k[2]), float(info.k[5])
    intrinsics.model = model
    intrinsics.coeffs = [float(c) for c in (list(info.d) + [0.0] * 5)[:5]]
    return intrinsics


class SoftwareDepthPipeline:
    """Runs the driver's host-side depth filters and alignment on raw frames.

    The composite depth+colour frameset ``rs.align`` needs is built with a
    processing block rather than a syncer: a software device's syncer never
    matched the injected depth frames.
    """

    def __init__(self, depth_info, color_info, extrinsics, args):
        import pyrealsense2 as rs

        self.rs = rs
        self.depth_size = (int(depth_info.width), int(depth_info.height))
        self.color_size = (int(color_info.width), int(color_info.height))
        self.device = rs.software_device()
        self.depth_sensor = self.device.add_sensor("Depth")
        self.color_sensor = self.device.add_sensor("Color")

        depth_stream = rs.video_stream()
        depth_stream.type, depth_stream.index, depth_stream.uid = rs.stream.depth, 0, 0
        depth_stream.width, depth_stream.height = self.depth_size
        depth_stream.fps, depth_stream.bpp, depth_stream.fmt = 30, 2, rs.format.z16
        depth_stream.intrinsics = _rs_intrinsics(rs, depth_info, rs.distortion.brown_conrady)
        self.depth_profile = self.depth_sensor.add_video_stream(depth_stream)
        self.depth_sensor.add_read_only_option(rs.option.depth_units, DEPTH_UNITS_M)
        self.depth_sensor.add_read_only_option(
            rs.option.stereo_baseline, float(args.stereo_baseline_mm)
        )

        color_stream = rs.video_stream()
        color_stream.type, color_stream.index, color_stream.uid = rs.stream.color, 0, 1
        color_stream.width, color_stream.height = self.color_size
        color_stream.fps, color_stream.bpp, color_stream.fmt = 30, 3, rs.format.rgb8
        color_stream.intrinsics = _rs_intrinsics(
            rs, color_info, rs.distortion.inverse_brown_conrady
        )
        self.color_profile = self.color_sensor.add_video_stream(color_stream)

        transform = rs.extrinsics()
        transform.rotation = [float(v) for v in extrinsics.rotation]
        transform.translation = [float(v) for v in extrinsics.translation]
        self.depth_profile.register_extrinsics_to(self.color_profile, transform)

        self.depth_queue = rs.frame_queue(2, keep_frames=True)
        self.color_queue = rs.frame_queue(2, keep_frames=True)
        self.depth_sensor.open(self.depth_profile)
        self.color_sensor.open(self.color_profile)
        self.depth_sensor.start(self.depth_queue)
        self.color_sensor.start(self.color_queue)

        self._pair = {}
        self._frameset_queue = rs.frame_queue(2, keep_frames=True)
        self._composer = rs.processing_block(self._compose)
        self._composer.start(self._frameset_queue)
        self._blank_color = np.zeros((self.color_size[1], self.color_size[0], 3), np.uint8)
        self._frame_number = 0
        self.align = rs.align(rs.stream.color)

        self.timings_ms = {"filters": deque(maxlen=30), "align": deque(maxlen=30)}
        self.args = args
        self.to_disparity = rs.disparity_transform(True)
        self.from_disparity = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.hole_filling = rs.hole_filling_filter()
        self._set(self.spatial, rs.option.filter_magnitude, args.spatial_magnitude)
        self._set(self.spatial, rs.option.filter_smooth_alpha, args.spatial_alpha)
        self._set(self.spatial, rs.option.filter_smooth_delta, args.spatial_delta)
        self._set(self.spatial, rs.option.holes_fill, args.spatial_holes_fill)
        self._set(self.hole_filling, rs.option.holes_fill, args.hole_filling_mode)
        self.reset_temporal()

    def _set(self, block, option, value):
        if value is not None:
            block.set_option(option, float(value))

    def reset_temporal(self):
        rs = self.rs
        self.temporal = rs.temporal_filter()
        self._set(self.temporal, rs.option.filter_smooth_alpha, self.args.temporal_alpha)
        self._set(self.temporal, rs.option.filter_smooth_delta, self.args.temporal_delta)
        self._set(self.temporal, rs.option.holes_fill, self.args.temporal_persistence)

    def describe(self) -> dict:
        rs = self.rs

        def get(block, option):
            return block.get_option(option)

        return {
            "spatial": (
                f"mag {get(self.spatial, rs.option.filter_magnitude):.0f} "
                f"a {get(self.spatial, rs.option.filter_smooth_alpha):.2f} "
                f"d {get(self.spatial, rs.option.filter_smooth_delta):.0f} "
                f"holes {get(self.spatial, rs.option.holes_fill):.0f}"
            ),
            "temporal": (
                f"a {get(self.temporal, rs.option.filter_smooth_alpha):.2f} "
                f"d {get(self.temporal, rs.option.filter_smooth_delta):.0f} "
                f"persist {get(self.temporal, rs.option.holes_fill):.0f}"
            ),
            "hole_filling": f"mode {get(self.hole_filling, rs.option.holes_fill):.0f}",
            "disparity": f"baseline {self.args.stereo_baseline_mm:g} mm",
        }

    def _compose(self, _frame, source):
        frameset = source.allocate_composite_frame([self._pair["depth"], self._pair["color"]])
        source.frame_ready(frameset)

    def _inject(self, depth_mm: np.ndarray):
        rs = self.rs
        self._frame_number += 1
        timestamp = self._frame_number * 1000.0 / 30.0
        depth = np.ascontiguousarray(depth_mm, dtype=np.uint16)
        for sensor, profile, pixels, bpp in (
            (self.depth_sensor, self.depth_profile, depth, 2),
            (self.color_sensor, self.color_profile, self._blank_color, 3),
        ):
            frame = rs.software_video_frame()
            frame.pixels = pixels
            frame.bpp = bpp
            frame.stride = pixels.shape[1] * bpp
            frame.timestamp = timestamp
            frame.domain = rs.timestamp_domain.hardware_clock
            frame.frame_number = self._frame_number
            frame.profile = profile.as_video_stream_profile()
            if bpp == 2:
                frame.depth_units = DEPTH_UNITS_M
            sensor.on_video_frame(frame)
        return self.depth_queue.wait_for_frame(1000), self.color_queue.wait_for_frame(1000)

    def _aligned(self, depth_frame, color_frame) -> np.ndarray:
        self._pair["depth"], self._pair["color"] = depth_frame, color_frame
        self._composer.invoke(depth_frame)
        frameset = self._frameset_queue.wait_for_frame(1000).as_frameset()
        aligned = self.align.process(frameset).get_depth_frame()
        return np.asanyarray(aligned.get_data()).copy()

    def align_only(self, depth_mm: np.ndarray) -> np.ndarray:
        depth_frame, color_frame = self._inject(depth_mm)
        return self._aligned(depth_frame, color_frame)

    def process(self, depth_mm: np.ndarray, state: dict):
        """Return (filtered sensor-frame depth, filtered colour-aligned depth), mm."""

        started = time.perf_counter()
        depth_frame, color_frame = self._inject(depth_mm)
        frame = depth_frame
        if state["disparity"]:
            frame = self.to_disparity.process(frame)
        if state["spatial"]:
            frame = self.spatial.process(frame)
        if state["temporal"]:
            frame = self.temporal.process(frame)
        if state["hole_filling"]:
            frame = self.hole_filling.process(frame)
        if state["disparity"]:
            frame = self.from_disparity.process(frame)
        filtered = np.asanyarray(frame.get_data()).copy()
        filtered_at = time.perf_counter()
        aligned = self._aligned(frame, color_frame)
        self.timings_ms["filters"].append(1000.0 * (filtered_at - started))
        self.timings_ms["align"].append(1000.0 * (time.perf_counter() - filtered_at))
        return filtered, aligned

    def close(self):
        # Stopping the software sensors before interpreter teardown avoids a
        # std::terminate from librealsense's dispatcher threads.
        for sensor in (self.depth_sensor, self.color_sensor):
            try:
                sensor.stop()
                sensor.close()
            except RuntimeError:
                pass


# ---------------------------------------------------------------------------
# ROS side and processing thread
# ---------------------------------------------------------------------------


def _stamp_key(message):
    return (int(message.header.stamp.sec), int(message.header.stamp.nanosec))


def _depth_mm(message) -> np.ndarray:
    if str(message.encoding).lower() not in {"16uc1", "mono16"}:
        raise ValueError(f"expected 16UC1 depth, got {message.encoding!r}")
    rows = np.frombuffer(message.data, dtype=np.uint16).reshape(
        int(message.height), int(message.step) // 2
    )
    return np.ascontiguousarray(rows[:, : int(message.width)])


class RateMeter:
    def __init__(self):
        self._times = deque(maxlen=30)

    def tick(self):
        self._times.append(time.monotonic())

    @property
    def hz(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


@dataclass
class Result:
    stamp: tuple
    raw_mm: np.ndarray
    filtered_mm: np.ndarray
    current_policy_m: np.ndarray
    filtered_policy_m: np.ndarray
    current_aligned_mm: np.ndarray
    filtered_aligned_mm: np.ndarray


class Comparison:
    PANELS = ("raw", "filtered", "current", "filtered_policy")

    def __init__(self, args):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from realsense2_camera_msgs.msg import Extrinsics
        from sensor_msgs.msg import CameraInfo, Image

        self.args = args
        self.profile = load_camera_calibration(args.camera_calibration)
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.running = True
        self.state = {
            "spatial": not args.start_unfiltered,
            "temporal": not args.start_unfiltered,
            "hole_filling": False,
            "disparity": False,
        }
        self.state_version = 0
        self.raw_queue: deque = deque(maxlen=2)
        self.aligned_by_stamp: dict = {}
        self.infos: dict = {}
        self.dropped = 0
        self.unpaired = 0
        self.camera_rate = RateMeter()
        self.processed_rate = RateMeter()
        self.latest: Result | None = None
        self.windows = {name: deque(maxlen=args.window) for name in self.PANELS}
        self.self_check = "self-check: waiting for a frame"
        self.pipeline: SoftwareDepthPipeline | None = None
        self.policy_view: PolicyView | None = None
        self.error: str | None = None

        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node("depth_filter_comparison")
        ns = args.camera_namespace.rstrip("/")
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.node.create_subscription(
            Image, f"{ns}/depth/image_rect_raw", self._on_raw, qos_profile_sensor_data
        )
        self.node.create_subscription(
            Image,
            f"{ns}/aligned_depth_to_color/image_raw",
            self._on_aligned,
            qos_profile_sensor_data,
        )
        for key, topic in (("depth", "depth/camera_info"), ("color", "color/camera_info")):
            self.node.create_subscription(
                CameraInfo,
                f"{ns}/{topic}",
                lambda message, key=key: self._on_info(key, message),
                qos_profile_sensor_data,
            )
        self.node.create_subscription(
            Extrinsics,
            f"{ns}/extrinsics/depth_to_color",
            lambda message: self._on_info("extrinsics", message),
            latched,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.worker_thread = threading.Thread(target=self._work, daemon=True)

    def _spin(self):
        from rclpy.executors import ExternalShutdownException

        try:
            self.executor.spin()
        except ExternalShutdownException:  # Ctrl+C: rclpy shut the context down
            self.running = False

    # ROS callbacks ---------------------------------------------------------

    def _on_raw(self, message):
        with self.wake:
            if len(self.raw_queue) == self.raw_queue.maxlen:
                self.dropped += 1
            self.raw_queue.append(message)
            self.camera_rate.tick()
            self.wake.notify()

    def _on_aligned(self, message):
        with self.wake:
            self.aligned_by_stamp[_stamp_key(message)] = message
            while len(self.aligned_by_stamp) > 10:
                self.aligned_by_stamp.pop(next(iter(self.aligned_by_stamp)))
            self.wake.notify()

    def _on_info(self, key, message):
        with self.lock:
            self.infos[key] = message

    # Processing ------------------------------------------------------------

    def _wait_for_aligned(self, stamp, timeout_s=0.25):
        deadline = time.monotonic() + timeout_s
        with self.wake:
            while self.running and stamp not in self.aligned_by_stamp:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.wake.wait(remaining)
            return self.aligned_by_stamp.pop(stamp, None)

    def _setup(self):
        with self.lock:
            if not all(key in self.infos for key in ("depth", "color", "extrinsics")):
                return False
            depth_info = self.infos["depth"]
            color_info = self.infos["color"]
            extrinsics = self.infos["extrinsics"]
        self.pipeline = SoftwareDepthPipeline(depth_info, color_info, extrinsics, self.args)
        self.policy_view = PolicyView(
            self.profile,
            CameraIntrinsics(
                width=int(color_info.width),
                height=int(color_info.height),
                camera_matrix=tuple(float(v) for v in color_info.k),
            ),
        )
        mode = "exact prepare_rgbd" if self.policy_view.exact else (
            f"stream {color_info.width}x{color_info.height} is not the profile's "
            f"{self.profile.physical_stream_size[0]}x{self.profile.physical_stream_size[1]}: "
            "policy panels are a centre-crop + nearest resize, not the exact rollout crop"
        )
        print(
            f"depth {depth_info.width}x{depth_info.height}, colour "
            f"{color_info.width}x{color_info.height}; policy view {mode}",
            flush=True,
        )
        return True

    def _work(self):
        try:
            self._work_loop()
        except Exception as exc:  # surfaced in the window and on exit
            self.error = f"{type(exc).__name__}: {exc}"
            self.running = False
            raise

    def _work_loop(self):
        seen_version = -1
        checked = False
        while self.running:
            with self.wake:
                while self.running and not self.raw_queue:
                    self.wake.wait(0.5)
                if not self.running:
                    return
                message = self.raw_queue.popleft()
                state = dict(self.state)
                version = self.state_version
            if self.pipeline is None and not self._setup():
                continue
            if version != seen_version:
                self.pipeline.reset_temporal()
                with self.lock:
                    for window in self.windows.values():
                        window.clear()
                seen_version = version

            stamp = _stamp_key(message)
            raw_mm = _depth_mm(message)
            if not checked:
                aligned_message = self._wait_for_aligned(stamp, timeout_s=1.0)
                if aligned_message is None:
                    continue
                software = self.pipeline.align_only(raw_mm)
                driver = _depth_mm(aligned_message)
                same = float(np.mean(software == driver)) if software.shape == driver.shape else 0.0
                self.self_check = (
                    f"self-check: unfiltered software align == driver align on "
                    f"{100.0 * same:.3f}% of pixels"
                )
                print(self.self_check, flush=True)
                checked = True
                continue

            filtered_mm, filtered_aligned_mm = self.pipeline.process(raw_mm, state)
            aligned_message = self._wait_for_aligned(stamp)
            if aligned_message is None:
                self.unpaired += 1
                continue
            current_aligned_mm = _depth_mm(aligned_message)
            result = Result(
                stamp=stamp,
                raw_mm=raw_mm,
                filtered_mm=filtered_mm,
                current_policy_m=self.policy_view.depth_m(current_aligned_mm),
                filtered_policy_m=self.policy_view.depth_m(filtered_aligned_mm),
                current_aligned_mm=current_aligned_mm,
                filtered_aligned_mm=filtered_aligned_mm,
            )
            step = self.args.sensor_stride
            samples = {
                "raw": raw_mm[::step, ::step].astype(np.float32) * DEPTH_UNITS_M,
                "filtered": filtered_mm[::step, ::step].astype(np.float32) * DEPTH_UNITS_M,
                "current": result.current_policy_m,
                "filtered_policy": result.filtered_policy_m,
            }
            with self.lock:
                if version == self.state_version:
                    for name, sample in samples.items():
                        self.windows[name].append(sample)
                self.latest = result
                self.processed_rate.tick()

    # Window ----------------------------------------------------------------

    def toggle(self, key):
        with self.lock:
            self.state[key] = not self.state[key]
            self.state_version += 1

    def metrics(self, roi):
        with self.lock:
            windows = {name: list(window) for name, window in self.windows.items()}
        out = {}
        for name, frames in windows.items():
            if not frames:
                out[name] = {}
                continue
            latest = frames[-1]
            entry = {
                "fill": fill_fraction(latest),
                "tnoise": temporal_noise_mm(frames),
                "roi": plane_rms_mm(latest, roi),
            }
            if name in ("current", "filtered_policy") and self.policy_view is not None:
                entry["dp3"] = self.policy_view.dp3_points(latest)
            out[name] = entry
        return out

    def start(self):
        self.spin_thread.start()
        self.worker_thread.start()

    def stop(self):
        with self.wake:
            self.running = False
            self.wake.notify_all()
        self.worker_thread.join(timeout=2.0)
        self.executor.shutdown()
        self.spin_thread.join(timeout=2.0)
        if self.pipeline is not None:
            self.pipeline.close()
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def colorize(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    import cv2

    scaled = np.clip((depth_m - near) / max(far - near, 1e-6), 0.0, 1.0)
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[depth_m <= 0] = 0
    return image


def colorize_difference(difference_mm: np.ndarray, valid: np.ndarray, span_mm: float):
    import cv2

    scaled = np.clip(difference_mm / span_mm * 0.5 + 0.5, 0.0, 1.0)
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    image[~valid] = 0
    return image


def _format(entry: dict) -> str:
    if not entry:
        return "collecting..."
    parts = [f"fill {100 * entry['fill']:.1f}%"]
    parts.append("tnoise " + ("--" if entry["tnoise"] is None else f"{entry['tnoise']:.2f}mm"))
    if entry["roi"] is not None:
        parts.append(f"roi {entry['roi']:.2f}mm")
    if "dp3" in entry:
        parts.append(f"dp3 {entry['dp3']}")
    return "  ".join(parts)


def _label(canvas, text, origin, scale=0.5, color=(255, 255, 255)):
    import cv2

    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(canvas, (x - 3, y - height - 4), (x + width + 3, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def run_window(comparison: Comparison, args) -> int:
    import cv2

    title = "D415 depth: raw | SDK-filtered  /  current policy input | filtered policy input"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    pw, ph = PANEL_SIZE
    ui = {"roi": None, "drag": None, "far": args.far, "diff": False}

    def on_mouse(event, x, y, _flags, _param):
        if y >= 2 * ph:
            return
        column, row = min(x // pw, 1), min(y // ph, 1)
        nx = (x - column * pw) / pw
        ny = (y - row * ph) / ph
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["drag"] = (column, row, nx, ny)
        elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and ui["drag"] is not None:
            c0, r0, x0, y0 = ui["drag"]
            if (column, row) == (c0, r0):
                roi = (x0, y0, nx, ny)
                if abs(nx - x0) > 0.01 and abs(ny - y0) > 0.01:
                    ui["roi"] = roi
            if event == cv2.EVENT_LBUTTONUP:
                ui["drag"] = None

    cv2.setMouseCallback(title, on_mouse)
    output_root = Path(args.output_dir)
    period = 1.0 / args.display_hz
    last_saved = ""
    while comparison.running:
        started = time.monotonic()
        with comparison.lock:
            result = comparison.latest
            state = dict(comparison.state)
            dropped, unpaired = comparison.dropped, comparison.unpaired
        canvas = np.zeros((2 * ph + STATUS_HEIGHT, 2 * pw, 3), np.uint8)
        near, far = args.near, ui["far"]
        metrics = comparison.metrics(ui["roi"])
        if result is not None:
            bottom_right_title = "SDK-filtered policy input"
            if ui["diff"]:
                current, filtered = result.current_policy_m, result.filtered_policy_m
                valid = (current > 0) & (filtered > 0)
                bottom_right = colorize_difference(
                    (filtered - current) * 1000.0, valid, args.diff_span_mm
                )
                bottom_right_title = f"filtered - current (+/-{args.diff_span_mm:g} mm, blue=closer)"
            else:
                bottom_right = colorize(result.filtered_policy_m, near, far)
            panels = [
                (colorize(result.raw_mm * DEPTH_UNITS_M, near, far), "raw sensor depth", "raw"),
                (colorize(result.filtered_mm * DEPTH_UNITS_M, near, far), "SDK-filtered sensor depth", "filtered"),
                (colorize(result.current_policy_m, near, far), "current policy input", "current"),
                (bottom_right, bottom_right_title, "filtered_policy"),
            ]
            for index, (image, name, key) in enumerate(panels):
                x, y = (index % 2) * pw, (index // 2) * ph
                canvas[y : y + ph, x : x + pw] = cv2.resize(
                    image, PANEL_SIZE, interpolation=cv2.INTER_NEAREST
                )
                if ui["roi"] is not None:
                    x0, y0, x1, y1 = ui["roi"]
                    cv2.rectangle(
                        canvas,
                        (x + int(min(x0, x1) * pw), y + int(min(y0, y1) * ph)),
                        (x + int(max(x0, x1) * pw), y + int(max(y0, y1) * ph)),
                        (255, 255, 255),
                        1,
                    )
                _label(canvas, name, (x + 8, y + 20))
                _label(canvas, _format(metrics.get(key, {})), (x + 8, y + ph - 10), 0.45)
            cv2.line(canvas, (pw, 0), (pw, 2 * ph), (80, 80, 80), 1)
            cv2.line(canvas, (0, ph), (2 * pw, ph), (80, 80, 80), 1)
        else:
            _label(canvas, "waiting for camera topics...", (20, 40), 0.7)

        descriptions = comparison.pipeline.describe() if comparison.pipeline else {}
        timing = {"filters": "--", "align": "--"}
        if comparison.pipeline is not None:
            for name, values in list(comparison.pipeline.timings_ms.items()):
                if values:
                    timing[name] = f"{np.median(list(values)):.0f}"
        toggles = "  ".join(
            f"[{key}] {name} {'ON' if state[name] else 'off'}"
            + (f" ({descriptions[name]})" if state[name] and name in descriptions else "")
            for key, name in (("s", "spatial"), ("t", "temporal"), ("h", "hole_filling"), ("d", "disparity"))
        )
        view_note = ""
        if comparison.policy_view is not None and not comparison.policy_view.exact:
            view_note = "   |   policy panels approximate: stream is not the profile's 640x480"
        lines = [
            toggles,
            f"driver args: {driver_launch_arguments(state)}",
            (
                f"camera {comparison.camera_rate.hz:.1f} Hz  processed {comparison.processed_rate.hz:.1f} Hz  "
                f"dropped {dropped}  unpaired {unpaired}  filters {timing['filters']} ms  "
                f"align {timing['align']} ms  range {near:.2f}-{far:.2f} m"
            ),
            f"{comparison.self_check}{view_note}",
            "drag=ROI  r clear ROI  x diff view  [ ] range  p save  q quit   (tnoise needs a static scene)"
            + (f"   saved {last_saved}" if last_saved else ""),
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                canvas, line, (8, 2 * ph + 20 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (230, 230, 230), 1, cv2.LINE_AA,
            )
        cv2.imshow(title, canvas)

        key = cv2.waitKey(max(1, int((period - (time.monotonic() - started)) * 1000))) & 0xFF
        if key in (ord("q"), 27):
            break
        toggles_by_key = {"s": "spatial", "t": "temporal", "h": "hole_filling", "d": "disparity"}
        if chr(key) in toggles_by_key:
            comparison.toggle(toggles_by_key[chr(key)])
        elif key == ord("x"):
            ui["diff"] = not ui["diff"]
        elif key == ord("r"):
            ui["roi"] = None
        elif key == ord("["):
            ui["far"] = max(near + 0.1, ui["far"] - 0.1)
        elif key == ord("]"):
            ui["far"] += 0.1
        elif key == ord("p") and result is not None:
            last_saved = str(save_snapshot(output_root, canvas, result, state, metrics, ui["roi"], comparison))
            print(f"saved {last_saved}", flush=True)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return 0


def save_snapshot(root: Path, canvas, result: Result, state, metrics, roi, comparison) -> Path:
    import cv2
    import json

    directory = root / datetime.now().strftime("%Y%m%dT%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "comparison.png"), canvas)
    np.savez_compressed(
        directory / "depth.npz",
        raw_sensor_mm=result.raw_mm,
        filtered_sensor_mm=result.filtered_mm,
        current_aligned_mm=result.current_aligned_mm,
        filtered_aligned_mm=result.filtered_aligned_mm,
        current_policy_m=result.current_policy_m,
        filtered_policy_m=result.filtered_policy_m,
    )
    summary = {
        "stamp": {"sec": result.stamp[0], "nanosec": result.stamp[1]},
        "filters": state,
        "filter_settings": comparison.pipeline.describe() if comparison.pipeline else {},
        "driver_launch_arguments": driver_launch_arguments(state),
        "roi_normalized": roi,
        "window_frames": comparison.args.window,
        "sensor_metric_stride": comparison.args.sensor_stride,
        "policy_view_exact": bool(comparison.policy_view and comparison.policy_view.exact),
        "self_check": comparison.self_check,
        "metrics": metrics,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera-namespace", default="/camera/camera")
    parser.add_argument("--camera-calibration", default=None, help="policy camera profile YAML")
    parser.add_argument("--start-unfiltered", action="store_true", help="start with every filter off")
    parser.add_argument("--near", type=float, default=0.3, help="near end of the colour scale, m")
    parser.add_argument("--far", type=float, default=1.5, help="far end of the colour scale, m")
    parser.add_argument("--diff-span-mm", type=float, default=20.0)
    parser.add_argument("--window", type=int, default=15, help="frames in the temporal-noise window")
    parser.add_argument("--sensor-stride", type=int, default=4, help="subsampling of sensor panels for metrics")
    parser.add_argument("--display-hz", type=float, default=15.0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "depth_filter_comparison"),
        help="where p saves snapshots",
    )
    group = parser.add_argument_group("filter settings (default: librealsense defaults, as the driver uses)")
    group.add_argument("--spatial-magnitude", type=float)
    group.add_argument("--spatial-alpha", type=float)
    group.add_argument("--spatial-delta", type=float)
    group.add_argument("--spatial-holes-fill", type=float)
    group.add_argument("--temporal-alpha", type=float)
    group.add_argument("--temporal-delta", type=float)
    group.add_argument("--temporal-persistence", type=float)
    group.add_argument("--hole-filling-mode", type=float)
    group.add_argument(
        "--stereo-baseline-mm",
        type=float,
        default=55.0,
        help="baseline for the disparity transform; the D415's nominal 55 mm (not on any topic)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.window < 3 or args.sensor_stride < 1 or args.display_hz <= 0 or args.far <= args.near:
        raise SystemExit("need --window >= 3, --sensor-stride >= 1, --display-hz > 0, --far > --near")
    comparison = Comparison(args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(comparison, "running", False))
    comparison.start()
    try:
        return run_window(comparison, args)
    except KeyboardInterrupt:
        return 0
    finally:
        comparison.stop()
        if comparison.error:
            print(f"processing stopped: {comparison.error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
