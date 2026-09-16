"""Record the D415's full-resolution RGB-D stream across a trajectory rollout.

    ros2 launch inspire_franka_trajectory_replay rgbd_camera.launch.py
    ros2 run inspire_franka_trajectory_replay replay_trajectory <trajectory> \\
        --record-rgbd --note "threading take 3"

The camera is launched separately, at the D415's largest colour mode, and the
run refuses to start unless the stream it finds on the graph is actually at
that resolution: only one ``realsense2_camera`` node can own the device, and
the one the policy rollout uses runs at 640x480, so a run that merely recorded
whatever was there would silently produce a quarter-resolution take.

What is recorded
----------------
One rosbag per run, beside the run's other output:

- ``<rgbd topic>`` -- ``realsense2_camera_msgs/RGBD``: colour and the depth
  aligned to it, from one RealSense frameset, at the colour resolution.
  Recorded as the composite rather than as two image topics so that a frame
  pair can never be mismatched afterwards.
- ``<prefix>/color/camera_info`` -- the intrinsics, so the depth can be
  unprojected without consulting anything outside the bag.
- the arm's and hand's joint states, the hand command channel, and ``/tf`` /
  ``/tf_static``, so every frame can be placed against the robot without
  re-deriving anything from the trajectory artifact.

Recording goes through :class:`franka_trajectory_replay.recording.BagRecorder`,
i.e. the ``ros2 bag record`` CLI, not a Python subscriber. At 1920x1080 one
RGBD message is about 10 MB and thirty of them arrive a second; an rclpy
callback writing PNGs would fall behind and drop frames silently, which is the
one failure a recording must not have. MCAP is the default storage because it
appends large messages sequentially; sqlite3 is kept as an option.

What a preflight proves
-----------------------
Before anything moves: that a message arrives on the RGBD topic and on the
camera-info topic, that the colour image is exactly the required size, that the
depth is aligned to it (same size, same frame), that both encodings are ones
the extractor can decode, and the rate the stream is actually delivered at.
Resolution is a refusal; a low rate is reported and written into the manifest,
because a USB 2 link or a CPU-bound aligner shows up there first.

Output
------
``<run directory>/rgbd_bag/`` and ``rgbd_manifest.json``. Turn the bag into
per-frame PNGs with ``extract_rgbd <run directory>``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from franka_trajectory_replay.recording import BagRecorder, read_messages, stamp_to_ns

from .capture import RecordedTopic, _atomic_json

SCHEMA_VERSION = 1

#: The D415's largest colour mode. Depth tops out at 1280x720, but the RGBD
#: message carries depth *aligned to colour*, so a full-resolution take is
#: 1920x1080 for both images.
D415_FULL_COLOR: Tuple[int, int] = (1920, 1080)
D415_FULL_DEPTH: Tuple[int, int] = (1280, 720)
DEFAULT_RATE_HZ = 30.0
DEFAULT_RGBD_TOPIC = "/camera/camera/rgbd"
DEFAULT_STORAGE_ID = "mcap"
STORAGE_IDS = ("mcap", "sqlite3")

#: Where a run lands when it is not also an intervention session.
DEFAULT_OUTPUT_ROOT = "logs/replay_rollout"
BAG_DIRECTORY = "rgbd_bag"
MANIFEST_NAME = "rgbd_manifest.json"
FRAMES_DIRECTORY = "rgbd_frames"

#: rosbag2 holds messages in a cache before its writer thread flushes them. The
#: default is 100 MB, which at ~310 MB/s (1920x1080 colour + aligned depth at
#: 30 Hz) is a third of a second of slack before a slow flush drops frames. A
#: gigabyte is several seconds, which is what a disk hiccup costs.
CACHE_BYTES = 1 << 30

RGB_ENCODINGS = frozenset({"rgb8", "bgr8", "rgba8", "bgra8"})
DEPTH_ENCODINGS = frozenset({"16uc1", "mono16", "32fc1"})


def parse_resolution(text: str) -> Tuple[int, int]:
    """``"1920x1080"`` -> ``(1920, 1080)``; anything else is a ``ValueError``."""
    parts = str(text).lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"resolution must look like 1920x1080, not {text!r}")
    try:
        width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"resolution must look like 1920x1080, not {text!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive, not {text!r}")
    return width, height


def camera_info_topic_for(rgbd_topic: str) -> str:
    """The colour intrinsics next to an RGBD topic, by the RealSense node's layout."""
    prefix = rgbd_topic.rstrip("/").rsplit("/", 1)[0]
    return f"{prefix}/color/camera_info"


@dataclass(frozen=True)
class StreamSpec:
    """What the run requires of the camera stream."""

    rgbd_topic: str = DEFAULT_RGBD_TOPIC
    width: int = D415_FULL_COLOR[0]
    height: int = D415_FULL_COLOR[1]
    rate_hz: float = DEFAULT_RATE_HZ

    @property
    def camera_info_topic(self) -> str:
        return camera_info_topic_for(self.rgbd_topic)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def recorded_topics(
    spec: StreamSpec,
    arm: bool = True,
    hand: bool = True,
    hand_topic: str = "/inspire_hand/command",
    hand_state_topic: str = "/inspire_hand/joint_states",
) -> List[RecordedTopic]:
    """The bag's topics for a run, and how each one is proven before recording.

    The camera topics lead. The robot channels follow the run's shape: a
    hand-only run has no arm bringup and so no ``/joint_states``; an arm-only
    run has no hand driver to be subscribed to its command channel. Requiring
    either where it cannot exist would refuse a run for the wrong reason.
    """
    topics = [
        RecordedTopic(
            spec.rgbd_topic,
            "live",
            f"colour and colour-aligned depth from one RealSense frameset at "
            f"{spec.resolution}; the composite, so a pair is never mismatched",
            "realsense2_camera_msgs/msg/RGBD",
        ),
        RecordedTopic(
            spec.camera_info_topic,
            "live",
            "the colour intrinsics, so the depth can be unprojected from the bag alone",
            "sensor_msgs/msg/CameraInfo",
        ),
    ]
    if arm:
        topics.append(
            RecordedTopic(
                "/joint_states",
                "live",
                "the 30 Hz merged joint view every other tool in this workspace reads",
                "sensor_msgs/msg/JointState",
            )
        )
    if hand:
        topics.extend(
            [
                RecordedTopic(
                    hand_state_topic,
                    "live",
                    "the measured hand joints in radians",
                    "sensor_msgs/msg/JointState",
                ),
                RecordedTopic(
                    hand_topic,
                    "subscriber",
                    "the hand action channel, so the bag shows what was commanded and when",
                    "sensor_msgs/msg/JointState",
                ),
            ]
        )
    topics.extend(
        [
            RecordedTopic(
                "/tf",
                "publisher",
                "frames over time, so each image can be placed against the robot",
                "tf2_msgs/msg/TFMessage",
            ),
            RecordedTopic(
                "/tf_static",
                "publisher",
                "the fixed frames, including the calibrated camera pose when it is published",
                "tf2_msgs/msg/TFMessage",
            ),
        ]
    )
    return topics


# --- the preflight --------------------------------------------------------------------


@dataclass
class StreamReport:
    """What the live stream looked like, and everything wrong with it."""

    rgbd_topic: str
    camera_info_topic: str
    required_width: int
    required_height: int
    rgb_width: Optional[int] = None
    rgb_height: Optional[int] = None
    rgb_encoding: Optional[str] = None
    rgb_frame_id: Optional[str] = None
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    depth_encoding: Optional[str] = None
    depth_frame_id: Optional[str] = None
    info_width: Optional[int] = None
    info_height: Optional[int] = None
    info_frame_id: Optional[str] = None
    expected_rate_hz: float = DEFAULT_RATE_HZ
    measured_rate_hz: Optional[float] = None
    frames_seen: int = 0
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> Dict[str, object]:
        document = asdict(self)
        document["ok"] = self.ok
        return document


def check_stream(report: StreamReport, rgbd, camera_info) -> StreamReport:
    """Fill ``report`` from one RGBD message and one CameraInfo; list every problem.

    Pure: the messages are read for their fields only, so the checks are
    testable with stand-ins. Every problem is listed rather than the first one,
    because a camera launched wrongly usually gets several things wrong at once
    and the operator should fix the launch, not iterate on refusals.
    """
    rgb, depth = rgbd.rgb, rgbd.depth
    report.rgb_width, report.rgb_height = int(rgb.width), int(rgb.height)
    report.rgb_encoding = str(rgb.encoding)
    report.rgb_frame_id = str(rgb.header.frame_id)
    report.depth_width, report.depth_height = int(depth.width), int(depth.height)
    report.depth_encoding = str(depth.encoding)
    report.depth_frame_id = str(depth.header.frame_id)
    report.info_width, report.info_height = int(camera_info.width), int(camera_info.height)
    report.info_frame_id = str(camera_info.header.frame_id)

    required = (report.required_width, report.required_height)
    if (report.rgb_width, report.rgb_height) != required:
        report.problems.append(
            f"colour is {report.rgb_width}x{report.rgb_height}, not the required "
            f"{required[0]}x{required[1]}; relaunch the camera at that colour profile"
        )
    if (report.depth_width, report.depth_height) != (report.rgb_width, report.rgb_height):
        report.problems.append(
            f"depth is {report.depth_width}x{report.depth_height} while colour is "
            f"{report.rgb_width}x{report.rgb_height}: the depth is not aligned to colour "
            "(align_depth.enable:=true)"
        )
    if report.depth_frame_id != report.rgb_frame_id:
        report.problems.append(
            f"depth frame {report.depth_frame_id!r} is not the colour frame "
            f"{report.rgb_frame_id!r}: the depth is not aligned to colour"
        )
    if (report.info_width, report.info_height) != (report.rgb_width, report.rgb_height):
        report.problems.append(
            f"CameraInfo is {report.info_width}x{report.info_height} while the colour "
            f"image is {report.rgb_width}x{report.rgb_height}; the intrinsics do not "
            "describe the recorded image"
        )
    if report.info_frame_id != report.rgb_frame_id:
        report.problems.append(
            f"CameraInfo frame {report.info_frame_id!r} is not the colour frame "
            f"{report.rgb_frame_id!r}"
        )
    if report.rgb_encoding.lower() not in RGB_ENCODINGS:
        report.problems.append(
            f"colour encoding {report.rgb_encoding!r} is not one the extractor decodes "
            f"({', '.join(sorted(RGB_ENCODINGS))})"
        )
    if report.depth_encoding.lower() not in DEPTH_ENCODINGS:
        report.problems.append(
            f"depth encoding {report.depth_encoding!r} is not one the extractor decodes "
            f"({', '.join(sorted(DEPTH_ENCODINGS))})"
        )
    return report


def note_rate(report: StreamReport, frames: int, seconds: float) -> StreamReport:
    """Record the delivered rate; well under the expected one is a warning, not a refusal."""
    report.frames_seen = int(frames)
    if seconds > 0:
        report.measured_rate_hz = frames / seconds
        if report.expected_rate_hz > 0 and report.measured_rate_hz < 0.8 * report.expected_rate_hz:
            report.warnings.append(
                f"the stream is delivered at {report.measured_rate_hz:.1f} Hz, under the "
                f"expected {report.expected_rate_hz:g} Hz: check the USB 3 link and the "
                "camera node's CPU load (aligning depth at full resolution is not free)"
            )
    return report


def inspect_stream(node, spec: StreamSpec, timeout: float, window: float = 1.0) -> StreamReport:
    """Subscribe to the stream, check the first frame pair, and measure the rate.

    ``node`` must not be spinning in an executor: the wait is driven with
    ``rclpy.spin_once`` here, exactly as the capture preflight does. The
    subscriptions are destroyed before returning.
    """
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from realsense2_camera_msgs.msg import RGBD
    from sensor_msgs.msg import CameraInfo

    report = StreamReport(
        spec.rgbd_topic, spec.camera_info_topic, spec.width, spec.height,
        expected_rate_hz=spec.rate_hz,
    )
    first = {"rgbd": None, "info": None}
    count = {"rgbd": 0}
    arrived = threading.Event()

    def on_rgbd(message):
        count["rgbd"] += 1
        if first["rgbd"] is None:
            first["rgbd"] = message
        if first["info"] is not None:
            arrived.set()

    def on_info(message):
        if first["info"] is None:
            first["info"] = message
        if first["rgbd"] is not None:
            arrived.set()

    subscriptions = [
        node.create_subscription(RGBD, spec.rgbd_topic, on_rgbd, qos_profile_sensor_data),
        node.create_subscription(
            CameraInfo, spec.camera_info_topic, on_info, qos_profile_sensor_data
        ),
    ]
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not arrived.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        if first["rgbd"] is None:
            report.problems.append(
                f"no RGBD message on {spec.rgbd_topic} within {timeout:g} s; is the camera "
                "launched with enable_rgbd:=true, and is this the topic it publishes on?"
            )
        if first["info"] is None:
            report.problems.append(
                f"no CameraInfo on {spec.camera_info_topic} within {timeout:g} s"
            )
        if report.problems:
            return report
        check_stream(report, first["rgbd"], first["info"])
        # The rate is measured after the first frame, over a fixed window, so a
        # slow start does not read as a slow stream.
        start_count = count["rgbd"]
        started = time.monotonic()
        while time.monotonic() - started < window:
            rclpy.spin_once(node, timeout_sec=0.05)
        note_rate(report, count["rgbd"] - start_count, time.monotonic() - started)
        return report
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)


def format_stream_report(report: StreamReport) -> str:
    lines = [f"RGB-D stream preflight on {report.rgbd_topic}:"]
    if report.rgb_width is not None:
        lines.append(
            f"  colour  {report.rgb_width}x{report.rgb_height} {report.rgb_encoding} "
            f"in {report.rgb_frame_id}"
        )
        lines.append(
            f"  depth   {report.depth_width}x{report.depth_height} {report.depth_encoding} "
            f"in {report.depth_frame_id}"
        )
        lines.append(
            f"  intrinsics {report.info_width}x{report.info_height} in {report.info_frame_id} "
            f"({report.camera_info_topic})"
        )
    if report.measured_rate_hz is not None:
        lines.append(
            f"  rate    {report.measured_rate_hz:.1f} Hz measured over {report.frames_seen} "
            f"frames (expected {report.expected_rate_hz:g} Hz)"
        )
    status = "OK" if report.ok else "REFUSED"
    lines.append(f"  required {report.required_width}x{report.required_height}: {status}")
    for problem in report.problems:
        lines.append(f"  ! {problem}")
    for warning in report.warnings:
        lines.append(f"  ~ {warning}")
    return "\n".join(lines)


# --- the recording --------------------------------------------------------------------


def bag_message_counts(bag_dir) -> Dict[str, int]:
    """Per-topic message counts from the bag's own ``metadata.yaml``; empty if unreadable.

    This is what says whether the take is complete. The preflight measures the
    rate the stream is *delivered* at; the recorder can still lose a 10 MB
    message on the transport, and rosbag2 only says so in its log. The count in
    the metadata is the number of frames actually on disk.
    """
    try:
        document = yaml.safe_load((Path(bag_dir) / "metadata.yaml").read_text(encoding="utf-8"))
        entries = document["rosbag2_bagfile_information"]["topics_with_message_count"]
        return {
            str(entry["topic_metadata"]["name"]): int(entry["message_count"]) for entry in entries
        }
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return {}


def describe_take(counts: Dict[str, int], spec: StreamSpec, duration_s: Optional[float]) -> str:
    """One line on how many frames the bag holds against how many the run lasted for."""
    frames = counts.get(spec.rgbd_topic)
    if frames is None:
        return f"no {spec.rgbd_topic} messages counted in the bag metadata"
    if not duration_s or duration_s <= 0:
        return f"{frames} RGB-D frames recorded"
    expected = spec.rate_hz * duration_s
    text = f"{frames} RGB-D frames over {duration_s:.1f} s ({frames / duration_s:.1f} Hz"
    if spec.rate_hz > 0:
        text += f"; {100.0 * frames / expected:.0f}% of {spec.rate_hz:g} Hz"
    return text + ")"


class RgbdRecording:
    """One run's RGB-D bag: start it, stop it, and describe it in a manifest."""

    def __init__(
        self,
        run_directory,
        spec: StreamSpec,
        topics: Sequence[RecordedTopic],
        storage_id: str = DEFAULT_STORAGE_ID,
        cache_bytes: int = CACHE_BYTES,
    ) -> None:
        if storage_id not in STORAGE_IDS:
            raise ValueError(f"storage must be one of {STORAGE_IDS}, not {storage_id!r}")
        self.run_directory = Path(run_directory)
        self.bag_dir = self.run_directory / BAG_DIRECTORY
        self.spec = spec
        self.topics = list(topics)
        self.storage_id = storage_id
        self.cache_bytes = int(cache_bytes)
        self.started_at: Optional[str] = None
        self.stopped_at: Optional[str] = None
        self._started_monotonic: Optional[float] = None
        self.duration_s: Optional[float] = None
        self.message_counts: Dict[str, int] = {}
        self._recorder: Optional[BagRecorder] = None

    def record_command_arguments(self) -> List[str]:
        return ["--max-cache-size", str(self.cache_bytes)]

    @property
    def recording(self) -> bool:
        return self._recorder is not None and self.stopped_at is None

    def start(self, timeout: float = 20.0) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._recorder = BagRecorder(
            self.bag_dir,
            [entry.topic for entry in self.topics],
            self.storage_id,
            extra_args=self.record_command_arguments(),
        )
        self._recorder.start(timeout=timeout)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._started_monotonic = time.monotonic()

    def stop(self) -> None:
        if self._recorder is None or self.stopped_at is not None:
            return
        self._recorder.stop()
        self.stopped_at = datetime.now(timezone.utc).isoformat()
        if self._started_monotonic is not None:
            self.duration_s = time.monotonic() - self._started_monotonic
        self.message_counts = bag_message_counts(self.bag_dir)

    def describe(self) -> str:
        return describe_take(self.message_counts, self.spec, self.duration_s)

    def write_manifest(
        self, report: Optional[StreamReport], extra: Optional[Dict[str, object]] = None
    ) -> Path:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "kind": "rollout_rgbd",
            "tool": "replay_trajectory --record-rgbd",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "bag_dir": BAG_DIRECTORY,
            "bag_storage_id": self.storage_id,
            "recorded": self.started_at is not None,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "duration_s": self.duration_s,
            "bag_message_counts": dict(self.message_counts),
            "rgbd_frames": self.message_counts.get(self.spec.rgbd_topic),
            "rgbd_topic": self.spec.rgbd_topic,
            "camera_info_topic": self.spec.camera_info_topic,
            "required_resolution": {"width": self.spec.width, "height": self.spec.height},
            "expected_rate_hz": self.spec.rate_hz,
            "stream": None if report is None else report.as_dict(),
            "topics": [
                {"topic": entry.topic, "check": entry.check, "why": entry.why}
                for entry in self.topics
            ],
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            **(extra or {}),
        }
        path = self.run_directory / MANIFEST_NAME
        _atomic_json(path, document)
        return path


# --- reading it back -------------------------------------------------------------------


def image_to_numpy(message) -> Tuple[np.ndarray, str]:
    """Decode a RealSense ``Image`` without cv_bridge: ``(array, units)``.

    Colour comes back as RGB ``uint8`` HxWx3; depth as ``uint16`` millimetres or
    ``float32`` metres, as published. The same encodings the preflight admits.
    """
    encoding = str(message.encoding).lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    raw = memoryview(message.data)
    if encoding in RGB_ENCODINGS:
        channels = 4 if "a" in encoding else 3
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)[..., :3]
        if encoding.startswith("bgr"):
            image = image[..., ::-1]
        return np.ascontiguousarray(image), "rgb"
    if encoding in {"16uc1", "mono16"}:
        rows = np.frombuffer(raw, dtype=np.uint16).reshape(height, step // 2)
        return np.ascontiguousarray(rows[:, :width]), "millimetres"
    if encoding == "32fc1":
        rows = np.frombuffer(raw, dtype=np.float32).reshape(height, step // 4)
        return np.ascontiguousarray(rows[:, :width]), "metres"
    raise ValueError(f"unsupported image encoding {message.encoding!r}")


def depth_to_millimetres(depth: np.ndarray, units: str) -> np.ndarray:
    """Depth as ``uint16`` millimetres, the RealSense's native unit, for a lossless PNG."""
    if units == "millimetres":
        return depth.astype(np.uint16, copy=False)
    if units == "metres":
        metres = np.nan_to_num(depth.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(np.rint(metres * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    raise ValueError(f"unknown depth units {units!r}")


def camera_info_document(info) -> Dict[str, object]:
    return {
        "frame_id": str(info.header.frame_id),
        "width": int(info.width),
        "height": int(info.height),
        "distortion_model": str(info.distortion_model),
        "d": [float(v) for v in info.d],
        "k": [float(v) for v in info.k],
        "r": [float(v) for v in info.r],
        "p": [float(v) for v in info.p],
    }


def locate_bag(recording) -> Tuple[Path, str]:
    """``(bag directory, RGBD topic)`` from a run directory, its manifest, or a bare bag."""
    recording = Path(recording)
    manifest = recording / MANIFEST_NAME
    if manifest.is_file():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        return recording / document["bag_dir"], str(document["rgbd_topic"])
    if (recording / "metadata.yaml").is_file():
        return recording, DEFAULT_RGBD_TOPIC
    if (recording / BAG_DIRECTORY / "metadata.yaml").is_file():
        return recording / BAG_DIRECTORY, DEFAULT_RGBD_TOPIC
    raise FileNotFoundError(
        f"{recording} is neither a run directory with {MANIFEST_NAME} nor a rosbag"
    )


def extract_frames(
    bag_dir,
    rgbd_topic: str,
    output,
    every: int = 1,
    limit: Optional[int] = None,
    color_format: str = "png",
    log=print,
) -> int:
    """Write every ``every``-th RGBD message as a colour image and a 16-bit depth PNG.

    ``color/NNNNNN.<format>``, ``depth/NNNNNN.png`` (uint16 millimetres),
    ``frames.csv`` with each frame's stamps, and ``camera_info.json``. Returns
    the number of frames written.
    """
    import cv2

    if every < 1:
        raise ValueError("every must be at least 1")
    output = Path(output)
    color_dir, depth_dir = output / "color", output / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    info_written = False
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        table = csv.writer(stream)
        table.writerow(
            ["index", "message_index", "header_stamp_ns", "rgb_stamp_ns",
             "depth_stamp_ns", "receive_ns", "color_file", "depth_file", "depth_units"]
        )
        for message_index, (_topic, message, receive_ns) in enumerate(
            read_messages(str(bag_dir), [rgbd_topic])
        ):
            if message_index % every:
                continue
            if limit is not None and written >= limit:
                break
            if not info_written:
                (output / "camera_info.json").write_text(
                    json.dumps(camera_info_document(message.rgb_camera_info), indent=2) + "\n",
                    encoding="utf-8",
                )
                info_written = True
            rgb, _ = image_to_numpy(message.rgb)
            depth, units = image_to_numpy(message.depth)
            color_name = f"{written:06d}.{color_format}"
            depth_name = f"{written:06d}.png"
            if not cv2.imwrite(str(color_dir / color_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"could not write {color_dir / color_name}")
            if not cv2.imwrite(str(depth_dir / depth_name), depth_to_millimetres(depth, units)):
                raise OSError(f"could not write {depth_dir / depth_name}")
            table.writerow(
                [
                    written, message_index, stamp_to_ns(message.header.stamp),
                    stamp_to_ns(message.rgb.header.stamp),
                    stamp_to_ns(message.depth.header.stamp), int(receive_ns),
                    f"color/{color_name}", f"depth/{depth_name}", "millimetres",
                ]
            )
            written += 1
            if written % 100 == 0:
                log(f"  {written} frames")
    return written


def _extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the frames of a --record-rgbd bag as colour images and "
                    "16-bit millimetre depth PNGs, with a frames.csv of their stamps."
    )
    parser.add_argument(
        "recording",
        help="a run directory written by replay_trajectory --record-rgbd, or the bag itself",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"where the frames go (default: <run directory>/{FRAMES_DIRECTORY})",
    )
    parser.add_argument("--topic", default=None, help="RGBD topic (default: from the manifest)")
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many frames")
    parser.add_argument(
        "--color-format", choices=("png", "jpg"), default="png",
        help="colour file format; png is lossless (default), jpg is a tenth the size",
    )
    return parser


def extract_main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    parser = _extract_parser()
    args = parser.parse_args(argv)
    if args.every < 1:
        parser.error("--every must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        bag_dir, topic = locate_bag(args.recording)
    except FileNotFoundError as exc:
        print(f"extract error: {exc}", file=sys.stderr)
        return 2
    topic = args.topic or topic
    output = Path(args.output) if args.output else bag_dir.parent / FRAMES_DIRECTORY
    print(f"reading {topic} from {bag_dir}")
    try:
        written = extract_frames(
            bag_dir, topic, output, every=args.every, limit=args.limit,
            color_format=args.color_format,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"extract error: {exc}", file=sys.stderr)
        return 2
    if written == 0:
        print(f"no frames on {topic}; is that the topic the bag holds?", file=sys.stderr)
        return 1
    print(f"{written} frames written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(extract_main())
