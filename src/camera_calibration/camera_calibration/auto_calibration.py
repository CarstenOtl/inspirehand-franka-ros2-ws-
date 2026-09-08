"""Automated eye-to-hand calibration: the arm drives itself through the poses.

Why this exists at all is worth stating, because it is not "the same thing but
without a human". The hand-guided recorder pairs every image with the robot
pose looked up at that image's timestamp, and the two clocks do not agree - the
first real run logged a steady stream of ``Lookup would require extrapolation
into the future`` and produced 85 mm / 20 deg of residual with a flange-to-tag
offset of 1.4 m, which is nonsense. While the arm moves, an unknown delay of a
few tens of milliseconds is an error of tens of millimetres, and it is
correlated with the motion, so it biases the solution rather than averaging out.

The automated run removes the problem instead of modelling it:

* it **stops** at every pose, so no delay can matter - a measurement taken at
  rest is the same measurement whenever it was taken;
* it proves the arm really stopped, from the scatter of the tag's corners
  across a burst of frames and the spread of the joint states over the same
  window, and discards the pose otherwise;
* it averages the burst's corners before running PnP once, which divides the
  detector's pixel noise by roughly the square root of the frame count;
* it chooses the poses from the camera's point of view (see ``pose_program``),
  so the set is guaranteed to fill the image and to rotate about three axes;
* and then, with the static calibration in hand, it drives a deliberately
  *moving* pass to measure the camera-to-robot delay itself, which is what
  makes the calibration usable while the robot is moving rather than only at
  standstill.

Nothing here commands the arm directly. Every motion is a ``goto`` on the
replay controller from ``franka_trajectory_replay``, which ramps all seven
joints on one quintic profile inside the FR3's limits, rejects anything it
cannot execute, and holds position otherwise. Ctrl-C sends its abort.
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener

from franka_trajectory_replay.kinematics import flange_transform
from franka_trajectory_replay.replay_client import Rejected, ReplayClient
from franka_trajectory_replay.runconfig import load_config, namespaced

from .apriltag_detector import AprilTagDetector
from .calibration_math import (
    CalibrationError,
    calibrate_eye_to_hand,
    invert_transform,
    make_transform,
    quaternion_xyzw_to_matrix,
)
from .capture import CaptureError, summarize_burst
from .pose_program import (
    CameraModel,
    PoseProgramError,
    PoseProgramLimits,
    generate_pose_program,
    orientation_excitation,
)
from .report import calibration_document, transform_document
from .tag_pose import estimate_tag_pose, tag_view_angle_deg, to_ippe_order
from .time_offset import TimeOffsetError, estimate_time_offset

APRILTAG_FAMILIES = ("tag16h5", "tag25h9", "tag36h10", "tag36h11")


def _clean_frame(frame: str) -> str:
    return frame.lstrip("/")


def _transform_from_message(message) -> np.ndarray:
    translation = message.translation
    rotation = message.rotation
    return make_transform(
        quaternion_xyzw_to_matrix([rotation.x, rotation.y, rotation.z, rotation.w]),
        [translation.x, translation.y, translation.z],
    )


class RunAborted(RuntimeError):
    """Raised when the sequence cannot continue."""


@dataclass(frozen=True)
class PoseSample:
    """One pose's worth of evidence, taken with the arm at a standstill."""

    index: int
    burst: object
    camera_to_tag: np.ndarray
    world_to_hand: np.ndarray
    world_to_hand_forward_kinematics: np.ndarray
    reprojection_error_px: float
    view_angle_deg: float
    distance_m: float
    commanded_joint_positions: Optional[np.ndarray]

    @property
    def forward_kinematics_disagreement_m(self) -> float:
        return float(
            np.linalg.norm(
                self.world_to_hand[:3, 3] - self.world_to_hand_forward_kinematics[:3, 3]
            )
        )

    def as_dict(self) -> dict:
        document = {
            "index": self.index,
            "camera_to_tag": self.camera_to_tag.tolist(),
            "world_to_hand": self.world_to_hand.tolist(),
            "reprojection_error_px": self.reprojection_error_px,
            "view_angle_deg": self.view_angle_deg,
            "distance_m": self.distance_m,
            "forward_kinematics_disagreement_m": self.forward_kinematics_disagreement_m,
            "burst": self.burst.as_dict(),
        }
        if self.commanded_joint_positions is not None:
            document["commanded_joint_positions"] = [
                float(value) for value in self.commanded_joint_positions
            ]
        return document


class CalibrationRecorder(Node):
    """The sensing half: tag detections, joint states, and the fixed transforms.

    Every observation carries two timestamps - the message's own header stamp
    and the moment it arrived here. Bursts are selected by arrival, so a skewed
    camera clock cannot make a burst window miss its frames; the header stamps
    are what the delay estimate is fitted to, which is the whole point of it.
    """

    def __init__(
        self,
        image_topic: str,
        camera_info_topic: str,
        joint_state_topic: str,
        world_frame: str,
        hand_frame: str,
        base_frame: str,
        camera_mount_frame: str,
        camera_optical_frame: str,
        arm_joint_names,
        tag_family: str,
        tag_id: int,
        tag_size_m: float,
        history: int = 40000,
    ) -> None:
        super().__init__("camera_calibration_recorder")
        self._world_frame = _clean_frame(world_frame)
        self._hand_frame = _clean_frame(hand_frame)
        self._base_frame = _clean_frame(base_frame)
        self._camera_mount_frame = _clean_frame(camera_mount_frame)
        self._camera_optical_frame = (
            _clean_frame(camera_optical_frame) if camera_optical_frame else ""
        )
        self._arm_joint_names = list(arm_joint_names)
        self._tag_id = int(tag_id)
        self._tag_size = float(tag_size_m)
        self._detector = AprilTagDetector(tag_family)
        self._bridge = CvBridge()

        self._lock = threading.Lock()
        self._detections = deque(maxlen=history)
        self._joint_samples = deque(maxlen=history * 4)
        self._camera_matrix = None
        self._distortion = None
        self._image_size = None
        self._frames = 0
        self._detected = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data
        )
        self.create_subscription(Image, image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_subscription(
            JointState, joint_state_topic, self._joint_state_callback, qos_profile_sensor_data
        )

    # -- clocks ----------------------------------------------------------

    def clock_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    @staticmethod
    def _stamp_seconds(header) -> float:
        return Time.from_msg(header.stamp).nanoseconds * 1.0e-9

    # -- callbacks -------------------------------------------------------

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if message.width == 0 or message.height == 0:
            return
        with self._lock:
            self._camera_matrix = np.asarray(message.k, dtype=float).reshape(3, 3)
            self._distortion = np.asarray(message.d, dtype=float)
            self._image_size = (int(message.width), int(message.height))
            if not self._camera_optical_frame:
                self._camera_optical_frame = _clean_frame(message.header.frame_id)

    def _image_callback(self, message: Image) -> None:
        with self._lock:
            camera_matrix = self._camera_matrix
            distortion = self._distortion
            self._frames += 1
        if camera_matrix is None:
            return
        try:
            color_image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception:  # cv_bridge exception types differ by ROS release
            return
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        matches = [
            detection
            for detection in self._detector.detect(gray_image)
            if detection.identifier == self._tag_id
        ]
        if len(matches) != 1:
            return
        corners = to_ippe_order(matches[0].corners)
        camera_to_tag, reprojection_error = estimate_tag_pose(
            corners, camera_matrix, distortion, self._tag_size
        )
        if camera_to_tag is None:
            return
        with self._lock:
            self._detected += 1
            self._detections.append(
                (
                    self._stamp_seconds(message.header),
                    self.clock_seconds(),
                    corners,
                    reprojection_error,
                    camera_to_tag,
                )
            )

    def _joint_state_callback(self, message: JointState) -> None:
        index = {name: position for name, position in zip(message.name, message.position)}
        try:
            positions = np.asarray([index[name] for name in self._arm_joint_names], dtype=float)
        except KeyError:
            # The hand publishes its own joint states; a message without the arm's
            # seven joints is simply not this stream.
            return
        with self._lock:
            self._joint_samples.append(
                (self._stamp_seconds(message.header), self.clock_seconds(), positions)
            )

    # -- queries ---------------------------------------------------------

    def camera_model(self) -> CameraModel:
        with self._lock:
            if self._camera_matrix is None or self._image_size is None:
                raise RunAborted("no CameraInfo has arrived yet")
            return CameraModel(
                matrix=self._camera_matrix.copy(),
                distortion=self._distortion.copy(),
                width=self._image_size[0],
                height=self._image_size[1],
            )

    def camera_optical_frame(self) -> str:
        with self._lock:
            return self._camera_optical_frame

    @property
    def arm_joint_names(self) -> list:
        return list(self._arm_joint_names)

    def counters(self) -> tuple:
        with self._lock:
            return self._frames, self._detected, len(self._joint_samples)

    def detections_since(self, arrival_s: float) -> list:
        with self._lock:
            return [entry for entry in self._detections if entry[1] >= arrival_s]

    def detections_between(self, first_arrival_s: float, last_arrival_s: float) -> list:
        with self._lock:
            return [
                entry
                for entry in self._detections
                if first_arrival_s <= entry[1] <= last_arrival_s
            ]

    def joint_samples_between(self, first_arrival_s: float, last_arrival_s: float) -> list:
        with self._lock:
            return [
                entry
                for entry in self._joint_samples
                if first_arrival_s <= entry[1] <= last_arrival_s
            ]

    def latest_joint_positions(self) -> np.ndarray:
        with self._lock:
            if not self._joint_samples:
                raise RunAborted("no joint states have arrived yet")
            return self._joint_samples[-1][2].copy()

    def wait_until(self, predicate, timeout_s: float, description: str, poll_s: float = 0.05):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(poll_s)
        raise RunAborted(f"timed out after {timeout_s:.0f} s waiting for {description}")

    def transform(self, parent: str, child: str, timeout_s: float = 5.0) -> np.ndarray:
        """The latest available transform. Everything asking for one is at a standstill."""
        parent = _clean_frame(parent)
        child = _clean_frame(child)
        deadline = time.monotonic() + timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                message = self._tf_buffer.lookup_transform(parent, child, Time())
                return _transform_from_message(message.transform)
            except TransformException as error:
                last_error = error
                time.sleep(0.05)
        raise RunAborted(f"no {parent} -> {child} transform: {last_error}")

    def stream_ages(self, duration_s: float = 2.0) -> dict:
        """How stale the newest image and joint state are, and how far apart they read.

        This is a symptom check, not the delay measurement: it mixes transport
        latency into the timestamps. A large ``stamp_difference_s`` is the same
        complaint the passive recorder made about extrapolation.
        """
        image_ages, joint_ages, differences = [], [], []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            with self._lock:
                newest_image = self._detections[-1][0] if self._detections else None
                newest_joint = self._joint_samples[-1][0] if self._joint_samples else None
            now = self.clock_seconds()
            if newest_image is not None:
                image_ages.append(now - newest_image)
            if newest_joint is not None:
                joint_ages.append(now - newest_joint)
            if newest_image is not None and newest_joint is not None:
                differences.append(newest_image - newest_joint)
            time.sleep(0.02)
        return {
            "image_age_s": float(np.median(image_ages)) if image_ages else None,
            "joint_state_age_s": float(np.median(joint_ages)) if joint_ages else None,
            "stamp_difference_s": float(np.median(differences)) if differences else None,
            "joint_state_rate_hz": self._observed_joint_rate(),
        }

    def _observed_joint_rate(self) -> Optional[float]:
        with self._lock:
            stamps = [entry[0] for entry in self._joint_samples][-200:]
        if len(stamps) < 10:
            return None
        intervals = np.diff(stamps)
        intervals = intervals[intervals > 0.0]
        return float(1.0 / np.median(intervals)) if len(intervals) else None


@dataclass
class RunOptions:
    tag_family: str
    tag_id: int
    tag_size_m: float
    world_frame: str
    hand_frame: str
    base_frame: str
    camera_mount_frame: str
    camera_optical_frame: str
    image_topic: str
    camera_info_topic: str
    joint_state_topic: str
    pose_count: int
    minimum_samples: int
    settle_seconds: float
    burst_frames: int
    burst_timeout_s: float
    max_corner_std_px: float
    max_joint_spread_rad: float
    max_reprojection_error_px: float
    seed_poses_path: Optional[str]
    seed_result_path: Optional[str]
    program_path: Optional[str]
    output_dir: str
    program_seed: int
    offset_pass_poses: int
    offset_pass_cycles: int
    skip_offset_pass: bool
    return_home: bool
    dry_run: bool
    assume_yes: bool
    limits: PoseProgramLimits


class AutoCalibrationRunner:
    """Drives the whole sequence. Every motion goes through the replay controller."""

    def __init__(self, recorder: CalibrationRecorder, replay, options: RunOptions, run_dir: str):
        self.recorder = recorder
        self.replay = replay
        self.options = options
        self.run_dir = run_dir
        self.world_to_base = np.eye(4)
        self.steps: list[dict] = []

    def log(self, text: str = "") -> None:
        print(text, flush=True)

    def gate(self, prompt: str) -> None:
        if self.options.assume_yes:
            self.log(f">> {prompt} (auto)")
            return
        try:
            input(f">> {prompt}  [Enter to continue, Ctrl-C to abort] ")
        except EOFError:
            raise RunAborted(
                "stdin closed with no answer; pass --yes for a non-interactive run"
            )

    # -- preparation -----------------------------------------------------

    def prepare(self) -> dict:
        options = self.options
        self.log("Waiting for the camera, the tag and the robot ...")
        self.recorder.wait_until(
            lambda: self.recorder.counters()[0] > 0,
            30.0,
            f"images on {options.image_topic} (is a realsense2_camera node running?)",
        )
        self.recorder.wait_until(
            lambda: self.recorder.camera_optical_frame() != "",
            15.0,
            f"CameraInfo on {options.camera_info_topic}",
        )
        self.recorder.wait_until(
            lambda: self.recorder.counters()[2] > 0, 30.0, "joint states from the arm"
        )
        self.recorder.wait_until(
            lambda: self.recorder.counters()[1] > 0,
            30.0,
            f"a detection of tag {options.tag_id} (is the tag in view and lit?)",
        )
        self.world_to_base = self.recorder.transform(options.world_frame, options.base_frame)
        self.recorder.transform(options.world_frame, options.hand_frame)

        model = self.recorder.camera_model()
        ages = self.recorder.stream_ages()
        frames, detected, _ = self.recorder.counters()
        self.log(
            f"   camera: {model.width}x{model.height}, tag seen in {detected} of {frames} frames"
        )
        if ages["joint_state_rate_hz"]:
            self.log(f"   joint states: {ages['joint_state_rate_hz']:.0f} Hz")
        if ages["stamp_difference_s"] is not None:
            self.log(
                "   newest image stamp reads %+.0f ms against the newest joint state "
                "(image age %.0f ms, joint age %.0f ms)"
                % (
                    ages["stamp_difference_s"] * 1000.0,
                    (ages["image_age_s"] or 0.0) * 1000.0,
                    (ages["joint_state_age_s"] or 0.0) * 1000.0,
                )
            )
            self.log(
                "   that skew is why this run stops at every pose; the offset pass at the "
                "end measures it properly."
            )
        if not np.allclose(self.world_to_base, np.eye(4), atol=1.0e-9):
            self.log(
                f"   note: {options.world_frame} and {options.base_frame} are not the same "
                "frame; the pose program accounts for the offset between them"
            )
        return ages

    def mount_to_optical(self) -> np.ndarray:
        options = self.options
        optical = self.recorder.camera_optical_frame()
        if _clean_frame(options.camera_mount_frame) == optical:
            return np.eye(4)
        return self.recorder.transform(options.camera_mount_frame, optical)

    # -- teaching --------------------------------------------------------

    def teach(self, path: str) -> None:
        options = self.options
        self.log(
            "\nTeaching seed poses. The arm must be in gravity compensation:\n"
            "  ros2 launch inspire_franka_bringup inspire_franka.launch.py \\\n"
            "      robot_ip:=<ip> hand_port:=/dev/ttyUSB0 gravity_compensation:=true\n\n"
            "Guide the hand to a pose where the whole tag is in view, let go, then press\n"
            "Enter. Six poses is the minimum; eight to ten spread across the camera's view,\n"
            "with clearly different tilts, make the coarse solve safe. Type 'done' to finish.\n"
        )
        taught: list[np.ndarray] = []
        while True:
            try:
                answer = input(f"[{len(taught)} taught] Enter to record, 'done' to finish: ")
            except EOFError:
                break
            if answer.strip().lower() in ("done", "q", "quit"):
                break
            detections = self.recorder.detections_since(self.recorder.clock_seconds() - 1.0)
            if not detections:
                self.log("   the tag is not being detected here - move so it is fully in view")
                continue
            joint_positions = self.recorder.latest_joint_positions()
            taught.append(joint_positions)
            self.log(
                "   recorded ["
                + " ".join("%+.4f" % value for value in joint_positions)
                + "]"
            )
        if len(taught) < 6:
            raise RunAborted(
                f"only {len(taught)} poses were taught; the coarse solve needs at least 6"
            )
        document = {
            "joint_names": self.recorder.arm_joint_names,
            "tag": {"id": options.tag_id, "size_m": options.tag_size_m},
            "poses": [[float(value) for value in pose] for pose in taught],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(document, handle, sort_keys=False)
        self.log(f"\nWrote {len(taught)} seed poses to {path}")
        self.log(
            "Now restart the arm under the replay controller and run without --teach:\n"
            "  ros2 launch franka_trajectory_replay replay.launch.py robot_config_file:=<yaml>\n"
            f"  ros2 run camera_calibration auto_calibrate --seed-poses {path}"
        )

    # -- capture ---------------------------------------------------------

    def capture_at_current_pose(
        self, index: int, commanded: Optional[np.ndarray] = None
    ) -> PoseSample:
        options = self.options
        time.sleep(options.settle_seconds)
        started = self.recorder.clock_seconds()
        self.recorder.wait_until(
            lambda: len(self.recorder.detections_since(started)) >= options.burst_frames,
            options.burst_timeout_s,
            f"{options.burst_frames} tag detections at pose {index}",
        )
        detections = self.recorder.detections_since(started)[: options.burst_frames]
        joint_samples = self.recorder.joint_samples_between(
            started, self.recorder.clock_seconds()
        )
        burst = summarize_burst(
            [(entry[0], entry[2], entry[3]) for entry in detections],
            [entry[2] for entry in joint_samples],
            minimum_frames=max(4, options.burst_frames // 2),
            max_corner_std_px=options.max_corner_std_px,
            max_joint_spread_rad=options.max_joint_spread_rad,
            max_reprojection_error_px=options.max_reprojection_error_px,
        )

        model = self.recorder.camera_model()
        camera_to_tag, reprojection_error = estimate_tag_pose(
            burst.corners, model.matrix, model.distortion, options.tag_size_m
        )
        if camera_to_tag is None:
            raise CaptureError("PnP failed on the averaged corners")
        if reprojection_error > options.max_reprojection_error_px:
            raise CaptureError(
                f"the averaged corners reproject {reprojection_error:.2f} px off, more than "
                f"{options.max_reprojection_error_px:.2f} px"
            )
        world_to_hand = self.recorder.transform(options.world_frame, options.hand_frame)
        return PoseSample(
            index=index,
            burst=burst,
            camera_to_tag=camera_to_tag,
            world_to_hand=world_to_hand,
            world_to_hand_forward_kinematics=self.world_to_base
            @ flange_transform(burst.joint_positions),
            reprojection_error_px=reprojection_error,
            view_angle_deg=tag_view_angle_deg(camera_to_tag),
            distance_m=float(np.linalg.norm(camera_to_tag[:3, 3])),
            commanded_joint_positions=None if commanded is None else np.asarray(commanded),
        )

    def goto(self, joint_positions, label: str) -> None:
        current = self.recorder.latest_joint_positions()
        step = float(np.abs(np.asarray(joint_positions) - current).max())
        self.log(f"   {label}: ramping, largest joint step {step:.3f} rad")
        self.replay.goto(joint_positions)

    def drive_and_capture(self, index: int, joint_positions, label: str) -> PoseSample:
        self.goto(joint_positions, label)
        sample = self.capture_at_current_pose(index, commanded=joint_positions)
        self.log(
            "   pose %d: %d frames, corner scatter %.2f px, reprojection %.2f px, "
            "%.2f m at %.0f deg, FK vs TF %.2f mm"
            % (
                index,
                sample.burst.frames_used,
                sample.burst.corner_std_px,
                sample.reprojection_error_px,
                sample.distance_m,
                sample.view_angle_deg,
                sample.forward_kinematics_disagreement_m * 1000.0,
            )
        )
        return sample

    # -- the passes ------------------------------------------------------

    def load_seed_poses(self, path: str) -> list:
        with open(os.path.expanduser(path), "r") as handle:
            document = yaml.safe_load(handle) or {}
        poses = [np.asarray(entry, dtype=float) for entry in document.get("poses", [])]
        if len(poses) < 6:
            raise RunAborted(f"{path} holds {len(poses)} seed poses; at least 6 are needed")
        if any(pose.shape != (7,) for pose in poses):
            raise RunAborted(f"{path} holds a pose that is not seven joint values")
        names = document.get("joint_names")
        if names and list(names) != self.recorder.arm_joint_names:
            raise RunAborted(
                f"{path} was taught for joints {list(names)}, but this arm reports "
                f"{self.recorder.arm_joint_names}"
            )
        return poses

    def load_seed_result(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        with open(os.path.expanduser(path), "r") as handle:
            document = json.load(handle)
        optical = document.get("transform_camera_optical") or document.get("transform")
        if optical is None:
            raise RunAborted(f"{path} carries no camera transform")
        world_to_camera = np.asarray(optical["matrix_4x4"], dtype=float)
        hand_to_tag = np.asarray(
            document["estimated_carrier_to_tag"]["matrix_4x4"], dtype=float
        )
        if document.get("transform_camera_optical") is None:
            self.log(
                f"   note: {path} predates the optical-frame field, so its mount transform "
                "is being used as the optical one; the coarse aim may be a few centimetres off"
            )
        return world_to_camera, hand_to_tag

    def load_program(self, path: str) -> list:
        with open(os.path.expanduser(path), "r") as handle:
            stored = yaml.safe_load(handle) or {}
        names = stored.get("joint_names")
        if names and list(names) != self.recorder.arm_joint_names:
            raise RunAborted(
                f"{path} was generated for joints {list(names)}, but this arm reports "
                f"{self.recorder.arm_joint_names}"
            )
        entries = [np.asarray(entry, dtype=float) for entry in stored.get("poses", [])]
        if any(entry.shape != (7,) for entry in entries):
            raise RunAborted(f"{path} holds a pose that is not seven joint values")
        if len(entries) < self.options.minimum_samples:
            raise RunAborted(
                f"{path} holds {len(entries)} poses, fewer than the "
                f"{self.options.minimum_samples} the solver requires"
            )
        self.log(f"\n[1-2/4] Replaying the {len(entries)} stored poses from {path}")
        return [_StoredPose(entry) for entry in entries]

    def coarse_pass(self, seed_poses) -> tuple:
        self.log(f"\n[1/4] Coarse pass over {len(seed_poses)} taught seed poses")
        samples = []
        for index, joint_positions in enumerate(seed_poses):
            try:
                samples.append(self.drive_and_capture(index, joint_positions, f"seed {index}"))
            except CaptureError as error:
                self.log(f"   pose {index} skipped: {error}")
        if len(samples) < 6:
            raise RunAborted(
                f"only {len(samples)} of {len(seed_poses)} seed poses produced a usable "
                "observation; re-teach them with the whole tag clearly in view"
            )
        result, _ = calibrate_eye_to_hand(
            [sample.world_to_hand for sample in samples],
            [sample.camera_to_tag for sample in samples],
            minimum_samples=6,
            reject_outliers=False,
        )
        self.log(
            "   coarse camera at [%s] m, residual %.1f mm / %.2f deg"
            % (
                " ".join("%+.3f" % value for value in result.world_to_camera[:3, 3]),
                result.translation_rmse_m * 1000.0,
                result.rotation_rmse_deg,
            )
        )
        self.log(
            "   coarse flange-to-tag offset %.3f m - it should be within a few centimetres "
            "of where the tag really sits on the hand" % np.linalg.norm(result.hand_to_tag[:3, 3])
        )
        return result, samples

    def build_program(self, world_to_camera: np.ndarray, hand_to_tag: np.ndarray) -> list:
        options = self.options
        self.log(f"\n[2/4] Generating {options.pose_count} calibration poses")
        poses = generate_pose_program(
            world_to_camera=world_to_camera,
            hand_to_tag=hand_to_tag,
            camera=self.recorder.camera_model(),
            tag_size_m=options.tag_size_m,
            count=options.pose_count,
            world_to_base=self.world_to_base,
            start_joint_positions=self.recorder.latest_joint_positions(),
            limits=options.limits,
            seed=options.program_seed,
        )
        spread, informative = orientation_excitation([pose.world_to_hand for pose in poses])
        self.log(
            "   %d poses, %.2f-%.2f m from the camera, view angles up to %.0f deg"
            % (
                len(poses),
                min(pose.distance_m for pose in poses),
                max(pose.distance_m for pose in poses),
                max(pose.view_angle_deg for pose in poses),
            )
        )
        self.log(
            f"   rotation axis spread {spread:.2f} over {informative} informative pose pairs"
        )
        return poses

    def program_pass(self, poses) -> list:
        options = self.options
        self.log(f"\n[3/4] Driving {len(poses)} poses, stopping at each one")
        samples = []
        for index, pose in enumerate(poses):
            try:
                samples.append(
                    self.drive_and_capture(
                        index, pose.joint_positions, f"pose {index + 1}/{len(poses)}"
                    )
                )
            except CaptureError as error:
                self.log(f"   pose {index} discarded: {error}")
        if len(samples) < options.minimum_samples:
            raise RunAborted(
                f"only {len(samples)} of {len(poses)} poses produced a usable observation, "
                f"fewer than the {options.minimum_samples} required"
            )
        return samples

    def offset_pass(self, poses, world_to_camera: np.ndarray, hand_to_tag: np.ndarray):
        options = self.options
        stride = max(1, len(poses) // options.offset_pass_poses)
        waypoints = poses[::stride][: options.offset_pass_poses]
        if len(waypoints) < 3:
            raise RunAborted("the offset pass needs at least three waypoints")
        route = list(waypoints) * options.offset_pass_cycles
        self.log(
            f"\n[4/4] Offset pass: {len(route)} ramps through {len(waypoints)} waypoints, "
            "recording while the arm moves"
        )
        started = self.recorder.clock_seconds()
        for index, pose in enumerate(route):
            self.goto(pose.joint_positions, f"leg {index + 1}/{len(route)}")
        ended = self.recorder.clock_seconds()

        detections = self.recorder.detections_between(started, ended)
        joint_samples = self.recorder.joint_samples_between(started, ended)
        self.log(
            f"   {len(detections)} tag observations and {len(joint_samples)} joint states "
            "over the pass"
        )
        camera_times = np.asarray([entry[0] for entry in detections])
        camera_positions = np.asarray(
            [(world_to_camera @ entry[4])[:3, 3] for entry in detections]
        )
        robot_times = np.asarray([entry[0] for entry in joint_samples])
        robot_positions = np.asarray(
            [
                (self.world_to_base @ flange_transform(entry[2]) @ hand_to_tag)[:3, 3]
                for entry in joint_samples
            ]
        )
        return estimate_time_offset(
            camera_times, camera_positions, robot_times, robot_positions
        )

    # -- output ----------------------------------------------------------

    def solve(self, samples) -> tuple:
        result, retained = calibrate_eye_to_hand(
            [sample.world_to_hand for sample in samples],
            [sample.camera_to_tag for sample in samples],
            minimum_samples=self.options.minimum_samples,
        )
        return result, retained

    def document(self, result, retained, samples, offset, ages, poses) -> dict:
        options = self.options
        optical = self.recorder.camera_optical_frame()
        world_to_mount = result.world_to_camera @ invert_transform(self.mount_to_optical())
        model = self.recorder.camera_model()
        spread, informative = orientation_excitation(
            [sample.world_to_hand for sample in samples]
        )
        quality = {
            "samples_collected": len(samples),
            "samples_used": int(np.count_nonzero(retained)),
            "translation_rmse_m": result.translation_rmse_m,
            "rotation_rmse_deg": result.rotation_rmse_deg,
            "capture_mode": "automated, stationary at every pose",
            "median_corner_scatter_px": float(
                np.median([sample.burst.corner_std_px for sample in samples])
            ),
            "median_reprojection_error_px": float(
                np.median([sample.reprojection_error_px for sample in samples])
            ),
            "max_forward_kinematics_disagreement_m": float(
                max(sample.forward_kinematics_disagreement_m for sample in samples)
            ),
            "rotation_axis_spread": spread,
            "informative_pose_pairs": informative,
        }
        extra = {
            "transform_camera_optical": transform_document(result.world_to_camera),
            "stream_ages": ages,
            "program": {
                "poses_planned": len(poses) if poses is not None else None,
                "pose_program_seed": options.program_seed,
                "distance_range_m": list(options.limits.distance_range_m),
                "max_tilt_deg": options.limits.max_tilt_deg,
            },
        }
        if offset is not None:
            extra["time_offset"] = {
                **offset.as_dict(),
                "meaning": (
                    "add offset_s to a camera timestamp to reach the robot clock; a "
                    "positive value means images are stamped early"
                ),
            }
        return calibration_document(
            parent_frame=options.world_frame,
            child_frame=options.camera_mount_frame,
            camera_optical_frame=optical,
            camera_matrix=model.matrix,
            distortion=model.distortion,
            world_to_child=world_to_mount,
            carrier_frame=options.hand_frame,
            hand_to_tag=result.hand_to_tag,
            tag_id=options.tag_id,
            tag_size_m=options.tag_size_m,
            quality=quality,
            extra=extra,
        )

    def write_outputs(self, document, samples, poses, offset) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "result.json"), "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
        with open(os.path.join(self.run_dir, "poses.json"), "w") as handle:
            json.dump([sample.as_dict() for sample in samples], handle, indent=2)
        if poses is not None:
            with open(os.path.join(self.run_dir, "program.yaml"), "w") as handle:
                yaml.safe_dump(
                    {
                        "joint_names": self.recorder.arm_joint_names,
                        "poses": [
                            [float(value) for value in pose.joint_positions] for pose in poses
                        ],
                        **(
                            {"predicted": [pose.as_dict() for pose in poses]}
                            if all(hasattr(pose, "as_dict") for pose in poses)
                            else {}
                        ),
                    },
                    handle,
                    sort_keys=False,
                )
        if offset is not None:
            with open(os.path.join(self.run_dir, "offset_scan.json"), "w") as handle:
                json.dump(
                    {
                        "offsets_s": offset.scan_offsets_s.tolist(),
                        "rms_m": offset.scan_rms_m.tolist(),
                        **offset.as_dict(),
                    },
                    handle,
                    indent=2,
                )

    # -- the whole run ---------------------------------------------------

    def run(self) -> dict:
        options = self.options
        ages = self.prepare()
        home = self.recorder.latest_joint_positions()

        # A stored program already says where to go, so it needs no coarse aim and
        # the taught seed poses are not driven at all.
        if options.program_path:
            poses = self.load_program(options.program_path)
        else:
            if options.seed_result_path:
                self.log(f"\n[1/4] Coarse aim taken from {options.seed_result_path}")
                coarse_camera, coarse_tag = self.load_seed_result(options.seed_result_path)
            else:
                seed_poses = self.load_seed_poses(options.seed_poses_path)
                self.gate(
                    f"about to drive {len(seed_poses)} taught seed poses - the arm will move"
                )
                coarse, _ = self.coarse_pass(seed_poses)
                coarse_camera, coarse_tag = coarse.world_to_camera, coarse.hand_to_tag
            poses = self.build_program(coarse_camera, coarse_tag)

        if options.dry_run:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "program.yaml"), "w") as handle:
                yaml.safe_dump(
                    {
                        "joint_names": self.recorder.arm_joint_names,
                        "poses": [
                            [float(value) for value in pose.joint_positions] for pose in poses
                        ],
                    },
                    handle,
                    sort_keys=False,
                )
            self.log(
                f"\n--dry-run: wrote the program to {self.run_dir}/program.yaml and stopped "
                "before driving it."
            )
            return {}

        self.gate(f"about to drive {len(poses)} calibration poses - the arm will move")
        samples = self.program_pass(poses)
        result, retained = self.solve(samples)
        rejected = len(retained) - int(np.count_nonzero(retained))
        self.log(
            "\n   %s -> %s: %d of %d samples used (%d rejected), residual %.1f mm / %.2f deg"
            % (
                options.world_frame,
                options.camera_mount_frame,
                int(np.count_nonzero(retained)),
                len(samples),
                rejected,
                result.translation_rmse_m * 1000.0,
                result.rotation_rmse_deg,
            )
        )

        offset = None
        if options.skip_offset_pass:
            self.log("\n[4/4] Offset pass skipped (--no-offset-pass)")
        else:
            self.gate("about to drive the moving offset pass - the arm will move continuously")
            try:
                offset = self.offset_pass(poses, result.world_to_camera, result.hand_to_tag)
                self.log(
                    "   camera stamps run %+.1f +/- %.1f ms against the robot clock "
                    "(residual %.1f mm, %.1f mm if the delay is ignored, %d observations)"
                    % (
                        offset.offset_s * 1000.0,
                        offset.uncertainty_s * 1000.0,
                        offset.rms_at_offset_m * 1000.0,
                        offset.rms_at_zero_m * 1000.0,
                        offset.samples_used,
                    )
                )
            except TimeOffsetError as error:
                self.log(f"   the delay could not be measured: {error}")

        document = self.document(result, retained, samples, offset, ages, poses)
        self.write_outputs(document, samples, poses, offset)

        if options.return_home:
            self.log("\nReturning to the configuration the run started from")
            self.goto(home, "home")

        self.log(f"\nWrote {self.run_dir}/result.json")
        self.log("CALIBRATION_RESULT " + json.dumps(document, sort_keys=True))
        self.summarize(result, offset)
        return document

    def summarize(self, result, offset) -> None:
        translation = result.world_to_camera[:3, 3]
        self.log("\n--- summary ---")
        self.log(
            "%s -> %s at [%s] m"
            % (
                self.options.world_frame,
                self.options.camera_mount_frame,
                " ".join("%+.4f" % value for value in translation),
            )
        )
        self.log(
            "residual %.1f mm / %.2f deg over the poses that were kept"
            % (result.translation_rmse_m * 1000.0, result.rotation_rmse_deg)
        )
        self.log(
            "flange-to-tag offset %.3f m (sanity check: the tag is on the palm, so this "
            "should be a few centimetres)" % np.linalg.norm(result.hand_to_tag[:3, 3])
        )
        if offset is not None:
            self.log(
                "camera-to-robot delay %+.1f ms +/- %.1f ms - add it to a camera timestamp "
                "before looking up a robot pose"
                % (offset.offset_s * 1000.0, offset.uncertainty_s * 1000.0)
            )
        if result.translation_rmse_m > 0.005 or result.rotation_rmse_deg > 0.5:
            self.log(
                "\nThe residual is larger than a good run's (a few millimetres, well under "
                "half a degree). Check the tag's measured edge length, that the tag is flat "
                "and rigid on the palm, and the lighting."
            )


@dataclass(frozen=True)
class _StoredPose:
    """A pose read back from a stored program, which carries no prediction."""

    joint_positions: np.ndarray


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_calibrate",
        description=(
            "Automated eye-to-hand calibration of a fixed RealSense from an AprilTag on "
            "the Inspire Hand. The arm drives itself through poses generated from the "
            "camera's point of view, stops at each one, and finally measures the delay "
            "between the camera's and the robot's clocks."
        ),
    )
    tag = parser.add_argument_group("tag")
    tag.add_argument("--tag-family", choices=APRILTAG_FAMILIES, default="tag36h11")
    tag.add_argument("--tag-id", type=int, default=0)
    tag.add_argument(
        "--tag-size-m",
        type=float,
        default=0.040,
        help="measured outer edge of the black square; every distance scales with it",
    )

    frames = parser.add_argument_group("frames and topics")
    frames.add_argument("--world-frame", default="world")
    frames.add_argument("--hand-frame", default="fr3_link8")
    frames.add_argument(
        "--base-frame",
        default="fr3_link0",
        help="frame the FR3 forward kinematics are rooted in",
    )
    frames.add_argument("--camera-mount-frame", default="camera_link")
    frames.add_argument(
        "--camera-optical-frame", default="", help="empty takes it from CameraInfo"
    )
    frames.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    frames.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    frames.add_argument(
        "--joint-state-topic",
        default="",
        help="empty derives it from the replay controller's namespace",
    )

    seeding = parser.add_argument_group("where the coarse aim comes from")
    seeding.add_argument(
        "--teach",
        action="store_true",
        help="record seed joint configurations by hand guiding, then exit",
    )
    seeding.add_argument(
        "--seed-poses",
        default="~/camera_calibration_runs/seed_poses.yaml",
        help="taught seed configurations, driven once for the coarse solve",
    )
    seeding.add_argument(
        "--seed-result",
        default=None,
        help="result.json from an earlier run; skips the coarse pass entirely",
    )
    seeding.add_argument(
        "--program",
        default=None,
        help="drive a stored program.yaml instead of generating a new one",
    )

    program = parser.add_argument_group("pose program")
    program.add_argument("--poses", type=int, default=28)
    program.add_argument("--program-seed", type=int, default=0)
    program.add_argument(
        "--distance-range-m", type=float, nargs=2, metavar=("NEAR", "FAR"), default=(0.35, 0.85)
    )
    program.add_argument("--max-tilt-deg", type=float, default=35.0)
    program.add_argument("--image-margin-px", type=float, default=50.0)
    program.add_argument("--minimum-samples", type=int, default=12)

    capture = parser.add_argument_group("capture at each pose")
    capture.add_argument("--settle-seconds", type=float, default=1.0)
    capture.add_argument("--burst-frames", type=int, default=10)
    capture.add_argument("--burst-timeout-s", type=float, default=8.0)
    capture.add_argument("--max-corner-std-px", type=float, default=0.35)
    capture.add_argument("--max-joint-spread-urad", type=float, default=200.0)
    capture.add_argument("--max-reprojection-error-px", type=float, default=1.5)

    offset = parser.add_argument_group("clock offset pass")
    offset.add_argument("--offset-pass-poses", type=int, default=6)
    offset.add_argument("--offset-pass-cycles", type=int, default=2)
    offset.add_argument("--no-offset-pass", action="store_true")

    parser.add_argument("--output-dir", default="~/camera_calibration_runs")
    parser.add_argument(
        "--replay-config", default=None, help="replay.yaml for the controller's namespace"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=(
            "override the replay config's namespace, which decides where the controller "
            "and joint states are looked for; pass an empty string for an unnamespaced "
            "arm. replay.yaml defaults to NS_1, which is not what the FR3 configs in "
            "this workspace use."
        ),
    )
    parser.add_argument("--no-return-home", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate and save the pose program without moving the arm",
    )
    parser.add_argument("--yes", action="store_true", help="do not prompt before moving")
    return parser


def _options(args) -> RunOptions:
    return RunOptions(
        tag_family=args.tag_family,
        tag_id=args.tag_id,
        tag_size_m=args.tag_size_m,
        world_frame=args.world_frame,
        hand_frame=args.hand_frame,
        base_frame=args.base_frame,
        camera_mount_frame=args.camera_mount_frame,
        camera_optical_frame=args.camera_optical_frame,
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        joint_state_topic=args.joint_state_topic,
        pose_count=args.poses,
        minimum_samples=args.minimum_samples,
        settle_seconds=args.settle_seconds,
        burst_frames=args.burst_frames,
        burst_timeout_s=args.burst_timeout_s,
        max_corner_std_px=args.max_corner_std_px,
        max_joint_spread_rad=args.max_joint_spread_urad * 1.0e-6,
        max_reprojection_error_px=args.max_reprojection_error_px,
        seed_poses_path=args.seed_poses,
        seed_result_path=args.seed_result,
        program_path=args.program,
        output_dir=os.path.expanduser(args.output_dir),
        program_seed=args.program_seed,
        offset_pass_poses=args.offset_pass_poses,
        offset_pass_cycles=args.offset_pass_cycles,
        skip_offset_pass=args.no_offset_pass,
        return_home=not args.no_return_home,
        dry_run=args.dry_run,
        assume_yes=args.yes,
        limits=PoseProgramLimits(
            distance_range_m=tuple(args.distance_range_m),
            max_tilt_deg=args.max_tilt_deg,
            image_margin_px=args.image_margin_px,
        ),
    )


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(remove_ros_args(argv or sys.argv)[1:])
    if args.tag_size_m <= 0.0:
        parser.error("--tag-size-m must be greater than zero")
    if args.poses < 8:
        parser.error("--poses must be at least 8")
    if args.minimum_samples < 4:
        parser.error("--minimum-samples must be at least 4")
    if args.burst_frames < 4:
        parser.error("--burst-frames must be at least 4")
    if args.distance_range_m[0] <= 0.0 or args.distance_range_m[1] <= args.distance_range_m[0]:
        parser.error("--distance-range-m must be an increasing pair of positive distances")
    options = _options(args)

    rclpy.init()
    config = load_config(args.replay_config)
    if args.namespace is not None:
        config["namespace"] = args.namespace
    run_dir = os.path.join(
        options.output_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )

    recorder_joint_state_topic = options.joint_state_topic or namespaced(
        config["namespace"], config["joint_state_topic"]
    )
    recorder = CalibrationRecorder(
        image_topic=options.image_topic,
        camera_info_topic=options.camera_info_topic,
        joint_state_topic=recorder_joint_state_topic,
        world_frame=options.world_frame,
        hand_frame=options.hand_frame,
        base_frame=options.base_frame,
        camera_mount_frame=options.camera_mount_frame,
        camera_optical_frame=options.camera_optical_frame,
        arm_joint_names=config["joint_names"],
        tag_family=options.tag_family,
        tag_id=options.tag_id,
        tag_size_m=options.tag_size_m,
    )
    replay = None if args.teach else ReplayClient(config, node_name="camera_calibration_replay")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(recorder)
    if replay is not None:
        executor.add_node(replay)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    runner = AutoCalibrationRunner(recorder, replay, options, run_dir)
    namespace = config["namespace"] or "(none)"
    runner.log(
        f"Controller namespace {namespace}; joint states on "
        f"{recorder_joint_state_topic}; writing to {run_dir}"
    )
    status = 0
    try:
        if args.teach:
            runner.teach(os.path.expanduser(args.seed_poses))
        else:
            replay.ensure_active(log=runner.log)
            runner.run()
    except KeyboardInterrupt:
        print("\nCtrl-C - sending abort to the controller", flush=True)
        if replay is not None:
            replay.abort()
        status = 130
    except (RunAborted, CalibrationError, PoseProgramError, CaptureError, Rejected) as error:
        print(f"\nerror: {error}", file=sys.stderr, flush=True)
        if replay is not None:
            replay.abort()
        status = 1
    finally:
        executor.shutdown()
        recorder.destroy_node()
        if replay is not None:
            replay.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
