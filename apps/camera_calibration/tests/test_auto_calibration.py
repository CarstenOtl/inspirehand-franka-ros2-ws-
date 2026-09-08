"""The automated sequence, driven end to end against a simulated robot and camera.

Six hundred lines of sequencing decide what a real FR3 does, so they are worth
exercising without one. The fake below is a real simulation rather than a set of
canned answers: it carries a true camera pose, a true flange-to-tag offset and a
true camera clock delay, moves a virtual arm through the forward kinematics on
every ``goto``, renders the tag's corners through the actual projection, and
stamps the images early by the delay the way the hardware does. The run has to
recover all three.

Needs rclpy (the runner is a ROS node module), so it runs inside the workspace
image alongside the rest of the tests.
"""

import json
import os

import numpy as np
import pytest

from camera_calibration.calibration_math import invert_transform, make_transform, rotation_angle_deg
from camera_calibration.pose_program import CameraModel, PoseProgramLimits, generate_pose_program
from camera_calibration.tag_pose import project_tag_corners, to_ippe_order
from franka_trajectory_replay.kinematics import READY_POSE, flange_transform

rclpy = pytest.importorskip("rclpy")
from camera_calibration.auto_calibration import (  # noqa: E402
    AutoCalibrationRunner,
    RunOptions,
)

TAG_SIZE = 0.040
OPTICAL_FRAME = "camera_color_optical_frame"
WORLD_TO_CAMERA = make_transform(
    np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), [0.95, 0.0, 0.55]
)
HAND_TO_TAG = make_transform(
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), [0.02, -0.03, 0.09]
)
CAMERA = CameraModel(
    matrix=np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]),
    distortion=np.zeros(5),
    width=640,
    height=480,
)
JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]
FRAME_INTERVAL = 1.0 / 30.0


class FakeRecorder:
    """A virtual D415 and FR3 sharing one virtual clock, with a skewed camera stamp."""

    def __init__(self, start_joint_positions, camera_delay_s=0.040, corner_noise_px=0.05,
                 seed=11):
        self.joint_positions = np.asarray(start_joint_positions, dtype=float)
        self.camera_delay_s = camera_delay_s
        self.corner_noise_px = corner_noise_px
        self.random = np.random.default_rng(seed)
        self.now = 1000.0
        self.detections = []
        self.joint_samples = []
        self.blur_poses = set()
        self.pose_counter = 0
        self.record(frames=12)

    # -- the simulation --------------------------------------------------

    def observe(self):
        camera_to_tag = (
            invert_transform(WORLD_TO_CAMERA) @ flange_transform(self.joint_positions) @ HAND_TO_TAG
        )
        corners = project_tag_corners(camera_to_tag, CAMERA.matrix, CAMERA.distortion, TAG_SIZE)
        # Well past max_corner_std_px, so the burst is unmistakably not a standstill.
        scale = 20.0 if self.pose_counter in self.blur_poses else 1.0
        corners = corners + self.random.normal(
            scale=self.corner_noise_px * scale, size=(4, 2)
        )
        # The detector hands its corners over in the opposite order, so the run has
        # to reorder them exactly as it does on real images.
        return to_ippe_order(corners[::-1])

    def record(self, frames=1):
        """Advance the clock, emitting one joint state and one image per frame."""
        for _ in range(frames):
            self.now += FRAME_INTERVAL
            self.joint_samples.append((self.now, self.now, self.joint_positions.copy()))
            corners = self.observe()
            camera_to_tag = (
                invert_transform(WORLD_TO_CAMERA)
                @ flange_transform(self.joint_positions)
                @ HAND_TO_TAG
            )
            # Images are stamped early by the delay - the skew the run measures.
            self.detections.append(
                (self.now - self.camera_delay_s, self.now, corners, 0.05, camera_to_tag)
            )

    def ramp(self, target, rate_rad_per_s=0.4):
        target = np.asarray(target, dtype=float)
        step = np.abs(target - self.joint_positions).max()
        frames = max(4, int(np.ceil(max(step / rate_rad_per_s, 1.0) / FRAME_INTERVAL)))
        start = self.joint_positions.copy()
        for index in range(1, frames + 1):
            # A quintic in time, like the controller's own ramp.
            unit = index / frames
            blend = 10.0 * unit**3 - 15.0 * unit**4 + 6.0 * unit**5
            self.joint_positions = start + (target - start) * blend
            self.record(frames=1)
        self.joint_positions = target.copy()
        self.pose_counter += 1
        self.record(frames=4)

    # -- the recorder interface the runner uses --------------------------

    arm_joint_names = JOINT_NAMES

    def clock_seconds(self):
        return self.now

    def camera_model(self):
        return CAMERA

    def camera_optical_frame(self):
        return OPTICAL_FRAME

    def counters(self):
        return len(self.detections), len(self.detections), len(self.joint_samples)

    def detections_since(self, arrival_s):
        # Frames keep arriving after a goto returns, so the burst the run asks for
        # is produced when it asks, not banked in advance.
        def selected():
            return [entry for entry in self.detections if entry[1] >= arrival_s]

        for _ in range(40):
            if len(selected()) >= 16:
                break
            self.record(frames=1)
        return selected()

    def detections_between(self, first_s, last_s):
        return [entry for entry in self.detections if first_s <= entry[1] <= last_s]

    def joint_samples_between(self, first_s, last_s):
        return [entry for entry in self.joint_samples if first_s <= entry[1] <= last_s]

    def latest_joint_positions(self):
        return self.joint_positions.copy()

    def wait_until(self, predicate, timeout_s, description, poll_s=0.05):
        value = predicate()
        if not value:
            raise AssertionError(f"the fake should already satisfy: {description}")
        return value

    def transform(self, parent, child, timeout_s=5.0):
        if child == "fr3_link0":
            return np.eye(4)
        if child == "fr3_link8":
            return flange_transform(self.joint_positions)
        raise AssertionError(f"unexpected transform {parent} -> {child}")

    def stream_ages(self, duration_s=2.0):
        return {
            "image_age_s": 0.02,
            "joint_state_age_s": 0.01,
            "stamp_difference_s": -self.camera_delay_s,
            "joint_state_rate_hz": 30.0,
        }


class FakeReplay:
    def __init__(self, recorder):
        self.recorder = recorder
        self.targets = []

    def goto(self, joint_positions, timeout=None):
        self.targets.append(np.asarray(joint_positions, dtype=float))
        self.recorder.ramp(joint_positions)
        return {"command_id": len(self.targets)}

    def abort(self):
        pass


def options(tmp_path, seed_poses_path=None, **overrides):
    values = dict(
        tag_family="tag36h11",
        tag_id=0,
        tag_size_m=TAG_SIZE,
        world_frame="world",
        hand_frame="fr3_link8",
        base_frame="fr3_link0",
        camera_mount_frame=OPTICAL_FRAME,
        camera_optical_frame=OPTICAL_FRAME,
        image_topic="/camera/camera/color/image_raw",
        camera_info_topic="/camera/camera/color/camera_info",
        joint_state_topic="/joint_states",
        pose_count=20,
        minimum_samples=10,
        settle_seconds=0.0,
        burst_frames=8,
        burst_timeout_s=1.0,
        max_corner_std_px=0.35,
        max_joint_spread_rad=2.0e-4,
        max_reprojection_error_px=1.5,
        seed_poses_path=seed_poses_path,
        seed_result_path=None,
        program_path=None,
        output_dir=str(tmp_path),
        program_seed=5,
        offset_pass_poses=6,
        offset_pass_cycles=2,
        skip_offset_pass=False,
        return_home=True,
        dry_run=False,
        assume_yes=True,
        limits=PoseProgramLimits(),
    )
    values.update(overrides)
    return RunOptions(**values)


def write_seed_poses(path, count=8):
    """Seed poses a human would have taught: valid, reachable, tag in view."""
    import yaml

    poses = generate_pose_program(
        WORLD_TO_CAMERA, HAND_TO_TAG, CAMERA, tag_size_m=TAG_SIZE, count=count, seed=99
    )
    with open(path, "w") as handle:
        yaml.safe_dump(
            {
                "joint_names": JOINT_NAMES,
                "poses": [[float(v) for v in pose.joint_positions] for pose in poses],
            },
            handle,
        )
    return poses


@pytest.fixture
def run(tmp_path):
    seed_path = str(tmp_path / "seed_poses.yaml")
    write_seed_poses(seed_path)
    recorder = FakeRecorder(READY_POSE)
    replay = FakeReplay(recorder)
    run_dir = str(tmp_path / "run")
    runner = AutoCalibrationRunner(
        recorder, replay, options(tmp_path, seed_poses_path=seed_path), run_dir
    )
    return runner, recorder, replay, run_dir


def test_the_run_recovers_the_camera_the_tag_and_the_clock_delay(run):
    runner, recorder, replay, run_dir = run
    document = runner.run()

    recovered = np.asarray(document["transform"]["matrix_4x4"])
    assert np.linalg.norm(recovered[:3, 3] - WORLD_TO_CAMERA[:3, 3]) < 0.003
    assert rotation_angle_deg((invert_transform(recovered) @ WORLD_TO_CAMERA)[:3, :3]) < 0.3

    tag = np.asarray(document["estimated_carrier_to_tag"]["matrix_4x4"])
    assert np.linalg.norm(tag[:3, 3] - HAND_TO_TAG[:3, 3]) < 0.003

    # The images were stamped 40 ms early, so the run must report +40 ms.
    offset = document["time_offset"]
    assert offset["offset_s"] == pytest.approx(recorder.camera_delay_s, abs=0.004)
    assert offset["rms_at_zero_m"] > offset["rms_at_offset_m"]


def test_the_run_writes_a_readable_result_and_program(run):
    runner, _, _, run_dir = run
    runner.run()
    with open(os.path.join(run_dir, "result.json")) as handle:
        document = json.load(handle)
    assert document["schema_version"] == 2
    assert document["parent_frame"] == "world"
    assert document["quality"]["capture_mode"].startswith("automated")
    assert document["quality"]["translation_rmse_m"] < 0.005
    assert document["quality"]["max_forward_kinematics_disagreement_m"] < 1e-9
    for name in ("poses.json", "program.yaml", "offset_scan.json"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    with open(os.path.join(run_dir, "poses.json")) as handle:
        poses = json.load(handle)
    assert len(poses) >= 10
    assert all(entry["burst"]["frames_used"] >= 4 for entry in poses)


def test_every_pose_is_driven_and_the_arm_is_returned_home(run):
    runner, recorder, replay, _ = run
    runner.run()
    # 8 seed poses + 20 program poses + 12 offset legs + 1 home.
    assert len(replay.targets) == 8 + 20 + 12 + 1
    np.testing.assert_allclose(replay.targets[-1], READY_POSE, atol=1e-12)
    np.testing.assert_allclose(recorder.joint_positions, READY_POSE, atol=1e-12)


def test_a_pose_that_was_still_moving_is_discarded(run):
    runner, recorder, replay, run_dir = run
    # A burst is taken one ramp after the pose it belongs to, and eight taught
    # seed poses come first, so these are program poses 3 and 5.
    recorder.blur_poses = {12, 14}
    document = runner.run()
    assert document["quality"]["samples_collected"] == 18
    recovered = np.asarray(document["transform"]["matrix_4x4"])
    assert np.linalg.norm(recovered[:3, 3] - WORLD_TO_CAMERA[:3, 3]) < 0.003


def test_a_dry_run_writes_the_program_without_moving(tmp_path):
    seed_path = str(tmp_path / "seed_poses.yaml")
    write_seed_poses(seed_path)
    recorder = FakeRecorder(READY_POSE)
    replay = FakeReplay(recorder)
    run_dir = str(tmp_path / "run")
    runner = AutoCalibrationRunner(
        recorder,
        replay,
        options(tmp_path, seed_poses_path=seed_path, dry_run=True, seed_result_path=None),
        run_dir,
    )
    runner.run()
    # The coarse pass still drives the taught poses; nothing beyond it does.
    assert len(replay.targets) == 8
    assert os.path.exists(os.path.join(run_dir, "program.yaml"))
    assert not os.path.exists(os.path.join(run_dir, "result.json"))


def test_a_previous_result_can_replace_the_coarse_pass(tmp_path, run):
    runner, _, _, run_dir = run
    runner.run()
    seed_result = os.path.join(run_dir, "result.json")

    recorder = FakeRecorder(READY_POSE)
    replay = FakeReplay(recorder)
    second_dir = str(tmp_path / "second")
    second = AutoCalibrationRunner(
        recorder,
        replay,
        options(tmp_path, seed_result_path=seed_result, skip_offset_pass=True),
        second_dir,
    )
    document = second.run()
    # No taught poses were driven this time: 20 program poses plus home.
    assert len(replay.targets) == 21
    recovered = np.asarray(document["transform"]["matrix_4x4"])
    assert np.linalg.norm(recovered[:3, 3] - WORLD_TO_CAMERA[:3, 3]) < 0.003
    assert "time_offset" not in document


def test_a_stored_program_is_driven_without_a_coarse_pass(tmp_path, run):
    runner, _, _, run_dir = run
    runner.run()
    program = os.path.join(run_dir, "program.yaml")

    recorder = FakeRecorder(READY_POSE)
    replay = FakeReplay(recorder)
    second = AutoCalibrationRunner(
        recorder,
        replay,
        options(tmp_path, program_path=program, skip_offset_pass=True),
        str(tmp_path / "third"),
    )
    document = second.run()
    # No seed poses and no generation: 20 stored poses plus home.
    assert len(replay.targets) == 21
    recovered = np.asarray(document["transform"]["matrix_4x4"])
    assert np.linalg.norm(recovered[:3, 3] - WORLD_TO_CAMERA[:3, 3]) < 0.003


def test_a_program_for_different_joints_is_refused(tmp_path, run):
    import yaml

    from camera_calibration.auto_calibration import RunAborted

    runner, _, _, _ = run
    path = tmp_path / "foreign.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump(
            {
                "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                "poses": [[0.0] * 7] * 20,
            },
            handle,
        )
    with pytest.raises(RunAborted, match="generated for joints"):
        runner.load_program(str(path))


def test_a_program_with_too_few_poses_is_refused(tmp_path, run):
    import yaml

    from camera_calibration.auto_calibration import RunAborted

    runner, _, _, _ = run
    path = tmp_path / "short.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump({"joint_names": JOINT_NAMES, "poses": [[0.0] * 7] * 3}, handle)
    with pytest.raises(RunAborted, match="fewer than"):
        runner.load_program(str(path))
