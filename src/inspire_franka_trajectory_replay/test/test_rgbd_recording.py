"""Full-resolution RGB-D recording: what is required, what is refused, what is written.

No camera and no ROS graph here. The stream checks take message stand-ins, the
recorder is faked, and the extractor is fed a fake bag, because what these
tests are about is the decisions: which stream is accepted, what the manifest
says, and that a frame read back is the frame that was recorded.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inspire_franka_trajectory_replay import rgbd_recording
from inspire_franka_trajectory_replay.rgbd_recording import (
    BAG_DIRECTORY,
    D415_FULL_COLOR,
    MANIFEST_NAME,
    RgbdRecording,
    StreamReport,
    StreamSpec,
    camera_info_topic_for,
    check_stream,
    depth_to_millimetres,
    extract_frames,
    extract_main,
    format_stream_report,
    image_to_numpy,
    locate_bag,
    note_rate,
    parse_resolution,
    recorded_topics,
)


# -- message stand-ins ---------------------------------------------------------


def _header(frame_id="camera_color_optical_frame", sec=12, nanosec=34):
    return SimpleNamespace(frame_id=frame_id, stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def _image(width, height, encoding, frame_id="camera_color_optical_frame", data=None, step=None):
    bytes_per_pixel = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "16UC1": 2, "32FC1": 4}[encoding]
    step = width * bytes_per_pixel if step is None else step
    if data is None:
        data = bytes(height * step)
    return SimpleNamespace(
        header=_header(frame_id), width=width, height=height, encoding=encoding,
        step=step, data=data,
    )


def _camera_info(width, height, frame_id="camera_color_optical_frame"):
    return SimpleNamespace(
        header=_header(frame_id), width=width, height=height,
        distortion_model="plumb_bob", d=[0.0] * 5,
        k=[600.0, 0.0, width / 2, 0.0, 600.0, height / 2, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[600.0, 0.0, width / 2, 0.0, 0.0, 600.0, height / 2, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def _rgbd(color=(1920, 1080), depth=None, depth_frame="camera_color_optical_frame",
          color_encoding="rgb8", depth_encoding="16UC1", rgb_data=None, depth_data=None):
    depth = color if depth is None else depth
    return SimpleNamespace(
        header=_header(),
        rgb_camera_info=_camera_info(*color),
        depth_camera_info=_camera_info(*depth, frame_id=depth_frame),
        rgb=_image(*color, color_encoding, data=rgb_data),
        depth=_image(*depth, depth_encoding, frame_id=depth_frame, data=depth_data),
    )


def _report(spec=StreamSpec()):
    return StreamReport(spec.rgbd_topic, spec.camera_info_topic, spec.width, spec.height)


# -- what is required ------------------------------------------------------------


def test_the_default_is_the_d415s_full_colour_resolution():
    spec = StreamSpec()
    assert (spec.width, spec.height) == D415_FULL_COLOR == (1920, 1080)
    assert spec.rgbd_topic == "/camera/camera/rgbd"
    assert spec.camera_info_topic == "/camera/camera/color/camera_info"


def test_resolution_is_parsed_strictly():
    assert parse_resolution("1920x1080") == (1920, 1080)
    assert parse_resolution(" 640X480 ") == (640, 480)
    for bad in ("1920", "1920x", "ax1080", "0x1080", "1920x1080x30"):
        with pytest.raises(ValueError, match="resolution must"):
            parse_resolution(bad)


def test_camera_info_topic_sits_beside_the_rgbd_topic():
    assert camera_info_topic_for("/camera/camera/rgbd") == "/camera/camera/color/camera_info"
    assert camera_info_topic_for("/d415/rgbd") == "/d415/color/camera_info"


def test_the_camera_topics_lead_and_the_robot_channels_follow_the_runs_shape():
    spec = StreamSpec()
    full = recorded_topics(spec)
    assert [entry.topic for entry in full[:2]] == [spec.rgbd_topic, spec.camera_info_topic]
    assert full[0].msg_type == "realsense2_camera_msgs/msg/RGBD"
    names = {entry.topic for entry in full}
    assert {"/joint_states", "/inspire_hand/joint_states", "/inspire_hand/command",
            "/tf", "/tf_static"} <= names

    hand_only = {entry.topic for entry in recorded_topics(spec, arm=False)}
    assert "/joint_states" not in hand_only
    assert "/inspire_hand/command" in hand_only

    arm_only = {entry.topic for entry in recorded_topics(spec, hand=False)}
    assert "/joint_states" in arm_only
    assert not any(topic.startswith("/inspire_hand") for topic in arm_only)


def test_every_recorded_topic_is_proven_and_explained():
    for entry in recorded_topics(StreamSpec(), hand_topic="/h/command", hand_state_topic="/h/js"):
        assert entry.check in ("live", "subscriber", "publisher")
        assert entry.why.strip()
        assert entry.msg_type.count("/") == 2, entry.topic


def test_the_command_channel_is_proven_by_its_subscriber_not_by_traffic():
    by_topic = {entry.topic: entry for entry in recorded_topics(StreamSpec())}
    assert by_topic["/inspire_hand/command"].check == "subscriber"
    assert by_topic["/camera/camera/rgbd"].check == "live"


# -- the preflight -----------------------------------------------------------------


def test_a_full_resolution_aligned_stream_is_accepted():
    report = check_stream(_report(), _rgbd(), _camera_info(1920, 1080))
    assert report.ok, report.problems
    assert (report.rgb_width, report.rgb_height) == (1920, 1080)
    assert report.depth_frame_id == report.rgb_frame_id


def test_the_policy_rollouts_640x480_stream_is_refused():
    report = check_stream(_report(), _rgbd(color=(640, 480)), _camera_info(640, 480))
    assert not report.ok
    assert any("640x480" in p and "1920x1080" in p for p in report.problems)


def test_unaligned_depth_is_refused_even_at_full_colour_resolution():
    report = check_stream(
        _report(), _rgbd(depth=(1280, 720), depth_frame="camera_depth_optical_frame"),
        _camera_info(1920, 1080),
    )
    assert not report.ok
    assert any("not aligned" in p and "1280x720" in p for p in report.problems)
    assert any("camera_depth_optical_frame" in p for p in report.problems)


def test_intrinsics_that_do_not_describe_the_image_are_refused():
    report = check_stream(_report(), _rgbd(), _camera_info(640, 480))
    assert any("CameraInfo is 640x480" in p for p in report.problems)
    report = check_stream(_report(), _rgbd(), _camera_info(1920, 1080, frame_id="other"))
    assert any("CameraInfo frame" in p for p in report.problems)


def test_encodings_the_extractor_cannot_decode_are_refused():
    report = check_stream(
        _report(), _rgbd(color_encoding="rgba8", depth_encoding="32FC1"), _camera_info(1920, 1080)
    )
    assert report.ok, report.problems
    odd = _rgbd()
    odd.rgb.encoding = "yuv422"
    odd.depth.encoding = "8UC1"
    report = check_stream(_report(), odd, _camera_info(1920, 1080))
    assert any("colour encoding 'yuv422'" in p for p in report.problems)
    assert any("depth encoding '8UC1'" in p for p in report.problems)


def test_every_problem_is_listed_not_just_the_first():
    report = check_stream(
        _report(), _rgbd(color=(640, 480), depth=(320, 240), depth_frame="d"),
        _camera_info(1280, 720, frame_id="x"),
    )
    assert len(report.problems) >= 4


def test_a_lower_resolution_can_be_required_deliberately():
    spec = StreamSpec(width=640, height=480)
    report = check_stream(_report(spec), _rgbd(color=(640, 480)), _camera_info(640, 480))
    assert report.ok


def test_a_slow_stream_is_a_warning_not_a_refusal():
    report = note_rate(_report(), frames=12, seconds=1.0)
    assert report.ok
    assert report.measured_rate_hz == pytest.approx(12.0)
    assert any("12.0 Hz" in w and "USB 3" in w for w in report.warnings)
    fine = note_rate(_report(), frames=29, seconds=1.0)
    assert not fine.warnings


def test_the_report_reads_as_a_verdict():
    report = check_stream(_report(), _rgbd(), _camera_info(1920, 1080))
    note_rate(report, 30, 1.0)
    text = format_stream_report(report)
    assert "colour  1920x1080 rgb8" in text
    assert "30.0 Hz" in text
    assert "required 1920x1080: OK" in text
    refused = check_stream(_report(), _rgbd(color=(640, 480)), _camera_info(640, 480))
    assert "REFUSED" in format_stream_report(refused)
    assert "! colour is 640x480" in format_stream_report(refused)


# -- the recording -----------------------------------------------------------------


_METADATA = """rosbag2_bagfile_information:
  version: 9
  storage_identifier: mcap
  topics_with_message_count:
    - topic_metadata:
        name: /camera/camera/rgbd
        type: realsense2_camera_msgs/msg/RGBD
      message_count: 87
    - topic_metadata:
        name: /camera/camera/color/camera_info
        type: sensor_msgs/msg/CameraInfo
      message_count: 90
"""


class _FakeRecorder:
    instances = []

    def __init__(self, bag_dir, topics, storage_id="sqlite3", logger=None, extra_args=()):
        self.bag_dir = Path(bag_dir)
        self.topics = list(topics)
        self.storage_id = storage_id
        self.extra_args = list(extra_args)
        self.started = self.stopped = False
        _FakeRecorder.instances.append(self)

    def start(self, timeout=20.0):
        self.bag_dir.mkdir(parents=True)
        (self.bag_dir / "metadata.yaml").write_text(_METADATA)
        self.started = True

    def stop(self, timeout=30.0):
        self.stopped = True


@pytest.fixture
def fake_recorder(monkeypatch):
    _FakeRecorder.instances = []
    monkeypatch.setattr(rgbd_recording, "BagRecorder", _FakeRecorder)
    return _FakeRecorder


def test_the_bag_is_written_through_ros2_bag_record_in_mcap_with_a_large_cache(
    tmp_path, fake_recorder
):
    spec = StreamSpec()
    recording = RgbdRecording(tmp_path / "run", spec, recorded_topics(spec))
    recording.start()
    recorder = fake_recorder.instances[-1]
    assert recorder.bag_dir == tmp_path / "run" / BAG_DIRECTORY
    assert recorder.storage_id == "mcap"
    assert recorder.topics[0] == spec.rgbd_topic
    assert recorder.extra_args == ["--max-cache-size", str(rgbd_recording.CACHE_BYTES)]
    assert recording.recording
    recording.stop()
    assert recorder.stopped and not recording.recording
    recording.stop()  # idempotent


def test_an_unknown_storage_plugin_is_refused_before_anything_starts(tmp_path):
    with pytest.raises(ValueError, match="storage"):
        RgbdRecording(tmp_path, StreamSpec(), [], storage_id="hdf5")


def test_the_manifest_describes_the_stream_the_topics_and_the_run(tmp_path, fake_recorder):
    spec = StreamSpec()
    report = check_stream(_report(spec), _rgbd(), _camera_info(1920, 1080))
    note_rate(report, 30, 1.0)
    recording = RgbdRecording(tmp_path / "run", spec, recorded_topics(spec, hand=False))
    recording.start()
    recording.stop()
    path = recording.write_manifest(report, {"exit_code": 0, "trajectory": "traj_2"})
    assert path == tmp_path / "run" / MANIFEST_NAME
    document = json.loads(path.read_text())
    assert document["kind"] == "rollout_rgbd"
    assert document["recorded"] is True
    assert document["bag_dir"] == BAG_DIRECTORY
    assert document["bag_storage_id"] == "mcap"
    assert document["rgbd_topic"] == spec.rgbd_topic
    assert document["required_resolution"] == {"width": 1920, "height": 1080}
    assert document["stream"]["ok"] is True
    assert document["stream"]["measured_rate_hz"] == pytest.approx(30.0)
    assert [t["topic"] for t in document["topics"]][:2] == [spec.rgbd_topic, spec.camera_info_topic]
    assert document["exit_code"] == 0 and document["trajectory"] == "traj_2"
    assert document["started_at"] and document["stopped_at"]
    assert document["duration_s"] >= 0.0
    # What is actually on disk, from the bag's own metadata, not the preflight's rate.
    assert document["rgbd_frames"] == 87
    assert document["bag_message_counts"]["/camera/camera/color/camera_info"] == 90


def test_the_take_is_described_by_frames_on_disk_against_the_run_length():
    from inspire_franka_trajectory_replay.rgbd_recording import bag_message_counts, describe_take

    spec = StreamSpec()
    assert describe_take({spec.rgbd_topic: 87}, spec, 3.0) == \
        "87 RGB-D frames over 3.0 s (29.0 Hz; 97% of 30 Hz)"
    assert describe_take({spec.rgbd_topic: 87}, spec, None) == "87 RGB-D frames recorded"
    assert "no /camera/camera/rgbd messages" in describe_take({}, spec, 3.0)
    assert bag_message_counts(Path("/nowhere")) == {}


def test_a_bag_whose_metadata_is_unreadable_still_gets_a_manifest(tmp_path, fake_recorder):
    from inspire_franka_trajectory_replay.rgbd_recording import bag_message_counts

    (tmp_path / "bag").mkdir()
    (tmp_path / "bag" / "metadata.yaml").write_text("not: [valid\n")
    assert bag_message_counts(tmp_path / "bag") == {}
    (tmp_path / "bag" / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    assert bag_message_counts(tmp_path / "bag") == {}


def test_a_run_that_never_recorded_still_writes_a_manifest_saying_so(tmp_path):
    recording = RgbdRecording(tmp_path / "run", StreamSpec(), [])
    recording.stop()  # nothing started; must not raise
    document = json.loads(recording.write_manifest(None, {"exit_code": 1}).read_text())
    assert document["recorded"] is False
    assert document["stream"] is None
    assert document["exit_code"] == 1


# -- reading it back ---------------------------------------------------------------


def test_colour_decodes_to_rgb_whatever_the_wire_order():
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rgb, units = image_to_numpy(_image(3, 2, "rgb8", data=pixels.tobytes()))
    assert units == "rgb" and np.array_equal(rgb, pixels)
    bgr, _ = image_to_numpy(_image(3, 2, "bgr8", data=pixels.tobytes()))
    assert np.array_equal(bgr, pixels[..., ::-1])


def test_depth_decodes_with_row_padding_and_in_either_unit():
    depth = np.array([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=np.uint16)
    padded = np.concatenate([depth, np.zeros((2, 1), dtype=np.uint16)], axis=1)
    millimetres, units = image_to_numpy(_image(3, 2, "16UC1", data=padded.tobytes(), step=8))
    assert units == "millimetres" and np.array_equal(millimetres, depth)
    metres = depth.astype(np.float32) / 1000.0
    decoded, units = image_to_numpy(_image(3, 2, "32FC1", data=metres.tobytes()))
    assert units == "metres" and np.allclose(decoded, metres)
    assert np.array_equal(depth_to_millimetres(decoded, units), depth)


def test_depth_in_metres_becomes_millimetres_with_holes_as_zero():
    metres = np.array([[0.5004, np.nan, np.inf, 70.0]], dtype=np.float32)
    assert depth_to_millimetres(metres, "metres").tolist() == [[500, 0, 0, 65535]]
    assert depth_to_millimetres(np.array([[7]], dtype=np.uint16), "millimetres").dtype == np.uint16


def _fake_bag(monkeypatch, frames):
    """Feed ``extract_frames`` these RGBD messages instead of reading a bag."""
    calls = []

    def read_messages(bag_dir, topics=None, storage_id=None):
        calls.append((bag_dir, list(topics or [])))
        for index, message in enumerate(frames):
            yield topics[0], message, 1_000_000_000 + index

    monkeypatch.setattr(rgbd_recording, "read_messages", read_messages)
    return calls


def _frame(index):
    rgb = np.full((4, 6, 3), index, dtype=np.uint8)
    depth = np.full((4, 6), 100 * (index + 1), dtype=np.uint16)
    message = _rgbd(color=(6, 4), rgb_data=rgb.tobytes(), depth_data=depth.tobytes())
    message.header.stamp.sec = 100 + index
    message.rgb.header.stamp.sec = 100 + index
    message.depth.header.stamp.sec = 100 + index
    return message


def test_frames_are_written_losslessly_with_their_stamps(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    import cv2

    calls = _fake_bag(monkeypatch, [_frame(i) for i in range(3)])
    written = extract_frames(tmp_path / "bag", "/camera/camera/rgbd", tmp_path / "frames")
    assert written == 3
    assert calls == [(str(tmp_path / "bag"), ["/camera/camera/rgbd"])]

    colour = cv2.imread(str(tmp_path / "frames" / "color" / "000002.png"), cv2.IMREAD_UNCHANGED)
    assert colour.shape == (4, 6, 3) and int(colour[0, 0, 0]) == 2
    depth = cv2.imread(str(tmp_path / "frames" / "depth" / "000002.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and int(depth[0, 0]) == 300

    with (tmp_path / "frames" / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["index"] for row in rows] == ["0", "1", "2"]
    assert rows[2]["header_stamp_ns"] == str(102 * 1_000_000_000 + 34)
    assert rows[2]["receive_ns"] == str(1_000_000_002)
    assert rows[2]["color_file"] == "color/000002.png"
    assert rows[2]["depth_units"] == "millimetres"

    info = json.loads((tmp_path / "frames" / "camera_info.json").read_text())
    assert info["width"] == 6 and info["frame_id"] == "camera_color_optical_frame"
    assert len(info["k"]) == 9


def test_every_and_limit_thin_the_extraction(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    _fake_bag(monkeypatch, [_frame(i) for i in range(10)])
    assert extract_frames(tmp_path, "/t", tmp_path / "a", every=3) == 4
    assert extract_frames(tmp_path, "/t", tmp_path / "b", every=3, limit=2) == 2
    with (tmp_path / "a" / "frames.csv").open() as stream:
        assert [row["message_index"] for row in csv.DictReader(stream)] == ["0", "3", "6", "9"]
    with pytest.raises(ValueError, match="every"):
        extract_frames(tmp_path, "/t", tmp_path / "c", every=0)


def test_the_bag_is_found_from_the_run_directory_its_manifest_or_the_bag_itself(tmp_path):
    run = tmp_path / "run"
    (run / BAG_DIRECTORY).mkdir(parents=True)
    (run / BAG_DIRECTORY / "metadata.yaml").write_text("{}\n")
    assert locate_bag(run) == (run / BAG_DIRECTORY, "/camera/camera/rgbd")
    assert locate_bag(run / BAG_DIRECTORY) == (run / BAG_DIRECTORY, "/camera/camera/rgbd")
    (run / MANIFEST_NAME).write_text(json.dumps({"bag_dir": BAG_DIRECTORY, "rgbd_topic": "/d415/rgbd"}))
    assert locate_bag(run) == (run / BAG_DIRECTORY, "/d415/rgbd")
    with pytest.raises(FileNotFoundError):
        locate_bag(tmp_path / "nowhere")


def test_extract_main_reports_a_missing_recording_and_a_bag_with_no_frames(
    tmp_path, monkeypatch, capsys
):
    assert extract_main([str(tmp_path / "nowhere")]) == 2
    assert "extract error" in capsys.readouterr().err
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("{}\n")
    _fake_bag(monkeypatch, [])
    assert extract_main([str(bag), "--output", str(tmp_path / "out")]) == 1
    assert "no frames" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        extract_main([str(bag), "--every", "0"])
