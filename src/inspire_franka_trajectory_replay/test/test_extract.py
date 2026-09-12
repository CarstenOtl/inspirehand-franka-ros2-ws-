"""Extraction, checked without a bag: the conversions, and the artifact contract."""

import json

import numpy as np
import pytest

from inspire_franka_trajectory_replay.extract import (
    ARTIFACT_JOINT_NAMES,
    ROBOT_STATE_TOPIC,
    marker_indices,
    pose16_from_joint_angles,
    DRIVER_TO_ARTIFACT_JOINT,
    Extraction,
    build_arrays,
    central_difference,
    expand_followers,
    filter_report,
    hand_command_radians,
    lowpass,
    sample_at,
    tcp_from_pose16,
    write_artifact,
    _segment_window,
)
from inspire_franka_trajectory_replay.trajectory import (
    ARM_JOINTS,
    FORGE_HAND_JOINTS,
    HAND_JOINTS,
    load_trajectory,
)
from inspire_hand_driver import kinematics as kin


# -- the joint-name contract -------------------------------------------------


def test_the_artifact_layout_is_seven_arm_joints_then_twelve_hand_joints():
    assert len(ARTIFACT_JOINT_NAMES) == 19
    assert ARTIFACT_JOINT_NAMES[:7] == tuple(ARM_JOINTS)
    assert len(set(ARTIFACT_JOINT_NAMES)) == 19


def test_the_hand_name_map_covers_every_driver_joint_exactly_once():
    assert set(DRIVER_TO_ARTIFACT_JOINT) == set(kin.ALL_JOINTS)
    assert len(set(DRIVER_TO_ARTIFACT_JOINT.values())) == len(kin.ALL_JOINTS)
    assert set(DRIVER_TO_ARTIFACT_JOINT.values()) == set(ARTIFACT_JOINT_NAMES[7:])


def test_the_driven_six_agree_with_the_mapping_the_replay_loader_uses():
    """``trajectory.py`` reads FORGE_HAND_JOINTS[i] as HAND_JOINTS[i].

    If this table disagreed with that one, every extracted demonstration would
    replay with its fingers permuted, and nothing else would notice.
    """
    for driver_name, artifact_name in zip(HAND_JOINTS, FORGE_HAND_JOINTS):
        assert DRIVER_TO_ARTIFACT_JOINT[driver_name] == artifact_name


# -- hand coupling -----------------------------------------------------------


def test_followers_are_recomputed_exactly_as_the_driver_would():
    ratios = np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 0.9, 0.5, 0.3, 0.1, 0.0]])
    driven = np.array(
        [[kin.open_ratio_to_rad(i, r) for i, r in enumerate(row)] for row in ratios]
    )

    columns = expand_followers(driven)

    for index, row in enumerate(ratios):
        assert [columns[name][index] for name in kin.ALL_JOINTS] == pytest.approx(
            kin.joint_positions(row)
        )


# -- signal handling ---------------------------------------------------------


def test_the_low_pass_has_no_phase_lag():
    """filtfilt, not lfilter: an exactly symmetric input must stay symmetric."""
    t = (np.arange(2001) - 1000) / 1000.0
    signal = np.exp(-(t ** 2) / 0.02)[:, None]

    filtered = lowpass(signal, 1000.0, 5.0)[:, 0]

    assert filtered == pytest.approx(filtered[::-1], abs=1e-7)  # peak is 1.0
    assert int(np.argmax(filtered)) == 1000


def test_the_low_pass_removes_tremor_and_passes_the_motion():
    t = np.arange(4000) / 1000.0
    motion = np.sin(2 * np.pi * 0.3 * t)
    tremor = 0.05 * np.sin(2 * np.pi * 12.0 * t)

    filtered = lowpass((motion + tremor)[:, None], 1000.0, 2.0)[:, 0]

    interior = slice(200, -200)
    assert np.max(np.abs(filtered[interior] - motion[interior])) < 0.1 * np.max(np.abs(tremor))


def test_the_low_pass_leaves_no_transient_where_the_homing_pose_is_read():
    """Sample 0 becomes homing.yaml, so an edge transient there is not cosmetic.

    scipy's default padding leaves about 18 mrad here; the settling-time padding
    this module uses brings it to a few tens of microradians.
    """
    t = np.arange(4000) / 1000.0
    ramp = (0.4 * t + 0.05 * np.sin(2 * np.pi * 15.0 * t))[:, None]

    filtered = lowpass(ramp, 1000.0, 2.0)[:, 0]

    assert abs(filtered[0]) < 1e-3


def test_a_zero_cutoff_leaves_the_signal_alone():
    values = np.random.default_rng(0).normal(size=(50, 7))
    assert lowpass(values, 1000.0, 0.0) == pytest.approx(values)


def test_a_cutoff_at_or_above_nyquist_is_refused():
    with pytest.raises(ValueError, match="Nyquist"):
        lowpass(np.zeros((100, 7)), 30.0, 15.0)


def test_central_difference_recovers_a_constant_slope():
    dt = 1.0 / 15.0
    ramp = (np.arange(60) * dt)[:, None] * np.array([2.0, -3.0])
    assert central_difference(ramp, dt) == pytest.approx(
        np.tile([2.0, -3.0], (60, 1))
    )


def test_sample_at_interpolates_each_column_independently():
    source = np.array([0.0, 1.0, 2.0])
    values = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    assert sample_at(source, values, np.array([0.5, 1.5])).tolist() == pytest.approx(
        np.array([[0.5, 15.0], [1.5, 25.0]])
    )


# -- TCP ---------------------------------------------------------------------


def test_the_tcp_quaternion_is_wxyz_and_the_tool_offset_is_applied():
    pose = np.eye(4).reshape(16, order="F")[None, :]
    tool = np.eye(4)
    tool[:3, 3] = [0.0, 0.0, 0.1]

    position, quaternion = tcp_from_pose16(pose, tool)

    assert position[0] == pytest.approx([0.0, 0.0, 0.1])
    # Identity rotation is w = 1 in WXYZ, and would be the last element in XYZW.
    assert quaternion[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_the_tcp_pose_is_read_column_major_as_libfranka_packs_it():
    transform = np.eye(4)
    transform[:3, 3] = [0.3, -0.2, 0.5]

    position, _ = tcp_from_pose16(transform.reshape(16, order="F")[None, :], np.eye(4))

    assert position[0] == pytest.approx([0.3, -0.2, 0.5])


# -- the action channel ------------------------------------------------------


def test_the_hand_target_holds_the_preset_in_force_and_says_when_there_was_none():
    grid = np.array([0, 100, 200, 300], dtype=np.int64)
    fallback = np.array([0.1] * 6)
    events = [
        {
            "event": "hand_command",
            "t_ros_ns": 150,
            "open_ratio_rad": {name: 0.5 for name in kin.DRIVEN_JOINTS},
        }
    ]

    target, commanded = hand_command_radians(events, grid, fallback)

    assert commanded.tolist() == [False, False, True, True]
    assert target[0] == pytest.approx(fallback)
    assert target[1] == pytest.approx(fallback)
    assert target[2] == pytest.approx([0.5] * 6)


def test_a_session_with_no_hand_command_falls_back_without_inventing_one():
    grid = np.array([0, 1], dtype=np.int64)
    fallback = np.array([0.2] * 6)

    target, commanded = hand_command_radians([], grid, fallback)

    assert not commanded.any()
    assert target == pytest.approx(np.tile(fallback, (2, 1)))


# -- segments ----------------------------------------------------------------


def test_segment_marks_split_the_session():
    events = [{"event": "segment", "t_ros_ns": 50}, {"event": "segment", "t_ros_ns": 80}]

    assert _segment_window(events, 0, 100, 0)[:2] == (0, 50)
    assert _segment_window(events, 0, 100, 1)[:2] == (50, 80)
    assert _segment_window(events, 0, 100, 2)[:2] == (80, 100)
    assert _segment_window(events, 0, 100, None)[:2] == (0, 100)


def test_selecting_a_segment_that_was_never_marked_is_an_error():
    with pytest.raises(ValueError, match="does not exist"):
        _segment_window([], 0, 100, 3)


# -- the artifact ------------------------------------------------------------


def _extraction(count=150, rate=15.0):
    """A calm synthetic demonstration: slow, smooth, and inside every limit."""
    t = np.arange(count) / rate
    arm = np.column_stack(
        [0.2 * np.sin(2 * np.pi * 0.1 * t + phase) for phase in np.linspace(0, 1.2, 7)]
    )
    arm[:, 3] -= 2.0  # joint 4 lives in [-3.08, -0.12]
    arm[:, 5] += 1.8  # joint 6 lives in [0.44, 4.62]
    driven = np.column_stack(
        [np.full(count, kin.open_ratio_to_rad(index, 0.6)) for index in range(6)]
    )
    pose = np.tile(np.eye(4).reshape(16, order="F"), (count, 1))
    return Extraction(
        time=t,
        arm=arm,
        arm_raw=arm + 1e-4,
        hand_driven=driven,
        hand_target=driven,
        hand_commanded=np.ones(count, dtype=bool),
        tcp_pos=np.zeros((count, 3)),
        tcp_quat=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        captures=np.array([10, 50, 100], dtype=np.int64),
        grid_ns=(t * 1e9).astype(np.int64),
        segment=None,
        segment_count=1,
    )


def test_build_arrays_produces_the_time_environment_component_layout():
    extraction = _extraction()

    arrays = build_arrays(extraction, 15.0)

    assert arrays["joint_pos"].shape == (150, 1, 19)
    assert arrays["joint_vel"].shape == (150, 1, 19)
    assert arrays["joint_pos_target"].shape == (150, 1, 19)
    assert arrays["tcp_pos"].shape == (150, 1, 3)
    assert arrays["tcp_quat"].shape == (150, 1, 4)
    assert arrays["joint_pos"][:, 0, :7] == pytest.approx(extraction.arm)


def test_the_arm_block_of_joint_pos_target_is_the_capture_itself():
    """A hand-guided demonstration has no arm command; the capture is the reference."""
    arrays = build_arrays(_extraction(), 15.0)
    assert arrays["joint_pos_target"][:, 0, :7] == pytest.approx(arrays["joint_pos"][:, 0, :7])


def test_a_written_artifact_is_loadable_by_the_replay_stack_unmodified(tmp_path):
    """The contract that matters: replay_trajectory's own loader accepts it."""
    extraction = _extraction()
    context = {"manifest": {}, "session_dir": str(tmp_path), "bag_dir": str(tmp_path / "bag"),
               "arm_rate_hz": 1000.0, "arm_samples": 10000, "hand_samples": 500}

    write_artifact(tmp_path / "artifact", extraction, context, 15.0, 2.0)
    trajectory = load_trajectory(str(tmp_path / "artifact"))

    assert trajectory.arm.shape == (150, 7)
    assert trajectory.hand.shape == (150, 6)
    assert trajectory.arm == pytest.approx(extraction.arm)
    assert np.all(np.diff(trajectory.time) > 0)


def test_the_homing_pose_is_the_first_waypoint_so_the_two_cannot_disagree(tmp_path):
    import yaml

    from inspire_franka_trajectory_replay.replay import load_home

    extraction = _extraction()
    context = {"manifest": {}, "session_dir": str(tmp_path), "bag_dir": "", "arm_rate_hz": 1000.0,
               "arm_samples": 1, "hand_samples": 1}
    write_artifact(tmp_path / "artifact", extraction, context, 15.0, 2.0)

    home_arm, home_hand = load_home(tmp_path / "artifact" / "homing.yaml")

    assert home_arm == pytest.approx(extraction.arm[0])
    assert home_hand == pytest.approx(extraction.hand_driven[0])
    document = yaml.safe_load((tmp_path / "artifact" / "homing.yaml").read_text())
    assert document["source"]["joint7_offset_rad"] == 0.0


def test_the_metadata_marks_the_provenance_and_the_joint_seven_contract(tmp_path):
    context = {"manifest": {"note": "pick the block"}, "session_dir": "/logs/x",
               "bag_dir": "/logs/x/bag", "arm_rate_hz": 1000.0,
               "arm_samples": 1, "hand_samples": 1}

    write_artifact(tmp_path / "artifact", _extraction(), context, 15.0, 2.0)
    metadata = json.loads((tmp_path / "artifact" / "metadata.json").read_text())

    assert metadata["replay"] == "hand_guided"
    assert metadata["source"]["session_dir"] == "/logs/x"
    assert metadata["hardware_orientation"]["offset_rad"] == 0.0
    assert metadata["recording_frequency_hz"] == 15.0
    assert metadata["dt"] == pytest.approx(1.0 / 15.0)
    assert metadata["joint_names"] == list(ARTIFACT_JOINT_NAMES)
    assert metadata["filter"]["cutoff_hz"] == 2.0


def test_the_extracted_trajectory_passes_the_fr3_limit_guards(tmp_path):
    """End to end through the real preparation, including the position-dependent guard."""
    from franka_trajectory_replay.prepare import prepare
    from franka_trajectory_replay.trajectory_io import Trajectory

    extraction = _extraction()
    context = {"manifest": {}, "session_dir": str(tmp_path), "bag_dir": "", "arm_rate_hz": 1000.0,
               "arm_samples": 1, "hand_samples": 1}
    write_artifact(tmp_path / "artifact", extraction, context, 15.0, 2.0)
    trajectory = load_trajectory(str(tmp_path / "artifact"))

    prepared = prepare(
        Trajectory(t=trajectory.time, q=trajectory.arm, source="test"),
        rate=1000, auto_scale=False, velocity_margin=0.8,
        acceleration_margin=0.5, jerk_margin=0.5,
    )

    assert prepared.report["ok"], prepared.report["violations"]


def test_the_filter_report_accounts_for_what_the_filter_changed():
    extraction = _extraction()

    report = filter_report(extraction, 15.0, 2.0)

    assert report["cutoff_hz"] == 2.0
    assert [entry["joint"] for entry in report["joints"]] == list(ARM_JOINTS)
    assert all(entry["filter_max_rad"] == pytest.approx(1e-4) for entry in report["joints"])


def test_operator_marks_snap_to_the_nearest_emitted_sample():
    grid = np.array([0, 100, 200, 300], dtype=np.int64)
    events = [
        {"event": "pose_capture", "t_ros_ns": 140},
        {"event": "pose_capture", "t_ros_ns": 210},
        {"event": "pose_capture", "t_ros_ns": 9999},  # outside the window
    ]

    assert marker_indices(events, "pose_capture", grid, 0, 300).tolist() == [1, 2]


def test_pose_captures_are_the_only_marker_the_artifact_carries():
    arrays = build_arrays(_extraction(), 15.0)
    assert arrays["pose_capture_index"].tolist() == [10, 50, 100]
    assert "waypoint_index" not in arrays


# -- the simulated fallback --------------------------------------------------


def test_forward_kinematics_packs_the_pose_the_way_libfranka_does():
    """The FK fallback has to be readable by the same column-major unpacking."""
    from franka_trajectory_replay.kinematics import flange_transform

    q = np.array([[0.0, -0.5, 0.0, -2.2, 0.0, 1.9, 0.75]])

    pose16 = pose16_from_joint_angles(q)
    position, _ = tcp_from_pose16(pose16, np.eye(4))

    assert pose16.shape == (1, 16)
    assert position[0] == pytest.approx(flange_transform(q[0])[:3, 3])


def test_the_fk_fallback_tracks_the_joint_angles_it_was_given():
    q = np.array([
        [0.0, -0.5, 0.0, -2.2, 0.0, 1.9, 0.75],
        [0.2, -0.5, 0.0, -2.2, 0.0, 1.9, 0.75],
    ])

    position, quaternion = tcp_from_pose16(pose16_from_joint_angles(q), np.eye(4))

    assert not np.allclose(position[0], position[1])
    assert np.linalg.norm(quaternion, axis=1) == pytest.approx([1.0, 1.0])


def test_a_simulated_artifact_is_named_differently_from_a_hardware_one(tmp_path):
    """No amount of copying directories can turn a rehearsal into training data."""
    context = {"manifest": {}, "session_dir": str(tmp_path), "bag_dir": "",
               "arm_rate_hz": 1000.0, "arm_samples": 1, "hand_samples": 1,
               "simulated": True, "arm_state_topic": "/joint_states",
               "tcp_source": "forward kinematics of the recorded joint angles"}

    write_artifact(tmp_path / "artifact", _extraction(), context, 15.0, 2.0)
    metadata = json.loads((tmp_path / "artifact" / "metadata.json").read_text())

    assert metadata["replay"] == "hand_guided_sim"
    assert metadata["simulated"] is True
    assert metadata["source"]["tcp_source"].startswith("forward kinematics")
    assert any("not training data" in line for line in metadata["limitations"])


def test_a_hardware_artifact_carries_no_simulation_caveat(tmp_path):
    context = {"manifest": {}, "session_dir": str(tmp_path), "bag_dir": "",
               "arm_rate_hz": 1000.0, "arm_samples": 1, "hand_samples": 1,
               "simulated": False, "arm_state_topic": ROBOT_STATE_TOPIC,
               "tcp_source": "recorded O_T_EE"}

    write_artifact(tmp_path / "artifact", _extraction(), context, 15.0, 2.0)
    metadata = json.loads((tmp_path / "artifact" / "metadata.json").read_text())

    assert metadata["replay"] == "hand_guided"
    assert metadata["simulated"] is False
    assert metadata["limitations"] == []
