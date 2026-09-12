import numpy as np
import pytest
import json

from inspire_franka_trajectory_replay.trajectory import (
    ARM_JOINTS, FINGER_FLEXION_INDICES, HAND_JOINTS, load_trajectory, resample,
    THUMB_ABDUCTION_DOF, THUMB_ABDUCTION_INDEX,
    THUMB_ABDUCTION_ZERO_OPEN_RATIO, scale_thumb_abduction,
    scale_finger_flexion,
)
from inspire_hand_driver import command_overlays
from inspire_hand_driver import kinematics as hand_kinematics


def test_finger_flexion_scale_changes_only_index_and_thumb_mcp_waypoints():
    hand = np.arange(18, dtype=float).reshape(3, 6) / 20.0
    original = hand.copy()

    scaled = scale_finger_flexion(hand, 1.3)

    flexion = list(FINGER_FLEXION_INDICES)
    untouched = [index for index in range(6) if index not in flexion]
    assert scaled[:, flexion] == pytest.approx(original[:, flexion] * 1.3)
    assert scaled[:, untouched] == pytest.approx(original[:, untouched])
    assert hand == pytest.approx(original)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_finger_flexion_scale_rejects_nonpositive_or_nonfinite_values(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        scale_finger_flexion(np.zeros((2, 6)), scale)


def test_finger_flexion_scale_requires_recorded_hand_positions():
    with pytest.raises(ValueError, match="requires Inspire hand positions"):
        scale_finger_flexion(None, 1.3)


def test_thumb_abduction_overlay_rescales_only_thumb_yaw_onto_the_usable_travel():
    hand = np.arange(24, dtype=float).reshape(4, 6) / 20.0
    original = hand.copy()
    open_pose_radians = hand_kinematics.DOFS[THUMB_ABDUCTION_DOF].lower

    scaled = scale_thumb_abduction(hand)

    untouched = [index for index in range(6) if index != THUMB_ABDUCTION_INDEX]
    assert scaled[:, untouched] == pytest.approx(original[:, untouched])
    assert scaled[:, THUMB_ABDUCTION_INDEX] == pytest.approx(
        open_pose_radians
        + (1.0 - THUMB_ABDUCTION_ZERO_OPEN_RATIO)
        * (original[:, THUMB_ABDUCTION_INDEX] - open_pose_radians)
    )
    assert all(
        hand_kinematics.rad_to_open_ratio(THUMB_ABDUCTION_DOF, radians)
        >= THUMB_ABDUCTION_ZERO_OPEN_RATIO
        for radians in scaled[:, THUMB_ABDUCTION_INDEX]
    )
    assert hand == pytest.approx(original)


def test_thumb_abduction_overlay_maps_the_command_range_onto_the_driver_range():
    """The radian contraction must agree with the driver's open-ratio rescale."""
    for commanded_ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        hand = np.zeros(6)
        hand[THUMB_ABDUCTION_INDEX] = hand_kinematics.open_ratio_to_rad(
            THUMB_ABDUCTION_DOF, commanded_ratio
        )

        scaled = scale_thumb_abduction(hand)

        expected_ratio = command_overlays.apply_open_ratio_overlay(
            THUMB_ABDUCTION_DOF, commanded_ratio
        )
        assert scaled[THUMB_ABDUCTION_INDEX] == pytest.approx(
            hand_kinematics.open_ratio_to_rad(THUMB_ABDUCTION_DOF, expected_ratio)
        )


def test_thumb_abduction_overlay_is_monotonic_and_loses_no_distinct_commands():
    """A floor collapsed the bottom of the range; the rescale must not."""
    commanded = np.linspace(0.0, 1.0, 9)
    hand = np.zeros((commanded.size, 6))
    hand[:, THUMB_ABDUCTION_INDEX] = [
        hand_kinematics.open_ratio_to_rad(THUMB_ABDUCTION_DOF, ratio)
        for ratio in commanded
    ]

    yaw = scale_thumb_abduction(hand)[:, THUMB_ABDUCTION_INDEX]

    assert np.all(np.diff(yaw) < 0.0)
    assert np.unique(yaw).size == commanded.size


def test_thumb_abduction_overlay_preserves_invalid_negative_radians_for_validation():
    hand = np.zeros((1, 6))
    hand[0, THUMB_ABDUCTION_INDEX] = -0.1

    scaled = scale_thumb_abduction(hand)[0, THUMB_ABDUCTION_INDEX]

    # Contracted, not clipped: _validate_hand must still see it as out of range.
    assert scaled < hand_kinematics.DOFS[THUMB_ABDUCTION_DOF].lower
    assert scaled == pytest.approx(-0.1 * (1.0 - THUMB_ABDUCTION_ZERO_OPEN_RATIO))


def test_thumb_abduction_overlay_also_rescales_a_one_dimensional_homing_pose():
    home = np.arange(6, dtype=float) / 10.0
    home[THUMB_ABDUCTION_INDEX] = 1.2

    scaled = scale_thumb_abduction(home)

    open_pose_radians = hand_kinematics.DOFS[THUMB_ABDUCTION_DOF].lower
    expected = open_pose_radians + (1.0 - THUMB_ABDUCTION_ZERO_OPEN_RATIO) * (
        1.2 - open_pose_radians
    )
    assert scaled[THUMB_ABDUCTION_INDEX] == pytest.approx(expected)
    assert np.delete(scaled, THUMB_ABDUCTION_INDEX) == pytest.approx(
        np.delete(home, THUMB_ABDUCTION_INDEX)
    )


@pytest.mark.parametrize("zero", [-0.1, 1.1, np.nan, np.inf])
def test_thumb_abduction_overlay_rejects_invalid_zero_open_ratio(zero):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        scale_thumb_abduction(np.zeros((2, 6)), zero)


def test_thumb_abduction_overlay_requires_recorded_hand_positions():
    with pytest.raises(ValueError, match="requires Inspire hand positions"):
        scale_thumb_abduction(None)


def test_load_and_resample_arm_only(tmp_path):
    path = tmp_path / "arm.npz"
    np.savez(path, joint_pos_arm=np.zeros((3, 7)), dt=0.1,
             arm_joint_names=np.array(ARM_JOINTS))
    trajectory = resample(load_trajectory(str(path)), 10.0)
    assert trajectory.arm.shape == (3, 7)
    assert trajectory.hand is None
    assert np.allclose(trajectory.time, [0.0, 0.1, 0.2])


def test_load_coordinated_data_reorders_named_columns(tmp_path):
    path = tmp_path / "both.npz"
    np.savez(path, arm=np.tile(np.arange(7), (2, 1)), hand=np.tile(np.arange(6), (2, 1)),
             arm_joint_names=np.array(ARM_JOINTS[::-1]),
             hand_joint_names=np.array(HAND_JOINTS[::-1]), rate=10.0)
    trajectory = load_trajectory(str(path))
    assert trajectory.arm[0].tolist() == list(reversed(range(7)))
    assert trajectory.hand[0].tolist() == list(reversed(range(6)))


def test_passive_hand_joint_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, joint_pos_arm=np.zeros((2, 7)), joint_pos_hand=np.zeros((2, 6)),
             hand_joint_names=np.array(["thumb_distal_joint"] * 6), dt=0.1)
    with pytest.raises(ValueError, match="passive"):
        load_trajectory(str(path))


def test_loads_forge_replay_directory_and_maps_driven_hand(tmp_path):
    trajectory_dir = tmp_path / "traj_1"
    trajectory_dir.mkdir()
    names = list(ARM_JOINTS) + [
        "index_joint_0", "little_joint_0", "middle_joint_0", "ring_joint_0",
        "thumb_joint_0", "index_joint_1", "little_joint_1", "middle_joint_1",
        "ring_joint_1", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    # Scaled to milliradians: the column mapping under test does not care, and
    # a raw 0..37 ramp would be a 19 rad step between two samples, which the
    # loader now correctly reads as a reset teleport rather than as motion.
    positions = np.arange(2 * 19, dtype=float).reshape(2, 1, 19) * 1e-3
    np.savez(
        trajectory_dir / "replay_data.npz",
        joint_pos=positions,
        sample_time_s=np.array([0.0, 0.1]),
    )
    (trajectory_dir / "metadata.json").write_text(
        json.dumps({"joint_names": names, "data_file": "replay_data.npz"})
    )

    trajectory = load_trajectory(str(trajectory_dir))

    assert trajectory.arm.shape == (2, 7)
    assert np.allclose(trajectory.arm[0], np.arange(7) * 1e-3)
    assert np.allclose(trajectory.hand[0], np.array([8, 10, 9, 7, 16, 11]) * 1e-3)


def test_multi_cycle_forge_recording_requires_and_applies_selection(tmp_path):
    trajectory_dir = tmp_path / "traj_1"
    trajectory_dir.mkdir()
    names = list(ARM_JOINTS) + [
        "index_joint_0", "little_joint_0", "middle_joint_0", "ring_joint_0",
        "thumb_joint_0", "index_joint_1", "little_joint_1", "middle_joint_1",
        "ring_joint_1", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    positions = np.arange(4 * 19, dtype=float).reshape(4, 1, 19) * 1e-3
    np.savez(
        trajectory_dir / "replay_data.npz",
        joint_pos=positions,
        sample_time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        cycle=np.array([1, 1, 2, 2]),
    )
    (trajectory_dir / "metadata.json").write_text(
        json.dumps({"joint_names": names, "data_file": "replay_data.npz"})
    )

    with pytest.raises(ValueError, match="--cycle N"):
        load_trajectory(str(trajectory_dir))

    trajectory = load_trajectory(str(trajectory_dir), cycle=2)
    assert trajectory.cycle == 2
    assert np.allclose(trajectory.arm[:, 0], [38e-3, 57e-3])
    assert np.allclose(trajectory.time, [0.0, 0.1])


def _forge(tmp_path, positions, time_s, **extra):
    """Write a minimal Forge replay directory and return its path."""
    trajectory_dir = tmp_path / "traj"
    trajectory_dir.mkdir()
    names = list(ARM_JOINTS) + [
        "index_joint_0", "little_joint_0", "middle_joint_0", "ring_joint_0",
        "thumb_joint_0", "index_joint_1", "little_joint_1", "middle_joint_1",
        "ring_joint_1", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    np.savez(
        trajectory_dir / "replay_data.npz",
        joint_pos=positions,
        sample_time_s=np.asarray(time_s, dtype=float),
        **extra,
    )
    (trajectory_dir / "metadata.json").write_text(
        json.dumps({"joint_names": names, "data_file": "replay_data.npz"})
    )
    return str(trajectory_dir)


def _episodes(lengths, dt=1 / 15.0, teleport=2.0):
    """Concatenate ``lengths`` slow episodes, each starting with a teleport."""
    rows = []
    for index, length in enumerate(lengths):
        # Well inside every FR3 velocity limit at 15 Hz, so nothing inside an
        # episode is ever mistaken for a reset.
        creep = np.arange(length, dtype=float)[:, None] * 0.001
        rows.append(np.tile(index * teleport + creep, (1, 19)))
    positions = np.vstack(rows)[:, None, :]
    return positions, np.arange(len(positions), dtype=float) * dt


def test_continuous_forge_recording_needs_no_segment_selection(tmp_path):
    positions, time_s = _episodes([40])
    trajectory = load_trajectory(_forge(tmp_path, positions, time_s))
    assert trajectory.segment is None
    assert len(trajectory.arm) == 40


def test_reset_teleport_splits_a_recording_and_selection_is_required(tmp_path):
    positions, time_s = _episodes([30, 40, 50])
    path = _forge(tmp_path, positions, time_s)

    with pytest.raises(ValueError, match="--segment N"):
        load_trajectory(path)

    trajectory = load_trajectory(path, segment=1)
    assert trajectory.segment == 1
    assert len(trajectory.arm) == 40
    # The middle episode, so its own first sample and not the recording's.
    assert trajectory.arm[0, 0] == pytest.approx(2.0)
    assert np.allclose(trajectory.time, np.arange(40) / 15.0)


def test_fragment_too_short_to_spline_is_listed_but_not_offered(tmp_path):
    positions, time_s = _episodes([30, 2])
    path = _forge(tmp_path, positions, time_s)

    # One usable episode plus a 2-sample tail: usable without selection.
    trajectory = load_trajectory(path)
    assert len(trajectory.arm) == 30

    with pytest.raises(ValueError, match="not a usable episode"):
        load_trajectory(path, segment=1)


def test_a_step_the_arm_could_make_is_not_read_as_a_reset(tmp_path):
    # 0.17 rad in one 15 Hz sample is just under joint 1's 2.62 rad/s limit,
    # so it is fast but real motion and must not split the recording.
    positions = np.zeros((20, 1, 19))
    positions[10:, 0, :] = 0.17
    time_s = np.arange(20, dtype=float) / 15.0
    trajectory = load_trajectory(_forge(tmp_path, positions, time_s))
    assert len(trajectory.arm) == 20


def test_cycle_selection_still_applies_before_segmentation(tmp_path):
    positions, time_s = _episodes([20, 20])
    path = _forge(tmp_path, positions, time_s, cycle=np.array([1] * 20 + [2] * 20))
    trajectory = load_trajectory(path, cycle=2)
    assert trajectory.cycle == 2
    assert len(trajectory.arm) == 20
