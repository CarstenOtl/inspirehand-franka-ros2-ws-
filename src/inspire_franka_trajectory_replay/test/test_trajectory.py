import numpy as np
import pytest
import json

from inspire_franka_trajectory_replay.trajectory import (
    ARM_JOINTS, HAND_JOINTS, load_trajectory, resample,
)


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
    positions = np.arange(2 * 19, dtype=float).reshape(2, 1, 19)
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
    assert trajectory.arm[0].tolist() == list(range(7))
    assert trajectory.hand[0].tolist() == [8.0, 10.0, 9.0, 7.0, 16.0, 11.0]


def test_multi_cycle_forge_recording_requires_and_applies_selection(tmp_path):
    trajectory_dir = tmp_path / "traj_1"
    trajectory_dir.mkdir()
    names = list(ARM_JOINTS) + [
        "index_joint_0", "little_joint_0", "middle_joint_0", "ring_joint_0",
        "thumb_joint_0", "index_joint_1", "little_joint_1", "middle_joint_1",
        "ring_joint_1", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    positions = np.arange(4 * 19, dtype=float).reshape(4, 1, 19)
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
    assert trajectory.arm[:, 0].tolist() == [38.0, 57.0]
    assert np.allclose(trajectory.time, [0.0, 0.1])
