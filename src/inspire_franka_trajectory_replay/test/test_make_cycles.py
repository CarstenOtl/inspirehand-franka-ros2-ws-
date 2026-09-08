"""The N-cycle generator: what it concatenates, and how it gets back to home."""

import json

import numpy as np
import pytest

from inspire_franka_trajectory_replay.make_cycles import (
    build, return_to_home, select_cycles, write,
)
from inspire_franka_trajectory_replay.trajectory import (
    ARM_JOINTS, HAND_JOINTS, load_trajectory,
)

FORGE_NAMES = list(ARM_JOINTS) + [
    "index_joint_0", "little_joint_0", "middle_joint_0", "ring_joint_0",
    "thumb_joint_0", "index_joint_1", "little_joint_1", "middle_joint_1",
    "ring_joint_1", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
]
HOME = {
    "joint_names": list(ARM_JOINTS) + list(HAND_JOINTS),
    "positions": [0.0] * 7 + [0.2, 0.2, 0.2, 0.2, 0.1, 0.3],
}


def _capture(tmp_path, cycles=4, per_cycle=20, dt=1 / 15.0):
    """A capture whose cycles run back to back with no reset between them."""
    directory = tmp_path / "capture"
    directory.mkdir()
    count = cycles * per_cycle
    # A slow ramp shared by every joint: continuous across cycle boundaries,
    # which is what the threading captures actually look like.
    positions = np.tile(np.arange(count, dtype=float)[:, None] * 0.002, (1, 19))[:, None, :]
    np.savez(
        directory / "replay_data.npz",
        joint_pos=positions,
        sample_time_s=np.arange(count, dtype=float) * dt,
        cycle=np.repeat(np.arange(1, cycles + 1), per_cycle),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"joint_names": FORGE_NAMES, "data_file": "replay_data.npz"})
    )
    (directory / "home.yaml").write_text(json.dumps(HOME))
    return directory


def test_selects_the_requested_cycles_as_one_contiguous_block():
    cycle_field = np.repeat([1, 2, 3, 4], 5)
    assert select_cycles(cycle_field, 2, 2).tolist() == list(range(5, 15))


def test_cycles_outside_the_recording_are_named():
    with pytest.raises(ValueError, match=r"cycles \[6, 7, 8\]"):
        select_cycles(np.repeat([1, 2, 3], 5), 6, 3)


def test_ramp_lands_exactly_on_the_home_pose():
    """The whole point of the ramp: the arm and hand are parked at home."""
    dt = 1 / 15.0
    home_arm, home_hand = np.full(7, 0.5), np.full(6, 0.3)

    ramp_arm, ramp_hand = return_to_home(
        np.zeros((2, 7)), np.zeros((2, 6)), home_arm, home_hand, 2.0, dt
    )

    assert np.allclose(ramp_arm[-1], home_arm)
    assert np.allclose(ramp_hand[-1], home_hand)


@pytest.mark.parametrize("rate", [15.0, 150.0, 1500.0])
def test_the_seam_is_c1_and_the_arrival_is_at_rest(rate):
    """Both endpoint velocities are exact in the continuous curve, so their
    finite differences converge first-order in dt. A ramp that started from
    rest instead would leave a 0.15 rad/s step at the seam at every rate, and
    the departure error below would not shrink."""
    dt = 1.0 / rate
    arm = np.zeros((2, 7))
    arm[1, 0] = 0.15 * dt  # moving at 0.15 rad/s when the recording runs out

    ramp_arm, _ = return_to_home(
        arm, np.zeros((2, 6)), np.full(7, 0.5), np.zeros(6), 2.0, dt
    )

    departure = (ramp_arm[0, 0] - arm[-1, 0]) / dt
    arrival = np.abs(ramp_arm[-1] - ramp_arm[-2]).max() / dt
    # Both errors are O(dt), so scaling them by the rate bounds every case at once.
    assert abs(departure - 0.15) * rate < 0.5
    assert arrival * rate < 0.5


def test_a_ramp_shorter_than_two_samples_is_refused():
    with pytest.raises(ValueError, match="two samples"):
        return_to_home(np.zeros((2, 7)), np.zeros((2, 6)),
                       np.zeros(7), np.zeros(6), 0.05, 1 / 15.0)


def test_generated_file_replays_without_cycle_or_segment_selection(tmp_path):
    directory = _capture(tmp_path)
    time, arm, hand, report = build(directory, directory / "home.yaml", 1, 3, 2.0, 0)
    output = write(tmp_path / "out", time, arm, hand, report)

    trajectory = load_trajectory(str(output))

    assert trajectory.cycle is None and trajectory.segment is None
    assert len(trajectory.arm) == 3 * 20 + 30
    assert np.allclose(trajectory.arm[-1], 0.0)          # the home pose above
    assert np.allclose(trajectory.hand[-1], HOME["positions"][7:])
    assert report["cycles"] == [1, 2, 3]


def test_a_capture_with_a_reset_inside_the_selection_is_refused(tmp_path):
    directory = _capture(tmp_path)
    with np.load(directory / "replay_data.npz") as data:
        fields = dict(data)
    fields["joint_pos"][25:] += 2.0  # a teleport in the middle of cycle 2
    np.savez(directory / "replay_data.npz", **fields)

    with pytest.raises(ValueError, match="not one continuous run"):
        build(directory, directory / "home.yaml", 1, 3, 2.0, 0)


def test_pickplace_style_capture_is_refused_with_a_reason(tmp_path):
    directory = tmp_path / "pickplace"
    directory.mkdir()
    np.savez(
        directory / "replay_data.npz",
        joint_pos=np.zeros((10, 1, 19)),
        sample_time_s=np.arange(10) / 15.0,
    )
    (directory / "metadata.json").write_text(
        json.dumps({"joint_names": FORGE_NAMES, "data_file": "replay_data.npz"})
    )
    (directory / "home.yaml").write_text(json.dumps(HOME))

    with pytest.raises(ValueError, match="cycle field"):
        build(directory, directory / "home.yaml", 1, 2, 2.0, 0)
