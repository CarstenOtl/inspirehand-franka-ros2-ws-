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


def test_optional_joint5_cap_changes_only_saturated_joint5_waypoints(tmp_path):
    directory = _capture(tmp_path)
    with np.load(directory / "replay_data.npz") as data:
        fields = dict(data)
    fields["joint_pos"][:, 0, 4] += 2.75
    np.savez(directory / "replay_data.npz", **fields)

    _, uncapped_arm, uncapped_hand, _ = build(
        directory, directory / "home.yaml", 1, 3, 2.0, 0
    )
    _, capped_arm, capped_hand, report = build(
        directory, directory / "home.yaml", 1, 3, 2.0, 0, joint5_cap=2.8
    )

    cycle_samples = report["cycle_samples"]
    assert np.all(capped_arm[:cycle_samples, 4] <= 2.8)
    assert np.allclose(
        capped_arm[:cycle_samples, :4], uncapped_arm[:cycle_samples, :4]
    )
    assert np.allclose(
        capped_arm[:cycle_samples, 5:], uncapped_arm[:cycle_samples, 5:]
    )
    assert np.allclose(capped_hand, uncapped_hand)
    assert report["joint5_capped_samples"] == 34
    assert report["joint5_recorded_max_rad"] == pytest.approx(2.868)
    assert report["joint5_maximum_change_rad"] == pytest.approx(0.068)


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


# --- the release flag the artifact has to carry -------------------------------------------


def _phased_capture(tmp_path, cycles=3, per_cycle=20, release_at=12, dt=1 / 15.0):
    """A capture whose samples carry ``replay_phase`` as a Forge rollout's do."""
    directory = _capture(tmp_path, cycles=cycles, per_cycle=per_cycle, dt=dt)
    data = dict(np.load(directory / "replay_data.npz", allow_pickle=False))
    phases = []
    for _ in range(cycles):
        phases.extend(["policy"] * release_at)
        phases.extend(["follow_waypoints"] * (per_cycle - release_at - 3))
        phases.extend(["return_to_reset"] * 3)
    data["replay_phase"] = np.array(phases)
    np.savez(directory / "replay_data.npz", **data)
    return directory


def test_the_output_metadata_carries_one_release_point_per_cycle(tmp_path):
    directory = _phased_capture(tmp_path)

    time, arm, hand, report = build(
        directory, directory / "home.yaml", first=1, count=3, seconds=1.0, environment=0
    )
    written = write(tmp_path / "out", time, arm, hand, report)
    metadata = json.loads((written / "metadata.json").read_text())

    assert metadata["release_phase"] == "follow_waypoints"
    assert [entry["cycle"] for entry in metadata["cycle_index"]] == [1, 2, 3]
    # Indices are in the *output's* numbering: cycles are taken whole and in
    # order, so cycle 2's release is 20 samples after cycle 1's.
    assert [entry["release_sample"] for entry in metadata["cycle_index"]] == [12, 32, 52]
    assert metadata["cycle_index"][0]["release_time_s"] == pytest.approx(12 / 15.0)


def test_the_flags_are_renumbered_when_the_selection_does_not_start_at_cycle_one(tmp_path):
    directory = _phased_capture(tmp_path, cycles=4)

    _, _, _, report = build(
        directory, directory / "home.yaml", first=3, count=2, seconds=1.0, environment=0
    )

    # Cycles 3 and 4 become samples 0-19 and 20-39 of the artifact.
    assert [entry["cycle"] for entry in report["cycle_index"]] == [3, 4]
    assert [entry["release_sample"] for entry in report["cycle_index"]] == [12, 32]


def test_the_flags_fall_inside_the_recorded_part_not_the_return_ramp(tmp_path):
    directory = _phased_capture(tmp_path, cycles=2)

    time, arm, hand, report = build(
        directory, directory / "home.yaml", first=1, count=2, seconds=1.0, environment=0
    )

    for entry in report["cycle_index"]:
        assert entry["release_sample"] < report["cycle_samples"]
    assert report["cycle_samples"] + report["ramp_samples"] == len(time)


def test_a_source_without_phases_produces_no_flags_rather_than_guessed_ones(tmp_path):
    directory = _capture(tmp_path, cycles=2)

    _, _, _, report = build(
        directory, directory / "home.yaml", first=1, count=2, seconds=1.0, environment=0
    )

    assert "cycle_index" not in report
    assert "release_phase" not in report
