"""The release-phase flag: finding it, carrying it, and placing it on a clock."""

import json

import numpy as np
import pytest

from franka_trajectory_replay.kinematics import READY_POSE
from franka_trajectory_replay.runconfig import load_config
from inspire_franka_trajectory_replay import release_phase
from inspire_franka_trajectory_replay.release_phase import (
    RELEASE_PHASE,
    ReleaseIndex,
    from_metadata,
    prepared_time_for_sample,
    releases_from_fields,
    sample_for_prepared_time,
)
from inspire_franka_trajectory_replay.replay import _prepare_arm
from inspire_franka_trajectory_replay.trajectory import CoordinatedTrajectory


def _phases(per_cycle=(("policy", 10), ("follow_waypoints", 6), ("return_to_reset", 4)),
            cycles=3):
    """Per-sample cycle and phase fields shaped like a threading capture's."""
    cycle_field, phase_field = [], []
    for cycle in range(1, cycles + 1):
        for name, count in per_cycle:
            cycle_field.extend([cycle] * count)
            phase_field.extend([name] * count)
    return np.array(cycle_field), np.array(phase_field)


def test_release_is_the_first_sample_of_the_release_phase_in_each_cycle():
    cycle_field, phase_field = _phases()

    releases = releases_from_fields(cycle_field, phase_field)

    assert [entry.cycle for entry in releases] == [1, 2, 3]
    # Twenty samples per cycle, the release phase ten into each.
    assert [entry.release_sample for entry in releases] == [10, 30, 50]
    assert [entry.start_sample for entry in releases] == [0, 20, 40]
    assert [entry.end_sample for entry in releases] == [19, 39, 59]


def test_the_offset_renumbers_every_index_for_a_sliced_caller():
    cycle_field, phase_field = _phases(cycles=2)

    releases = releases_from_fields(cycle_field, phase_field, offset=100)

    assert [entry.release_sample for entry in releases] == [110, 130]
    assert releases[0].start_sample == 100


def test_a_cycle_that_never_releases_is_an_error_rather_than_a_cycle_without_one():
    cycle_field, phase_field = _phases(cycles=2)
    phase_field = np.where(phase_field == RELEASE_PHASE, "policy", phase_field)

    with pytest.raises(ValueError, match="never enters"):
        releases_from_fields(cycle_field, phase_field)


def test_a_cycle_split_across_the_file_is_refused():
    cycle_field = np.array([1, 1, 2, 2, 1, 1])
    phase_field = np.array([RELEASE_PHASE] * 6)

    with pytest.raises(ValueError, match="contiguous"):
        releases_from_fields(cycle_field, phase_field)


def test_mismatched_field_lengths_are_refused():
    with pytest.raises(ValueError, match="replay_phase has"):
        releases_from_fields(np.array([1, 1, 1]), np.array([RELEASE_PHASE]))


def test_the_containing_cycle_and_the_next_release_are_different_questions():
    index = ReleaseIndex(releases_from_fields(*_phases()), rate_hz=15.0)

    # Sample 12 is past cycle 1's release at 10: the cycle it is in has already
    # let go, so the next release to aim at is cycle 2's.
    assert index.cycle_for_sample(12).cycle == 1
    assert index.next_release_at_or_after(12).cycle == 2
    # Before the release of the cycle it is in, that release is the next one.
    assert index.next_release_at_or_after(5).cycle == 1
    assert index.next_release_at_or_after(5).release_sample == 10


def test_past_the_last_cycle_there_is_no_cycle_and_nothing_left_to_rejoin():
    index = ReleaseIndex(releases_from_fields(*_phases()), rate_hz=15.0)

    # The return-to-home ramp make_cycles appends belongs to no cycle.
    assert index.cycle_for_sample(60) is None
    assert index.next_release_at_or_after(51) is None


def test_shifting_drops_finished_cycles_and_renumbers_the_rest():
    index = ReleaseIndex(releases_from_fields(*_phases()), rate_hz=15.0)

    shifted = index.shifted(30)

    # Cycle 1 is wholly behind the slice; cycle 2 is the one it starts inside.
    assert [entry.cycle for entry in shifted.releases] == [2, 3]
    assert shifted.releases[0].release_sample == 0
    assert shifted.releases[1].release_sample == 20


def test_metadata_round_trips_through_the_artifact_shape():
    index = ReleaseIndex(releases_from_fields(*_phases()), rate_hz=15.0)

    document = {
        "release_phase": RELEASE_PHASE,
        "rate_hz": 15.0,
        "cycle_index": index.as_metadata(),
    }
    restored = from_metadata(json.loads(json.dumps(document)))

    assert [entry.release_sample for entry in restored.releases] == [10, 30, 50]
    assert restored.release_phase == RELEASE_PHASE
    assert document["cycle_index"][0]["release_time_s"] == pytest.approx(10 / 15.0)


def test_an_artifact_without_the_flag_reads_as_none_rather_than_raising():
    assert from_metadata({"schema_version": 1}) is None
    assert from_metadata({"cycle_index": []}) is None


def test_a_malformed_entry_is_named_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="malformed cycle_index entry"):
        from_metadata({"cycle_index": [{"cycle": 1, "start_sample": 0}]})


def test_load_returns_none_for_a_directory_with_no_metadata(tmp_path):
    assert release_phase.load(tmp_path) is None


def test_load_reads_a_directory_or_its_npz(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "rate_hz": 15.0,
                "release_phase": RELEASE_PHASE,
                "cycle_index": [
                    {"cycle": 1, "start_sample": 0, "end_sample": 9, "release_sample": 4}
                ],
            }
        )
    )

    assert len(release_phase.load(tmp_path)) == 1
    assert len(release_phase.load(tmp_path / "replay_data.npz")) == 1


# --- placing a sample on the controller's prepared clock ----------------------------------


def _prepared(samples=60, rate_hz=15.0, time_scale=1.0):
    """A prepared stream and the coordinated trajectory behind it.

    Built around ``READY_POSE`` rather than the origin: joints 4 and 6 have
    limits that exclude zero, so a trajectory of small sinusoids about zero is
    not merely unrealistic, it is outside the position limits and preparation
    rightly refuses it.
    """
    time = np.arange(samples, dtype=float) / rate_hz
    arm = READY_POSE + np.column_stack(
        [0.2 * np.sin(time / 3.0 + j) for j in range(7)]
    )
    trajectory = CoordinatedTrajectory(
        time=time, arm=arm, hand=None, source="test"
    )
    prepared = _prepare_arm(trajectory, load_config(None), 600.0, time_scale=time_scale)
    return trajectory, prepared


@pytest.mark.parametrize("time_scale", [1.0, 5.0])
def test_a_sample_maps_to_a_time_and_back_to_itself(time_scale):
    trajectory, prepared = _prepared(time_scale=time_scale)

    for sample in (0, 1, 17, 40, 59):
        seconds = prepared_time_for_sample(prepared, trajectory, sample)
        assert sample_for_prepared_time(prepared, trajectory, seconds) == sample


def test_the_mapping_accounts_for_the_lead_in_and_the_time_scale():
    trajectory, prepared = _prepared(time_scale=5.0)

    capture_start = prepared.t[prepared.params["capture_start_index"]]
    # Sample 0 is where the capture starts, after the hold and the lead-in ramp.
    assert prepared_time_for_sample(prepared, trajectory, 0) == pytest.approx(capture_start)
    # And every source interval is stretched by the scale the check settled on.
    span = prepared_time_for_sample(
        prepared, trajectory, 30
    ) - prepared_time_for_sample(prepared, trajectory, 0)
    assert span == pytest.approx(
        trajectory.time[30] * prepared.params["time_scale"], rel=1e-6
    )


def test_times_before_and_after_the_capture_clamp_to_its_ends():
    trajectory, prepared = _prepared()

    # The opening hold reports the first sample, the closing one the last.
    assert sample_for_prepared_time(prepared, trajectory, 0.0) == 0
    assert sample_for_prepared_time(prepared, trajectory, prepared.duration) == len(
        trajectory.time
    ) - 1
    assert prepared_time_for_sample(prepared, trajectory, 10**6) == pytest.approx(
        prepared_time_for_sample(prepared, trajectory, len(trajectory.time) - 1)
    )


def test_the_flag_of_the_shipped_ten_cycle_artifact_names_real_samples():
    """The checked-in artifact carries the flag, and it describes its own NPZ."""
    index = release_phase.load(
        "apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8"
    )
    if index is None:  # pragma: no cover - only when run outside the workspace
        pytest.skip("the demo trajectories are not on this path")

    assert len(index) == 10
    assert index.release_phase == RELEASE_PHASE
    samples = [entry.release_sample for entry in index.releases]
    assert samples == [56, 155, 246, 330, 427, 622, 745, 851, 948, 1034]
    # Every release falls inside the cycle it belongs to, and the cycles run
    # back to back with no gap, which is what makes the run continuous.
    for entry in index.releases:
        assert entry.start_sample <= entry.release_sample <= entry.end_sample
    for previous, following in zip(index.releases, index.releases[1:]):
        assert following.start_sample == previous.end_sample + 1
