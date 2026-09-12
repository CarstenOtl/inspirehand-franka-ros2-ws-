import json
from pathlib import Path

import numpy as np
import pytest

from inspire_franka_trajectory_replay.splice_intervention import (
    build_splice,
    remap_cycle_index,
    write_splice,
)
from inspire_franka_trajectory_replay.trajectory import (
    ARM_JOINTS,
    HAND_JOINTS,
    CoordinatedTrajectory,
    load_trajectory,
)
from inspire_franka_trajectory_replay.waypoints import Waypoint


RATE = 15.0


def _original(tmp_path, count=80):
    time = np.arange(count, dtype=float) / RATE
    arm = np.arange(count, dtype=float)[:, None] * np.full((1, 7), 0.001)
    hand = np.arange(count, dtype=float)[:, None] * np.full((1, 6), 0.0005)
    source = tmp_path / "original" / "replay_data.npz"
    source.parent.mkdir()
    np.savez(
        source,
        joint_pos_arm=arm,
        joint_pos_hand=hand,
        arm_joint_names=np.asarray(ARM_JOINTS),
        hand_joint_names=np.asarray(HAND_JOINTS),
        sample_time_s=time,
    )
    return CoordinatedTrajectory(time, arm, hand, source)


def _waypoint(index, arm, hand):
    return Waypoint(
        index=index,
        stamp_ns=index,
        arm=np.asarray(arm, dtype=float),
        hand=np.asarray(hand, dtype=float),
        source="test snapshot",
    )


def _metadata(count=80):
    return {
        "sample_count": count,
        "recording_frequency_hz": RATE,
        "release_phase": "follow_waypoints",
        "cycle_index": [
            {
                "cycle": 1,
                "start_sample": 0,
                "end_sample": 39,
                "release_sample": 30,
                "release_time_s": 2.0,
            },
            {
                "cycle": 2,
                "start_sample": 40,
                "end_sample": 69,
                "release_sample": 55,
                "release_time_s": 55 / RATE,
            },
        ],
    }


def test_splice_keeps_the_prefix_and_suffix_and_visits_every_mark(tmp_path):
    original = _original(tmp_path)
    points = [
        _waypoint(1, np.full(7, 0.10), np.full(6, 0.20)),
        _waypoint(2, np.full(7, 0.15), np.full(6, 0.25)),
    ]

    result = build_splice(
        original, 10, 30, points, RATE, dwell_s=0.2, peak_speed=0.35
    )

    assert result.arm[:11] == pytest.approx(original.arm[:11])
    assert result.hand[:11] == pytest.approx(original.hand[:11])
    assert result.arm[-49:] == pytest.approx(original.arm[31:])
    assert result.hand[-49:] == pytest.approx(original.hand[31:])
    assert result.arm[result.waypoint_samples[0]] == pytest.approx(points[0].arm)
    assert result.arm[result.waypoint_samples[1]] == pytest.approx(points[1].arm)
    assert result.arm[result.composite_rejoin_sample] == pytest.approx(original.arm[30])
    assert result.arm[result.composite_suffix_sample] == pytest.approx(original.arm[31])
    assert np.diff(result.time) == pytest.approx(np.full(len(result.time) - 1, 1 / RATE))


def test_release_and_later_cycles_are_renumbered_after_the_replacement(tmp_path):
    original = _original(tmp_path)
    point = _waypoint(1, np.full(7, 0.10), np.full(6, 0.20))
    result = build_splice(original, 10, 30, [point], RATE, dwell_s=0.2)

    remapped = remap_cycle_index(_metadata(), result, len(result.time), RATE)
    shift = len(result.time) - len(original.time)

    assert remapped[0]["start_sample"] == 0
    assert remapped[0]["release_sample"] == result.composite_rejoin_sample
    assert remapped[0]["end_sample"] == 39 + shift
    assert remapped[1]["start_sample"] == 40 + shift
    assert remapped[1]["release_sample"] == 55 + shift
    assert remapped[1]["end_sample"] == 69 + shift


def test_written_splice_is_an_ordinary_replay_artifact_with_matching_home(tmp_path):
    original = _original(tmp_path)
    metadata = _metadata()
    (original.source.parent / "metadata.json").write_text(json.dumps(metadata))
    home = tmp_path / "homing.yaml"
    home.write_text("schema_version: 1\nname: original_home\n")
    point = _waypoint(1, np.full(7, 0.10), np.full(6, 0.20))
    result = build_splice(original, 10, 30, [point], RATE, dwell_s=0.2)
    record = {
        "index": 1,
        "paused_sample": 10,
        "rejoin_sample": 30,
    }

    output = write_splice(
        tmp_path / "composite",
        result,
        original,
        metadata,
        home,
        tmp_path / "session",
        record,
        [point],
        RATE,
        0.2,
        0.35,
    )

    loaded = load_trajectory(str(output))
    assert loaded.arm == pytest.approx(result.arm)
    assert loaded.hand == pytest.approx(result.hand)
    assert (output / "homing.yaml").read_text() == home.read_text()
    document = json.loads((output / "metadata.json").read_text())
    assert document["splice"]["continuous_human_path_recorded"] is False
    assert document["splice"]["captured_waypoints"] == 1
    assert document["splice"]["composite_suffix_sample"] == result.composite_suffix_sample
    assert document["cycle_index"][0]["release_sample"] == result.composite_rejoin_sample


@pytest.mark.parametrize("pause,rejoin", [(-1, 30), (30, 30), (40, 30), (10, 80)])
def test_invalid_splice_bounds_are_refused(tmp_path, pause, rejoin):
    original = _original(tmp_path)
    point = _waypoint(1, np.full(7, 0.10), np.full(6, 0.20))

    with pytest.raises(ValueError, match="splice bounds"):
        build_splice(original, pause, rejoin, [point], RATE)


def test_source_without_hand_positions_is_refused(tmp_path):
    original = _original(tmp_path)
    arm_only = CoordinatedTrajectory(
        original.time, original.arm, None, original.source
    )
    point = _waypoint(1, np.full(7, 0.10), np.full(6, 0.20))

    with pytest.raises(ValueError, match="no hand channel"):
        build_splice(arm_only, 10, 30, [point], RATE)
