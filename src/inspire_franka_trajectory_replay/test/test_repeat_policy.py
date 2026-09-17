"""Repeating one V2 rollout around the scripted release/reset recipe."""

from pathlib import Path

import numpy as np
import pytest

from inspire_franka_trajectory_replay.make_cycles import load_home
from inspire_franka_trajectory_replay.repeat_policy import (
    build,
    load_recipe,
    select_release_sample,
)
from inspire_franka_trajectory_replay.trajectory import load_trajectory


ROOT = Path(__file__).resolve().parents[3]
TRAJ3 = ROOT / "apps/traj_replay/demo_trajs/traj_3_joint5_cap_2p8"
TRAJ3_MULTI = ROOT / "apps/traj_replay/demo_trajs/traj_3_multi_joint5_cap_2p8"
CURRENT_WAYPOINTS = (
    ROOT
    / "apps/traj_replay/scripted_waypoints/"
    "waypoints_release_and_reset_franka_20260910_current.yaml"
)
HISTORICAL_WAYPOINTS = (
    ROOT
    / "apps/traj_replay/scripted_waypoints/"
    "waypoints_release_and_reset_franka_20260905_traj1.yaml"
)


@pytest.fixture(scope="module")
def policy():
    return load_trajectory(str(TRAJ3))


def test_first_hybrid_handoff_selects_traj3_sample_56(policy):
    release = select_release_sample(
        policy, environment=0, reference=TRAJ3_MULTI, reference_cycle=1
    )

    assert release.sample == 56
    assert release.reference_handoff_sample == 55
    assert np.degrees(release.source_turn_progress_rad) == pytest.approx(50.50, abs=0.01)
    assert release.reference_turn_progress_rad is None
    assert release.arm_max_delta_rad == pytest.approx(0.1018, abs=0.0001)
    assert release.hand_max_delta_rad == pytest.approx(0.0113, abs=0.0001)


def test_explicit_release_is_inclusive_and_each_cycle_returns_home(policy):
    home_arm, home_hand = load_home(TRAJ3 / "homing.yaml")
    recipe = load_recipe(CURRENT_WAYPOINTS)

    result = build(
        policy,
        home_arm,
        home_hand,
        recipe,
        cycles=2,
        release_sample=56,
    )

    assert result.policy_samples == 57
    assert result.cycle_samples == 116
    assert len(result.time) == 232
    assert np.allclose(result.arm[:57], policy.arm[:57])
    assert np.allclose(result.hand[:57], policy.hand[:57])
    assert np.allclose(result.arm[result.cycle_samples - 1], home_arm)
    assert np.allclose(result.hand[result.cycle_samples - 1], home_hand)
    assert result.seam_delta < 1e-12
    assert result.phase_index[0]["release_start_sample"] == 57


def test_release_choice_is_required_and_bounds_checked(policy):
    with pytest.raises(ValueError, match="exactly one"):
        select_release_sample(policy, environment=0)
    with pytest.raises(ValueError, match="within"):
        select_release_sample(policy, environment=0, sample=len(policy.arm))


def test_old_cartesian_waypoint_recipe_is_not_mixed_into_joint_pd_replay():
    with pytest.raises(ValueError, match="robot_joint_position_pd"):
        load_recipe(HISTORICAL_WAYPOINTS)
