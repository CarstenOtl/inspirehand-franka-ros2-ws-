import pytest

torch = pytest.importorskip("torch")

from policy_rollout.observation import (
    FR3_POLICY_JOINT_NAMES,
    build_native_osc_proprio,
    one_hot_process_phase,
)
from policy_rollout.session import OscActionFilter


def test_native_osc_proprio_has_exact_29_value_order():
    value = build_native_osc_proprio(range(10), range(10, 20), range(20, 29))
    assert value.tolist() == pytest.approx(list(range(29)))
    assert len(FR3_POLICY_JOINT_NAMES) == 10


def test_process_phase_uses_checkpoint_order_and_aliases():
    schema = ("policy", "follow_waypoints", "return_to_reset")
    assert one_hot_process_phase("follow", schema).tolist() == [0.0, 1.0, 0.0]


def test_native_filter_matches_forge_ema_and_updates_history():
    action_filter = OscActionFilter("native")
    action_filter.reset(torch.zeros(9))
    _, filtered, _ = action_filter.apply(torch.ones(9))
    assert filtered[:6].tolist() == pytest.approx([0.0625] * 6)
    assert filtered[6:].tolist() == pytest.approx([0.1875] * 3)


def test_unified_filter_applies_checkpoint_scale_without_second_ema():
    action_filter = OscActionFilter("unified", range(1, 10))
    action_filter.reset(torch.zeros(9))
    bounded, filtered, clipped = action_filter.apply(torch.tensor([2.0] + [0.5] * 8))
    assert bounded[0] == 1.0
    assert filtered.tolist() == pytest.approx([1.0] + [0.5 * i for i in range(2, 10)])
    assert clipped == 1
