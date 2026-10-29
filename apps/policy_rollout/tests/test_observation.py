import pytest

torch = pytest.importorskip("torch")

from policy_rollout.observation import (
    FR3_ARM_EFFORT_LIMITS_NM,
    FR3_POLICY_JOINT_NAMES,
    build_native_osc_proprio,
    native_osc_proprio_dim,
    one_hot_process_phase,
)
from policy_rollout.session import OscActionFilter


def test_native_osc_proprio_has_exact_29_value_order():
    value = build_native_osc_proprio(range(10), range(10, 20), range(20, 29))
    assert value.tolist() == pytest.approx(list(range(29)))
    assert len(FR3_POLICY_JOINT_NAMES) == 10


def test_omitted_hand_velocity_drops_only_the_three_finger_rates():
    value = build_native_osc_proprio(
        range(10),
        range(10, 20),
        range(20, 29),
        hand_velocity_observation="omitted",
    )
    # q10, then the seven arm velocities only, then the previous action: the
    # three hand velocities 17, 18, 19 are the whole difference.
    assert value.tolist() == pytest.approx(
        list(range(10)) + list(range(10, 17)) + list(range(20, 29))
    )
    assert native_osc_proprio_dim("omitted") == 26
    assert native_osc_proprio_dim("included") == 29


def test_unknown_hand_velocity_convention_is_refused():
    with pytest.raises(ValueError, match="hand_velocity_observation"):
        build_native_osc_proprio(
            range(10), range(10, 20), range(20, 29), hand_velocity_observation="zeroed"
        )


def test_force_observation_appends_pad_forces_and_normalized_torques():
    torque = [87.0, -43.5, 0.0, 200.0, 6.0, -12.0, -30.0]
    value = build_native_osc_proprio(
        range(10),
        range(10, 20),
        range(20, 29),
        force_observation="included",
        fingertip_force_n=[3.5, -0.2],
        arm_joint_torque_nm=torque,
    )
    assert native_osc_proprio_dim("included", "included") == 38
    assert value.shape == (38,)
    # q10, dq10, the previous action, then [thumb, index] in N / 20
    # (unilateral) and the torques over the FR3 limits, clipped to 1.
    assert value[:29].tolist() == pytest.approx(list(range(29)))
    assert value[29:31].tolist() == pytest.approx([3.5 / 20.0, 0.0])
    expected = [max(-1.0, min(1.0, t / l)) for t, l in zip(torque, FR3_ARM_EFFORT_LIMITS_NM)]
    assert value[31:].tolist() == pytest.approx(expected)


def test_force_observation_without_hand_velocities_is_35_wide():
    torque = [8.7, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2]
    value = build_native_osc_proprio(
        range(10),
        range(10, 20),
        range(20, 29),
        hand_velocity_observation="omitted",
        force_observation="included",
        fingertip_force_n=[1.0, 2.0],
        arm_joint_torque_nm=torque,
    )
    assert native_osc_proprio_dim("omitted", "included") == 35
    # q10, the arm velocities only, the previous action, then the forces: the
    # m24-dp3-nominal-da-force-nohv student's first proprio layer keeps the
    # previous-action column fingerprint at 17-25, so the forces follow it.
    assert value.tolist() == pytest.approx(
        list(range(10)) + list(range(10, 17)) + list(range(20, 29))
        + [0.05, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1]
    )


def test_force_observing_checkpoint_refuses_missing_forces():
    with pytest.raises(ValueError, match="fingertip_force_n"):
        build_native_osc_proprio(
            range(10), range(10, 20), range(20, 29), force_observation="included"
        )


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
