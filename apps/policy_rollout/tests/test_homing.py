import numpy as np
import pytest

from policy_rollout import forge_osc as fo
from policy_rollout.hardware import MAX_HOME_PRECOMPENSATION_RAD, next_home_command

HOME = np.asarray(fo.FRANKA_ARM_RESET_JOINTS_M24, dtype=float)
# The 2026-10-26 hardware residual (measured minus home), joint 7 held by friction.
RESIDUAL = np.array([-0.0087, 0.0125, 0.0093, -0.0063, 0.0244, 0.0159, -0.0557])


def test_the_retry_goal_is_shifted_against_the_residual():
    command = next_home_command(HOME, HOME, RESIDUAL)
    np.testing.assert_allclose(command - HOME, -RESIDUAL)


def test_a_pd_arm_with_a_constant_offset_comes_to_rest_on_home():
    # A PD law settles a fixed amount short of its goal when friction or an
    # unmodelled load holds it; shifting the goal cancels that.
    command = HOME.copy()
    for _ in range(3):
        measured = command + RESIDUAL
        command = next_home_command(command, HOME, measured - HOME)
    np.testing.assert_allclose(command + RESIDUAL, HOME, atol=1e-12)


def test_the_shift_accumulates_but_is_clipped():
    command = HOME.copy()
    for _ in range(10):
        command = next_home_command(command, HOME, np.full(7, 0.04))
    assert np.max(np.abs(command - HOME)) == pytest.approx(MAX_HOME_PRECOMPENSATION_RAD)
