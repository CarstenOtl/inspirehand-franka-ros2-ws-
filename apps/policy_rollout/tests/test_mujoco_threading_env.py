"""The MuJoCo threading scene's arm dynamics switch."""

import numpy as np
import pytest

from policy_rollout import forge_osc as fo


def _scene(**kwargs):
    pytest.importorskip("mujoco")
    from policy_rollout.mujoco_threading_env import ThreadingScene

    nut_quat = fo.quat_from_euler_xyz(0.0, 0.0, np.deg2rad(30.0))
    return ThreadingScene(nut_quat_wxyz=nut_quat, **kwargs)


def test_arm_friction_scale_sets_the_menagerie_values_on_the_arm_dofs():
    from policy_rollout.mujoco_threading_env import (
        MENAGERIE_ARM_DAMPING,
        MENAGERIE_ARM_FRICTIONLOSS,
    )

    frictionless = _scene()
    assert frictionless.arm_friction_scale == 0.0
    np.testing.assert_allclose(frictionless.model.dof_frictionloss[frictionless.arm_dofs], 0.0)
    np.testing.assert_allclose(frictionless.model.dof_damping[frictionless.arm_dofs], 0.0)

    half = _scene(arm_friction_scale=0.5)
    np.testing.assert_allclose(
        half.model.dof_frictionloss[half.arm_dofs], 0.5 * np.asarray(MENAGERIE_ARM_FRICTIONLOSS)
    )
    np.testing.assert_allclose(
        half.model.dof_damping[half.arm_dofs], 0.5 * np.asarray(MENAGERIE_ARM_DAMPING)
    )
    # Only the arm: the hand and thread-pair dofs keep their own values.
    other = [i for i in range(half.model.nv) if i not in half.arm_dofs]
    np.testing.assert_allclose(
        half.model.dof_frictionloss[other], frictionless.model.dof_frictionloss[other]
    )
    with pytest.raises(ValueError):
        _scene(arm_friction_scale=-1.0)
