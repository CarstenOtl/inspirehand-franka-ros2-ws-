import math

import numpy as np
import pytest

from policy_rollout import DEFAULT_CHECKPOINT
from policy_rollout import forge_osc as fo

DRIVER_HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)


def test_mimic_profile_keeps_the_module_constants():
    profile = fo.MIMIC_M24_PROFILE
    assert np.allclose(profile.arm_reset_joints, fo.FRANKA_ARM_RESET_JOINTS_M24)
    assert profile.ee_base_orn_euler == fo.EE_BASE_ORN_EULER
    assert profile.yaw_wrap_upper_deg == 235.0
    assert profile.entry_pinch == fo.ISAAC_ENTRY_PINCH


def test_yaw_seam_follows_the_profile():
    assert fo.wrap_yaw(math.radians(240.0)) == pytest.approx(math.radians(-120.0))
    assert fo.wrap_yaw(math.radians(240.0), 280.0) == pytest.approx(math.radians(240.0))


