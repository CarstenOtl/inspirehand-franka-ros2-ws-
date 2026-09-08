import numpy as np
import pytest

from camera_calibration.auto_waypoints import (
    AUTO_WAYPOINTS,
    FR3_JOINT_LIMITS,
    JOINT_NAMES,
    validate_auto_waypoints,
)


def test_auto_calibration_has_twelve_distinct_safe_fr3_waypoints():
    validate_auto_waypoints()
    waypoints = np.asarray(AUTO_WAYPOINTS)
    limits = np.asarray(FR3_JOINT_LIMITS)
    assert waypoints.shape == (12, len(JOINT_NAMES))
    assert np.unique(waypoints, axis=0).shape[0] == 12
    assert np.all(waypoints > limits[:, 0] + 0.02)
    assert np.all(waypoints < limits[:, 1] - 0.02)


def test_auto_waypoint_validation_rejects_wrong_count():
    with pytest.raises(ValueError, match="requires 12 waypoints"):
        validate_auto_waypoints(AUTO_WAYPOINTS[:-1])
