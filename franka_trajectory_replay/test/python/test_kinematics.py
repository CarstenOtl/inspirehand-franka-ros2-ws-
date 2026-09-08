import os

import numpy as np
import pytest

from franka_trajectory_replay import kinematics

URDF_XACRO = None
try:
    from ament_index_python.packages import get_package_share_directory

    URDF_XACRO = os.path.join(get_package_share_directory('franka_description'), 'robots', 'fr3', 'fr3.urdf.xacro')
except Exception:  # noqa: BLE001
    pass


def test_ready_pose_matches_documented_flange():
    position, quaternion = kinematics.flange_pose(kinematics.READY_POSE)
    np.testing.assert_allclose(position, [0.3069, 0.0, 0.5903], atol=1e-4)
    np.testing.assert_allclose(np.abs(quaternion), [0.92388, 0.38268, 0.0, 0.0], atol=1e-4)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = kinematics.matrix_to_quaternion(kinematics.quaternion_to_matrix(q))
        assert min(np.linalg.norm(back - q), np.linalg.norm(back + q)) < 1e-9


def test_quaternion_angle():
    a = np.array([0.0, 0.0, 0.0, 1.0])
    b = kinematics.matrix_to_quaternion(kinematics.tool_transform((0, 0, 0), (0, 0, 0.3))[:3, :3])
    assert abs(kinematics.quaternion_angle(a, b) - 0.3) < 1e-9


@pytest.mark.skipif(URDF_XACRO is None or not os.path.exists(URDF_XACRO), reason='franka_description not available')
def test_dh_matches_urdf():
    pinocchio = pytest.importorskip('pinocchio')  # noqa: F841
    import subprocess
    import tempfile

    urdf = subprocess.check_output(['xacro', URDF_XACRO, 'hand:=false', 'ee_id:=none']).decode()
    with tempfile.NamedTemporaryFile('w', suffix='.urdf', delete=False) as handle:
        handle.write(urdf)
        path = handle.name
    fk = kinematics.UrdfKinematics(path, ['fr3_joint%d' % i for i in range(1, 8)])
    rng = np.random.default_rng(0)
    from franka_trajectory_replay import limits

    for _ in range(50):
        q = rng.uniform(limits.POSITION_LOWER, limits.POSITION_UPPER)
        p_urdf, r_urdf = fk.pose(q, 'fr3_link8', 'fr3_link0')
        p_dh, r_dh = kinematics.flange_pose(q)
        np.testing.assert_allclose(p_dh, p_urdf, atol=1e-9)
        assert kinematics.quaternion_angle(r_dh, r_urdf) < 1e-6
    os.unlink(path)
