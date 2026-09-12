import numpy as np
import pytest

from franka_trajectory_replay import cartesian, kinematics
from franka_trajectory_replay.kinematics import READY_POSE
from franka_trajectory_replay.prepare import prepare
from franka_trajectory_replay.trajectory_io import Trajectory


def synthetic(rate=15.0, duration=4.0, amplitude=0.3):
    t = np.arange(int(duration * rate) + 1) / rate
    q = np.tile(READY_POSE, (len(t), 1))
    envelope = np.sin(np.pi * t / duration) ** 2
    for j in range(7):
        q[:, j] += amplitude * envelope * np.sin(2 * np.pi * (0.2 + 0.05 * j) * t)
    return Trajectory(t=t, q=q)


def test_batched_flange_transforms_match_the_per_sample_kinematics():
    rng = np.random.default_rng(3)
    q = READY_POSE + 0.5 * rng.standard_normal((25, 7))
    batched = kinematics.flange_transforms(q)
    for k in range(len(q)):
        np.testing.assert_allclose(batched[k], kinematics.flange_transform(q[k]), atol=1e-12)


def test_pose_stream_is_continuous_and_consistent_with_the_jacobian():
    prepared = prepare(synthetic(amplitude=0.2), rate=1000)
    stream = cartesian.from_joint_stream(prepared)
    assert stream.p.shape == (len(prepared.t), 3)
    assert stream.quat.shape == (len(prepared.t), 4)
    assert stream.q_null is prepared.q or np.shares_memory(stream.q_null, prepared.q)
    # No hemisphere flips along the stream, and every quaternion is a unit one.
    assert np.all(np.sum(stream.quat[1:] * stream.quat[:-1], axis=1) > 0.0)
    np.testing.assert_allclose(np.linalg.norm(stream.quat, axis=1), 1.0, atol=1e-12)
    # Finite-difference velocities agree with the geometric Jacobian times the joint velocity.
    for k in (700, 1500, 2600):
        jacobian = kinematics.flange_jacobian(prepared.q[k])
        twist = jacobian @ prepared.qd[k]
        np.testing.assert_allclose(stream.v[k], twist[:3], atol=2e-4)
        np.testing.assert_allclose(stream.w[k], twist[3:], atol=2e-3)
    # The tool offset moves the position and leaves the orientation alone.
    tool = kinematics.tool_transform((0.0, 0.0, 0.1))
    offset = cartesian.from_joint_stream(prepared, tool)
    np.testing.assert_allclose(offset.quat, stream.quat, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(offset.p - stream.p, axis=1), 0.1, atol=1e-12)


def test_cartesian_limit_check_passes_a_gentle_stream_and_flags_a_violent_one():
    gentle = cartesian.from_joint_stream(prepare(synthetic(amplitude=0.2), rate=1000))
    report = cartesian.check_cartesian_limits(gentle)
    assert report['ok'], report['violations']
    assert gentle.report is report
    assert 0.0 < report['fractions']['linear_velocity'] < 1.0
    assert 'within limits' in cartesian.summarize_cartesian(gentle)

    fast = prepare(synthetic(rate=30.0, duration=1.0, amplitude=0.6), rate=1000, auto_scale=False)
    violent = cartesian.from_joint_stream(fast)
    report = cartesian.check_cartesian_limits(violent, velocity_margin=0.02,
                                              acceleration_margin=0.02, jerk_margin=0.02)
    assert not report['ok']
    assert any('velocity' in text for text in report['violations'])
    assert 'VIOLATIONS' in cartesian.summarize_cartesian(violent)


def test_angular_velocity_of_a_constant_rate_rotation():
    dt = 0.001
    rate = 0.7  # rad/s about base z
    t = np.arange(500) * dt
    from scipy.spatial.transform import Rotation

    quat = Rotation.from_rotvec(np.outer(rate * t, [0, 0, 1])).as_quat()
    w = cartesian.angular_velocity(quat, dt)
    np.testing.assert_allclose(w[:, 2], rate, atol=1e-9)
    np.testing.assert_allclose(w[:, :2], 0.0, atol=1e-12)


def test_end_effector_frame_check():
    tool = np.eye(4)
    cartesian.check_end_effector_frame(np.eye(4), tool)
    offset = kinematics.tool_transform((0.0, 0.0, 0.1034))
    with pytest.raises(ValueError, match='end-effector frame F_T_EE'):
        cartesian.check_end_effector_frame(offset, tool)
    # The check is about matching the stream's tool, not about the flange specifically.
    cartesian.check_end_effector_frame(offset, offset)


def test_forward_kinematics_check_against_the_robot_pose():
    q = READY_POSE + 0.1
    o_t_ee = kinematics.flange_transform(q)
    cartesian.check_forward_kinematics(q, o_t_ee, np.eye(4))
    with pytest.raises(ValueError, match='disagrees'):
        cartesian.check_forward_kinematics(q + np.array([0.02, 0, 0, 0, 0, 0, 0]), o_t_ee, np.eye(4))
    with pytest.raises(ValueError, match='disagrees'):
        cartesian.check_forward_kinematics(q, o_t_ee, kinematics.tool_transform((0, 0, 0.05)))


def test_transform_error():
    a = kinematics.tool_transform((0.1, 0.0, 0.0), (0.0, 0.0, 0.3))
    position, angle = cartesian.transform_error(np.eye(4), a)
    assert position == pytest.approx(0.1)
    assert angle == pytest.approx(0.3)


def test_trajectory_message_packs_the_stream():
    pytest.importorskip('franka_trajectory_replay_msgs.msg')
    stream = cartesian.from_joint_stream(prepare(synthetic(amplitude=0.2), rate=1000))
    message = cartesian.trajectory_message(stream, 'fr3_link0', send_rate=100)
    assert message.header.frame_id == 'fr3_link0'
    assert len(message.points) == (len(stream.t) - 1) // 10 + 1 + (1 if (len(stream.t) - 1) % 10 else 0)
    first, last = message.points[0], message.points[-1]
    assert first.time_from_start.sec == 0 and first.time_from_start.nanosec == 0
    assert last.time_from_start.sec + last.time_from_start.nanosec * 1e-9 == pytest.approx(stream.duration)
    assert len(first.nullspace_positions) == 7
    np.testing.assert_allclose(
        [last.pose.position.x, last.pose.position.y, last.pose.position.z], stream.p[-1])
    goto = cartesian.goto_message(stream.p[0], stream.quat[0], stream.q_null[0], 0.0)
    assert len(goto.nullspace_positions) == 7
    assert goto.pose.orientation.w == pytest.approx(stream.quat[0][3])


GRASP_CENTRE = [-0.0874, -0.0327, 0.1453]


def test_end_effector_frame_check_expects_the_bare_flange_by_default():
    cartesian.check_end_effector_frame(np.eye(4))
    with pytest.raises(ValueError, match="bare flange"):
        cartesian.check_end_effector_frame(kinematics.tool_transform(GRASP_CENTRE))


def test_controller_tool_check_matches_the_stream_tool():
    tool = kinematics.tool_transform(GRASP_CENTRE, [0.0, 0.0, 0.0])
    actual = cartesian.check_controller_tool(GRASP_CENTRE, [0.0, 0.0, 0.0], tool)
    np.testing.assert_allclose(actual, tool, atol=1e-12)
    with pytest.raises(ValueError, match="does not match the tool"):
        cartesian.check_controller_tool([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], tool)
    with pytest.raises(ValueError, match="does not match the tool"):
        cartesian.check_controller_tool(GRASP_CENTRE, [0.0, 0.0, 0.5], tool)
    with pytest.raises(ValueError, match="does not expose"):
        cartesian.check_controller_tool(None, None, tool)


def test_pose_stream_through_the_grasp_centre_tool():
    prepared = prepare(synthetic(amplitude=0.2), rate=1000)
    tool = kinematics.tool_transform(GRASP_CENTRE)
    stream = cartesian.from_joint_stream(prepared, tool)
    flange = cartesian.from_joint_stream(prepared)
    # Same orientation, and every sample is the grasp centre offset from the flange.
    np.testing.assert_allclose(stream.quat, flange.quat, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(stream.p - flange.p, axis=1), np.linalg.norm(GRASP_CENTRE), atol=1e-12)
    for k in (0, 1500, len(prepared.t) - 1):
        expected = (kinematics.flange_transform(prepared.q[k]) @ tool)[:3, 3]
        np.testing.assert_allclose(stream.p[k], expected, atol=1e-12)
