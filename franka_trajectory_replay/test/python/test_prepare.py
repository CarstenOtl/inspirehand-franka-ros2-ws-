import numpy as np
import pytest

from franka_trajectory_replay import limits
from franka_trajectory_replay.kinematics import READY_POSE
from franka_trajectory_replay.prepare import goto_duration, prepare, resample
from franka_trajectory_replay.trajectory_io import Trajectory, load_trajectory, save_prepared, load_prepared


def synthetic(rate=15.0, duration=4.0, amplitude=0.3):
    t = np.arange(int(duration * rate) + 1) / rate
    q = np.tile(READY_POSE, (len(t), 1))
    envelope = np.sin(np.pi * t / duration) ** 2
    for j in range(7):
        q[:, j] += amplitude * envelope * np.sin(2 * np.pi * (0.2 + 0.05 * j) * t)
    return Trajectory(t=t, q=q)


def test_resample_layout_and_samples():
    source = synthetic(rate=20.0)  # 50 ms samples land exactly on the 1 ms grid
    prepared = resample(source, rate=1000, hold_start=0.5, hold_end=0.25, lead_in=0.4, lead_out=0.4)
    assert abs(prepared.rate - 1000) < 1e-6
    assert prepared.t[0] == 0.0
    k0 = prepared.params['capture_start_index']
    # holds are exact plateaus at the ramp ends
    np.testing.assert_allclose(prepared.q[:500], np.tile(prepared.q[0], (500, 1)), atol=1e-12)
    np.testing.assert_allclose(prepared.q[-250:], np.tile(prepared.q[-1], (250, 1)), atol=1e-12)
    # the capture samples are hit exactly from the capture start index on
    for k in range(0, len(source.t), 7):
        index = k0 + int(round(source.t[k] * 1000))
        np.testing.assert_allclose(prepared.q[index], source.q[k], atol=1e-6)
    # the synthetic capture starts and ends at rest, so the lead ramps go nowhere
    np.testing.assert_allclose(prepared.q[0], source.q[0], atol=1e-6)
    assert abs(prepared.params['lead_in'] - 0.4) < 1e-9
    # velocity is continuous everywhere: no step larger than a few mrad/s between samples
    assert np.abs(np.diff(prepared.qd, axis=0)).max() < 0.02


def test_lead_in_matches_a_moving_start():
    source = synthetic(rate=20.0)
    # chop the first second off so the capture starts mid-motion
    source = Trajectory(t=source.t[20:] - source.t[20], q=source.q[20:])
    prepared = resample(source, rate=1000, hold_start=0.2, hold_end=0.2)
    k0 = prepared.params['capture_start_index']
    v_capture = (source.q[1] - source.q[0]) / (source.t[1] - source.t[0])
    assert np.abs(v_capture).max() > 0.05
    np.testing.assert_allclose(prepared.q[k0], source.q[0], atol=1e-9)
    np.testing.assert_allclose(prepared.qd[k0], v_capture, atol=0.03)
    assert np.abs(prepared.qd[199]).max() < 1e-9  # at rest at the end of the hold
    assert np.abs(np.diff(prepared.qd, axis=0)).max() < 0.02


def test_prepare_passes_for_a_gentle_trajectory():
    prepared = prepare(synthetic(amplitude=0.2), rate=1000)
    assert prepared.report['ok'], prepared.report['violations']
    assert prepared.params['time_scale'] == 1.0


def test_auto_scale_slows_a_violent_trajectory():
    fast = synthetic(rate=30.0, duration=1.0, amplitude=0.6)
    unscaled = prepare(fast, rate=1000, auto_scale=False)
    assert not unscaled.report['ok']
    scaled = prepare(fast, rate=1000, auto_scale=True)
    assert scaled.report['ok'], scaled.report['violations']
    assert scaled.params['time_scale'] > 1.0
    assert scaled.duration > unscaled.duration


def test_cutoff_keeps_stream_smooth():
    prepared = resample(synthetic(), rate=1000, cutoff_hz=5.0, hold_start=0.3, hold_end=0.3)
    np.testing.assert_allclose(prepared.q[-1], prepared.q[-300], atol=1e-9)
    assert np.abs(np.diff(prepared.qd, axis=0)).max() < 0.02
    assert np.abs(prepared.q[0] - READY_POSE).max() < 0.02


def test_velocity_limits_shape():
    upper = limits.upper_velocity_limits(READY_POSE)
    lower = limits.lower_velocity_limits(READY_POSE)
    assert np.all(upper > 0) and np.all(lower < 0)
    near = READY_POSE.copy()
    near[0] = 2.7
    assert limits.upper_velocity_limits(near)[0] < 0.6
    assert limits.lower_velocity_limits(near)[0] < -2.5


def test_goto_duration_rules():
    assert goto_duration(np.zeros(7)) == 2.0
    assert abs(goto_duration(np.array([1.0, 0, 0, 0, 0, 0, 0]), max_velocity=0.5, max_acceleration=100.0) - 3.75) < 1e-9


def test_load_npz_isaac_layout(tmp_path):
    source = synthetic()
    path = tmp_path / 'capture.npz'
    order = [6, 5, 4, 3, 2, 1, 0]
    np.savez(path, joint_pos_arm=source.q[:, order].astype(np.float32),
             arm_joint_names=np.array(['panda_joint%d' % (i + 1) for i in order]), dt=np.float64(1 / 15))
    loaded = load_trajectory(path)
    np.testing.assert_allclose(loaded.q, source.q, atol=1e-6)
    np.testing.assert_allclose(loaded.t, source.t, atol=1e-9)


def test_load_refuses_foreign_joint_names(tmp_path):
    path = tmp_path / 'foreign.npz'
    np.savez(path, joint_pos_arm=np.zeros((10, 7)), dt=0.1,
             arm_joint_names=np.array(['shoulder_pitch_r_joint'] + ['x%d' % i for i in range(6)]))
    with pytest.raises(ValueError):
        load_trajectory(path)
    loaded = load_trajectory(path, assume_order=True)
    assert loaded.q.shape == (10, 7)


def test_load_plain_npy_needs_rate(tmp_path):
    path = tmp_path / 'plain.npy'
    np.save(path, np.tile(READY_POSE, (20, 1)))
    with pytest.raises(ValueError):
        load_trajectory(path)
    assert load_trajectory(path, rate=10.0).duration == pytest.approx(1.9)


def test_load_refuses_degrees(tmp_path):
    path = tmp_path / 'deg.npy'
    np.save(path, np.tile(np.degrees(READY_POSE), (20, 1)))
    with pytest.raises(ValueError):
        load_trajectory(path, rate=10.0)
    assert np.allclose(load_trajectory(path, rate=10.0, degrees=True).q[0], READY_POSE)


def test_prepared_roundtrip(tmp_path):
    prepared = prepare(synthetic(), rate=500, joint_names=['fr3_joint%d' % i for i in range(1, 8)])
    save_prepared(tmp_path / 'p.npz', prepared, None, {'x': 1})
    back, extras, meta = load_prepared(tmp_path / 'p.npz')
    np.testing.assert_allclose(back.q, prepared.q)
    assert back.joint_names == prepared.joint_names
    assert meta == {'x': 1}


def test_load_multi_env_npz_with_sidecar(tmp_path):
    import json

    source = synthetic()
    n = len(source.t)
    q19 = np.zeros((n, 3, 19), dtype=np.float32)
    q19[:, 1, :7] = source.q
    q19[:, 1, 7:] = 0.5
    tcp = np.zeros((n, 3, 3), dtype=np.float32)
    np.savez(tmp_path / 'replay_data.npz', step=np.arange(n), joint_pos=q19, tcp_pos=tcp)
    names = ['fr3_joint%d' % i for i in range(1, 8)] + ['index_joint_0', 'thumb_joint_1'] + ['h%d' % i for i in range(10)]
    (tmp_path / 'metadata.json').write_text(json.dumps({
        'dt': 1 / 15, 'joint_names': names, 'arm_joint_ids': list(range(7)), 'task': 'x'}))
    loaded = load_trajectory(tmp_path / 'replay_data.npz', env=1)
    np.testing.assert_allclose(loaded.q, source.q, atol=1e-6)
    np.testing.assert_allclose(loaded.t, source.t, atol=1e-9)
    assert loaded.meta['environments'] == 3 and loaded.meta['task'] == 'x'
    with pytest.raises(ValueError):
        load_trajectory(tmp_path / 'replay_data.npz', env=5)


def test_load_csv_segment(tmp_path):
    source = synthetic()
    names = ['fr3_joint%d' % i for i in range(1, 8)] + ['thumb_joint_0']
    header = ['time_s'] + ['pos_' + n for n in names] + ['vel_' + n for n in names] + ['cmd_' + n for n in names]
    rows = []
    for k in range(len(source.t)):
        q = list(source.q[k]) + [0.3]
        rows.append([source.t[k] + 5.0] + q + [0.0] * 8 + [v + 0.01 for v in q])
    path = tmp_path / 'env0_segment1.csv'
    path.write_text('\n'.join(','.join('%.9g' % v for v in row) for row in [header] + rows).replace(
        ','.join('%.9g' % 0 for _ in range(0)), '') if False else
        ','.join(header) + '\n' + '\n'.join(','.join('%.9g' % v for v in row) for row in rows) + '\n')
    loaded = load_trajectory(path)
    np.testing.assert_allclose(loaded.q, source.q, atol=1e-6)
    assert loaded.t[0] == 0.0
    assert loaded.qd is not None and loaded.qd.shape == (len(source.t), 7)
    cmd = load_trajectory(path, csv_prefix='cmd')
    np.testing.assert_allclose(cmd.q, source.q + 0.01, atol=1e-6)


def test_thumb_joint_cannot_steal_an_arm_joint():
    from franka_trajectory_replay.trajectory_io import joint_order_from_names

    names = ['thumb_joint_1', 'index_joint_2'] + ['fr3_joint%d' % i for i in range(1, 8)]
    assert joint_order_from_names(names) == list(range(2, 9))


def test_linear_interpolation_is_straight_between_blends_and_smooth():
    source = synthetic(rate=15.0)
    prepared = resample(source, rate=1000, interpolation='linear', blend_time=0.04)
    k0 = prepared.params['capture_start_index']
    # mid-segment samples lie on the straight line between the two waypoints
    for k in range(3, 40, 5):
        t_mid = 0.5 * (source.t[k] + source.t[k + 1])
        index = k0 + int(round(t_mid * 1000))
        t_grid = (index - k0) / 1000.0
        expected = np.array([np.interp(t_grid, source.t, source.q[:, j]) for j in range(7)])
        np.testing.assert_allclose(prepared.q[index], expected, atol=1e-6)
    # velocity is continuous (corner blends), so no 1 ms velocity steps
    assert np.abs(np.diff(prepared.qd, axis=0)).max() < 0.02
    assert prepared.params['interpolation'] == 'linear'
