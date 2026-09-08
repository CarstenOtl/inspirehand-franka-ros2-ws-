import json

import numpy as np

from franka_trajectory_replay.analysis import analyze, format_report, segments
from franka_trajectory_replay.kinematics import READY_POSE
from franka_trajectory_replay.plots import FIGURES, Context, plot_all


def fake_run(tmp_path, lag_ms=20):
    t = np.arange(0, 6.0, 0.001)
    q_ref = np.tile(READY_POSE, (len(t), 1))
    phase = np.zeros(len(t), dtype=int)
    elapsed = np.zeros(len(t))
    # goto 0.5..2.5, idle, trajectory 3..5
    goto = (t >= 0.5) & (t < 2.5)
    traj = (t >= 3.0) & (t < 5.0)
    phase[goto] = 1
    phase[traj] = 2
    elapsed[goto] = t[goto] - 0.5
    elapsed[traj] = t[traj] - 3.0
    q_ref[goto, 0] += 0.2 * (t[goto] - 0.5) / 2.0
    q_ref[t >= 2.5, 0] += 0.2
    q_ref[traj, 1] += 0.1 * np.sin(np.pi * (t[traj] - 3.0))
    q = np.array([np.interp(t - lag_ms / 1000.0, t, q_ref[:, j]) for j in range(7)]).T
    dq = np.gradient(q, 0.001, axis=0)
    data = {
        't0_ns': np.int64(0), 't': t, 'stamp_ns': (t * 1e9).astype(np.int64),
        'q_ref': q_ref, 'qd_ref': np.gradient(q_ref, 0.001, axis=0), 'q': q, 'dq': dq,
        'tau': np.zeros_like(q), 'out': q_ref.copy(), 'phase': phase, 'elapsed': elapsed,
        'mode': np.asarray('position'), 'joint_names': np.asarray(['fr3_joint%d' % i for i in range(1, 8)]),
        'source': np.asarray('test'),
    }
    steps = [{'name': 'goto_start', 'start_ns': int(0.5e9), 'end_ns': int(2.5e9)},
             {'name': 'trajectory', 'start_ns': int(3.0e9), 'end_ns': int(5.0e9)}]
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'run.json').write_text(json.dumps({'steps': steps}))
    return run_dir, data, {'steps': steps}


def test_segments_and_report(tmp_path):
    run_dir, data, meta = fake_run(tmp_path)
    segs = segments(data)
    assert [s['phase'] for s in segs] == [1, 2]
    report = analyze(run_dir, data, meta)
    names = [s['name'] for s in report['segments']]
    assert names == ['goto_start', 'trajectory']
    traj = report['segments'][1]
    assert abs(traj['joint']['joints'][1]['lag'] - 0.02) < 0.002
    assert traj['tcp']['position_error_max_mm'] > 0
    text = format_report(report, list(data['joint_names']))
    assert 'Headline' in text and 'trajectory' in text


def test_all_figures_render(tmp_path):
    run_dir, data, meta = fake_run(tmp_path)
    ctx = Context(run_dir, data, meta)
    written = plot_all(ctx)
    produced = {p.split('/')[-1].split('.')[0] for p in written}
    # figures needing robot_state data are skipped without it
    assert {'joints', 'joint_errors', 'velocities', 'torques', 'tcp_path', 'tcp_error', 'sampling',
            'command_limits'} <= produced
    assert 'external_wrench' not in produced
    assert set(FIGURES) >= produced


def test_jacobian_decomposition_matches_fk():
    from franka_trajectory_replay import kinematics

    rng = np.random.default_rng(3)
    q_ref = np.tile(READY_POSE, (5, 1)) + rng.normal(scale=0.3, size=(5, 7))
    q_meas = q_ref + rng.normal(scale=1e-3, size=(5, 7))
    linear, angular, error, angular_error = kinematics.tcp_error_contributions(q_ref, q_meas)
    np.testing.assert_allclose(linear.sum(axis=1), error, atol=2e-6)
    np.testing.assert_allclose(angular.sum(axis=1), angular_error, atol=1e-5)  # second-order terms


def test_contribution_figure_and_metrics(tmp_path):
    from franka_trajectory_replay.analysis import tcp_contributions

    run_dir, data, meta = fake_run(tmp_path)
    ctx = Context(run_dir, data, meta)
    seg = [s for s in ctx.segments if s['phase'] == 2][0]
    summary = tcp_contributions(data, seg)
    assert len(summary['share_rms_mm']) == 7 and summary['at_peaks']
    # only joint 2 moves in the fake trajectory segment, so it carries the error
    assert summary['at_peaks'][0]['dominant_joint'] == 2
    written = plot_all(ctx, only=['tcp_contributions'])
    assert written
