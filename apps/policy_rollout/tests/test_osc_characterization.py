"""Planning helpers and the payload fit of the OSC characterization (no ROS, no MuJoCo)."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

pytest.importorskip("franka_trajectory_replay")

from utils import osc_characterization as oc  # noqa: E402
from utils import osc_characterization_analysis as oca  # noqa: E402


def _postures():
    """Three wrist-down postures plus the wrist spin: what a real session holds at."""

    poses = [
        np.array([0.55, -0.17, 0.18, -2.14, -0.39, 1.53, 0.86]),
        np.array([0.58, -0.61, -0.02, -2.54, -0.20, 2.05, 0.58]),
        np.array([0.53, -0.38, -0.12, -2.35, 0.01, 1.78, 0.68]),
    ]
    return poses + oc.spin_stops(SimpleNamespace(spin_range_rad=2.7, spin_stops=13))


def test_payload_fit_recovers_mass_com_and_offsets():
    mass, com = 0.58, np.array([0.004, -0.004, 0.098])
    offsets = np.array([0.3, -1.0, -0.6, 0.6, -0.4, 0.01, 0.0])
    holds = [
        {"q": q, "tau_ext": oca.payload_columns(q) @ np.concatenate(([mass], mass * com)) + offsets}
        for q in _postures()
    ]
    fit = oca.fit_payload(holds, offsets=True)
    assert fit["m"] == pytest.approx(mass, abs=1e-9)
    assert np.allclose(fit["com"], com, atol=1e-9)
    assert np.allclose(fit["offsets"], offsets, atol=1e-9)


def test_difference_fit_sets_the_sign_convention():
    """none - modelled carries exactly the applied payload, so its sign says which way tau_ext points."""

    mass, com = 0.5789, np.array([0.00358, -0.00413, 0.09805])
    report = {"payloads": {"modelled": {"load": {"m": mass, "com": com.tolist()}}}}
    true_hand = np.concatenate(([0.60], 0.60 * np.array([-0.01, 0.0, 0.085])))
    applied = np.concatenate(([mass], mass * com))
    offsets = np.array([0.2, -0.9, -0.5, 0.6, -0.3, 0.0, 0.0])
    holds = []
    for i, q in enumerate(_postures()):
        columns = oca.payload_columns(q)
        # Franka's tau_ext with the opposite sign of J^T w, to exercise the convention.
        holds.append({"label": str(i), "payload": "none", "q": q,
                      "tau_ext": -(columns @ true_hand) + offsets})
        holds.append({"label": str(i), "payload": "modelled", "q": q,
                      "tau_ext": -(columns @ (true_hand - applied)) + offsets})
    result = oca.analyze_payload(holds, report)
    assert result["none_minus_modelled"]["sign_convention"] == -1.0
    assert result["identified_hand"]["m"] == pytest.approx(0.60, abs=1e-9)
    assert np.allclose(result["identified_hand"]["com"], [-0.01, 0.0, 0.085], atol=1e-9)


def test_joint_moves_are_split_below_the_step_limit():
    start, end = np.zeros(7), np.array([2.5, -0.4, 0.0, 0.3, 0.0, 0.0, -1.2])
    waypoints = oc.chunks(start, end, 1.0)
    assert np.allclose(waypoints[-1], end)
    previous = start
    for waypoint in waypoints:
        assert np.max(np.abs(waypoint - previous)) <= 1.0 + 1e-12
        previous = waypoint


def test_probe_triangle_returns_to_the_start():
    legs = oc.probe_profile(SimpleNamespace(probe_cycles=2))
    assert legs[0][1] == 0.0 and legs[-1][2] == 0.0
    for (_, _, end), (_, start, _) in zip(legs[:-1], legs[1:]):
        assert end == start


def test_probe_levels_stretch_the_stroke_for_fast_levels():
    args = SimpleNamespace(probe_speed_mm_s=6.0, probe_rate_deg_s=2.0, probe_translation_mm=15.0,
                           probe_rotation_deg=8.0, probe_min_leg_s=1.5,
                           probe_speeds_mm_s=[2.0, 50.0], probe_rates_deg_s=[1.0, 15.0])
    slow, fast = oc.probe_levels(args)
    assert slow["amplitude_m"] == pytest.approx(0.015)
    assert fast["amplitude_m"] == pytest.approx(0.075)
    assert np.degrees(fast["amplitude_rad"]) == pytest.approx(22.5)
    # The planner checks the largest stroke.
    translation, _ = oc.probe_offsets(args)[1]
    assert translation[0] == pytest.approx(0.075)
    # Without a sweep the single level is the old probe.
    args.probe_speeds_mm_s = args.probe_rates_deg_s = None
    (only,) = oc.probe_levels(args)
    assert only["amplitude_m"] == pytest.approx(0.015) and only["speed_m_s"] == pytest.approx(0.006)


def test_speed_sweep_fit_recovers_dead_zone_and_viscous_loss():
    k = 565.0
    damping = 2.0 * np.sqrt(k)
    dead_zone, viscous, bias = 5.0, 20.0, 0.7
    probes = {}
    for level, v in enumerate((0.002, 0.006, 0.02, 0.05)):
        points = []
        for direction in (1, -1, -1, 1):
            # A constant bias pushes the spring up on + legs and down on - legs.
            points.append([v, dead_zone + (damping + viscous) * v + direction * bias, direction])
        probes[f"modelled/o0/L{level}"] = {"x": {"steady": {
            "goal_speed": v, "speed": v, "spring": float(np.mean([p[1] for p in points])),
            "lost": 0.0, "points": points}}}
    fit = oca.fit_speed_sweep(probes, [k] * 3 + [28.0] * 3)["modelled/o0"]["x"]
    assert fit["intercept"] == pytest.approx(dead_zone, abs=1e-9)
    assert fit["viscous"] == pytest.approx(viscous, abs=1e-6)


def test_loss_curve_fit_recovers_a_smoothed_coulomb_step():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.0002, 0.05, 4000)
    lost = 4.0 * np.tanh(v / 0.002) + 50.0 * v + rng.normal(0.0, 0.05, v.size)
    fit = oca.fit_loss_curve(np.column_stack((v, lost)), True,
                             [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 15, 20, 30, 40, 60])
    assert fit["coulomb"] == pytest.approx(4.0, abs=0.1)
    assert fit["viscous"] == pytest.approx(50.0, abs=3.0)
    assert 1000.0 * fit["v0"] == pytest.approx(2.0, rel=0.25)
    assert fit["bins"][-1]["lost"] > fit["bins"][0]["lost"]


def test_smoothing_keeps_the_window_edges():
    values = np.linspace(0.30, 0.31, 200)
    smoothed = oca.smooth(values, 25)
    assert smoothed.shape == values.shape
    assert abs(smoothed[0] - values[0]) < 1e-3 and abs(smoothed[-1] - values[-1]) < 1e-3
