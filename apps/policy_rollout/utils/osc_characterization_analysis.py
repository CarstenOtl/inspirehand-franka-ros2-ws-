#!/usr/bin/env python3
"""Analyse an ``osc_characterization.py`` run: dead zones and the payload fit.

    python3 apps/policy_rollout/utils/osc_characterization_analysis.py debug/osc_characterization/<run>

Dead-zone probe, per pose and base axis, from ``~/cartesian_state_fast`` (1 kHz):
* the spring wrench is -K x (measured - target), which is what the impedance law
  applies before damping and any dead zone;
* **breakaway**: the spring wrench when the arm first follows the goal in a leg;
* **creep**: the median spring wrench while it follows;
* per axis, half the sum of the + and - creep values is the dead zone (friction-like)
  and half the difference is a bias (an unmodelled force, e.g. payload or gravity).

Payload (hardware only), from ``tau_ext_hat_filtered`` at the static holds:
``tau_ext = s J(q)^T [m g ; (R p) x g] + b`` with p = m c in the flange frame and a
constant per-joint offset b, fitted per payload level. The difference between the
"none" and "modelled" levels at the same postures has no offsets and a known answer
(the payload set_load applied), which fixes Franka's sign convention s.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from franka_trajectory_replay import kinematics as kin  # noqa: E402
from utils import osc_pose_repeatability as rep  # noqa: E402

AXES = ("x", "y", "z", "rx", "ry", "rz")
GRAVITY = np.array([0.0, 0.0, -9.81])
# Training URDF hand at reset, flange frame, +40 g cable (2026-11-10 estimate).
TRAINING_HAND = {"m": 0.579, "com": [-0.0107, -0.0018, 0.0845]}


# --- reading --------------------------------------------------------------------
def read_bag(path: Path, windows: list[tuple[float, float]]) -> dict:
    """Fast controller state and robot state inside the given time windows (header time)."""

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    lower = min(w[0] for w in windows)
    upper = max(w[1] for w in windows)
    edges = np.array(sorted(windows))

    def wanted(t):
        if t < lower or t > upper:
            return False
        index = np.searchsorted(edges[:, 0], t, side="right") - 1
        return index >= 0 and t <= edges[index, 1]

    fast, state = [], []
    classes = {}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if not (topic.endswith("/cartesian_state_fast") or topic.endswith("/robot_state")):
            continue
        cls = classes.setdefault(topic, get_message(types[topic]))
        message = deserialize_message(raw, cls)
        t = message.header.stamp.sec + 1.0e-9 * message.header.stamp.nanosec
        if not wanted(t):
            continue
        if topic.endswith("/cartesian_state_fast"):
            m, g = message.measured, message.goal
            fast.append((
                t,
                m.position.x, m.position.y, m.position.z,
                m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w,
                g.position.x, g.position.y, g.position.z,
                g.orientation.x, g.orientation.y, g.orientation.z, g.orientation.w,
                *message.error, *message.tau_command,
            ))
        else:
            state.append((
                t,
                *message.measured_joint_state.position,
                *message.measured_joint_state.velocity,
                *message.measured_joint_state.effort,
                *message.desired_joint_state.effort,
                *message.tau_ext_hat_filtered.effort,
                *message.measured_joint_motor_state.position,
            ))
    out = {}
    if fast:
        a = np.array(fast)
        out["fast"] = {"t": a[:, 0], "position": a[:, 1:4], "quaternion": a[:, 4:8],
                       "goal_position": a[:, 8:11], "goal_quaternion": a[:, 11:15],
                       "error": a[:, 15:21], "tau_command": a[:, 21:28]}
    if state:
        a = np.array(state)
        out["state"] = {"t": a[:, 0], "q": a[:, 1:8], "dq": a[:, 8:15], "tau_j": a[:, 15:22],
                        "tau_j_d": a[:, 22:29], "tau_ext": a[:, 29:36], "theta": a[:, 36:43]}
    return out


def segments(events, start: str, end: str, key: str = "label") -> list[dict]:
    out, open_ = [], {}
    for e in events:
        if e["event"] == start:
            open_[(e.get(key), e.get("payload"), e.get("axis"))] = e
        elif e["event"] == end:
            begin = open_.pop((e.get(key), e.get("payload"), e.get("axis")), None)
            if begin is not None:
                out.append({**begin, "t_end": e["t"], "end": e})
    return out


# --- probe ----------------------------------------------------------------------
def smooth(values, window: int):
    """Moving average, padded with the end values (zero padding made the first and last
    samples of every window look like metres per second)."""

    values = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    before, after = window // 2, window - 1 - window // 2
    padded = np.concatenate((np.full(before, values[0]), values, np.full(after, values[-1])))
    return np.convolve(padded, kernel, mode="valid")


def analyze_probe_axis(fast, axis: int, legs: list[dict], t_end: float, stiffness, args,
                       goal_speed=None) -> dict:
    t = fast["t"]
    inside = (t >= legs[0]["t"]) & (t <= t_end)
    if np.count_nonzero(inside) < 50:
        return {"error": "no controller state in the probe window"}
    times = t[inside]
    spring = -stiffness[axis] * fast["error"][inside, axis]
    if axis < 3:
        coordinate = fast["position"][inside, axis]
    else:
        reference = kin.quaternion_to_matrix(fast["quaternion"][inside][0])
        coordinate = np.array([
            rep.rotation_vector(kin.quaternion_to_matrix(q) @ reference.T)[axis - 3]
            for q in fast["quaternion"][inside]
        ])
    dt = float(np.median(np.diff(times)))
    window = max(3, int(round(args.velocity_window_s / dt)))
    velocity = np.gradient(smooth(coordinate, window), times)
    threshold = args.moving_mm_s / 1000.0 if axis < 3 else math.radians(args.moving_deg_s)
    others = np.delete(np.arange(6), axis)
    results = []
    curve_points = []
    for index, leg in enumerate(legs):
        leg_end = legs[index + 1]["t"] if index + 1 < len(legs) else t_end
        mask = (times >= leg["t"]) & (times < leg_end)
        if np.count_nonzero(mask) < 10:
            continue
        direction = int(leg["direction"])
        if direction == 0:
            # A rest: the goal sits at the start pose. Read where the arm settled from the
            # second half; approached from + it stalls on the + side, so the spring is < 0.
            tail = mask & (times >= leg["t"] + 0.5 * (leg_end - leg["t"]))
            results.append({"leg": leg["leg"], "direction": 0,
                            "approach": 1 if "plus" in leg["leg"] else -1,
                            "rest_spring": float(np.mean(spring[tail])),
                            "rest_offset": float(np.mean(-fast["error"][inside][tail, axis])),
                            "rest_speed": float(np.max(np.abs(velocity[tail])))})
            continue
        moving = direction * velocity[mask] > threshold
        sustain = max(1, int(round(args.sustain_s / dt)))
        run = np.convolve(moving.astype(int), np.ones(sustain, dtype=int), mode="valid")
        onsets = np.flatnonzero(run == sustain)
        w = direction * spring[mask]
        entry = {"leg": leg["leg"], "direction": direction}
        # Steady state: the goal moves at constant speed; past the transient the arm trails
        # it at the same speed, and the spring carries damping plus whatever the arm loses
        # (dead zone, friction). Read from the leg's second half, at least --steady-after-s in.
        steady_from = leg["t"] + max(args.steady_after_s, 0.5 * (leg_end - leg["t"]))
        steady = (times[mask] >= steady_from)
        if np.count_nonzero(steady) >= 10:
            entry["steady_speed"] = float(np.median(direction * velocity[mask][steady]))
            entry["steady_spring"] = float(np.median(w[steady]))
        # Every sample of the leg as (speed along the leg, lost wrench): spring minus the
        # controller's damping. On the real arm the slow legs never reach steady state (the
        # loss rises steeply over the first mm/s), so the loss curve is read sample by sample;
        # the arm's inertia is ignored (accelerations are small except at the turnarounds).
        step = max(1, int(round(args.curve_decimate_s / dt)))
        v_leg = direction * velocity[mask]
        lost_leg = w - 2.0 * math.sqrt(stiffness[axis]) * v_leg
        curve_points.extend(zip(v_leg[::step].tolist(), lost_leg[::step].tolist()))
        leg_velocity = direction * velocity[mask]
        leg_coordinate = direction * coordinate[mask]
        travel = args.min_travel_mm / 1000.0 if axis < 3 else math.radians(args.min_travel_deg)
        ahead = max(1, int(round(args.travel_window_s / dt)))
        accepted = None
        for candidate in onsets:
            # A real breakaway keeps going; the spring-back of an arm stopping after a
            # reversal moves a few hundredths of a degree and stops.
            later = min(len(w) - 1, int(candidate) + ahead)
            if leg_coordinate[later] - leg_coordinate[candidate] < travel:
                continue
            onset = int(candidate)
            # The threshold and the sustain window trigger late while the spring keeps
            # loading at K x goal speed: walk back to where the motion really began, but
            # never past the moment the spring started pulling the leg's way.
            while (onset > 0 and leg_velocity[onset - 1] > args.onset_fraction * threshold
                   and w[onset - 1] > 0.0):
                onset -= 1
            accepted = onset
            break
        onsets = [] if accepted is None else [accepted]
        if len(onsets):
            onset = int(onsets[0])
            # Moving before the spring even pulls its way means no static dead zone at all.
            # Only a leg that starts from rest has a breakaway; after a turnaround the arm
            # is still moving and unwinding.
            if leg["leg"].startswith("out"):
                entry["breakaway"] = max(float(w[onset]), 0.0)
            entry["breakaway_after_s"] = float(times[mask][onset] - leg["t"])
            tail = np.arange(len(w)) >= onset + sustain
            entry["creep"] = float(np.median(w[tail])) if np.any(tail) else float(w[onset])
            entry["tau_command_at_breakaway"] = fast["tau_command"][inside][mask][onset].tolist()
            entry["off_axis_error_at_breakaway"] = fast["error"][inside][mask][onset][others].tolist()
        elif leg["leg"].startswith("out"):
            entry["breakaway_above"] = float(np.max(w))
        results.append(entry)
    summary = {"legs": results}
    after_plus = [r["rest_spring"] for r in results if r.get("approach") == 1]
    after_minus = [r["rest_spring"] for r in results if r.get("approach") == -1]
    if after_plus and after_minus:
        p, m = float(np.mean(after_plus)), float(np.mean(after_minus))
        half = 0.5 * (m - p)
        summary["rest"] = {"after_plus": p, "after_minus": m, "dead_zone": half,
                           "bias": 0.5 * (m + p), "stall": half / stiffness[axis]}
    for name in ("breakaway", "creep"):
        plus = [r[name] for r in results if name in r and r["direction"] > 0]
        minus = [r[name] for r in results if name in r and r["direction"] < 0]
        if plus and minus:
            p, m = float(np.mean(plus)), float(np.mean(minus))
            summary[name] = {"plus": p, "minus": m, "dead_zone": 0.5 * (p + m), "bias": 0.5 * (p - m)}
    if "creep" in summary and goal_speed is not None:
        # While the arm creeps it moves at the goal's speed, so 2 sqrt(K) v of the spring
        # goes into the damper, not into breaking friction.
        damping = 2.0 * math.sqrt(stiffness[axis]) * goal_speed
        summary["creep"]["damping_share"] = damping
        summary["creep"]["dead_zone_minus_damping"] = summary["creep"]["dead_zone"] - damping
    if any("breakaway_above" in r for r in results):
        summary["no_breakaway_legs"] = sum("breakaway_above" in r for r in results)
    steady = [r for r in results if "steady_spring" in r]
    if steady:
        damping = 2.0 * math.sqrt(stiffness[axis])
        speed = float(np.mean([r["steady_speed"] for r in steady]))
        spring_mean = float(np.mean([r["steady_spring"] for r in steady]))
        # Averaged over both directions, so a constant bias cancels.
        summary["steady"] = {"goal_speed": goal_speed, "speed": speed, "spring": spring_mean,
                             "damping": damping * speed, "lost": spring_mean - damping * speed,
                             "points": [[r["steady_speed"], r["steady_spring"], r["direction"]] for r in steady]}
    summary["curve_points"] = curve_points
    return summary


def fit_loss_curve(points, translation: bool, speed_edges) -> dict:
    """Lost wrench against speed: binned medians, and lost = Fc tanh(v / v0) + b v.

    The tanh is a smoothed Coulomb step: a sim dead zone is a sharp step (v0 -> 0) with
    b = 0; the FR3 creeps, so its step is spread over v0. Only samples moving the leg's
    way are used (v > 0).
    """

    points = np.asarray(points, dtype=float)
    if len(points) < 50:
        return {}
    unit = 1000.0 if translation else math.degrees(1.0)
    v, lost = points[:, 0], points[:, 1]
    keep = v > 0.1 / unit
    v, lost = v[keep], lost[keep]
    bins = []
    edges = np.asarray(speed_edges, dtype=float) / unit
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (v >= lo) & (v < hi)
        if np.count_nonzero(inside) >= 10:
            bins.append({"speed_lo": float(lo), "speed_hi": float(hi), "n": int(np.count_nonzero(inside)),
                         "speed": float(np.median(v[inside])), "lost": float(np.median(lost[inside])),
                         "lost_p25": float(np.percentile(lost[inside], 25)),
                         "lost_p75": float(np.percentile(lost[inside], 75))})
    if len(bins) < 3:
        return {"samples": int(len(v)), "bins": bins}
    # Fitted to the bin medians (weight sqrt(n)): robust to the few samples a turnaround
    # or a velocity glitch puts far off the curve. Both directions are folded onto the
    # leg's direction, so a constant bias cancels and needs no term of its own.
    speed = np.array([b["speed"] for b in bins])
    median = np.array([b["lost"] for b in bins])
    weight = np.sqrt([b["n"] for b in bins])
    best = None
    for v0 in np.geomspace(0.05 / unit, 10.0 / unit, 80):
        design = np.column_stack((np.tanh(speed / v0), speed))
        solution, *_ = np.linalg.lstsq(design * weight[:, None], median * weight, rcond=None)
        rms = float(np.sqrt(np.sum(weight ** 2 * (median - design @ solution) ** 2) / np.sum(weight ** 2)))
        if best is None or rms < best[0]:
            best = (rms, v0, solution)
    rms, v0, (coulomb, viscous) = best
    return {"coulomb": float(coulomb), "v0": float(v0), "viscous": float(viscous),
            "rms_residual": rms, "samples": int(len(v)), "bins": bins}


def loss_curves(probes: dict, args) -> dict:
    """Pool every level's samples per pose and axis, then fit the loss curve."""

    pooled = {}
    for key, axes in probes.items():
        pose = key.rsplit("/L", 1)[0] if "/L" in key else key
        for axis, s in axes.items():
            pooled.setdefault(pose, {}).setdefault(axis, []).extend(s.get("curve_points", []))
    out = {}
    for pose, axes in pooled.items():
        for axis, points in axes.items():
            translation = axis in AXES[:3]
            fit = fit_loss_curve(points, translation,
                                 args.curve_edges_mm_s if translation else args.curve_edges_deg_s)
            if fit:
                out.setdefault(pose, {})[axis] = fit
    return out


def fit_speed_sweep(probes: dict, stiffness) -> dict:
    """Per pose and axis, across speed levels: spring = intercept + slope * speed.

    Every moving leg of every level is one point (both directions, folded onto the leg's
    direction, so a constant bias cancels in the fit). The intercept is the wrench the arm
    loses at vanishing speed: the dead zone in sim, Coulomb friction on the FR3. The slope is
    the damper 2 sqrt(K) plus any viscous loss; slope - 2 sqrt(K) is the viscous part.
    """

    grouped = {}
    for key, axes in probes.items():
        if "/L" not in key:
            continue
        pose = key.rsplit("/L", 1)[0]
        for axis, s in axes.items():
            if "steady" in s:
                grouped.setdefault(pose, {}).setdefault(axis, []).append(s["steady"])
    out = {}
    for pose, axes in grouped.items():
        for axis, levels in axes.items():
            if len(levels) < 2:
                continue
            points = np.array([p[:2] for lv in levels for p in lv["points"]])
            design = np.column_stack((np.ones(len(points)), points[:, 0]))
            (intercept, slope), *_ = np.linalg.lstsq(design, points[:, 1], rcond=None)
            residual = points[:, 1] - design @ (intercept, slope)
            damping = 2.0 * math.sqrt(stiffness[AXES.index(axis)])
            out.setdefault(pose, {})[axis] = {
                "intercept": float(intercept), "slope": float(slope), "damping": damping,
                "viscous": float(slope - damping),
                "rms_residual": float(np.sqrt(np.mean(residual ** 2))), "points": len(points),
                "levels": [{"goal_speed": lv["goal_speed"], "speed": lv["speed"], "spring": lv["spring"],
                            "lost": lv["lost"]} for lv in levels],
            }
    return out


def analyze_probes(data, events, parameters, args) -> dict:
    fast = data.get("fast")
    if fast is None:
        return {}
    k_t = float(parameters.get("translational_stiffness", 565.0))
    k_r = float(parameters.get("rotational_stiffness", 28.0))
    stiffness = [k_t] * 3 + [k_r] * 3
    out = {}
    for axis_segment in segments(events, "probe_axis_start", "probe_axis_end"):
        legs = [e for e in events if e["event"] == "probe_leg_start"
                and e.get("label") == axis_segment["label"] and e.get("axis") == axis_segment["axis"]
                and axis_segment["t"] <= e["t"] <= axis_segment["t_end"]]
        if not legs:
            continue
        axis = AXES.index(axis_segment["axis"])
        payload = next((e.get("payload") for e in reversed(events) if e["event"] == "probe_start"
                        and e.get("label") == axis_segment["label"] and e["t"] <= axis_segment["t"]), None)
        key = f"{payload}/{axis_segment['label']}"
        if int(axis_segment.get("levels") or 1) > 1:
            key += f"/L{axis_segment['level']}"
        out.setdefault(key, {})[axis_segment["axis"]] = (
            analyze_probe_axis(fast, axis, legs, axis_segment["t_end"], stiffness, args,
                               axis_segment.get("speed")))
    return out


# --- payload --------------------------------------------------------------------
def hold_means(data, events, args) -> list[dict]:
    state = data.get("state")
    if state is None:
        return []
    out = []
    for hold in segments(events, "static_start", "static_end"):
        begin = hold["t"] + args.hold_skip_fraction * (hold["t_end"] - hold["t"])
        mask = (state["t"] >= begin) & (state["t"] <= hold["t_end"])
        if np.count_nonzero(mask) < 10:
            continue
        out.append({"label": hold["label"], "payload": hold["payload"],
                    "q": state["q"][mask].mean(axis=0), "tau_ext": state["tau_ext"][mask].mean(axis=0),
                    "tau_ext_std": state["tau_ext"][mask].std(axis=0),
                    "dq_max": float(np.max(np.abs(state["dq"][mask])))})
    return out


def payload_columns(q) -> np.ndarray:
    """7 x 4: joint torque per unit of (m, p_x, p_y, p_z) for the weight of a flange payload."""

    jacobian = kin.flange_jacobian(q)
    rotation = kin.flange_transform(q)[:3, :3]
    skew = np.array([[0.0, -GRAVITY[2], GRAVITY[1]], [GRAVITY[2], 0.0, -GRAVITY[0]],
                     [-GRAVITY[1], GRAVITY[0], 0.0]])
    mass = jacobian.T @ np.concatenate((GRAVITY, np.zeros(3)))
    moment = jacobian.T @ np.vstack((np.zeros((3, 3)), -skew @ rotation))
    return np.column_stack((mass, moment))


def fit_payload(holds, offsets: bool) -> dict:
    rows = [payload_columns(h["q"]) for h in holds]
    design = np.vstack([np.hstack((r, np.eye(7))) if offsets else r for r in rows])
    target = np.concatenate([h["tau_ext"] for h in holds])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ solution
    m, p = float(solution[0]), solution[1:4]
    return {
        "m": m, "p": p.tolist(), "com": (p / m).tolist() if abs(m) > 1.0e-3 else None,
        "offsets": solution[4:].tolist() if offsets else None,
        "rms_residual_per_joint": np.sqrt(np.mean(residual.reshape(-1, 7) ** 2, axis=0)).tolist(),
        "condition_number": float(np.linalg.cond(design)),
        "holds": len(holds),
    }


def analyze_payload(holds, report) -> dict:
    if not holds:
        return {}
    out = {}
    by_level = {}
    for h in holds:
        by_level.setdefault(h["payload"], []).append(h)
    for level, group in by_level.items():
        out[level] = fit_payload(group, offsets=True)
    if "none" in by_level and "modelled" in by_level:
        modelled = {h["label"]: h for h in by_level["modelled"]}
        pairs = [{"q": h["q"], "tau_ext": h["tau_ext"] - modelled[h["label"]]["tau_ext"]}
                 for h in by_level["none"] if h["label"] in modelled]
        if pairs:
            diff = fit_payload(pairs, offsets=False)
            applied = report.get("payloads", {}).get("modelled", {}).get("load", {})
            sign = 1.0 if diff["m"] > 0.0 else -1.0
            diff["sign_convention"] = sign
            diff["applied_mass"] = applied.get("m")
            diff["applied_com"] = applied.get("com")
            out["none_minus_modelled"] = diff
            if "none" in out:
                none = out["none"]
                out["identified_hand"] = {"m": sign * none["m"], "com": none["com"],
                                          "desk": {"m": applied.get("m"), "com": applied.get("com")},
                                          "training": TRAINING_HAND}
            if "modelled" in out:
                out["model_error"] = {"m": sign * out["modelled"]["m"], "p": [sign * v for v in out["modelled"]["p"]]}
    return out


# --- output ---------------------------------------------------------------------
def print_probe(probes, sim_dead_zone) -> None:
    """The dead zone is where the arm comes to rest, approached from either side."""

    if not probes:
        return
    print(f"\n{'pose':<14}{'axis':>5}{'rest after +/-':>17}{'dead zone':>12}{'stall':>10}{'bias':>8}"
          f"{'breakaway +/-':>16}{'creep-damp':>11}{'expected':>9}")
    for pose, axes in probes.items():
        for axis, s in axes.items():
            translation = axis in AXES[:3]
            unit = "N " if translation else "Nm"
            r, b, c = s.get("rest"), s.get("breakaway", {}), s.get("creep", {})
            if r is None:
                print(f"{pose:<14}{axis:>5}   {s.get('error') or 'no rest legs (older probe profile)'}")
                continue
            stall = f"{1000 * r['stall']:.1f} mm" if translation else f"{math.degrees(r['stall']):.1f} deg"
            expected = "" if sim_dead_zone is None else f"{sim_dead_zone[AXES.index(axis)]:.2f}"
            print(f"{pose:<14}{axis:>5}{r['after_plus']:>+8.2f}/{r['after_minus']:<+8.2f}{r['dead_zone']:>9.2f} {unit}"
                  f"{stall:>10}{r['bias']:>+8.2f}{b.get('plus', float('nan')):>7.2f}/{b.get('minus', float('nan')):<7.2f}"
                  f"{c.get('dead_zone_minus_damping', float('nan')):>10.2f}{expected:>9}")


def print_sweep(sweep, sim_dead_zone) -> None:
    """Steady lost wrench per speed level, and the line through all levels."""

    if not sweep:
        return
    print(f"\nspeed sweep: steady spring = intercept + slope x speed (intercept = dead zone / Coulomb)")
    print(f"{'pose':<14}{'axis':>5}  {'per level: speed -> lost (spring - 2 sqrt(K) v)':<58}"
          f"{'intercept':>10}{'viscous':>9}{'rms':>7}{'expected':>9}")
    for pose, axes in sweep.items():
        for axis, f in axes.items():
            translation = axis in AXES[:3]
            scale, unit = (1000.0, "mm/s") if translation else (math.degrees(1.0), "deg/s")
            levels = " ".join(f"{scale * lv['speed']:.1f}->{lv['lost']:+.2f}" for lv in f["levels"])
            expected = "" if sim_dead_zone is None else f"{sim_dead_zone[AXES.index(axis)]:.2f}"
            print(f"{pose:<14}{axis:>5}  {levels + ' (' + unit + ')':<58}{f['intercept']:>10.2f}"
                  f"{f['viscous']:>9.1f}{f['rms_residual']:>7.2f}{expected:>9}")


def print_curves(curves, sim_dead_zone) -> None:
    """Lost wrench vs speed from every moving sample: lost = Fc tanh(v/v0) + b v + c."""

    if not curves:
        return
    print("\nloss curve (all moving samples): lost = Fc tanh(v/v0) + b v, fitted to the binned medians")
    for pose, axes in curves.items():
        for axis, f in axes.items():
            translation = axis in AXES[:3]
            unit, name = (1000.0, "mm/s") if translation else (math.degrees(1.0), "deg/s")
            expected = "" if sim_dead_zone is None else f" expected {sim_dead_zone[AXES.index(axis)]:.2f}"
            if "coulomb" in f:
                viscous = (f"b {f['viscous']:6.1f} N s/m" if translation
                           else f"b {f['viscous']:6.2f} N m s/rad")
                print(f"{pose:<14}{axis:>4}  Fc {f['coulomb']:5.2f}  v0 {unit * f['v0']:5.2f} {name}  "
                      f"{viscous}  rms {f['rms_residual']:.2f}{expected}")
            else:
                print(f"{pose:<14}{axis:>4}  too few speed bins for a fit")
            print("      " + "  ".join(f"{unit * b['speed']:.2g}:{b['lost']:+.2f}" for b in f["bins"]))


def print_payload(payload) -> None:
    if not payload:
        return
    print()
    for level in ("modelled", "none", "none_minus_modelled"):
        if level in payload:
            f = payload[level]
            com = "-" if f["com"] is None else np.array2string(1000 * np.asarray(f["com"]), precision=1)
            print(f"{level:<20} m {f['m']:+.3f} kg, com {com} mm, rms residual "
                  f"{np.array2string(np.asarray(f['rms_residual_per_joint']), precision=2)} Nm, "
                  f"cond {f['condition_number']:.0f}, {f['holds']} holds")
    if "identified_hand" in payload:
        h = payload["identified_hand"]
        print(f"identified hand: {h['m']:.3f} kg at {np.array2string(1000 * np.asarray(h['com']), precision=1)} mm"
              f" (Desk {h['desk']['m']} kg at {h['desk']['com']}, training {TRAINING_HAND})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("run", type=Path)
    p.add_argument("--velocity-window-s", type=float, default=0.05)
    p.add_argument("--moving-mm-s", type=float, default=0.5)
    p.add_argument("--moving-deg-s", type=float, default=0.15)
    p.add_argument("--sustain-s", type=float, default=0.05)
    p.add_argument("--min-travel-mm", type=float, default=0.2)
    p.add_argument("--min-travel-deg", type=float, default=0.1)
    p.add_argument("--travel-window-s", type=float, default=0.5)
    p.add_argument("--onset-fraction", type=float, default=0.3,
                   help="breakaway is where the velocity first rose above this share of the threshold")
    p.add_argument("--hold-skip-fraction", type=float, default=0.4)
    p.add_argument("--steady-after-s", type=float, default=0.6,
                   help="steady state is read from the leg's second half, but not before this")
    p.add_argument("--curve-decimate-s", type=float, default=0.02, help="loss-curve sample spacing")
    p.add_argument("--curve-edges-mm-s", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120])
    p.add_argument("--curve-edges-deg-s", type=float, nargs="+",
                   default=[0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30])
    args = p.parse_args(argv)
    report = json.loads((args.run / "report.json").read_text())
    events = report["events"]
    windows = [(s["t"], s["t_end"]) for s in segments(events, "probe_start", "probe_end")]
    windows += [(s["t"], s["t_end"]) for s in segments(events, "static_start", "static_end")]
    if not windows:
        raise SystemExit("no probe or hold segments in the report")
    data = read_bag(args.run / "bag", windows)
    parameters = report.get("controller_parameters", {})
    probes = analyze_probes(data, events, parameters, args)
    dead_zone = parameters.get("task_wrench_dead_zone")
    if isinstance(dead_zone, str):
        dead_zone = [float(v) for v in dead_zone.split()]
    if dead_zone is not None and not any(float(v) > 0 for v in dead_zone):
        dead_zone = None
    payload = analyze_payload(hold_means(data, events, args), report)
    stiffness = ([float(parameters.get("translational_stiffness", 565.0))] * 3
                 + [float(parameters.get("rotational_stiffness", 28.0))] * 3)
    sweep = fit_speed_sweep(probes, stiffness)
    curves = loss_curves(probes, args)
    # The raw (speed, lost) samples go to an npz for plotting, not into the JSON.
    raw = {}
    for key, axes in probes.items():
        for axis, s in axes.items():
            points = s.pop("curve_points", [])
            if points:
                raw[f"{key}/{axis}".replace("/", "__")] = np.asarray(points)
    if raw:
        np.savez_compressed(args.run / "loss_curve_points.npz", **raw)
    print_probe(probes, dead_zone)
    print_sweep(sweep, dead_zone)
    print_curves(curves, dead_zone)
    print_payload(payload)
    out = {"probes": probes, "speed_sweep": sweep, "loss_curve": curves, "payload": payload,
           "configured_dead_zone": dead_zone}
    (args.run / "analysis.json").write_text(json.dumps(out, indent=2, default=rep._plain) + "\n")
    print(f"\nwrote {args.run / 'analysis.json'}")


if __name__ == "__main__":
    main()
