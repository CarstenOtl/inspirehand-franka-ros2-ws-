"""The repeatability test's motion plan and its split of error into friction and load."""

from __future__ import annotations

from argparse import Namespace
import math
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("franka_trajectory_replay")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from franka_trajectory_replay import kinematics as kin  # noqa: E402
from utils import osc_pose_repeatability as test_tool  # noqa: E402

A = np.array([0.30, 0.20, 0.30])
B = np.array([0.30, 0.40, 0.30])


def test_every_pose_is_reached_from_both_sides_for_every_orientation():
    moves = test_tool.build_moves(A, B, [("o0", np.eye(3)), ("o1", np.eye(3))], 2, 0.05)
    arrivals = [m for m in moves if m.is_arrival]
    assert len(arrivals) == 2 * 2 * 4
    for label in ("A", "B"):
        for orientation in ("o0", "o1"):
            sides = [
                m.approach for m in arrivals
                if m.position_label == label and m.orientation_label == orientation
            ]
            assert sorted(sides) == [-1, -1, 1, 1]


def test_approach_sign_is_the_real_direction_of_the_last_leg():
    moves = test_tool.build_moves(A, B, [("o0", np.eye(3))], 3, 0.05)
    axis = (B - A) / np.linalg.norm(B - A)
    previous = A
    for move in moves:
        if move.is_arrival:
            assert np.sign(np.dot(move.position - previous, axis)) == move.approach
        previous = move.position


def test_clip_speed_is_the_spring_at_the_clip_over_the_damping():
    assert test_tool.expected_clip_speed_m_s(565.0, 0.02) == pytest.approx(0.2377, abs=1e-3)


def _result(label, side, error):
    return {
        "position_label": label,
        "orientation_label": "o0",
        "approach": side,
        "cycle": 0,
        "position_error_m": list(error),
        "rotation_error_rad": [0.0, 0.0, 0.0],
        "in_return_gate_at_end": True,
        "peak_speed_m_s": 0.2,
    }


def test_summary_separates_friction_from_load():
    # 1 mm of friction each way along y, and 1 mm of sag in z.
    results = []
    for i in range(3):
        results.append(_result("A", 1, [0.0, -0.001, -0.001]))
        results.append(_result("A", -1, [0.0, 0.001, -0.001]))
    target = test_tool.summarize(results, 565.0)["targets"]["A/o0"]
    assert target["hysteresis_mm"] == pytest.approx([0.0, -2.0, 0.0], abs=1e-9)
    assert target["bias_mm"] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
    assert target["implied_extra_load_kg"] == pytest.approx(0.0576, abs=1e-3)


def test_iso_repeatability_is_zero_for_identical_arrivals():
    assert test_tool.iso_repeatability(np.ones((5, 3))) == pytest.approx(0.0)


def test_ik_reaches_the_pose_and_the_euler_check_sees_a_seam():
    tool = np.eye(4)
    rotation = kin.flange_transform(kin.READY_POSE)[:3, :3]
    q = test_tool.solve_ik(A + [0.0, 0.0, 0.2], rotation, kin.READY_POSE.copy(), tool)
    assert q is not None
    pose = kin.flange_transform(q)
    assert np.linalg.norm(pose[:3, 3] - (A + [0.0, 0.0, 0.2])) < 1e-4

    # Yaw 234 and 236 deg sit on either side of the 235 deg seam: 2 deg apart, a 360 deg jump.
    below = test_tool.clip_euler(test_tool.rotation_from_vector([0, 0, math.radians(234 - 180)]), 235.0)
    above = test_tool.clip_euler(test_tool.rotation_from_vector([0, 0, math.radians(236 - 180)]), 235.0)
    assert abs(below[2] - above[2]) == pytest.approx(2.0 * math.pi - math.radians(2.0), abs=1e-6)


def test_analyze_arrival_reads_the_rest_window_and_the_gate():
    time_s = np.arange(0.0, 3.0, 0.02)
    goal = np.array([0.3, 0.2, 0.3])
    position = np.tile(goal + [0.0, 0.0, -0.002], (len(time_s), 1))
    position[time_s < 1.0, 1] -= 0.05
    quaternion = kin.matrix_to_quaternion(np.eye(3))
    samples = {
        "time_s": time_s,
        "position": position,
        "quaternion_xyzw": np.tile(quaternion, (len(time_s), 1)),
        "position_clipped": np.zeros(len(time_s)),
        "tau_task": np.zeros((len(time_s), 7)),
        "q": np.zeros((len(time_s), 7)),
    }
    arrival = {
        "position_label": "A", "orientation_label": "o0", "approach": 1, "cycle": 0,
        "goal_position": goal.tolist(), "goal_rotation": np.eye(3).tolist(),
        "t_start": 0.0, "t_goal_end": 0.5, "t_end": 2.9,
    }
    args = Namespace(settle_window=0.5, return_tolerance_mm=5.0, return_tolerance_deg=5.0)
    result = test_tool.analyze_arrival(samples, arrival, args)
    assert result["position_error_m"] == pytest.approx([0.0, 0.0, -0.002], abs=1e-9)
    assert result["in_return_gate_at_end"] is True
    assert result["time_to_return_gate_s"] == pytest.approx(0.5, abs=0.03)


def _plan_args():
    return Namespace(
        speed=0.5, acceleration=2.5, angular_speed=1.5, angular_acceleration=8.0,
        transit_speed=0.05, transit_angular_speed=0.25, settle=2.0, waypoint_dwell=0.15,
        min_tcp_z=0.15, min_tcp_radius=0.2, bolt_keepout=0.15, bolt_clear_z=0.30,
        min_joint_margin=0.1, max_start_joint_error=1.0, max_posture_move_tcp=0.1,
        min_singular_value=0.03, yaw_seam_deg=235.0,
    )


# Where the arm stood on 2026-11-10, after a rollout: the joint homing refused it (3.03 rad on j7).
AFTER_ROLLOUT_Q = np.array([-0.3502, 0.3955, -0.0285, -1.7889, 1.3698, 2.0279, -1.8464])


def test_transit_lifts_with_the_orientation_held_then_goes_straight_to_the_start():
    tool = np.eye(4)
    goal_rotation = kin.flange_transform(kin.READY_POSE)[:3, :3]
    goal_q = kin.READY_POSE.copy()
    lift, line = test_tool.build_transit(AFTER_ROLLOUT_Q, A, goal_rotation, goal_q, tool, 0.1)
    assert lift.posture_goal is None
    assert line.posture_goal == pytest.approx(goal_q)
    pose = kin.flange_transform(AFTER_ROLLOUT_Q)
    assert lift.position == pytest.approx(pose[:3, 3] + [0.0, 0.0, 0.1])
    assert lift.rotation == pytest.approx(pose[:3, :3])
    assert line.position == pytest.approx(A)
    assert lift.transit and line.transit and not line.is_arrival


def test_transit_may_not_move_towards_the_bolt_inside_the_keepout():
    args = _plan_args()
    # Bolt at base (0.59, 0): 0.10 m away and then moving closer.
    last = test_tool._distance_to_bolt_m([0.69, 0.0, 0.2])
    with pytest.raises(ValueError, match="towards the bolt"):
        test_tool._check_transit_tcp(np.array([0.68, 0.0, 0.2]), last, 0.2, args, "here")
    assert test_tool._check_transit_tcp(np.array([0.70, 0.0, 0.2]), last, 0.2, args, "here") > last
    # High enough, it may pass over.
    assert test_tool._check_transit_tcp(np.array([0.68, 0.0, 0.34]), last, 0.34, args, "here") < last


def test_the_nullspace_path_starts_and_ends_on_the_ik():
    path = np.array([[0.0] * 7, [1.0] * 7, [3.0] * 7])
    assert test_tool._interpolate_path(path, 0.0) == pytest.approx([0.0] * 7)
    assert test_tool._interpolate_path(path, 0.75) == pytest.approx([2.0] * 7)
    assert test_tool._interpolate_path(path, 1.0) == pytest.approx([3.0] * 7)


def test_each_orientation_block_starts_with_a_slow_turn_in_place():
    moves = test_tool.build_moves(A, B, [("o0", np.eye(3)), ("o1", test_tool.rotation_from_vector([0.5, 0, 0]))], 1, 0.05)
    turns = [i for i, m in enumerate(moves) if m.wait_arrival]
    assert len(turns) == 2 and turns[0] == 0
    # The second block's turn happens at the pose the first block ended on.
    assert np.allclose(moves[turns[1]].position, moves[turns[1] - 1].position)
    assert moves[turns[1]].transit and not moves[turns[1]].is_arrival


def test_posture_move_refuses_a_long_joint_move_or_a_wandering_tcp():
    args = _plan_args()
    start = np.array([0.391, -0.657, 0.015, -2.574, 0.009, 1.917, 1.184])
    tool = np.eye(4)
    test_tool.check_posture_move(start + 0.02, start, tool, args)
    with pytest.raises(ValueError, match="start posture"):
        test_tool.check_posture_move(start + np.array([1.5, 0, 0, 0, 0, 0, 0]), start, tool, args)
    with pytest.raises(ValueError, match="posture move takes the TCP"):
        test_tool.check_posture_move(start + np.array([0.9, 0, 0, 0, 0, 0, 0]), start, tool, args)
