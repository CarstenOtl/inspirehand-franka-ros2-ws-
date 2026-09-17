#!/usr/bin/env python3
"""Compare where a recorded rollout sent the grasp with where the policy aimed it.

    python3 apps/policy_rollout/utils/analyze_rollout_motion.py \\
        logs/policy_rollout/hardware-20260912-184327-287247 [--plot out.png]

The recording holds joint positions and filtered policy actions but no poses.
This rebuilds the fingertip grasp frame with forward kinematics of
``inspire_franka.urdf.xacro`` (hand on the flange, thumb-yaw overlay applied,
reset approach axis transported like training) and decodes each action into
its unclipped target, the pose the policy is steering towards, in the
training world. A healthy rollout closes the distance (the MuJoCo student
reaches ~10 mm within 50 steps); a frame or unit bug shows the grasp walking
away from a steady goal. Needs a sourced ROS
workspace (xacro, inspire_hand_driver).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout import forge_osc as fo  # noqa: E402


def _rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class UrdfKinematics:
    """Minimal serial-chain FK over the flange-mounted description."""

    def __init__(self):
        import xacro

        path = WORKSPACE_ROOT / "src/inspire_franka_description/urdf/inspire_franka.urdf.xacro"
        root = ET.fromstring(xacro.process_file(str(path), mappings={"hand_mount": "flange"}).toxml())
        self.joints = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            axis = joint.find("axis")
            mimic = joint.find("mimic")
            self.joints[joint.find("child").get("link")] = {
                "name": joint.get("name"),
                "parent": joint.find("parent").get("link"),
                "type": joint.get("type"),
                "xyz": np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]),
                "rpy": [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()],
                "axis": np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.array([0.0, 0.0, 1.0]),
                "mimic": (mimic.get("joint"), float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))) if mimic is not None else None,
            }

    def pose(self, link: str, positions: dict) -> np.ndarray:
        chain = []
        while link in self.joints and link != "fr3_link0":
            chain.append(self.joints[link])
            link = chain[-1]["parent"]
        transform = np.eye(4)
        for joint in reversed(chain):
            fixed = np.eye(4)
            fixed[:3, :3] = _rpy_matrix(*joint["rpy"])
            fixed[:3, 3] = joint["xyz"]
            value = positions.get(joint["name"], 0.0)
            if joint["mimic"]:
                leader, multiplier, offset = joint["mimic"]
                value = positions.get(leader, 0.0) * multiplier + offset
            motion = np.eye(4)
            if joint["type"] in ("revolute", "continuous"):
                motion[:3, :3] = _axis_angle(joint["axis"], value)
            elif joint["type"] == "prismatic":
                motion[:3, 3] = joint["axis"] * value
            transform = transform @ fixed @ motion
        return transform


def physical_positions(q10) -> dict:
    """Recorded policy coordinates -> URDF joint positions (overlay re-applied)."""

    from inspire_hand_driver import command_overlays, kinematics as kin

    positions = {f"fr3_joint{i + 1}": float(v) for i, v in enumerate(q10[:7])}
    dof = kin.dof_index(command_overlays.THUMB_ABDUCTION_JOINT)
    ratio = command_overlays.apply_open_ratio_overlay(dof, kin.rad_to_open_ratio(dof, float(q10[7])))
    positions["thumb_proximal_yaw_joint"] = kin.open_ratio_to_rad(dof, ratio)
    positions["thumb_proximal_pitch_joint"] = float(q10[8])
    positions["index_proximal_joint"] = float(q10[9])
    return positions


def analyze(run_dir: Path):
    data = np.load(run_dir / "data" / "rollout_data.npz")
    q = data["joint_position"]
    actions = data["filtered_native_action"]
    t = data["sample_time_s"] - data["sample_time_s"][0]
    kinematics = UrdfKinematics()
    q_yaw = fo.quat_from_euler_xyz(0.0, 0.0, math.pi)
    world_rotation = fo.matrix_from_quat(q_yaw)

    def world(point):
        return fo.ROBOT_BASE_POSITION + world_rotation @ point

    rows = []
    z_transport = None
    reference_quaternion = None
    for k in range(len(q)):
        positions = physical_positions(q[k])
        flange = kinematics.pose("fr3_link8", positions)
        thumb = world(kinematics.pose("thumb_tip", positions)[:3, 3])
        index = world(kinematics.pose("index_tip", positions)[:3, 3])
        flange_quaternion = fo.quat_mul(q_yaw, fo.quat_from_matrix(flange[:3, :3]))
        if z_transport is None:
            z_transport = fo.reset_z_transport(thumb, index, world(flange[:3, 3]), flange_quaternion)
        grasp_position, grasp_rotation = fo.hand_grasp_frame(
            thumb, index, fo.quat_rotate(flange_quaternion, z_transport)
        )
        grasp_quaternion = fo.quat_from_matrix(grasp_rotation)
        if reference_quaternion is None:
            reference_quaternion = grasp_quaternion
        goal = fo.decode_action_target(
            actions[k],
            fo.GraspFrameState(grasp_position, grasp_quaternion, np.zeros(3), np.zeros(3), np.zeros((6, 7))),
            clip=False,
        )
        rows.append({
            "t": float(t[k]),
            "grasp": grasp_position,
            "goal": goal.pos,
            "distance_to_goal_m": float(np.linalg.norm(goal.pos - grasp_position)),
            "rotation_from_start_deg": math.degrees(
                2.0 * math.acos(min(1.0, abs(float(np.dot(grasp_quaternion, reference_quaternion)))))
            ),
        })
    return rows, t


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rows", type=int, default=12, help="table rows to print")
    parser.add_argument("--plot", type=Path, default=None, help="save a PNG of the traces")
    args = parser.parse_args(argv)

    report_path = args.run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print("report:", {k: report.get(k) for k in ("status", "error", "steps", "missed_policy_deadlines", "limited_policy_targets")})
    rows, t = analyze(args.run_dir)
    dt = np.diff(t)
    if len(dt):
        print(f"{len(rows)} steps over {t[-1]:.1f} s: loop period median {1000 * np.median(dt):.0f} ms, max {1000 * dt.max():.0f} ms")
    print(f"bolt tip (training world) {fo.BOLT_TIP_POSITION.round(3)}")
    # Orientation goals are not compared: while turning, the policy aims its yaw
    # far beyond the per-step clip even in successful MuJoCo rollouts.
    print("   t [s]  grasp (world) [m]          policy goal (world) [m]    |goal-grasp| [mm]  rot from start [deg]")
    for row in rows[:: max(1, len(rows) // args.rows)] + rows[-1:]:
        print(f"  {row['t']:6.1f}  {np.array2string(row['grasp'], precision=3):25s}  "
              f"{np.array2string(row['goal'], precision=3):25s}  {1000 * row['distance_to_goal_m']:8.0f}"
              f"  {row['rotation_from_start_deg']:18.1f}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grasp = np.array([r["grasp"] for r in rows])
        goal = np.array([r["goal"] for r in rows])
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for axis, label in enumerate("xyz"):
            (line,) = axes[0].plot(t, grasp[:, axis], label=f"grasp {label}")
            axes[0].plot(t, goal[:, axis], "--", color=line.get_color(), label=f"goal {label}")
        axes[0].set_ylabel("training world [m]")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(t, [r["rotation_from_start_deg"] for r in rows], label="grasp rotation from start")
        axes[1].set_ylabel("deg")
        axes[1].set_xlabel("time [s]")
        axes[1].legend(fontsize=8)
        fig.suptitle(args.run_dir.name)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
