#!/usr/bin/env python3
"""Plot a physical rollout's controlled point against the nominal teacher/nut."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("teacher_episode")
    parser.add_argument(
        "--config",
        default=str(
            WORKSPACE_ROOT / "src/inspire_franka_trajectory_replay/config/replay.yaml"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from franka_trajectory_replay.kinematics import tool_transform
    from franka_trajectory_replay.runconfig import load_config
    from inspire_franka_trajectory_replay.extract import (
        pose16_from_joint_angles,
        tcp_from_pose16,
    )
    from policy_rollout.hardware import TrainingFrameAdapter

    with np.load(args.recording, allow_pickle=False) as source:
        actual_time = source["sample_time_s"].copy()
        actual_q = source["joint_position"][:, :7].copy()
    with np.load(args.teacher_episode, allow_pickle=False) as source:
        teacher_time = source["sample_time_s"].copy()
        teacher_q = source["student_osc_proprio"][:, :7].copy()
        nut_world = source["nut_pos"].copy()

    config = load_config(args.config)
    tool = tool_transform(
        config["tcp"]["offset_xyz"], config["tcp"]["offset_rpy"]
    )
    actual_tcp, _ = tcp_from_pose16(pose16_from_joint_angles(actual_q), tool)
    teacher_tcp, _ = tcp_from_pose16(pose16_from_joint_angles(teacher_q), tool)
    frame = TrainingFrameAdapter()
    nut_base = np.stack(
        [frame.pose_world_to_base(position, [1.0, 0.0, 0.0, 0.0])[0] for position in nut_world]
    )
    comparison_end = min(float(actual_time[-1]), float(teacher_time[-1]))
    teacher_mask = teacher_time <= comparison_end
    nominal_nut = nut_base[0]
    actual_distance = np.linalg.norm(actual_tcp - nominal_nut, axis=1)
    teacher_distance = np.linalg.norm(teacher_tcp - nut_base, axis=1)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(15, 11))
    axis3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis3d.plot(*teacher_tcp.T, color="0.7", alpha=0.55, label="teacher, full")
    axis3d.plot(
        *teacher_tcp[teacher_mask].T,
        color="tab:orange",
        linestyle="--",
        label="teacher, same elapsed time",
    )
    axis3d.plot(*actual_tcp.T, color="tab:blue", linewidth=2.0, label="physical")
    axis3d.scatter(*nominal_nut, marker="*", s=180, color="black", label="nominal nut")
    axis3d.scatter(*actual_tcp[0], marker="o", color="tab:green", label="physical start")
    axis3d.scatter(*actual_tcp[-1], marker="x", s=80, color="tab:red", label="physical end")
    axis3d.set_xlabel("base x (m)")
    axis3d.set_ylabel("base y (m)")
    axis3d.set_zlabel("base z (m)")
    axis3d.set_title("Controlled-point path in fr3_link0")
    axis3d.legend(fontsize=8)

    def projection(axis, first, second, first_label, second_label, title):
        axis.plot(
            teacher_tcp[teacher_mask, first],
            teacher_tcp[teacher_mask, second],
            color="tab:orange",
            linestyle="--",
            label="teacher",
        )
        axis.plot(actual_tcp[:, first], actual_tcp[:, second], color="tab:blue", label="physical")
        axis.scatter(nominal_nut[first], nominal_nut[second], marker="*", s=150, color="black")
        axis.scatter(actual_tcp[0, first], actual_tcp[0, second], color="tab:green")
        axis.scatter(actual_tcp[-1, first], actual_tcp[-1, second], marker="x", s=70, color="tab:red")
        axis.set_xlabel(first_label)
        axis.set_ylabel(second_label)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.axis("equal")

    projection(figure.add_subplot(2, 2, 2), 0, 1, "base x (m)", "base y (m)", "Top view")
    projection(figure.add_subplot(2, 2, 3), 0, 2, "base x (m)", "base z (m)", "Side view")
    distance_axis = figure.add_subplot(2, 2, 4)
    distance_axis.plot(
        teacher_time[teacher_mask],
        teacher_distance[teacher_mask],
        color="tab:orange",
        linestyle="--",
        label="teacher",
    )
    distance_axis.plot(actual_time, actual_distance, color="tab:blue", label="physical")
    distance_axis.set_xlabel("elapsed rollout time (s)")
    distance_axis.set_ylabel("controlled-point distance to nominal nut (m)")
    distance_axis.set_title("Distance to nominal nut centre")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend()
    figure.suptitle("Physical rollout versus nominal teacher workspace path")
    figure.tight_layout()
    plot_path = output / "workspace_path_vs_teacher.png"
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)

    report = {
        "recording": str(Path(args.recording).resolve()),
        "teacher_episode": str(Path(args.teacher_episode).resolve()),
        "actual_duration_s": float(actual_time[-1]),
        "nominal_nut_base_m": nominal_nut.tolist(),
        "physical_tcp_start_m": actual_tcp[0].tolist(),
        "physical_tcp_end_m": actual_tcp[-1].tolist(),
        "physical_distance_to_nut_m": {
            "start": float(actual_distance[0]),
            "minimum": float(np.min(actual_distance)),
            "end": float(actual_distance[-1]),
        },
        "teacher_distance_to_nut_during_physical_window_m": {
            "start": float(teacher_distance[0]),
            "minimum": float(np.min(teacher_distance[teacher_mask])),
            "end": float(teacher_distance[np.flatnonzero(teacher_mask)[-1]]),
        },
        "plot": str(plot_path),
        "qualification": (
            "Positions are reconstructed from recorded joints with nominal FR3 FK; "
            "the nut marker is the teacher episode position mapped into fr3_link0."
        ),
    }
    report_path = output / "workspace_path_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
