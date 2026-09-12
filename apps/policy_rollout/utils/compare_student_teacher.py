#!/usr/bin/env python3
"""Compare a MuJoCo student rollout with the recorded ForgeUltra teacher episode.

Both runs perform the same cyclic threading task but at their own pace, so the
meaningful alignment is per threading cycle and per process phase, not per
wall-clock sample. For every (cycle, phase) segment present in both runs the
script resamples onto normalized phase time and reports, per joint, the RMS
and maximum difference, plus task-level traces (nut rotation, grasp frame,
phase timing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from policy_rollout.observation import FORGE_POLICY_JOINT_NAMES  # noqa: E402

DEFAULT_STUDENT = APP_ROOT / "checkpoints" / "reference_episode" / "student_rollout"
DEFAULT_TEACHER = (
    APP_ROOT / "checkpoints" / "reference_episode" / "episode_000_sequential_threading.npz"
)
PHASES = ("policy", "follow_waypoints", "return_to_reset")


def segments(phases: np.ndarray, cycles: np.ndarray) -> list[tuple[int, str, int, int]]:
    """Split a run into ``(cycle, phase, start, stop)`` segments."""

    result: list[tuple[int, str, int, int]] = []
    start = 0
    for index in range(1, len(phases) + 1):
        boundary = (
            index == len(phases)
            or phases[index] != phases[start]
            or cycles[index] != cycles[start]
        )
        if boundary:
            result.append((int(cycles[start]), str(phases[start]), start, index))
            start = index
    return result


def resample(values: np.ndarray, count: int) -> np.ndarray:
    """Resample a segment onto ``count`` normalized phase-time points."""

    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    return np.stack(
        [np.interp(target, source, values[:, j]) for j in range(values.shape[1])],
        axis=1,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default=str(DEFAULT_STUDENT))
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    student_dir = Path(args.student)
    sim = np.load(student_dir / "sim_state.npz")
    student_q = np.concatenate((sim["arm_q"], sim["hand_q"][:, :3]), axis=1)
    student_phase = sim["phase"].astype(str)
    student_cycle = sim["cycle"].astype(int)
    student_turn = sim["turn_progress_deg"].astype(float)
    student_twist = np.degrees(sim["nut_twist_rad"].astype(float))
    student_grasp = sim["grasp_pos"].astype(float)
    student_time = sim["sim_time_s"].astype(float)

    teacher = np.load(args.teacher, allow_pickle=False)
    teacher_q = teacher["student_osc_proprio"][:, :10].astype(float)
    teacher_phase = teacher["replay_phase"].astype(str)
    teacher_cycle = teacher["cycle"].astype(int)
    teacher_time = teacher["sample_time_s"].astype(float)
    nut_quat = teacher["nut_quat"].astype(float)
    teacher_twist = np.degrees(
        np.unwrap(
            2.0
            * np.arctan2(
                nut_quat[:, 3] * np.sign(nut_quat[:, 0] + 1e-12),
                np.abs(nut_quat[:, 0]) + 1e-12,
            )
        )
    )
    teacher_twist -= teacher_twist[0]

    student_index = {(c, p): (a, b) for c, p, a, b in segments(student_phase, student_cycle)}
    teacher_index = {(c, p): (a, b) for c, p, a, b in segments(teacher_phase, teacher_cycle)}
    shared = sorted(key for key in teacher_index if key in student_index)
    if not shared:
        raise SystemExit("student and teacher share no (cycle, phase) segments")

    names = list(FORGE_POLICY_JOINT_NAMES)
    per_phase: dict[str, dict[str, list]] = {}
    stacked_student, stacked_teacher, labels = [], [], []
    for cycle, phase in shared:
        sa, sb = student_index[(cycle, phase)]
        ta, tb = teacher_index[(cycle, phase)]
        s = resample(student_q[sa:sb], args.samples)
        t = resample(teacher_q[ta:tb], args.samples)
        stacked_student.append(s)
        stacked_teacher.append(t)
        labels.append((cycle, phase, sb - sa, tb - ta))
        entry = per_phase.setdefault(phase, {"student": [], "teacher": [], "cycles": []})
        entry["student"].append(s)
        entry["teacher"].append(t)
        entry["cycles"].append(cycle)

    def stats(s: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        d = s - t
        return {
            "rms": np.sqrt((d**2).mean(axis=0)),
            "max": np.abs(d).max(axis=0),
            "bias": d.mean(axis=0),
        }

    all_student = np.concatenate(stacked_student)
    all_teacher = np.concatenate(stacked_teacher)
    overall = stats(all_student, all_teacher)

    report = {
        "student_rollout": str(student_dir),
        "teacher_episode": str(args.teacher),
        "student_steps": int(len(student_q)),
        "teacher_steps": int(len(teacher_q)),
        "student_duration_s": float(student_time[-1]),
        "teacher_duration_s": float(teacher_time[-1]),
        "aligned_segments": len(shared),
        "alignment": "per (cycle, phase) segment, resampled onto normalized phase time",
        "samples_per_segment": args.samples,
        "joints": names,
        "overall": {
            "rms_deg": np.degrees(overall["rms"]).round(2).tolist(),
            "max_deg": np.degrees(overall["max"]).round(2).tolist(),
            "bias_deg": np.degrees(overall["bias"]).round(2).tolist(),
        },
        "per_phase": {},
        "task": {
            "student_total_nut_rotation_deg": float(abs(student_twist[-1])),
            "teacher_total_nut_rotation_deg": float(abs(teacher_twist[-1])),
            "student_turn_per_cycle_deg": [],
            "teacher_turn_per_cycle_deg": [],
            "student_phase_steps": {},
            "teacher_phase_steps": {},
        },
    }
    for phase in PHASES:
        if phase not in per_phase:
            continue
        s = np.concatenate(per_phase[phase]["student"])
        t = np.concatenate(per_phase[phase]["teacher"])
        st = stats(s, t)
        report["per_phase"][phase] = {
            "cycles": per_phase[phase]["cycles"],
            "rms_deg": np.degrees(st["rms"]).round(2).tolist(),
            "max_deg": np.degrees(st["max"]).round(2).tolist(),
        }
        report["task"]["student_phase_steps"][phase] = [
            int(b - a) for (c, p), (a, b) in student_index.items() if p == phase
        ]
        report["task"]["teacher_phase_steps"][phase] = [
            int(b - a) for (c, p), (a, b) in teacher_index.items() if p == phase
        ]
    for cycle in sorted({c for c, _ in shared}):
        rows = np.where(student_cycle == cycle)[0]
        if len(rows):
            report["task"]["student_turn_per_cycle_deg"].append(
                round(float(student_turn[rows].max()), 1)
            )
        rows = np.where(teacher_cycle == cycle)[0]
        if len(rows):
            span = teacher_twist[rows]
            report["task"]["teacher_turn_per_cycle_deg"].append(
                round(float(abs(span.max() - span.min())), 1)
            )

    destination = Path(args.output_dir or student_dir / "analysis")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "joint_comparison.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        destination / "aligned_joint_traces.npz",
        student=all_student,
        teacher=all_teacher,
        joint_names=np.array(names),
        segment_cycle=np.array([c for c, _p, _s, _t in labels]),
        segment_phase=np.array([p for _c, p, _s, _t in labels]),
        segment_student_steps=np.array([s for _c, _p, s, _t in labels]),
        segment_teacher_steps=np.array([t for _c, _p, _s, t in labels]),
    )

    width = max(len(n) for n in names) + 1
    print(
        f"aligned {len(shared)} (cycle, phase) segments | "
        f"student {len(student_q)} steps / {student_time[-1]:.1f} s, "
        f"teacher {len(teacher_q)} steps / {teacher_time[-1]:.1f} s"
    )
    header = (
        f"{'joint':{width}}"
        + "".join(f"{p.split('_')[0][:10]:>12}" for p in PHASES)
        + f"{'overall':>12}{'max':>10}{'bias':>9}"
    )
    print(header)
    print("-" * len(header))
    for j, name in enumerate(names):
        row = f"{name:{width}}"
        for phase in PHASES:
            value = report["per_phase"].get(phase)
            row += f"{value['rms_deg'][j]:12.2f}" if value else f"{'-':>12}"
        row += (
            f"{np.degrees(overall['rms'][j]):12.2f}"
            f"{np.degrees(overall['max'][j]):10.2f}"
            f"{np.degrees(overall['bias'][j]):9.2f}"
        )
        print(row)
    print("\ndegrees; phase columns and 'overall' are RMS difference, 'max' is the largest")
    print("absolute difference, 'bias' is the mean signed difference (student minus teacher)")
    print(
        f"\nnut rotation: student {report['task']['student_total_nut_rotation_deg']:.1f} deg, "
        f"teacher {report['task']['teacher_total_nut_rotation_deg']:.1f} deg"
    )

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        boundaries = np.arange(1, len(shared)) * args.samples
        cycle_starts = [
            i * args.samples
            for i, (c, p, _s, _t) in enumerate(labels)
            if i == 0 or labels[i - 1][0] != c
        ]
        cycle_ids = [labels[i // args.samples][0] for i in cycle_starts]

        fig, axes = plt.subplots(5, 2, figsize=(15, 15), sharex=True)
        x = np.arange(len(all_student))
        for j, name in enumerate(names):
            ax = axes[j % 5, j // 5]
            ax.plot(x, np.degrees(all_teacher[:, j]), label="teacher (Isaac)", lw=1.3, color="#1f77b4")
            ax.plot(x, np.degrees(all_student[:, j]), label="student (MuJoCo)", lw=1.1, color="#d62728", alpha=0.85)
            for b in boundaries:
                ax.axvline(b, color="0.9", lw=0.4, zorder=0)
            for b in cycle_starts:
                ax.axvline(b, color="0.6", lw=0.8, ls=":", zorder=0)
            ax.set_ylabel(f"{name}\n[deg]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, lw=0.4)
            if j == 0:
                ax.legend(fontsize=8, loc="best")
        for col in range(2):
            axes[-1, col].set_xlabel("aligned (cycle, phase) segments on normalized phase time")
            axes[-1, col].set_xticks(cycle_starts)
            axes[-1, col].set_xticklabels([f"c{c}" for c in cycle_ids], fontsize=7)
        fig.suptitle(
            "Per-joint student (MuJoCo, closed loop) vs teacher (Isaac, recorded), aligned per cycle and phase",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(destination / "joint_comparison.png", dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
        axes[0].plot(teacher_time, teacher_twist, label="teacher", color="#1f77b4")
        axes[0].plot(student_time, student_twist, label="student", color="#d62728")
        axes[0].set_ylabel("cumulative nut rotation [deg]")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(student_time, student_turn, color="#d62728", label="student turn progress")
        axes[1].axhline(55.0, color="0.4", ls="--", lw=0.8, label="release threshold 55 deg")
        axes[1].set_ylabel("per-cycle turn progress [deg]")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        for axis, series, color, label in (
            (axes[2], student_grasp, "#d62728", "student"),
            (axes[2], None, None, None),
        ):
            if series is None:
                continue
            for k, component in enumerate("xyz"):
                axis.plot(student_time, series[:, k], lw=1.0, label=f"{label} grasp {component}")
        axes[2].set_ylabel("grasp frame [m]")
        axes[2].set_xlabel("simulation time [s]")
        axes[2].legend(fontsize=8, ncol=3)
        axes[2].grid(alpha=0.3)
        fig.suptitle("Task progress: nut rotation, per-cycle turn, grasp frame", fontsize=12)
        fig.tight_layout()
        fig.savefig(destination / "task_progress.png", dpi=110)
        plt.close(fig)
        print(f"\nplots: {destination / 'joint_comparison.png'}")
        print(f"       {destination / 'task_progress.png'}")
    print(f"report: {destination / 'joint_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
