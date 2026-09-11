"""Headless plots for rollout state, actions, conditioning, and task signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_collection import LoadedRollout, load_rollout


JOINT_NAMES = (
    *(f"fr3_joint{index}" for index in range(1, 8)),
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
)
ACTION_NAMES = (
    "tcp_dx",
    "tcp_dy",
    "tcp_dz",
    "tcp_drx",
    "tcp_dry",
    "tcp_drz",
    "thumb_yaw",
    "thumb_pitch",
    "index_pitch",
)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError as exc:
        raise ImportError(
            "plotting rollout data requires matplotlib; install requirements.txt"
        ) from exc
    return pyplot


def _output_dir(rollout: LoadedRollout, output_dir: str | Path | None) -> Path:
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else rollout.run_dir / "analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _reference_on_times(
    reference: LoadedRollout, target_times: np.ndarray, field: str
) -> np.ndarray:
    source_times = reference.arrays["sample_time_s"].astype(np.float64)
    values = reference.arrays[field].astype(np.float64)
    if len(source_times) == 1:
        return np.repeat(values, len(target_times), axis=0)
    return np.column_stack(
        [
            np.interp(target_times, source_times, values[:, component])
            for component in range(values.shape[1])
        ]
    )


def _plot_joint_state(
    rollout: LoadedRollout,
    destination: Path,
    reference: LoadedRollout | None,
) -> Path:
    plt = _pyplot()
    times = rollout.arrays["sample_time_s"]
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for index, name in enumerate(JOINT_NAMES):
        color = f"C{index % 10}"
        axes[0].plot(
            times, rollout.arrays["joint_position"][:, index], label=name, color=color
        )
        axes[1].plot(
            times, rollout.arrays["joint_velocity"][:, index], label=name, color=color
        )
    if reference is not None:
        overlap = times <= reference.arrays["sample_time_s"][-1]
        overlap_times = times[overlap]
        for axis, field in zip(axes, ("joint_position", "joint_velocity")):
            expected = _reference_on_times(reference, overlap_times, field)
            for index in range(expected.shape[1]):
                axis.plot(
                    overlap_times,
                    expected[:, index],
                    color=f"C{index % 10}",
                    alpha=0.35,
                    linestyle="--",
                )
    axes[0].set_ylabel("position (rad)")
    axes[1].set_ylabel("velocity (rad/s)")
    axes[1].set_xlabel("elapsed time (s)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_title("Measured joint state (dashed: reference)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path = destination / "joint_state.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _plot_actions(
    rollout: LoadedRollout,
    destination: Path,
    reference: LoadedRollout | None,
) -> Path:
    plt = _pyplot()
    times = rollout.arrays["sample_time_s"]
    raw = rollout.arrays["policy_action"]
    filtered = rollout.arrays["filtered_native_action"]
    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    groups = ((0, 3, "TCP translation"), (3, 6, "TCP rotation"), (6, 9, "hand"))
    for axis, (start, stop, title) in zip(axes, groups):
        for index in range(start, stop):
            color = f"C{index - start}"
            axis.plot(
                times,
                raw[:, index],
                color=color,
                alpha=0.35,
                label=f"{ACTION_NAMES[index]} policy",
            )
            axis.plot(
                times,
                filtered[:, index],
                color=color,
                label=f"{ACTION_NAMES[index]} filtered",
            )
        if reference is not None:
            overlap = times <= reference.arrays["sample_time_s"][-1]
            overlap_times = times[overlap]
            expected = _reference_on_times(
                reference, overlap_times, "filtered_native_action"
            )
            for index in range(start, stop):
                axis.plot(
                    overlap_times,
                    expected[:, index],
                    color=f"C{index - start}",
                    linestyle="--",
                    alpha=0.65,
                )
        axis.set_title(title)
        axis.set_ylabel("native OSC action")
        axis.legend(ncol=3, fontsize=7)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("elapsed time (s)")
    figure.suptitle("Policy and filtered actions (dashed: reference)")
    figure.tight_layout()
    path = destination / "actions.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _plot_diagnostics(rollout: LoadedRollout, destination: Path) -> Path:
    plt = _pyplot()
    arrays = rollout.arrays
    times = arrays["sample_time_s"]
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].step(times, arrays["clipped_elements"], where="post")
    axes[0].set_ylabel("clipped values")
    progress = arrays["trajectory_progress"].astype(float)
    axes[1].plot(times, progress, color="tab:green")
    axes[1].set_ylabel("progress")
    phases = arrays["process_phase"].astype(str)
    unique_phases = list(dict.fromkeys(phases.tolist()))
    phase_index = np.asarray([unique_phases.index(value) for value in phases])
    axes[2].step(times, phase_index, where="post", color="tab:purple")
    axes[2].set_yticks(
        range(len(unique_phases)),
        [name or "unspecified" for name in unique_phases],
    )
    axes[2].set_ylabel("process phase")
    task_signal_plotted = False
    for name, label in (
        ("pickup_success", "pickup"),
        ("threading_entered", "threading"),
        ("completed_cycles", "cycles"),
        ("watchdog_stop", "watchdog"),
    ):
        if name not in arrays:
            continue
        values = arrays[name].astype(float)
        known = values >= 0.0
        if not np.any(known):
            continue
        axes[3].step(
            times,
            np.where(known, values, np.nan),
            where="post",
            label=label,
        )
        task_signal_plotted = True
    if task_signal_plotted:
        axes[3].legend(ncol=4, fontsize=8)
    else:
        axes[3].text(
            0.5,
            0.5,
            "task signals not supplied",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
    axes[3].set_ylabel("task signals")
    axes[3].set_xlabel("elapsed time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Rollout conditioning and safety diagnostics")
    figure.tight_layout()
    path = destination / "diagnostics.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_rollout(
    recording: str | Path,
    *,
    reference: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Generate the standard plot set and return the created paths."""

    rollout = load_rollout(recording)
    reference_rollout = load_rollout(reference) if reference is not None else None
    destination = _output_dir(rollout, output_dir)
    return (
        _plot_joint_state(rollout, destination, reference_rollout),
        _plot_actions(rollout, destination, reference_rollout),
        _plot_diagnostics(rollout, destination),
    )


__all__ = ["ACTION_NAMES", "JOINT_NAMES", "plot_rollout"]
