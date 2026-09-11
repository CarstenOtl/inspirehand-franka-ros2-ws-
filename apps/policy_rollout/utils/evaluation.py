"""Numerical diagnostics and reference comparison for recorded rollouts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .data_collection import LoadedRollout, load_rollout


def _statistics(values: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(flattened)))),
        "mae": float(np.mean(np.abs(flattened))),
        "max_abs": float(np.max(np.abs(flattened))),
    }


def _component_statistics(values: np.ndarray) -> list[dict[str, float]]:
    return [_statistics(values[:, index]) for index in range(values.shape[1])]


def _known_boolean(values: np.ndarray) -> tuple[bool, bool]:
    array = np.asarray(values).reshape(-1)
    known = array >= 0
    return bool(known.any()), bool(np.any(array[known] == 1))


def _task_outcome(
    arrays: dict[str, np.ndarray], required_cycles: int
) -> dict[str, Any]:
    required = ("pickup_success", "threading_entered", "completed_cycles")
    if any(name not in arrays for name in required):
        return {
            "status": "not_evaluable",
            "reason": "recording predates task-signal columns",
            "required_cycles": required_cycles,
        }
    pickup_known, pickup = _known_boolean(arrays["pickup_success"])
    threading_known, threading_entered = _known_boolean(arrays["threading_entered"])
    cycle_values = np.asarray(arrays["completed_cycles"]).reshape(-1)
    cycles_known = cycle_values >= 0
    completed_cycles = (
        int(np.max(cycle_values[cycles_known])) if cycles_known.any() else None
    )
    watchdog_known, watchdog_stop = (
        _known_boolean(arrays["watchdog_stop"])
        if "watchdog_stop" in arrays
        else (False, False)
    )
    missing = []
    if not pickup_known:
        missing.append("pickup_success")
    if not threading_known:
        missing.append("threading_entered")
    if completed_cycles is None:
        missing.append("completed_cycles")
    if missing:
        return {
            "status": "not_evaluable",
            "reason": f"task adapter did not supply: {', '.join(missing)}",
            "required_cycles": required_cycles,
            "watchdog_stop_observed": watchdog_stop if watchdog_known else None,
        }
    passed = bool(
        pickup
        and threading_entered
        and completed_cycles >= required_cycles
        and not watchdog_stop
    )
    turn_progress = np.asarray(
        arrays.get("threading_turn_progress_rad", ()), dtype=np.float64
    ).reshape(-1)
    finite_turn_progress = turn_progress[np.isfinite(turn_progress)]
    return {
        "status": "passed" if passed else "failed",
        "pickup_success": pickup,
        "threading_entered": threading_entered,
        "completed_cycles": completed_cycles,
        "required_cycles": required_cycles,
        "watchdog_stop_observed": watchdog_stop if watchdog_known else None,
        "max_threading_turn_deg": (
            float(np.rad2deg(np.max(np.abs(finite_turn_progress))))
            if len(finite_turn_progress)
            else None
        ),
        "task_success": passed,
        "qualification_note": (
            "This numerical result is diagnostic and does not by itself qualify "
            "the checkpoint or enable physical execution."
        ),
    }


def _self_metrics(rollout: LoadedRollout, required_cycles: int) -> dict[str, Any]:
    arrays = rollout.arrays
    times = arrays["sample_time_s"].astype(np.float64)
    periods = np.diff(times)
    duration = float(times[-1])
    timing: dict[str, Any] = {
        "sample_count": int(len(times)),
        "duration_s": duration,
        "mean_rate_hz": float((len(times) - 1) / duration) if duration > 0.0 else None,
    }
    if len(periods):
        timing.update(
            {
                "period_mean_s": float(np.mean(periods)),
                "period_std_s": float(np.std(periods)),
                "period_max_s": float(np.max(periods)),
            }
        )

    action = arrays["policy_action"].astype(np.float64)
    filtered = arrays["filtered_native_action"].astype(np.float64)
    filtered_delta = np.diff(filtered, axis=0)
    clipped = arrays["clipped_elements"].astype(np.int64)
    phase_values, phase_counts = np.unique(
        arrays["process_phase"].astype(str), return_counts=True
    )
    progress = arrays["trajectory_progress"].astype(np.float64)
    finite_progress = progress[np.isfinite(progress)]
    return {
        "timing": timing,
        "actions": {
            "clipped_element_count": int(np.sum(clipped)),
            "steps_with_clipping": int(np.count_nonzero(clipped)),
            "steps_with_clipping_fraction": float(np.mean(clipped > 0)),
            "policy_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
            "filtered_action": _statistics(filtered),
            "filtered_step_delta": (
                _statistics(filtered_delta)
                if len(filtered_delta)
                else {"rmse": 0.0, "mae": 0.0, "max_abs": 0.0}
            ),
        },
        "state": {
            "joint_velocity": _statistics(arrays["joint_velocity"]),
            "joint_velocity_per_joint": _component_statistics(
                arrays["joint_velocity"]
            ),
        },
        "conditioning": {
            "phase_sample_counts": {
                (name if name else "unspecified"): int(count)
                for name, count in zip(phase_values, phase_counts)
            },
            "progress_samples": int(len(finite_progress)),
            "progress_monotonicity_violations": int(
                np.count_nonzero(np.diff(finite_progress) < -1.0e-6)
            ),
        },
        "task_outcome": _task_outcome(arrays, required_cycles),
    }


def _interpolate_reference(
    reference: LoadedRollout, target_times: np.ndarray, name: str
) -> np.ndarray:
    source_times = reference.arrays["sample_time_s"].astype(np.float64)
    values = reference.arrays[name].astype(np.float64)
    if len(source_times) == 1:
        return np.repeat(values, len(target_times), axis=0)
    return np.column_stack(
        [
            np.interp(target_times, source_times, values[:, component])
            for component in range(values.shape[1])
        ]
    )


def _reference_metrics(
    rollout: LoadedRollout, reference: LoadedRollout
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    times = rollout.arrays["sample_time_s"].astype(np.float64)
    reference_times = reference.arrays["sample_time_s"].astype(np.float64)
    overlap_end = min(float(times[-1]), float(reference_times[-1]))
    selected = times <= overlap_end + 1.0e-12
    overlap_times = times[selected]
    if len(overlap_times) < 1:
        raise ValueError(
            "rollout and reference have no overlapping elapsed-time samples"
        )

    errors: dict[str, np.ndarray] = {"sample_time_s": overlap_times}
    report: dict[str, Any] = {
        "reference_data": str(reference.data_path),
        "alignment": "linear interpolation on overlapping elapsed monotonic time",
        "overlap_duration_s": overlap_end,
        "compared_samples": int(len(overlap_times)),
        "channels": {},
    }
    for name in ("joint_position", "joint_velocity", "filtered_native_action"):
        expected = _interpolate_reference(reference, overlap_times, name)
        error = rollout.arrays[name][selected].astype(np.float64) - expected
        errors[f"{name}_error"] = error
        channel = {
            "overall": _statistics(error),
            "per_component": _component_statistics(error),
        }
        if name in {"joint_position", "joint_velocity"}:
            channel["arm"] = _statistics(error[:, :7])
            channel["hand"] = _statistics(error[:, 7:])
        else:
            channel["translation"] = _statistics(error[:, :3])
            channel["rotation"] = _statistics(error[:, 3:6])
            channel["hand"] = _statistics(error[:, 6:])
        report["channels"][name] = channel
    return report, errors


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def evaluate_rollout(
    recording: str | Path,
    *,
    reference: str | Path | None = None,
    output_dir: str | Path | None = None,
    required_cycles: int = 6,
) -> tuple[Path, dict[str, Any]]:
    """Write a JSON evaluation and optional reference-aligned error trace."""

    if required_cycles < 1:
        raise ValueError("required_cycles must be positive")
    rollout = load_rollout(recording)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else rollout.run_dir / "analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "evaluation": "fr3_dp3_policy_rollout",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recording": str(rollout.data_path),
        "checkpoint": rollout.metadata.get("checkpoint"),
        "checkpoint_sha256": rollout.metadata.get("checkpoint_sha256"),
        "metrics": _self_metrics(rollout, required_cycles),
        "reference_comparison": None,
    }
    if reference is not None:
        reference_rollout = load_rollout(reference)
        comparison, trace = _reference_metrics(rollout, reference_rollout)
        report["reference_comparison"] = comparison
        _atomic_npz(destination / "reference_error_trace.npz", trace)
    report_path = destination / "evaluation.json"
    _atomic_json(report_path, report)
    return report_path, report


__all__ = ["evaluate_rollout"]
