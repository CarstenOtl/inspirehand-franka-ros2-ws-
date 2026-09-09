"""Portable, framework-light recording for policy rollout observations and actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
DATA_FILENAME = "rollout_data.npz"
METADATA_FILENAME = "metadata.json"

REQUIRED_FIELDS = (
    "sample_time_s",
    "joint_position",
    "joint_velocity",
    "proprio",
    "policy_action",
    "filtered_native_action",
    "clipped_elements",
    "trajectory_progress",
    "process_phase",
)

TASK_SIGNAL_DEFAULTS: dict[str, tuple[np.dtype, float | int]] = {
    "pickup_success": (np.dtype(np.int8), -1),
    "threading_entered": (np.dtype(np.int8), -1),
    "completed_cycles": (np.dtype(np.int32), -1),
    "threading_turn_progress_rad": (np.dtype(np.float64), np.nan),
    "terminated": (np.dtype(np.int8), -1),
    "truncated": (np.dtype(np.int8), -1),
    "watchdog_stop": (np.dtype(np.int8), -1),
}

_SAFE_RUN_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_vector(value: Any, width: int, label: str) -> np.ndarray:
    result = _numpy(value).astype(np.float32, copy=False).reshape(-1)
    if result.shape != (width,):
        raise ValueError(f"{label} must contain {width} values, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result.copy()


def _optional_progress(value: float | None) -> float:
    if value is None:
        return float("nan")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("trajectory_progress must be finite and within [0, 1]")
    return result


def _task_signal(name: str, value: Any) -> np.ndarray:
    dtype, default = TASK_SIGNAL_DEFAULTS[name]
    if value is None:
        return np.asarray(default, dtype=dtype)
    if name in {
        "pickup_success",
        "threading_entered",
        "terminated",
        "truncated",
        "watchdog_stop",
    }:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"task signal {name!r} must be boolean or None")
        return np.asarray(int(value), dtype=dtype)
    if name == "completed_cycles":
        result = int(value)
        if result < 0 or result != float(value):
            raise ValueError("completed_cycles must be a non-negative integer")
        return np.asarray(result, dtype=dtype)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"task signal {name!r} must be finite")
    return np.asarray(result, dtype=dtype)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def create_rollout_dir(root: str | Path, name: str = "rollout") -> Path:
    """Atomically reserve a timestamped rollout directory below ``root``."""

    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    safe_name = _SAFE_RUN_NAME.sub("-", name.strip()).strip("-.") or "rollout"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    candidate = base / f"{safe_name}-{stamp}"
    candidate.mkdir(exist_ok=False)
    return candidate


@dataclass(frozen=True)
class RolloutArtifact:
    run_dir: Path
    data_path: Path
    metadata_path: Path
    sample_count: int


@dataclass(frozen=True)
class LoadedRollout:
    run_dir: Path
    data_path: Path
    metadata_path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


class RolloutDataCollector:
    """Collect aligned policy-rate samples and persist one auditable artifact.

    RGB-D storage is opt-in because uncompressed policy images dominate artifact
    size. Images passed by :class:`PolicyRolloutSession` are already transformed
    into the checkpoint's policy input convention.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        record_rgbd: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.metadata = dict(metadata or {})
        self.record_rgbd = bool(record_rgbd)
        self._rows: dict[str, list[np.ndarray]] = {
            name: [] for name in REQUIRED_FIELDS
        }
        self._rows.update({name: [] for name in TASK_SIGNAL_DEFAULTS})
        if self.record_rgbd:
            self._rows.update({"head_rgb": [], "head_depth": [], "depth_valid": []})
        self._clock_origin_s: float | None = None
        self._last_source_time_s: float | None = None
        self._artifact: RolloutArtifact | None = None

    @property
    def sample_count(self) -> int:
        return len(self._rows["sample_time_s"])

    def record(
        self,
        *,
        joint_position: Any,
        joint_velocity: Any,
        proprio: Any,
        policy_action: Any,
        filtered_native_action: Any,
        clipped_elements: int,
        sample_time_s: float | None = None,
        trajectory_progress: float | None = None,
        process_phase: str | None = None,
        rgb: Any | None = None,
        depth: Any | None = None,
        valid_mask: Any | None = None,
        task_signals: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one policy-rate sample after validating every channel."""

        if self._artifact is not None:
            raise RuntimeError("cannot record after the collector has been closed")
        source_time = (
            time.monotonic() if sample_time_s is None else float(sample_time_s)
        )
        if not np.isfinite(source_time):
            raise ValueError("sample_time_s must be finite")
        if (
            self._last_source_time_s is not None
            and source_time <= self._last_source_time_s
        ):
            raise ValueError("sample_time_s must increase strictly")
        if self._clock_origin_s is None:
            self._clock_origin_s = source_time

        clipped = int(clipped_elements)
        if clipped < 0 or clipped > 9:
            raise ValueError("clipped_elements must be within [0, 9]")
        phase = "" if process_phase is None else str(process_phase).strip()
        if "\x00" in phase:
            raise ValueError("process_phase cannot contain a NUL byte")

        values = {
            "sample_time_s": np.asarray(
                source_time - self._clock_origin_s, dtype=np.float64
            ),
            "joint_position": _finite_vector(joint_position, 10, "joint_position"),
            "joint_velocity": _finite_vector(joint_velocity, 10, "joint_velocity"),
            "proprio": _finite_vector(proprio, 29, "proprio"),
            "policy_action": _finite_vector(policy_action, 9, "policy_action"),
            "filtered_native_action": _finite_vector(
                filtered_native_action, 9, "filtered_native_action"
            ),
            "clipped_elements": np.asarray(clipped, dtype=np.int16),
            "trajectory_progress": np.asarray(
                _optional_progress(trajectory_progress), dtype=np.float64
            ),
            "process_phase": np.asarray(phase),
        }
        if not np.allclose(values["proprio"][:10], values["joint_position"]):
            raise ValueError("proprio position prefix does not match joint_position")
        if not np.allclose(values["proprio"][10:20], values["joint_velocity"]):
            raise ValueError("proprio velocity prefix does not match joint_velocity")
        if np.any(np.abs(values["policy_action"]) > 1.0 + 1.0e-6):
            raise ValueError("policy_action must contain the bounded [-1, 1] action")
        signals = dict(task_signals or {})
        unknown = sorted(set(signals) - set(TASK_SIGNAL_DEFAULTS))
        if unknown:
            raise ValueError(f"unsupported task signals: {', '.join(unknown)}")
        values.update(
            {
                name: _task_signal(name, signals.get(name))
                for name in TASK_SIGNAL_DEFAULTS
            }
        )

        if self.record_rgbd:
            if rgb is None or depth is None or valid_mask is None:
                raise ValueError("record_rgbd requires rgb, depth, and valid_mask")
            rgb_array = _numpy(rgb).astype(np.float32, copy=False)
            depth_array = _numpy(depth).astype(np.float32, copy=False)
            valid_array = _numpy(valid_mask).astype(np.bool_, copy=False)
            if rgb_array.ndim != 3 or rgb_array.shape[0] != 3:
                raise ValueError("recorded RGB must have policy CHW shape (3, H, W)")
            expected = (1, rgb_array.shape[1], rgb_array.shape[2])
            if depth_array.shape != expected or valid_array.shape != expected:
                raise ValueError(
                    f"depth and valid_mask must have policy shape {expected}"
                )
            if not np.isfinite(rgb_array).all() or not np.isfinite(depth_array).all():
                raise ValueError("recorded RGB-D contains non-finite values")
            values.update(
                {
                    "head_rgb": rgb_array.copy(),
                    "head_depth": depth_array.copy(),
                    "depth_valid": valid_array.copy(),
                }
            )

        for name, value in values.items():
            self._rows[name].append(value)
        self._last_source_time_s = source_time

    def close(self) -> RolloutArtifact:
        """Persist data and metadata once; repeated calls are idempotent."""

        if self._artifact is not None:
            return self._artifact
        if self.sample_count == 0:
            raise RuntimeError("cannot close an empty rollout recording")
        data_dir = self.run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / DATA_FILENAME
        metadata_path = data_dir / METADATA_FILENAME
        if data_path.exists() or metadata_path.exists():
            raise FileExistsError(f"rollout artifact already exists under {data_dir}")

        arrays = {name: np.stack(rows, axis=0) for name, rows in self._rows.items()}
        fields = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        }
        document = {
            **self.metadata,
            "schema_version": SCHEMA_VERSION,
            "artifact": "fr3_dp3_policy_rollout",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": self.sample_count,
            "record_rgbd": self.record_rgbd,
            "time_axis": "elapsed monotonic seconds from first recorded sample",
            "fields": fields,
        }
        # Validate caller metadata before creating either final artifact.
        json.dumps(document, allow_nan=False)
        _atomic_npz(data_path, arrays)
        try:
            _atomic_json(metadata_path, document)
        except Exception:
            data_path.unlink(missing_ok=True)
            raise
        self._artifact = RolloutArtifact(
            run_dir=self.run_dir,
            data_path=data_path,
            metadata_path=metadata_path,
            sample_count=self.sample_count,
        )
        return self._artifact

    def __enter__(self) -> "RolloutDataCollector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None and self.sample_count:
            self.close()


def resolve_rollout_path(path: str | Path) -> tuple[Path, Path, Path]:
    """Resolve ``(data_path, metadata_path, run_dir)`` from file or directory."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"rollout data must be an NPZ file: {source}")
        data_path = source
    elif source.is_dir():
        candidates = [source / DATA_FILENAME, source / "data" / DATA_FILENAME]
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"could not uniquely resolve {DATA_FILENAME} under {source}"
            )
        data_path = matches[0]
    else:
        raise FileNotFoundError(f"rollout path does not exist: {source}")
    metadata_path = data_path.with_name(METADATA_FILENAME)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"rollout metadata does not exist: {metadata_path}")
    run_dir = (
        data_path.parent.parent
        if data_path.parent.name == "data"
        else data_path.parent
    )
    return data_path, metadata_path, run_dir


def load_rollout(path: str | Path) -> LoadedRollout:
    """Load and strictly validate a rollout without enabling pickle."""

    data_path, metadata_path, run_dir = resolve_rollout_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse rollout metadata: {metadata_path}") from exc
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported rollout schema in {metadata_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(
                f"rollout is missing required fields: {', '.join(missing)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}

    sample_count = arrays["sample_time_s"].shape[0]
    expected_widths = {
        "joint_position": 10,
        "joint_velocity": 10,
        "proprio": 29,
        "policy_action": 9,
        "filtered_native_action": 9,
    }
    if sample_count < 1 or arrays["sample_time_s"].shape != (sample_count,):
        raise ValueError("rollout must contain a non-empty one-dimensional time axis")
    if int(metadata.get("sample_count", -1)) != sample_count:
        raise ValueError("metadata sample_count does not match rollout data")
    for name, width in expected_widths.items():
        if arrays[name].shape != (sample_count, width):
            raise ValueError(f"{name} must have shape ({sample_count}, {width})")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("clipped_elements", "trajectory_progress", "process_phase"):
        if arrays[name].shape != (sample_count,):
            raise ValueError(f"{name} must have shape ({sample_count},)")
    clipped = arrays["clipped_elements"]
    if not np.issubdtype(clipped.dtype, np.integer) or np.any(
        (clipped < 0) | (clipped > 9)
    ):
        raise ValueError("clipped_elements must contain integers within [0, 9]")
    progress = arrays["trajectory_progress"].astype(np.float64, copy=False)
    if np.any(np.isinf(progress)) or np.any(
        np.isfinite(progress) & ((progress < 0.0) | (progress > 1.0))
    ):
        raise ValueError("trajectory_progress must contain [0, 1] values or NaN")
    if arrays["process_phase"].dtype.kind not in {"U", "S"}:
        raise ValueError("process_phase must contain plain strings")
    for name in TASK_SIGNAL_DEFAULTS:
        if name in arrays and arrays[name].shape != (sample_count,):
            raise ValueError(f"{name} must have shape ({sample_count},)")
    camera_fields = {"head_rgb", "head_depth", "depth_valid"}
    present_camera_fields = camera_fields.intersection(arrays)
    if present_camera_fields and present_camera_fields != camera_fields:
        raise ValueError("RGB-D recording must contain all three camera fields")
    if present_camera_fields:
        rgb = arrays["head_rgb"]
        if rgb.ndim != 4 or rgb.shape[:2] != (sample_count, 3):
            raise ValueError("head_rgb must have shape (time, 3, height, width)")
        expected_depth_shape = (sample_count, 1, rgb.shape[2], rgb.shape[3])
        if arrays["head_depth"].shape != expected_depth_shape:
            raise ValueError(f"head_depth must have shape {expected_depth_shape}")
        if arrays["depth_valid"].shape != expected_depth_shape:
            raise ValueError(f"depth_valid must have shape {expected_depth_shape}")
        if not np.isfinite(rgb).all() or not np.isfinite(arrays["head_depth"]).all():
            raise ValueError("recorded RGB-D contains non-finite values")
    times = arrays["sample_time_s"].astype(np.float64, copy=False)
    if not np.isfinite(times).all() or times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_time_s must start at zero and increase strictly")
    return LoadedRollout(
        run_dir=run_dir,
        data_path=data_path,
        metadata_path=metadata_path,
        arrays=arrays,
        metadata=metadata,
    )


__all__ = [
    "DATA_FILENAME",
    "LoadedRollout",
    "RolloutArtifact",
    "RolloutDataCollector",
    "SCHEMA_VERSION",
    "TASK_SIGNAL_DEFAULTS",
    "create_rollout_dir",
    "load_rollout",
    "resolve_rollout_path",
]
