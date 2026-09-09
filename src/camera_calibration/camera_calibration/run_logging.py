"""Filesystem artifacts for one camera-calibration run."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .calibration_math import matrix_to_quaternion_xyzw


def create_run_directory(
    output_root: str | Path, timestamp: datetime | None = None
) -> Path:
    root = Path(output_root).expanduser().resolve()
    moment = timestamp or datetime.now(timezone.utc)
    stem = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def transform_record(transform: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    quaternion = matrix_to_quaternion_xyzw(matrix[:3, :3])
    return {
        "translation": {
            "x": float(matrix[0, 3]),
            "y": float(matrix[1, 3]),
            "z": float(matrix[2, 3]),
        },
        "rotation_xyzw": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
        "matrix_4x4": matrix.tolist(),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_sample_snapshot(
    path: str | Path,
    color_image: np.ndarray,
    image_corners: np.ndarray,
    camera_to_tag: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
    sample_index: int,
    reprojection_error_px: float,
) -> None:
    annotated = np.asarray(color_image).copy()
    corners = np.rint(image_corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [corners], True, (0, 255, 255), 3, cv2.LINE_AA)
    corner_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255))
    for corner, color in zip(corners.reshape(-1, 2), corner_colors):
        cv2.circle(annotated, tuple(int(value) for value in corner), 7, color, -1, cv2.LINE_AA)

    transform = np.asarray(camera_to_tag, dtype=float).reshape(4, 4)
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    cv2.drawFrameAxes(
        annotated,
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
        rotation_vector,
        transform[:3, 3],
        float(tag_size_m) * 0.75,
        3,
    )
    cv2.putText(
        annotated,
        f"sample {sample_index:03d}  reprojection {reprojection_error_px:.2f}px",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), annotated):
        raise OSError(f"OpenCV could not write snapshot to {path}")
