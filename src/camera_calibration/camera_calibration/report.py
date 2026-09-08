"""The machine-readable calibration result, in one place.

Both the passive recorder and the automated runner write the same document, so
whatever reads one can read the other.  Version 2 adds the optional
``time_offset`` and ``program`` sections the automated run fills in; every
version 1 key is still present and still means the same thing.
"""

import numpy as np

from .calibration_math import matrix_to_quaternion_xyzw


SCHEMA_VERSION = 2


def transform_document(transform: np.ndarray) -> dict:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "translation": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
        "rotation_xyzw": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
        "matrix_4x4": transform.tolist(),
    }


def calibration_document(
    parent_frame: str,
    child_frame: str,
    camera_optical_frame: str,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    world_to_child: np.ndarray,
    carrier_frame: str,
    hand_to_tag: np.ndarray,
    tag_id: int,
    tag_size_m: float,
    quality: dict,
    extra: dict = None,
) -> dict:
    document = {
        "schema_version": SCHEMA_VERSION,
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "camera_optical_frame": camera_optical_frame,
        "camera_intrinsics": {
            "matrix_3x3": np.asarray(camera_matrix, dtype=float).reshape(3, 3).tolist(),
            "distortion": np.asarray(distortion, dtype=float).tolist(),
        },
        "transform": transform_document(world_to_child),
        "estimated_carrier_to_tag": {
            "parent_frame": carrier_frame,
            "child_frame": f"apriltag_{int(tag_id)}",
            **transform_document(hand_to_tag),
        },
        "quality": dict(quality),
        "tag": {"id": int(tag_id), "size_m": float(tag_size_m)},
    }
    if extra:
        document.update(extra)
    return document
