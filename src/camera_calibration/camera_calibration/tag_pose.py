"""Tag corner handling: canonical ordering, averaging, and the PnP pose.

Kept in one place because the passive recorder, the automated runner and the
pose program all need the same convention.  Corner order is the subtle part:

* the native AprilTag detector returns its four corners as ``lb-rb-rt-lt``
  (bottom-left, bottom-right, top-right, top-left of the *decoded* tag);
* ``cv2.SOLVEPNP_IPPE_SQUARE`` insists on object points in the order
  ``(-h, +h), (+h, +h), (+h, -h), (-h, -h)``, that is top-left, top-right,
  bottom-right, bottom-left.

The two run in opposite directions, so the detector's array is reversed before
it reaches solvePnP.  Feeding it in unreversed mirrors the tag's y axis, which
is exactly a 180 deg rotation about the tag's x axis: constant across frames,
so it cancels out of ``world -> camera``, but it leaves the reported tag frame
upside down with respect to the physical tag.
"""

import math

import cv2
import numpy as np

from .calibration_math import make_transform


DETECTOR_TO_IPPE = (3, 2, 1, 0)


class TagPoseError(ValueError):
    """Raised when corners cannot produce a usable tag pose."""


def object_corners(tag_size_m: float) -> np.ndarray:
    """The tag's own corners in its own frame, in SOLVEPNP_IPPE_SQUARE order."""
    if tag_size_m <= 0.0:
        raise TagPoseError("tag_size_m must be positive")
    half = float(tag_size_m) * 0.5
    return np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def to_ippe_order(detector_corners: np.ndarray) -> np.ndarray:
    """Reorder the detector's lb-rb-rt-lt corners into lt-rt-rb-lb."""
    corners = np.asarray(detector_corners, dtype=np.float64).reshape(4, 2)
    return corners[list(DETECTOR_TO_IPPE)]


def estimate_tag_pose(
    image_corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> tuple[np.ndarray, float]:
    """Tag pose in the camera's optical frame, plus the RMS reprojection error in pixels.

    ``image_corners`` must already be in SOLVEPNP_IPPE_SQUARE order; run the
    detector's output through :func:`to_ippe_order` first.
    """
    corners = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
    corners_3d = object_corners(tag_size_m)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        corners_3d,
        corners,
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, math.inf
    translation = np.asarray(translation_vector, dtype=float).reshape(3)
    if translation[2] <= 0.0:
        return None, math.inf
    reprojection_error = corner_reprojection_error(
        corners, rotation_vector, translation_vector, camera_matrix, distortion, tag_size_m
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return make_transform(rotation, translation), reprojection_error


def corner_reprojection_error(
    image_corners: np.ndarray,
    rotation_vector: np.ndarray,
    translation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> float:
    projected, _ = cv2.projectPoints(
        object_corners(tag_size_m),
        rotation_vector,
        translation_vector,
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64),
    )
    difference = projected.reshape(4, 2) - np.asarray(image_corners, dtype=float).reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def project_tag_corners(
    camera_to_tag: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    tag_size_m: float,
) -> np.ndarray:
    """Where a tag at ``camera_to_tag`` would appear, in IPPE corner order."""
    transform = np.asarray(camera_to_tag, dtype=float).reshape(4, 4)
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    projected, _ = cv2.projectPoints(
        object_corners(tag_size_m),
        rotation_vector,
        transform[:3, 3],
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64),
    )
    return projected.reshape(4, 2)


def average_corners(corner_sets) -> tuple[np.ndarray, float]:
    """Mean corners over a stationary burst, and the largest per-coordinate std in pixels.

    Averaging the corners rather than the poses keeps the average in the space
    where the measurement noise actually lives; the std is the honest test of
    whether the arm really had stopped.
    """
    stack = np.asarray([np.asarray(entry, dtype=float).reshape(4, 2) for entry in corner_sets])
    if len(stack) == 0:
        raise TagPoseError("no corner observations to average")
    return stack.mean(axis=0), float(stack.std(axis=0).max())


def tag_view_angle_deg(camera_to_tag: np.ndarray) -> float:
    """Angle between the tag's normal and the camera's line of sight to it.

    0 deg is a tag squarely facing the camera; beyond roughly 60 deg the corners
    smear and both the detection and its pose degrade.
    """
    transform = np.asarray(camera_to_tag, dtype=float).reshape(4, 4)
    to_tag = transform[:3, 3]
    distance = float(np.linalg.norm(to_tag))
    if distance < 1.0e-9:
        raise TagPoseError("tag is at the camera's optical centre")
    # The tag's own z axis points out of its face, back towards a viewer.
    cosine = float(np.dot(transform[:3, 2], -to_tag / distance))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
