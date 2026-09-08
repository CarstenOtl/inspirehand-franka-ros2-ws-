"""Deciding whether a burst of frames was really taken with the arm at rest.

The automated run stops at every pose and takes a short burst of frames instead
of one frame in passing.  That buys two things.  Averaging the corners over the
burst divides the detector's pixel noise by roughly the square root of the
frame count, and - more importantly - the *spread* within the burst is a direct
measurement of whether the arm had actually settled.  A burst taken while the
arm still creeps or vibrates shows it in the corner scatter, and is thrown away
rather than quietly biasing the calibration the way the hand-guided run did.
"""

from dataclasses import dataclass

import numpy as np


class CaptureError(RuntimeError):
    """Raised when a burst cannot be trusted as a stationary observation."""


@dataclass(frozen=True)
class BurstSummary:
    corners: np.ndarray
    corner_std_px: float
    joint_positions: np.ndarray
    joint_spread_rad: float
    reprojection_error_px: float
    frames_used: int
    frames_seen: int
    first_stamp_s: float
    last_stamp_s: float

    def as_dict(self) -> dict:
        return {
            "corners": self.corners.tolist(),
            "corner_std_px": self.corner_std_px,
            "joint_positions": [float(value) for value in self.joint_positions],
            "joint_spread_rad": self.joint_spread_rad,
            "reprojection_error_px": self.reprojection_error_px,
            "frames_used": self.frames_used,
            "frames_seen": self.frames_seen,
            "first_stamp_s": self.first_stamp_s,
            "last_stamp_s": self.last_stamp_s,
        }


def summarize_burst(
    detections,
    joint_positions,
    minimum_frames: int = 6,
    max_corner_std_px: float = 0.35,
    max_joint_spread_rad: float = 2.0e-4,
    max_reprojection_error_px: float = 1.5,
) -> BurstSummary:
    """Average a stationary burst, or raise if it does not look stationary.

    ``detections`` is a sequence of ``(stamp_s, corners, reprojection_error_px)``
    covering the burst, ``joint_positions`` an ``(M, 7)`` array of the joint
    states recorded over the same window.
    """
    usable = [
        entry for entry in detections if entry[2] <= max_reprojection_error_px
    ]
    if len(usable) < minimum_frames:
        raise CaptureError(
            f"only {len(usable)} of {len(detections)} frames carried a clean tag "
            f"detection, fewer than the {minimum_frames} needed"
        )
    joint_positions = np.asarray(joint_positions, dtype=float).reshape(-1, 7)
    if len(joint_positions) < 2:
        raise CaptureError("no joint states were recorded over the burst")

    stack = np.asarray([np.asarray(entry[1], dtype=float).reshape(4, 2) for entry in usable])
    corner_std = float(stack.std(axis=0).max())
    if corner_std > max_corner_std_px:
        raise CaptureError(
            f"the tag's corners moved by {corner_std:.2f} px across the burst, more than "
            f"{max_corner_std_px:.2f} px: the arm had not settled, or the tag is blurred"
        )
    spread = float((joint_positions.max(axis=0) - joint_positions.min(axis=0)).max())
    if spread > max_joint_spread_rad:
        raise CaptureError(
            f"the joints moved by {spread * 1.0e6:.0f} urad across the burst, more than "
            f"{max_joint_spread_rad * 1.0e6:.0f} urad: the arm had not settled"
        )

    return BurstSummary(
        corners=stack.mean(axis=0),
        corner_std_px=corner_std,
        joint_positions=joint_positions.mean(axis=0),
        joint_spread_rad=spread,
        reprojection_error_px=float(np.mean([entry[2] for entry in usable])),
        frames_used=len(usable),
        frames_seen=len(detections),
        first_stamp_s=float(usable[0][0]),
        last_stamp_s=float(usable[-1][0]),
    )
