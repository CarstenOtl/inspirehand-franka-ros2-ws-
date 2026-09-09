"""Stable adapter around AprilTag's native Python detector."""

from dataclasses import dataclass

import numpy as np

try:
    from apriltag import apriltag as _NativeDetector
except ImportError as error:  # Give a useful error outside the development image.
    raise RuntimeError(
        "The native AprilTag detector is missing; install ros-$ROS_DISTRO-apriltag."
    ) from error


@dataclass(frozen=True)
class AprilTagDetection:
    identifier: int
    corners: np.ndarray


class AprilTagDetector:
    """Detect tags without OpenCV 4.6's crash-prone ArUco implementation."""

    def __init__(self, family: str) -> None:
        self._detector = _NativeDetector(family)

    def detect(self, gray_image: np.ndarray) -> tuple[AprilTagDetection, ...]:
        image = np.ascontiguousarray(gray_image, dtype=np.uint8)
        return tuple(
            AprilTagDetection(
                identifier=int(detection["id"]),
                # Native AprilTag exposes lower-left first (lb, rb, rt, lt),
                # while OpenCV IPPE_SQUARE requires top-left first for our
                # object points (lt, rt, rb, lb).  Reversing the native order
                # also makes the solved tag +Z axis point out of its printed
                # face, matching the MuJoCo tag frame.
                corners=np.asarray(
                    detection["lb-rb-rt-lt"], dtype=np.float64
                ).reshape(4, 2)[[3, 2, 1, 0]],
            )
            for detection in self._detector.detect(image)
        )
