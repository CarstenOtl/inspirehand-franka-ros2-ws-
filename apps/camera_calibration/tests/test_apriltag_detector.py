"""Tests for the native-AprilTag to OpenCV corner-order adapter."""

import importlib
import sys
import types

import numpy as np


class _FakeNativeDetector:
    def __init__(self, family):
        self.family = family

    def detect(self, image):
        assert image.flags.c_contiguous
        assert image.dtype == np.uint8
        return [
            {
                "id": 7,
                "lb-rb-rt-lt": np.asarray(
                    [
                        [10.0, 30.0],  # lower-left
                        [30.0, 30.0],  # lower-right
                        [30.0, 10.0],  # upper-right
                        [10.0, 10.0],  # upper-left
                    ]
                ),
            }
        ]


def _module_with_fake_native(monkeypatch):
    native_module = types.ModuleType("apriltag")
    native_module.apriltag = _FakeNativeDetector
    monkeypatch.setitem(sys.modules, "apriltag", native_module)
    sys.modules.pop("camera_calibration.apriltag_detector", None)
    return importlib.import_module("camera_calibration.apriltag_detector")


def test_native_bottom_left_order_is_converted_to_opencv_ippe_order(monkeypatch):
    module = _module_with_fake_native(monkeypatch)
    detector = module.AprilTagDetector("tag36h11")

    (detection,) = detector.detect(np.zeros((8, 8), dtype=np.uint8))

    assert detection.identifier == 7
    np.testing.assert_array_equal(
        detection.corners,
        [
            [10.0, 10.0],  # upper-left
            [30.0, 10.0],  # upper-right
            [30.0, 30.0],  # lower-right
            [10.0, 30.0],  # lower-left
        ],
    )
