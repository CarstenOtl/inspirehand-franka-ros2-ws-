"""Fail-closed placeholder for the not-yet-validated FR3 hardware boundary."""

from __future__ import annotations

from dataclasses import dataclass

from utils.camera_calibration import CameraCalibrationProfile


@dataclass(frozen=True)
class HardwareReadiness:
    ready: bool
    blockers: tuple[str, ...]


def assess_hardware_readiness(
    calibration: CameraCalibrationProfile,
) -> HardwareReadiness:
    blockers = list(calibration.hardware_blockers())
    blockers.extend(
        (
            "the supplied student checkpoint has not been multi-seed replay-qualified for this workcell",
            "live FR3 FK/Jacobian and 1 kHz OSC command transport are not implemented",
            "the cyclic release/return process-phase coordinator is not implemented",
            "the Forge official-hand to workspace RH56 command mapping is not validated",
            "an operator gate, watchdog, limits, and emergency-stop path are not implemented",
        )
    )
    return HardwareReadiness(ready=not blockers, blockers=tuple(blockers))


def run_hardware_placeholder(calibration: CameraCalibrationProfile) -> None:
    readiness = assess_hardware_readiness(calibration)
    details = "\n".join(f"- {item}" for item in readiness.blockers)
    raise RuntimeError(
        "physical policy execution is intentionally disabled; unresolved blockers:\n"
        + details
    )


__all__ = ["HardwareReadiness", "assess_hardware_readiness", "run_hardware_placeholder"]
