"""Stateful policy-rate boundary shared by future simulation and ROS adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .flow_policy import FlowPolicyRunner
from .observation import build_native_osc_proprio, one_hot_process_phase
from utils.camera_calibration import CameraCalibrationProfile, prepare_rgbd
from utils.data_collection import RolloutDataCollector


@dataclass(frozen=True)
class PolicyStep:
    policy_action: torch.Tensor
    filtered_native_action: torch.Tensor
    clipped_elements: int
    proprio: torch.Tensor


class OscActionFilter:
    """Apply ForgeUltra's native-action EMA or unified-action scale."""

    def __init__(self, representation: str, native_scale=()) -> None:
        if representation not in {"native", "unified"}:
            raise ValueError("OSC representation must be native or unified")
        self.representation = representation
        self.native_scale = torch.as_tensor(native_scale, dtype=torch.float32)
        if representation == "unified" and self.native_scale.shape != (9,):
            raise ValueError("unified OSC requires a 9-D native scale")
        self.previous_filtered: torch.Tensor | None = None

    def reset(self, previous_filtered_native_action) -> None:
        value = torch.as_tensor(
            previous_filtered_native_action, dtype=torch.float32
        ).flatten()
        if value.shape != (9,) or not bool(torch.isfinite(value).all().item()):
            raise ValueError("reset action must contain nine finite values")
        self.previous_filtered = value.clone()

    def apply(self, policy_action) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.previous_filtered is None:
            raise RuntimeError(
                "session must be reset with the task's previous-action seed"
            )
        raw = torch.as_tensor(policy_action, dtype=torch.float32).flatten()
        if raw.shape != (9,) or not bool(torch.isfinite(raw).all().item()):
            raise ValueError("policy action must contain nine finite values")
        bounded = raw.clamp(-1.0, 1.0)
        clipped = int((bounded != raw).sum().item())
        if self.representation == "unified":
            filtered = bounded * self.native_scale
        else:
            alpha = torch.full((9,), 0.0625, dtype=torch.float32)
            alpha[-3:] = 0.1875
            filtered = alpha * bounded + (1.0 - alpha) * self.previous_filtered
        self.previous_filtered = filtered.clone()
        return bounded, filtered, clipped


class PolicyRolloutSession:
    """Prepare one observation, run the flow student, and update action history."""

    def __init__(
        self,
        runner: FlowPolicyRunner,
        calibration: CameraCalibrationProfile,
        collector: RolloutDataCollector | None = None,
    ) -> None:
        calibration.assert_checkpoint_compatible(runner.config.to_dict())
        self.runner = runner
        self.calibration = calibration
        self.collector = collector
        self.action_filter = OscActionFilter(
            runner.config.osc_action_representation,
            runner.config.osc_native_action_scale,
        )

    def reset(
        self, *, previous_filtered_native_action, seed: int | None = None
    ) -> None:
        self.runner.reset(seed=seed)
        self.action_filter.reset(previous_filtered_native_action)

    def step(
        self,
        *,
        joint_position,
        joint_velocity,
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_units: str,
        trajectory_progress: float | None = None,
        process_phase: str | None = None,
        sample_time_s: float | None = None,
        task_signals: dict[str, object] | None = None,
    ) -> PolicyStep:
        previous = self.action_filter.previous_filtered
        if previous is None:
            raise RuntimeError("session must be reset before its first policy step")
        proprio = build_native_osc_proprio(joint_position, joint_velocity, previous)
        prepared = prepare_rgbd(rgb, depth, self.calibration, depth_units=depth_units)
        phase = None
        if self.runner.config.cyclic_process_phase_conditioning:
            if process_phase is None:
                raise ValueError("checkpoint requires the coordinator's process phase")
            phase = one_hot_process_phase(
                process_phase, self.runner.config.cyclic_process_phase_features
            )
        action = self.runner.step(
            proprio=proprio,
            head_rgb=prepared.rgb,
            head_depth=prepared.depth,
            valid_mask=prepared.valid_mask,
            trajectory_progress=trajectory_progress,
            cyclic_process_phase=phase,
        )
        bounded, filtered, clipped = self.action_filter.apply(action)
        result = PolicyStep(
            policy_action=bounded,
            filtered_native_action=filtered,
            clipped_elements=clipped,
            proprio=proprio,
        )
        if self.collector is not None:
            self.collector.record(
                sample_time_s=sample_time_s,
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                proprio=proprio,
                policy_action=bounded,
                filtered_native_action=filtered,
                clipped_elements=clipped,
                trajectory_progress=trajectory_progress,
                process_phase=process_phase,
                rgb=prepared.rgb,
                depth=prepared.depth,
                valid_mask=prepared.valid_mask,
                task_signals=task_signals,
            )
        return result


__all__ = ["OscActionFilter", "PolicyRolloutSession", "PolicyStep"]
