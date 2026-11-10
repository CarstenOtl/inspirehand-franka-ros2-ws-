"""exp_20-style rl_games teacher in ros-sim.

The checkpoint is the PPO teacher, not a distilled student: an LSTM actor over
46 state observations with no vision. Two consequences for the rollout:

- it reads the nut pose, so it runs only where the nut is known. That is
  ros-sim (``/thread_state``); there is no nut-pose topic on hardware;
- it was trained on one episode, approach, grip and one turn, and never learned
  a release or a return. A run is one turn and ends on the task's own success.

Sources in ForgeUltra (``forge_ultra/tasks``): ``mdp/policy_observations.py``
``_get_observations``, ``forge_franka_threading_v2/forge_franka_env.py``
``_compute_intermediate_values``, ``_update_task_goal_obs_state``,
``_update_threading_phase``, ``utils/helical_goal.py`` ``build_threading_target``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import forge_osc as fo
from .observation import FR3_ARM_EFFORT_LIMITS_NM, build_native_osc_proprio
from .session import OscActionFilter, PolicyStep

#: Actor observation order of the run's env.yaml (obs_order), then prev_actions.
TEACHER_OBS_ORDER = (
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "fingertip_force",
    "arm_joint_torque",
    "fingertip_pos_rel_held",
    "held_quat",
    "wp_pos_err",
    "wp_quat_err",
    "wp_progress",
)
TEACHER_OBS_DIM = 46
TEACHER_ACTION_DIM = 9

# threading_target_rotation_range_deg is 55..65, sampled per episode; a
# deterministic run takes the middle.
TARGET_TURN_RAD = math.radians(60.0)
SUCCESS_TOLERANCE_RAD = math.radians(2.5)  # threading_success_tolerance_deg
# threading_grasp_hold_steps, and adr_state.yaml's final success_hold steps.
GRASP_HOLD_STEPS = 10
SUCCESS_HOLD_STEPS = 12
# Training's proper_grip also requires opposed faces and a perpendicular pinch.
# ros-sim publishes only the pad forces, so this is an approximation: both pads
# loaded, at training's threading_grip_hold_min_pad_force_n.
MIN_PAD_FORCE_N = 1.0
#: 30 s at 15 Hz, the training episode.
EPISODE_STEPS = 450
#: Isaac's nut world yaw minus the configured spawn yaw: the recorded episodes
#: show 150 deg for a 30 deg config, and exp_20's normalizer agrees (a 75 deg
#: config gives a mean near 168 deg, 195 less the mean turn of about 28).
ISAAC_NUT_YAW_OFFSET_DEG = 120.0


def is_teacher_checkpoint(path) -> bool:
    """rl_games writes ``.pth``; the distilled students are ``checkpoint.pt``."""

    return Path(path).suffix == ".pth"


def read_env_contract(checkpoint) -> dict:
    """The fields of the run's ``params/env.yaml`` the rollout depends on.

    Parsed by line because the file carries ``!!python`` tags a safe YAML
    loader refuses.
    """

    path = Path(checkpoint).resolve().parents[1] / "params" / "env.yaml"
    lines = path.read_text().splitlines()
    start = lines.index("obs_order:") + 1
    order = []
    for line in lines[start:]:
        if not line.startswith("- "):
            break
        order.append(line[2:].strip())
    hw_fit = any(line.strip() == "hand_hw_fit: true" for line in lines)
    return {"obs_order": tuple(order), "hand_hw_fit": hw_fit, "path": str(path)}


class TeacherNetwork(nn.Module):
    """rl_games ``actor_critic`` actor: normalize, LSTM, layer norm, MLP, mean."""

    def __init__(self, obs_dim=TEACHER_OBS_DIM, action_dim=TEACHER_ACTION_DIM) -> None:
        super().__init__()
        self.rnn = nn.LSTM(obs_dim, 1024, 2)
        self.layer_norm = nn.LayerNorm(1024)
        self.actor_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU()
        )
        self.mu = nn.Linear(64, action_dim)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """rl_games ``RunningMeanStd``, eval path, with its 1e-5 epsilon and +-5 clip."""

        return ((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1.0e-5)).clamp(-5.0, 5.0)

    def forward(self, obs, state):
        out, state = self.rnn(self.normalize(obs).reshape(1, 1, -1), state)
        return self.mu(self.actor_mlp(self.layer_norm(out.reshape(1, -1)))).flatten(), state


@dataclass(frozen=True)
class TeacherConfig:
    """The fields the shared rollout loop reads from a runner's config."""

    force_observation: str = "included"
    hand_velocity_observation: str = "omitted"
    trajectory_progress_conditioning: bool = False
    cyclic_process_phase_conditioning: bool = False
    trajectory_progress_duration_s: float = 1.0


class TeacherPolicyRunner:
    """Deterministic actor mean of an rl_games checkpoint, with its LSTM state."""

    is_teacher = True

    def __init__(self, checkpoint, device="cpu") -> None:
        self.path = Path(checkpoint)
        self.device = torch.device(device)
        self.config = TeacherConfig()
        contract = read_env_contract(self.path)
        if contract["obs_order"] != TEACHER_OBS_ORDER:
            raise ValueError(
                f"{contract['path']}: obs_order {contract['obs_order']} is not the "
                "layout this runner builds"
            )
        if not contract["hand_hw_fit"]:
            raise ValueError(
                f"{contract['path']}: not trained on the hardware-fitted hand, "
                "which is the only hand ros-sim simulates"
            )
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        weights = torch.load(self.path, map_location="cpu", weights_only=False)["model"]
        prefix = "a2c_network."
        network = TeacherNetwork()
        state = {
            "rnn." + key[len(prefix) + len("rnn.rnn."):]: value
            for key, value in weights.items()
            if key.startswith(prefix + "rnn.rnn.")
        }
        for name in ("actor_mlp", "layer_norm", "mu"):
            state.update(
                {
                    key[len(prefix):]: value
                    for key, value in weights.items()
                    if key.startswith(f"{prefix}{name}.")
                }
            )
        state["obs_mean"] = weights["running_mean_std.running_mean"].float()
        state["obs_var"] = weights["running_mean_std.running_var"].float()
        network.load_state_dict(state, strict=True)
        self.model = network.to(self.device).eval()
        self._state = None

    def metadata(self) -> dict:
        return {
            "checkpoint": str(self.path),
            "sha256": self.sha256,
            "weight_source": "rl_games a2c_network (actor mean)",
            "integration_steps": 0,
            "temporal_ensemble_decay": 0.0,
            "device": str(self.device),
        }

    def reset(self, *, seed=None) -> None:
        self._state = None

    @torch.no_grad()
    def step(self, observation) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).flatten()
        if obs.shape != (TEACHER_OBS_DIM,) or not bool(torch.isfinite(obs).all().item()):
            raise ValueError(f"teacher observation must be {TEACHER_OBS_DIM} finite values")
        action, self._state = self.model(obs, self._state)
        return action.cpu()


def nut_pose_world(axial_m: float, twist_rad: float, spawn_yaw_deg: float):
    """Isaac's ``held_pos`` / ``held_quat`` from ros-sim's thread joints.

    ``nut_carrier`` sits at the bolt tip and ``nut_axial`` starts at
    ``held_asset_spawn_z_offset``, so the sum is the nut root. ``nut_twist``
    turns about +z, the same axis in the base frame and the training world.

    Isaac composes the nut from the bolt's frame and its twist joint, so its
    world yaw is the configured spawn yaw plus ``ISAAC_NUT_YAW_OFFSET_DEG``.
    MuJoCo's nut sits 120 deg away from that, which a hex nut cannot show, but
    the actor reads this quaternion directly. The yaw is deliberately not
    wrapped: the quaternion's sign follows the unwrapped joint, as in Isaac.
    """

    position = fo.BOLT_TIP_POSITION + np.array([0.0, 0.0, float(axial_m)])
    yaw = math.radians(float(spawn_yaw_deg) + ISAAC_NUT_YAW_OFFSET_DEG) + float(twist_rad)
    return position, fo.quat_from_angle_axis(yaw, np.array([0.0, 0.0, 1.0]))


def threading_target(nut_position, nut_quaternion, turn_rad=TARGET_TURN_RAD):
    """``build_threading_target`` for a clockwise turn about the upright bolt."""

    axis = np.array([0.0, 0.0, 1.0])
    position = np.asarray(nut_position, dtype=float) - axis * fo.M24_THREAD_PITCH * turn_rad / (2.0 * math.pi)
    turn = fo.quat_from_angle_axis(fo.THREADING_DIRECTION_SIGN * turn_rad, axis)
    return position, fo.quat_mul(turn, np.asarray(nut_quaternion, dtype=float))


def _orientation_observation(quaternion) -> np.ndarray:
    """Factory's ``noisy_fingertip_quat``: w and z are zeroed before the policy sees it."""

    value = np.array(quaternion, dtype=float)
    value[[0, 3]] = 0.0
    return value


@dataclass
class TeacherState:
    observation: np.ndarray
    grasped_once: bool
    turn_progress_rad: float
    success: bool


class TeacherEpisode:
    """One training episode's bookkeeping, and the 46-D observation."""

    def __init__(self, rate_hz: float, spawn_yaw_deg: float) -> None:
        self.dt = 1.0 / float(rate_hz)
        self.spawn_yaw_deg = float(spawn_yaw_deg)
        self.target_position = None
        self.target_quaternion = None
        self.previous_position = None
        self.previous_orientation = None
        self.proper_grasp_steps = 0
        self.grasped_once = False
        self.target_hold_steps = 0
        self.grip_loss_ticks = 0
        self.success = False
        self.peak_turn_rad = 0.0

    def observe(
        self,
        *,
        grasp_position_world,
        grasp_quaternion_world,
        axial_m: float,
        twist_rad: float,
        turn_progress_rad: float,
        fingertip_force_n,
        arm_joint_torque_nm,
        previous_filtered_action,
    ) -> TeacherState:
        position = np.asarray(grasp_position_world, dtype=float)
        orientation = _orientation_observation(grasp_quaternion_world)
        forces = np.asarray(fingertip_force_n, dtype=float).flatten()
        nut_position, nut_quaternion = nut_pose_world(axial_m, twist_rad, self.spawn_yaw_deg)
        if self.target_position is None:
            self.target_position, self.target_quaternion = threading_target(nut_position, nut_quaternion)
            self.previous_position = position
            self.previous_orientation = orientation

        # Finite differences, as training's; it divides by its 1/120 s physics
        # step over the last substep, this by the tick, so both are a rate.
        linear_velocity = (position - self.previous_position) / self.dt
        difference = fo.quat_mul(orientation, fo.quat_conjugate(self.previous_orientation))
        difference = difference * (1.0 if difference[0] >= 0.0 else -1.0)
        if np.linalg.norm(difference[1:]) < 1.0e-9:
            angular_velocity = np.zeros(3)
        else:
            angular_velocity = fo.axis_angle_from_quat(difference) / self.dt
        angular_velocity[0:2] = 0.0
        self.previous_position = position
        self.previous_orientation = orientation

        # _update_threading_phase runs before the observation in training's step.
        proper_grip = bool(np.all(forces >= MIN_PAD_FORCE_N))
        self.proper_grasp_steps = self.proper_grasp_steps + 1 if proper_grip else 0
        self.grasped_once |= self.proper_grasp_steps >= GRASP_HOLD_STEPS
        if self.grasped_once and not proper_grip:
            self.grip_loss_ticks += 1
        self.peak_turn_rad = max(self.peak_turn_rad, float(turn_progress_rad))
        reached = (
            self.grasped_once
            and proper_grip
            and abs(float(turn_progress_rad) - TARGET_TURN_RAD) <= SUCCESS_TOLERANCE_RAD
        )
        self.target_hold_steps = self.target_hold_steps + 1 if reached else 0
        self.success |= self.target_hold_steps >= SUCCESS_HOLD_STEPS

        quaternion_error = fo.quat_mul(fo.quat_conjugate(nut_quaternion), self.target_quaternion)
        if quaternion_error[0] < 0.0:
            quaternion_error = -quaternion_error
        previous = np.array(previous_filtered_action, dtype=float).flatten()
        previous[3:5] = 0.0
        observation = np.concatenate(
            (
                position - fo.BOLT_TIP_POSITION,
                orientation,
                linear_velocity,
                angular_velocity,
                forces,
                np.clip(
                    np.asarray(arm_joint_torque_nm, dtype=float).flatten()
                    / np.asarray(FR3_ARM_EFFORT_LIMITS_NM),
                    -1.0,
                    1.0,
                ),
                position - nut_position,
                nut_quaternion,
                self.target_position - nut_position,
                quaternion_error,
                [float(self.grasped_once)],
                previous,
            )
        )
        return TeacherState(observation, self.grasped_once, float(turn_progress_rad), self.success)


class TeacherCycleCoordinator:
    """Stands in for ``GripCycleCoordinator``: one turn, and no scripted release."""

    gate_source = "nut_twist"

    def __init__(self, episode: TeacherEpisode) -> None:
        self.episode = episode
        self.completed_cycles = 0
        self.failed = False
        self.last_proxy_turn_rad = 0.0
        self.proxy_vetoes = 0
        self.last_veto = None
        self.release_turn_rad = TARGET_TURN_RAD

    def process_phase(self) -> str:
        return "policy"

    def return_errors(self, *_args) -> list:
        return [0.0, 0.0, 0.0]

    def update(self, *, measured_turn_progress=None, **_kwargs):
        turn = 0.0 if measured_turn_progress is None else float(measured_turn_progress)
        if self.episode.success and self.completed_cycles == 0:
            self.completed_cycles = 1
            return "cycle_completed", turn
        return None, turn


class TeacherRolloutSession:
    """``PolicyRolloutSession`` for the teacher: state observations, native-action EMA.

    ``ema_factor`` was randomized per episode over 0.025..0.1 in training and the
    actor never observes it; the filter keeps the rollout's 0.0625 (arm) and
    0.1875 (pinch), the middle of that range.
    """

    def __init__(self, runner, calibration, collector=None) -> None:
        self.runner = runner
        self.calibration = calibration
        self.collector = collector
        self.action_filter = OscActionFilter("native")

    def reset(self, *, previous_filtered_native_action, seed=None) -> None:
        self.runner.reset(seed=seed)
        self.action_filter.reset(previous_filtered_native_action)

    def step(
        self,
        *,
        joint_position,
        joint_velocity,
        rgb,
        depth,
        depth_units,
        trajectory_progress=None,
        process_phase=None,
        sample_time_s=None,
        fingertip_force_n=None,
        arm_joint_torque_nm=None,
        task_signals=None,
        extras=None,
        teacher_state=None,
    ) -> PolicyStep:
        from utils.camera_calibration import prepare_rgbd

        previous = self.action_filter.previous_filtered
        if previous is None:
            raise RuntimeError("session must be reset before its first policy step")
        # The preflight only times the network, before the nut state exists.
        observation = (
            np.zeros(TEACHER_OBS_DIM) if teacher_state is None else teacher_state.observation
        )
        action = self.runner.step(observation)
        # The student layout is what the recorder and the analysis tools read.
        proprio = build_native_osc_proprio(
            joint_position,
            joint_velocity,
            previous,
            hand_velocity_observation="omitted",
            force_observation="included",
            fingertip_force_n=fingertip_force_n,
            arm_joint_torque_nm=arm_joint_torque_nm,
        )
        prepared = prepare_rgbd(rgb, depth, self.calibration, depth_units=depth_units)
        bounded, filtered, clipped = self.action_filter.apply(action)
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
                extras={**(extras or {}), "teacher_observation": observation},
            )
        return PolicyStep(
            policy_action=bounded,
            filtered_native_action=filtered,
            clipped_elements=clipped,
            proprio=proprio,
            prepared=prepared,
        )


__all__ = [
    "EPISODE_STEPS",
    "TeacherCycleCoordinator",
    "TeacherEpisode",
    "TeacherPolicyRunner",
    "TeacherRolloutSession",
    "is_teacher_checkpoint",
    "nut_pose_world",
    "threading_target",
]
