"""Isaac-free inference for ForgeUltra DP3 conditional-flow checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dp3 import build_dp3_encoder
from .flow_matching import sample_local_ode


@dataclass(frozen=True)
class FlowPolicyConfig:
    joint_dim: int
    student_control_domain: str | None = None
    osc_action_representation: str = "native"
    osc_native_action_scale: tuple[float, ...] = ()
    proprio_dim: int = 0
    action_horizon: int = 1
    visual_dim: int = 64
    condition_dim: int = 256
    hidden_dim: int = 512
    time_embedding_dim: int = 64
    proprio_hidden_dims: tuple[int, ...] = (128, 128)
    fusion_activation: str = "elu"
    vision_encoder_config: dict = field(default_factory=dict)
    observation_mode: str = "vision"
    backbone_type: str = "trajectory_transformer"
    transformer_num_layers: int = 4
    transformer_num_heads: int = 8
    transformer_feedforward_dim: int = 2048
    transformer_dropout: float = 0.0
    trajectory_progress_conditioning: bool = False
    trajectory_progress_duration_s: float = 0.0
    trajectory_progress_horizon_steps: int = 0
    cyclic_process_phase_conditioning: bool = False
    cyclic_process_phase_schema: str = ""
    cyclic_process_phase_features: tuple[str, ...] = ()
    vision_nut_position_head: bool = False
    nut_position_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nut_position_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "osc_native_action_scale",
            "proprio_hidden_dims",
            "cyclic_process_phase_features",
            "nut_position_mean",
            "nut_position_std",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.joint_dim < 1 or self.action_horizon < 1 or self.proprio_dim < 0:
            raise ValueError("invalid flow action or proprioception dimensions")
        if self.student_control_domain not in {None, "pd", "osc"}:
            raise ValueError("student_control_domain must be 'pd' or 'osc'")
        if self.osc_action_representation not in {"native", "unified"}:
            raise ValueError("unsupported OSC action representation")
        if self.osc_action_representation == "unified":
            if self.control_domain != "osc":
                raise ValueError("unified OSC actions require OSC control")
            if len(self.osc_native_action_scale) != self.joint_dim or any(
                value <= 0 for value in self.osc_native_action_scale
            ):
                raise ValueError("unified OSC requires one positive scale per action")
        if self.observation_mode not in {"vision", "nut_pose_3d", "nut_pose_6d"}:
            raise ValueError("unsupported student observation mode")
        if self.backbone_type not in {"mlp", "trajectory_transformer"}:
            raise ValueError("unsupported flow backbone")
        if self.time_embedding_dim < 2 or self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even and at least two")
        if self.backbone_type == "trajectory_transformer":
            if self.transformer_num_layers < 1 or self.transformer_num_heads < 1:
                raise ValueError("transformer layers and heads must be positive")
            if self.hidden_dim % self.transformer_num_heads:
                raise ValueError("hidden_dim must be divisible by transformer heads")
        if self.trajectory_progress_conditioning and (
            self.trajectory_progress_duration_s <= 0
            and self.trajectory_progress_horizon_steps < 1
        ):
            raise ValueError(
                "progress-conditioned checkpoint has no duration or horizon"
            )
        if self.cyclic_process_phase_conditioning and (
            not self.cyclic_process_phase_schema
            or len(self.cyclic_process_phase_features) < 2
        ):
            raise ValueError("cyclic process conditioning has no feature schema")

    @property
    def control_domain(self) -> str:
        return self.student_control_domain or "pd"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FlowPolicyConfig":
        values = dict(values)
        values.setdefault("backbone_type", "mlp")
        values.setdefault("osc_action_representation", "native")
        values.setdefault("osc_native_action_scale", ())
        values.setdefault("trajectory_progress_conditioning", False)
        values.setdefault("trajectory_progress_duration_s", 0.0)
        values.setdefault("trajectory_progress_horizon_steps", 0)
        values.setdefault("cyclic_process_phase_conditioning", False)
        values.setdefault("cyclic_process_phase_schema", "")
        values.setdefault("cyclic_process_phase_features", ())
        legacy_backend = values.pop("flow_backend", None)
        if legacy_backend == "custom":
            raise ValueError(
                "checkpoint requires ForgeUltra's removed custom flow backend"
            )
        if legacy_backend not in (None, "standalone", "local"):
            raise ValueError(f"unsupported legacy flow backend: {legacy_backend!r}")
        return cls(**values)


def _activation(name: str) -> nn.Module:
    activation = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }.get(str(name).lower())
    if activation is None:
        raise ValueError(f"unsupported fusion activation: {name!r}")
    return activation()


class DextrahConcatFusion(nn.Module):
    """Match ForgeUltra's image-first late fusion and state-dict hierarchy."""

    def __init__(self, config: FlowPolicyConfig, visual_encoder: nn.Module) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = config.proprio_dim
        for output in config.proprio_hidden_dims:
            layers.extend(
                (nn.Linear(width, output), _activation(config.fusion_activation))
            )
            width = output
        self.proprio_encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.visual_encoder = visual_encoder
        self.proprio_dim = config.proprio_dim
        self.proprio_feature_dim = width
        self.visual_dim = config.visual_dim
        self.output_dim = width + config.visual_dim

    def encode_visual(self, *, head_rgb, head_depth, valid_mask=None):
        if head_rgb is None or head_depth is None:
            raise ValueError("DP3 rollout requires aligned head_rgb and head_depth")
        value = self.visual_encoder(head_rgb, head_depth, valid_mask)
        if value.shape[-1] != self.visual_dim:
            raise ValueError("DP3 encoder returned an unexpected latent width")
        return value

    def from_features(self, proprio, visual_features):
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"expected proprio[...,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        return torch.cat(
            (visual_features, self.proprio_encoder(proprio.float())), dim=-1
        )

    def forward(self, proprio, *, head_rgb, head_depth, valid_mask=None):
        return self.from_features(
            proprio,
            self.encode_visual(
                head_rgb=head_rgb, head_depth=head_depth, valid_mask=valid_mask
            ),
        )


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000) / (half - 1))
        )
        angle = time[:, None] * frequency[None] * 1000
        return torch.cat((angle.sin(), angle.cos()), dim=-1)


class _TrajectoryTransformer(nn.Module):
    def __init__(self, config: FlowPolicyConfig) -> None:
        super().__init__()
        self.joint_dim = config.joint_dim
        self.action_horizon = config.action_horizon
        self.action_input = nn.Linear(config.joint_dim, config.hidden_dim)
        self.condition_input = nn.Linear(config.condition_dim, config.hidden_dim)
        self.time_input = nn.Linear(config.time_embedding_dim, config.hidden_dim)
        self.token_position = nn.Parameter(
            torch.zeros(1, config.action_horizon + 2, config.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=config.transformer_feedforward_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_num_layers,
            norm=nn.LayerNorm(config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.velocity_output = nn.Linear(config.hidden_dim, config.joint_dim)
        nn.init.normal_(self.token_position, std=0.02)

    def forward(self, x_t, condition, time_embedding):
        actions = self.action_input(
            x_t.reshape(x_t.shape[0], self.action_horizon, self.joint_dim)
        )
        tokens = torch.cat(
            (
                self.condition_input(condition).unsqueeze(1),
                self.time_input(time_embedding).unsqueeze(1),
                actions,
            ),
            dim=1,
        )
        return self.velocity_output(
            self.encoder(tokens + self.token_position)[:, 2:]
        ).flatten(1)


class ConditionalJointFlowPolicy(nn.Module):
    def __init__(self, config: FlowPolicyConfig) -> None:
        super().__init__()
        self.cfg = config
        visual_encoder, visual_dim = build_dp3_encoder(config.vision_encoder_config)
        if visual_dim != config.visual_dim:
            raise ValueError("DP3 latent width does not match checkpoint config")
        self.fusion = DextrahConcatFusion(config, visual_encoder)
        self.condition = nn.Sequential(
            nn.Linear(self.fusion.output_dim, config.condition_dim), nn.ELU()
        )
        self.nut_position_head = (
            nn.Sequential(
                nn.Linear(config.visual_dim, 128), nn.ELU(), nn.Linear(128, 3)
            )
            if config.vision_nut_position_head
            else None
        )
        self.register_buffer(
            "nut_position_mean",
            torch.tensor(config.nut_position_mean),
            persistent=False,
        )
        self.register_buffer(
            "nut_position_std", torch.tensor(config.nut_position_std), persistent=False
        )
        self.progress_condition = (
            nn.Sequential(
                nn.Linear(1, config.condition_dim),
                nn.ELU(),
                nn.Linear(config.condition_dim, config.condition_dim),
            )
            if config.trajectory_progress_conditioning
            else None
        )
        self.cyclic_process_phase_condition = (
            nn.Sequential(
                nn.Linear(
                    len(config.cyclic_process_phase_features), config.condition_dim
                ),
                nn.ELU(),
                nn.Linear(config.condition_dim, config.condition_dim),
            )
            if config.cyclic_process_phase_conditioning
            else None
        )
        self.time = _TimeEmbedding(config.time_embedding_dim)
        action_dim = config.joint_dim * config.action_horizon
        self.velocity = (
            _TrajectoryTransformer(config)
            if config.backbone_type == "trajectory_transformer"
            else nn.Sequential(
                nn.Linear(
                    action_dim + config.condition_dim + config.time_embedding_dim,
                    config.hidden_dim,
                ),
                nn.ELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ELU(),
                nn.Linear(config.hidden_dim, action_dim),
            )
        )

    @property
    def action_dim(self) -> int:
        return self.cfg.joint_dim * self.cfg.action_horizon

    def encode_condition(
        self,
        *,
        proprio,
        head_rgb,
        head_depth,
        valid_mask=None,
        trajectory_progress=None,
        cyclic_process_phase=None,
    ):
        fused = self.fusion(
            proprio,
            head_rgb=head_rgb,
            head_depth=head_depth,
            valid_mask=valid_mask,
        )
        condition = self.condition(fused)
        if self.progress_condition is not None:
            if trajectory_progress is None:
                raise ValueError("checkpoint requires trajectory_progress")
            progress = trajectory_progress.reshape(-1, 1).to(fused)
            if progress.shape[0] != fused.shape[0] or not bool(
                torch.isfinite(progress).all().item()
            ):
                raise ValueError("trajectory_progress has an invalid batch or value")
            if bool(((progress < 0) | (progress > 1)).any().item()):
                raise ValueError("trajectory_progress must be in [0,1]")
            condition = condition + self.progress_condition(progress)
        if self.cyclic_process_phase_condition is not None:
            if cyclic_process_phase is None:
                raise ValueError("checkpoint requires cyclic_process_phase")
            phase = cyclic_process_phase.to(fused)
            expected = (fused.shape[0], len(self.cfg.cyclic_process_phase_features))
            if tuple(phase.shape) != expected:
                raise ValueError(f"cyclic_process_phase must have shape {expected}")
            if not bool(torch.isfinite(phase).all().item()) or not bool(
                torch.isclose(
                    phase.sum(-1), torch.ones(fused.shape[0], device=fused.device)
                )
                .all()
                .item()
            ):
                raise ValueError("cyclic_process_phase must be finite and one-hot")
            condition = condition + self.cyclic_process_phase_condition(phase)
        return condition

    def forward(self, x_t, time, *, condition):
        if x_t.shape[-1] != self.action_dim:
            raise ValueError("incorrect flow action dimension")
        embedding = self.time(time.reshape(-1).to(x_t))
        if self.cfg.backbone_type == "trajectory_transformer":
            return self.velocity(x_t, condition, embedding)
        return self.velocity(torch.cat((x_t, condition, embedding), dim=-1))

    @torch.inference_mode()
    def sample(
        self,
        *,
        proprio,
        head_rgb,
        head_depth,
        valid_mask=None,
        trajectory_progress=None,
        cyclic_process_phase=None,
        steps=16,
        generator=None,
        noise=None,
    ):
        condition = self.encode_condition(
            proprio=proprio,
            head_rgb=head_rgb,
            head_depth=head_depth,
            valid_mask=valid_mask,
            trajectory_progress=trajectory_progress,
            cyclic_process_phase=cyclic_process_phase,
        )
        value = noise
        if value is None:
            value = torch.randn(
                (condition.shape[0], self.action_dim),
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        value = sample_local_ode(self, value, condition, steps=steps)
        return value.reshape(
            value.shape[0], self.cfg.action_horizon, self.cfg.joint_dim
        )


class FlowActionChunkEnsembler:
    """Blend overlapping action chunks for the current absolute policy tick."""

    def __init__(self, action_horizon: int, joint_dim: int, decay: float = 0.5):
        if action_horizon < 1 or joint_dim < 1 or not 0 <= decay < 1:
            raise ValueError("invalid temporal ensemble configuration")
        self.action_horizon = int(action_horizon)
        self.joint_dim = int(joint_dim)
        self.decay = float(decay)
        self._history: list[torch.Tensor] = []

    def reset(self) -> None:
        self._history.clear()

    def add(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 3 or tuple(chunk.shape[1:]) != (
            self.action_horizon,
            self.joint_dim,
        ):
            raise ValueError("incorrect action chunk shape")
        self._history.insert(0, chunk.detach())
        del self._history[self.action_horizon :]
        weighted = torch.zeros_like(chunk[:, 0])
        total = 0.0
        for age, proposal in enumerate(self._history):
            weight = self.decay**age
            weighted.add_(proposal[:, age], alpha=weight)
            total += weight
        return weighted / total


class FlowPolicyRunner:
    """Strict checkpoint loader plus seeded receding-horizon DP3 inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        integration_steps: int = 16,
        temporal_ensemble_decay: float = 0.5,
        raw_weights: bool = False,
        seed: int = 0,
    ) -> None:
        self.path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device)
        try:
            payload = torch.load(
                self.path, map_location=self.device, weights_only=False
            )
        except TypeError:  # pragma: no cover - PyTorch before weights_only
            payload = torch.load(self.path, map_location=self.device)
        if not isinstance(payload, dict) or not {"config", "model"}.issubset(payload):
            raise ValueError("not a ForgeUltra offline-flow checkpoint")
        self.payload = payload
        self.config = FlowPolicyConfig.from_dict(payload["config"])
        self._assert_deployment_contract()
        self.model = ConditionalJointFlowPolicy(self.config).to(self.device)
        state_key = (
            "model" if raw_weights or "ema_model" not in payload else "ema_model"
        )
        self.model.load_state_dict(payload[state_key], strict=True)
        self.model.eval()
        self.weight_source = "raw" if state_key == "model" else "ema"
        self.integration_steps = int(integration_steps)
        if self.integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        self.ensembler = FlowActionChunkEnsembler(
            self.config.action_horizon,
            self.config.joint_dim,
            temporal_ensemble_decay,
        )
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.last_chunk: torch.Tensor | None = None

    def _assert_deployment_contract(self) -> None:
        vision_type = self.config.vision_encoder_config.get("encoder", {}).get("type")
        errors = []
        if self.config.control_domain != "osc" or self.config.joint_dim != 9:
            errors.append("student must use the 9-D OSC control domain")
        if self.config.observation_mode != "vision" or self.config.proprio_dim != 29:
            errors.append("student must use vision with 29-D native-OSC proprioception")
        if vision_type != "rgb_pointcloud_dp3_encoder":
            errors.append("student must use the RGB point-cloud DP3 vision head")
        if errors:
            raise ValueError("unsupported deployment checkpoint: " + "; ".join(errors))

    def reset(self, *, seed: int | None = None) -> None:
        self.ensembler.reset()
        self.last_chunk = None
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @staticmethod
    def _batch(value, *, device):
        result = torch.as_tensor(value, dtype=torch.float32, device=device)
        return result.unsqueeze(0) if result.ndim in (1, 3) else result

    @torch.inference_mode()
    def step(
        self,
        *,
        proprio,
        head_rgb,
        head_depth,
        valid_mask=None,
        trajectory_progress=None,
        cyclic_process_phase=None,
    ) -> torch.Tensor:
        proprio_t = self._batch(proprio, device=self.device)
        rgb_t = self._batch(head_rgb, device=self.device)
        depth_t = torch.as_tensor(head_depth, dtype=torch.float32, device=self.device)
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0).unsqueeze(0)
        elif depth_t.ndim == 3:
            depth_t = depth_t.unsqueeze(0)
        mask_t = None
        if valid_mask is not None:
            mask_t = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
            if mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(0)
        progress_t = None
        if trajectory_progress is not None:
            progress_t = torch.as_tensor(
                trajectory_progress, dtype=torch.float32, device=self.device
            ).reshape(-1)
        phase_t = None
        if cyclic_process_phase is not None:
            phase_t = self._batch(cyclic_process_phase, device=self.device)
        chunk = self.model.sample(
            proprio=proprio_t,
            head_rgb=rgb_t,
            head_depth=depth_t,
            valid_mask=mask_t,
            trajectory_progress=progress_t,
            cyclic_process_phase=phase_t,
            steps=self.integration_steps,
            generator=self.generator,
        )
        self.last_chunk = chunk.detach().cpu()[0]
        return self.ensembler.add(chunk)[0].detach().cpu()

    def metadata(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {
            "checkpoint": str(self.path),
            "sha256": digest,
            "weight_source": self.weight_source,
            "integration_steps": self.integration_steps,
            "epoch": self.payload.get("epoch"),
            "config": self.config.to_dict(),
            "dataset_contract": self.payload.get("dataset_contract"),
            "training_environment": self.payload.get("training_environment"),
        }


__all__ = [
    "ConditionalJointFlowPolicy",
    "FlowActionChunkEnsembler",
    "FlowPolicyConfig",
    "FlowPolicyRunner",
]
