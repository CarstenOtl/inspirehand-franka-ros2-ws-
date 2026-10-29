"""Checkpoint-compatible two-channel RGB-D vision head.

A port of ForgeUltra's ``two_channel_rgbd_encoder`` (DUSt3R-inspired: separate
RGB and depth ResNet-18 stems, bidirectional RGB<->depth cross-attention, and a
learned control query that pools the fused tokens). Unlike DP3 it sees the
whole 320x180 frame, not a cropped point cloud.

ForgeUltra builds its stems from torchvision's ``resnet18``. The image does not
ship torchvision and inference never needs its ImageNet download, so the trunk
is rebuilt here from plain torch with torchvision's module layout; the strict
state-dict load is what proves the two agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F


TWO_CHANNEL_RGBD_ENCODER_TYPES = ("two_channel_rgbd_encoder", "rgbd_custom_encoder")


class _BasicBlock(nn.Module):
    """torchvision's ResNet ``BasicBlock``, with its parameter names."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            if stride != 1 or inplanes != planes
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _resnet18_stage3(in_channels: int) -> nn.Sequential:
    """``resnet18().children()[:7]``: conv1 .. layer3, 1/16 resolution, 256 ch."""

    def layer(inplanes: int, planes: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            _BasicBlock(inplanes, planes, stride), _BasicBlock(planes, planes)
        )

    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
        layer(64, 64, 1),
        layer(64, 128, 2),
        layer(128, 256, 2),
    )


class _Stage3TokenStem(nn.Module):
    def __init__(
        self, in_channels: int, embedding_dim: int, token_grid: Sequence[int]
    ) -> None:
        super().__init__()
        if len(token_grid) != 2 or min(int(v) for v in token_grid) < 1:
            raise ValueError("token_grid must contain two positive integers")
        self.body = _resnet18_stage3(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(tuple(int(v) for v in token_grid))
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.body(image)).flatten(2).transpose(1, 2)
        return self.projection(features)


class _SelfCrossBlock(nn.Module):
    """Self-attention, cross-attention to the other modality, feed-forward."""

    def __init__(
        self, embedding_dim: int, num_heads: int, feedforward_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_query_norm = nn.LayerNorm(embedding_dim)
        self.cross_memory_norm = nn.LayerNorm(embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(query)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        query = query + attended
        cross_query = self.cross_query_norm(query)
        cross_memory = self.cross_memory_norm(memory)
        attended, _ = self.cross_attention(
            cross_query, cross_memory, cross_memory, need_weights=False
        )
        query = query + attended
        return query + self.mlp(self.mlp_norm(query))


class _BidirectionalFusionLayer(nn.Module):
    """Update RGB from depth and depth from RGB, both from the same old state."""

    def __init__(
        self, embedding_dim: int, num_heads: int, feedforward_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.rgb_from_depth = _SelfCrossBlock(
            embedding_dim, num_heads, feedforward_dim, dropout
        )
        self.depth_from_rgb = _SelfCrossBlock(
            embedding_dim, num_heads, feedforward_dim, dropout
        )

    def forward(self, rgb_tokens, depth_tokens):
        return (
            self.rgb_from_depth(rgb_tokens, depth_tokens),
            self.depth_from_rgb(depth_tokens, rgb_tokens),
        )


class TwoChannelRGBDEncoder(nn.Module):
    """Encode aligned RGB-D into one policy feature vector."""

    def __init__(
        self,
        *,
        camera_matrix: Sequence[float],
        output_dim: int = 64,
        depth_min_m: float = 0.10,
        depth_max_m: float = 2.0,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        token_grid: Sequence[int] = (8, 14),
        dropout: float = 0.0,
        output_activation: str = "elu",
    ) -> None:
        super().__init__()
        camera = torch.as_tensor(camera_matrix, dtype=torch.float32)
        if camera.numel() != 9:
            raise ValueError("camera_matrix must contain nine values")
        camera = camera.reshape(3, 3)
        if camera[0, 0] <= 0 or camera[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not 0.0 < depth_min_m < depth_max_m:
            raise ValueError("depth bounds must satisfy 0 < min < max")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.register_buffer("camera_matrix", camera)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.token_grid = tuple(int(v) for v in token_grid)

        self.rgb_stem = _Stage3TokenStem(3, embedding_dim, self.token_grid)
        # Metric log-depth plus the validity mask.
        self.depth_stem = _Stage3TokenStem(2, embedding_dim, self.token_grid)
        self.spatial_position = nn.Parameter(
            torch.zeros(1, math.prod(self.token_grid), embedding_dim)
        )
        self.modality_embedding = nn.Parameter(torch.zeros(1, 2, 1, embedding_dim))
        self.geometry_embedding = nn.Sequential(
            nn.Linear(4, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.fusion_layers = nn.ModuleList(
            _BidirectionalFusionLayer(embedding_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        )
        self.control_query = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pool_query_norm = nn.LayerNorm(embedding_dim)
        self.pool_memory_norm = nn.LayerNorm(embedding_dim)
        self.pool_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        activation = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "none": nn.Identity,
        }.get(str(output_activation).lower())
        if activation is None:
            raise ValueError(f"unsupported output activation: {output_activation!r}")
        self.output_head = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, output_dim), activation()
        )
        self.register_buffer(
            "rgb_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def _prepare_depth(self, depth, valid_mask):
        depth = depth.float()
        valid = torch.isfinite(depth) & (depth > 0.0)
        if valid_mask is not None:
            valid = valid & valid_mask.bool()
        safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
        clipped = safe_depth.clamp(self.depth_min_m, self.depth_max_m)
        log_min = math.log(self.depth_min_m)
        log_span = math.log(self.depth_max_m) - log_min
        log_depth = (torch.log(clipped) - log_min) / log_span
        log_depth = torch.where(valid, log_depth, torch.zeros_like(log_depth))
        return torch.cat((log_depth, valid.float()), dim=1), clipped, valid

    def _camera_geometry(self, depth, valid):
        """Pool valid depth per token and unproject token centres to camera XYZ."""

        grid_h, grid_w = self.token_grid
        valid_float = valid.float()
        valid_fraction = F.adaptive_avg_pool2d(valid_float, self.token_grid)
        depth_sum = F.adaptive_avg_pool2d(depth * valid_float, self.token_grid)
        pooled_depth = depth_sum / valid_fraction.clamp_min(1.0e-6)
        pooled_depth = torch.where(
            valid_fraction > 0.0, pooled_depth, torch.zeros_like(pooled_depth)
        )
        image_h, image_w = depth.shape[-2:]
        u = (torch.arange(grid_w, device=depth.device, dtype=depth.dtype) + 0.5)
        u = u * (float(image_w) / float(grid_w)) - 0.5
        v = (torch.arange(grid_h, device=depth.device, dtype=depth.dtype) + 0.5)
        v = v * (float(image_h) / float(grid_h)) - 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        z = pooled_depth[:, 0]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        x = (uu.unsqueeze(0) - cx) * z / fx
        y = (vv.unsqueeze(0) - cy) * z / fy
        geometry = torch.stack((x, y, z, valid_fraction[:, 0]), dim=-1)
        return geometry.reshape(depth.shape[0], grid_h * grid_w, 4)

    def forward(self, rgb, depth, valid_mask=None):
        if rgb is None or rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if depth is None or depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [B,1,H,W]")
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and depth must be batch- and pixel-aligned")
        if valid_mask is not None and valid_mask.shape != depth.shape:
            raise ValueError("valid_mask must have the same shape as depth")
        rgb = (rgb.float().clamp(0.0, 1.0) - self.rgb_mean) / self.rgb_std
        depth_input, metric_depth, valid = self._prepare_depth(depth, valid_mask)
        rgb_tokens = (
            self.rgb_stem(rgb) + self.spatial_position + self.modality_embedding[:, 0]
        )
        depth_tokens = (
            self.depth_stem(depth_input)
            + self.spatial_position
            + self.modality_embedding[:, 1]
            + self.geometry_embedding(self._camera_geometry(metric_depth, valid))
        )
        for layer in self.fusion_layers:
            rgb_tokens, depth_tokens = layer(rgb_tokens, depth_tokens)
        memory = self.pool_memory_norm(torch.cat((rgb_tokens, depth_tokens), dim=1))
        query = self.control_query.expand(rgb.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            self.pool_query_norm(query), memory, memory, need_weights=False
        )
        return self.output_head((query + pooled)[:, 0])


def build_two_channel_rgbd_encoder(config: Mapping) -> tuple[TwoChannelRGBDEncoder, int]:
    encoder = config.get("encoder", {})
    if encoder.get("type") not in TWO_CHANNEL_RGBD_ENCODER_TYPES:
        raise ValueError(f"not a two-channel RGB-D encoder: {encoder.get('type')!r}")
    stem = encoder.get("stem", {})
    depth = encoder.get("depth", {})
    fusion = encoder.get("fusion", {})
    output = config.get("output", {})
    output_dim = int(output.get("feature_dim", 64))
    module = TwoChannelRGBDEncoder(
        camera_matrix=config.get("input", {}).get("camera_matrix", ()),
        output_dim=output_dim,
        depth_min_m=float(depth.get("min_depth_m", 0.10)),
        depth_max_m=float(depth.get("max_depth_m", 2.0)),
        embedding_dim=int(fusion.get("embedding_dim", 128)),
        num_heads=int(fusion.get("num_attention_heads", 4)),
        num_layers=int(fusion.get("num_layers", 2)),
        feedforward_dim=int(fusion.get("feedforward_dim", 512)),
        token_grid=tuple(stem.get("token_grid", (8, 14))),
        dropout=float(fusion.get("dropout", 0.0)),
        output_activation=output.get("activation", "elu"),
    )
    module.required_streams = ("head_rgb", "head_depth")
    return module, output_dim


__all__ = [
    "TWO_CHANNEL_RGBD_ENCODER_TYPES",
    "TwoChannelRGBDEncoder",
    "build_two_channel_rgbd_encoder",
]
