"""Checkpoint-compatible DP3 RGB point-cloud vision head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn


class RGBPointCloudDP3Encoder(nn.Module):
    """Back-project aligned RGB-D, encode XYZRGB points, and globally pool."""

    def __init__(
        self,
        *,
        camera_matrix: Sequence[float],
        image_shape: Sequence[int] = (180, 320),
        output_dim: int = 64,
        num_points: int = 4096,
        point_widths: Sequence[int] = (64, 128, 256),
        depth_min_m: float = 0.10,
        depth_max_m: float = 2.0,
        crop_min_m: Sequence[float] | None = None,
        crop_max_m: Sequence[float] | None = None,
        xyz_center_m: Sequence[float] = (0.0, 0.0, 1.0),
        xyz_scale_m: Sequence[float] = (0.5, 0.5, 1.0),
        use_rgb: bool = True,
        output_activation: str = "elu",
    ) -> None:
        super().__init__()
        camera = torch.as_tensor(camera_matrix, dtype=torch.float32)
        if camera.numel() != 9:
            raise ValueError("camera_matrix must contain nine values")
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("image_shape must contain two positive values")
        if num_points < 1 or not point_widths:
            raise ValueError("num_points and point_widths must be positive")
        if not 0.0 < depth_min_m < depth_max_m:
            raise ValueError("depth bounds must satisfy 0 < min < max")

        self.num_points = int(num_points)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.use_rgb = bool(use_rgb)
        self.register_buffer("camera_matrix", camera.reshape(3, 3))
        self.register_buffer(
            "xyz_center_m",
            self._vector3(xyz_center_m, "xyz_center_m").view(1, 1, 3),
        )
        scale = self._vector3(xyz_scale_m, "xyz_scale_m")
        if torch.any(scale <= 0):
            raise ValueError("xyz_scale_m entries must be positive")
        self.register_buffer("xyz_scale_m", scale.view(1, 1, 3))
        crop_min = (
            None if crop_min_m is None else self._vector3(crop_min_m, "crop_min_m")
        )
        crop_max = (
            None if crop_max_m is None else self._vector3(crop_max_m, "crop_max_m")
        )
        if (crop_min is None) != (crop_max is None):
            raise ValueError("crop_min_m and crop_max_m must be provided together")
        if crop_min is not None and torch.any(crop_min >= crop_max):
            raise ValueError("each crop minimum must be below its maximum")
        self.register_buffer(
            "crop_min_m", torch.empty(0) if crop_min is None else crop_min.view(1, 1, 3)
        )
        self.register_buffer(
            "crop_max_m", torch.empty(0) if crop_max is None else crop_max.view(1, 1, 3)
        )

        height, width = self.image_shape
        pixel_v, pixel_u = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("pixel_u", pixel_u.flatten().view(1, -1))
        self.register_buffer("pixel_v", pixel_v.flatten().view(1, -1))
        indices = torch.arange(height * width, dtype=torch.float32)
        self.register_buffer(
            "evaluation_priority",
            torch.frac(indices * ((math.sqrt(5.0) - 1.0) / 2.0)).view(1, -1),
        )

        channels = 6 if self.use_rgb else 3
        layers: list[nn.Module] = []
        for width_out in (int(value) for value in point_widths):
            if width_out < 1:
                raise ValueError("point_widths must contain positive values")
            layers.extend(
                (nn.Linear(channels, width_out), nn.LayerNorm(width_out), nn.ReLU())
            )
            channels = width_out
        self.point_mlp = nn.Sequential(*layers)
        activation = {
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "none": nn.Identity,
        }.get(str(output_activation).lower())
        if activation is None:
            raise ValueError(f"unsupported output activation: {output_activation!r}")
        self.output_head = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, int(output_dim)), activation()
        )

    @staticmethod
    def _vector3(value: Sequence[float], label: str) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=torch.float32)
        if result.numel() != 3:
            raise ValueError(f"{label} must contain three values")
        return result.reshape(3)

    def _sample_points(self, rgb, depth, valid_mask):
        batch, _, height, width = rgb.shape
        if (height, width) != self.image_shape:
            raise ValueError(
                f"RGB-D image shape must be {self.image_shape}, got {(height, width)}"
            )
        z = depth[:, 0].float().reshape(batch, -1)
        valid = torch.isfinite(z) & (z >= self.depth_min_m) & (z <= self.depth_max_m)
        if valid_mask is not None:
            valid = valid & valid_mask[:, 0].bool().reshape(batch, -1)
        safe_z = torch.where(valid, z, torch.zeros_like(z))
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        x = (self.pixel_u - cx) * safe_z / fx
        y = (self.pixel_v - cy) * safe_z / fy
        xyz_metric = torch.stack((x, y, safe_z), dim=-1)
        if self.crop_min_m.numel():
            valid = valid & (xyz_metric >= self.crop_min_m).all(dim=-1)
            valid = valid & (xyz_metric <= self.crop_max_m).all(dim=-1)
        xyz = (xyz_metric - self.xyz_center_m) / self.xyz_scale_m
        if self.use_rgb:
            colour = rgb.float().clamp(0.0, 1.0).flatten(2).transpose(1, 2)
            points = torch.cat((xyz, colour.mul(2.0).sub(1.0)), dim=-1)
        else:
            points = xyz

        count = min(self.num_points, points.shape[1])
        priority = (
            torch.rand(batch, points.shape[1], device=points.device, dtype=points.dtype)
            if self.training
            else self.evaluation_priority.expand(batch, -1)
        )
        selected = torch.topk(
            priority + valid.to(priority.dtype) * 2.0, count, dim=1
        ).indices
        sampled = points.gather(
            1, selected.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        )
        sampled_valid = valid.gather(1, selected)
        if count < self.num_points:
            padding = self.num_points - count
            sampled = torch.cat(
                (sampled, sampled.new_zeros(batch, padding, sampled.shape[-1])), dim=1
            )
            sampled_valid = torch.cat(
                (sampled_valid, sampled_valid.new_zeros(batch, padding)), dim=1
            )
        return sampled, sampled_valid

    def forward(self, rgb, depth, valid_mask=None):
        if rgb is None or rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if depth is None or depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [B,1,H,W]")
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and depth must be batch- and pixel-aligned")
        if valid_mask is not None and valid_mask.shape != depth.shape:
            raise ValueError("valid_mask must have the same shape as depth")
        points, valid = self._sample_points(rgb, depth, valid_mask)
        point_features = self.point_mlp(points)
        floor = torch.finfo(point_features.dtype).min
        pooled = point_features.masked_fill(~valid.unsqueeze(-1), floor).amax(dim=1)
        pooled = torch.where(
            valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled)
        )
        return self.output_head(pooled)


def build_dp3_encoder(config: Mapping) -> tuple[RGBPointCloudDP3Encoder, int]:
    encoder = config.get("encoder", {})
    if encoder.get("type") != "rgb_pointcloud_dp3_encoder":
        raise ValueError("only ForgeUltra's RGB point-cloud DP3 encoder is supported")
    input_config = config.get("input", {})
    point_cloud = encoder.get("point_cloud", {})
    backbone = encoder.get("backbone", {})
    output = config.get("output", {})
    output_dim = int(output.get("feature_dim", 64))
    module = RGBPointCloudDP3Encoder(
        camera_matrix=input_config.get("camera_matrix", ()),
        image_shape=input_config.get("image_shape", (180, 320)),
        output_dim=output_dim,
        num_points=int(point_cloud.get("num_points", 4096)),
        point_widths=backbone.get("point_widths", (64, 128, 256)),
        depth_min_m=float(point_cloud.get("depth_min_m", 0.10)),
        depth_max_m=float(point_cloud.get("depth_max_m", 2.0)),
        crop_min_m=point_cloud.get("crop_min_m"),
        crop_max_m=point_cloud.get("crop_max_m"),
        xyz_center_m=point_cloud.get("xyz_center_m", (0.0, 0.0, 1.0)),
        xyz_scale_m=point_cloud.get("xyz_scale_m", (0.5, 0.5, 1.0)),
        use_rgb=bool(point_cloud.get("use_rgb", True)),
        output_activation=output.get("activation", "elu"),
    )
    module.required_streams = ("head_rgb", "head_depth")
    return module, output_dim


__all__ = ["RGBPointCloudDP3Encoder", "build_dp3_encoder"]
