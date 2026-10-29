"""The student vision heads this runtime implements, selected by checkpoint."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .dp3 import build_dp3_encoder
from .rgbd_encoder import TWO_CHANNEL_RGBD_ENCODER_TYPES, build_two_channel_rgbd_encoder


DP3_ENCODER_TYPE = "rgb_pointcloud_dp3_encoder"
SUPPORTED_VISION_ENCODER_TYPES = (DP3_ENCODER_TYPE, *TWO_CHANNEL_RGBD_ENCODER_TYPES)


def vision_encoder_type(config: Mapping) -> str | None:
    return (config or {}).get("encoder", {}).get("type")


def build_vision_encoder(config: Mapping) -> tuple[nn.Module, int]:
    """Build the checkpoint's vision head; both take ``(rgb, depth, valid_mask)``."""

    encoder_type = vision_encoder_type(config)
    if encoder_type == DP3_ENCODER_TYPE:
        return build_dp3_encoder(config)
    if encoder_type in TWO_CHANNEL_RGBD_ENCODER_TYPES:
        return build_two_channel_rgbd_encoder(config)
    raise ValueError(
        f"unsupported vision encoder {encoder_type!r}; this runtime implements "
        + ", ".join(SUPPORTED_VISION_ENCODER_TYPES)
    )


__all__ = [
    "DP3_ENCODER_TYPE",
    "SUPPORTED_VISION_ENCODER_TYPES",
    "build_vision_encoder",
    "vision_encoder_type",
]
