import numpy as np
import pytest

torch = pytest.importorskip("torch")

from policy_rollout import DEFAULT_CHECKPOINT
from policy_rollout.dp3 import RGBPointCloudDP3Encoder
from policy_rollout.rgbd_encoder import TwoChannelRGBDEncoder
from policy_rollout.vision import build_vision_encoder


CAMERA = (40.0, 0.0, 32.0, 0.0, 40.0, 16.0, 0.0, 0.0, 1.0)


def _rgbd_config(encoder_type="two_channel_rgbd_encoder"):
    return {
        "input": {"camera_matrix": list(CAMERA), "image_shape": [32, 64]},
        "encoder": {
            "type": encoder_type,
            "stem": {"token_grid": [2, 4], "pretrained": "imagenet1k_v1"},
            "depth": {"min_depth_m": 0.1, "max_depth_m": 2.0},
            "fusion": {
                "embedding_dim": 16,
                "num_layers": 1,
                "num_attention_heads": 2,
                "feedforward_dim": 32,
            },
        },
        "output": {"feature_dim": 5, "activation": "elu"},
    }


def _rgbd_encoder():
    encoder, width = build_vision_encoder(_rgbd_config())
    assert width == 5
    return encoder.eval()


def test_the_rgbd_stems_carry_torchvision_s_resnet18_parameter_names():
    """ForgeUltra's checkpoints name the stems after torchvision's modules.

    conv1, bn1, relu, maxpool, layer1..layer3 are ``body.0`` .. ``body.6``;
    the strict load of a real checkpoint depends on every one of these.
    """

    state = _rgbd_encoder().state_dict()
    for stem, channels in (("rgb_stem", 3), ("depth_stem", 2)):
        keys = [key for key in state if key.startswith(stem + ".")]
        assert len(keys) == 92
        assert tuple(state[f"{stem}.body.0.weight"].shape) == (64, channels, 7, 7)
        assert f"{stem}.body.1.running_var" in state
        assert f"{stem}.body.4.1.bn2.num_batches_tracked" in state
        assert f"{stem}.body.4.0.downsample.0.weight" not in state
        assert tuple(state[f"{stem}.body.6.0.downsample.0.weight"].shape) == (256, 128, 1, 1)
        assert tuple(state[f"{stem}.projection.weight"].shape) == (16, 256)


def test_the_rgbd_encoder_reads_only_valid_depth():
    """Missing depth is the mask channel, whatever value the hole holds."""

    encoder = _rgbd_encoder()
    rgb = torch.rand(1, 3, 32, 64)
    depth = 0.3 + torch.rand(1, 1, 32, 64)
    hole = torch.zeros_like(depth, dtype=torch.bool)
    hole[..., 4:12, 10:30] = True
    with torch.inference_mode():
        zero = encoder(rgb, depth.masked_fill(hole, 0.0))
        nan = encoder(rgb, depth.masked_fill(hole, float("nan")))
        masked = encoder(rgb, depth, ~hole)
        full = encoder(rgb, depth)
    assert zero.shape == (1, 5)
    assert torch.isfinite(zero).all()
    torch.testing.assert_close(zero, nan)
    torch.testing.assert_close(zero, masked)
    assert not torch.allclose(zero, full)


def test_vision_heads_are_built_from_the_checkpoint_s_encoder_type():
    assert isinstance(build_vision_encoder(_rgbd_config())[0], TwoChannelRGBDEncoder)
    # ForgeUltra's older name for the same encoder.
    assert isinstance(
        build_vision_encoder(_rgbd_config("rgbd_custom_encoder"))[0], TwoChannelRGBDEncoder
    )
    dp3 = {
        "input": {"camera_matrix": list(CAMERA), "image_shape": [32, 64]},
        "encoder": {"type": "rgb_pointcloud_dp3_encoder", "point_cloud": {"num_points": 8}},
        "output": {"feature_dim": 5},
    }
    assert isinstance(build_vision_encoder(dp3)[0], RGBPointCloudDP3Encoder)
    with pytest.raises(ValueError, match="unsupported vision encoder"):
        build_vision_encoder(_rgbd_config("adapt3r_head_encoder"))


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(), reason="deployed checkpoint not copied in")
def test_the_deployed_checkpoint_loads_strictly_and_samples_an_action():
    from policy_rollout.flow_policy import FlowPolicyRunner
    from utils.camera_calibration import load_camera_calibration

    runner = FlowPolicyRunner(DEFAULT_CHECKPOINT, device="cpu", integration_steps=1)
    load_camera_calibration().assert_checkpoint_compatible(runner.config.to_dict())
    phase = np.zeros(len(runner.config.cyclic_process_phase_features), dtype=np.float32)
    phase[0] = 1.0
    action = runner.step(
        proprio=np.zeros(runner.config.proprio_dim, dtype=np.float32),
        head_rgb=np.full((3, 180, 320), 0.5, dtype=np.float32),
        head_depth=np.full((1, 180, 320), 0.8, dtype=np.float32),
        trajectory_progress=0.0,
        cyclic_process_phase=phase,
    )
    assert action.shape == (9,)
    assert torch.isfinite(action).all()
