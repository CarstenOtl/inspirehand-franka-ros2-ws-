"""The offline DP3 view re-samples a frame the way the deployed encoder does."""

import numpy as np
import pytest

from policy_rollout import DEFAULT_CHECKPOINT


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(), reason="deployed checkpoint not copied in")
def test_a_flat_frame_inside_the_crop_box_comes_back_at_its_depth():
    pytest.importorskip("torch")
    from utils.render_policy_view import load_encoder, sample_dp3_points

    encoder = load_encoder(DEFAULT_CHECKPOINT)
    rgb = np.full((3, 180, 320), 0.5, dtype=np.float32)
    depth = np.full((1, 180, 320), 0.9, dtype=np.float32)
    xyz = sample_dp3_points(encoder, rgb, depth, depth > 0)
    assert 0 < len(xyz) <= 4096
    np.testing.assert_allclose(xyz[:, 2], 0.9, atol=1e-5)
    # Only the crop box survives: x in [-0.39, 0.12], y in [-0.25, 0.20].
    assert xyz[:, 0].min() >= -0.39 - 1e-6 and xyz[:, 0].max() <= 0.12 + 1e-6
    assert xyz[:, 1].min() >= -0.25 - 1e-6 and xyz[:, 1].max() <= 0.20 + 1e-6
