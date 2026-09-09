import pytest

torch = pytest.importorskip("torch")

from policy_rollout.dp3 import RGBPointCloudDP3Encoder
from policy_rollout.flow_matching import require_vendored_flow_matching
from policy_rollout.flow_policy import FlowActionChunkEnsembler


def _encoder():
    return RGBPointCloudDP3Encoder(
        camera_matrix=(20.0, 0.0, 2.0, 0.0, 20.0, 1.0, 0.0, 0.0, 1.0),
        image_shape=(2, 4),
        output_dim=5,
        num_points=6,
        point_widths=(8, 12),
        crop_min_m=(-1.0, -1.0, 0.1),
        crop_max_m=(1.0, 1.0, 1.0),
        xyz_center_m=(0.0, 0.0, 0.5),
        xyz_scale_m=(1.0, 1.0, 0.5),
    ).eval()


def test_dp3_is_deterministic_in_eval_and_handles_an_empty_cloud():
    encoder = _encoder()
    rgb = torch.rand(2, 3, 2, 4)
    depth = torch.stack((torch.full((1, 2, 4), 0.5), torch.zeros(1, 2, 4)))
    first = encoder(rgb, depth)
    second = encoder(rgb, depth)
    assert first.shape == (2, 5)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_flow_matching_is_loaded_from_workspace_vendor_tree():
    path = require_vendored_flow_matching()
    assert path.parts[-3:] == ("third_party", "flow_matching", "flow_matching")


def test_action_chunk_ensemble_aligns_overlapping_predictions():
    ensemble = FlowActionChunkEnsembler(3, 1, decay=0.5)
    first = ensemble.add(torch.tensor([[[1.0], [2.0], [3.0]]]))
    second = ensemble.add(torch.tensor([[[10.0], [20.0], [30.0]]]))
    torch.testing.assert_close(first, torch.tensor([[1.0]]))
    # newest t=0 and prior t=1 refer to the same absolute command tick
    torch.testing.assert_close(second, torch.tensor([[(10.0 + 0.5 * 2.0) / 1.5]]))
