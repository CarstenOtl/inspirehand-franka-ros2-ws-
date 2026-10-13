import pytest

torch = pytest.importorskip("torch")

from policy_rollout.dp3 import RGBPointCloudDP3Encoder
from policy_rollout.flow_matching import require_vendored_flow_matching, sample_local_ode
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


class _ConstantVelocity(torch.nn.Module):
    """u_t(x) = condition, so the exact flow is x(1) = x(0) + condition."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x_t, time, *, condition):
        self.calls.append((tuple(x_t.shape), tuple(time.shape)))
        return condition.expand_as(x_t)


def test_sample_local_ode_drives_the_vendored_euler_solver_with_the_condition():
    model = _ConstantVelocity()
    x_init = torch.zeros(2, 3)
    condition = torch.tensor([[1.0, -2.0, 0.5], [0.0, 0.25, 4.0]])
    result = sample_local_ode(model, x_init, condition, steps=4)
    torch.testing.assert_close(result, condition)
    assert len(model.calls) == 4
    assert all(call == ((2, 3), (2,)) for call in model.calls)
    with pytest.raises(ValueError, match="positive"):
        sample_local_ode(model, x_init, condition, steps=0)


def test_the_encoder_keeps_the_cloud_it_sampled_for_the_debug_view():
    """The debug view draws these rather than sampling the same frame again.

    It has to be what the policy was actually given: a second pass is both
    duplicated work inside the policy loop and, in principle, a different
    answer.
    """

    encoder = _encoder()
    assert encoder.last_sampled_points is None
    assert encoder.last_sampled_valid is None

    encoder.eval()
    rgb = torch.rand(1, 3, 2, 4)
    depth = torch.full((1, 1, 2, 4), 0.5)
    with torch.inference_mode():
        encoder(rgb, depth)

    points, valid = encoder.last_sampled_points, encoder.last_sampled_valid
    assert points.shape == (1, 6, 6)
    assert valid.shape == (1, 6)
    with torch.inference_mode():
        expected, expected_valid = encoder._sample_points(rgb, depth, None)
    torch.testing.assert_close(points, expected)
    torch.testing.assert_close(valid, expected_valid)


def test_the_sampled_cloud_is_not_part_of_the_state_dict():
    """Otherwise every checkpoint load would fail its strict key check."""

    encoder = _encoder()
    with torch.inference_mode():
        encoder(torch.rand(1, 3, 2, 4), torch.full((1, 1, 2, 4), 0.5))
    assert not [key for key in encoder.state_dict() if "last_sampled" in key]
