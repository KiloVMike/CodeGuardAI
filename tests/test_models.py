import torch

from codeguard_ai.models import NeuralHead


def test_forward_applies_train_fitted_statistics() -> None:
    head = NeuralHead("linear", [512, 128], 0.2).eval()
    head.set_statistics(torch.full((768,), 1.0), torch.full((768,), 2.0))
    features = torch.full((3, 768), 5.0)
    with torch.no_grad():
        observed = head(features)
        expected = head.layers((features - 1.0) / 2.0)
    assert torch.allclose(observed, expected)


def test_statistics_survive_the_checkpoint_round_trip() -> None:
    head = NeuralHead("mlp", [512, 128], 0.2).eval()
    head.set_statistics(torch.randn(768), torch.rand(768) + 0.5)
    state = head.state_dict()
    assert "feature_mean" in state and "feature_std" in state
    restored = NeuralHead("mlp", [512, 128], 0.2).eval()
    restored.load_state_dict(state)
    features = torch.randn(5, 768)
    with torch.no_grad():
        assert torch.allclose(restored(features), head(features))


def test_zero_variance_features_never_divide_by_zero() -> None:
    head = NeuralHead("linear", [512, 128], 0.2).eval()
    head.set_statistics(torch.zeros(768), torch.zeros(768))
    assert torch.all(head.feature_std >= 1e-6)
    with torch.no_grad():
        assert torch.isfinite(head(torch.randn(4, 768))).all()
