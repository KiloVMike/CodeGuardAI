import torch

from codeguard_ai.losses import FocalLoss


def test_focal_loss_is_finite_and_backpropagates() -> None:
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_focal_loss_penalizes_bad_predictions_more() -> None:
    targets = torch.tensor([0, 1])
    good = FocalLoss()(torch.tensor([[3.0, -2.0], [-2.0, 3.0]]), targets)
    bad = FocalLoss()(torch.tensor([[-2.0, 3.0], [3.0, -2.0]]), targets)
    assert bad > good


def test_focal_loss_applies_class_alpha_after_modulation() -> None:
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1])
    unweighted = FocalLoss(gamma=2.0)(logits, targets)
    weighted = FocalLoss(alpha=torch.tensor([1.0, 3.0]), gamma=2.0)(logits, targets)
    assert weighted == 2 * unweighted
