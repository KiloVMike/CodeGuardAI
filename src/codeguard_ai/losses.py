from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        probability = torch.exp(-cross_entropy)
        focal = ((1 - probability) ** self.gamma) * cross_entropy
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()
