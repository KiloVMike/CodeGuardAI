from __future__ import annotations

import numpy as np
import torch
from torch import nn

from codeguard_ai.config import AppConfig

EXPERIMENTS = {
    "linear": ("C0_LINEAR_CE", "linear", "ce"),
    "mlp": ("C1_MLP_CE", "mlp", "ce"),
    "weighted": ("C2_MLP_WEIGHTED", "mlp", "weighted"),
    "focal": ("C3_MLP_FOCAL", "mlp", "focal"),
    "lora": ("L0_LORA_CE", "lora", "ce"),
    "lora-weighted": ("L1_LORA_WEIGHTED", "lora", "weighted"),
}
CORE_EXPERIMENTS = ("linear", "mlp", "weighted")


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class NeuralHead(nn.Module):
    def __init__(self, kind: str, hidden_sizes: list[int], dropout: float, width: int = 768):
        super().__init__()
        # Per-feature train-set standardization. Buffers travel inside every
        # checkpoint, so training, evaluation and inference apply one transform.
        self.register_buffer("feature_mean", torch.zeros(width))
        self.register_buffer("feature_std", torch.ones(width))
        if kind == "linear":
            self.layers = nn.Linear(width, 2)
        elif kind == "mlp":
            first, second = hidden_sizes
            self.layers = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, first), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(first, second), nn.GELU(), nn.Dropout(dropout), nn.Linear(second, 2),
            )
        else:
            raise ValueError(f"Unknown head architecture: {kind}")

    def set_statistics(self, mean, std) -> None:
        with torch.no_grad():
            self.feature_mean.copy_(torch.as_tensor(np.asarray(mean, dtype=np.float32)))
            self.feature_std.copy_(torch.as_tensor(np.maximum(np.asarray(std, dtype=np.float32),
                                                              1e-6)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.layers(normalized)


def load_tokenizer(config: AppConfig, local_only: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.project.model_name, revision=config.project.model_revision,
        cache_dir=str(config.paths.cache / "models"), use_fast=True,
        local_files_only=local_only or config.runtime.offline,
    )


def load_encoder(config: AppConfig, local_only: bool = False):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        config.project.model_name, revision=config.project.model_revision,
        cache_dir=str(config.paths.cache / "models"),
        local_files_only=local_only or config.runtime.offline,
    )
    model.requires_grad_(False)
    return model.eval()


def build_lora(config: AppConfig):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError('Install adapters: python -m pip install -e ".[lora]"') from error
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        config.project.model_name, revision=config.project.model_revision,
        cache_dir=str(config.paths.cache / "models"), num_labels=2,
        local_files_only=config.runtime.offline,
    )
    return get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=config.lora.rank, lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout, target_modules=["query", "value"],
        modules_to_save=["classifier"], bias="none",
    ))
