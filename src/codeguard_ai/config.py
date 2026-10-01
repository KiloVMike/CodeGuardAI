from __future__ import annotations

import os
import random
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(os.environ.get("CODEGUARD_HOME", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"


@dataclass(frozen=True)
class ProjectConfig:
    study: str
    seed: int
    model_name: str
    model_revision: str
    dataset_name: str
    dataset_revision: str


@dataclass(frozen=True)
class TrainingConfig:
    max_length: int
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    num_workers: int


@dataclass(frozen=True)
class QuickTestConfig:
    train_samples: int
    validation_samples: int
    max_length: int
    epochs: int


@dataclass(frozen=True)
class InferenceConfig:
    threshold: float
    explanation_batch_size: int
    explanation_max_tokens: int
    top_lines: int


@dataclass(frozen=True)
class RuntimeConfig:
    cpu_threads: int
    offline: bool


@dataclass(frozen=True)
class FeatureConfig:
    batch_size: int
    shard_size: int
    pooling: str


@dataclass(frozen=True)
class HeadConfig:
    hidden_sizes: list[int]
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    study: str = "cpu_v3"
    quick: bool = False

    @property
    def cache(self) -> Path:
        return self.root / ".cache"

    @property
    def artifacts(self) -> Path:
        base = self.root / "artifacts" / self.study
        return base / "smoke" if self.quick else base

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def results(self) -> Path:
        return self.artifacts / "results"

    @property
    def plots(self) -> Path:
        return self.artifacts / "plots"

    @property
    def predictions(self) -> Path:
        return self.artifacts / "predictions"

    @property
    def research_evidence(self) -> Path:
        return self.artifacts / "research_evidence"

    @property
    def final_model(self) -> Path:
        return self.artifacts / "final_model"

    @property
    def final_config(self) -> Path:
        return self.artifacts / "final_config.json"

    @property
    def experiment_log(self) -> Path:
        return self.artifacts / "experiment_log.csv"

    def create(self) -> None:
        for path in (
            self.cache,
            self.checkpoints,
            self.results,
            self.plots,
            self.predictions,
            self.research_evidence,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    training: TrainingConfig
    quick_test: QuickTestConfig
    inference: InferenceConfig
    paths: Paths
    runtime: RuntimeConfig
    features: FeatureConfig
    head: HeadConfig
    lora: LoraConfig

    @property
    def max_length(self) -> int:
        return self.quick_test.max_length if self.paths.quick else self.training.max_length


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing [{name}] section in configuration")
    return value


def load_config(path: Path | str | None = None, quick: bool = False) -> AppConfig:
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    project = ProjectConfig(**_section(raw, "project"))
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not project.study or any(c not in allowed for c in project.study):
        raise ValueError("Study name must contain lowercase letters, digits, '_' or '-' only")
    config = AppConfig(
        project=project,
        training=TrainingConfig(**_section(raw, "training")),
        quick_test=QuickTestConfig(**_section(raw, "quick_test")),
        inference=InferenceConfig(**_section(raw, "inference")),
        paths=Paths(study=project.study, quick=quick),
        runtime=RuntimeConfig(**_section(raw, "runtime")),
        features=FeatureConfig(**_section(raw, "features")),
        head=HeadConfig(**_section(raw, "head")),
        lora=LoraConfig(**_section(raw, "lora")),
    )
    for value in (
        config.runtime.cpu_threads, config.features.batch_size, config.features.shard_size,
        config.head.batch_size, config.head.epochs, config.head.patience, config.training.epochs,
        config.training.train_batch_size, config.training.eval_batch_size,
        config.training.gradient_accumulation_steps, config.inference.explanation_batch_size,
        config.inference.explanation_max_tokens, config.lora.rank,
    ):
        if value < 1:
            raise ValueError("Thread, batch, epoch, rank and limit settings must be positive")
    if not 8 <= config.max_length <= 512 or config.features.pooling != "masked_mean":
        raise ValueError("Use max_length in [8,512] and pooling='masked_mean'")
    if len(config.head.hidden_sizes) != 2 or min(config.head.hidden_sizes) < 1:
        raise ValueError("The deep head requires two positive hidden layer sizes")
    if quick:
        config = replace(config, head=replace(config.head, epochs=config.quick_test.epochs))
    return config


def configure_local_cache(config: AppConfig) -> None:
    config.paths.create()
    cache = str(config.paths.cache)
    os.environ.setdefault("HF_HOME", str(config.paths.cache / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(config.paths.cache / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(config.paths.cache / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if config.runtime.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(config.runtime.cpu_threads)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
