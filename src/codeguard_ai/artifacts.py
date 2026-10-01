"""Small explicit persistence helpers. Only load checkpoints created by this project."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from codeguard_ai.config import AppConfig


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, allow_nan=False)
    atomic_write(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    atomic_write(path, lambda temporary: torch.save(state, temporary))


def load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def require_development(config: AppConfig) -> None:
    if (config.paths.results / "test_opened.json").exists():
        raise RuntimeError("Test evaluation has started. This study is frozen; no further tuning.")


def provenance() -> dict:
    packages = ("torch", "transformers", "datasets", "numpy", "scikit-learn", "peft")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "created_at": datetime.now(UTC).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "packages": versions,
        "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
    }
