from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import DatasetDict, load_dataset, load_from_disk

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from codeguard_ai.artifacts import digest
from codeguard_ai.config import AppConfig

REQUIRED_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {"id", "func", "target"}
OFFICIAL_SIZES = {"train": 21854, "validation": 2732, "test": 2732}


def load_raw_dataset(config: AppConfig) -> DatasetDict:
    snapshot = config.paths.cache / "devign" / digest({
        "name": config.project.dataset_name, "revision": config.project.dataset_revision,
    })[:16]
    if snapshot.exists():
        dataset = load_from_disk(str(snapshot))
    else:
        dataset = load_dataset(
            config.project.dataset_name,
            revision=config.project.dataset_revision,
            cache_dir=str(config.paths.cache / "datasets"),
        )
        validate_dataset(dataset)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=snapshot.parent) as temporary:
            dataset.save_to_disk(temporary)
            Path(temporary).replace(snapshot)
    validate_dataset(dataset)
    sizes = {name: len(dataset[name]) for name in REQUIRED_SPLITS}
    if sizes != OFFICIAL_SIZES:
        raise ValueError(f"Official dataset counts changed: expected {OFFICIAL_SIZES}, got {sizes}")
    return dataset


def validate_dataset(dataset: DatasetDict) -> None:
    missing_splits = set(REQUIRED_SPLITS) - set(dataset)
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")
    for split in REQUIRED_SPLITS:
        missing_columns = REQUIRED_COLUMNS - set(dataset[split].column_names)
        if missing_columns:
            raise ValueError(f"{split} is missing columns: {sorted(missing_columns)}")
        labels = set(dataset[split].unique("target"))
        if not labels or not labels.issubset({0, 1}):
            raise ValueError(f"{split} contains non-binary targets: {labels}")
        if any(not isinstance(code, str) or not code.strip() for code in dataset[split]["func"]):
            raise ValueError(f"{split} has missing/empty code; investigate, do not drop rows")
        ids = dataset[split]["id"]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split} contains duplicate IDs")


def subset_for_quick_test(dataset: DatasetDict, config: AppConfig) -> DatasetDict:
    seed = config.project.seed
    train_count = min(config.quick_test.train_samples, len(dataset["train"]))
    val_count = min(config.quick_test.validation_samples, len(dataset["validation"]))
    return DatasetDict(
        train=_balanced_subset(dataset["train"], train_count, seed),
        validation=_balanced_subset(dataset["validation"], val_count, seed),
        test=dataset["test"],
    )


def _balanced_subset(split: Any, count: int, seed: int) -> Any:
    shuffled = split.shuffle(seed=seed)
    labels = [int(value) for value in shuffled["target"]]
    per_class = count // 2
    selected = [index for index, label in enumerate(labels) if label == 0][:per_class]
    selected += [index for index, label in enumerate(labels) if label == 1][:per_class]
    selected_set = set(selected)
    selected += [
        index for index in range(len(shuffled)) if index not in selected_set
    ][: count - len(selected)]
    return shuffled.select(sorted(selected))


def tokenize_split(
    split: Any,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Any:
    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["func"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = [int(value) for value in batch["target"]]
        return encoded

    return split.map(
        tokenize,
        batched=True,
        remove_columns=split.column_names,
        desc=f"Tokenizing {getattr(split, 'split', 'split')}",
    )


def _code_hashes(values: list[str | None]) -> set[str]:
    return {
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in values
        if value is not None
    }


def audit_dataset(dataset: DatasetDict, output_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "splits": {},
        "exact_cross_split_overlap": {},
    }
    hashes: dict[str, set[str]] = {}
    for split_name in REQUIRED_SPLITS:
        split = dataset[split_name]
        code = split["func"]
        labels = [int(value) for value in split["target"]]
        counts = Counter(labels)
        split_hashes = _code_hashes(code)
        hashes[split_name] = split_hashes
        lengths = [len(value) for value in code if value is not None]
        lines = [value.count("\n") + 1 for value in code if value is not None]
        report["splits"][split_name] = {
            "samples": len(split),
            "class_distribution": {"0": counts[0], "1": counts[1]},
            "missing_code": sum(value is None for value in code),
            "empty_code": sum(not value.strip() for value in code if value is not None),
            "very_short_code": sum(len(value.strip()) < 20 for value in code if value is not None),
            "exact_duplicates": len(code) - len(split_hashes),
            "character_length": _summary(lengths),
            "line_count": _summary(lines),
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        report["exact_cross_split_overlap"][f"{left}_{right}"] = len(
            hashes[left].intersection(hashes[right])
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "min": ordered[0],
        "median": median,
        "mean": sum(ordered) / count,
        "max": ordered[-1],
    }
