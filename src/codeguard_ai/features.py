"""Once-only, resumable CodeBERT feature extraction; never fits anything to test labels."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from codeguard_ai.artifacts import atomic_write, digest, file_hash, read_json, write_json
from codeguard_ai.config import AppConfig, select_device
from codeguard_ai.data import load_raw_dataset, subset_for_quick_test
from codeguard_ai.models import load_encoder, load_tokenizer, mean_pool


def feature_contract(config: AppConfig, split, name: str) -> dict:
    return {
        "schema": 2, "split": name, "rows": len(split), "fingerprint": split._fingerprint,
        "ids_labels": digest([list(split["id"]), list(split["target"])]),
        "model": config.project.model_name, "revision": config.project.model_revision,
        "dataset_revision": config.project.dataset_revision, "max_length": config.max_length,
        "pooling": config.features.pooling, "dtype": "float32", "quick": config.paths.quick,
        "shard_size": config.features.shard_size,
    }


def feature_directory(config: AppConfig, contract: dict) -> Path:
    return config.paths.cache / "features" / digest(contract)


def study_dataset(config: AppConfig):
    raw = load_raw_dataset(config)
    return subset_for_quick_test(raw, config) if config.paths.quick else raw


def read_manifest(directory: Path, contract: dict) -> dict:
    path = directory / "manifest.json"
    if not path.exists():
        return {"contract": contract, "chunks": [], "completed_rows": 0, "seconds": 0.0}
    manifest = read_json(path)
    if manifest["contract"] != contract:
        raise ValueError("Feature cache contract mismatch")
    expected_start = 0
    for chunk in manifest["chunks"]:
        valid_hash = file_hash(directory / chunk["file"]) == chunk["sha256"]
        if chunk["start"] != expected_start or not valid_hash:
            raise ValueError("Feature cache is incomplete/corrupt; investigate before reusing it")
        expected_start += chunk["rows"]
    if expected_start != manifest["completed_rows"]:
        raise ValueError("Feature progress does not match committed chunks")
    return manifest


@torch.inference_mode()
def extract_split(config: AppConfig, split, name: str, encoder=None, tokenizer=None) -> Path:
    contract = feature_contract(config, split, name)
    directory = feature_directory(config, contract)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(directory, contract)
    if manifest["completed_rows"] == len(split):
        return directory
    device = select_device()
    encoder = (encoder or load_encoder(config)).to(device).eval()
    tokenizer = tokenizer or load_tokenizer(config)
    start_time = time.perf_counter()
    elapsed_before = manifest["seconds"]
    for start in tqdm(
        range(manifest["completed_rows"], len(split), config.features.shard_size),
        desc=f"Encoding {name}", unit="shard",
    ):
        batch = split[start : start + config.features.shard_size]
        codes = batch["func"]
        # Token counts are exact, including special tokens, before truncation.
        full = tokenizer(codes, truncation=False, padding=False, verbose=False)
        lengths = np.asarray([len(tokens) for tokens in full["input_ids"]], dtype=np.int64)
        order = np.argsort(np.minimum(lengths, config.max_length), kind="stable")
        vectors = np.empty((len(codes), encoder.config.hidden_size), dtype=np.float32)
        for offset in range(0, len(order), config.features.batch_size):
            indices = order[offset : offset + config.features.batch_size]
            encoded = tokenizer(
                [codes[int(index)] for index in indices], truncation=True,
                max_length=config.max_length, padding=True, return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            vectors[indices] = mean_pool(hidden, encoded["attention_mask"]).float().cpu().numpy()
        path = directory / f"{start:08d}.npz"

        def write_chunk(temporary: Path, vectors=vectors, batch=batch, lengths=lengths) -> None:
            with temporary.open("wb") as handle:
                np.savez(handle, features=vectors, ids=np.asarray(batch["id"], dtype=np.int64),
                         labels=np.asarray(batch["target"], dtype=np.int64), lengths=lengths)

        atomic_write(path, write_chunk)
        manifest["chunks"].append({
            "file": path.name, "start": start, "rows": len(codes), "sha256": file_hash(path),
        })
        manifest["completed_rows"] = start + len(codes)
        manifest["seconds"] = elapsed_before + time.perf_counter() - start_time
        write_json(directory / "manifest.json", manifest)
    return directory


def load_features(config: AppConfig, split, name: str) -> tuple[dict, dict]:
    contract = feature_contract(config, split, name)
    directory = feature_directory(config, contract)
    manifest = read_manifest(directory, contract)
    if manifest["completed_rows"] != len(split):
        raise RuntimeError(f"{name} features are not complete. Run 'codeguard extract' first.")
    parts: dict[str, list] = {key: [] for key in ("features", "ids", "labels", "lengths")}
    for chunk in manifest["chunks"]:
        with np.load(directory / chunk["file"], allow_pickle=False) as arrays:
            for key in parts:
                parts[key].append(arrays[key])
    result = {key: np.concatenate(values) for key, values in parts.items()}
    if not np.array_equal(result["ids"], split["id"]) or not np.array_equal(
        result["labels"], np.asarray(split["target"], dtype=np.int64)
    ):
        raise ValueError("Cached embeddings are not aligned with dataset IDs/labels")
    if not np.isfinite(result["features"]).all():
        raise ValueError("Non-finite cached embeddings")
    return result, contract


def extract_development(config: AppConfig) -> dict:
    from codeguard_ai.artifacts import require_development

    require_development(config)
    raw = study_dataset(config)
    encoder = None
    tokenizer = None
    reports = {}
    for name in ("train", "validation"):
        contract = feature_contract(config, raw[name], name)
        directory = feature_directory(config, contract)
        manifest = read_manifest(directory, contract)
        if manifest["completed_rows"] != len(raw[name]):
            if encoder is None:
                encoder, tokenizer = load_encoder(config), load_tokenizer(config)
            extract_split(config, raw[name], name, encoder, tokenizer)
        arrays, _ = load_features(config, raw[name], name)
        reports[name] = {
            "rows": len(raw[name]), "contract": contract,
            "truncated": int((arrays["lengths"] > config.max_length).sum()),
            "truncation_fraction": float((arrays["lengths"] > config.max_length).mean()),
            "cache": str(directory.relative_to(config.paths.root)),
            "seconds": read_json(directory / "manifest.json")["seconds"],
        }
    write_json(config.paths.results / "feature_report.json", reports)
    return reports


@torch.inference_mode()
def benchmark(config: AppConfig, samples: int = 32) -> dict:
    # Timing sample from training only. Does not fit a model or evaluate performance.
    raw = load_raw_dataset(config)["train"].shuffle(seed=config.project.seed)
    codes = raw.select(range(min(samples, len(raw))))["func"]
    tokenizer = load_tokenizer(config)
    device = select_device()
    encoder = load_encoder(config).to(device).eval()
    times = []
    for start in range(0, len(codes), config.features.batch_size):
        encoded = tokenizer(list(codes[start : start + config.features.batch_size]),
                            padding=True, truncation=True, max_length=config.max_length,
                            return_tensors="pt").to(device)
        if start == 0:
            encoder(**encoded)  # untimed warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        encoder(**encoded)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
    rate = len(codes) / sum(times)
    report = {
        "device": str(device), "cpu_threads": torch.get_num_threads(),
        "timing_samples": len(codes), "max_length": config.max_length,
        "batch_size": config.features.batch_size, "samples_per_second": rate,
        "estimated_all_split_encoding_hours": 27318 / rate / 3600,
        "note": "Extrapolation, not a deadline. Excludes downloads, I/O, heads and explanations.",
    }
    write_json(config.paths.results / "benchmark.json", report)
    return report
