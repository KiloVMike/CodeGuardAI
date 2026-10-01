"""Validation-only selection, immutable model bundles, and frozen baseline/final evaluation."""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from codeguard_ai.artifacts import (
    digest,
    file_hash,
    read_json,
    require_development,
    write_json,
)
from codeguard_ai.config import AppConfig, select_device
from codeguard_ai.data import OFFICIAL_SIZES, load_raw_dataset, tokenize_split
from codeguard_ai.features import extract_split, feature_contract, load_features, study_dataset
from codeguard_ai.metrics import classification_metrics, find_best_threshold
from codeguard_ai.models import CORE_EXPERIMENTS, EXPERIMENTS, load_encoder, load_tokenizer
from codeguard_ai.training import load_head, predict


def collect_candidates(config: AppConfig, experiments: list[str]) -> list[dict]:
    if not set(CORE_EXPERIMENTS).issubset(experiments):
        raise ValueError("A study requires linear, mlp and weighted experiments")
    candidates = []
    reference = None
    raw = study_dataset(config)
    for name in experiments:
        path = config.paths.checkpoints / EXPERIMENTS[name][0]
        if not (path / "completed.json").exists():
            raise RuntimeError(f"Incomplete required experiment: {name}")
        spec = read_json(path / "run_config.json")
        for split in ("train", "validation"):
            if spec["data"][split] != feature_contract(config, raw[split], split):
                raise ValueError("Current data/encoder configuration differs "
                                 "from saved experiments")
        if read_json(path / "completed.json")["signature"] != spec["signature"]:
            raise ValueError("Completed run marker does not match configuration")
        comparable = {key: spec[key] for key in ("data", "seed", "max_length", "head", "quick")}
        if reference is not None and comparable != reference:
            raise ValueError("Experiments differ in data/seed/preprocessing/head settings")
        reference = comparable
        if spec["quick"] != config.paths.quick:
            raise ValueError("Smoke and full results cannot be mixed")
        if not spec["quick"] and any(spec["data"][s]["rows"] != OFFICIAL_SIZES[s]
                                     for s in ("train", "validation")):
            raise ValueError("Full experiments must use every official development row")
        frame = pd.read_csv(path / "validation_predictions.csv")
        if len(frame) != spec["data"]["validation"]["rows"] or frame["id"].duplicated().any():
            raise ValueError("Invalid validation predictions")
        if not np.array_equal(frame["id"], raw["validation"]["id"]) or not np.array_equal(
            frame["label"], np.asarray(raw["validation"]["target"], dtype=np.int64)
        ):
            raise ValueError("Validation predictions have wrong IDs/labels/order")
        metrics = classification_metrics(frame["label"], frame["probability"])
        saved = read_json(path / "validation_metrics.json")
        if any(abs(metrics[k] - saved[k]) > 1e-8 for k in ("f1", "recall", "accuracy")):
            raise ValueError("Saved metrics do not match validation predictions")
        tuned_threshold, _ = find_best_threshold(frame["label"], frame["probability"])
        tuned = classification_metrics(frame["label"], frame["probability"], tuned_threshold)
        candidates.append({"name": name, "path": path, "spec": spec, "metrics": metrics,
                           "tuned": tuned, "tuned_threshold": tuned_threshold})
    return candidates


def _selection_key(candidate: dict) -> tuple:
    """Validation ROC-AUC first (threshold-free), then achievable F1, then recall."""
    return (
        candidate["metrics"]["roc_auc"] or 0.0,
        candidate["tuned"]["f1"],
        candidate["tuned"]["recall"],
    )


def compare(config: AppConfig, experiments: list[str]) -> list[dict]:
    rows = []
    for candidate in collect_candidates(config, experiments):
        path = candidate["path"]
        metadata = read_json(path / "provenance.json")
        history = read_json(path / "history.json")
        rows.append({
            "experiment": candidate["name"], **candidate["metrics"],
            "tuned_threshold": candidate["tuned_threshold"],
            "tuned_accuracy": candidate["tuned"]["accuracy"],
            "tuned_precision": candidate["tuned"]["precision"],
            "tuned_recall": candidate["tuned"]["recall"],
            "tuned_f1": candidate["tuned"]["f1"],
            "trainable_parameters": metadata["trainable_parameters"],
            "training_seconds": sum(epoch["seconds"] for epoch in history),
        })
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        frame = pd.DataFrame(history)
        axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
        axes[0].plot(frame["epoch"], frame["loss"], label="Validation")
        axes[0].legend()
        axes[0].set(xlabel="Epoch", ylabel="Objective loss")
        axes[1].plot(frame["epoch"], frame["f1"], label="Vulnerable F1")
        axes[1].set(xlabel="Epoch", ylabel="Validation F1", ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(config.paths.plots / f"{candidate['name']}_training.png", dpi=120)
        plt.close(figure)
    frame = pd.DataFrame(rows)
    frame.to_csv(config.paths.results / "validation_comparison.csv", index=False)
    frame.to_csv(config.paths.experiment_log, index=False)
    return rows


def finalize_model(config: AppConfig, experiments: list[str]) -> dict:
    require_development(config)
    candidates = collect_candidates(config, experiments)
    compare(config, experiments)
    ranked = sorted(candidates, key=_selection_key, reverse=True)
    selected = ranked[0]
    baseline = next(c for c in candidates if c["name"] == "linear")
    frames = {c["name"]: pd.read_csv(c["path"] / "validation_predictions.csv")
              for c in candidates}

    # Validation-guarded ensemble: average the top-two heads, keep it only if the
    # ensemble beats the best single model on validation (otherwise single model).
    ensemble, scored, rows = None, None, None
    if len(ranked) >= 2:
        first, second = ranked[0], ranked[1]
        labels = frames[first["name"]]["label"].to_numpy()
        blended = (frames[first["name"]]["probability"].to_numpy()
                   + frames[second["name"]]["probability"].to_numpy()) / 2.0
        ensemble_threshold, _ = find_best_threshold(labels, blended)
        ensemble_metrics = classification_metrics(labels, blended, ensemble_threshold)
        if ensemble_metrics["f1"] > first["tuned"]["f1"]:
            ensemble = {
                "members": [first["name"], second["name"]],
                "threshold": ensemble_threshold, "metrics": ensemble_metrics,
            }
            scored = blended
            selected = first

    if scored is None:
        frame = frames[selected["name"]]
        threshold, rows = find_best_threshold(frame["label"], frame["probability"])
    else:
        threshold = ensemble["threshold"]
        rows = [classification_metrics(labels, scored, float(value)) for value in
                np.round(np.arange(0.05, 0.95 + 0.005, 0.01), 10)]
    pd.DataFrame(rows).to_csv(config.paths.results / "threshold_analysis.csv", index=False)
    _plot_threshold(rows, config.paths.plots / "threshold_metrics.png")
    selection = {
        "schema": 2, "quick": config.paths.quick, "selected": selected["name"],
        "candidate_signatures": {c["name"]: c["spec"]["signature"] for c in candidates},
        "prediction_hashes": {c["name"]: file_hash(c["path"] / "validation_predictions.csv")
                              for c in candidates},
        "ranking_rule": "validation ROC-AUC, then tuned F1, then recall",
        "threshold_rule": "0.05..0.95 step .01; F1, recall, fewer FN, closest to .50",
        "ensemble": ensemble,
        "tuned_thresholds": {c["name"]: c["tuned_threshold"] for c in candidates},
        "threshold": threshold, "max_length": selected["spec"]["max_length"],
        "checkpoint_hashes": {
            role: {p.name: file_hash(p) for p in candidate["path"].joinpath("best_model").iterdir()
                   if p.is_file()}
            for role, candidate in (("baseline", baseline), ("selected", selected))
        },
    }
    if ensemble:
        for member in ensemble["members"]:
            candidate = next(c for c in candidates if c["name"] == member)
            selection["checkpoint_hashes"][f"member_{member}"] = {
                p.name: file_hash(p)
                for p in candidate["path"].joinpath("best_model").iterdir() if p.is_file()}
    bundle_id = digest(selection)[:20]
    bundle = config.paths.final_model / bundle_id
    if not (bundle / "bundle.json").exists():
        bundle.mkdir(parents=True, exist_ok=True)
        load_encoder(config).save_pretrained(bundle / "encoder", safe_serialization=True)
        load_tokenizer(config).save_pretrained(bundle / "tokenizer")
        roles = [("baseline", baseline), ("selected", selected)]
        if ensemble:
            roles += [(f"member_{member}",
                       next(c for c in candidates if c["name"] == member))
                      for member in ensemble["members"]]
        for role, candidate in roles:
            shutil.copytree(candidate["path"] / "best_model", bundle / role, dirs_exist_ok=True)
        write_json(bundle / "selection.json", selection)
        files = {p.relative_to(bundle).as_posix(): file_hash(p)
                 for p in bundle.rglob("*") if p.is_file() and p.name != "bundle.json"}
        write_json(bundle / "bundle.json", {"files": files})
    final = {**selection, "bundle": bundle.relative_to(config.paths.artifacts).as_posix(),
             "bundle_manifest_sha256": file_hash(bundle / "bundle.json")}
    verify_bundle(config, final)
    write_json(config.paths.final_config, final)
    return final


def verify_bundle(config: AppConfig, final: dict) -> Path:
    bundle = (config.paths.artifacts / final["bundle"]).resolve()
    if not bundle.is_relative_to(config.paths.artifacts.resolve()):
        raise ValueError("Invalid bundle path")
    if file_hash(bundle / "bundle.json") != final["bundle_manifest_sha256"]:
        raise ValueError("Bundle manifest changed after freeze")
    for name, expected in read_json(bundle / "bundle.json")["files"].items():
        if file_hash(bundle / name) != expected:
            raise ValueError(f"Frozen model file changed: {name}")
    frozen = read_json(bundle / "selection.json")
    if any(final.get(key) != value for key, value in frozen.items()):
        raise ValueError("Final selection/threshold differs from the frozen bundle")
    return bundle


def evaluate_final(config: AppConfig) -> dict:
    final = read_json(config.paths.final_config)
    if final["quick"] or config.paths.quick:
        raise RuntimeError("Smoke models cannot open the official test set")
    bundle = verify_bundle(config, final)
    opened_path = config.paths.results / "test_opened.json"
    if opened_path.exists() and read_json(opened_path)["final_signature"] != digest(final):
        raise ValueError("The frozen configuration changed after test evaluation started")
    result_path = config.paths.results / "test_comparison.json"
    if result_path.exists():
        return read_json(result_path)  # re-display the existing evidence; no new inspection/tuning
    spec = read_json(bundle / "selected" / "model_spec.json")
    config = replace(config, training=replace(config.training, max_length=final["max_length"]))
    raw = load_raw_dataset(config)
    if feature_contract(config, raw["train"], "train") != spec["data"]["train"]:
        raise ValueError("Dataset/encoder configuration differs from the frozen study")
    write_json(opened_path, {"final_signature": digest(final)})
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(bundle / "encoder", local_files_only=True).eval()
    extract_split(config, raw["test"], "test", encoder, tokenizer)
    del encoder
    arrays, _ = load_features(config, raw["test"], "test")
    loader = DataLoader(TensorDataset(torch.from_numpy(arrays["features"]),
                                     torch.from_numpy(arrays["labels"])),
                        batch_size=config.head.batch_size)
    device = select_device()
    comparison = {}
    for role in ("baseline", "selected"):
        role_spec = read_json(bundle / role / "model_spec.json")
        threshold = 0.5 if role == "baseline" else final["threshold"]
        if role_spec["architecture"] == "lora":
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

            base = AutoModelForSequenceClassification.from_pretrained(
                bundle / "encoder", num_labels=2,
            )
            model = PeftModel.from_pretrained(base, bundle / role).to(device)
            data = tokenize_split(raw["test"], tokenizer, final["max_length"])
            role_loader = DataLoader(data, batch_size=config.training.eval_batch_size,
                                     collate_fn=DataCollatorWithPadding(tokenizer))
            _, labels, probabilities = predict(model, role_loader, device)
            del model
        elif role == "selected" and final.get("ensemble"):
            labels, blended = None, None
            for member in final["ensemble"]["members"]:
                member_head = load_head(bundle / f"member_{member}", device)
                _, member_labels, member_probabilities = predict(member_head, loader, device)
                labels = member_labels if labels is None else labels
                blended = (np.asarray(member_probabilities) if blended is None
                           else blended + np.asarray(member_probabilities))
                del member_head
            probabilities = (blended / len(final["ensemble"]["members"])).tolist()
        else:
            model = load_head(bundle / role, device)
            _, labels, probabilities = predict(model, loader, device)
            del model
        metrics = classification_metrics(labels, probabilities, threshold)
        comparison[role] = metrics
        frame = pd.DataFrame({
            "id": arrays["ids"], "label": labels, "probability": probabilities,
            "prediction": (np.asarray(probabilities) >= threshold).astype(int),
            "token_length": arrays["lengths"], "truncated": arrays["lengths"] > final["max_length"],
        })
        frame.to_csv(config.paths.predictions / f"{role}_test.csv", index=False)
        errors = frame[frame["label"] != frame["prediction"]].copy()
        source_by_id = dict(zip(raw["test"]["id"], raw["test"]["func"], strict=True))
        errors["source_code"] = errors["id"].map(source_by_id)
        errors.to_csv(config.paths.results / f"{role}_errors.csv", index=False)
        _plot_evaluation(labels, probabilities, threshold, config.paths.plots, role)
    comparison["delta_selected_minus_baseline"] = {
        key: comparison["selected"][key] - comparison["baseline"][key]
        for key in ("accuracy", "precision", "recall", "f1", "false_negatives")
    }
    write_json(result_path, comparison)
    return comparison


def _plot_threshold(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(9, 5))
    for column in ("precision", "recall", "f1"):
        axis.plot(frame["threshold"], frame[column], label=column.title())
    axis.set(xlabel="Vulnerability threshold", ylabel="Score", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_evaluation(labels, probabilities, threshold, plot_dir: Path, role: str) -> None:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    for name, display, values in (
        ("confusion_matrix", ConfusionMatrixDisplay, predictions),
        ("roc_curve", RocCurveDisplay, probabilities),
        ("precision_recall_curve", PrecisionRecallDisplay, probabilities),
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        display.from_predictions(labels, values, ax=axis)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{role}_{name}.png", dpi=120)
        plt.close(figure)
