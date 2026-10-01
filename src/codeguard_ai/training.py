"""CPU neural-head experiments and optional genuinely trained LoRA adapters."""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from codeguard_ai.artifacts import (
    digest,
    load_state,
    provenance,
    read_json,
    require_development,
    restore_rng,
    rng_state,
    save_state,
    write_json,
)
from codeguard_ai.config import AppConfig, seed_everything, select_device
from codeguard_ai.data import tokenize_split
from codeguard_ai.features import feature_contract, load_features, study_dataset
from codeguard_ai.metrics import classification_metrics
from codeguard_ai.models import EXPERIMENTS, NeuralHead, build_lora, load_tokenizer


def loss_sum(logits, labels, kind, weights):
    ce = F.cross_entropy(logits, labels, reduction="none")
    if kind == "weighted":
        return (ce * weights[labels]).sum()
    if kind == "focal":
        return (weights[labels] * (1 - (-ce).exp()).pow(2) * ce).sum()
    return ce.sum()


def normalization(labels, kind, weights) -> float:
    return float(weights[labels].sum()) if kind == "weighted" else float(labels.numel())


def unpack(batch, device):
    if isinstance(batch, dict):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        return inputs, batch["labels"].to(device)
    return batch[0].to(device), batch[1].to(device)


def forward(model, inputs):
    return model(**inputs).logits if isinstance(inputs, dict) else model(inputs)


def train_epoch(model, loader, optimizer, scaler, scheduler, device, kind, weights, accumulation):
    model.train()
    iterator = iter(loader)
    total, denominator = 0.0, 0.0
    with tqdm(total=len(loader), desc="Training", leave=False) as progress:
        while group := list(itertools.islice(iterator, accumulation)):
            # Normalize the entire effective batch, including a short final accumulation group.
            group_norm = sum(
                normalization(unpack(batch, device)[1], kind, weights) for batch in group
            )
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                inputs, labels = unpack(batch, device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    value = loss_sum(forward(model, inputs), labels, kind, weights)
                scaler.scale(value / group_norm).backward()
                total += float(value.detach())
                progress.update(1)
            denominator += group_norm
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= old_scale:
                scheduler.step()
    return total / denominator


@torch.inference_mode()
def predict(model, loader, device, kind="ce", weights=None):
    model.eval()
    weights = weights if weights is not None else torch.ones(2, device=device)
    probabilities, targets = [], []
    total, denominator = 0.0, 0.0
    for batch in loader:
        inputs, labels = unpack(batch, device)
        logits = forward(model, inputs).float()
        total += float(loss_sum(logits, labels, kind, weights))
        denominator += normalization(labels, kind, weights)
        probabilities.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        targets.extend(labels.cpu().tolist())
    result = classification_metrics(targets, probabilities)
    result["loss"] = total / denominator
    return result, targets, probabilities


def run_training(config: AppConfig, experiment: str, resume: bool = False) -> Path:
    require_development(config)
    experiment_id, architecture, loss_kind = EXPERIMENTS[experiment]
    output = config.paths.checkpoints / experiment_id
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError(f"Run exists: {output}. Use --resume; outputs are not overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if resume and not latest.exists():
        raise FileNotFoundError(f"No committed epoch exists at {latest}")
    seed_everything(config.project.seed)
    device = select_device()
    raw = study_dataset(config)
    contracts = {
        name: feature_contract(config, raw[name], name) for name in ("train", "validation")
    }
    is_lora = architecture == "lora"
    spec = {
        "schema": 2, "experiment": experiment, "experiment_id": experiment_id,
        "architecture": architecture, "loss": loss_kind, "seed": config.project.seed,
        "quick": config.paths.quick, "model": config.project.model_name,
        "revision": config.project.model_revision, "max_length": config.max_length,
        "pooling": config.features.pooling, "data": contracts,
        "head": asdict(config.head), "training": asdict(config.training) if is_lora else None,
        "lora": asdict(config.lora) if is_lora else None,
        "epochs": config.quick_test.epochs if config.paths.quick else (
            config.training.epochs if is_lora else config.head.epochs
        ),
    }
    signature = digest(spec)
    state = load_state(latest) if resume else None
    if state and state["signature"] != signature:
        raise ValueError("Resume configuration/data changed. Restore config or use a new study.")
    if state and state["completed"]:
        write_json(output / "history.json", state["history"])
        write_json(output / "completed.json", {
            "signature": signature, "epochs": state["epoch"] + 1,
        })
        print(f"Already completed: {experiment_id}")
        return output

    generator = torch.Generator()
    tokenizer = load_tokenizer(config)
    if is_lora:
        from transformers import DataCollatorWithPadding

        model = build_lora(config).to(device)
        train_data = tokenize_split(raw["train"], tokenizer, config.max_length)
        val_data = tokenize_split(raw["validation"], tokenizer, config.max_length)
        collator = DataCollatorWithPadding(tokenizer)
        train_batch, val_batch = config.training.train_batch_size, config.training.eval_batch_size
    else:
        train_arrays, _ = load_features(config, raw["train"], "train")
        val_arrays, _ = load_features(config, raw["validation"], "validation")
        model = NeuralHead(architecture, config.head.hidden_sizes, config.head.dropout).to(device)
        # Fit standardization on train only; buffers are stored in every checkpoint.
        model.set_statistics(train_arrays["features"].mean(axis=0),
                             train_arrays["features"].std(axis=0))
        train_data = TensorDataset(torch.from_numpy(train_arrays["features"]),
                                   torch.from_numpy(train_arrays["labels"]))
        val_data = TensorDataset(torch.from_numpy(val_arrays["features"]),
                                 torch.from_numpy(val_arrays["labels"]))
        collator = None
        train_batch = val_batch = config.head.batch_size
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True,
                              collate_fn=collator, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=val_batch, collate_fn=collator, num_workers=0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.training.learning_rate if is_lora else config.head.learning_rate,
        weight_decay=config.training.weight_decay if is_lora else config.head.weight_decay,
    )
    accumulation = config.training.gradient_accumulation_steps if is_lora else 1
    scheduler = None
    if is_lora:
        from transformers import get_linear_schedule_with_warmup

        steps = math.ceil(len(train_loader) / accumulation) * spec["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(steps * config.training.warmup_ratio), steps,
        )
    counts = np.bincount(np.asarray(raw["train"]["target"], dtype=np.int64), minlength=2)
    if min(counts) == 0:
        raise ValueError("Both classes must occur in training")
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best, stale, start = [], -1.0, 0, 0
    if state:
        model.load_state_dict(state["model"], strict=not is_lora)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best = state["history"], state["best"]
        stale, start = state["stale"], state["epoch"] + 1
        restore_rng(state["rng"])
    spec["signature"] = signature
    write_json(output / "run_config.json", spec)
    metadata = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "encoder_frozen": not is_lora, "device": str(device), "class_weights": weights.tolist(),
        "environment": provenance(),
    }
    write_json(output / "provenance.json", metadata)
    tokenizer.save_pretrained(output / "tokenizer")

    for epoch in range(start, spec["epochs"]):
        generator.manual_seed(config.project.seed + epoch)
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler, device,
                                 loss_kind, weights, accumulation)
        metrics, labels, probabilities = predict(model, val_loader, device, loss_kind, weights)
        row = {"epoch": epoch + 1, "train_loss": train_loss, **metrics,
               "seconds": time.perf_counter() - started}
        history.append(row)
        print(f"{experiment_id}: {row}")
        if metrics["f1"] > best:
            best, stale = metrics["f1"], 0
            best_dir = output / "best_model"
            best_dir.mkdir(exist_ok=True)
            if is_lora:
                model.save_pretrained(best_dir, safe_serialization=True)
            else:
                save_state(best_dir / "head.pt", model.state_dict())
            write_json(best_dir / "model_spec.json", spec)
            write_json(output / "validation_metrics.json", metrics)
            pd.DataFrame({"id": raw["validation"]["id"], "label": labels,
                          "probability": probabilities}).to_csv(
                              output / "validation_predictions.csv", index=False)
        else:
            stale += 1
        completed = epoch + 1 == spec["epochs"] or stale >= config.head.patience
        model_state = model.state_dict()
        if is_lora:
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            model_state = {name: value for name, value in model_state.items() if name in names}
        save_state(latest, {
            "signature": signature, "epoch": epoch, "best": best, "stale": stale,
            "model": model_state, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(), "rng": rng_state(), "history": history,
            "completed": completed,
        })
        write_json(output / "history.json", history)
        if completed:
            write_json(output / "completed.json", {"signature": signature, "epochs": epoch + 1})
            break
    return output


def load_head(directory: Path, device):
    spec = read_json(directory / "model_spec.json")
    head = NeuralHead(spec["architecture"], spec["head"]["hidden_sizes"], spec["head"]["dropout"])
    head.load_state_dict(load_state(directory / "head.pt"))
    return head.to(device).eval()
