from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from codeguard_ai.artifacts import read_json
from codeguard_ai.config import AppConfig, select_device
from codeguard_ai.models import mean_pool
from codeguard_ai.training import load_head
from codeguard_ai.workflows import verify_bundle

DISCLAIMER = (
    "Highlighted regions influenced the predicted class; they are not guaranteed vulnerability "
    "locations. Softmax scores are uncalibrated model scores, not security guarantees."
)


class CodeAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if not config.paths.final_config.exists():
            raise FileNotFoundError("No v2 bundle. Complete training and 'codeguard finalize'.")
        final = read_json(config.paths.final_config)
        bundle = verify_bundle(config, final)
        self.device = select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True)
        self.spec = read_json(bundle / "selected" / "model_spec.json")
        self.head = None
        if self.spec["architecture"] == "lora":
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification

            base = AutoModelForSequenceClassification.from_pretrained(
                bundle / "encoder", local_files_only=True, num_labels=2,
            )
            self.model = PeftModel.from_pretrained(base, bundle / "selected").to(self.device).eval()
        else:
            self.model = AutoModel.from_pretrained(
                bundle / "encoder", local_files_only=True,
            ).to(self.device).eval()
            members = (final.get("ensemble") or {}).get("members") or []
            self.ensemble_members = [bundle / f"member_{name}" for name in members]
            self.head = load_head(
                self.ensemble_members[0] if members else bundle / "selected", self.device,
            )
            self.extra_heads = [load_head(directory, self.device)
                                for directory in self.ensemble_members[1:]]
        self.threshold = final["threshold"]
        self.max_length = final["max_length"]
        self.quick_test_model = final["quick"]

    def logits(self, inputs):
        output = self.model(**inputs)
        if self.head is None:
            return output.logits
        pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
        if not self.extra_heads:
            return self.head(pooled)
        # Frozen ensemble: exact mean of member probabilities; softmax(log p) == p.
        probabilities = torch.stack(
            [head(pooled).softmax(-1) for head in [self.head, *self.extra_heads]]
        ).mean(0)
        return probabilities.clamp_min(1e-12).log()

    @torch.inference_mode()
    def analyze(self, source_code: str, explain: bool = True) -> dict:
        if not isinstance(source_code, str) or not source_code.strip():
            raise ValueError("Paste a non-empty C function")
        if len(source_code) > 200_000:
            raise ValueError("Input exceeds 200,000 characters; analyze one function at a time")
        started = time.perf_counter()
        full = self.tokenizer(source_code, truncation=False, verbose=False)
        total_tokens = len(full["input_ids"])
        encoded = self.tokenizer(
            source_code, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        probability = float(self.logits(inputs).softmax(-1)[0, 1])
        label = int(probability >= self.threshold)
        lines, tokens = ([], [])
        if explain:
            lines, tokens = self._occlusion(source_code, inputs, offsets, probability, label)
        return {
            "prediction": "Potentially Vulnerable" if label else "Low Vulnerability Likelihood",
            "label": label, "vulnerability_probability": probability,
            "confidence": probability if label else 1 - probability,
            "threshold": self.threshold, "influential_lines": lines, "influential_tokens": tokens,
            "original_token_count": total_tokens, "input_truncated": total_tokens > self.max_length,
            "max_length": self.max_length, "explanation_target_class": label,
            "explained_tokens": len(tokens), "processing_seconds": time.perf_counter() - started,
            "quick_test_model": self.quick_test_model, "architecture": self.spec["architecture"],
            "disclaimer": DISCLAIMER,
        }

    def _occlusion(self, source, inputs, offsets, probability, label):
        positions = [index for index, (start, end) in enumerate(offsets) if end > start]
        limit = self.config.inference.explanation_max_tokens
        if len(positions) > limit:
            positions = [positions[i] for i in np.linspace(0, len(positions) - 1, limit, dtype=int)]
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("Tokenizer has no mask token for occlusion")
        tokens = []
        for start in range(0, len(positions), self.config.inference.explanation_batch_size):
            group = positions[start : start + self.config.inference.explanation_batch_size]
            batch = {key: value.repeat(len(group), 1) for key, value in inputs.items()}
            for row, position in enumerate(group):
                batch["input_ids"][row, position] = mask_id
            changed = self.logits(batch).softmax(-1)[:, 1].cpu().tolist()
            for position, score in zip(group, changed, strict=True):
                begin, end = offsets[position]
                influence = (probability - score) * (1 if label == 1 else -1)
                tokens.append({"start": begin, "end": end, "text": source[begin:end],
                               "influence": influence})
        scores = defaultdict(float)
        for token in tokens:
            first = source.count("\n", 0, token["start"]) + 1
            last = source.count("\n", 0, max(token["start"], token["end"] - 1)) + 1
            for number in range(first, last + 1):
                scores[number] += token["influence"] / (last - first + 1)
        scale = max((abs(value) for value in scores.values()), default=1.0) or 1.0
        source_lines = source.split("\n")
        ranked = sorted(scores.items(), key=lambda item: abs(item[1]), reverse=True)
        lines = [{"line_number": number, "code": source_lines[number - 1],
                  "influence": score, "normalized_influence": score / scale,
                  "direction": "supports" if score >= 0 else "opposes"}
                 for number, score in ranked[:self.config.inference.top_lines]]
        return lines, tokens


def analyze_code(source_code: str, analyzer: CodeAnalyzer, explain: bool = True) -> dict:
    """Reuse loaded weights for repeated calls."""
    return analyzer.analyze(source_code, explain)
