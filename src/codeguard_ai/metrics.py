from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    aligned = labels_array.ndim == 1 and probability_array.shape == labels_array.shape
    if not aligned or not len(labels_array):
        raise ValueError("Labels and probabilities must be aligned non-empty 1D arrays")
    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("Labels must be binary")
    in_range = np.all((probability_array >= 0) & (probability_array <= 1))
    if not np.isfinite(probability_array).all() or not in_range:
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0,1]")
    labels_array = labels_array.astype(np.int64)
    predictions = (probability_array >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    roc_auc = None
    if len(np.unique(labels_array)) == 2:
        roc_auc = float(roc_auc_score(labels_array, probability_array))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "average_precision": float(average_precision_score(labels_array, probability_array))
        if len(np.unique(labels_array)) == 2 else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def find_best_threshold(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.01,
) -> tuple[float, list[dict[str, Any]]]:
    if not 0 <= minimum <= maximum <= 1 or step <= 0:
        raise ValueError("Invalid threshold grid")
    thresholds = np.round(np.arange(minimum, maximum + (step / 2), step), decimals=10)
    rows = [classification_metrics(labels, probabilities, float(value)) for value in thresholds]
    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -row["false_negatives"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), rows
