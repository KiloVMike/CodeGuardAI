import pytest

from codeguard_ai.metrics import classification_metrics, find_best_threshold


def test_classification_metrics_use_vulnerable_class_as_positive() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9], threshold=0.5)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == pytest.approx(2 / 3)


def test_threshold_selection_uses_validation_f1() -> None:
    threshold, rows = find_best_threshold(
        [0, 0, 1, 1], [0.1, 0.4, 0.45, 0.9], minimum=0.4, maximum=0.5, step=0.05
    )
    assert threshold == pytest.approx(0.45)
    assert len(rows) == 3


def test_default_grid_covers_005_to_095() -> None:
    _, rows = find_best_threshold([0, 1], [0.10, 0.90])
    assert rows[0]["threshold"] == pytest.approx(0.05)
    assert rows[-1]["threshold"] == pytest.approx(0.95)
    assert len(rows) == 91


def test_thresholds_below_the_old_020_floor_are_reachable() -> None:
    # The pre-v3 grid started at 0.20, so this case could only ever predict
    # "not vulnerable" and collapsed to zero recall.
    threshold, _ = find_best_threshold([0, 1], [0.10, 0.15])
    assert threshold == pytest.approx(0.15)
    assert classification_metrics([0, 1], [0.10, 0.15], threshold)["f1"] == 1.0
