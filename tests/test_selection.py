from codeguard_ai.workflows import _selection_key


def _candidate(auc, f1, recall):
    return {"metrics": {"roc_auc": auc}, "tuned": {"f1": f1, "recall": recall}}


def test_selection_prefers_higher_auc_before_f1() -> None:
    weak_f1_high = _candidate(0.60, 0.80, 0.90)
    strong_auc_low_f1 = _candidate(0.75, 0.50, 0.40)
    assert _selection_key(strong_auc_low_f1) > _selection_key(weak_f1_high)


def test_f1_breaks_auc_ties_before_recall() -> None:
    better_f1 = _candidate(0.70, 0.60, 0.10)
    worse_f1 = _candidate(0.70, 0.50, 0.90)
    assert _selection_key(better_f1) > _selection_key(worse_f1)


def test_missing_auc_scores_as_zero() -> None:
    broken = _candidate(None, 0.99, 0.99)
    working = _candidate(0.55, 0.10, 0.10)
    assert _selection_key(working) > _selection_key(broken)
