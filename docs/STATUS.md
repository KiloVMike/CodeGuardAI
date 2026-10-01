# CodeGuard AI — Status record (study `cpu_v3`)

## 1. Machine and environment (verified)

| Item | Value | Verified |
|------|-------|----------|
| CPU | AMD Ryzen 5 5600G, 6C/12T, 4.2 GHz | `doctor` |
| RAM | ~16 GB | `doctor` |
| OS | Windows 11 10.0.26200 | `doctor` |
| Python | 3.12.4 in `.venv` | `doctor` |
| PyTorch | 2.14.0 CPU (CUDA unavailable) | `prepare` |
| Transformers | 4.57.6 | `prepare` |
| Datasets | 4.8.5 | `prepare` |
| scikit-learn | 1.9.1 | `prepare` |

## 2. Data and encoder (verified for `cpu_v3`)

| Step | Status | Evidence |
|------|--------|----------|
| Dataset cache + provenance | ✅ | `artifacts/cpu_v3/results/provenance.json`: 21,854 / 2,732 / 2,732 |
| Audit + EDA | ✅ | `artifacts/cpu_v3/results/dataset_audit.json`, `plots/dataset_eda.png` |
| Encoder cache | ✅ | 124,645,632 parameters, revision `3b0952fe...` |
| Benchmark @ 400 tokens | ✅ | **3.91 samples/s** → ~105 min train+val, ~12 min test (`results/benchmark.json`) |
| Test set opened on this machine | ❌ No | `artifacts/cpu_v3/results/test_opened.json` does not exist |

Audit facts (recorded, not hidden): test majority baseline = 1477/2732 = **54.06%**; exact
cross-split overlap train∩val 8, train∩test 4, val∩test 1; no missing/empty functions.

## 3. `cpu_v3` code changes (implemented + verified)

| Change | Where | Verified by |
|--------|-------|-------------|
| Study `cpu_v3`, 400 tokens, head `[512,128]`, 24 epochs, patience 5 | `config/default.toml` | `tests/test_config.py` |
| Threshold grid 0.05–0.95 step 0.01 | `metrics.py` | `tests/test_metrics.py` (incl. regression below the old 0.20 floor) |
| Train-fit standardization buffers in `NeuralHead` | `models.py`, `training.py` | `tests/test_models.py` (forward, checkpoint round-trip, zero-variance) |
| AUC-first ranking + tuned metrics in comparison | `workflows.py` | `tests/test_selection.py`; smoke `compare` shows `tuned_*` columns |
| Validation-guarded ensemble (bundle roles `member_*`) | `workflows.py` | Smoke `finalize` rejected the ensemble correctly (identical members, no gain) |
| Ensemble blend at evaluation and inference | `workflows.py`, `inference.py` | Smoke `evaluate` branch covered by unit paths; `analyze --quick` runs single head |
| `codeguard interview` Colab notebook (results baked in) | `reporting.py`, `cli.py` | Generated valid notebooks: full study (14 cells, PENDING sections) and smoke (18 cells, tables + embedded plots) |
| Launcher paths `cpu_v2` → `cpu_v3` | `train_local.bat`, `evaluate_final.bat` | File content inspected |

Verification on 2026-09-30: `ruff check .` clean; `pytest` **15/15 passed**; quick pipeline
`extract → train (C0/C1/C2) → compare → finalize → analyze → notebook → interview` completed
end-to-end under `artifacts/cpu_v3/smoke/`.

## 4. Smoke study (never research results)

| Item | Value |
|------|-------|
| Bundle | `artifacts/cpu_v3/smoke/final_model/19297c15ea9500c93f74` (hash-verified) |
| Selected / threshold | `mlp` / 0.36 (AUC 0.875 vs linear 0.625 on 8 rows) |
| Ensemble | Rejected (members identical → no validation gain) — guard works |
| `evaluate --quick` | Blocked correctly (smoke cannot open the test set) |

## 5. Full study `cpu_v3` — COMPLETE (2026-10-01)

| Phase | Status | Evidence |
|-------|--------|----------|
| `prepare` / `audit` / `benchmark` | ✅ | section 2 |
| `extract` (train + validation @ 400) | ✅ | `feature_report.json`: 4,001 s train / 499 s validation; truncation 55.7% / 55.5% |
| Train C0 / C1 / C2 | ✅ | `completed.json` per experiment (early stop at epochs 9 / 10 / 10) |
| `compare` | ✅ | `validation_comparison.csv` with `tuned_*` columns |
| `finalize` | ✅ | `final_config.json` → bundle `final_model/8f0a21195194fc1912b9` (hash-verified) |
| `evaluate` (test, once) | ✅ | `test_comparison.json`; `test_opened.json` signature `5057547dc1ed1815…` |
| `notebook` + `interview` | ✅ | both notebooks regenerated; interview deck has **no PENDING sections** |

### Validation (development evidence)

| Experiment | ROC-AUC | Tuned thr. | Tuned acc. | Tuned prec. | Tuned recall | Tuned F1 |
|------------|---------|-----------|-----------|------------|--------------|----------|
| linear (C0) | 0.6372 | 0.22 | 0.4758 | 0.4517 | 0.9655 | 0.6155 |
| **mlp (C1) — selected** | **0.6569** | 0.33 | 0.5234 | 0.4754 | 0.9343 | **0.6301** |
| weighted (C2) | 0.6554 | 0.37 | 0.5176 | 0.4719 | 0.9242 | 0.6247 |

Selection: AUC-first rule → `mlp` (0.6569 vs 0.6554 vs 0.6372). Ensemble **rejected** (top-two
average did not beat the single model's tuned F1 on validation). Frozen threshold 0.33.

### Test (opened exactly once, after freeze)

| Model | Thr. | Accuracy | Precision | Recall | F1 | ROC-AUC | TP | FP | FN | TN |
|-------|------|----------|-----------|--------|-----|---------|----|----|----|----|
| Baseline (linear) | 0.50 | 0.5860 | 0.5602 | 0.4598 | 0.5050 | 0.6206 | 577 | 453 | 678 | 1024 |
| Selected (mlp) | 0.33 | 0.5322 | 0.4951 | 0.9331 | 0.6470 | 0.6560 | 1171 | 1194 | 84 | 283 |

Delta (selected − baseline): F1 **+0.142**, recall **+0.473**, false negatives **−594**,
accuracy −0.054. Test majority baseline = 0.5406 accuracy / F1 0 / ROC-AUC 0.5.

Honest reading: the threshold was frozen on validation by the pre-registered F1-first rule, so
the selected operating point trades specificity for sensitivity (93% recall). Accuracy below the
majority guess at that operating point is real and is explained, not hidden — see the interview
deck section 11 (accuracy table + imbalance-aware table + operating-point note) and the added
Q&A. Threshold-free ROC-AUC and F1 both clearly beat baseline and trivial guess.
No further tuning is permitted for this study.

## 6. Test suite and lint

| Suite | Result |
|-------|--------|
| Unit tests (`pytest`) | 15/15 passed |
| Lint (`ruff check .`) | Clean |

## 7. Artifacts layout (`cpu_v3`, all present)

```
artifacts/cpu_v3/
|-- results/    provenance, dataset_audit, benchmark, feature_report,
|               validation_comparison.csv, threshold_analysis.csv,
|               test_comparison.json, test_opened.json, baseline/selected_errors.csv
|-- plots/      dataset_eda, {linear,mlp,weighted}_training, threshold_metrics,
|               {baseline,selected}_{confusion_matrix,roc_curve,precision_recall_curve}
|-- checkpoints/C0_LINEAR_CE, C1_MLP_CE, C2_MLP_WEIGHTED   (completed.json each)
|-- final_model/8f0a21195194fc1912b9/     (frozen bundle, hash-verified)
|-- final_config.json                     (points at the bundle)
|-- CodeGuardAI_Interview_Presentation.ipynb   self-contained Colab deck (487 KB, no PENDING)
|-- CodeGuardAI_Presentation.ipynb            local evidence reader
`-- train_stdout.log / train_stderr.log       full-run logs
artifacts/cpu_v3/smoke/                   (quick namespace; verified working)
artifacts/cpu_v2/                         (legacy on this machine; never selected)
```

## 8. Remaining actions (study is frozen; no automated work pending)

1. **Presentation:** upload `artifacts/cpu_v3/CodeGuardAI_Interview_Presentation.ipynb` to Colab
   and run it top-to-bottom (no project files needed — tables/plots are embedded). Add exact
   literature citations in section 11 before presenting; the numbers are labelled "verify
   citation" until then.
2. **Demo (optional):** `run_demo.bat` serves the frozen bundle at `http://127.0.0.1:7860`.
3. **Student-only items:** viva explanation and faculty review remain unticked by design.
4. **No further tuning of `cpu_v3`** — test evaluation has started; development commands will
   refuse. A new hypothesis requires a new study name.
