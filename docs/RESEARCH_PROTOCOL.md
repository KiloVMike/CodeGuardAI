# CodeGuard AI — Research protocol

## 1. Data identity

| Item | Value |
|------|-------|
| Dataset | CodeXGLUE–Devign (Defect Detection) |
| Source | Hugging Face `google/code_x_glue_cc_defect_detection` |
| Revision | `69bd48c03223c2104342acd9a807caf61ac3efb8` (train 21,854 / val 2,732 / test 2,732) |
| Labels | 0 = non-vulnerable, 1 = vulnerable |
| Split preservation | Official train/validation/test never merged or resampled |
| Missing/empty code | Present in the official data; **not dropped** (logged as limitation) |
| Exact overlaps | 8 train/val, 4 train/test, 1 val/test; preserved and reported |

## 2. Encoder identity

| Item | Value |
|------|-------|
| Model | `microsoft/codebert-base` |
| Revision | `3b0952feddeffad0063f274080e3c23d75e7eb39` |
| Parameters | 124,645,632 total (frozen in core study) |
| Tokenizer | RoBERTa fast, vocab 50,265 |
| Max tokens | 400 (full run) / 128 (smoke) |
| Pooling | Masked mean of final hidden states |

## 3. Core experiments (required)

| Exp | Architecture | Loss | Purpose |
|-----|--------------|------|---------|
| C0 | Linear (768→2) | CE | Frozen-encoder baseline; minimal trainable params |
| C1 | 768→512→128→2 (GELU+Dropout) | CE | Deep non-linear head capacity |
| C2 | 768→512→128→2 | Weighted CE | False-negative-sensitive hypothesis |

Inputs are standardized with mean/std fitted on the train split only; the statistics live inside
the head checkpoints.

### Optional extensions
| Exp | Architecture | Loss | Note |
|-----|--------------|------|------|
| C3 | 768→512→128→2 | Focal (α=weights, γ=2) | Alternative imbalance handling |
| L0 | LoRA (r=8, α=16) | CE | Genuine transformer adaptation |
| L1 | LoRA (r=8, α=16) | Weighted CE | LoRA + imbalance |

**LoRA is not required for the CPU study.** It genuinely trains attention query/value adapters but does not guarantee a 1–2 hour full-dataset run on this hardware.

## 4. Controlled comparison invariants

For a fair experiment group:
- Same dataset split, same revision
- Same random seed (123456)
- Same `max_length` (400 full, 128 smoke)
- Same encoder revision and pooling
- Same head architecture for C1/C2 (linear probe uses single layer)
- Same training/validation split
- Same epoch budget and early-stopping patience
- Validation metrics computed from saved predictions

Any deviation from the above invalidates direct comparison and must be documented.

## 5. Development freeze rule

Before test evaluation:
1. All required core experiments complete and verified
2. Validation comparison table generated (including tuned-threshold metrics per candidate)
3. Model selected by validation ROC-AUC, then tuned F1, then recall; threshold optimized on
   validation probabilities over the grid 0.05–0.95 step 0.01 (ties: F1, recall, fewer FN,
   closest to 0.50)
4. Optional two-model probability ensemble kept only if it beats the single model on validation
5. Model bundle created and hash-verified
6. `final_config.json` written

After step 6, **no further training, threshold change, preprocessing change, or experiment addition is allowed for that study**. Any new study must change the `study` name in configuration.

## 6. Test evaluation protocol

- Baseline (linear probe, threshold 0.50) and selected model (optimized threshold) evaluated once
- Metrics: accuracy, precision, recall, F1, ROC-AUC, average precision, TP/TN/FP/FN
- Deliverables per role: confusion matrix, ROC curve, PR curve, prediction CSV, error CSV
- Delta table: selected minus baseline for accuracy, precision, recall, F1, false negatives
- If interrupted, `evaluate` re-displays `test_comparison.json` without re-running inference

## 7. Error and limitation analysis

Post-evaluation, the following are saved for review (not for tuning):
- `baseline_errors.csv` and `selected_errors.csv` with source code
- False negatives stratified by token length and truncation
- High-confidence errors (>0.90 or <0.10)
- Qualitative notes recorded separately by the student

## 8. Explainability scope

- Method: token occlusion with `[MASK]`
- Batch size: configurable (default 4)
- Token budget: `explanation_max_tokens` (default 128)
- Signed influence: supports or opposes the predicted class
- Line aggregation: distribute token influence across spanned lines
- Disclaimer: **not ground-truth vulnerability localization**
- UI displays signed values and direction

## 9. Reproducibility requirements

Every study must produce:
- `provenance.json`: packages, platform, threads, CUDA
- `run_config.json`: complete experiment specification
- `completed.json`: finished epoch count and signature
- `history.json`: per-epoch train/validation metrics
- `validation_predictions.csv`: full validation split
- `validation_metrics.json`: final validation metrics
- `feature_report.json`: truncation rates, cache locations
- `experiment_log.csv`: validation comparison history
- Bundle manifest with per-file SHA-256

## 10. Presentation expectations

The local notebook (`CodeGuardAI_Presentation.ipynb`) must:
- Load saved artifacts; not train or evaluate test data
- Display dataset audit, feature truncation, validation comparison
- Show machine-generated plots only
- Include explicit "PENDING" markers for missing artifacts
- Provide an optional live inference cell (explicit opt-in)
- End with limitations and pending student/faculty review

The interview notebook (`CodeGuardAI_Interview_Presentation.ipynb`, `codeguard interview`) is
self-contained: tables and plots are baked in as text/base64 so it opens in Colab without project
files. It cites published reference numbers only as literature (marked "verify citation") and
labels every measured number with its source.

## 11. Negative results are valid

If weighted CE or deep head does not improve over linear probe, that outcome is a valid result. Do not silently drop experiments or modify the protocol to force improvement.