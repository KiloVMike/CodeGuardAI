# CodeGuard AI

**CPU-first deep transfer learning for C function vulnerability detection (study `cpu_v3`).**

This project trains a multilayer neural classifier on frozen CodeBERT representations of the official
CodeXGLUE–Devign dataset. It does not fine-tune the 125M-parameter encoder — that would require a GPU.
Instead, CodeBERT is encoded once, cached, and reused for controlled experiments comparing:

- C0: Linear probe (baseline)
- C1: Deep MLP head with cross-entropy
- C2: Deep MLP head with weighted cross-entropy (false-negative-sensitive)

All development uses the full official splits (21,854 train / 2,732 validation / 2,732 test). Model
selection and threshold tuning use validation data only. The test set is evaluated exactly once,
after all choices are frozen.

No Google Colab, Google Drive, or cloud runtime is required.

## What is included

- Authentic CodeXGLUE Devign data with pinned revisions
- Dataset quality, duplicate, and cross-split leakage audit
- Resumable, checksummed feature extraction (one-time encoder pass)
- Three core neural-head experiments with identical representations
- Validation-only model selection and threshold optimization
- Frozen test evaluation with delta metrics, confusion matrices, ROC/PR curves
- Token-occlusion explainability with signed influence and disclaimer
- CLI inference, local Gradio UI, artifact-reading presentation notebook
- Self-contained Colab interview notebook (`codeguard interview`) with results baked in
- Unit tests for configuration, metrics, thresholds, losses, selection and head standardization

## Quick start (Windows)

```powershell
cd C:\Users\A_R_COMPUTERS\Downloads\CodeGuardAI
.\setup.bat        # creates .venv with pinned Python 3.12 deps
.\prepare.bat      # downloads dataset and CodeBERT (~200 MB)
```

## Verify installation

```powershell
.\.venv\Scripts\codeguard.exe doctor
pytest
.\.venv\Scripts\codeguard.exe audit
```

## Smoke validation (30 seconds)

```powershell
.\quick_train.bat
# extract --quick → train --experiment all --quick → finalize --quick
# verifies the entire pipeline on 8 samples
```

## Full research study (~2 hours on this CPU)

```powershell
.\train_local.bat
# 1. Feature extraction (train + validation, ~105 min at 400 tokens, resumable per shard)
# 2. Train C0 linear probe (resumable epochs)
# 3. Train C1 MLP (resumable)
# 4. Train C2 weighted MLP (resumable)
# 5. Compare → finalize → notebooks
```

Then evaluate the frozen study once:

```powershell
.\evaluate_final.bat
# Evaluates baseline (C0, threshold 0.50) and selected model on test
# Saves delta metrics, error CSVs, confusion matrices, ROC/PR curves
```

## Inference and UI

```powershell
# CLI single analysis (loads frozen bundle)
.\.venv\Scripts\codeguard.exe analyze .\examples\sample.c

# Local browser UI
.\run_demo.bat
# http://127.0.0.1:7860 (loopback only)
```

## Presentation notebooks

```powershell
# Local evidence reader (reads saved artifacts; never trains)
.\.venv\Scripts\codeguard.exe notebook
# artifacts/cpu_v3/CodeGuardAI_Presentation.ipynb

# Self-contained interview deck for Colab (tables + plots baked in as text/base64)
.\.venv\Scripts\codeguard.exe interview
# artifacts/cpu_v3/CodeGuardAI_Interview_Presentation.ipynb
```

## Documentation

| File | Purpose |
|------|---------|
| `docs/PROJECT_SPEC.md` | Identity, constraints, scope |
| `docs/RESEARCH_PROTOCOL.md` | Data, experiments, freeze rules |
| `docs/ARCHITECTURE.md` | Implementation and artifact contracts |
| `docs/STATUS.md` | Current measured state |
| `docs/DEVELOPMENT_PLAN.md` | Next tasks and acceptance criteria |
| `docs/AI_HANDOFF.md` | Continuation procedure for AI agents |
| `docs/DECISIONS.md` | Methodological decision log |
| `docs/ORIGINAL_PLAN_MAP.md` | Original plan → v2 mapping |
| `docs/RUNBOOK.md` | Step-by-step execution guide |

## Project layout

```
CodeGuardAI/
|-- config/default.toml          Central study configuration
|-- src/codeguard_ai/            Core package
|-- tests/                       Fast offline unit tests
|-- docs/                        Technical documentation
|-- .venv/                       Python 3.12 environment (git-ignored)
|-- .cache/                      Downloads + feature cache (git-ignored)
|-- artifacts/                   Generated checkpoints + evidence (git-ignored)
|-- setup.bat                    Creates .venv + installs pinned deps
|-- prepare.bat                  Downloads dataset + CodeBERT
|-- quick_train.bat              End-to-end smoke validation
|-- train_local.bat              Full study runner
|-- evaluate_final.bat           Frozen test evaluation
|-- run_demo.bat                 Local browser UI
`-- README.md                    This file
```

## Hardware note

This PC has an AMD Ryzen 5 5600G (6C/12T, ~16 GB RAM) with **no NVIDIA GPU**.
- Measured encoder benchmark: 3.91 functions/second at 400 tokens → ~105 min train+validation, ~12 min test
- Neural-head training: minutes on cached features
- The project runs entirely on CPU; no CUDA is used

If a CUDA device becomes available, the same code will use it automatically.

## Interpretation warning

Predictions are **statistical likelihoods**, not vulnerability proofs.
- Highlighted lines show signed model influence, not ground-truth locations
- Softmax scores are uncalibrated
- Frozen encoder limits adaptability
- Truncation at 400 tokens still discards long-function suffixes
- Labels are noisy (function-level, not line-level)
- Source-code overlaps exist in official splits
- No guarantee of unseen-project generalization

Use as supporting evidence alongside static analysis, testing, and expert review.