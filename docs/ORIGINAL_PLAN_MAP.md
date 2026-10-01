# CodeGuard AI — Original plan mapping (v1 → v2)

This document records every phase from the original `CODEGUARD_AI_DEVELOPMENT_PLAN.md` and its disposition in the CPU-first v2 project.

## Phase mapping

| Original phase | Original title | v2 disposition | Notes |
|----------------|----------------|----------------|-------|
| Phase 0 | Project Initialization | ✅ Merged into `setup.bat`, `prepare.bat`, config | No Colab; local venv, pinned deps, Drive → local artifacts |
| Phase 1 | Dataset Acquisition & Verification | ✅ `prepare`, `audit`, `audit_and_plot` | Full official splits; audit preserves overlaps |
| Phase 2 | Exploratory Data Analysis | ✅ `audit` + `reporting.audit_and_plot` | Token lengths, truncation rates, class distributions, plots |
| Phase 3 | CodeBERT Preprocessing | ✅ `features.extract_split` | Masked mean pooling; cached shards; truncation accounting |
| Phase 4 | Baseline CodeBERT Model | ⚠️ **Revised** | Official full fine-tuning → frozen encoder + linear probe (C0) |
| Phase 5 | Baseline Test Evaluation | ⚠️ **Revised** | Baseline test eval moved after freeze (with final model) |
| Phase 6 | Weighted CE Experiment | ✅ C2 `weighted` | Same data, representations, head; loss variant only |
| Phase 7 | Focal Loss Experiment | ✅ Optional C3 `focal` | Optional; same data/representations |
| Phase 8 | Validation Comparison | ✅ `compare` | CSV + training plots; core experiments required |
| Phase 9 | Final Model Selection | ✅ `finalize` | Validation F1 → recall → order; threshold on validation |
| Phase 10 | Final Test Evaluation | ✅ `evaluate` | Once, after freeze; baseline + selected; delta table |
| Phase 11 | Error Analysis | ✅ `evaluate` saves error CSVs | Qualitative review by student; no post-test tuning |
| Phase 12 | Explainable AI | ✅ `CodeAnalyzer._occlusion` | Token occlusion, signed influence, disclaimer |
| Phase 13 | Unified Inference Function | ✅ `analyze_code` / `CodeAnalyzer` | Reusable analyzer; CLI, notebook, UI |
| Phase 14 | Interactive Colab Demo | ⚠️ **Revised** | Local Gradio UI bound to 127.0.0.1 |
| Phase 15 | Presentation Notebook Cleanup | ✅ `generate_notebook` | Artifact-reading notebook; explicit PENDING markers |
| Phase 16 | Reproducibility & Final Saving | ✅ Bundle manifest + provenance | Checksums, per-file hashes, full config freeze |
| Phase 17 | Final End-to-End Verification | ✅ `verify_bundle` + re-eval idempotency | Bundle hash check; test evaluation resumable |
| Phase 18 | Reproducibility (dupe) | — | Consolidated into Phase 16 |
| Phase 19 | Final Audit | ✅ `AI_HANDOFF.md` checklist | State inspection, validation, documentation update |

## Key methodological revisions

| Original assumption | v2 implementation | Rationale |
|---------------------|-------------------|-----------|
| Google Colab + Drive | Local venv + project artifacts | No cloud dependency; runs on this PC |
| Full 125M fine-tuning | Frozen encoder + neural heads | CPU feasible; 1× feature pass amortized |
| Official baseline reproduction | Linear probe (C0) + deep heads (C1/C2) | Honest about methodology; original baseline needs GPU |
| 400 token length | 400 tokens in study `cpu_v3` (256 in legacy `cpu_v2`) | Original length restores context; truncation still disclosed |
| 5 epochs full fine-tune | 24 epochs head training (early stop, patience 5) | Head converges fast on cached features |
| Early baseline test eval | Both baseline + final test after freeze | Prevents test-set leakage into development |
| Google Drive persistence | Checksummed local bundles | Portable; no account needed |
| `ipywidgets` demo | Gradio 6 server | Modern, maintained, works locally |

## Preserved research integrity rules

From the original plan, these are **unchanged**:
- ✅ Official CodeXGLUE–Devign data, official splits
- ✅ Binary labels 0/1, vulnerable = positive class
- ✅ Validation-only model selection and threshold tuning
- ✅ Test set evaluated exactly once, after all choices frozen
- ✅ No post-test retraining or threshold changes
- ✅ Explainability = influential regions, not ground truth
- ✅ Machine-generated tables/plots only
- ✅ Student must understand and explain all code

## What was dropped (with reason)

| Original item | Reason |
|---------------|--------|
| Colab-specific setup cells | Replaced by `setup.bat` + `configure_local_cache` |
| Drive mount + folder creation | Local artifacts; `.gitignore` excludes large dirs |
| `/content/` temporary paths | All paths project-relative |
| `QUICK_TEST_MODE` / `RESUME_TRAINING` / `TRAIN_MODEL` / `LOAD_FINAL_MODEL` global flags | Explicit CLI commands with `--quick` / `--resume` |
| 30 recent papers requirement | Outside code scope; literature review is student work |
| Streamlit/FastAPI/database/backend | Out of scope in both plans |
| CWE-type classification | Out of scope in both plans |
| Cloud deployment | Out of scope in both plans |

## Smoke vs full namespace

The eight-sample smoke runs (`--quick`) have a **separate artifact namespace**:
- `artifacts/cpu_v2/smoke/...`
- Never mixed with full study in `compare`, `finalize`, or `evaluate`
- Exist only to verify the pipeline end-to-end in < 30 seconds