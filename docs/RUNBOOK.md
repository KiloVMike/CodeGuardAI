# CodeGuard AI — Runbook

## Before you run anything expensive

1. **Read the current state:** `docs/STATUS.md`
2. **Confirm the next task:** `docs/DEVELOPMENT_PLAN.md`
3. **Verify environment:** `.\.venv\Scripts\codeguard.exe doctor`
4. **Check disk space:** artifacts and cache can reach several GB for full study

## One-time setup (if not done)

```powershell
cd C:\Users\A_R_COMPUTERS\Downloads\CodeGuardAI
.\setup.bat          # creates .venv, installs pinned deps (Python 3.12)
.\prepare.bat        # downloads dataset and CodeBERT (~200 MB)
```

## Smoke validation (30 seconds, safe to run anytime)

```powershell
.\quick_train.bat
# runs: extract --quick → train --experiment all --quick → finalize --quick
# verifies: download, tokenization, encoder, heads, freeze, bundle
```

## Full study (~2 hours on this CPU, measured @ 400 tokens)

### Option A: Guided batch script (recommended)
```powershell
.\train_local.bat
# 1. Full feature extraction (resumable, ~105 min at 3.91 samples/s)
# 2. Train C0 linear probe (resumable)
# 3. Train C1 MLP (resumable)
# 4. Train C2 weighted MLP (resumable)
# 5. Compare → finalize → notebooks
```

### Option B: Step-by-step (for debugging)

```powershell
# 1. Feature extraction (idempotent, resumable)
.\.venv\Scripts\codeguard.exe extract
# Watch: .cache/features/<contract>/manifest.json completed_rows

# 2. Core experiments (each resumable with --resume)
.\.venv\Scripts\codeguard.exe train --experiment linear
.\.venv\Scripts\codeguard.exe train --experiment mlp
.\.venv\Scripts\codeguard.exe train --experiment weighted

# 3. Validation comparison
.\.venv\Scripts\codeguard.exe compare

# 4. Freeze study
.\.venv\Scripts\codeguard.exe finalize

# 5. Test evaluation (idempotent after freeze)
.\.venv\Scripts\codeguard.exe evaluate

# 6. Presentation notebooks (local reader + self-contained Colab deck)
.\.venv\Scripts\codeguard.exe notebook
.\.venv\Scripts\codeguard.exe interview
```

## Resume after interruption

| Interrupted at | Resume command |
|----------------|----------------|
| Feature extraction | `codeguard extract` |
| C0 training | `codeguard train --experiment linear --resume` |
| C1 training | `codeguard train --experiment mlp --resume` |
| C2 training | `codeguard train --experiment weighted --resume` |
| Test evaluation | `codeguard evaluate` (re-displays existing result) |

**Do not re-run completed steps.** Each command checks `completed.json` signatures and exits immediately if the work is already done with the same configuration.

## Monitoring progress

```powershell
# Feature extraction
Get-Content .cache/features/*/manifest.json -Raw | ConvertFrom-Json

# Training epochs
Get-Content artifacts/cpu_v3/checkpoints/*/history.json -Raw | ConvertFrom-Json
```

## Expected outputs (full study)

```
artifacts/cpu_v3/
|-- checkpoints/
|   |-- C0_LINEAR_CE/best_model/head.pt
|   |-- C1_MLP_CE/best_model/head.pt
|   |-- C2_MLP_WEIGHTED/best_model/head.pt
|-- results/
|   |-- provenance.json
|   |-- dataset_audit.json
|   |-- feature_report.json
|   |-- validation_comparison.csv
|   |-- threshold_analysis.csv
|   |-- experiment_log.csv
|   |-- test_comparison.json
|   |-- baseline_test.csv
|   |-- selected_test.csv
|   |-- baseline_errors.csv
|   `-- selected_errors.csv
|-- plots/
|   |-- dataset_eda.png
|   |-- *_training.png
|   |-- threshold_metrics.png
|   |-- baseline_confusion_matrix.png
|   |-- baseline_roc_curve.png
|   |-- baseline_precision_recall_curve.png
|   |-- selected_confusion_matrix.png
|   |-- selected_roc_curve.png
|   `-- selected_precision_recall_curve.png
|-- predictions/
|   |-- baseline_test.csv
|   `-- selected_test.csv
|-- final_model/<DIGEST>/
|   |-- encoder/, tokenizer/
|   |-- baseline/, selected/
|   |-- selection.json, bundle.json
|   `-- member_*/  (only when a validation-guarded ensemble wins)
|-- final_config.json
|-- experiment_log.csv
|-- CodeGuardAI_Presentation.ipynb
`-- CodeGuardAI_Interview_Presentation.ipynb
```

## Local inference and UI

```powershell
# CLI single analysis
.\.venv\Scripts\codeguard.exe analyze .\examples\sample.c

# Local browser UI (loads frozen bundle)
.\run_demo.bat
# Opens http://127.0.0.1:7860
```

## Presentation notebooks

```powershell
# Local evidence reader
.\.venv\Scripts\codeguard.exe notebook
# artifacts/cpu_v3/CodeGuardAI_Presentation.ipynb

# Self-contained Colab interview deck (results baked in; upload as-is)
.\.venv\Scripts\codeguard.exe interview
# artifacts/cpu_v3/CodeGuardAI_Interview_Presentation.ipynb
```

## Common troubleshooting

| Issue | Resolution |
|-------|------------|
| `codeguard: command not found` | `.\setup.bat` not run; run in project directory |
| `HF_HUB_OFFLINE` errors | `runtime.offline = false` in config or ensure cache populated |
| `Bundle manifest changed` | Never edit frozen files; re-run `finalize` if config unchanged |
| `Validation predictions have wrong IDs` | Re-run `extract` (encoder/data contract changed) |
| `peft not installed` for LoRA | `python -m pip install -e ".[lora]"` |
| Smoke UI says "not a research-quality model" | Expected; smoke models are 8-sample verification only |
| CPU at 100%, fans loud | Normal for encoder pass; ensure laptop is on power + ventilated |

## Time budgeting

| Phase | Wall time (this machine) | Can pause? |
|-------|--------------------------|------------|
| `setup.bat` + `prepare.bat` | 5–10 min | N/A |
| Feature extraction (train+val @400) | ~105 min | Yes (per shard) |
| C0 linear probe | < 5 min | Yes (per epoch) |
| C1 MLP | < 10 min | Yes (per epoch) |
| C2 weighted MLP | < 10 min | Yes (per epoch) |
| Compare + finalize | < 3 min | N/A |
| Test extraction + evaluation | ~15 min | N/A (idempotent) |
| Notebooks (both) | < 1 min | N/A |

**Do not start the full study without confirming you have the time window.** The encoder pass
alone is ~105 minutes. Use `train_local.bat` which resumes automatically if interrupted.

## After full study completes

1. Verify `docs/STATUS.md` matches actual artifacts
2. Update `docs/DEVELOPMENT_PLAN.md` task checkboxes
3. Review `docs/AI_HANDOFF.md` for new failure modes
4. Regenerate `codeguard interview` — the Colab deck must show no PENDING sections
5. Open the local presentation notebook and walk through cells
6. Run `run_demo.bat` for the live demo
7. Prepare student explanation of:
   - Architecture: frozen encoder → mean pool → neural head
   - Why this is deep learning (transfer learning, not fine-tuning)
   - How weighted loss targets false negatives
   - What the threshold optimizes (validation grid 0.05–0.95)
   - What occlusion measures (signed influence)
   - All limitations (frozen encoder, truncation at 400, overlaps, uncalibrated scores)

## Safety and integrity reminders

- Test evaluation is **irreversible** for a study. After `evaluate`, no further training or selection changes are allowed.
- If you must explore a new hypothesis, change `study` in `config/default.toml` (e.g., `cpu_v4`) and re-run from `extract`.
- Never present smoke-model metrics as research results.
- All tables and plots must be generated from saved CSVs/JSONs, not manually typed.