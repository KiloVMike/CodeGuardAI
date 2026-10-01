# CodeGuard AI — AI agent handoff procedure

## 1. Read order (mandatory)
1. `AGENTS.md`
2. `docs/PROJECT_SPEC.md`
3. `docs/RESEARCH_PROTOCOL.md`
4. `docs/ARCHITECTURE.md`
5. `docs/STATUS.md` — current measured state
6. `docs/DEVELOPMENT_PLAN.md` — next incomplete task
7. `docs/RUNBOOK.md` — before running any long job

## 2. State inspection (do this first)

```powershell
cd C:\Users\A_R_COMPUTERS\Downloads\CodeGuardAI
.\.venv\Scripts\codeguard.exe doctor
# Check artifact timestamps and completed.json existence under artifacts/cpu_v3/
```

**Never assume a task is done because it appears in a plan. Check the actual artifacts.**
Current study is **`cpu_v3`** (400 tokens, `[512,128]` head, threshold grid 0.05–0.95,
train-fit standardization). `artifacts/cpu_v2/` is legacy on this machine and never selected.
The official test set for `cpu_v3` has **not** been opened yet
(`artifacts/cpu_v3/results/test_opened.json` must not exist until task 8).

## 3. Common continuation scenarios

### Scenario A: Fresh environment (new machine or deleted `.venv`)
```powershell
.\setup.bat
.\prepare.bat
```

### Scenario B: Feature extraction incomplete
```powershell
.\.venv\Scripts\codeguard.exe extract
# or run train_local.bat which calls extract + all core experiments
```

### Scenario C: Training interrupted
```powershell
.\.venv\Scripts\codeguard.exe train --experiment linear --resume
# repeat for mlp, weighted
```

### Scenario D: Core experiments done, need comparison
```powershell
.\.venv\Scripts\codeguard.exe compare
```

### Scenario E: Comparison done, need freeze
```powershell
.\.venv\Scripts\codeguard.exe finalize
```

### Scenario F: Frozen bundle exists, need test evaluation
```powershell
.\.venv\Scripts\codeguard.exe evaluate
```

### Scenario G: Full pipeline (unattended)
```powershell
.\train_local.bat
# then later
.\evaluate_final.bat
```

### Scenario H: Presentation deck for interview/Colab
```powershell
.\.venv\Scripts\codeguard.exe interview
# -> artifacts/cpu_v3/CodeGuardAI_Interview_Presentation.ipynb
# Self-contained: tables and plots are baked in; upload to Colab as-is.
# Before task 8 the notebook lists its PENDING sections; regenerate after evaluate.
```

## 4. Known failure modes and resolutions

| Symptom | Likely cause | Resolution |
|---------|--------------|------------|
| `Resumable checkpoint at ... not found` | No `latest.pt` | Run without `--resume` or start new study |
| `Resume configuration/data changed` | Config edited after run | Restore original config or rename study |
| `Feature cache is incomplete/corrupt` | Partial shard or moved cache | Delete `.cache/features/<contract>/` and re-extract |
| `Bundle manifest changed after freeze` | File modified manually | Never edit frozen artifacts; re-run finalize |
| `Smoke models cannot open the official test set` | Running `evaluate` with `quick=true` | Run full study or use `--quick` only for smoke UI |
| `Validation predictions have wrong IDs/labels/order` | Dataset fingerprint mismatch | Re-run extract (data/encoder contract changed) |
| `Test evaluation already started with different config` | `test_opened.json` signature mismatch | Do not modify frozen study; start new study if needed |
| `CUDA out of memory` | Not applicable (CPU) | N/A |
| `peft not installed` for LoRA | Optional dependency missing | `python -m pip install -e ".[lora]"` |
| Ensemble members missing in bundle | `finalize` stored only single-model roles, or bundle edited | Re-run `finalize` before freeze; never edit a frozen bundle |
| `interview` notebook shows PENDING sections | Generated before the corresponding command ran | Regenerate with `codeguard interview` after the run |

## 5. Numerical validation checklist

Before declaring any experiment complete:
- [ ] `completed.json` exists and signature matches `run_config.json`
- [ ] `validation_predictions.csv` row count = validation split size
- [ ] `validation_predictions.csv` IDs match current validation split
- [ ] Recomputed metrics from predictions match `validation_metrics.json`
- [ ] Training history shows reasonable loss decrease (or plateau)
- [ ] Early stopping triggered or epoch budget exhausted

## 6. Prohibited actions

- **Never** delete or modify files under `artifacts/cpu_v3/final_model/` after freeze
- **Never** run `evaluate` on a smoke study (`quick=true`)
- **Never** silently drop missing/empty code rows — they are limitations
- **Never** present test metrics as if they were validation metrics
- **Never** merge train/validation/test splits
- **Never** tune or refinalize `cpu_v3` after `test_opened.json` exists
- **Never** select or mix `artifacts/cpu_v2/` (legacy) or smoke outputs into `cpu_v3`
- **Never** claim LoRA, probe, or smoke results are the official CodeBERT baseline
- **Never** fabricate metrics, plots, or references
- **Never** tick student-understanding or faculty-approval boxes

## 7. Updating documentation before hand-off

Always update:
- `STATUS.md` — measured results, timings, current phase
- `DEVELOPMENT_PLAN.md` — check off completed tasks
- `docs/AI_HANDOFF.md` — this file, if new failure modes found

## 8. Hand-off checklist

Before responding "done" or handing off:

- [ ] Read all docs in section 1 order
- [ ] Ran `doctor` and confirmed environment
- [ ] Verified current artifact state matches `STATUS.md`
- [ ] Identified the actual next incomplete task from `DEVELOPMENT_PLAN.md`
- [ ] Ran targeted tests for any changed logic
- [ ] Updated `STATUS.md`, `DEVELOPMENT_PLAN.md` with new state
- [ ] Did not start multi-hour jobs without user confirmation and `RUNBOOK.md` reference