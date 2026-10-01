# CodeGuard AI — Development plan (study `cpu_v3`)

## Current study: `cpu_v3`

**All tasks below are complete (2026-10-01).** The test set has been opened once; the study is
frozen. Measured results live in `STATUS.md` section 5. Nothing may be tuned for this study.

## Task list

| # | Task | Command | Prerequisites | Acceptance criteria | Status |
|---|------|---------|---------------|---------------------|--------|
| 1 | Provenance + audit + benchmark @400 | `codeguard prepare` / `audit` / `benchmark` | `.venv` ready | provenance, audit, `benchmark.json` written under `artifacts/cpu_v3/results/` | ✅ |
| 2 | Full feature extraction (train+val @400) | `codeguard extract` | Task 1 | `feature_report.json`; completed_rows 21,854 / 2,732; truncation fraction recorded | ✅ 4,001 s + 499 s; 55.7% / 55.5% truncated |
| 3 | Train C0 linear probe | `codeguard train --experiment linear` | Task 2 | `C0_LINEAR_CE/completed.json`; best epoch saved; validation metrics | ✅ stopped epoch 9 |
| 4 | Train C1 MLP | `codeguard train --experiment mlp` | Task 2 | `C1_MLP_CE/completed.json`; best epoch saved; validation metrics | ✅ stopped epoch 10 |
| 5 | Train C2 weighted MLP | `codeguard train --experiment weighted` | Task 2 | `C2_MLP_WEIGHTED/completed.json`; best epoch saved; validation metrics | ✅ stopped epoch 10 |
| 6 | Compare (tuned metrics + AUC ranking) | `codeguard compare` | Tasks 3–5 | `validation_comparison.csv` with `tuned_*` columns; training plots | ✅ |
| 7 | Freeze study (selection + bundle) | `codeguard finalize` | Task 6 | `final_config.json` points to hash-verified bundle; ranking/threshold rules recorded; ensemble kept or rejected by validation | ✅ selected `mlp`, thr 0.33, ensemble rejected, bundle `8f0a21195194fc1912b9` |
| 8 | One-time test evaluation | `codeguard evaluate` | Task 7 | `test_comparison.json`; baseline/selected CSVs, error CSVs, plots; `test_opened.json` written | ✅ F1 0.647 vs 0.505 baseline; signature `5057547dc1ed1815…` |
| 9 | Presentation notebooks | `codeguard notebook` + `codeguard interview` | Task 8 | local evidence notebook + self-contained Colab interview notebook with no PENDING sections | ✅ 25-cell deck, 487 KB |

Tasks 2–9 are automated by `train_local.bat` (2–7 + notebooks) and `evaluate_final.bat` (8 + 9).

## Resume rules

- `extract` is idempotent: restarts from last committed shard (`.cache/features/<contract>/`).
- `train --resume` loads `latest.pt`, validates the signature, continues from the next epoch.
- If `completed.json` exists with a matching signature, the command exits immediately.
- Configuration or data changes between resume and the original run raise an error.
- `evaluate` may be re-run after interruption: it resumes the same frozen evaluation and simply
  re-displays results if `test_comparison.json` already exists.

## Optional extensions (after the core study)

| Task | Command | Note |
|------|---------|------|
| Focal loss MLP | `codeguard train --experiment focal` | Same data, different loss |
| LoRA CE | `python -m pip install -e ".[lora]"` then `codeguard train --experiment lora` | Optional; installs `peft` |
| LoRA weighted | `codeguard train --experiment lora-weighted` | LoRA + class-weighted loss |

Each extension must use the same feature contract and data split to be comparable. Extensions are
**not** required for tomorrow's presentation.

## Blocking conditions

| Condition | Resolution |
|-----------|------------|
| `train` finds an existing incomplete run without `--resume` | Use `--resume` or rename the study |
| Resume signature mismatch | Restore original config or start a new study |
| `finalize` experiment configs differ | Re-run experiments with identical config |
| `evaluate` on a smoke study | Run the full study first |
| `evaluate` on a modified bundle | Bundle verification fails; never edit frozen artifacts |
| `interview` shows PENDING sections | Expected before task 8; regenerate afterwards |

## Measured full-study wall time (`cpu_v3`, this machine, 400 tokens)

| Phase | Measured |
|-------|----------|
| Encoder pass train + validation (3.91 samples/s) | 4,001 s + 499 s ≈ **75 min** |
| C0 / C1 / C2 head training (early stop) | ~20–25 s each |
| Compare + finalize (threshold grid + ensemble check) | < 1 min |
| Test extraction + evaluation + notebooks | ~9 min |
| **Total** | **≈ 85 minutes** |

## Checkpoint locations (`cpu_v3`)

| Artifact | Path |
|----------|------|
| Feature shards | `.cache/features/<contract>/` |
| Training checkpoints | `artifacts/cpu_v3/checkpoints/<EXP_ID>/` |
| Best models | `<EXP_DIR>/best_model/` (head carries standardization buffers) |
| Resume state | `<EXP_DIR>/latest.pt` |
| Validation predictions | `<EXP_DIR>/validation_predictions.csv` |
| Study comparison | `artifacts/cpu_v3/results/validation_comparison.csv` |
| Frozen bundle | `artifacts/cpu_v3/final_model/<DIGEST>/` (+ `member_*` roles when an ensemble wins) |
| Test outputs | `artifacts/cpu_v3/results/` (`test_comparison.json`, `*_test.csv`, `*_errors.csv`) |
| Presentation notebooks | `artifacts/cpu_v3/CodeGuardAI_Presentation.ipynb` and `CodeGuardAI_Interview_Presentation.ipynb` |

## Documentation update requirements

Before hand-off, update:
- `STATUS.md` with measured timings and results
- `AI_HANDOFF.md` with current state and next command
- This file's task table
