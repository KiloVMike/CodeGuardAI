# CodeGuard AI — Implementation and artifact contracts

## 1. Runtime structure

```
CodeGuardAI/
|-- config/default.toml          # Central study configuration
|-- src/codeguard_ai/            # Core package
|-- tests/                       # Fast offline unit tests
|-- .venv/                       # Python 3.12 environment
|-- .cache/                      # Local downloads (git-ignored)
|-- artifacts/                   # Generated checkpoints and evidence (git-ignored)
|    |-- cpu_v3/                 # Active study
|    |    |-- smoke/             # Eight-sample validation runs
|    |    |-- checkpoints/       # Training checkpoints per experiment
|    |    |-- results/           # Validation metrics, comparisons, plots
|    |    |-- plots/             # PNG figures
|    |    |-- predictions/       # CSV predictions for test set
|    |    |-- research_evidence/ # Per-experiment markdown
|    |    |-- final_model/       # Frozen bundle directory
|    |    |-- final_config.json  # Frozen selection + threshold
|    |    `-- experiment_log.csv # Validation comparison history
|-- docs/                        # This documentation
|-- setup.bat                    # Creates .venv and installs pinned deps
|-- prepare.bat                  # Downloads dataset and CodeBERT
|-- train_local.bat              # Full-data feature extraction + all core experiments
|-- evaluate_final.bat           # Frozen baseline + final test evaluation
|-- quick_train.bat              # Eight-sample smoke run
|-- run_demo.bat                 # Local browser UI
```

## 2. Configuration contract (`config/default.toml`)

All downstream code reads from this single file. Sections:

| Section | Purpose |
|---------|---------|
| `[project]` | Study identity, seeds, model/dataset revisions |
| `[training]` | Encoder cap, LoRA epochs, batch/accumulation, LR |
| `[quick_test]` | Smoke sample counts and short sequence length |
| `[inference]` | Default threshold, occlusion limits |
| `[runtime]` | CPU threads, offline mode, telemetry |
| `[features]` | Shard size, batch size, pooling method |
| `[head]` | Hidden sizes, dropout, batch, epochs, patience, optimizer |
| `[lora]` | Rank, alpha, dropout |

Values are validated in `load_config`. Invalid configurations raise before any network call.

## 3. Feature extraction contract

The encoder is frozen. Representations are cached once and reused.

### Contract key
`codeguard_ai.features.feature_contract` creates a deterministic dictionary that includes:
- Split name and row count
- Dataset fingerprint
- IDs+labels hash
- Model and dataset revisions
- `max_length`, `pooling`, `quick` flag
- Shard size

### Cache layout
```
.cache/features/<contract-digest>/
|-- manifest.json              # Completed rows, chunk checksums, elapsed time
|-- 00000000.npz               # features (N, 768), ids, labels, lengths
|-- 00000256.npz
|-- ...
```

### Atomicity
Each chunk is written to a temporary file then renamed. `manifest.json` is updated after the chunk is committed. On restart, completed chunks are verified by SHA-256 before continuing.

### Alignment checks
`load_features` raises if:
- Manifest completed rows ≠ split length
- Concatenated IDs/labels do not match the current split
- Features contain non-finite values

### Truncation accounting
Original token lengths (no truncation) are saved per sample. `feature_report.json` records truncated count and fraction for the current cap.

## 4. Neural-head experiments

Three core experiments are required for a complete CPU study:

| ID | Architecture | Loss | Experiment key |
|----|--------------|------|----------------|
| C0 | Linear probe | Cross-entropy | `linear` |
| C1 | 768→512→128→2 | Cross-entropy | `mlp` |
| C2 | 768→512→128→2 | Weighted CE | `weighted` |

Heads standardize inputs with train-split statistics stored as buffers inside the model state.

Optional focal and LoRA experiments use the same data and feature contract.

### Training loop contract
- Optimizer: AdamW with configured LR, weight decay
- Scheduler: Linear warm-up + decay (LoRA only); none for neural heads
- Gradient clipping: 1.0
- Early stopping: Patience on validation vulnerable-class F1
- Checkpointed per epoch:
  - Model state (parameters + standardization buffers)
  - Optimizer, scheduler, scaler, RNG states
  - Epoch index, best F1, stale counter, history
  - Configuration signature
- Best-model checkpoint: highest validation F1; contains full model spec

### Loss functions
- CE: `F.cross_entropy`
- Weighted CE: class weights from training labels, applied after modulation
- Focal: `(1-p)^γ * CE` with α applied per class after modulation (same as weighted CE modulation)
- Validation loss uses the same formula as training

### Metrics contract
`classification_metrics` returns:
- accuracy, precision, recall, F1, ROC-AUC, average precision
- TP, TN, FP, FN at the given threshold
- Vulnerable class = positive label = 1

## 5. Validation selection and freezing

### Candidate requirements
- Completed experiment with matching data/seed/head/quick settings
- Validation predictions aligned with split IDs/labels
- Saved metrics match recomputed metrics from predictions

### Selection rule
Highest validation ROC-AUC (threshold-free); ties by tuned F1, then recall.

### Threshold optimization
Grid 0.05–0.95 step 0.01 on validation probabilities. Optimize:
1. F1, 2. recall, 3. fewer false negatives, 4. closest to 0.50

### Ensemble guard
The top two candidates' probabilities are averaged on validation; the ensemble replaces the
single model only if its tuned F1 is strictly higher. Members are stored as `member_<name>/`
bundle roles and blended identically during evaluation and inference.

### Frozen bundle
Directory `final_model/<selection-digest>/` containing:
```
encoder/        # Frozen CodeBERT, saved via `save_pretrained`
tokenizer/      # Tokenizer, saved via `save_pretrained`
baseline/       # Linear probe best head
selected/       # Selected head
member_*/       # Ensemble members (only when the ensemble was kept)
selection.json  # All selection data, thresholds, hashes
bundle.json     # Manifest: every file path + SHA-256
```
`final_config.json` points to this bundle by relative path. `verify_bundle` re-hashes every file and compares against `bundle.json` and `selection.json` before any test evaluation.

## 6. Test evaluation contract

### Rules
- Smoke studies cannot call `evaluate_final`.
- Test evaluation is blocked if the frozen configuration has changed.
- Evaluation is idempotent: re-running reproduces identical outputs without re-inspecting the test set.
- After evaluation starts, the study is marked "test opened". Any development command refuses further tuning.

### Outputs
Per role (baseline, selected):
- CSV predictions with `id`, `label`, `probability`, `prediction`, `token_length`, `truncated`
- Error CSV with source code for false positives and false negatives
- Confusion matrix, ROC curve, PR curve PNG
- `test_comparison.json` with per-role metrics and delta

## 7. Inference contract

`CodeAnalyzer` loads the frozen bundle and exposes:
```python
analyze(source_code: str, explain: bool = True) -> dict
```
Return keys:
- `prediction`, `label`, `vulnerability_probability`, `confidence`
- `threshold`, `influential_lines`, `influential_tokens`
- `original_token_count`, `input_truncated`, `max_length`
- `explanation_target_class`, `explained_tokens`
- `processing_seconds`, `quick_test_model`, `architecture`
- `disclaimer`

Occlusion uses `[MASK]` token replacement in batches. Influences are signed (supports/opposes predicted class) and normalized to the maximum absolute line score.

## 8. Artifact integrity functions

`codeguard_ai.artifacts` provides:
- `digest(obj)` — SHA-256 of canonical JSON
- `file_hash(path)` — SHA-256 of file contents
- `atomic_write(path, writer)` — temporary file + rename
- `save_state/load_state` — torch state dicts with `weights_only=True`
- `rng_state/restore_rng` — Python, torch, numpy, CUDA
- `provenance()` — packages, platform, threads, CUDA
- `require_development(config)` — refuses if test opened
- `write_json/read_json` — UTF-8, sorted keys, no NaN

All checkpoints and manifests use these primitives.

## 9. Offline capability

- `runtime.offline = true` sets `HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE`
- `local_files_only=True` is passed to tokenizer/encoder loading
- All required files must already exist in `.cache`
- No network call is made by any downstream operation