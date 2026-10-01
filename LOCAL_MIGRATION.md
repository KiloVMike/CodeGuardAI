# Local Migration Record

## Decision

CodeGuard AI now runs as an installable Python project on Windows. Google Colab, `/content`,
Google Drive mounts, notebook setup cells, and Drive checkpoint paths are not part of the runtime.

## Local Replacements

| Former assumption | Local implementation |
|---|---|
| Colab package-install cells | Project-owned Python 3.12 `.venv` via `setup.bat` |
| `/content` temporary storage | Paths rooted at this project directory |
| Google Drive persistence | `artifacts/` with resumable checkpoints |
| Colab model/data downloads | Pinned revisions cached in `.cache/` |
| Notebook switches | Explicit `codeguard` CLI commands |
| `ipywidgets` demo | Local Gradio server bound to `127.0.0.1` |
| Manual result tables | JSON/CSV/PNG files generated from predictions |

## Verified On This PC

- Python 3.12 virtual environment creation
- CPU-only PyTorch runtime detection
- Official CodeXGLUE Devign download and split validation
- Pinned CodeBERT download and classifier construction
- Dataset duplicate and leakage audit
- Quick forward pass, backward pass, optimizer step, and validation
- Best-model and resumable checkpoint saving
- Checkpoint resume loading
- Validation-only threshold selection
- CLI prediction and batched occlusion explanation
- Local browser server response at `http://127.0.0.1:7860`

## Current Artifact Status

The model currently finalized in `artifacts/final_model` is an eight-sample smoke-test model so
the local CLI and UI can be exercised immediately. It is clearly marked `quick_test_model: true`
and must not be cited as a research result. The code blocks test-set evaluation until a full model
has been trained and finalized.

## Official-Data Audit Finding

The preserved official splits contain exact source-code overlap. The verified local audit found
8 train/validation overlaps, 4 train/test overlaps, and 1 validation/test overlap. These records
are not silently removed because the project contract requires preserving official splits. The
finding is stored in `artifacts/results/dataset_audit.json` and must be disclosed as a limitation.
