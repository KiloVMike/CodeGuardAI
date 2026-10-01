# Decision log

## ADR-001 — CPU-first transfer learning (accepted for implementation)

User asked for genuine deep learning on the existing AMD CPU with all original data. The earlier
conversation's 1–2 hour LoRA estimates were not measured and are not reliable promises. LoRA
reduces optimizer/trainable-parameter memory but still requires Transformer forward passes and
backpropagation through adapted layers. It does not cache representations across training epochs.

Default: encode each function once with frozen CodeBERT, cache mean-pooled vectors, train a deep
head cheaply, and compare loss variants on identical inputs. Optional LoRA is an extension.
This preserves the original task and data while honestly revising the methodology. Academic
approval of the revision remains the student's/faculty's decision.

## ADR-002 — Full data and sequence cap

Keep all official functions and splits. Cap token sequences at 256 rather than silently shrinking
the dataset. Save original token lengths and truncation counts. The visible prefix is not the
whole function; all reports and inference output disclose this limitation.

## ADR-003 — Mean pooling and deep head

Use attention-mask-weighted mean of the final hidden states (including attended special tokens,
excluding padding). A CodeBERT CLS token has not necessarily been trained for this target task;
mean pooling is a simple, explicit choice. Network: LayerNorm(768), Linear(768,256), GELU,
Dropout(0.2), Linear(256,64), GELU, Dropout(0.2), Linear(64,2). Linear probe uses a single linear
layer on the same vectors. No sklearn classifier masquerades as the deep model.

## ADR-004 — Isolation and integrity

New outputs live in `artifacts/cpu_v2/`, smoke outputs in `artifacts/cpu_v2/smoke/`.
Earlier full-fine-tuning smoke artifacts are historical evidence only. Feature keys include
data identity, source revisions, pooling, cap, and smoke mode. Checkpoints include configuration
signatures and RNG states. Never reuse stale features or checkpoints across incompatible settings.

## ADR-005 — Test protocol

Freeze model, baseline, threshold, preprocessing, and file hashes together. Evaluation marks the
study as opened before computing test predictions. Development commands refuse changes after
this point. Interrupted evaluation may restart against the same frozen bundle. This is a
research-integrity rule inherited from the original plan, not a claim that repeated deterministic
evaluation is intrinsically invalid.

## ADR-006 — Evidence over acceptance promises

No target accuracy is fabricated. Loss weighting is a hypothesis and may fail. A negative result
with proper controls remains useful. Do not call a frozen encoder LoRA, a probe an official
baseline, an eight-sample run a study, or a library integration a novel algorithm.

## ADR-007 — Study `cpu_v3`: 400-token cap (supersedes ADR-002 for this study)

A first `cpu_v2` run reported 54% test accuracy, which equals the majority-class baseline
(1477/2732 non-vulnerable test rows). Diagnosis on validation data (no test inspection here):
prefix truncation at 256 tokens cuts most functions (median source length ~934 characters), and
the tuned-threshold floor could not recover. `cpu_v3` raises the cap to 400 tokens, matching the
original plan's sequence length. Truncation counts stay recorded per split; the visible-prefix
limitation remains disclosed. The 256-token `cpu_v2` cap is superseded for this study only.

## ADR-008 — Decision-threshold grid 0.05–0.95 (fixes majority collapse)

The previous grid (0.20–0.80) could not express the low thresholds a poorly calibrated probe
needs; the effective result was "predict non-vulnerable almost always" at 54% accuracy. The grid
is now 0.05–0.95 step 0.01, still selected on validation only, still breaking ties toward
recall, fewer false negatives, and thresholds near 0.50. Selection ordering added per-candidate
tuned metrics to `validation_comparison.csv` for auditability.

## ADR-009 — Train-fit feature standardization inside the head

Frozen CodeBERT features are not standardized; the MLP head optimizes harder on standardized
inputs. Mean and standard deviation are fitted on the **train split only** and registered as
buffers in `NeuralHead`, so they travel inside every checkpoint, bundle, and inference call —
training, evaluation, and analysis always apply exactly one transform. Zero-variance features are
clamped (std ≥ 1e-6). No validation/test statistic is ever fitted.

## ADR-010 — AUC-first selection, larger head, validation-guarded ensemble

- Ranking: validation ROC-AUC (threshold-free) first, then tuned F1, then recall — an experiment
  that only wins because of one lucky threshold should not own the study.
- Head: 768→512→128→2 (≈461K parameters), 24 epochs, patience 5 — deeper than the v2
  768→256→64→2 head, still trivial on CPU because features are cached.
- Ensemble: average the top two candidates' probabilities on validation and keep it **only** if
  it beats the best single model's tuned F1. Members are stored as extra bundle roles and
  blended identically at evaluation and inference. If validation shows no gain, the single model
  is frozen (observed on smoke: identical members → no gain → ensemble rejected).

## ADR-011 — Study rename `cpu_v2` → `cpu_v3`

Sequence cap, head architecture, threshold protocol, standardization, and selection rule all
changed, and the laptop's `cpu_v2` run already consumed its test set on another machine. A new
study name documents a genuinely new study, keeps this machine's untouched test set
(`test_opened.json` absent) under a clean protocol, and preserves `artifacts/cpu_v2/` as legacy
evidence that is never reselected.
