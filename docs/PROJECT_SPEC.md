# CodeGuard AI — CPU-first v2 specification

## 1. Identity and objective

**Title:** Resource-Efficient Explainable Source-Code Vulnerability Detection with CodeBERT
and Deep Transfer Learning.

Build, train, evaluate, and demonstrate a local neural system that accepts one C function and
predicts its Devign-style binary vulnerability label. The research question is whether a
multilayer nonlinear classifier and class-weighted training improve validation vulnerable-class
F1/recall over a linear probe of the same CodeBERT representations, within a measured CPU budget.

The application is a presentation surface. The deliverable is the reproducible research pipeline,
trained checkpoints, controlled experiments, results, analysis, and student explanation.

## 2. Fixed constraints

| Item | Contract |
|---|---|
| Actual target machine | Ryzen 5 5600G; 6 physical / 12 logical cores; ~16 GB RAM; Windows |
| Required runtime | Local Python 3.12, PyTorch, Transformers |
| Backbone | `microsoft/codebert-base` at the revision pinned in configuration |
| Dataset | Official CodeXGLUE–Devign via the pinned Google Hugging Face mirror |
| Full-run rows | 21,854 train, 2,732 validation, 2,732 test |
| Input | One C source-code function, preserving punctuation and whitespace |
| Labels | 0 non-vulnerable; 1 vulnerable |
| Main training | Frozen contextual CodeBERT representations + neural classifier |
| Extension | Optional LoRA fine-tuning; separate IDs, checkpoints, and time measurements |
| Persistence | Project-local download caches and versioned study artifacts |
| Final interface | CLI, loopback-only browser application, local presentation notebook |

## 3. Why this is deep learning

The encoder is a pretrained 12-layer Transformer. Its contextual features feed a PyTorch network
with two hidden layers (768 → 512 → 128 → 2), GELU nonlinearities, normalization, and dropout.
The network is genuinely trained with gradient-based optimization on the full training split.
The shallow linear probe is an ablation baseline, not the only model. Loss functions, validation
curves, learned checkpoints, and error analysis demonstrate actual training.

Freezing a pretrained backbone is a valid transfer-learning method. It does not become a dummy
because it is frozen. Equally, it must never be described as fine-tuning the encoder. If the
faculty specifically requires trainable Transformer layers or exact official baseline reproduction,
the optional LoRA or legacy full-fine-tuning research would need separate time and scope approval.
Only the faculty can confirm acceptance; this document does not certify it.

## 4. Scope retained from the first plan

- Authentic Devign data and fixed official splits.
- CodeBERT preprocessing, data-quality/leakage audit, and token-length analysis.
- Baseline, class-sensitive loss experiments, validation selection, threshold optimization.
- Untouched-test baseline versus final comparison after all development choices are frozen.
- False-negative and truncation analysis; genuine predictions, metrics, and plots.
- Token-occlusion attribution and one reusable inference API.
- Presentation-ready local demo and notebook; reproducibility, checkpoints, provenance.

## 5. Explicit revisions

- Colab/Drive become a local virtual environment and portable project-relative artifacts.
- Full 125M-parameter fine-tuning is no longer the default CPU methodology.
- Baseline is named **linear probe**, never "official CodeBERT reproduction".
- Sequence cap is 400 by default (study `cpu_v3`); the actual truncation rate is saved and disclosed.
- Full dataset is retained. Time pressure is not permission to silently drop samples.
- A 1–2 hour total budget is an objective to measure, not a performance guarantee.
- One-time feature extraction is amortized across repeated neural-head experiments.
- The newer original plan's delayed-test discipline supersedes the first plan's early baseline
  test inspection: both baseline and final test results are generated only after freeze.

## 6. Out of scope

Multi-language/repository scanning, CWE labeling, exploit generation, automatic repairs,
authentication, cloud deployment, hosted inference, production security certification, and
claims of a novel Transformer architecture. Literature work must use real checked references.

## 7. Completion criteria

All required full-data experiments run; their metrics/predictions are saved; a final bundle is
frozen; baseline/final test evaluation and plots are generated; failures and limitations are
reviewed; CLI/browser/notebook work offline from saved artifacts; resume is demonstrated; the
student can explain the model and experimental choices. Implementation completion alone is not
research completion. `STATUS.md` is the live record.
