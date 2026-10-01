# CodeGuard AI: start here

## Authority and required reading

This is the active **CPU-first v2** project, rooted in this directory. Read, in order:

1. `docs/PROJECT_SPEC.md` — identity, fixed scope, original-plan mapping.
2. `docs/RESEARCH_PROTOCOL.md` — data, experiments, selection, test freeze.
3. `docs/ARCHITECTURE.md` — implementation and artifact contracts.
4. `docs/STATUS.md` — what actually exists and what was verified.
5. `docs/DEVELOPMENT_PLAN.md` — next incomplete task and its acceptance criteria.
6. `docs/AI_HANDOFF.md` — continuation procedure and known failure modes.
7. `docs/DECISIONS.md` — reasons behind methodological choices.
8. `docs/RUNBOOK.md` before running anything expensive.

The four `../CODEGUARD_AI*.md` files are historical inputs, not active runtime instructions.
`docs/ORIGINAL_PLAN_MAP.md` records every original phase and its disposition. Latest explicit
user instructions take precedence over project documents; record any resulting design change.

## Invariants

- Target: AMD Ryzen 5 5600G, 6C/12T, about 16 GB RAM, native Windows, Python 3.12, CPU.
- Full official CodeXGLUE–Devign splits: 21,854 train / 2,732 validation / 2,732 test.
- `0 = non-vulnerable`, `1 = vulnerable`; binary precision/recall/F1 refer to class 1.
- Default: frozen CodeBERT mean-pooled representations + trainable neural heads. This is deep
  transfer learning, **not** CodeBERT fine-tuning or an official-baseline reproduction.
- Optional LoRA really trains attention adapters but is not required for the CPU study. Do not
  promise a 1–2 hour full-dataset LoRA run or guaranteed academic acceptance.
- Never replace official data with generated code, silently subsample full runs, merge splits,
  use test performance for development, or claim unrun experiments as complete.
- Quick/smoke data and outputs have a separate namespace. They are never research results.
- Preserve legacy artifacts under `artifacts/` outside the v2 study; never select them as v2 models.
- Freeze the baseline, selected checkpoint, preprocessing, and threshold before test evaluation.
- After test evaluation starts, do not tune/retrain/refinalize that study. Resume the same frozen
  evaluation if interrupted; change the study name only for a genuinely new, documented study.
- Explainability describes signed influence, not ground-truth vulnerability localization.
- No cloud, Drive, Colab, paid API, account, external inference, database, or Node build required.

## Work discipline

Inspect current code and artifacts before editing. Do not trust old prose over machine evidence.
Use explicit implemented / verified / pending / blocked statuses. Never tick student-understanding
or faculty-approval items on the student's behalf. Preserve user work and existing checkpoints.
Run targeted tests for changed numerical, persistence, and research-protocol logic.
Update `STATUS.md`, `DEVELOPMENT_PLAN.md`, and `AI_HANDOFF.md` before handing off.
Do not start multi-hour jobs silently; use the benchmark and runbook to make the cost visible.
Use the local `.venv\Scripts\python.exe`. Do not install into system Python.
Do not delegate unless the user explicitly asks.
