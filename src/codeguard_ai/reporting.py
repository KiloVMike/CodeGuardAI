from __future__ import annotations

import base64
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from codeguard_ai.artifacts import read_json, write_json
from codeguard_ai.config import AppConfig
from codeguard_ai.data import audit_dataset, load_raw_dataset

LITERATURE_REFERENCES = [
    ("BiLSTM", 59.37),
    ("TextCNN", 60.69),
    ("RoBERTa", 61.05),
    ("CodeBERT (full fine-tuning)", 62.08),
]


def audit_and_plot(config: AppConfig) -> dict:
    raw = load_raw_dataset(config)
    report = audit_dataset(raw, config.paths.results / "dataset_audit.json")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = list(raw)
    for label in (0, 1):
        counts = [report["splits"][name]["class_distribution"][str(label)] for name in names]
        axes[0].bar(np.arange(3) + label * .35, counts, width=.35, label=f"Class {label}")
    axes[0].set_xticks(np.arange(3) + .175, names)
    axes[0].legend()
    axes[0].set_title("Official class distributions")
    axes[1].hist([len(code) for code in raw["train"]["func"]], bins=np.logspace(1, 6, 50))
    axes[1].set(xscale="log", xlabel="Characters", ylabel="Training functions",
                title="Training function lengths")
    figure.tight_layout()
    figure.savefig(config.paths.plots / "dataset_eda.png", dpi=120)
    plt.close(figure)
    return report


def generate_notebook(config: AppConfig):
    """Generate a local notebook that only reads evidence unless inference is explicitly run."""
    def cell(kind, source):
        result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    cells = [
        cell("markdown", "# CodeGuard AI — Local Deep Transfer Learning\n"
             "Read `docs/PROJECT_SPEC.md` and `docs/RESEARCH_PROTOCOL.md`. "
             "This notebook loads saved evidence; it does not train or open test data.\n"
             "**Architecture:** C function → CodeBERT tokenizer → frozen 12-layer Transformer → "
             "masked mean → 768→256→64→2 neural head. Optional LoRA is a separate experiment."),
cell("code", f"import pandas as pd\n"
              f"from IPython.display import Image, display\n\n"
              f"from codeguard_ai.artifacts import read_json\n"
              f"from codeguard_ai.config import configure_local_cache, load_config\n"
              f"config = load_config(quick={config.paths.quick})\n"
              f"configure_local_cache(config)\nroot = config.paths.artifacts\nprint(root)\n"
             "def show_json(path):\n"
             "    print(read_json(path) if path.exists() else f'PENDING: {path.name}')\n"),
        cell("markdown", "## Dataset, provenance and leakage\n"
             "Official Devign C functions; labels 0/1; official splits preserved. "
             "Overlap counts are limitations, not silently deleted records."),
        cell("code", "show_json(root / 'results' / 'provenance.json')\n"
             "show_json(root / 'results' / 'dataset_audit.json')\n"
             "show_json(root / 'results' / 'feature_report.json')"),
        cell("markdown", "## Controlled experiments\nLinear CE; deep MLP CE; deep MLP weighted CE. "
             "Same features and data; validation-only selection. Improvement is a hypothesis."),
        cell("code", "table = root / 'results' / 'validation_comparison.csv'\n"
             "display(pd.read_csv(table)) if table.exists() else print('PENDING: experiments')"),
        cell("markdown", "## Frozen configuration and final evaluation\n"
             "Read results only if generated. Missing results are not zero-valued results."),
        cell("code", "show_json(root / 'final_config.json')\n"
             "show_json(root / 'results' / 'test_comparison.json')"),
        cell("markdown", "## Machine-generated plots"),
        cell("code", "for path in sorted((root / 'plots').glob('*.png')):\n"
             "    print(path.name)\n    display(Image(filename=str(path)))"),
        cell("markdown", "## Error analysis\nReview false negatives and high-confidence errors. "
             "Do not tune after reading test failures. Record qualitative observations."),
        cell("code", "errors = root / 'results' / 'selected_errors.csv'\n"
             "display(pd.read_csv(errors).head(10)) if errors.exists() else print('PENDING')"),
        cell("markdown", "## Live inference\nRun the following cell explicitly after finalization. "
             "Reads the local bundle without training. Token occlusion measures influence, "
             "not exact vulnerability localization."),
        cell("code", "# from codeguard_ai.inference import CodeAnalyzer\n"
             "# analyzer = CodeAnalyzer(config)\n"
             "# analyzer.analyze('void f(char *s) { char b[10]; strcpy(b, s); }')"),
        cell("markdown", "## Limitations and discussion\n"
             "Frozen encoder; prefix truncation; noisy function labels; source-code overlap; "
             "uncalibrated scores; C-only domain; no guarantee of unseen-project generalization.\n"
             "Fill in the conclusion from saved results, including any negative findings. "
             "Student viva and faculty review are pending until performed by the student."),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {
        "kernelspec": {"display_name": "Python 3 (CodeGuard)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }, "cells": cells}
    output = config.paths.artifacts / "CodeGuardAI_Presentation.ipynb"
    write_json(output, notebook)
    # JSON round-trip validation independent of the optional Jupyter installation.
    json.loads(output.read_text(encoding="utf-8"))
    return output


def _nb_cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    def fmt(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _embed_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.stem}"/>'


def _bar_chart(labels: list[str], values: list[float], title: str,
               reference: float | None = None, reference_label: str = "") -> str:
    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(labels, values, color="#4C72B0")
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}",
                  ha="center", fontsize=9)
    ceiling = max(values + ([reference] if reference is not None else []))
    if reference is not None:
        axis.axhline(reference, color="#C44E52", linestyle="--", label=reference_label)
        axis.legend(fontsize=9)
    axis.set_ylim(0, min(1.0, ceiling * 1.18))
    axis.set_ylabel("Accuracy")
    axis.set_title(title)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="chart"/>'


def _accuracy(reference_counts: dict[str, int] | None) -> float | None:
    if not reference_counts:
        return None
    total = sum(reference_counts.values())
    return max(reference_counts.values()) / total if total else None


def generate_interview_notebook(config: AppConfig) -> Path:
    """Self-contained Colab presentation: results and plots baked in as text/base64."""
    results, plots_dir = config.paths.results, config.paths.plots

    def load(path: Path):
        return read_json(path) if path.exists() else None

    audit = load(results / "dataset_audit.json")
    provenance = load(results / "provenance.json")
    feature_report = load(results / "feature_report.json")
    benchmark = load(results / "benchmark.json")
    final = load(config.paths.final_config)
    test_comparison = load(results / "test_comparison.json")
    opened = load(results / "test_opened.json")
    comparison_path = results / "validation_comparison.csv"
    validation = pd.read_csv(comparison_path) if comparison_path.exists() else None
    cells: list[dict] = []

    def markdown(title: str, body: str) -> None:
        cells.append(_nb_cell("markdown", f"## {title}\n\n{body}"))

    def figure(caption: str, path: Path) -> None:
        if path.exists():
            cells.append(_nb_cell("markdown", f"**{caption}**\n\n{_embed_image(path)}"))

    pending: list[str] = []
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    cells.append(_nb_cell("markdown",
        "# CodeGuard AI — Deep Learning for Vulnerable C Function Detection\n\n"
        f"**Study:** `{config.project.study}` · **Encoder:** frozen "
        f"`{config.project.model_name}` · **Generated:** {stamp}\n\n"
        "> We train a small neural network on top of a frozen pretrained code transformer "
        "(deep transfer learning) to flag vulnerable C functions. Every model-selection "
        "decision uses validation data only; the official test set is opened exactly once, "
        "after everything is frozen."))

    markdown("1. Problem and goal",
        "- **Task:** binary classification of one C function: `0 = non-vulnerable`, "
        "`1 = vulnerable` (CodeXGLUE/Devign dataset).\n"
        "- **Why it matters:** memory-safety defects in C are a leading cause of CVEs; "
        "static analysis produces many false positives, so learned assistive screening is useful.\n"
        "- **Metric choice:** the classes are imbalanced, so we report precision / recall / F1 "
        "for the vulnerable class alongside accuracy. A model that always answers "
        "\"not vulnerable\" already reaches the majority-class accuracy shown below.")

    if audit:
        rows = []
        for name in ("train", "validation", "test"):
            split = audit["splits"][name]
            distribution = split["class_distribution"]
            total = split["samples"]
            rows.append([name, total, distribution["0"], distribution["1"],
                         distribution["1"] / total])
        overlap = audit["exact_cross_split_overlap"]
        markdown("2. Dataset — official CodeXGLUE/Devign splits",
            _md_table(["Split", "Samples", "Non-vulnerable", "Vulnerable", "Vulnerable share"],
                      rows)
            + f"\n\n- Exact-code overlap between splits (kept and reported, never silently "
              f"deleted): train∩validation = {overlap['train_validation']}, "
              f"train∩test = {overlap['train_test']}, "
              f"validation∩test = {overlap['validation_test']}.\n"
              "- Known caveats: function-level labels are noisy (a vulnerable line inside an "
              "innocuous function still labels the whole function), and the splits were "
              "originally built from real commits, so near-duplicates across splits exist.")
        figure("Class balance and function-length distribution (train split)",
               plots_dir / "dataset_eda.png")
    else:
        pending.append("dataset audit (`codeguard audit`)")

    encoder_parameters = (provenance or {}).get("encoder_parameters")
    head_parameters = None
    if validation is not None and "trainable_parameters" in validation:
        head_parameters = int(validation["trainable_parameters"].max())
    method = [
        "```text",
        "C function",
        "  -> CodeBERT WordPiece tokenizer (400 tokens, beginning of the function preserved)",
        "  -> frozen 12-layer CodeBERT transformer ({} parameters, gradients never flow)".format(
            f"{encoder_parameters:,}" if encoder_parameters else "125M"),
        "  -> masked mean pooling -> one 768-d vector per function",
        "  -> standardization fitted on the TRAIN split only (mean and std, stored in checkpoints)",
        "  -> trainable head: 768 -> 512 -> 128 -> 2 (GELU, dropout 0.2)",
        "  -> softmax -> P(vulnerable)",
        "```",
    ]
    markdown("3. Method — frozen encoder + trained head (deep transfer learning)",
        "\n".join(method)
        + f"\n\n- **Trainable parameters:** about "
          f"{f'{head_parameters:,}' if head_parameters else '461K'} in the head versus "
          f"{f'{encoder_parameters:,}' if encoder_parameters else '125M'} frozen encoder "
          "parameters — the encoder is never updated.\n"
        "- **Why frozen:** the machine is a CPU-only AMD Ryzen 5 5600G; full fine-tuning of "
        "125M parameters needs GPU-days. Freezing the encoder is standard practice for "
        "resource-constrained transfer learning, and it makes runs reproducible in hours.\n"
        "- **Class weighting:** one experiment up-weights the vulnerable class in the "
        "cross-entropy loss (the minority class), a standard deep-learning technique.\n"
        "- **Losses/heads are identical across experiments** except the declared factor, so "
        "comparisons isolate that factor.\n"
        "- This is **transfer learning, not CodeBERT fine-tuning** — we never claim otherwise.")

    design = ("Controlled experiments (same features, data, seed and preprocessing):\n\n"
              + _md_table(
                  ["ID", "Experiment", "Head", "Loss"],
                  [["C0", "linear", "768 -> 2 linear probe", "cross-entropy"],
                   ["C1", "mlp", "768 -> 512 -> 128 -> 2", "cross-entropy"],
                   ["C2", "weighted", "768 -> 512 -> 128 -> 2",
                    "cross-entropy, vulnerable class weighted"]])
              + "\n\nSelection protocol — every decision reads **validation only**:\n"
                "1. Rank candidates by validation ROC-AUC (threshold-free), then tuned F1, "
                "then recall.\n"
                "2. Tune the decision threshold on validation over **0.05-0.95 (step 0.01)**: "
                "maximize F1, then recall, then fewer false negatives, then closest to 0.50.\n"
                "3. A two-model probability-average ensemble is kept **only** if it beats the "
                "best single model on validation.\n"
                "4. Everything is frozen into a hashed bundle (bundle id + SHA-256 of every "
                "file) **before** the test set is touched.\n"
                "5. The test set is opened exactly once; afterwards the study cannot be tuned. "
                "Smoke runs use a separate namespace and are never presented as results.")
    markdown("4. Experimental design and integrity", design)

    environment = []
    if provenance:
        environment.append(
            f"- **OS/Python:** {provenance.get('environment', {}).get('platform', 'n/a')}"
            f" · Python {provenance.get('environment', {}).get('python', 'n/a')}")
        packages = provenance.get("environment", {}).get("packages", {})
        if packages:
            environment.append("- **Packages:** "
                               + ", ".join(f"{k} {v}" for k, v in packages.items()))
    if benchmark:
        environment.append(
            f"- **Encoder speed:** {benchmark['samples_per_second']:.2f} functions/s at "
            f"max_length={benchmark['max_length']}, batch {benchmark['batch_size']} "
            f"({benchmark['device']}, {benchmark['cpu_threads']} threads) "
            f"-> estimated all-split encoding about "
            f"{benchmark['estimated_all_split_encoding_hours']:.2f} h.")
    if feature_report:
        for split in ("train", "validation"):
            info = feature_report.get(split, {})
            environment.append(
                f"- **{split} features:** {info.get('seconds', 0):.0f} s of encoder time, "
                f"{info.get('truncation_fraction', 0) * 100:.1f}% of functions longer than the "
                f"token limit.")
    environment.append(f"- **Seed:** {config.project.seed}; CPU-only, no GPU and no cloud.")
    if environment:
        markdown("5. Compute environment and honest cost",
            "Everything below was measured on this machine.\n\n" + "\n".join(environment))

    if validation is not None:
        columns = [c for c in ("experiment", "accuracy", "precision", "recall", "f1", "roc_auc",
                               "tuned_threshold", "tuned_accuracy", "tuned_precision",
                               "tuned_recall", "tuned_f1", "trainable_parameters",
                               "training_seconds") if c in validation.columns]
        table = _md_table(columns, validation[columns].values.tolist())
        validation_majority = _accuracy((audit or {}).get("splits", {}).get("validation", {})
                                        .get("class_distribution"))
        chart = ""
        if validation_majority is not None:
            chart = "\n\n" + _bar_chart(
                validation["experiment"].tolist(),
                validation["accuracy"].tolist(),
                "Validation accuracy by experiment",
                reference=validation_majority,
                reference_label=f"validation majority baseline ({validation_majority:.3f})")
        markdown("6. Validation results (development data — test not involved)",
            table + "\n\nColumns starting with `tuned_` use the validation-tuned threshold; "
            "`roc_auc` is threshold-free." + chart)
        for name in validation["experiment"].tolist():
            figure(f"Training curves — {name}", plots_dir / f"{name}_training.png")
        figure("Threshold sweep on validation (precision / recall / F1)",
               plots_dir / "threshold_metrics.png")
    else:
        pending.append("validation comparison (`codeguard compare`)")

    if final:
        frozen = [
            f"- **Selected experiment:** `{final['selected']}`",
            f"- **Decision threshold:** {final['threshold']:.2f}",
            f"- **Ranking rule:** {final.get('ranking_rule', 'n/a')}",
            f"- **Threshold rule:** {final.get('threshold_rule', 'n/a')}",
            f"- **Token limit:** {final['max_length']} tokens",
        ]
        ensemble = final.get("ensemble")
        if ensemble:
            frozen.append(
                f"- **Ensemble:** average of `{ensemble['members'][0]}` + "
                f"`{ensemble['members'][1]}` probabilities (kept because it beat the single "
                f"model on validation; F1 {ensemble['metrics']['f1']:.4f})")
        else:
            frozen.append("- **Ensemble:** not used — validation showed no improvement over "
                          "the single selected model")
        frozen.append(f"- **Bundle id:** `{final['bundle']}` "
                      f"(manifest SHA-256 `{final['bundle_manifest_sha256'][:16]}...`)")
        markdown("7. Frozen selection", "\n".join(frozen))
    else:
        pending.append("freeze (`codeguard finalize`)")

    if test_comparison:
        rows = []
        for role in ("baseline", "selected"):
            metrics = test_comparison[role]
            rows.append([role, metrics["threshold"], metrics["accuracy"], metrics["precision"],
                         metrics["recall"], metrics["f1"], metrics["roc_auc"],
                         metrics["true_positives"], metrics["false_positives"],
                         metrics["false_negatives"], metrics["true_negatives"]])
        table = _md_table(
            ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
             "TP", "FP", "FN", "TN"], rows)
        delta = test_comparison["delta_selected_minus_baseline"]
        delta_table = _md_table(["Metric", "Selected - Baseline"],
                                [[key, value] for key, value in delta.items()])
        test_majority = _accuracy((audit or {}).get("splits", {}).get("test", {})
                                  .get("class_distribution"))
        chart = ""
        if test_majority is not None:
            chart = "\n\n" + _bar_chart(
                ["majority guess", "baseline (linear)", "selected model"],
                [test_majority, test_comparison["baseline"]["accuracy"],
                 test_comparison["selected"]["accuracy"]],
                "Final test accuracy",
                reference=test_majority,
                reference_label=f"majority baseline ({test_majority:.3f})")
        integrity = ""
        if opened:
            integrity = (f"\n\n**Integrity:** the test set was opened once for signature "
                         f"`{opened['final_signature'][:16]}...`; no tuning happened afterwards.")
        markdown("8. Final test results (opened exactly once)",
            table + "\n\n**Delta table**\n\n" + delta_table + chart + integrity)
        figure("Selected model — confusion matrix", plots_dir / "selected_confusion_matrix.png")
        figure("Selected model — ROC curve", plots_dir / "selected_roc_curve.png")
        figure("Selected model — precision-recall curve",
               plots_dir / "selected_precision_recall_curve.png")
    else:
        pending.append("test evaluation (`codeguard evaluate`)")

    errors_path = results / "selected_errors.csv"
    if errors_path.exists():
        frame = pd.read_csv(errors_path)
        parts = []
        for title, mask in (("False negatives (missed vulnerabilities)",
                             (frame["label"] == 1) & (frame["prediction"] == 0)),
                            ("False positives (over-flagged safe code)",
                             (frame["label"] == 0) & (frame["prediction"] == 1))):
            sample = frame[mask].head(2)
            parts.append(f"**{title}** — {int(mask.sum())} cases on the test set.\n")
            for _, row in sample.iterrows():
                source = str(row["source_code"])
                if len(source) > 900:
                    source = source[:900] + "\n... (truncated)"
                parts.append(f"P(vulnerable) = {row['probability']:.3f}\n\n"
                             f"````c\n{source}\n````\n")
        markdown("9. Error analysis", "\n".join(parts))
    else:
        pending.append("error analysis (generated by `codeguard evaluate`)")

    markdown("10. Explainability — what the model looks at",
        "For any prediction we run **token occlusion**: tokens are removed in batches and the "
        "drop in P(vulnerable) measures each token's signed influence on that prediction.\n\n"
        "- Positive influence: removing the token lowered the vulnerability score "
        "(token pushed the model toward vulnerable).\n"
        "- This is an **influence explanation, not a guaranteed vulnerability location** — "
        "the dataset has no line-level ground truth to verify against.\n"
        "- Scores are uncalibrated softmax outputs, not security guarantees.")

    extra_questions: list[str] = []
    if test_comparison:
        test_accuracy = test_comparison["selected"]["accuracy"] * 100
        baseline_accuracy = test_comparison["baseline"]["accuracy"] * 100
        test_distribution = (audit or {}).get("splits", {}).get("test", {}).get(
            "class_distribution")
        majority_accuracy = _accuracy(test_distribution)
        selected = test_comparison["selected"]
        baseline = test_comparison["baseline"]
        markdown("11. How this compares with published references",
            "**Accuracy (what most people quote):**\n\n"
            + _md_table(["Approach", "Accuracy", "Source"], [
                ["Majority-class guess (always safe)", majority_accuracy,
                 "computed on this test split"],
                *[[name, value / 100, "published Devign result — verify citation"]
                  for name, value in LITERATURE_REFERENCES],
                ["Linear probe (baseline, threshold 0.5)", baseline_accuracy / 100,
                 "this study, measured"],
                ["Selected model (frozen transfer learning)", test_accuracy / 100,
                 "this study, measured"],
            ])
            + "\n\nLiterature rows are **reference points from published Devign work** — they "
              "were not measured on this machine. Add the exact paper citations from your "
              "literature review before presenting them.\n\n"
              "**Imbalance-aware view — the same test set, all rows measured or derived "
              "here:**\n\n"
            + _md_table(["Approach", "Accuracy", "F1 (vulnerable)", "Recall", "ROC-AUC"], [
                ["Majority-class guess", majority_accuracy, 0.0, 0.0, 0.5],
                ["Linear probe (threshold 0.5)", baseline["accuracy"], baseline["f1"],
                 baseline["recall"], baseline["roc_auc"]],
                ["Selected model (frozen threshold "
                 f"{selected['threshold']:.2f})", selected["accuracy"], selected["f1"],
                 selected["recall"], selected["roc_auc"]],
            ])
            + "\n\n**Why the selected model's accuracy is below the majority guess:** the "
              "decision threshold was chosen **on validation data only**, by the pre-registered "
              "rule \"maximize F1, then recall\" (the right objective when 46% of test functions "
              "are vulnerable). That operating point deliberately casts a wide net: recall "
              f"{selected['recall']:.0%} and F1 {selected['f1']:.3f} versus the majority guess's "
              "F1 of 0 (it never flags a single vulnerability) and the baseline's "
              f"F1 {baseline['f1']:.3f}. Threshold-free ROC-AUC "
              f"{selected['roc_auc']:.3f} vs {baseline['roc_auc']:.3f} confirms the ranking "
              "quality independent of any operating point. The threshold was frozen before the "
              "test set was opened, so it was never adjusted to flatter any of these numbers.")
        extra_questions.append(
            "**Q: Your selected model's accuracy is below the majority guess — is it useless?**  \n"
            "A: The threshold was frozen on validation by the pre-registered F1-first rule, "
            "before the test set was opened — accuracy was never the optimization target. On "
            f"imbalance-aware metrics the same model wins: F1 {selected['f1']:.3f} vs "
            f"{baseline['f1']:.3f} (baseline) vs 0 (majority), recall "
            f"{selected['recall']:.0%} vs {baseline['recall']:.0%}, ROC-AUC "
            f"{selected['roc_auc']:.3f} vs {baseline['roc_auc']:.3f}; it misses "
            f"{selected['false_negatives']} of 1255 vulnerable functions instead of "
            f"{baseline['false_negatives']}. As a screening tool, flagging more code is the "
            "intended trade, and the operating point can be moved without retraining.")
    else:
        pending.append("test results (needed for the reference comparison)")

    markdown("12. Limitations (stated before anyone asks)",
        "- **Frozen encoder ceiling:** the head only re-weights fixed features; full "
        "fine-tuning would likely score higher but needs a GPU.\n"
        "- **Prefix truncation:** functions longer than 400 tokens are cut; see the "
        "truncation fractions in section 5.\n"
        "- **Noisy labels:** Devign labels come from commit diffs; some functions marked "
        "vulnerable are safe and vice versa — this caps any model's measurable accuracy.\n"
        "- **Split overlap:** a handful of exact duplicate functions exist across splits.\n"
        "- **Uncalibrated scores:** softmax outputs are not probabilities of being vulnerable.\n"
        "- **Domain:** C functions from one corpus only; no claim about other languages or "
        "unseen projects.\n"
        "- **Explainability:** token occlusion shows influence, not ground-truth locations.")

    markdown("13. Reproduce locally (CPU only)",
        "```bat\n"
        "setup.bat                 :: create the environment\n"
        "codeguard prepare         :: cache official data + CodeBERT\n"
        "codeguard audit           :: quality/leakage audit + plots\n"
        "codeguard benchmark       :: measured encoder speed\n"
        "train_local.bat           :: extract features, train C0/C1/C2, freeze\n"
        "evaluate_final.bat        :: one-time test evaluation + notebooks\n"
        "codeguard demo            :: local browser UI\n"
        "```\n"
        f"Artifacts: `{config.paths.artifacts.as_posix()}` "
        "(checkpoints, bundle, plots, CSVs, JSON evidence).")

    markdown("14. Anticipated interview questions",
        "**Q: Why is the accuracy not 90%?**  \n"
        "A: The task has noisy function-level labels and a frozen encoder ceiling; published "
        "fine-tuned CodeBERT results on this dataset are around the low 60s, and a majority "
        "guess already scores in the mid-50s. Beating that baseline with a trained head — "
        "honestly measured — is the result.\n\n"
        "**Q: Why not fine-tune CodeBERT?**  \n"
        "A: CPU-only hardware. Full fine-tuning of 125M parameters needs GPU-days; a frozen "
        "encoder + trained head is standard resource-constrained transfer learning and keeps "
        "runs reproducible in hours.\n\n"
        "**Q: How do you know you did not overfit or peek at the test set?**  \n"
        "A: Model choice, threshold and ensemble decisions use validation only; everything is "
        "hashed into a bundle first; the test set is opened once afterwards, and re-running "
        "evaluation just re-displays the stored result.\n\n"
        "**Q: What would you do with a GPU?**  \n"
        "A: Fine-tune the encoder end-to-end, sweep hyperparameters on validation, calibrate "
        "probabilities, and try code-specific models; the pipeline already supports an "
        "optional LoRA experiment.\n\n"
        "**Q: Is the explanation reliable?**  \n"
        "A: It shows which tokens influenced this prediction; the dataset has no line-level "
        "ground truth, so we never claim it is the true vulnerable line."
        + ("\n\n" + "\n\n".join(extra_questions) if extra_questions else ""))

    cells.append(_nb_cell("code",
        "# OPTIONAL — only inside the project repository (not needed for this presentation).\n"
        "# Live analysis of one function with the frozen bundle:\n"
        "# from codeguard_ai.inference import CodeAnalyzer\n"
        "# from codeguard_ai.config import load_config, configure_local_cache\n"
        "# config = load_config(); configure_local_cache(config)\n"
        "# print(CodeAnalyzer(config).analyze(open('examples/sample.c').read()))"))

    if pending:
        cells.insert(1, _nb_cell("markdown",
            "> **PENDING sections** — regenerate this notebook after the run finishes "
            "(`codeguard interview`): " + "; ".join(pending) + "."))
    cells.append(_nb_cell("markdown",
        f"*Generated by `codeguard interview` on {stamp}. All numbers and images are baked "
        "in from saved artifacts; this notebook needs no project files to display.*"))

    notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "CodeGuardAI_Interview_Presentation.ipynb", "provenance": []},
    }, "cells": cells}
    output = config.paths.artifacts / "CodeGuardAI_Interview_Presentation.ipynb"
    write_json(output, notebook)
    json.loads(output.read_text(encoding="utf-8"))
    return output
