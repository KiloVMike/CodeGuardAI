from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codeguard_ai.artifacts import provenance, write_json
from codeguard_ai.config import configure_local_cache, load_config, select_device


def build_parser() -> argparse.ArgumentParser:
    from codeguard_ai.models import CORE_EXPERIMENTS, EXPERIMENTS

    parser = argparse.ArgumentParser(prog="codeguard", description="CPU-first CodeGuard AI v2")
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Local runtime and package information")
    commands.add_parser("prepare", help="Cache official data and pretrained encoder")
    commands.add_parser("audit", help="Quality/leakage audit and EDA; no model test metrics")
    commands.add_parser("benchmark", help="Time encoder inference on 32 training functions")
    extract = commands.add_parser("extract", help="Resume train/validation feature extraction")
    extract.add_argument("--quick", action="store_true")
    train = commands.add_parser("train", help="Train neural heads or optional LoRA")
    train.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--quick", action="store_true")
    for name in ("compare", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS),
                             default=list(CORE_EXPERIMENTS))
        command.add_argument("--quick", action="store_true")
    commands.add_parser("evaluate", help="Evaluate frozen models; resume after interruption")
    analyze = commands.add_parser("analyze", help="Analyze one C file with a saved bundle")
    analyze.add_argument("file", type=Path)
    analyze.add_argument("--no-explain", action="store_true")
    analyze.add_argument("--quick", action="store_true")
    demo = commands.add_parser("demo", help="Start local browser UI")
    demo.add_argument("--open", action="store_true")
    demo.add_argument("--quick", action="store_true")
    demo.add_argument("--port", type=int, default=7860,
                      help="Preferred port; the demo falls back to the next free port")
    notebook = commands.add_parser("notebook", help="Generate the presentation notebook")
    notebook.add_argument("--quick", action="store_true")
    commands.add_parser("interview", help="Generate the self-contained Colab interview notebook")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, quick=getattr(args, "quick", False))
        configure_local_cache(config)
        if args.command == "doctor":
            result = {**provenance(), "device": str(select_device()),
                      "project": str(config.paths.root), "artifacts": str(config.paths.artifacts)}
        elif args.command == "prepare":
            from codeguard_ai.data import load_raw_dataset
            from codeguard_ai.models import load_encoder, load_tokenizer

            dataset = load_raw_dataset(config)
            tokenizer, encoder = load_tokenizer(config), load_encoder(config)
            result = {"splits": {name: len(split) for name, split in dataset.items()},
                      "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
                      "vocabulary": len(tokenizer), "project": config.project.__dict__,
                      "environment": provenance()}
            write_json(config.paths.results / "provenance.json", result)
        elif args.command == "audit":
            from codeguard_ai.reporting import audit_and_plot

            result = audit_and_plot(config)
        elif args.command == "benchmark":
            from codeguard_ai.features import benchmark

            result = benchmark(config)
        elif args.command == "extract":
            from codeguard_ai.features import extract_development

            result = extract_development(config)
        elif args.command == "train":
            from codeguard_ai.models import CORE_EXPERIMENTS
            from codeguard_ai.training import run_training

            experiments = CORE_EXPERIMENTS if args.experiment == "all" else [args.experiment]
            result = {
                name: str(run_training(config, name, resume=args.resume)) for name in experiments
            }
        elif args.command in ("compare", "finalize", "evaluate"):
            from codeguard_ai.workflows import compare, evaluate_final, finalize_model

            if args.command == "evaluate":
                result = evaluate_final(config)
            else:
                action = compare if args.command == "compare" else finalize_model
                result = action(config, args.experiments)
        elif args.command == "analyze":
            from codeguard_ai.inference import CodeAnalyzer

            result = CodeAnalyzer(config).analyze(
                args.file.read_text(encoding="utf-8"), not args.no_explain,
            )
        elif args.command == "notebook":
            from codeguard_ai.reporting import generate_notebook

            result = {"notebook": str(generate_notebook(config))}
        elif args.command == "interview":
            from codeguard_ai.reporting import generate_interview_notebook

            result = {"interview_notebook": str(generate_interview_notebook(config))}
        elif args.command == "demo":
            from codeguard_ai.web import launch_demo

            return launch_demo(config, open_browser=args.open, port=args.port)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
