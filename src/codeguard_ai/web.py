from __future__ import annotations

import json
import socket
import time
import traceback

import gradio as gr

from codeguard_ai.artifacts import read_json
from codeguard_ai.config import AppConfig
from codeguard_ai.inference import CodeAnalyzer

VULNERABLE_EXAMPLE = """void copy_data(const char *src) {
    char buffer[10];
    strcpy(buffer, src);
}"""

SAFER_EXAMPLE = """int copy_data(char *dst, size_t dst_size, const char *src) {
    if (dst == NULL || src == NULL || dst_size == 0) return -1;
    int written = snprintf(dst, dst_size, "%s", src);
    return (written < 0 || (size_t)written >= dst_size) ? -1 : 0;
}"""

SUBTLE_EXAMPLE = """int sum_positive(const int *values, int count) {
    int total = 0;
    for (int i = 0; i <= count; i++) {
        total += values[i];
    }
    return total;
}"""

IDLE_SUMMARY = "## Awaiting code\n\nPaste a C function or click an example."
IDLE_BADGE = (
    '<div style="border-left:6px solid #64748b; background:#f1f5f9; padding:14px 18px;'
    ' border-radius:6px;"><div style="font-size:1.15rem; font-weight:700; color:#334155;">'
    'Awaiting code</div></div>'
)
ERROR_BADGE_COLOR = "#b91c1c"
ERROR_BADGE_FILL = "#fee2e2"

CSS = """
.gradio-container { max-width: 1180px !important; margin: auto !important; }
.hero { padding: 20px 0 12px; border-bottom: 2px solid #314158; margin-bottom: 14px; }
.model-banner p { margin: 0 !important; }
footer { margin-top: 8px; }
"""


def find_free_port(preferred: int = 7860, span: int = 10) -> int | None:
    """Return the first bindable localhost port, or None if every candidate is taken."""
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    return None


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def format_badge(result: dict) -> str:
    vulnerable = bool(result["label"])
    color = "#b91c1c" if vulnerable else "#166534"
    fill = "#fee2e2" if vulnerable else "#dcfce7"
    probability = result["vulnerability_probability"] * 100
    width = min(max(probability, 0.0), 100.0)
    return (
        '<div style="border-left:6px solid {color}; background:{fill}; padding:14px 18px;'
        ' border-radius:6px;">'
        '<div style="font-size:1.15rem; font-weight:700; color:{color};">{prediction}</div>'
        '<div style="margin-top:4px;">Probability <b>{probability:.1f}%</b>'
        ' &middot; decision threshold <b>{threshold:.0f}%</b></div>'
        '<div style="background:#e5e7eb; border-radius:4px; height:8px; margin-top:8px;">'
        '<div style="width:{width:.1f}%; background:{color}; height:8px; border-radius:4px;">'
        '</div></div></div>'
    ).format(
        color=color, fill=fill, prediction=result["prediction"],
        probability=probability, threshold=result["threshold"] * 100, width=width,
    )


def format_summary(result: dict) -> str:
    lines = [
        f"## {result['prediction']}",
        f"- **Vulnerability probability:** {result['vulnerability_probability'] * 100:.2f}%",
        f"- **Decision threshold:** {result['threshold'] * 100:.2f}%",
        f"- **Predicted-class score (uncalibrated):** {result['confidence'] * 100:.2f}%",
        f"- **Processing time:** {result['processing_seconds']:.2f}s",
        f"- **Tokens:** {result['original_token_count']} seen, "
        f"{result['max_length']} analyzed",
        f"- **Head:** {result['architecture']}",
    ]
    if result["quick_test_model"]:
        lines += ["", "> Smoke-test model, not a research-quality trained model."]
    if result["input_truncated"]:
        lines += ["", f"> Long function: only the first {result['max_length']} tokens "
                      "were analyzed."]
    return "\n".join(lines)


def format_explanation(result: dict) -> str:
    items = result["influential_lines"]
    if not items:
        body = ("No line-level influence was computed (explanation unchecked, or the "
                "function contained no explainable tokens).")
    else:
        rows = "\n".join(
            f"| {item['line_number']} | `{_cell(item['code'])}` "
            f"| {item['influence']:+.4f} | {item['direction']} the predicted class |"
            for item in items
        )
        body = ("| Line | Code | Influence | Effect |\n| --- | --- | --- | --- |\n" + rows)
    return (
        f"{body}\n\nOcclusion over **{result['explained_tokens']}** tokens."
        f"\n\n_{result['disclaimer']}_"
    )


def error_outputs(message: str) -> tuple[str, str, str, str]:
    color, fill = ERROR_BADGE_COLOR, ERROR_BADGE_FILL
    badge = (
        f'<div style="border-left:6px solid {color}; background:{fill};'
        ' padding:14px 18px; border-radius:6px;">'
        f'<div style="font-size:1.15rem; font-weight:700; color:{color};">'
        "Analysis not completed"
        "</div></div>"
    )
    summary = f"## Could not analyze\n\n{message}"
    return badge, summary, "", ""


def model_banner(analyzer: CodeAnalyzer, config: AppConfig) -> str:
    final = read_json(config.paths.final_config)
    bundle = str(final.get("bundle", "")).replace("\\", "/").rstrip("/").split("/")[-1] or "-"
    mode = ("smoke-test model (not a research result)"
            if analyzer.quick_test_model else "frozen research bundle")
    return (
        f"> **Model:** `{analyzer.spec['experiment']}` head on frozen "
        f"`{analyzer.spec['model']}` &middot; {mode} &middot; **threshold** "
        f"{analyzer.threshold:.2f} &middot; **max length** {analyzer.max_length} tokens "
        f"&middot; **device** `{analyzer.device}` &middot; **bundle** `{bundle}`"
    )


def launch_demo(config: AppConfig, open_browser: bool = False,
                port: int = 7860) -> int:
    print(f"[demo] loading model from {config.paths.final_config.parent} ...", flush=True)
    started = time.perf_counter()
    analyzer = CodeAnalyzer(config)
    print(f"[demo] model ready in {time.perf_counter() - started:.1f}s", flush=True)

    def run_analysis(code: str | None, explain: bool = True) -> tuple[str, str, str, str]:
        try:
            result = analyzer.analyze(code or "", explain=bool(explain))
        except (ValueError, RuntimeError) as error:
            return error_outputs(str(error))
        except Exception:
            traceback.print_exc()
            return error_outputs(
                "Unexpected internal error while analyzing this input — the full "
                "traceback was printed in the console running the demo."
            )
        try:
            return (
                format_badge(result),
                format_summary(result),
                format_explanation(result),
                json.dumps(result, indent=2, allow_nan=False),
            )
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
            return error_outputs("The model returned an unexpected result structure.")

    def reset_outputs() -> tuple[str, str, str, str, str]:
        return IDLE_BADGE, IDLE_SUMMARY, "", "", ""

    with gr.Blocks(title="CodeGuard AI", analytics_enabled=False) as demo:
        gr.Markdown(
            "# CodeGuard AI\nCPU-first deep transfer learning for C functions. "
            "Predictions are advisory, not a security review.",
            elem_classes="hero",
        )
        gr.Markdown(model_banner(analyzer, config), elem_classes="model-banner")
        with gr.Row():
            with gr.Column(scale=3):
                code = gr.Code(label="C function", language="c", lines=20)
                explain = gr.Checkbox(label="Generate occlusion explanation", value=True)
                with gr.Row():
                    analyze = gr.Button("Analyze", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                badge = gr.HTML(value=IDLE_BADGE)
                summary = gr.Markdown(IDLE_SUMMARY)
                explanation = gr.Markdown()
        with gr.Accordion("Machine-readable result", open=False):
            raw_result = gr.Code(label="JSON", language="json")

        outputs = [badge, summary, explanation, raw_result]
        gr.Examples(
            examples=[[VULNERABLE_EXAMPLE], [SAFER_EXAMPLE], [SUBTLE_EXAMPLE]],
            inputs=[code],
            outputs=outputs,
            fn=run_analysis,
            run_on_click=True,
            label="Example functions — click one to run the analysis",
        )
        analyze.click(run_analysis, [code, explain], outputs, concurrency_limit=1)
        clear.click(reset_outputs, [], [*outputs, code])
    demo.queue(max_size=8, default_concurrency_limit=1)

    chosen = find_free_port(port)
    if chosen is None:
        raise RuntimeError(f"No free localhost port in {port}..{port + 9}")
    if chosen != port:
        print(f"[demo] port {port} is busy (another demo may be running) — using {chosen}",
              flush=True)
    url = f"http://127.0.0.1:{chosen}"
    print(f"[demo] open {url} — press Ctrl+C to stop", flush=True)
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=chosen,
            inbrowser=open_browser,
            theme=gr.themes.Monochrome(),
            css=CSS,
            show_error=False,
        )
    except KeyboardInterrupt:
        print("\n[demo] stopped.", flush=True)
    return 0
