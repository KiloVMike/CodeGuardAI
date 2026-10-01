import socket

import pytest

from codeguard_ai.web import (
    IDLE_BADGE,
    IDLE_SUMMARY,
    error_outputs,
    find_free_port,
    format_badge,
    format_explanation,
    format_summary,
)


def _result(**overrides) -> dict:
    result = {
        "prediction": "Potentially Vulnerable",
        "label": 1,
        "vulnerability_probability": 0.72,
        "confidence": 0.72,
        "threshold": 0.33,
        "processing_seconds": 0.41,
        "original_token_count": 120,
        "max_length": 400,
        "architecture": "mlp",
        "input_truncated": False,
        "quick_test_model": False,
        "explained_tokens": 18,
        "influential_lines": [
            {"line_number": 3, "code": "strcpy(buffer, src);", "influence": 0.2143,
             "direction": "supports"},
        ],
        "disclaimer": "Signed influence, not a guaranteed location.",
    }
    result.update(overrides)
    return result


def test_find_free_port_returns_bindable_port_in_range() -> None:
    port = find_free_port(7860, span=10)
    assert port is not None
    assert 7860 <= port < 7870
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_find_free_port_skips_an_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        port = find_free_port(taken, span=5)
        assert port is not None
        assert port != taken
        assert taken < port <= taken + 4


def test_badge_shows_verdict_probability_and_threshold() -> None:
    html = format_badge(_result())
    assert "Potentially Vulnerable" in html
    assert "72.0%" in html
    assert "33%" in html
    assert "#b91c1c" in html  # vulnerable styling
    safe = format_badge(_result(label=0, prediction="Low Vulnerability Likelihood",
                                vulnerability_probability=0.12, confidence=0.88))
    assert "Low Vulnerability Likelihood" in safe
    assert "#166534" in safe  # safe styling


def test_summary_reports_metrics_modes_and_truncation() -> None:
    text = format_summary(_result())
    assert "**Vulnerability probability:** 72.00%" in text
    assert "**Decision threshold:** 33.00%" in text
    assert "**Head:** mlp" in text
    assert "Smoke-test model" not in text
    smoke = format_summary(_result(quick_test_model=True, input_truncated=True,
                                   original_token_count=900))
    assert "Smoke-test model" in smoke
    assert "only the first 400 tokens" in smoke


def test_explanation_is_a_table_with_escaped_pipes() -> None:
    text = format_explanation(_result(
        influential_lines=[{"line_number": 2, "code": "if (a | b) {", "influence": -0.5,
                            "direction": "opposes"}],
        explained_tokens=7,
    ))
    assert text.startswith("| Line | Code | Influence | Effect |")
    assert "`if (a \\| b) {`" in text
    assert "| 2 |" in text
    assert "-0.5000" in text
    assert "Occlusion over **7** tokens." in text
    assert "Signed influence" in text


def test_explanation_without_lines_is_explicit() -> None:
    text = format_explanation(_result(influential_lines=[]))
    assert "No line-level influence" in text


def test_error_outputs_are_graceful_and_do_not_leak_details() -> None:
    badge, summary, explanation, raw = error_outputs("Paste a non-empty C function")
    assert "Analysis not completed" in badge
    assert "Paste a non-empty C function" in summary
    assert explanation == ""
    assert raw == ""


def test_idle_state_is_not_an_error_state() -> None:
    assert "Awaiting code" in IDLE_BADGE
    assert "#64748b" in IDLE_BADGE
    assert IDLE_SUMMARY.startswith("## Awaiting code")
    assert "Analysis not completed" not in IDLE_BADGE


def test_summary_and_explanation_never_raise_on_extremes() -> None:
    extreme = _result(vulnerability_probability=1.0, confidence=1.0, label=1)
    assert "100.00%" in format_summary(extreme)
    assert "100.0%" in format_badge(extreme)
    with pytest.raises(KeyError):
        format_summary({"prediction": "x"})
