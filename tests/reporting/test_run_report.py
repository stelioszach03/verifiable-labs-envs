"""Tests for the run-driven Markdown report renderer."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.reporting import REQUIRED_SECTIONS, render_run_report
from verifiable_labs_envs.traces import FailureType, Trace


def _make_traces(n: int = 5, parse_fail: int = 0) -> list[Trace]:
    traces: list[Trace] = []
    for i in range(n):
        ok = i >= parse_fail
        traces.append(Trace.new(
            env_name="sparse-fourier-recovery",
            agent_name="test-agent",
            reward=0.4 + 0.05 * i if ok else 0.0,
            parse_success=ok,
            seed=i,
            reward_components=({"nmse": 0.5 + 0.01 * i, "support": 0.3, "conformal": 0.85} if ok else {}),
            classical_baseline_reward=0.55,
            gap_to_classical=(0.4 + 0.05 * i - 0.55) if ok else None,
            coverage=0.85 if ok else None,
            latency_ms=100.0 + i * 10,
            failure_type=FailureType.NONE if ok else FailureType.PARSE_ERROR,
        ))
    return traces


def test_render_writes_file_with_all_sections(tmp_path):
    traces = _make_traces()
    out = tmp_path / "report.md"
    written = render_run_report(traces, out)
    assert written == out
    text = out.read_text()
    for section in REQUIRED_SECTIONS:
        assert section in text


def test_render_handles_all_parse_fail(tmp_path):
    """When every episode parse-fails the renderer must not crash."""
    traces = _make_traces(n=3, parse_fail=3)
    out = tmp_path / "report.md"
    render_run_report(traces, out)
    text = out.read_text()
    assert "All episodes failed to parse" in text
    assert "parse-failure rate: **100.0 %**" in text


def test_render_recommendations_flag_high_parse_fail(tmp_path):
    traces = _make_traces(n=10, parse_fail=4)
    out = tmp_path / "report.md"
    render_run_report(traces, out)
    text = out.read_text()
    assert "High parse-fail rate" in text


def test_render_recommendations_clean_run(tmp_path):
    traces = _make_traces(n=5, parse_fail=0)
    # Make agent beat baseline so the gap-recommendation isn't triggered.
    for t in traces:
        t.classical_baseline_reward = 0.30
        t.gap_to_classical = t.reward - 0.30
    out = tmp_path / "report.md"
    render_run_report(traces, out)
    text = out.read_text()
    assert "No critical issues flagged" in text


def test_render_recommendations_under_baseline(tmp_path):
    """Agent meaningfully under-performing the classical baseline triggers the
    capability-gap recommendation."""
    traces = _make_traces(n=5)
    # Force a -0.10 gap on every episode.
    for t in traces:
        t.classical_baseline_reward = t.reward + 0.10
        t.gap_to_classical = -0.10
    out = tmp_path / "report.md"
    render_run_report(traces, out)
    text = out.read_text()
    assert "under-performs the classical baseline" in text


def test_render_creates_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "nested" / "report.md"
    render_run_report(_make_traces(), out)
    assert out.exists()


def test_render_rejects_empty_traces(tmp_path):
    with pytest.raises(ValueError, match="at least one trace"):
        render_run_report([], tmp_path / "x.md")


def test_render_includes_env_and_agent_in_title(tmp_path):
    traces = _make_traces()
    out = tmp_path / "report.md"
    render_run_report(traces, out)
    text = out.read_text()
    first_line = text.splitlines()[0]
    assert "sparse-fourier-recovery" in first_line
    assert "test-agent" in first_line


def test_render_includes_p95_latency(tmp_path):
    traces = _make_traces()
    out = tmp_path / "report.md"
    render_run_report(traces, out)
    text = out.read_text()
    assert "p95 latency" in text
