"""Tests for the M4 timeout / retry / fail-fast / error-handling layer.

Covers the four CLI flags introduced in M4
(``--timeout-seconds``, ``--max-retries``, ``--continue-on-error``,
``--fail-fast``), the SIGALRM-based wall-clock guard, the no-retry
policy for deterministic failures, error-message redaction, and the
schema-uniformity guarantee that failure traces carry the same three
M3 hashes as success traces.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verifiable_labs_envs.cli import _redact_secrets

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "agents"


# ── _redact_secrets unit tests ─────────────────────────────────────────


def test_redact_secrets_strips_env_var_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-deadbeef-1234567890abcdef")
    msg = "Auth failed: sk-deadbeef-1234567890abcdef is invalid"
    redacted = _redact_secrets(msg)
    assert "sk-deadbeef-1234567890abcdef" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_inline_pattern() -> None:
    msg = "Failed to connect: API_KEY=mysecret123 rejected"
    redacted = _redact_secrets(msg)
    assert "mysecret123" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_secrets_inline_token() -> None:
    msg = "header: token=abc123xyz"
    redacted = _redact_secrets(msg)
    assert "abc123xyz" not in redacted


def test_redact_secrets_no_secrets_passthrough() -> None:
    msg = "ValueError: index 5 out of range"
    assert _redact_secrets(msg) == msg


def test_redact_secrets_empty() -> None:
    assert _redact_secrets("") == ""
    assert _redact_secrets(None) is None  # type: ignore[arg-type]


def test_redact_secrets_short_env_value_not_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values shorter than 8 chars are not replaced (false-positive risk)."""
    monkeypatch.setenv("MY_TOKEN", "abc")
    msg = "the abc value"
    assert _redact_secrets(msg) == msg


# ── CLI integration tests ──────────────────────────────────────────────


def _run_cli(args: list[str]) -> int:
    from verifiable_labs_envs.cli import main
    return main(args)


def _read_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sleep_agent_times_out_continue_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "sleep.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "sleep_agent.py"),
        "--n", "3",
        "--timeout-seconds", "2",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0, "continue-on-error is the default; run should complete"
    traces = _read_traces(out)
    assert len(traces) == 3
    for t in traces:
        assert t["failure_type"] == "timeout", t
        assert t["metadata"]["status"] == "failed"
        assert t["metadata"]["retries"] == 0  # default --max-retries


def test_raise_agent_continue_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "raise.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "raise_agent.py"),
        "--n", "3",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    traces = _read_traces(out)
    assert len(traces) == 3
    for t in traces:
        # raise_agent → caught as generic Exception → FailureType.UNKNOWN
        assert t["failure_type"] != "none"
        assert t["metadata"]["status"] == "failed"
        # error_message should mention ValueError but NOT contain a traceback.
        assert "ValueError" in t["metadata"]["error_message"]


def test_flaky_agent_with_max_retries_succeeds_with_retries_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flaky_agent fails 1st call, succeeds on retry → retries=1, parse_success=True."""
    monkeypatch.chdir(REPO_ROOT)
    state_file = tmp_path / "flaky_state.txt"
    monkeypatch.setenv("FLAKY_AGENT_STATE", str(state_file))

    out = tmp_path / "flaky.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "flaky_agent.py"),
        "--n", "1",
        "--max-retries", "2",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    traces = _read_traces(out)
    assert len(traces) == 1
    t = traces[0]
    assert t["parse_success"] is True
    assert t["failure_type"] == "none"
    assert t["metadata"]["status"] == "ok"
    assert t["metadata"]["retries"] == 1


def test_sleep_agent_with_fail_fast_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "sleep_ff.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "sleep_agent.py"),
        "--n", "3",
        "--timeout-seconds", "2",
        "--fail-fast",
        "--out", str(out),
        "--quiet",
    ])
    assert rc != 0, "fail-fast should produce a non-zero exit code"
    traces = _read_traces(out)
    # First episode timed out → file has exactly 1 trace.
    assert len(traces) == 1
    assert traces[0]["failure_type"] == "timeout"


def test_no_traceback_or_stack_trace_in_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stack traces NEVER end up in the JSONL — only on stderr."""
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "raise.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "raise_agent.py"),
        "--n", "2",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    text = out.read_text()
    assert "Traceback" not in text
    assert "traceback" not in text.lower()
    assert "File \"" not in text  # filename refs from stack trace formatting
    assert "  File " not in text


def test_failure_trace_has_three_hashes_and_status_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3 + M4 schema parity: failure traces carry config_hash, instance_hash,
    reward_hash="0.000000", status="failed"."""
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "raise.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "raise_agent.py"),
        "--n", "2",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    traces = _read_traces(out)
    assert len(traces) == 2
    cfgs = set()
    insts = set()
    for t in traces:
        m = t["metadata"]
        assert m["status"] == "failed"
        assert m["config_hash"].startswith("sha256:")
        assert m["instance_hash"].startswith("sha256:")
        assert m["reward_hash"] == "0.000000"
        cfgs.add(m["config_hash"])
        insts.add(m["instance_hash"])
    assert len(cfgs) == 1, "config_hash should be identical across the run"
    assert len(insts) == 2, "instance_hashes should differ per seed even on failure"


def test_parse_fail_agent_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parse_error is deterministic → no retry, even with --max-retries 5."""
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "pf.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", str(FIXTURES / "parse_fail_agent.py"),
        "--n", "1",
        "--max-retries", "5",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    traces = _read_traces(out)
    assert len(traces) == 1
    t = traces[0]
    # The parse_fail agent returns a dict, so it gets past the
    # invalid_shape gate and into _score_episode → PARSE_ERROR.
    assert t["failure_type"] in ("parse_error", "invalid_json", "invalid_shape")
    assert t["metadata"]["retries"] == 0  # despite max-retries=5


def test_no_timeout_when_seconds_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--timeout-seconds 0 disables the alarm; fast agents complete normally."""
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "zero_ok.jsonl"
    rc = _run_cli([
        "run",
        "--env", "sparse-fourier-recovery",
        "--agent", "examples/agents/zero_agent.py",
        "--n", "2",
        "--timeout-seconds", "0",
        "--out", str(out),
        "--quiet",
    ])
    assert rc == 0
    traces = _read_traces(out)
    assert len(traces) == 2
    for t in traces:
        assert t["parse_success"] is True
        assert t["metadata"]["status"] == "ok"


# ── argparse / --help inspection ───────────────────────────────────────


def test_help_shows_all_four_new_flags() -> None:
    """`verifiable run --help` text mentions all four M4 flags."""
    res = subprocess.run(
        [sys.executable, "-m", "verifiable_labs_envs.cli", "run", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    h = res.stdout
    assert "--timeout-seconds" in h
    assert "--max-retries" in h
    assert "--continue-on-error" in h
    assert "--fail-fast" in h


def test_help_flags_are_mutually_exclusive() -> None:
    """--continue-on-error and --fail-fast are mutually exclusive."""
    res = subprocess.run(
        [
            sys.executable, "-m", "verifiable_labs_envs.cli", "run",
            "--env", "sparse-fourier-recovery",
            "--agent", "examples/agents/zero_agent.py",
            "--n", "1",
            "--out", "/tmp/_test_mutex.jsonl",
            "--continue-on-error",
            "--fail-fast",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "not allowed with argument" in res.stderr or "mutually exclusive" in res.stderr.lower()
