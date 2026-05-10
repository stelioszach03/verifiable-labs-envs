"""Cost-gate + env-gate tests for the ``vlabs-reward-data judge`` subcommand.

Phase 29.B's frontier-judge slice is the only path in this CLI that
spends real OpenRouter dollars. PHASE_29_PLAN.md §5 D1-D requires:

    1. A hard ``--cost-cap`` USD ceiling that REFUSES to proceed when
       the estimated cost exceeds it.
    2. An explicit env-var gate (``VLABS_PHASE29_COLLECT_FRONTIER=1``)
       so a misconfigured CI run can't silently bill the maintainer.
    3. ``--force-stub`` for offline / smoke runs.
    4. Stub fall-through when the gate is enabled but the API key is
       missing — refuse to proceed rather than silently emit empty
       judgments.

These five tests pin those four behaviours plus the JSONL shape of
the merged output (cap behaviour can ship through OpenRouter, env-gate,
key-missing, force-stub, JSONL shape).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vlabs_reward_data.cli import DEFAULT_COST_CAP_USD, app

runner = CliRunner()


def _write_input_rows(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_borderline_input(n: int = 5) -> list[dict[str, Any]]:
    """Build an in-spec RewardTrainingRow input with rewards in the
    borderline band so the judge actually has work to do."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "row_id": f"rwd_test_{i:04d}",
                "env_id": "math-algebra",
                "seed": i,
                "prompt": f"prompt-{i}",
                "completion": f"completion-{i}",
                "env_reward": 0.5,  # borderline
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.5,
                "conformal_interval": [0.4, 0.6],
                "metadata": {},
                "schema_version": "v0.1.0",
                "source": "env",
            }
        )
    return rows


# ── 1. cost cap REFUSES past the threshold ─────────────────────────


def test_judge_aborts_when_estimate_exceeds_cap(tmp_path: Path) -> None:
    """Setting an absurdly low cap and asking for the full borderline
    slice forces the cost-estimator past the cap → exit 2."""
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _write_input_rows(inp, _make_borderline_input(n=5))
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-rows", "10",
            "--cost-cap", "0.0001",  # forces the abort
            "--force-stub",
        ],
    )
    assert result.exit_code == 2
    assert "ABORT" in result.stdout or "ABORT" in (result.stderr or "")


# ── 2. env-var gate — VLABS_PHASE29_COLLECT_FRONTIER ───────────────


def test_judge_uses_stub_when_env_gate_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``VLABS_PHASE29_COLLECT_FRONTIER`` is not set to ``1``, the
    CLI MUST fall through to the stub caller — even if a real
    ``OPENROUTER_API_KEY`` is present in the environment."""
    monkeypatch.delenv("VLABS_PHASE29_COLLECT_FRONTIER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake-test-key-12345678")

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _write_input_rows(inp, _make_borderline_input(n=3))
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-rows", "10",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # The CLI says it's using the stub when the gate is off.
    assert "stub" in result.stdout.lower()


# ── 3. env gate ON but no key → ABORT ──────────────────────────────


def test_judge_aborts_when_env_gate_on_but_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the user explicitly opted in via the env gate but the API
    key isn't actually available, the CLI must abort instead of
    silently falling through to the stub (which would produce
    misleading "frontier judgments" in production)."""
    monkeypatch.setenv("VLABS_PHASE29_COLLECT_FRONTIER", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _write_input_rows(inp, _make_borderline_input(n=3))
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-rows", "10",
        ],
    )
    assert result.exit_code == 2
    err = (result.stderr or "") + result.stdout
    assert "ABORT" in err
    assert "OPENROUTER_API_KEY" in err


# ── 4. --force-stub overrides everything ───────────────────────────


def test_judge_force_stub_overrides_env_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--force-stub`` short-circuits to the deterministic stub even
    when the env gate IS on and a key IS present — useful for
    reproducible smoke runs that would otherwise burn dollars."""
    monkeypatch.setenv("VLABS_PHASE29_COLLECT_FRONTIER", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake-test-key-87654321")

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _write_input_rows(inp, _make_borderline_input(n=3))
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-rows", "10",
            "--force-stub",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "stub" in result.stdout.lower()


# ── 5. successful run produces a JSONL with judgment fields ────────


def test_judge_stub_run_writes_jsonl_with_judgments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub run on borderline rows → output JSONL contains every input
    row, with the borderline subset carrying ``frontier_judgment`` /
    ``frontier_rationale`` populated."""
    monkeypatch.delenv("VLABS_PHASE29_COLLECT_FRONTIER", raising=False)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _write_input_rows(inp, _make_borderline_input(n=4))
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-rows", "10",
            "--force-stub",
        ],
    )
    assert result.exit_code == 0, result.stdout

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    judged = [r for r in rows if r.get("frontier_judgment") is not None]
    # All 4 inputs are borderline → all 4 should be judged in stub mode.
    assert len(judged) == 4


def test_judge_default_cost_cap_constant_is_30() -> None:
    """Pin the default cost cap so a refactor can't silently raise it."""
    assert DEFAULT_COST_CAP_USD == 30.0
