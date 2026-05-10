"""Cost-gate + env-gate tests for the ``vlabs-prm-data judge-steps`` CLI.

Phase 30.B's per-step frontier-judge slice is the only PRM CLI path
that spends real OpenRouter dollars. PHASE_30_PLAN.md §19 mirrors the
Phase 29 contract:

    1. A hard ``--cost-cap`` USD ceiling that REFUSES past it.
    2. An explicit env-var gate
       (``VLABS_PHASE30_COLLECT_FRONTIER=1``).
    3. ``--force-stub`` for offline runs.
    4. Hard-fail (NOT silent stub fall-through) when the gate is
       enabled but the API key is missing.

These five tests pin those four behaviours plus the JSONL shape of
the merged output (cost cap, env-gate-off, env-gate-on-no-key,
--force-stub, JSONL shape).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vlabs_prm_data.cli import DEFAULT_COST_CAP_USD, app

runner = CliRunner()


def _write_input_traces(p: Path, traces: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t) + "\n")


def _make_borderline_trace(idx: int, n_steps: int = 4) -> dict[str, Any]:
    """One ProcessRewardTraceRow with all step rewards in the borderline
    band (0.3, 0.7) so the per-step judge actually has work to do."""
    return {
        "row_id": f"prm_test_{idx:04d}",
        "env_id": "math-algebra",
        "prompt": f"prompt-{idx}",
        "steps": [f"step-{idx}-{j}" for j in range(n_steps)],
        "step_rewards": [0.5 for _ in range(n_steps)],
        "step_components": [None for _ in range(n_steps)],
        "step_conformal_intervals": [None for _ in range(n_steps)],
        "step_frontier_judgments": [None for _ in range(n_steps)],
        "step_frontier_rationales": [None for _ in range(n_steps)],
        "step_consensus_rewards": [0.5 for _ in range(n_steps)],
        "aggregate_reward": 0.5,
        "metadata": {},
        "schema_version": "v0.1.0",
        "source": "env",
    }


# ── 1. cost cap REFUSES past the threshold ─────────────────────────


def test_judge_steps_aborts_when_estimate_exceeds_cap(tmp_path: Path) -> None:
    inp = tmp_path / "traces.jsonl"
    out = tmp_path / "judged.jsonl"
    _write_input_traces(inp, [_make_borderline_trace(i) for i in range(3)])
    result = runner.invoke(
        app,
        [
            "judge-steps",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-steps", "100",
            "--cost-cap", "0.0001",  # way under estimate
            "--force-stub",
        ],
    )
    assert result.exit_code == 2
    assert "ABORT" in result.stdout or "ABORT" in (result.stderr or "")


# ── 2. env-var gate — VLABS_PHASE30_COLLECT_FRONTIER ───────────────


def test_judge_steps_uses_stub_when_env_gate_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default behaviour — the gate is off, the CLI uses the stub
    even when a real key is in the environment."""
    monkeypatch.delenv("VLABS_PHASE30_COLLECT_FRONTIER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake-prm-12345678")

    inp = tmp_path / "traces.jsonl"
    out = tmp_path / "judged.jsonl"
    _write_input_traces(inp, [_make_borderline_trace(i) for i in range(2)])
    result = runner.invoke(
        app,
        [
            "judge-steps",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-steps", "20",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "stub" in result.stdout.lower()


# ── 3. env gate ON but no key → ABORT ──────────────────────────────


def test_judge_steps_aborts_when_env_gate_on_but_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 29-shaped invariant: explicit opt-in + missing key →
    hard abort. NOT silent stub fall-through."""
    monkeypatch.setenv("VLABS_PHASE30_COLLECT_FRONTIER", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    inp = tmp_path / "traces.jsonl"
    out = tmp_path / "judged.jsonl"
    _write_input_traces(inp, [_make_borderline_trace(i) for i in range(2)])
    result = runner.invoke(
        app,
        [
            "judge-steps",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-steps", "20",
        ],
    )
    assert result.exit_code == 2
    err = (result.stderr or "") + result.stdout
    assert "ABORT" in err
    assert "OPENROUTER_API_KEY" in err


# ── 4. --force-stub overrides everything ───────────────────────────


def test_judge_steps_force_stub_overrides_env_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLABS_PHASE30_COLLECT_FRONTIER", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake-prm-87654321")

    inp = tmp_path / "traces.jsonl"
    out = tmp_path / "judged.jsonl"
    _write_input_traces(inp, [_make_borderline_trace(i) for i in range(2)])
    result = runner.invoke(
        app,
        [
            "judge-steps",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-steps", "20",
            "--force-stub",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "stub" in result.stdout.lower()


# ── 5. successful stub run produces a JSONL with judgments ─────────


def test_judge_steps_stub_run_writes_jsonl_with_per_step_judgments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLABS_PHASE30_COLLECT_FRONTIER", raising=False)

    inp = tmp_path / "traces.jsonl"
    out = tmp_path / "judged.jsonl"
    _write_input_traces(inp, [_make_borderline_trace(i, n_steps=3) for i in range(3)])
    result = runner.invoke(
        app,
        [
            "judge-steps",
            "--input", str(inp),
            "--output", str(out),
            "--fraction", "1.0",
            "--max-steps", "20",
            "--force-stub",
        ],
    )
    assert result.exit_code == 0, result.stdout

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    judged_traces = [
        r for r in rows
        if any(j is not None for j in r.get("step_frontier_judgments", []))
    ]
    # All input traces have borderline steps → all should land judgments.
    assert len(judged_traces) >= 1


def test_judge_steps_default_cost_cap_constant_is_50() -> None:
    """Pin the default cost cap so a refactor can't silently raise it.

    Phase 30 sits at $50 (vs $30 in Phase 29) — per-step prompts are
    denser than outcome judgments, so the per-trace cost is higher.
    See vlabs_prm_data.cli + PHASE_30_PLAN.md §19.
    """
    assert DEFAULT_COST_CAP_USD == 50.0
