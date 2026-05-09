"""Unit tests for ``vlabs_reward_data.cli`` — Typer command wiring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vlabs_reward_data import __version__
from vlabs_reward_data.cli import DEFAULT_COST_CAP_USD, app

runner = CliRunner()


# ── help + version ──────────────────────────────────────────────────


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    for cmd in ("extract", "extract-external", "judge", "merge", "summary", "version"):
        assert cmd in out


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


# ── extract command ─────────────────────────────────────────────────


def test_extract_smoke(tmp_path: Path) -> None:
    """Single-env extraction at small N — smoke check the CLI plumbing."""
    output = tmp_path / "rows.jsonl"
    result = runner.invoke(
        app,
        [
            "extract",
            "--envs", "math-algebra",
            "--n-per-env", "2",
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    for row in parsed:
        assert row["env_id"] == "math-algebra"
        assert row["source"] == "env"


def test_extract_writes_summary_to_stdout(tmp_path: Path) -> None:
    output = tmp_path / "rows.jsonl"
    result = runner.invoke(
        app,
        ["extract", "--envs", "math-algebra", "--n-per-env", "1", "--output", str(output)],
    )
    assert "math-algebra" in result.stdout
    assert "schema_version" in result.stdout


def test_extract_rejects_empty_envs_list(tmp_path: Path) -> None:
    output = tmp_path / "rows.jsonl"
    result = runner.invoke(
        app,
        ["extract", "--envs", "  ,  ", "--n-per-env", "1", "--output", str(output)],
    )
    assert result.exit_code != 0


# ── extract-external ────────────────────────────────────────────────


def test_extract_external_synthetic_path(tmp_path: Path) -> None:
    """Falls back to synthetic when offline; uses real HF data when
    available. Either path is a valid 29.B harness response."""
    output = tmp_path / "external.jsonl"
    result = runner.invoke(
        app,
        ["extract-external", "--n", "5", "--seed", "0", "--output", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 5
    for row in rows:
        assert row["source"] == "external"
        assert row["frontier_judgment"] is not None
        assert 0.0 <= row["frontier_judgment"] <= 1.0
        # Whether the row is synthetic or real, the metadata flag must
        # be present so downstream pipelines can filter cleanly.
        assert "synthetic" in row["metadata"]


# ── merge command ───────────────────────────────────────────────────


def test_merge_concatenates_shards(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    out = tmp_path / "merged.jsonl"
    runner.invoke(
        app, ["extract", "--envs", "math-algebra", "--n-per-env", "1", "--output", str(a)]
    )
    runner.invoke(
        app, ["extract", "--envs", "math-algebra", "--n-per-env", "1", "--output", str(b)]
    )
    result = runner.invoke(
        app,
        [
            "merge",
            "--inputs", f"{a},{b}",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    merged_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(merged_rows) == 2


def test_merge_rejects_empty_inputs(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app,
        ["merge", "--inputs", "", "--output", str(out)],
    )
    assert result.exit_code != 0


# ── summary command ─────────────────────────────────────────────────


def test_summary_prints_aggregate_stats(tmp_path: Path) -> None:
    src = tmp_path / "rows.jsonl"
    runner.invoke(
        app,
        ["extract", "--envs", "math-algebra", "--n-per-env", "3", "--output", str(src)],
    )
    result = runner.invoke(app, ["summary", "--input", str(src)])
    assert result.exit_code == 0, result.stdout
    payload = _extract_json_block(result.stdout)
    assert payload["n_rows"] == 3
    assert payload["by_env"] == {"math-algebra": 3}
    assert payload["schema_version"]


# ── judge command (stub path) ───────────────────────────────────────


def test_judge_uses_stub_when_gates_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLABS_PHASE29_COLLECT_FRONTIER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    src = tmp_path / "src.jsonl"
    out = tmp_path / "judged.jsonl"
    runner.invoke(
        app,
        ["extract", "--envs", "math-algebra", "--n-per-env", "2", "--output", str(src)],
    )
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(src),
            "--output", str(out),
            "--fraction", "1.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Stub mode is announced in stdout.
    assert "stub" in result.stdout.lower()
    assert out.exists()


def test_judge_force_stub_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--force-stub bypasses the gate even when both env vars are set."""
    monkeypatch.setenv("VLABS_PHASE29_COLLECT_FRONTIER", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    src = tmp_path / "src.jsonl"
    out = tmp_path / "judged.jsonl"
    runner.invoke(
        app,
        ["extract", "--envs", "math-algebra", "--n-per-env", "2", "--output", str(src)],
    )
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(src),
            "--output", str(out),
            "--fraction", "1.0",
            "--force-stub",
        ],
    )
    assert result.exit_code == 0
    assert "stub" in result.stdout.lower()


def test_judge_aborts_on_cost_cap_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the borderline-row budget exceeds the cap, the CLI aborts."""
    monkeypatch.delenv("VLABS_PHASE29_COLLECT_FRONTIER", raising=False)
    src = tmp_path / "src.jsonl"
    out = tmp_path / "judged.jsonl"
    # Hand-build a borderline-row JSONL so the cap math has something to
    # bite on (the math-algebra baseline scores 0.0 which is NOT borderline).
    lines = []
    for i in range(40):
        lines.append(
            json.dumps(
                {
                    "row_id": f"rwd_{i:016x}",
                    "env_id": "math-algebra",
                    "prompt": f"p-{i}",
                    "completion": f"c-{i}",
                    "env_reward": 0.5,
                    "env_components": None,
                    "conformal_interval": None,
                    "frontier_judgment": None,
                    "frontier_rationale": None,
                    "consensus_reward": 0.5,
                    "disagreement": None,
                    "source": "env",
                    "metadata": {},
                },
                sort_keys=True,
            )
        )
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "judge",
            "--input", str(src),
            "--output", str(out),
            "--fraction", "1.0",
            "--cost-cap", "0.0001",  # tighter than 1 row × $0.005
        ],
    )
    assert result.exit_code == 2, result.stdout
    assert "ABORT" in result.stdout or "ABORT" in result.stderr


def test_default_cost_cap_locked_at_30() -> None:
    """Plan §5 D1-D fixes the cap at $30; regression guard."""
    assert pytest.approx(30.0) == DEFAULT_COST_CAP_USD


# ── env-status helper command ───────────────────────────────────────


def test_env_status_emits_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLABS_PHASE29_COLLECT_FRONTIER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = runner.invoke(app, ["env-status"])
    assert result.exit_code == 0
    payload = _extract_json_block(result.stdout)
    assert payload["frontier_gate_enabled"] is False
    assert payload["OPENROUTER_API_KEY_present"] is False


# ── helpers ─────────────────────────────────────────────────────────


def _extract_json_block(stdout: str) -> dict:
    """Pull the first {...} JSON block out of the CLI stdout."""
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < 0:
        raise AssertionError(f"no JSON block in stdout: {stdout!r}")
    return json.loads(stdout[start : end + 1])
