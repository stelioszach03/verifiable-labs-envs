"""Tests for ``vlabs_prm_data.cli``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vlabs_prm_data import __version__
from vlabs_prm_data.cli import DEFAULT_COST_CAP_USD, app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    for cmd in ("extract", "extend-from-rm", "judge-steps", "merge", "summary", "version"):
        assert cmd in out


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_extract_smoke(tmp_path: Path) -> None:
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
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    for row in parsed:
        assert row["env_id"] == "math-algebra"
        assert row["source"] == "env"
        # `step_count` is a @property, not a JSON field — use len(steps).
        assert len(row["steps"]) >= 1


def test_extract_writes_summary_to_stdout(tmp_path: Path) -> None:
    output = tmp_path / "rows.jsonl"
    result = runner.invoke(
        app,
        ["extract", "--envs", "math-algebra", "--n-per-env", "1", "--output", str(output)],
    )
    assert result.exit_code == 0
    assert "math-algebra" in result.stdout
    assert "schema_version" in result.stdout


def test_extract_rejects_empty_envs(tmp_path: Path) -> None:
    output = tmp_path / "rows.jsonl"
    result = runner.invoke(
        app,
        ["extract", "--envs", "  ,  ", "--n-per-env", "1", "--output", str(output)],
    )
    assert result.exit_code != 0


def test_extend_from_rm_round_trip(tmp_path: Path) -> None:
    """Hand-build a Phase 29 JSONL → extend → check shape."""
    src = tmp_path / "src.jsonl"
    out = tmp_path / "out.jsonl"
    payload = {
        "row_id": "rwd_x",
        "env_id": "math-algebra",
        "prompt": "p",
        "completion": "Step 1: A.\nStep 2: B.",
        "env_reward": 0.5,
        "env_components": {"parse_valid": 1.0},
        "conformal_interval": None,
        "frontier_judgment": None,
        "frontier_rationale": None,
        "consensus_reward": 0.5,
        "disagreement": None,
        "source": "env",
        "metadata": {"seed": 42},
    }
    src.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "extend-from-rm",
            "--input", str(src),
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert len(rows[0]["steps"]) == 2
    assert rows[0]["segmentation_strategy"] == "explicit_step_marker"


def test_merge_concatenates(tmp_path: Path) -> None:
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
        app, ["merge", "--inputs", f"{a},{b}", "--output", str(out)]
    )
    assert result.exit_code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2


def test_merge_rejects_empty_inputs(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = runner.invoke(app, ["merge", "--inputs", "", "--output", str(out)])
    assert result.exit_code != 0


def test_summary_prints_stats(tmp_path: Path) -> None:
    src = tmp_path / "rows.jsonl"
    runner.invoke(
        app,
        ["extract", "--envs", "math-algebra", "--n-per-env", "3", "--output", str(src)],
    )
    result = runner.invoke(app, ["summary", "--input", str(src)])
    assert result.exit_code == 0
    payload = _extract_json_block(result.stdout)
    assert payload["n_traces"] == 3
    assert payload["by_env"] == {"math-algebra": 3}


def test_judge_steps_uses_stub_when_gates_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLABS_PHASE30_COLLECT_FRONTIER", raising=False)
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
            "judge-steps",
            "--input", str(src),
            "--output", str(out),
            "--fraction", "1.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "stub" in result.stdout.lower()


def test_judge_steps_force_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLABS_PHASE30_COLLECT_FRONTIER", "1")
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
            "judge-steps",
            "--input", str(src),
            "--output", str(out),
            "--fraction", "1.0",
            "--force-stub",
        ],
    )
    assert result.exit_code == 0
    assert "stub" in result.stdout.lower()


def test_judge_steps_aborts_on_cost_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hand-build a borderline-rich JSONL so the cap math has bite."""
    monkeypatch.delenv("VLABS_PHASE30_COLLECT_FRONTIER", raising=False)
    src = tmp_path / "src.jsonl"
    out = tmp_path / "judged.jsonl"
    lines = []
    for i in range(20):
        payload = {
            "row_id": f"prw_{i:016x}",
            "env_id": "math-algebra",
            "prompt": f"p-{i}",
            "steps": ["a", "b", "c"],
            "step_rewards": [0.5, 0.5, 0.5],
            "step_components": [None, None, None],
            "step_conformal_intervals": [None, None, None],
            "step_frontier_judgments": [None, None, None],
            "step_frontier_rationales": [None, None, None],
            "step_consensus_rewards": [0.5, 0.5, 0.5],
            "step_disagreements": [None, None, None],
            "aggregate_reward": 0.5,
            "aggregate_conformal_interval": None,
            "decomposition": "text_progress",
            "segmentation_strategy": "explicit_step_marker",
            "segmentation_confidence": 0.95,
            "truncated": False,
            "source": "env",
            "metadata": {},
        }
        lines.append(json.dumps(payload, sort_keys=True))
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "judge-steps",
            "--input", str(src),
            "--output", str(out),
            "--fraction", "1.0",
            "--cost-cap", "0.0001",
        ],
    )
    assert result.exit_code == 2
    assert "ABORT" in (result.stdout + result.stderr)


def test_default_cost_cap_locked() -> None:
    """Plan §5 D8 / §19: cap = $50."""
    assert pytest.approx(50.0) == DEFAULT_COST_CAP_USD


def test_env_status_emits_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLABS_PHASE30_COLLECT_FRONTIER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = runner.invoke(app, ["env-status"])
    assert result.exit_code == 0
    payload = _extract_json_block(result.stdout)
    assert payload["frontier_gate_enabled"] is False


def _extract_json_block(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    return json.loads(stdout[start : end + 1])
