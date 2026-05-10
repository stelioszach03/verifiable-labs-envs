"""Tests for ``vlabs_prm_eval.cli``."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vlabs_prm_eval import __version__
from vlabs_prm_eval.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    for cmd in ("eval-processbench", "bon-rerank", "calibration", "card", "version"):
        assert cmd in out


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_eval_processbench_smoke() -> None:
    result = runner.invoke(app, ["eval-processbench", "--n", "10"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["n_traces"] == 10
    assert "overall_accuracy" in payload


def test_bon_rerank_smoke() -> None:
    result = runner.invoke(app, ["bon-rerank", "--n", "5", "--n-per", "3"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["n_problems"] == 5
    assert "prm_bon_accuracy" in payload


def test_calibration_command(tmp_path: Path) -> None:
    """Hand-build a calibration JSONL and run the calibration command."""
    src = tmp_path / "calib.jsonl"
    lines = []
    for i in range(20):
        payload = {
            "row_id": f"prw_{i:016x}",
            "env_id": "math-algebra",
            "prompt": f"p-{i}",
            "steps": ["a", "b"],
            "step_rewards": [0.5, 0.5],
            "step_components": [None, None],
            "step_conformal_intervals": [None, None],
            "step_frontier_judgments": [None, None],
            "step_frontier_rationales": [None, None],
            "step_consensus_rewards": [0.5, 0.5],
            "step_disagreements": [None, None],
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
        app, ["calibration", "--calib-set", str(src), "--target-alpha", "0.10"]
    )
    assert result.exit_code == 0, result.stdout
    body = json.loads(result.stdout)
    assert body["alpha"] == 0.10
    assert body["n_traces"] == 20
    assert "is_calibration_suspect" in body


def test_calibration_rejects_empty(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["calibration", "--calib-set", str(src)])
    assert result.exit_code != 0


def test_card_smoke(tmp_path: Path) -> None:
    out = tmp_path / "card.json"
    result = runner.invoke(
        app,
        [
            "card",
            "--n-processbench", "5",
            "--n-bon-problems", "3",
            "--n-per-bon", "2",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "processbench" in payload
    assert "bon" in payload


def test_card_skips_calibration_without_calib_set() -> None:
    result = runner.invoke(
        app,
        [
            "card",
            "--n-processbench", "3",
            "--n-bon-problems", "2",
            "--n-per-bon", "2",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["calibration"] is None
