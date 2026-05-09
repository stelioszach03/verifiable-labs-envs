"""Tests for ``vlabs_reward_train.eval``."""
from __future__ import annotations

import numpy as np
import pytest
from verifiable_labs_envs.reward_distillation.dataset import RewardTrainingRow

from vlabs_reward_train.eval import (
    HeldOutEnvScore,
    HeldOutEvalReport,
    calibration_mse,
    evaluate_held_out_envs,
    report_to_dict,
    spearman_rho,
)


def _row(reward: float, idx: int, env_id: str = "math-algebra") -> RewardTrainingRow:
    return RewardTrainingRow(
        row_id=f"rwd_{idx:016x}",
        env_id=env_id,
        prompt=f"prompt-{idx}",
        completion=f"completion-{idx}",
        env_reward=reward,
        env_components=None,
        conformal_interval=None,
        frontier_judgment=None,
        frontier_rationale=None,
        consensus_reward=reward,
        disagreement=None,
        source="env",
        metadata={},
    )


def test_spearman_rho_perfect_correlation() -> None:
    x = np.arange(10, dtype=np.float64)
    assert spearman_rho(x, x) == pytest.approx(1.0)
    assert spearman_rho(x, -x) == pytest.approx(-1.0)


def test_spearman_rho_uncorrelated_returns_low_value() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    rho = spearman_rho(x, y)
    assert -0.3 < rho < 0.3


def test_spearman_rho_short_input_returns_zero() -> None:
    assert spearman_rho(np.array([1.0]), np.array([1.0])) == 0.0


def test_spearman_rho_constant_returns_zero() -> None:
    assert spearman_rho(np.zeros(5), np.arange(5, dtype=np.float64)) == 0.0


def test_spearman_rho_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman_rho(np.zeros(3), np.zeros(5))


def test_calibration_mse_zero_when_equal() -> None:
    a = np.array([0.1, 0.2, 0.3])
    assert calibration_mse(a, a) == pytest.approx(0.0)


def test_calibration_mse_positive_on_difference() -> None:
    a = np.array([0.1, 0.2])
    b = np.array([0.3, 0.4])
    # mean((0.2)^2, (0.2)^2) = 0.04
    assert calibration_mse(a, b) == pytest.approx(0.04)


def test_calibration_mse_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        calibration_mse(np.zeros(3), np.zeros(5))


def test_calibration_mse_empty_returns_zero() -> None:
    assert calibration_mse(np.array([]), np.array([])) == 0.0


def test_evaluate_held_out_envs_with_injected_rows() -> None:
    """Use rows_by_env to bypass the real env load (CI-fast)."""
    rows = {
        "math-algebra": [_row(0.5, i) for i in range(5)],
        "sql-single-turn": [_row(0.7, i, env_id="sql-single-turn") for i in range(5)],
    }

    def perfect(prompt: str, completion: str) -> float:
        del prompt, completion
        return 0.5

    report = evaluate_held_out_envs(
        perfect,
        env_ids=("math-algebra", "sql-single-turn"),
        n_per_env=5,
        rows_by_env=rows,
    )
    assert isinstance(report, HeldOutEvalReport)
    assert len(report.per_env) == 2
    # Constant predictor → Spearman undefined / 0; passes() should be False.
    assert report.passes(spearman_floor=0.5) is False


def test_evaluate_held_out_envs_handles_empty_env() -> None:
    report = evaluate_held_out_envs(
        lambda p, c: 0.5,
        env_ids=("math-algebra",),
        n_per_env=2,
        rows_by_env={"math-algebra": []},
    )
    assert report.per_env[0].n_rows == 0


def test_evaluate_held_out_envs_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="env_ids"):
        evaluate_held_out_envs(lambda p, c: 0.5, env_ids=())
    with pytest.raises(ValueError, match="positive"):
        evaluate_held_out_envs(
            lambda p, c: 0.5, env_ids=("math-algebra",), n_per_env=0
        )


def test_evaluate_held_out_envs_passes_when_correlated() -> None:
    """Increasing predictor on increasing targets → Spearman = 1.0."""
    rows = {
        f"env-{e}": [_row(i / 10.0, i, env_id=f"env-{e}") for i in range(10)]
        for e in ("a", "b")
    }

    def correlated(prompt: str, completion: str) -> float:
        # The synthetic prompts are "prompt-0", "prompt-1", ...
        idx = int(prompt.rsplit("-", 1)[-1])
        return idx / 10.0 + 0.001 * idx  # slight scaling — Spearman still 1

    report = evaluate_held_out_envs(
        correlated,
        env_ids=("env-a", "env-b"),
        n_per_env=10,
        rows_by_env=rows,
    )
    assert report.passes(spearman_floor=0.9)
    assert report.spearman_avg > 0.9


def test_held_out_score_dataclass_fields() -> None:
    score = HeldOutEnvScore(
        env_id="math-algebra", n_rows=10, spearman=0.8, mae=0.05, bias=0.01
    )
    assert score.env_id == "math-algebra"


def test_report_to_dict_serializable() -> None:
    report = HeldOutEvalReport(
        per_env=(
            HeldOutEnvScore(env_id="x", n_rows=5, spearman=0.8, mae=0.1, bias=0.0),
        ),
        spearman_avg=0.8,
        mae_avg=0.1,
    )
    d = report_to_dict(report)
    assert d["passes"] is True
    assert d["per_env"][0]["env_id"] == "x"
