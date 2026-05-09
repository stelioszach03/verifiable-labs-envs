"""In-loop held-out env evaluation (Phase 29.C scaffold).

Per :doc:`PHASE_29_PLAN.md` §9 / §11 the trainer runs an eval pass at
every checkpoint:

- score 200 fresh-seed completions per held-out env (D7-A);
- compute Spearman ρ vs the env's true reward;
- compute calibration MSE on the conformal CI (D10).

This module lays down the **adapter shape** so the 29.F training step
just plugs in `student_predict`. Tests pass a deterministic stub
predictor and assert the metric maths.

The full 29.D eval harness (which adds RewardBench cross-check + the
external benchmark surface) lives at
`verifiable_labs_envs.reward_distillation.eval` and is built in 29.D;
this module is the *training-time* slim version.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_HELD_OUT_ENVS,
    RewardTrainingRow,
    collect_env_rows,
)


@dataclass(frozen=True)
class HeldOutEnvScore:
    """Spearman ρ + MAE for a single held-out env."""

    env_id: str
    n_rows: int
    spearman: float
    mae: float
    bias: float


@dataclass(frozen=True)
class HeldOutEvalReport:
    """Aggregate report across the held-out env list."""

    per_env: tuple[HeldOutEnvScore, ...]
    spearman_avg: float
    mae_avg: float

    def passes(self, *, spearman_floor: float = 0.70) -> bool:
        """Pass criterion: every env Spearman ρ ≥ ``spearman_floor``."""
        return all(s.spearman >= spearman_floor for s in self.per_env)


def evaluate_held_out_envs(
    student_predict: Callable[[str, str], float],
    *,
    env_ids: Sequence[str] = DEFAULT_HELD_OUT_ENVS,
    n_per_env: int = 5,
    seed_start: int = 10_000,
    rows_by_env: dict[str, list[RewardTrainingRow]] | None = None,
) -> HeldOutEvalReport:
    """Score the student against held-out env rows.

    Parameters:

    - ``student_predict``: ``(prompt, completion) -> reward in [0, 1]``.
    - ``env_ids``: defaults to D7-A held-out envs (long-context-synthesis,
      sql-multiturn, code-mini-repo).
    - ``n_per_env``: scoring budget per env.
    - ``seed_start``: base seed; tests use a different value to avoid
      overlap with training seeds.
    - ``rows_by_env``: optional injection point for tests so they don't
      need to spin up the real env every time.
    """
    if not env_ids:
        raise ValueError("env_ids must be non-empty")
    if n_per_env <= 0:
        raise ValueError(f"n_per_env must be positive; got {n_per_env}")

    per_env_scores: list[HeldOutEnvScore] = []
    for env_id in env_ids:
        if rows_by_env is not None:
            rows = list(rows_by_env.get(env_id, []))
        else:
            rows = collect_env_rows([env_id], n_per_env=n_per_env, seed_start=seed_start)
        if not rows:
            per_env_scores.append(
                HeldOutEnvScore(env_id=env_id, n_rows=0, spearman=0.0, mae=0.0, bias=0.0)
            )
            continue
        per_env_scores.append(_score_env(env_id, rows, student_predict))
    spearman_avg = float(np.mean([s.spearman for s in per_env_scores])) if per_env_scores else 0.0
    mae_avg = float(np.mean([s.mae for s in per_env_scores])) if per_env_scores else 0.0
    return HeldOutEvalReport(
        per_env=tuple(per_env_scores),
        spearman_avg=spearman_avg,
        mae_avg=mae_avg,
    )


def _score_env(
    env_id: str,
    rows: Sequence[RewardTrainingRow],
    student_predict: Callable[[str, str], float],
) -> HeldOutEnvScore:
    targets = np.asarray([float(r.consensus_reward) for r in rows], dtype=np.float64)
    predictions = np.asarray(
        [float(student_predict(r.prompt, r.completion)) for r in rows],
        dtype=np.float64,
    )
    return HeldOutEnvScore(
        env_id=env_id,
        n_rows=int(targets.size),
        spearman=spearman_rho(targets, predictions),
        mae=float(np.mean(np.abs(targets - predictions))),
        bias=float(np.mean(predictions - targets)),
    )


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-numpy Spearman rank correlation; no scipy dep.

    Returns ``0.0`` for any of:

    - input arrays shorter than 2,
    - constant inputs (zero variance after ranking).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"shape mismatch: x={x.shape}, y={y.shape}")
    if x.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank ranking; mirrors ``scipy.stats.rankdata(method="average")``."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def calibration_mse(
    predicted_widths: np.ndarray, empirical_widths: np.ndarray
) -> float:
    """Mean-squared error between the student's predicted CI widths and
    the empirical widths observed on a held-out set (D10-A moat metric).
    """
    predicted_widths = np.asarray(predicted_widths, dtype=np.float64).ravel()
    empirical_widths = np.asarray(empirical_widths, dtype=np.float64).ravel()
    if predicted_widths.shape != empirical_widths.shape:
        raise ValueError(
            f"shape mismatch: predicted={predicted_widths.shape}, "
            f"empirical={empirical_widths.shape}"
        )
    if predicted_widths.size == 0:
        return 0.0
    return float(np.mean((predicted_widths - empirical_widths) ** 2))


def report_to_dict(report: HeldOutEvalReport) -> dict[str, Any]:
    """JSON-serialisable view of the report; persisted next to the
    checkpoint manifest in 29.F."""
    return {
        "per_env": [
            {
                "env_id": s.env_id,
                "n_rows": s.n_rows,
                "spearman": s.spearman,
                "mae": s.mae,
                "bias": s.bias,
            }
            for s in report.per_env
        ],
        "spearman_avg": report.spearman_avg,
        "mae_avg": report.mae_avg,
        "passes": report.passes(),
    }


__all__ = [
    "HeldOutEnvScore",
    "HeldOutEvalReport",
    "calibration_mse",
    "evaluate_held_out_envs",
    "report_to_dict",
    "spearman_rho",
]
