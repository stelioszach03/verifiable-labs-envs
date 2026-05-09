"""End-to-end eval harness for the distilled reward model (Phase 29.D).

Per :doc:`PHASE_29_PLAN.md` §9 the harness exposes three eval surfaces
plus a top-level orchestrator that runs all three and produces the
checkpoint's eval card:

1. **Held-out env eval** (:func:`evaluate_held_out_envs`) — D7-A
   primary, Spearman ρ ≥ 0.70 pass criterion.
2. **RewardBench cross-check** (:func:`evaluate_rewardbench`) — D7-C
   external, ≥ 65 % accuracy pass criterion.
3. **Calibration quality** (:func:`evaluate_calibration`) — D10 moat
   metric, empirical coverage within ±5 pp of target.

29.D ships the surface backed by
:func:`~verifiable_labs_envs.reward_distillation.stub_inference.stub_predictor`.
29.G swaps the stub for the trained student.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_HELD_OUT_ENVS,
    RewardTrainingRow,
    collect_env_rows,
)
from verifiable_labs_envs.reward_distillation.eval_metrics import (
    bias as compute_bias,
)
from verifiable_labs_envs.reward_distillation.eval_metrics import (
    calibration_drift,
    empirical_coverage,
    mae,
    passes_calibration_drift,
    passes_rewardbench,
    passes_spearman_floor,
    spearman_rho,
)
from verifiable_labs_envs.reward_distillation.rewardbench_adapter import (
    PreferencePair,
    RewardBenchReport,
    build_synthetic_rewardbench,
    evaluate_rewardbench,
)
from verifiable_labs_envs.reward_distillation.stub_inference import stub_predictor

DEFAULT_N_PER_ENV: int = 5
DEFAULT_CALIB_ALPHA: float = 0.10
DEFAULT_CALIB_DRIFT_TOL: float = 0.05
DEFAULT_RB_PAIRS: int = 40

StudentPredictor = Callable[[str, str], float]


# ── held-out env eval ───────────────────────────────────────────────


@dataclass(frozen=True)
class HeldOutEnvReport:
    env_id: str
    n_rows: int
    spearman: float
    mae: float
    bias: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "n_rows": int(self.n_rows),
            "spearman": float(self.spearman),
            "mae": float(self.mae),
            "bias": float(self.bias),
        }


@dataclass(frozen=True)
class HeldOutEvalReport:
    per_env: tuple[HeldOutEnvReport, ...]
    spearman_avg: float
    mae_avg: float

    def passes(self, *, floor: float = 0.70) -> bool:
        return passes_spearman_floor((s.spearman for s in self.per_env), floor=floor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_env": [s.to_dict() for s in self.per_env],
            "spearman_avg": float(self.spearman_avg),
            "mae_avg": float(self.mae_avg),
            "passes": self.passes(),
        }


def evaluate_held_out_envs(
    student: StudentPredictor | None = None,
    *,
    env_ids: Sequence[str] = DEFAULT_HELD_OUT_ENVS,
    n_per_env: int = DEFAULT_N_PER_ENV,
    seed_start: int = 10_000,
    rows_by_env: dict[str, list[RewardTrainingRow]] | None = None,
) -> HeldOutEvalReport:
    """For each held-out env, score ``n_per_env`` fresh-seed completions
    and measure Spearman ρ + MAE vs the env's true reward.

    ``student`` defaults to :func:`stub_predictor` so the 29.D-shaped
    eval can land without trained weights. Tests pass a deterministic
    callable.
    """
    if not env_ids:
        raise ValueError("env_ids must be non-empty")
    if n_per_env <= 0:
        raise ValueError(f"n_per_env must be positive; got {n_per_env}")
    predict = student or stub_predictor()
    per_env: list[HeldOutEnvReport] = []
    for env_id in env_ids:
        rows = (
            list(rows_by_env.get(env_id, []))
            if rows_by_env is not None
            else collect_env_rows([env_id], n_per_env=n_per_env, seed_start=seed_start)
        )
        per_env.append(_score_held_out_env(env_id, rows, predict))
    if per_env:
        spearman_avg = float(np.mean([s.spearman for s in per_env]))
        mae_avg = float(np.mean([s.mae for s in per_env]))
    else:
        spearman_avg = 0.0
        mae_avg = 0.0
    return HeldOutEvalReport(
        per_env=tuple(per_env), spearman_avg=spearman_avg, mae_avg=mae_avg
    )


def _score_held_out_env(
    env_id: str,
    rows: Sequence[RewardTrainingRow],
    predict: StudentPredictor,
) -> HeldOutEnvReport:
    if not rows:
        return HeldOutEnvReport(env_id=env_id, n_rows=0, spearman=0.0, mae=0.0, bias=0.0)
    targets = [float(r.consensus_reward) for r in rows]
    predictions = [float(predict(r.prompt, r.completion)) for r in rows]
    rho = spearman_rho(targets, predictions).rho
    return HeldOutEnvReport(
        env_id=env_id,
        n_rows=len(rows),
        spearman=rho,
        mae=mae(targets, predictions),
        bias=compute_bias(predictions, targets),
    )


# ── calibration eval ────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationReport:
    quantile: float
    target_coverage: float
    empirical_coverage: float
    drift: float
    n_rows: int

    def passes(self, *, tol: float = DEFAULT_CALIB_DRIFT_TOL) -> bool:
        return passes_calibration_drift(self.drift, tol=tol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile": float(self.quantile),
            "target_coverage": float(self.target_coverage),
            "empirical_coverage": float(self.empirical_coverage),
            "drift": float(self.drift),
            "n_rows": int(self.n_rows),
            "passes": self.passes(),
        }


def evaluate_calibration(
    student: StudentPredictor,
    calib_set: Sequence[RewardTrainingRow],
    *,
    target_alpha: float = DEFAULT_CALIB_ALPHA,
) -> CalibrationReport:
    """Compute the conformal quantile + empirical coverage over the
    held-out calibration set."""
    if not calib_set:
        raise ValueError("calibration set is empty")
    if not 0.0 < target_alpha < 1.0:
        raise ValueError(f"target_alpha must be in (0, 1); got {target_alpha}")

    targets = np.asarray(
        [float(r.consensus_reward) for r in calib_set], dtype=np.float64
    )
    predictions = np.asarray(
        [float(student(r.prompt, r.completion)) for r in calib_set], dtype=np.float64
    )
    residuals = np.abs(predictions - targets)
    quantile = float(split_conformal_quantile(residuals, alpha=target_alpha))
    coverage = empirical_coverage(predictions, targets, quantile=quantile)
    target_coverage = 1.0 - target_alpha
    return CalibrationReport(
        quantile=quantile,
        target_coverage=target_coverage,
        empirical_coverage=float(coverage),
        drift=calibration_drift(coverage, target_coverage),
        n_rows=int(targets.size),
    )


# ── rewardbench passthrough ─────────────────────────────────────────


def evaluate_rewardbench_default(
    student: StudentPredictor | None = None,
    *,
    n_pairs: int = DEFAULT_RB_PAIRS,
    seed: int = 0,
    pairs: Sequence[PreferencePair] | None = None,
) -> RewardBenchReport:
    """Score the synthetic (default) or supplied RewardBench pairs."""
    predict = student or stub_predictor()
    use_pairs = list(pairs) if pairs is not None else build_synthetic_rewardbench(n=n_pairs, seed=seed)
    return evaluate_rewardbench(use_pairs, predict)


# ── top-level orchestrator (eval card) ──────────────────────────────


@dataclass(frozen=True)
class EvalCard:
    """Combined 29.D eval card persisted next to a trained checkpoint."""

    held_out: HeldOutEvalReport
    calibration: CalibrationReport | None
    rewardbench: RewardBenchReport | None
    notes: dict[str, Any] = field(default_factory=dict)

    def passes(
        self,
        *,
        spearman_floor: float = 0.70,
        rb_floor: float = 0.65,
        calib_tol: float = DEFAULT_CALIB_DRIFT_TOL,
    ) -> bool:
        ok = self.held_out.passes(floor=spearman_floor)
        if self.rewardbench is not None and not passes_rewardbench(
            self.rewardbench.overall_accuracy, floor=rb_floor
        ):
            ok = False
        if self.calibration is not None and not self.calibration.passes(tol=calib_tol):
            ok = False
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out": self.held_out.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "rewardbench": self.rewardbench.to_dict() if self.rewardbench else None,
            "passes": self.passes(),
            "notes": dict(self.notes),
        }


def run_eval_card(
    student: StudentPredictor | None = None,
    *,
    held_out_envs: Sequence[str] = DEFAULT_HELD_OUT_ENVS,
    n_per_env: int = DEFAULT_N_PER_ENV,
    seed_start: int = 10_000,
    rows_by_env: dict[str, list[RewardTrainingRow]] | None = None,
    calib_set: Sequence[RewardTrainingRow] | None = None,
    rb_pairs: Sequence[PreferencePair] | None = None,
    n_rb_pairs: int = DEFAULT_RB_PAIRS,
    rb_seed: int = 0,
    target_alpha: float = DEFAULT_CALIB_ALPHA,
    notes: dict[str, Any] | None = None,
) -> EvalCard:
    """Run all three eval surfaces + return a single combined report.

    All three surfaces are independent: ``calib_set=None`` skips
    calibration; ``rb_pairs=None`` and ``n_rb_pairs=0`` together skip
    RewardBench. Held-out env eval is mandatory (returns an empty card
    if you opt out of it via ``held_out_envs=()``).
    """
    predict = student or stub_predictor()
    held_out = evaluate_held_out_envs(
        predict,
        env_ids=held_out_envs,
        n_per_env=n_per_env,
        seed_start=seed_start,
        rows_by_env=rows_by_env,
    ) if held_out_envs else HeldOutEvalReport(per_env=(), spearman_avg=0.0, mae_avg=0.0)
    calibration = (
        evaluate_calibration(predict, calib_set, target_alpha=target_alpha)
        if calib_set
        else None
    )
    rb_report = (
        evaluate_rewardbench_default(
            predict, pairs=rb_pairs, n_pairs=n_rb_pairs, seed=rb_seed
        )
        if (rb_pairs is not None or n_rb_pairs > 0)
        else None
    )
    return EvalCard(
        held_out=held_out,
        calibration=calibration,
        rewardbench=rb_report,
        notes=dict(notes or {}),
    )


__all__ = [
    "DEFAULT_CALIB_ALPHA",
    "DEFAULT_CALIB_DRIFT_TOL",
    "DEFAULT_N_PER_ENV",
    "DEFAULT_RB_PAIRS",
    "CalibrationReport",
    "EvalCard",
    "HeldOutEnvReport",
    "HeldOutEvalReport",
    "evaluate_calibration",
    "evaluate_held_out_envs",
    "evaluate_rewardbench_default",
    "run_eval_card",
]
