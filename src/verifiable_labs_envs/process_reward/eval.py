"""End-to-end PRM eval harness (Phase 30.D).

Per :doc:`PHASE_30_PLAN.md` §9 the harness exposes three eval surfaces
plus a top-level orchestrator that runs all three and produces the
checkpoint's eval card:

1. **ProcessBench eval** (:func:`evaluate_processbench`) — D6-A
   external benchmark, step-error detection accuracy ≥ 60 % pass
   criterion.
2. **BoN reranking eval** (:func:`evaluate_bon`) — D6-B internal,
   PRM lift over Phase 29 distilled RM baseline ≥ +5 pp pass
   criterion.
3. **Calibration quality** (:func:`evaluate_calibration`) — D9-C
   moat metric, aggregate empirical coverage within ±5 pp of target.

30.D ships the surface backed by
:func:`~verifiable_labs_envs.process_reward.inference.stub_full_predictor`.
30.G swaps the stub for the trained student.

D6-C RL training capability lift is **scaffolded only** — the actual
RL run is gated to 30.G when the trained PRM exists + a customer
policy training pipeline is set up.
"""
from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from verifiable_labs_envs.process_reward.bon_rerank import (
    BonCandidate,
    bon_lift_metrics,
    make_synthetic_bon_problems,
    passes_bon_lift_floor,
)
from verifiable_labs_envs.process_reward.calibration import (
    DEFAULT_ALPHA,
    DEFAULT_DRIFT_TOL,
    CalibrationResult,
    calibrate_residuals,
)
from verifiable_labs_envs.process_reward.dataset import (
    ProcessRewardTraceRow,
)
from verifiable_labs_envs.process_reward.inference import (
    stub_aggregate_predictor,
    stub_step_predictor,
)

logger = logging.getLogger(__name__)

DEFAULT_PROCESSBENCH_DATASET: str = "Qwen/ProcessBench"
DEFAULT_PROCESSBENCH_PASS_THRESHOLD: float = 0.60
"""Plan §5 D6: step-error detection ≥ 60 % overall."""

DEFAULT_BON_PASS_THRESHOLD: float = 0.05
"""Plan §5 D6: PRM BoN lift over Phase 29 RM baseline ≥ +5 pp."""


# ── ProcessBench ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ProcessBenchTrace:
    """One ProcessBench trace.

    ProcessBench-shaped data: a (problem, reasoning_trace) pair plus
    a per-step ground-truth label marking which step (if any) is the
    first error. ``first_error_step`` is the 0-based step index of the
    first incorrect step, or ``None`` for fully-correct traces.
    """

    problem: str
    steps: tuple[str, ...]
    first_error_step: int | None
    subset: str = "math"
    trace_id: str = ""


@dataclass(frozen=True)
class ProcessBenchReport:
    """Aggregate ProcessBench accuracy + per-subset breakdown."""

    n_traces: int
    overall_accuracy: float
    per_subset: dict[str, float] = field(default_factory=dict)
    per_subset_count: dict[str, int] = field(default_factory=dict)
    n_correct_traces: int = 0
    n_error_traces: int = 0

    def passes(self, *, threshold: float = DEFAULT_PROCESSBENCH_PASS_THRESHOLD) -> bool:
        return float(self.overall_accuracy) >= float(threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_traces": int(self.n_traces),
            "overall_accuracy": float(self.overall_accuracy),
            "per_subset": dict(self.per_subset),
            "per_subset_count": dict(self.per_subset_count),
            "n_correct_traces": int(self.n_correct_traces),
            "n_error_traces": int(self.n_error_traces),
            "passes": self.passes(),
        }


def evaluate_processbench(
    traces: Sequence[ProcessBenchTrace],
    step_predictor: Callable[[str, Sequence[str], int], float],
    *,
    error_threshold: float = 0.5,
) -> ProcessBenchReport:
    """For each trace, find the first step whose PRM-predicted reward
    falls below ``error_threshold`` (≈ "the model thinks this step is
    wrong"). Match against the ground-truth ``first_error_step``.

    Counting rules:

    - Fully-correct trace (``first_error_step is None``): correct iff
      the PRM predicts every step ≥ ``error_threshold``.
    - Error trace: correct iff the PRM's predicted-error step matches
      the ground-truth ``first_error_step`` exactly.

    Per-subset accuracy reported separately.
    """
    if not traces:
        return ProcessBenchReport(n_traces=0, overall_accuracy=0.0)

    correct = 0
    n_correct = 0
    n_error = 0
    by_subset_correct: dict[str, int] = {}
    by_subset_total: dict[str, int] = {}

    for trace in traces:
        predicted_error: int | None = None
        for i, _step in enumerate(trace.steps):
            pred = float(step_predictor(trace.problem, trace.steps, i))
            if pred < error_threshold:
                predicted_error = i
                break

        match = predicted_error == trace.first_error_step
        if match:
            correct += 1
            by_subset_correct[trace.subset] = (
                by_subset_correct.get(trace.subset, 0) + 1
            )
        by_subset_total[trace.subset] = by_subset_total.get(trace.subset, 0) + 1

        if trace.first_error_step is None:
            n_correct += 1
        else:
            n_error += 1

    overall = correct / len(traces)
    per_subset = {
        s: by_subset_correct.get(s, 0) / total
        for s, total in by_subset_total.items()
    }
    return ProcessBenchReport(
        n_traces=len(traces),
        overall_accuracy=float(overall),
        per_subset=per_subset,
        per_subset_count=by_subset_total,
        n_correct_traces=n_correct,
        n_error_traces=n_error,
    )


def build_synthetic_processbench(
    n_traces: int = 40, *, seed: int = 0
) -> list[ProcessBenchTrace]:
    """Deterministic offline ProcessBench stand-in.

    Generates ``n_traces`` traces split across subsets
    ``("math", "olympiadbench", "gsm8k")``; ~half are fully-correct,
    ~half have a randomly-placed first-error step. Used by CI when
    the real `Qwen/ProcessBench` HF dataset isn't reachable.
    """
    import numpy as np

    if n_traces < 0:
        raise ValueError(f"n_traces must be non-negative; got {n_traces}")

    rng = np.random.default_rng(seed)
    subsets = ("math", "olympiadbench", "gsm8k")
    traces: list[ProcessBenchTrace] = []
    for i in range(n_traces):
        n_steps = int(rng.integers(3, 8))
        is_correct = bool(rng.integers(0, 2))
        steps = tuple(
            f"Step {j + 1}: synthetic-step #{i}-{j}"
            for j in range(n_steps)
        )
        if is_correct:
            first_error: int | None = None
        else:
            first_error = int(rng.integers(0, n_steps))
        traces.append(
            ProcessBenchTrace(
                problem=f"Synthetic ProcessBench problem #{i:04d}",
                steps=steps,
                first_error_step=first_error,
                subset=subsets[i % len(subsets)],
                trace_id=f"pb-synth-{i:04d}",
            )
        )
    return traces


def load_processbench_subset(
    n: int,
    *,
    seed: int = 0,
    subset: str = "all",
    dataset_name: str = DEFAULT_PROCESSBENCH_DATASET,
    fallback_to_synthetic: bool = True,
) -> list[ProcessBenchTrace]:
    """Pull ``n`` rows from the HF ProcessBench dataset. Falls back to
    :func:`build_synthetic_processbench` when the dataset isn't
    reachable (mirrors the Phase 29 RewardBench adapter pattern)."""
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    if n == 0:
        return []

    try:
        datasets_mod = importlib.import_module("datasets")
    except (ImportError, AttributeError) as exc:
        if not fallback_to_synthetic:
            raise RuntimeError("datasets library unavailable") from exc
        logger.info("datasets unavailable; falling back to synthetic ProcessBench")
        return build_synthetic_processbench(n_traces=n, seed=seed)

    try:
        ds = datasets_mod.load_dataset(dataset_name, split="train")
    except Exception as exc:  # noqa: BLE001
        if not fallback_to_synthetic:
            raise RuntimeError(f"failed to load {dataset_name}: {exc}") from exc
        logger.info("ProcessBench load failed (%s); falling back to synthetic", exc)
        return build_synthetic_processbench(n_traces=n, seed=seed)

    try:
        total = len(ds)
    except Exception:  # noqa: BLE001
        if not fallback_to_synthetic:
            raise
        return build_synthetic_processbench(n_traces=n, seed=seed)
    if total == 0:
        return build_synthetic_processbench(n_traces=n, seed=seed)

    import numpy as np

    rng = np.random.default_rng(seed)
    take = min(n, total)
    indices = rng.choice(total, size=take, replace=False)
    indices.sort()
    traces: list[ProcessBenchTrace] = []
    for idx in indices:
        record = ds[int(idx)]
        if subset != "all" and record.get("subset") != subset:
            continue
        problem = str(record.get("problem", record.get("question", "")))
        steps_raw = record.get("steps", []) or []
        steps = tuple(str(s) for s in steps_raw)
        first_err = record.get("first_error_step")
        first_err_int = int(first_err) if first_err is not None else None
        if not problem or not steps:
            continue
        traces.append(
            ProcessBenchTrace(
                problem=problem,
                steps=steps,
                first_error_step=first_err_int,
                subset=str(record.get("subset", "math")),
                trace_id=str(record.get("id", f"pb-{idx}")),
            )
        )
    return traces


# ── BoN reranking eval ─────────────────────────────────────────────


def evaluate_bon(
    problems: Sequence[Sequence[BonCandidate]],
    *,
    aggregate_predictor: Callable[[str, Sequence[str]], float] | None = None,
    rm_predictor: Callable[[str, str], float] | None = None,
    correct_threshold: float = 0.5,
) -> dict[str, float]:
    """Wrapper around :func:`bon_lift_metrics` with the locked PRM
    pass criterion (D6-B). ``aggregate_predictor`` defaults to the
    stub predictor — tests pass a deterministic stub; 30.G swaps in
    the real student."""
    predict = aggregate_predictor or stub_aggregate_predictor()
    metrics = bon_lift_metrics(
        problems,
        prm_aggregate_predictor=predict,
        rm_predictor=rm_predictor,
        correct_threshold=correct_threshold,
    )
    metrics["passes_bon_lift_floor"] = passes_bon_lift_floor(
        metrics, floor=DEFAULT_BON_PASS_THRESHOLD
    )
    return metrics


# ── Calibration eval (delegates to calibration.calibrate_residuals) ─


def evaluate_calibration(
    rows: Sequence[ProcessRewardTraceRow],
    *,
    step_predictor: Callable[[str, Sequence[str], int], float] | None = None,
    aggregate_predictor: Callable[[str, Sequence[str]], float] | None = None,
    target_alpha: float = DEFAULT_ALPHA,
) -> CalibrationResult:
    """30.D D9-C calibration eval — delegates to
    :func:`verifiable_labs_envs.process_reward.calibration.calibrate_residuals`
    with the stub predictors as defaults."""
    if not rows:
        raise ValueError("calibration set is empty")
    step_pred = step_predictor or stub_step_predictor()
    agg_pred = aggregate_predictor or stub_aggregate_predictor()
    return calibrate_residuals(
        rows,
        step_pred,
        agg_pred,
        alpha=target_alpha,
    )


# ── Top-level orchestrator (eval card) ─────────────────────────────


@dataclass(frozen=True)
class PrmEvalCard:
    """Combined 30.D eval card persisted next to a trained PRM
    checkpoint."""

    processbench: ProcessBenchReport | None
    bon: dict[str, float] | None
    calibration: CalibrationResult | None
    notes: dict[str, Any] = field(default_factory=dict)

    def passes(
        self,
        *,
        processbench_threshold: float = DEFAULT_PROCESSBENCH_PASS_THRESHOLD,
        bon_floor: float = DEFAULT_BON_PASS_THRESHOLD,
        calib_drift_tol: float = DEFAULT_DRIFT_TOL,
    ) -> bool:
        ok = True
        if self.processbench is not None and not self.processbench.passes(
            threshold=processbench_threshold
        ):
            ok = False
        if self.bon is not None:
            lift = self.bon.get("prm_bon_lift_vs_rm")
            if lift is not None and float(lift) < bon_floor:
                ok = False
        if self.calibration is not None and self.calibration.is_calibration_suspect(
            drift_tol=calib_drift_tol
        ):
            ok = False
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "processbench": (
                self.processbench.to_dict() if self.processbench else None
            ),
            "bon": dict(self.bon) if self.bon else None,
            "calibration": (
                {
                    "per_step_quantiles": dict(
                        self.calibration.per_step_quantiles
                    ),
                    "aggregate_quantile": float(
                        self.calibration.aggregate_quantile
                    ),
                    "aggregate_target_coverage": float(
                        self.calibration.aggregate_target_coverage
                    ),
                    "aggregate_empirical_coverage": float(
                        self.calibration.aggregate_empirical_coverage
                    ),
                    "aggregate_drift": float(self.calibration.aggregate_drift),
                    "n_traces": int(self.calibration.n_traces),
                    "alpha": float(self.calibration.alpha),
                }
                if self.calibration
                else None
            ),
            "passes": self.passes(),
            "notes": dict(self.notes),
        }


def run_eval_card(
    *,
    step_predictor: Callable[[str, Sequence[str], int], float] | None = None,
    aggregate_predictor: Callable[[str, Sequence[str]], float] | None = None,
    rm_predictor: Callable[[str, str], float] | None = None,
    processbench_traces: Sequence[ProcessBenchTrace] | None = None,
    n_processbench: int = 40,
    bon_problems: Sequence[Sequence[BonCandidate]] | None = None,
    n_bon_problems: int = 10,
    n_per_bon: int = 4,
    calib_set: Sequence[ProcessRewardTraceRow] | None = None,
    target_alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    notes: dict[str, Any] | None = None,
) -> PrmEvalCard:
    """Run all three eval surfaces + return a combined eval card.

    All three surfaces are independent: pass empty / None to skip any
    one. The ``seed`` controls deterministic synthetic-fixture
    generation when ``processbench_traces=None`` /
    ``bon_problems=None``.
    """
    step_pred = step_predictor or stub_step_predictor()
    agg_pred = aggregate_predictor or stub_aggregate_predictor()

    pb_report: ProcessBenchReport | None = None
    if processbench_traces is not None or n_processbench > 0:
        pb_traces = (
            list(processbench_traces)
            if processbench_traces is not None
            else build_synthetic_processbench(n_traces=n_processbench, seed=seed)
        )
        pb_report = evaluate_processbench(pb_traces, step_pred)

    bon_metrics: dict[str, float] | None = None
    if bon_problems is not None or n_bon_problems > 0:
        problems = (
            list(bon_problems)
            if bon_problems is not None
            else make_synthetic_bon_problems(
                n_problems=n_bon_problems, n_per_problem=n_per_bon, seed=seed
            )
        )
        bon_metrics = evaluate_bon(
            problems,
            aggregate_predictor=agg_pred,
            rm_predictor=rm_predictor,
        )

    calibration_result: CalibrationResult | None = None
    if calib_set:
        calibration_result = evaluate_calibration(
            calib_set,
            step_predictor=step_pred,
            aggregate_predictor=agg_pred,
            target_alpha=target_alpha,
        )

    return PrmEvalCard(
        processbench=pb_report,
        bon=bon_metrics,
        calibration=calibration_result,
        notes=dict(notes or {}),
    )


__all__ = [
    "DEFAULT_BON_PASS_THRESHOLD",
    "DEFAULT_PROCESSBENCH_DATASET",
    "DEFAULT_PROCESSBENCH_PASS_THRESHOLD",
    "PrmEvalCard",
    "ProcessBenchReport",
    "ProcessBenchTrace",
    "build_synthetic_processbench",
    "evaluate_bon",
    "evaluate_calibration",
    "evaluate_processbench",
    "load_processbench_subset",
    "run_eval_card",
]
