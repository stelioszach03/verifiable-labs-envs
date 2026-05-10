"""Tests for ``verifiable_labs_envs.process_reward.eval``."""
from __future__ import annotations

import sys
import types

import pytest

from verifiable_labs_envs.process_reward.bon_rerank import (
    make_synthetic_bon_problems,
)
from verifiable_labs_envs.process_reward.dataset import (
    SCHEMA_VERSION,
    ProcessRewardTraceRow,
)
from verifiable_labs_envs.process_reward.eval import (
    DEFAULT_BON_PASS_THRESHOLD,
    DEFAULT_PROCESSBENCH_PASS_THRESHOLD,
    PrmEvalCard,
    ProcessBenchReport,
    ProcessBenchTrace,
    build_synthetic_processbench,
    evaluate_bon,
    evaluate_calibration,
    evaluate_processbench,
    load_processbench_subset,
    run_eval_card,
)


def _trace(step_rewards: tuple[float, ...]) -> ProcessRewardTraceRow:
    n = len(step_rewards)
    return ProcessRewardTraceRow(
        row_id="prw_x",
        env_id="math-algebra",
        prompt="p",
        steps=tuple(f"step-{i}" for i in range(n)),
        step_rewards=step_rewards,
        step_components=tuple(None for _ in range(n)),
        step_conformal_intervals=tuple(None for _ in range(n)),
        step_frontier_judgments=tuple(None for _ in range(n)),
        step_frontier_rationales=tuple(None for _ in range(n)),
        step_consensus_rewards=tuple(step_rewards),
        step_disagreements=tuple(None for _ in range(n)),
        aggregate_reward=sum(step_rewards) / max(1, n),
        aggregate_conformal_interval=None,
        decomposition="text_progress",
        segmentation_strategy="explicit_step_marker",
        segmentation_confidence=0.95,
        truncated=False,
        source="env",
        metadata={"schema_version": SCHEMA_VERSION},
    )


# ── locked thresholds ───────────────────────────────────────────────


def test_default_thresholds_locked() -> None:
    """Plan §5 D6: 60% ProcessBench, +5pp BoN."""
    assert pytest.approx(0.60) == DEFAULT_PROCESSBENCH_PASS_THRESHOLD
    assert pytest.approx(0.05) == DEFAULT_BON_PASS_THRESHOLD


# ── ProcessBench ───────────────────────────────────────────────────


def test_build_synthetic_processbench_basic() -> None:
    traces = build_synthetic_processbench(n_traces=20, seed=0)
    assert len(traces) == 20
    for t in traces:
        assert isinstance(t, ProcessBenchTrace)
        assert len(t.steps) >= 3
        assert t.subset in ("math", "olympiadbench", "gsm8k")


def test_build_synthetic_processbench_deterministic() -> None:
    a = build_synthetic_processbench(n_traces=10, seed=42)
    b = build_synthetic_processbench(n_traces=10, seed=42)
    assert [t.trace_id for t in a] == [t.trace_id for t in b]


def test_build_synthetic_processbench_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_synthetic_processbench(n_traces=-1)


def test_evaluate_processbench_correct_trace() -> None:
    """A predictor that always returns 1.0 → all-correct verdict on
    fully-correct traces."""
    traces = [ProcessBenchTrace(problem="p", steps=("a", "b"), first_error_step=None)]
    report = evaluate_processbench(traces, lambda p, s, t: 1.0)
    assert report.overall_accuracy == pytest.approx(1.0)
    assert report.n_correct_traces == 1


def test_evaluate_processbench_detects_first_error() -> None:
    traces = [
        ProcessBenchTrace(problem="p", steps=("a", "b", "c"), first_error_step=1)
    ]

    def predictor(prompt: str, steps, step_index: int) -> float:
        return 1.0 if step_index < 1 else 0.0  # error at step 1

    report = evaluate_processbench(traces, predictor)
    assert report.overall_accuracy == pytest.approx(1.0)


def test_evaluate_processbench_misdetect() -> None:
    traces = [
        ProcessBenchTrace(problem="p", steps=("a", "b"), first_error_step=1)
    ]
    # Predict no error → mismatch.
    report = evaluate_processbench(traces, lambda p, s, t: 1.0)
    assert report.overall_accuracy == pytest.approx(0.0)


def test_evaluate_processbench_per_subset() -> None:
    traces = [
        ProcessBenchTrace(
            problem="p1", steps=("a",), first_error_step=None, subset="math"
        ),
        ProcessBenchTrace(
            problem="p2", steps=("a",), first_error_step=None, subset="gsm8k"
        ),
    ]
    report = evaluate_processbench(traces, lambda p, s, t: 1.0)
    assert "math" in report.per_subset
    assert "gsm8k" in report.per_subset


def test_evaluate_processbench_empty() -> None:
    report = evaluate_processbench([], lambda p, s, t: 1.0)
    assert report.n_traces == 0


def test_processbench_report_passes_at_threshold() -> None:
    report = ProcessBenchReport(n_traces=10, overall_accuracy=0.65)
    assert report.passes(threshold=0.60) is True
    assert report.passes(threshold=0.70) is False


def test_processbench_report_to_dict_serialisable() -> None:
    report = ProcessBenchReport(
        n_traces=5,
        overall_accuracy=0.6,
        per_subset={"math": 0.8},
        per_subset_count={"math": 5},
    )
    d = report.to_dict()
    assert d["n_traces"] == 5
    assert d["per_subset"]["math"] == pytest.approx(0.8)


def test_load_processbench_subset_falls_back_to_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("forced unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    traces = load_processbench_subset(n=5, seed=0)
    assert len(traces) == 5


def test_load_processbench_subset_zero_returns_empty() -> None:
    assert load_processbench_subset(n=0) == []


def test_load_processbench_subset_falls_back_when_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("datasets")

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    fake_module.load_dataset = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    traces = load_processbench_subset(n=3, seed=0)
    assert len(traces) == 3


# ── BoN eval ───────────────────────────────────────────────────────


def test_evaluate_bon_basic() -> None:
    problems = make_synthetic_bon_problems(n_problems=5, n_per_problem=3, seed=0)
    metrics = evaluate_bon(problems, aggregate_predictor=lambda p, s: 0.5)
    assert "prm_bon_accuracy" in metrics
    assert "passes_bon_lift_floor" in metrics


def test_evaluate_bon_with_rm_predictor() -> None:
    problems = make_synthetic_bon_problems(n_problems=3, n_per_problem=2, seed=0)
    metrics = evaluate_bon(
        problems,
        aggregate_predictor=lambda p, s: 0.5,
        rm_predictor=lambda p, c: 0.5,
    )
    assert "rm_bon_accuracy" in metrics


def test_evaluate_bon_uses_stub_when_predictor_none() -> None:
    problems = make_synthetic_bon_problems(n_problems=3, n_per_problem=2, seed=0)
    metrics = evaluate_bon(problems, aggregate_predictor=None)
    assert "n_problems" in metrics


# ── calibration eval ───────────────────────────────────────────────


def test_evaluate_calibration_perfect_predictor() -> None:
    rows = [_trace((0.5, 0.7)) for _ in range(20)]

    def perfect_step(p, s, t):
        return [0.5, 0.7][t]

    def perfect_agg(p, s):
        return 0.6

    result = evaluate_calibration(
        rows,
        step_predictor=perfect_step,
        aggregate_predictor=perfect_agg,
        target_alpha=0.10,
    )
    assert result.aggregate_quantile == pytest.approx(0.0)
    assert result.aggregate_empirical_coverage == pytest.approx(1.0)


def test_evaluate_calibration_uses_stub_predictors_by_default() -> None:
    rows = [_trace((0.5, 0.5)) for _ in range(10)]
    result = evaluate_calibration(rows)
    assert result.n_traces == 10


def test_evaluate_calibration_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_calibration([])


# ── eval card ──────────────────────────────────────────────────────


def test_run_eval_card_full_pipeline() -> None:
    rows = [_trace((0.5, 0.6)) for _ in range(10)]
    card = run_eval_card(
        n_processbench=8,
        n_bon_problems=3,
        n_per_bon=2,
        calib_set=rows,
        seed=0,
    )
    assert card.processbench is not None
    assert card.bon is not None
    assert card.calibration is not None


def test_run_eval_card_skips_when_zero() -> None:
    card = run_eval_card(
        n_processbench=0, n_bon_problems=0, calib_set=None, seed=0
    )
    assert card.processbench is None
    assert card.bon is None
    assert card.calibration is None


def test_run_eval_card_to_dict_round_trippable() -> None:
    rows = [_trace((0.5,)) for _ in range(5)]
    card = run_eval_card(
        n_processbench=4,
        n_bon_problems=2,
        n_per_bon=2,
        calib_set=rows,
    )
    payload = card.to_dict()
    assert "processbench" in payload
    assert "bon" in payload
    assert "calibration" in payload
    assert "passes" in payload


def test_eval_card_passes_when_all_three_pass() -> None:
    """Hand-build a card with passing values."""
    pb = ProcessBenchReport(n_traces=10, overall_accuracy=0.65)
    card = PrmEvalCard(processbench=pb, bon=None, calibration=None)
    assert card.passes() is True


def test_eval_card_fails_when_processbench_below_threshold() -> None:
    pb = ProcessBenchReport(n_traces=10, overall_accuracy=0.30)
    card = PrmEvalCard(processbench=pb, bon=None, calibration=None)
    assert card.passes() is False


def test_eval_card_fails_when_bon_below_threshold() -> None:
    bon = {"prm_bon_lift_vs_rm": 0.01}
    card = PrmEvalCard(processbench=None, bon=bon, calibration=None)
    assert card.passes() is False


def test_eval_card_notes_passthrough() -> None:
    card = run_eval_card(
        n_processbench=0,
        n_bon_problems=0,
        calib_set=None,
        notes={"experiment": "smoke", "dataset_id": "v0.0.1"},
    )
    assert card.notes["experiment"] == "smoke"
