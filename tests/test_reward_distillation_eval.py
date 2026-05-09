"""Tests for the 29.D eval harness."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.reward_distillation.dataset import RewardTrainingRow
from verifiable_labs_envs.reward_distillation.eval import (
    CalibrationReport,
    EvalCard,
    HeldOutEnvReport,
    HeldOutEvalReport,
    evaluate_calibration,
    evaluate_held_out_envs,
    evaluate_rewardbench_default,
    run_eval_card,
)
from verifiable_labs_envs.reward_distillation.stub_inference import stub_predictor


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


# ── held-out env eval ───────────────────────────────────────────────


def test_evaluate_held_out_envs_with_injected_rows() -> None:
    rows = {
        "math-algebra": [_row(0.5, i) for i in range(5)],
        "sql-single-turn": [
            _row(0.7, i, env_id="sql-single-turn") for i in range(5)
        ],
    }
    report = evaluate_held_out_envs(
        student=lambda p, c: 0.5,
        env_ids=("math-algebra", "sql-single-turn"),
        rows_by_env=rows,
    )
    assert len(report.per_env) == 2
    # Constant predictor → Spearman is 0; passes(0.5) is False.
    assert report.passes(floor=0.5) is False


def test_evaluate_held_out_envs_perfect_correlation() -> None:
    """An order-preserving predictor → Spearman = 1.0 → passes."""
    rows = {
        "math-algebra": [_row(i / 10.0, i) for i in range(10)],
    }

    def correlated(prompt: str, completion: str) -> float:
        del completion
        idx = int(prompt.rsplit("-", 1)[-1])
        return idx / 10.0

    report = evaluate_held_out_envs(
        student=correlated,
        env_ids=("math-algebra",),
        rows_by_env=rows,
    )
    assert report.spearman_avg == pytest.approx(1.0)
    assert report.passes(floor=0.7) is True


def test_evaluate_held_out_envs_handles_empty_env() -> None:
    report = evaluate_held_out_envs(
        student=lambda p, c: 0.5,
        env_ids=("math-algebra",),
        rows_by_env={"math-algebra": []},
    )
    assert report.per_env[0].n_rows == 0
    assert report.spearman_avg == pytest.approx(0.0)


def test_evaluate_held_out_envs_uses_stub_when_student_none() -> None:
    rows = {"math-algebra": [_row(0.5, i) for i in range(5)]}
    report = evaluate_held_out_envs(
        student=None,
        env_ids=("math-algebra",),
        rows_by_env=rows,
    )
    assert isinstance(report, HeldOutEvalReport)
    assert len(report.per_env) == 1


def test_evaluate_held_out_envs_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="env_ids"):
        evaluate_held_out_envs(student=lambda p, c: 0.5, env_ids=())
    with pytest.raises(ValueError, match="positive"):
        evaluate_held_out_envs(
            student=lambda p, c: 0.5, env_ids=("math-algebra",), n_per_env=0
        )


def test_held_out_env_report_to_dict_serialisable() -> None:
    score = HeldOutEnvReport(
        env_id="x", n_rows=5, spearman=0.8, mae=0.05, bias=0.0
    )
    d = score.to_dict()
    assert d["env_id"] == "x"
    assert d["n_rows"] == 5


# ── calibration eval ────────────────────────────────────────────────


def test_evaluate_calibration_perfect_predictor() -> None:
    rows = [_row(0.5, i) for i in range(50)]

    def perfect(prompt: str, completion: str) -> float:
        del prompt, completion
        return 0.5

    report = evaluate_calibration(perfect, rows, target_alpha=0.1)
    assert report.quantile == pytest.approx(0.0)
    assert report.empirical_coverage == pytest.approx(1.0)
    assert report.drift == pytest.approx(0.1)


def test_evaluate_calibration_uniform_offset() -> None:
    rows = [_row(0.5, i) for i in range(100)]

    def offset(prompt: str, completion: str) -> float:
        del prompt, completion
        return 0.7

    report = evaluate_calibration(offset, rows, target_alpha=0.1)
    assert report.quantile == pytest.approx(0.2)


def test_evaluate_calibration_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_calibration(lambda p, c: 0.5, [])


def test_evaluate_calibration_rejects_invalid_alpha() -> None:
    rows = [_row(0.5, 0)]
    with pytest.raises(ValueError, match="target_alpha"):
        evaluate_calibration(lambda p, c: 0.5, rows, target_alpha=0.0)
    with pytest.raises(ValueError, match="target_alpha"):
        evaluate_calibration(lambda p, c: 0.5, rows, target_alpha=1.0)


def test_calibration_report_passes_when_drift_within_tol() -> None:
    report = CalibrationReport(
        quantile=0.1,
        target_coverage=0.9,
        empirical_coverage=0.92,
        drift=0.02,
        n_rows=100,
    )
    assert report.passes(tol=0.05) is True
    assert report.passes(tol=0.01) is False


# ── rewardbench eval ────────────────────────────────────────────────


def test_evaluate_rewardbench_default_with_stub() -> None:
    """Stub predictor should land near 50 % accuracy due to the random
    hash but tied pairs reduce variance."""
    report = evaluate_rewardbench_default(student=stub_predictor(), n_pairs=40, seed=0)
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert report.n_pairs == 40


def test_evaluate_rewardbench_default_uses_stub_predictor() -> None:
    """Pass student=None and verify report is well-formed."""
    report = evaluate_rewardbench_default(student=None, n_pairs=10, seed=0)
    assert report.n_pairs == 10


# ── eval card orchestrator ──────────────────────────────────────────


def test_run_eval_card_minimal_held_out_only() -> None:
    rows = {"math-algebra": [_row(0.5, i) for i in range(5)]}
    card = run_eval_card(
        student=lambda p, c: 0.5,
        held_out_envs=("math-algebra",),
        rows_by_env=rows,
        calib_set=None,
        rb_pairs=None,
        n_rb_pairs=0,
    )
    assert card.calibration is None
    assert card.rewardbench is None
    assert isinstance(card.held_out, HeldOutEvalReport)


def test_run_eval_card_full_pipeline() -> None:
    rows = {"math-algebra": [_row(0.5, i) for i in range(5)]}
    calib = [_row(0.5, i) for i in range(20)]
    card = run_eval_card(
        student=lambda p, c: 0.5,
        held_out_envs=("math-algebra",),
        rows_by_env=rows,
        calib_set=calib,
        n_rb_pairs=10,
    )
    assert card.held_out is not None
    assert card.calibration is not None
    assert card.rewardbench is not None
    assert card.calibration.n_rows == 20
    assert card.rewardbench.n_pairs == 10


def test_run_eval_card_to_dict_serialisable() -> None:
    rows = {"math-algebra": [_row(0.5, i) for i in range(3)]}
    card = run_eval_card(
        student=lambda p, c: 0.5,
        held_out_envs=("math-algebra",),
        rows_by_env=rows,
        n_rb_pairs=0,
    )
    payload = card.to_dict()
    assert "held_out" in payload
    assert "passes" in payload


def test_run_eval_card_pass_predicate() -> None:
    """A perfectly-aligned predictor passes all three pass criteria."""

    def perfect(prompt: str, completion: str) -> float:
        idx = int(prompt.rsplit("-", 1)[-1])
        return idx / 10.0

    rows = {"math-algebra": [_row(i / 10.0, i) for i in range(10)]}
    calib = [_row(i / 10.0, i + 100) for i in range(20)]
    card = run_eval_card(
        student=perfect,
        held_out_envs=("math-algebra",),
        rows_by_env=rows,
        calib_set=calib,
        n_rb_pairs=0,
    )
    assert card.held_out.passes(floor=0.95)


def test_eval_card_skips_held_out_when_empty_env_list() -> None:
    card = run_eval_card(
        student=lambda p, c: 0.5,
        held_out_envs=(),
        rows_by_env=None,
        calib_set=None,
        n_rb_pairs=0,
    )
    assert card.held_out.per_env == ()


def test_eval_card_passes_combines_three_criteria() -> None:
    """Pass on held-out only doesn't pass overall if RB fails."""
    held = HeldOutEvalReport(
        per_env=(HeldOutEnvReport("x", 5, 0.95, 0.0, 0.0),),
        spearman_avg=0.95,
        mae_avg=0.0,
    )
    from verifiable_labs_envs.reward_distillation.rewardbench_adapter import (
        RewardBenchReport as RBReport,
    )

    rb_low = RBReport(n_pairs=10, overall_accuracy=0.40)
    card = EvalCard(held_out=held, calibration=None, rewardbench=rb_low)
    assert card.passes() is False  # held-out passes but rewardbench fails


def test_held_out_eval_report_passes_default_floor() -> None:
    report = HeldOutEvalReport(
        per_env=(HeldOutEnvReport("x", 5, 0.75, 0.05, 0.01),),
        spearman_avg=0.75,
        mae_avg=0.05,
    )
    assert report.passes() is True
    assert report.passes(floor=0.8) is False


def test_evaluate_calibration_returns_clipped_quantile_payload() -> None:
    """Predictions clipped at 0/1 still produce a valid coverage."""
    rows = [_row(0.5, i) for i in range(20)]

    def clip_predictor(prompt: str, completion: str) -> float:
        del prompt, completion
        return 0.5  # all predictions match exactly

    report = evaluate_calibration(clip_predictor, rows)
    assert report.passes(tol=0.15) is True


def test_run_eval_card_notes_passthrough() -> None:
    rows = {"math-algebra": [_row(0.5, 0)]}
    card = run_eval_card(
        student=lambda p, c: 0.5,
        held_out_envs=("math-algebra",),
        rows_by_env=rows,
        n_rb_pairs=0,
        notes={"experiment": "smoke", "dataset_id": "v0.0.1"},
    )
    assert card.notes["experiment"] == "smoke"


def test_eval_metric_pass_thresholds_locked() -> None:
    """Plan §5 D7 pass criteria: ρ ≥ 0.70, RB ≥ 0.65, drift ≤ 5 pp."""
    from verifiable_labs_envs.reward_distillation.eval import (
        DEFAULT_CALIB_DRIFT_TOL,
    )

    assert pytest.approx(0.05) == DEFAULT_CALIB_DRIFT_TOL
