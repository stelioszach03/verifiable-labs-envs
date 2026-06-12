"""Tests for ``verifiable_labs_envs.process_reward.calibration``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.calibration import (
    DEFAULT_ALPHA,
    DEFAULT_DRIFT_TOL,
    DEFAULT_POSITION_BUCKETS,
    DEFAULT_TARGET_COVERAGE,
    CalibrationResult,
    PerStepBucketResult,
    bucket_label,
    calibrate_residuals,
    evaluate_per_step_coverage,
    position_to_bucket_label,
    score_with_ci,
)
from verifiable_labs_envs.process_reward.dataset import (
    SCHEMA_VERSION,
    ProcessRewardTraceRow,
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


# ── locked constants ────────────────────────────────────────────────


def test_default_alpha_locked() -> None:
    assert pytest.approx(0.10) == DEFAULT_ALPHA
    assert pytest.approx(0.90) == DEFAULT_TARGET_COVERAGE


def test_default_drift_tol_locked() -> None:
    assert pytest.approx(0.05) == DEFAULT_DRIFT_TOL


def test_default_position_buckets_locked() -> None:
    """Plan §11: 4 step-position buckets."""
    assert len(DEFAULT_POSITION_BUCKETS) == 4
    assert DEFAULT_POSITION_BUCKETS[0] == range(0, 1)
    assert DEFAULT_POSITION_BUCKETS[3] == range(7, 32)


# ── bucket_label / position_to_bucket_label ────────────────────────


def test_bucket_label_format() -> None:
    assert bucket_label(range(0, 1)) == "range(0, 1)"
    assert bucket_label(range(7, 32)) == "range(7, 32)"


def test_position_to_bucket_label_routes_correctly() -> None:
    assert position_to_bucket_label(0) == "range(0, 1)"
    assert position_to_bucket_label(2) == "range(1, 3)"
    assert position_to_bucket_label(5) == "range(3, 7)"
    assert position_to_bucket_label(15) == "range(7, 32)"


def test_position_to_bucket_label_returns_none_when_unbounded() -> None:
    assert position_to_bucket_label(1000) is None


# ── calibrate_residuals ─────────────────────────────────────────────


def test_calibrate_residuals_perfect_predictor() -> None:
    rows = [_trace((0.5, 0.7, 0.6)) for _ in range(20)]

    def perfect_step(prompt: str, steps, step_index: int) -> float:
        del prompt, steps
        # Return the same step reward as the row for perfect calibration.
        return [0.5, 0.7, 0.6][step_index]

    def perfect_agg(prompt: str, steps) -> float:
        del prompt, steps
        return (0.5 + 0.7 + 0.6) / 3

    result = calibrate_residuals(rows, perfect_step, perfect_agg, alpha=0.10)
    assert result.aggregate_quantile == pytest.approx(0.0)
    assert result.aggregate_empirical_coverage == pytest.approx(1.0)
    assert result.aggregate_drift == pytest.approx(0.10)


def test_calibrate_residuals_uniform_offset_predictor() -> None:
    rows = [_trace((0.5, 0.5)) for _ in range(50)]

    def offset_step(prompt: str, steps, step_index: int) -> float:
        del prompt, steps, step_index
        return 0.7  # constant +0.2 offset on every step

    def offset_agg(prompt: str, steps) -> float:
        del prompt, steps
        return 0.7

    result = calibrate_residuals(rows, offset_step, offset_agg, alpha=0.10)
    assert result.aggregate_quantile == pytest.approx(0.2)


def test_calibrate_residuals_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        calibrate_residuals(
            [], lambda p, s, t: 0.5, lambda p, s: 0.5
        )


def test_calibrate_residuals_rejects_invalid_alpha() -> None:
    rows = [_trace((0.5,))]
    with pytest.raises(ValueError, match="alpha"):
        calibrate_residuals(
            rows, lambda p, s, t: 0.5, lambda p, s: 0.5, alpha=0.0
        )


def test_calibrate_residuals_returns_per_step_bucket_records() -> None:
    rows = [_trace((0.5,) * 10) for _ in range(20)]

    def predict(prompt: str, steps, step_index: int) -> float:
        del prompt, steps, step_index
        return 0.6

    def agg(prompt: str, steps) -> float:
        del prompt, steps
        return 0.6

    result = calibrate_residuals(rows, predict, agg)
    # Should produce records for buckets that have residuals.
    assert len(result.per_step_bucket_results) > 0
    for record in result.per_step_bucket_results:
        assert record.n_residuals > 0


def test_calibration_result_drift_tolerance() -> None:
    result = CalibrationResult(
        per_step_quantiles={"range(0, 1)": 0.05},
        per_step_bucket_results=(
            PerStepBucketResult(
                bucket_label="range(0, 1)", n_residuals=10, quantile=0.05
            ),
        ),
        aggregate_quantile=0.1,
        aggregate_target_coverage=0.9,
        aggregate_empirical_coverage=0.92,
        aggregate_drift=0.02,
        n_traces=10,
        alpha=0.1,
    )
    assert result.is_calibration_suspect(drift_tol=0.05) is False
    assert result.is_calibration_suspect(drift_tol=0.01) is True


# ── evaluate_per_step_coverage ─────────────────────────────────────


def test_evaluate_per_step_coverage_perfect_predictor() -> None:
    rows = [_trace((0.5, 0.7, 0.6))]

    def perfect_step(prompt: str, steps, step_index: int) -> float:
        return [0.5, 0.7, 0.6][step_index]

    quantiles = {label: 0.05 for label in ("range(0, 1)", "range(1, 3)", "range(3, 7)", "range(7, 32)")}
    report = evaluate_per_step_coverage(
        rows, perfect_step, per_step_quantiles=quantiles
    )
    # Perfect predictions → coverage 1.0 across all populated buckets.
    assert report.overall == pytest.approx(1.0)


def test_evaluate_per_step_coverage_zero_when_residuals_exceed_quantile() -> None:
    rows = [_trace((0.0, 1.0))]

    def predictor(prompt: str, steps, step_index: int) -> float:
        # Big offset; residuals exceed any small quantile.
        return 1.0 - [0.0, 1.0][step_index]

    quantiles = {"range(0, 1)": 0.05, "range(1, 3)": 0.05}
    report = evaluate_per_step_coverage(
        rows, predictor, per_step_quantiles=quantiles
    )
    assert report.overall == pytest.approx(0.0)


# ── score_with_ci ─────────────────────────────────────────────────


def test_score_with_ci_clips_to_unit() -> None:
    payload = score_with_ci(
        prompt="p",
        steps=("s1", "s2"),
        step_predictor=lambda p, s, t: 0.95,
        aggregate_predictor=lambda p, s: 0.95,
        per_step_quantiles={"range(0, 1)": 0.2, "range(1, 3)": 0.2},
        aggregate_quantile=0.2,
    )
    assert payload["calibrated"] is True
    # Step 0: pred=0.95, ci=[0.75, min(1.0, 1.15)] → [0.75, 1.0].
    assert payload["step_confidence_intervals"][0][1] == pytest.approx(1.0)
    assert payload["aggregate_confidence_interval"][1] == pytest.approx(1.0)


def test_score_with_ci_low_clip() -> None:
    payload = score_with_ci(
        prompt="p",
        steps=("s1",),
        step_predictor=lambda p, s, t: 0.05,
        aggregate_predictor=lambda p, s: 0.05,
        per_step_quantiles={"range(0, 1)": 0.2},
        aggregate_quantile=0.2,
    )
    assert payload["step_confidence_intervals"][0][0] == pytest.approx(0.0)


def test_score_with_ci_rejects_negative_quantile() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        score_with_ci(
            prompt="p",
            steps=("s",),
            step_predictor=lambda p, s, t: 0.5,
            aggregate_predictor=lambda p, s: 0.5,
            per_step_quantiles={},
            aggregate_quantile=-0.1,
        )


def test_score_with_ci_rejects_empty_steps() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        score_with_ci(
            prompt="p",
            steps=(),
            step_predictor=lambda p, s, t: 0.5,
            aggregate_predictor=lambda p, s: 0.5,
            per_step_quantiles={},
            aggregate_quantile=0.1,
        )


def test_score_with_ci_falls_back_to_largest_quantile_for_unbounded_position() -> None:
    """A position past every defined bucket should use a conservative
    fallback (largest available quantile)."""
    payload = score_with_ci(
        prompt="p",
        steps=tuple("s" for _ in range(50)),  # position 49 is unbounded
        step_predictor=lambda p, s, t: 0.5,
        aggregate_predictor=lambda p, s: 0.5,
        per_step_quantiles={"range(0, 1)": 0.05, "range(1, 3)": 0.30},
        aggregate_quantile=0.1,
    )
    # Position 49 uses largest quantile (0.30); width = 0.6.
    last_ci = payload["step_confidence_intervals"][-1]
    assert last_ci[1] - last_ci[0] == pytest.approx(0.6)
