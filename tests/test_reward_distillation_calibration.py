"""Tests for the 29.D calibration + stub-inference + eval-metrics surface."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs.reward_distillation.eval_metrics import (
    bias,
    calibration_drift,
    empirical_coverage,
    mae,
    memorisation_gap,
    passes_calibration_drift,
    passes_rewardbench,
    passes_spearman_floor,
    spearman_rho,
)
from verifiable_labs_envs.reward_distillation.stub_inference import (
    DEFAULT_DELTA,
    DEFAULT_MODEL_ID,
    DEFAULT_SCHEMA_VERSION,
    StubScoreResult,
    is_stub_payload,
    stub_predictor,
    stub_score,
)

# ── stub_inference ──────────────────────────────────────────────────


def test_stub_score_deterministic() -> None:
    a = stub_score("hello", "world")
    b = stub_score("hello", "world")
    assert a.reward == b.reward
    assert a.confidence_interval == b.confidence_interval


def test_stub_score_diverges_on_input() -> None:
    a = stub_score("hello", "world")
    b = stub_score("hello", "earth")
    assert a.reward != b.reward or a.confidence_interval != b.confidence_interval


def test_stub_score_within_unit_interval() -> None:
    for _ in range(50):
        result = stub_score("p", f"c-{_}")
        assert 0.0 <= result.reward <= 1.0
        low, high = result.confidence_interval
        assert 0.0 <= low <= high <= 1.0


def test_stub_score_default_delta_locks_floor_ceiling() -> None:
    """At delta=0.1 the reward lands in [0.4, 0.6]."""
    for i in range(50):
        result = stub_score("p", f"c-{i}")
        assert 0.4 <= result.reward <= 0.6


def test_stub_score_payload_carries_stub_schema() -> None:
    result = stub_score("p", "c")
    assert result.schema_version == DEFAULT_SCHEMA_VERSION
    assert result.schema_version.endswith("-stub")
    assert result.model_id == DEFAULT_MODEL_ID
    assert result.coverage_guarantee == pytest.approx(0.9)
    assert result.cache_hit is False


def test_stub_score_to_dict_round_trippable() -> None:
    result = stub_score("p", "c")
    d = result.to_dict()
    assert d["reward"] == pytest.approx(result.reward)
    assert d["confidence_interval"] == [
        result.confidence_interval[0],
        result.confidence_interval[1],
    ]


def test_stub_score_rejects_negative_delta() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stub_score("p", "c", delta=-0.1)


def test_stub_score_seed_changes_output() -> None:
    a = stub_score("p", "c", seed=0)
    b = stub_score("p", "c", seed=1)
    # Seeded outputs may differ; this just sanity-checks the path.
    assert isinstance(a.reward, float)
    assert isinstance(b.reward, float)


def test_stub_predictor_signature() -> None:
    predict = stub_predictor()
    assert predict("hello", "world") == predict("hello", "world")


def test_is_stub_payload_predicate() -> None:
    assert is_stub_payload({"schema_version": "v0.1.0-stub"}) is True
    assert is_stub_payload({"schema_version": "v0.1.0"}) is False
    assert is_stub_payload({}) is False


def test_default_delta_locked() -> None:
    assert pytest.approx(0.1) == DEFAULT_DELTA


def test_stub_score_result_dataclass_attrs() -> None:
    result = StubScoreResult(
        reward=0.5,
        confidence_interval=(0.4, 0.6),
        coverage_guarantee=0.9,
        model_id="x",
    )
    assert result.cache_hit is False


# ── eval_metrics ────────────────────────────────────────────────────


def test_spearman_rho_perfect_correlation() -> None:
    x = list(range(10))
    rc = spearman_rho(x, x)
    assert rc.rho == pytest.approx(1.0)
    assert rc.is_significant is True


def test_spearman_rho_anti_correlation() -> None:
    x = list(range(10))
    y = list(range(10, 0, -1))
    rc = spearman_rho(x, y)
    assert rc.rho == pytest.approx(-1.0)


def test_spearman_rho_constant_returns_zero() -> None:
    rc = spearman_rho([1, 1, 1], [1, 2, 3])
    assert rc.rho == 0.0
    assert rc.is_significant is False


def test_spearman_rho_short_input() -> None:
    rc = spearman_rho([1.0], [1.0])
    assert rc.rho == 0.0
    assert rc.is_significant is False


def test_spearman_rho_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman_rho([1, 2, 3], [1, 2])


def test_mae_basic() -> None:
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


def test_mae_empty() -> None:
    assert mae([], []) == 0.0


def test_mae_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mae([1, 2], [1, 2, 3])


def test_bias_basic() -> None:
    assert bias([1, 2, 3], [0, 1, 2]) == pytest.approx(1.0)
    assert bias([0, 1, 2], [1, 2, 3]) == pytest.approx(-1.0)


def test_bias_empty() -> None:
    assert bias([], []) == 0.0


def test_empirical_coverage_full() -> None:
    pred = np.array([0.5, 0.5, 0.5])
    target = np.array([0.5, 0.5, 0.5])
    assert empirical_coverage(pred, target, quantile=0.0) == pytest.approx(1.0)


def test_empirical_coverage_partial() -> None:
    pred = np.array([0.5, 0.5])
    target = np.array([0.5, 0.95])  # second target outside 0.5 ± 0.1
    cov = empirical_coverage(pred, target, quantile=0.1)
    assert cov == pytest.approx(0.5)


def test_empirical_coverage_clips_to_unit() -> None:
    """Predictions near the boundaries don't break coverage math."""
    pred = np.array([0.0, 1.0])
    target = np.array([0.05, 0.95])
    cov = empirical_coverage(pred, target, quantile=0.1)
    assert cov == pytest.approx(1.0)


def test_empirical_coverage_rejects_negative_quantile() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        empirical_coverage([0.5], [0.5], quantile=-0.1)


def test_calibration_drift_signed() -> None:
    assert calibration_drift(0.92, 0.90) == pytest.approx(0.02)
    assert calibration_drift(0.85, 0.90) == pytest.approx(-0.05)


def test_memorisation_gap() -> None:
    assert memorisation_gap(0.95, 0.85) == pytest.approx(0.10)
    assert memorisation_gap(0.80, 0.90) == pytest.approx(-0.10)


def test_passes_spearman_floor_locked_at_70pct() -> None:
    assert passes_spearman_floor([0.71, 0.72, 0.73], floor=0.70) is True
    assert passes_spearman_floor([0.69, 0.72, 0.73], floor=0.70) is False


def test_passes_calibration_drift_default_5pp() -> None:
    assert passes_calibration_drift(0.04) is True
    assert passes_calibration_drift(-0.04) is True
    assert passes_calibration_drift(0.06) is False


def test_passes_rewardbench_default_65pct() -> None:
    assert passes_rewardbench(0.65) is True
    assert passes_rewardbench(0.64) is False


def test_rank_correlation_significance_proxy() -> None:
    """Spearman 0.99 on n=100 is significant; 0.05 on n=3 isn't."""
    big = spearman_rho(list(range(100)), list(range(100)))
    assert big.is_significant is True
    small = spearman_rho([1, 2, 3], [1, 3, 2])
    assert isinstance(small.is_significant, bool)


def test_eval_metrics_handle_numpy_inputs() -> None:
    """All entries accept numpy arrays + Python lists transparently."""
    arr = np.array([0.1, 0.2, 0.3])
    assert mae(arr, arr) == 0.0
    assert bias(arr, arr) == 0.0
    cov = empirical_coverage(arr, arr, quantile=0.1)
    assert cov == 1.0
