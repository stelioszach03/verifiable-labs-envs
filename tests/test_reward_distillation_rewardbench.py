"""Tests for the 29.D RewardBench adapter."""
from __future__ import annotations

import sys
import types

import pytest

from verifiable_labs_envs.reward_distillation.rewardbench_adapter import (
    DEFAULT_CATEGORIES,
    DEFAULT_PASS_THRESHOLD,
    PreferencePair,
    RewardBenchReport,
    build_synthetic_rewardbench,
    evaluate_rewardbench,
    length_bias_baseline,
    load_rewardbench_subset,
)


def test_build_synthetic_rewardbench_basic() -> None:
    pairs = build_synthetic_rewardbench(n=20, seed=0)
    assert len(pairs) == 20
    assert all(isinstance(p, PreferencePair) for p in pairs)
    categories = {p.category for p in pairs}
    assert categories.issubset(set(DEFAULT_CATEGORIES))


def test_build_synthetic_rewardbench_deterministic() -> None:
    a = build_synthetic_rewardbench(n=10, seed=42)
    b = build_synthetic_rewardbench(n=10, seed=42)
    assert [p.pair_id for p in a] == [p.pair_id for p in b]


def test_build_synthetic_rewardbench_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_synthetic_rewardbench(n=-1)


def test_evaluate_rewardbench_perfect_predictor() -> None:
    """A predictor that always picks the longer completion lands above
    50 % on the synthetic benchmark (chosen completions are intentionally
    longer in non-safety categories)."""
    pairs = build_synthetic_rewardbench(n=80, seed=0)
    report = evaluate_rewardbench(pairs, length_bias_baseline)
    assert report.overall_accuracy > 0.5


def test_evaluate_rewardbench_constant_predictor_at_zero_accuracy() -> None:
    """A constant predictor produces ties on every pair → 0 correct
    but ties counted in n_ties; accuracy is 0 since neither chosen
    nor rejected scored higher."""
    pairs = build_synthetic_rewardbench(n=20, seed=0)
    report = evaluate_rewardbench(pairs, lambda p, c: 0.5)
    assert report.overall_accuracy == 0.0
    assert report.n_ties == 20


def test_evaluate_rewardbench_empty_returns_zero() -> None:
    report = evaluate_rewardbench([], lambda p, c: 0.5)
    assert report.n_pairs == 0
    assert report.overall_accuracy == 0.0


def test_evaluate_rewardbench_per_category_breakdown() -> None:
    pairs = build_synthetic_rewardbench(n=40, seed=0)
    report = evaluate_rewardbench(pairs, length_bias_baseline)
    assert report.per_category
    assert report.per_category_count
    for cat, count in report.per_category_count.items():
        assert count > 0
        assert 0.0 <= report.per_category.get(cat, 0.0) <= 1.0


def test_rewardbench_report_passes_at_threshold() -> None:
    report = RewardBenchReport(n_pairs=100, overall_accuracy=0.66)
    assert report.passes(threshold=0.65) is True
    assert report.passes(threshold=0.70) is False


def test_rewardbench_report_to_dict_serialisable() -> None:
    report = RewardBenchReport(
        n_pairs=10,
        overall_accuracy=0.7,
        per_category={"chat": 0.8, "safety": 0.6},
        per_category_count={"chat": 5, "safety": 5},
    )
    d = report.to_dict()
    assert d["n_pairs"] == 10
    assert d["per_category"]["chat"] == pytest.approx(0.8)


def test_default_pass_threshold_locked_per_plan() -> None:
    """Plan §5 D7 pass criterion 3: RewardBench overall ≥ 65%."""
    assert pytest.approx(0.65) == DEFAULT_PASS_THRESHOLD


def test_load_rewardbench_subset_falls_back_to_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `datasets` isn't importable, the loader falls back to the
    deterministic synthetic benchmark."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("forced unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    pairs = load_rewardbench_subset(n=10, seed=0)
    assert len(pairs) == 10


def test_load_rewardbench_subset_zero_returns_empty() -> None:
    assert load_rewardbench_subset(n=0) == []


def test_load_rewardbench_subset_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        load_rewardbench_subset(n=-1)


def test_load_rewardbench_subset_falls_back_when_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `datasets.load_dataset` raises, fallback kicks in."""
    fake_module = types.ModuleType("datasets")

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    fake_module.load_dataset = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    pairs = load_rewardbench_subset(n=5, seed=0)
    assert len(pairs) == 5


def test_load_rewardbench_subset_real_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake datasets module that returns a fake reward-bench dataset."""

    class _FakeDS:
        def __init__(self, records):
            self._records = records

        def __len__(self):
            return len(self._records)

        def __getitem__(self, idx):
            return self._records[idx]

    records = [
        {
            "prompt": f"q-{i}",
            "chosen": f"good-{i}",
            "rejected": f"bad-{i}",
            "subset": "chat",
            "id": f"rb-{i}",
        }
        for i in range(20)
    ]
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda *a, **k: _FakeDS(records)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    pairs = load_rewardbench_subset(n=8, seed=0)
    assert len(pairs) == 8
    for p in pairs:
        assert p.prompt.startswith("q-")
        assert p.chosen.startswith("good-")


def test_synthetic_benchmark_includes_safety_category() -> None:
    pairs = build_synthetic_rewardbench(n=40, seed=0)
    safety = [p for p in pairs if p.category == "safety"]
    assert safety  # at least one safety-category pair


def test_evaluate_rewardbench_handles_ties_diagnostic() -> None:
    pairs = [
        PreferencePair("p", "same", "same", "chat", "rb-1"),
        PreferencePair("p", "winner", "loser", "chat", "rb-2"),
    ]
    report = evaluate_rewardbench(pairs, lambda p, c: float(len(c)))
    # One tie + one correct => 0.5 accuracy, 1 tie counted.
    assert report.overall_accuracy == pytest.approx(0.5)
    assert report.n_ties == 1


def test_length_bias_baseline_clips_to_unit() -> None:
    short = length_bias_baseline("p", "x")
    long = length_bias_baseline("p", "x" * 500)
    assert 0.0 <= short <= 1.0
    assert long == pytest.approx(1.0)
