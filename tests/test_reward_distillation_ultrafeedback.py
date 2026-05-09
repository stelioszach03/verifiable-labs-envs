"""Unit tests for ``verifiable_labs_envs.reward_distillation.ultrafeedback``."""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from verifiable_labs_envs.reward_distillation.ultrafeedback import (
    DEFAULT_DATASET_NAME,
    DEFAULT_OVERALL_SCALE,
    build_synthetic_subset,
    collect_external_subset,
    collect_ultrafeedback_subset,
    deterministic_subset_indices,
    fingerprint_text,
)

# ── synthetic subset (offline) ──────────────────────────────────────


def test_build_synthetic_subset_count() -> None:
    rows = build_synthetic_subset(5, seed=0)
    assert len(rows) == 5


def test_build_synthetic_subset_determinism() -> None:
    a = build_synthetic_subset(3, seed=42)
    b = build_synthetic_subset(3, seed=42)
    assert [r.row_id for r in a] == [r.row_id for r in b]
    assert [r.consensus_reward for r in a] == [r.consensus_reward for r in b]


def test_build_synthetic_subset_diverges_on_seed() -> None:
    a = build_synthetic_subset(5, seed=0)
    b = build_synthetic_subset(5, seed=1)
    assert [r.row_id for r in a] != [r.row_id for r in b]


def test_build_synthetic_subset_zero_returns_empty() -> None:
    assert build_synthetic_subset(0) == []


def test_build_synthetic_subset_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_synthetic_subset(-1)


def test_build_synthetic_subset_row_shape() -> None:
    row = build_synthetic_subset(1, seed=0)[0]
    assert row.env_id is None
    assert row.env_reward is None
    assert row.frontier_judgment is not None
    assert 0.0 <= row.frontier_judgment <= 1.0
    assert row.source == "external"
    assert row.metadata.get("synthetic") is True
    assert row.metadata["external_dataset"] == DEFAULT_DATASET_NAME


def test_build_synthetic_subset_consensus_equals_frontier() -> None:
    """For external rows env_reward is None, so consensus = frontier (no blend)."""
    rows = build_synthetic_subset(10, seed=7)
    for row in rows:
        assert row.consensus_reward == pytest.approx(row.frontier_judgment)


# ── real-loader path (datasets unavailable) ─────────────────────────


def test_collect_ultrafeedback_falls_back_when_datasets_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the ``datasets`` library import fails, the loader falls back to
    the synthetic path and never raises."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("synthetic test: datasets not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rows = collect_ultrafeedback_subset(3, seed=0)
    assert len(rows) == 3
    for row in rows:
        assert row.metadata["synthetic"] is True


def test_collect_ultrafeedback_raises_when_synthetic_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("forced unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="UltraFeedback unavailable"):
        collect_ultrafeedback_subset(2, fallback_to_synthetic=False)


# ── real-loader path (datasets shape) ───────────────────────────────


class _FakeDatasetRecord(dict):
    """Dict-with-getitem matching the HuggingFace Dataset row API."""


class _FakeDataset:
    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        return _FakeDatasetRecord(self._records[idx])


def _install_fake_datasets(records: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stand-in ``datasets`` module under ``sys.modules`` and
    purge any real submodules from a partial earlier import so the fake
    fully owns the ``datasets`` namespace for the duration of the test.
    monkeypatch restores everything on teardown."""
    for key in list(sys.modules):
        if key == "datasets" or key.startswith("datasets."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = lambda *args, **kwargs: _FakeDataset(records)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_collect_ultrafeedback_real_path_projects_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "instruction": f"q-{i}",
            "completions": [
                {
                    "response": f"a-{i}",
                    "annotations": {"overall-rating": str((i % 9) + 1)},
                }
            ],
        }
        for i in range(20)
    ]
    _install_fake_datasets(records, monkeypatch)

    rows = collect_ultrafeedback_subset(5, seed=0)
    assert len(rows) == 5
    for row in rows:
        assert row.metadata["synthetic"] is False
        assert row.frontier_judgment is not None
        assert 0.0 <= row.frontier_judgment <= 1.0
        assert row.prompt.startswith("q-")
        assert row.completion.startswith("a-")


def test_collect_ultrafeedback_drops_malformed_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"instruction": "good", "completions": [{"response": "fine"}]},
        {"instruction": "", "completions": [{"response": "skipme"}]},
        {"completions": [{"response": "missing-instruction"}]},
        {"instruction": "another good", "completions": [{"response": "ok"}]},
    ]
    _install_fake_datasets(records, monkeypatch)

    # Force the loader to pick all 4 records (n >= len(records) is clamped).
    rows = collect_ultrafeedback_subset(4, seed=0)
    # The two malformed rows are dropped, leaving the 2 good ones.
    assert len(rows) == 2


def test_collect_ultrafeedback_falls_back_when_dataset_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_datasets([], monkeypatch)
    rows = collect_ultrafeedback_subset(3, seed=0)
    # Empty real dataset → fallback to synthetic.
    assert len(rows) == 3
    for row in rows:
        assert row.metadata["synthetic"] is True


def test_collect_ultrafeedback_falls_back_when_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("datasets")

    def boom(*args, **kwargs):
        raise RuntimeError("network unreachable")

    fake_module.load_dataset = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    rows = collect_ultrafeedback_subset(2, seed=0)
    assert len(rows) == 2
    for row in rows:
        assert row.metadata["synthetic"] is True


def test_overall_score_rescales_to_unit_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real UltraFeedback ratings are 1-10 strings; we divide by 10."""
    records = [
        {
            "instruction": "q",
            "completions": [
                {"response": "a", "annotations": {"overall-rating": "10"}}
            ],
        }
    ]
    _install_fake_datasets(records, monkeypatch)
    rows = collect_ultrafeedback_subset(1, seed=0)
    assert rows[0].frontier_judgment == pytest.approx(10.0 / DEFAULT_OVERALL_SCALE)


def test_overall_score_neutral_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"instruction": "q", "completions": [{"response": "a"}]}]
    _install_fake_datasets(records, monkeypatch)
    rows = collect_ultrafeedback_subset(1, seed=0)
    assert rows[0].frontier_judgment == pytest.approx(0.5)


# ── deterministic_subset_indices ────────────────────────────────────


def test_deterministic_subset_indices_size() -> None:
    idxs = deterministic_subset_indices(100, 10, seed=0)
    assert len(idxs) == 10
    assert all(0 <= i < 100 for i in idxs)
    assert idxs == sorted(idxs)


def test_deterministic_subset_indices_reproducible() -> None:
    a = deterministic_subset_indices(100, 10, seed=42)
    b = deterministic_subset_indices(100, 10, seed=42)
    assert a == b


def test_deterministic_subset_indices_clamps_to_total() -> None:
    """Asking for more than the total returns the whole population."""
    idxs = deterministic_subset_indices(5, 10, seed=0)
    assert len(idxs) == 5
    assert sorted(idxs) == [0, 1, 2, 3, 4]


def test_deterministic_subset_indices_zero_total() -> None:
    assert deterministic_subset_indices(0, 5, seed=0) == []


# ── fingerprint helper ──────────────────────────────────────────────


def test_fingerprint_text_deterministic() -> None:
    a = fingerprint_text("hello world")
    b = fingerprint_text("hello world")
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_fingerprint_text_diverges_on_change() -> None:
    assert fingerprint_text("a") != fingerprint_text("b")


# ── alias surface ───────────────────────────────────────────────────


def test_collect_external_subset_alias_matches() -> None:
    """``collect_external_subset`` is an alias for the canonical name."""
    assert collect_external_subset is collect_ultrafeedback_subset


# ── direct rejection of negative n ──────────────────────────────────


def test_collect_ultrafeedback_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        collect_ultrafeedback_subset(-3)


# ── seed reproducibility on real path ───────────────────────────────


def test_real_path_seed_picks_same_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"instruction": f"i{i}", "completions": [{"response": f"r{i}"}]}
        for i in range(50)
    ]
    _install_fake_datasets(records, monkeypatch)
    a = collect_ultrafeedback_subset(8, seed=123)
    b = collect_ultrafeedback_subset(8, seed=123)
    assert [r.row_id for r in a] == [r.row_id for r in b]


def test_zero_request_short_circuits() -> None:
    """``n=0`` returns immediately without touching the loader."""
    with patch(
        "verifiable_labs_envs.reward_distillation.ultrafeedback._load_real_dataset"
    ) as loader:
        rows = collect_ultrafeedback_subset(0)
    loader.assert_not_called()
    assert rows == []
