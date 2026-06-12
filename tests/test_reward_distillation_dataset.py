"""Unit tests for ``verifiable_labs_envs.reward_distillation.dataset``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verifiable_labs_envs import _REGISTRY
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_HELD_OUT_ENVS,
    DEFAULT_TRAINING_ENVS,
    SCHEMA_VERSION,
    RewardTrainingRow,
    baseline_completion_source,
    collect_env_rows,
    dataset_summary,
    default_train_envs,
    env_disk_size_estimate,
    env_loader_safe,
    is_held_out,
    is_phase29_collect_frontier_enabled,
    make_row_id,
    merge_jsonl,
    output_path_default,
    read_jsonl,
    write_jsonl,
)

# ── locked constants ────────────────────────────────────────────────


def test_held_out_envs_locked_per_plan() -> None:
    """D7-A held-out envs (plan §5 D4) — order is part of the contract."""
    assert DEFAULT_HELD_OUT_ENVS == (
        "long-context-synthesis",
        "sql-multiturn",
        "code-mini-repo",
    )


def test_held_out_envs_subset_of_registry() -> None:
    assert set(DEFAULT_HELD_OUT_ENVS).issubset(set(_REGISTRY))


def test_default_training_envs_disjoint_from_held_out() -> None:
    held = set(DEFAULT_HELD_OUT_ENVS)
    train = set(DEFAULT_TRAINING_ENVS)
    assert train.isdisjoint(held)
    assert train | held == set(_REGISTRY)


def test_default_train_envs_returns_22_for_25_env_catalogue() -> None:
    # Plan §5 D4 promises 22 training envs for v0.0.1.
    envs = default_train_envs()
    assert len(envs) == len(_REGISTRY) - len(DEFAULT_HELD_OUT_ENVS)
    assert sorted(envs) == envs  # sorted convention


def test_is_held_out_predicate() -> None:
    assert is_held_out("sql-multiturn") is True
    assert is_held_out("math-algebra") is False
    assert is_held_out(None) is False


def test_output_path_default_under_reports() -> None:
    path = output_path_default()
    parts = path.parts
    assert "reports" in parts
    assert "reward_distillation" in parts


# ── make_row_id determinism ─────────────────────────────────────────


def test_make_row_id_deterministic() -> None:
    a = make_row_id("math-algebra", "p", "c", seed=0)
    b = make_row_id("math-algebra", "p", "c", seed=0)
    assert a == b


def test_make_row_id_diverges_on_seed() -> None:
    a = make_row_id("math-algebra", "p", "c", seed=0)
    b = make_row_id("math-algebra", "p", "c", seed=1)
    assert a != b


def test_make_row_id_diverges_on_env() -> None:
    a = make_row_id("math-algebra", "p", "c", seed=0)
    b = make_row_id("sql-single-turn", "p", "c", seed=0)
    assert a != b


def test_make_row_id_handles_none_env() -> None:
    rid = make_row_id(None, "external prompt", "external completion")
    assert rid.startswith("rwd_")
    assert len(rid) == 4 + 16  # "rwd_" + 16 hex chars


# ── RewardTrainingRow ───────────────────────────────────────────────


def _sample_row() -> RewardTrainingRow:
    return RewardTrainingRow(
        row_id="rwd_" + "0" * 16,
        env_id="math-algebra",
        prompt="2 + 2 = ?",
        completion="4",
        env_reward=1.0,
        env_components={"format_valid": 1.0, "correct": 1.0},
        conformal_interval=(0.85, 1.0),
        frontier_judgment=None,
        frontier_rationale=None,
        consensus_reward=1.0,
        disagreement=None,
        source="env",
        metadata={"seed": 0, "schema_version": SCHEMA_VERSION},
    )


def test_row_to_dict_roundtrip() -> None:
    row = _sample_row()
    d = row.to_dict()
    assert d["env_id"] == "math-algebra"
    assert d["conformal_interval"] == [0.85, 1.0]
    restored = RewardTrainingRow.from_dict(d)
    assert restored == row


def test_row_from_dict_handles_none_ci() -> None:
    payload = {
        "row_id": "rwd_x",
        "env_id": None,
        "prompt": "p",
        "completion": "c",
        "env_reward": None,
        "env_components": None,
        "conformal_interval": None,
        "frontier_judgment": 0.7,
        "frontier_rationale": None,
        "consensus_reward": 0.7,
        "disagreement": None,
        "source": "external",
        "metadata": {},
    }
    row = RewardTrainingRow.from_dict(payload)
    assert row.conformal_interval is None
    assert row.frontier_judgment == 0.7


# ── env-row extraction (real env) ───────────────────────────────────


def test_collect_env_rows_basic() -> None:
    """The fastest text env (math-algebra) supplies a real signal — use it
    for the smoke run."""
    rows = collect_env_rows(["math-algebra"], n_per_env=2)
    assert len(rows) == 2
    for row in rows:
        assert row.env_id == "math-algebra"
        assert row.source == "env"
        assert row.env_reward is not None
        assert 0.0 <= row.env_reward <= 1.0
        assert row.consensus_reward == pytest.approx(row.env_reward)
        assert row.metadata.get("schema_version") == SCHEMA_VERSION
        assert row.frontier_judgment is None


def test_collect_env_rows_seed_deterministic() -> None:
    rows_a = collect_env_rows(["math-algebra"], n_per_env=3, seed_start=100)
    rows_b = collect_env_rows(["math-algebra"], n_per_env=3, seed_start=100)
    assert [r.row_id for r in rows_a] == [r.row_id for r in rows_b]
    assert [r.env_reward for r in rows_a] == [r.env_reward for r in rows_b]


def test_collect_env_rows_distinct_seeds_distinct_ids() -> None:
    rows = collect_env_rows(["math-algebra"], n_per_env=4, seed_start=200)
    ids = {row.row_id for row in rows}
    assert len(ids) == 4


def test_collect_env_rows_empty_when_n_zero() -> None:
    rows = collect_env_rows(["math-algebra"], n_per_env=0)
    assert rows == []


def test_collect_env_rows_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        collect_env_rows(["math-algebra"], n_per_env=-1)


def test_collect_env_rows_continue_on_error_drops_row() -> None:
    """When the completion source raises, the row is dropped (default)."""

    def broken_source(env_id, env, instance, seed):  # noqa: ARG001
        if seed % 2 == 1:
            raise RuntimeError("synthetic failure")
        return baseline_completion_source(env_id, env, instance, seed)

    rows = collect_env_rows(
        ["math-algebra"],
        n_per_env=4,
        completion_source=broken_source,
    )
    # Seeds 0, 2 succeed; 1, 3 are dropped.
    assert len(rows) == 2


def test_collect_env_rows_fail_fast_propagates_error() -> None:
    def broken_source(env_id, env, instance, seed):  # noqa: ARG001
        raise RuntimeError("explode")

    with pytest.raises(RuntimeError, match="explode"):
        collect_env_rows(
            ["math-algebra"],
            n_per_env=2,
            completion_source=broken_source,
            fail_fast=True,
        )


def test_collect_env_rows_uses_on_error_callback() -> None:
    captured: list[tuple[str, int, str]] = []

    def broken_source(env_id, env, instance, seed):  # noqa: ARG001
        raise RuntimeError(f"err-{seed}")

    rows = collect_env_rows(
        ["math-algebra"],
        n_per_env=2,
        completion_source=broken_source,
        on_error=lambda env_id, seed, exc: captured.append((env_id, seed, str(exc))),
    )
    assert rows == []
    assert len(captured) == 2
    assert captured[0][0] == "math-algebra"


def test_baseline_completion_source_returns_text() -> None:
    from verifiable_labs_envs import load_environment

    env = load_environment("math-algebra")
    instance = env.generate_instance(0)
    prompt, completion, score = baseline_completion_source(
        "math-algebra", env, instance, 0
    )
    assert isinstance(prompt, str) and prompt
    assert isinstance(completion, str)
    assert isinstance(score, dict) and "reward" in score


def test_baseline_completion_source_unknown_env_raises() -> None:
    """``importlib.import_module`` raises KeyError when the env id isn't in
    the registry; baseline_completion_source surfaces it as KeyError too."""
    with pytest.raises(KeyError):
        baseline_completion_source("does-not-exist", None, None, 0)


# ── env_loader_safe ─────────────────────────────────────────────────


def test_env_loader_safe_resolves_known_env() -> None:
    env = env_loader_safe("math-algebra")
    assert env is not None
    assert hasattr(env, "generate_instance")


def test_env_loader_safe_rejects_unknown_env() -> None:
    with pytest.raises(KeyError, match="unknown env"):
        env_loader_safe("not-a-real-env")


# ── JSONL IO ────────────────────────────────────────────────────────


def test_jsonl_write_read_roundtrip(tmp_path: Path) -> None:
    rows = collect_env_rows(["math-algebra"], n_per_env=3, seed_start=10)
    path = tmp_path / "rows.jsonl"
    n = write_jsonl(rows, path)
    assert n == 3
    assert path.exists()
    restored = read_jsonl(path)
    assert restored == rows


def test_jsonl_byte_stable_on_repeat_write(tmp_path: Path) -> None:
    """Same input → byte-identical file (write_jsonl uses sort_keys)."""
    rows = collect_env_rows(["math-algebra"], n_per_env=2)
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(rows, a)
    write_jsonl(rows, b)
    assert a.read_bytes() == b.read_bytes()


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    """Reader tolerates trailing newlines / blank lines from manual edits."""
    path = tmp_path / "with_blanks.jsonl"
    payload = json.dumps(_sample_row().to_dict(), sort_keys=True)
    path.write_text(f"\n{payload}\n\n{payload}\n", encoding="utf-8")
    rows = read_jsonl(path)
    assert len(rows) == 2


def test_jsonl_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "deeper" / "out.jsonl"
    rows = [_sample_row()]
    write_jsonl(rows, nested)
    assert nested.exists()


def test_merge_jsonl_concatenates_in_order(tmp_path: Path) -> None:
    rows_a = collect_env_rows(["math-algebra"], n_per_env=2, seed_start=0)
    rows_b = collect_env_rows(["math-algebra"], n_per_env=2, seed_start=100)
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    write_jsonl(rows_a, path_a)
    write_jsonl(rows_b, path_b)
    merged = merge_jsonl([path_a, path_b])
    assert merged == [*rows_a, *rows_b]


# ── summary + footprint ──────────────────────────────────────────────


def test_dataset_summary_empty() -> None:
    summary = dataset_summary([])
    assert summary["n_rows"] == 0
    assert summary["by_env"] == {}
    assert summary["schema_version"] == SCHEMA_VERSION


def test_dataset_summary_aggregates_per_env() -> None:
    rows = collect_env_rows(["math-algebra"], n_per_env=3)
    summary = dataset_summary(rows)
    assert summary["n_rows"] == 3
    assert summary["by_env"] == {"math-algebra": 3}
    assert summary["by_source"] == {"env": 3}
    assert 0.0 <= summary["consensus_min"] <= summary["consensus_max"] <= 1.0


def test_env_disk_size_estimate_positive() -> None:
    rows = [_sample_row()]
    size = env_disk_size_estimate(rows)
    assert size > 0
    assert env_disk_size_estimate([]) == 0


# ── frontier slice gate ─────────────────────────────────────────────


def test_frontier_gate_default_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLABS_PHASE29_COLLECT_FRONTIER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert is_phase29_collect_frontier_enabled() is False


def test_frontier_gate_requires_both_flag_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLABS_PHASE29_COLLECT_FRONTIER", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert is_phase29_collect_frontier_enabled() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert is_phase29_collect_frontier_enabled() is True


def test_frontier_gate_accepts_truthy_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("VLABS_PHASE29_COLLECT_FRONTIER", value)
        assert is_phase29_collect_frontier_enabled() is True


def test_frontier_gate_rejects_falsy_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("VLABS_PHASE29_COLLECT_FRONTIER", value)
        assert is_phase29_collect_frontier_enabled() is False


# ── numpy-encoder safety ────────────────────────────────────────────


def test_baseline_source_handles_numpy_arrays_in_inputs() -> None:
    """Envs with numpy arrays in ``as_inputs()`` (sparse-fourier) must not
    crash JSON serialization."""
    from verifiable_labs_envs import load_environment

    env = load_environment("sparse-fourier-recovery")
    instance = env.generate_instance(0)
    prompt, completion, score = baseline_completion_source(
        "sparse-fourier-recovery", env, instance, 0
    )
    # Both must be valid JSON strings.
    parsed_prompt = json.loads(prompt)
    assert parsed_prompt["env_id"] == "sparse-fourier-recovery"
    assert parsed_prompt["seed"] == 0
    assert isinstance(score["reward"], float)
    assert isinstance(json.loads(completion), dict)
    # Numpy arrays must be coerced to lists.
    inputs = parsed_prompt["inputs"]
    if "y" in inputs:
        assert isinstance(inputs["y"], list)


def test_collect_env_rows_handles_numpy_envs_too() -> None:
    """End-to-end smoke that sparse-fourier rows survive JSONL roundtrip."""
    rows = collect_env_rows(["sparse-fourier-recovery"], n_per_env=1)
    assert len(rows) == 1
    assert rows[0].env_id == "sparse-fourier-recovery"
    # Roundtrippable through the dict form.
    restored = RewardTrainingRow.from_dict(rows[0].to_dict())
    assert restored == rows[0]


def test_numpy_floats_are_serializable() -> None:
    """Defence against a regression: env score components sometimes carry
    numpy scalar floats that json.dumps can't handle natively."""
    rows = collect_env_rows(["math-algebra"], n_per_env=1)
    payload = json.dumps(rows[0].to_dict(), sort_keys=True)
    assert "math-algebra" in payload
    assert "env_reward" in payload
    assert "consensus_reward" in payload
    # Round-trippable through json.loads / from_dict.
    restored = RewardTrainingRow.from_dict(json.loads(payload))
    assert restored == rows[0]


def test_collect_env_rows_supplies_conformal_interval_when_calibrated() -> None:
    rows = collect_env_rows(["math-algebra"], n_per_env=2)
    # math-algebra exposes a conformal quantile in score meta, so the row
    # carries a CI; this is the moat-aligned signal for D10 calibration.
    for row in rows:
        if row.conformal_interval is not None:
            low, high = row.conformal_interval
            assert 0.0 <= low <= high <= 1.0
            return
    pytest.fail("expected at least one row to carry a conformal interval")


def test_default_training_envs_includes_math_algebra() -> None:
    """math-algebra is the canonical fast text env — it must remain in the
    training pool (regression guard against accidental held-out moves)."""
    assert "math-algebra" in DEFAULT_TRAINING_ENVS


def test_dataset_summary_consensus_stats_match_rows() -> None:
    rows = collect_env_rows(["math-algebra"], n_per_env=4, seed_start=500)
    summary = dataset_summary(rows)
    consensus = np.asarray([r.consensus_reward for r in rows])
    assert summary["consensus_mean"] == pytest.approx(float(consensus.mean()))
    assert summary["consensus_min"] == pytest.approx(float(consensus.min()))
    assert summary["consensus_max"] == pytest.approx(float(consensus.max()))
