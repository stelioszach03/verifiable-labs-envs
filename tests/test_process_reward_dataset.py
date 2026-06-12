"""Tests for ``verifiable_labs_envs.process_reward.dataset``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifiable_labs_envs.process_reward.dataset import (
    DEFAULT_AGGREGATION_METHOD,
    DEFAULT_HELD_OUT_ENVS,
    DEFAULT_TRAINING_ENVS,
    SCHEMA_VERSION,
    ProcessRewardTraceRow,
    collect_env_traces,
    default_train_envs,
    extend_from_phase29_rows,
    is_held_out,
    is_phase30_collect_frontier_enabled,
    make_row_id,
    merge_jsonl,
    output_path_default,
    read_jsonl,
    trace_dataset_summary,
    write_jsonl,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_HELD_OUT_ENVS as PHASE29_HELD_OUT,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    RewardTrainingRow,
)

# ── locked constants ────────────────────────────────────────────────


def test_held_out_envs_match_phase29() -> None:
    """Plan §3: D7-A held-out envs are the same as Phase 29's."""
    assert DEFAULT_HELD_OUT_ENVS == PHASE29_HELD_OUT
    assert DEFAULT_HELD_OUT_ENVS == (
        "long-context-synthesis",
        "sql-multiturn",
        "code-mini-repo",
    )


def test_default_training_envs_disjoint() -> None:
    held = set(DEFAULT_HELD_OUT_ENVS)
    train = set(DEFAULT_TRAINING_ENVS)
    assert train.isdisjoint(held)


def test_default_aggregation_method_locked() -> None:
    assert DEFAULT_AGGREGATION_METHOD == "mean"


def test_output_path_default_under_reports() -> None:
    path = output_path_default()
    assert "process_reward" in path.parts


# ── make_row_id determinism ────────────────────────────────────────


def test_make_row_id_deterministic() -> None:
    a = make_row_id("math-algebra", "p", ["s1", "s2"], seed=0)
    b = make_row_id("math-algebra", "p", ["s1", "s2"], seed=0)
    assert a == b
    assert a.startswith("prw_")


def test_make_row_id_diverges_on_steps() -> None:
    a = make_row_id("math-algebra", "p", ["s1"], seed=0)
    b = make_row_id("math-algebra", "p", ["s1", "s2"], seed=0)
    assert a != b


def test_make_row_id_handles_none_env() -> None:
    rid = make_row_id(None, "external", ["a"])
    assert rid.startswith("prw_")
    assert len(rid) == 4 + 16


# ── ProcessRewardTraceRow round-trip ───────────────────────────────


def _sample_row() -> ProcessRewardTraceRow:
    return ProcessRewardTraceRow(
        row_id="prw_" + "0" * 16,
        env_id="math-algebra",
        prompt="2+2=?",
        steps=("Step 1: ...", "Step 2: 4."),
        step_rewards=(0.5, 1.0),
        step_components=({"parse_valid": 1.0}, {"parse_valid": 1.0}),
        step_conformal_intervals=(None, None),
        step_frontier_judgments=(None, None),
        step_frontier_rationales=(None, None),
        step_consensus_rewards=(0.5, 1.0),
        step_disagreements=(None, None),
        aggregate_reward=0.75,
        aggregate_conformal_interval=None,
        decomposition="text_progress",
        segmentation_strategy="explicit_step_marker",
        segmentation_confidence=0.95,
        truncated=False,
        source="env",
        metadata={"schema_version": SCHEMA_VERSION, "seed": 0},
    )


def test_row_round_trip_via_dict() -> None:
    row = _sample_row()
    d = row.to_dict()
    restored = ProcessRewardTraceRow.from_dict(d)
    assert restored == row


def test_row_to_dict_serialises_ci_as_lists() -> None:
    row = ProcessRewardTraceRow(
        row_id="prw_x",
        env_id="x",
        prompt="p",
        steps=("a",),
        step_rewards=(0.5,),
        step_components=(None,),
        step_conformal_intervals=((0.4, 0.6),),
        step_frontier_judgments=(None,),
        step_frontier_rationales=(None,),
        step_consensus_rewards=(0.5,),
        step_disagreements=(None,),
        aggregate_reward=0.5,
        aggregate_conformal_interval=(0.4, 0.6),
        decomposition="text_progress",
        segmentation_strategy="single_step",
        segmentation_confidence=0.3,
        truncated=False,
        source="env",
    )
    d = row.to_dict()
    assert d["step_conformal_intervals"] == [[0.4, 0.6]]
    assert d["aggregate_conformal_interval"] == [0.4, 0.6]


def test_row_step_count_property() -> None:
    row = _sample_row()
    assert row.step_count == 2


# ── collect_env_traces ─────────────────────────────────────────────


def test_collect_env_traces_basic() -> None:
    rows = collect_env_traces(["math-algebra"], n_per_env=2, max_steps=8)
    assert len(rows) == 2
    for row in rows:
        assert row.env_id == "math-algebra"
        assert row.source == "env"
        assert row.step_count >= 1
        assert 0.0 <= row.aggregate_reward <= 1.0
        assert row.metadata.get("schema_version") == SCHEMA_VERSION


def test_collect_env_traces_seed_deterministic() -> None:
    a = collect_env_traces(["math-algebra"], n_per_env=3, seed_start=100)
    b = collect_env_traces(["math-algebra"], n_per_env=3, seed_start=100)
    assert [r.row_id for r in a] == [r.row_id for r in b]


def test_collect_env_traces_zero_returns_empty() -> None:
    rows = collect_env_traces(["math-algebra"], n_per_env=0)
    assert rows == []


def test_collect_env_traces_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        collect_env_traces(["math-algebra"], n_per_env=-1)


def test_collect_env_traces_continue_on_error_drops_failed() -> None:
    """Broken trace_source on odd seeds → those rows drop silently."""

    def broken(env_id, env, instance, seed):
        if seed % 2 == 1:
            raise RuntimeError("synthetic failure")
        from verifiable_labs_envs.process_reward.dataset import (
            baseline_trace_source,
        )

        return baseline_trace_source(env_id, env, instance, seed)

    rows = collect_env_traces(
        ["math-algebra"], n_per_env=4, trace_source=broken
    )
    assert len(rows) == 2


def test_collect_env_traces_fail_fast_propagates() -> None:
    def boom(env_id, env, instance, seed):
        raise RuntimeError("explode")

    with pytest.raises(RuntimeError, match="explode"):
        collect_env_traces(
            ["math-algebra"],
            n_per_env=2,
            trace_source=boom,
            fail_fast=True,
        )


def test_collect_env_traces_on_error_callback() -> None:
    captured: list[tuple[str, int, str]] = []

    def boom(env_id, env, instance, seed):
        raise RuntimeError(f"err-{seed}")

    rows = collect_env_traces(
        ["math-algebra"],
        n_per_env=2,
        trace_source=boom,
        on_error=lambda env_id, seed, exc: captured.append(
            (env_id, seed, str(exc))
        ),
    )
    assert rows == []
    assert len(captured) == 2


def test_collect_env_traces_multiple_envs() -> None:
    rows = collect_env_traces(
        ["math-algebra", "math-algebra"], n_per_env=1
    )
    # Same env twice, n=1 each → 2 rows.
    assert len(rows) == 2


# ── extend_from_phase29_rows ───────────────────────────────────────


def _phase29_row(reward: float, idx: int) -> RewardTrainingRow:
    return RewardTrainingRow(
        row_id=f"rwd_{idx:016x}",
        env_id="math-algebra",
        prompt=f"prompt-{idx}",
        completion=f"Step 1: A.\nStep 2: B.\nStep 3: C-{idx}",
        env_reward=reward,
        env_components={"parse_valid": 1.0, "correct": reward},
        conformal_interval=None,
        frontier_judgment=None,
        frontier_rationale=None,
        consensus_reward=reward,
        disagreement=None,
        source="env",
        metadata={"seed": idx, "schema_version": "v0.1.0"},
    )


def test_extend_from_phase29_rows_segments_completion() -> None:
    rows = extend_from_phase29_rows([_phase29_row(0.7, 0), _phase29_row(0.5, 1)])
    assert len(rows) == 2
    for r in rows:
        assert r.step_count == 3
        assert r.segmentation_strategy == "explicit_step_marker"


def test_extend_from_phase29_rows_propagates_metadata() -> None:
    rows = extend_from_phase29_rows([_phase29_row(0.7, 5)])
    assert rows[0].metadata.get("seed") == 5


def test_extend_from_phase29_rows_handles_empty_input() -> None:
    rows = extend_from_phase29_rows([])
    assert rows == []


# ── JSONL IO ───────────────────────────────────────────────────────


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [_sample_row(), _sample_row()]
    path = tmp_path / "rows.jsonl"
    n = write_jsonl(rows, path)
    assert n == 2
    restored = read_jsonl(path)
    assert restored == rows


def test_jsonl_byte_stable_on_repeat_write(tmp_path: Path) -> None:
    rows = collect_env_traces(["math-algebra"], n_per_env=2)
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(rows, a)
    write_jsonl(rows, b)
    assert a.read_bytes() == b.read_bytes()


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with_blanks.jsonl"
    payload = json.dumps(_sample_row().to_dict(), sort_keys=True)
    path.write_text(f"\n{payload}\n\n", encoding="utf-8")
    rows = read_jsonl(path)
    assert len(rows) == 1


def test_jsonl_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.jsonl"
    write_jsonl([_sample_row()], nested)
    assert nested.exists()


def test_merge_jsonl_concatenates(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    rows_a = collect_env_traces(["math-algebra"], n_per_env=2, seed_start=0)
    rows_b = collect_env_traces(["math-algebra"], n_per_env=2, seed_start=100)
    write_jsonl(rows_a, a)
    write_jsonl(rows_b, b)
    merged = merge_jsonl([a, b])
    assert merged == [*rows_a, *rows_b]


# ── summary + helpers ──────────────────────────────────────────────


def test_dataset_summary_empty() -> None:
    summary = trace_dataset_summary([])
    assert summary["n_traces"] == 0
    assert summary["schema_version"] == SCHEMA_VERSION


def test_dataset_summary_aggregates_per_env() -> None:
    rows = collect_env_traces(["math-algebra"], n_per_env=3)
    summary = trace_dataset_summary(rows)
    assert summary["n_traces"] == 3
    assert summary["by_env"] == {"math-algebra": 3}
    assert summary["by_source"] == {"env": 3}


def test_dataset_summary_step_count_mean() -> None:
    rows = collect_env_traces(["math-algebra"], n_per_env=2)
    summary = trace_dataset_summary(rows)
    assert summary["step_count_mean"] >= 1.0


def test_default_train_envs_returns_22() -> None:
    envs = default_train_envs()
    assert len(envs) == len(DEFAULT_TRAINING_ENVS)
    assert "math-algebra" in envs


def test_is_held_out_predicate() -> None:
    assert is_held_out("sql-multiturn") is True
    assert is_held_out("math-algebra") is False
    assert is_held_out(None) is False


# ── frontier gate ──────────────────────────────────────────────────


def test_frontier_gate_default_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLABS_PHASE30_COLLECT_FRONTIER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert is_phase30_collect_frontier_enabled() is False


def test_frontier_gate_requires_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLABS_PHASE30_COLLECT_FRONTIER", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert is_phase30_collect_frontier_enabled() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert is_phase30_collect_frontier_enabled() is True


def test_frontier_gate_rejects_falsy_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("VLABS_PHASE30_COLLECT_FRONTIER", value)
        assert is_phase30_collect_frontier_enabled() is False
