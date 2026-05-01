"""Unit tests for ``vlabs_audit.anonymize`` — model-id substitution."""
from __future__ import annotations

import pytest

from vlabs_audit.anonymize import (
    DEFAULT_LABELS,
    anonymize_audit_stats,
    parse_anonymize_labels,
    resolve_anonymize_label,
)
from vlabs_audit.latex import render_tex
from vlabs_audit.stats import AuditStats, EnvStats


def _mk_stats(model: str = "anthropic/claude-haiku-4.5") -> AuditStats:
    es = EnvStats(
        env="sparse-fourier-recovery",
        n_episodes=3, n_success=3, n_failed=0,
        mean_reward=0.5, ci_low=0.45, ci_high=0.55,
        parse_failure_rate=0.0, format_valid_rate=1.0,
        coverage_holdout=0.95,
        rewards=[0.45, 0.50, 0.55], total_cost_usd=0.005,
    )
    return AuditStats(
        audit_id="aud_anon_1234",
        model=model,
        alpha=0.1,
        n_episodes_per_env=3,
        per_env=[es],
        aggregate_mean_reward=0.5,
        aggregate_ci_low=0.45,
        aggregate_ci_high=0.55,
        aggregate_parse_failure_rate=0.0,
        aggregate_format_valid_rate=1.0,
        aggregate_coverage_holdout=0.95,
    )


# ── parse_anonymize_labels ────────────────────────────────────────────


def test_parse_anonymize_labels_handles_csv_whitespace_and_empty() -> None:
    assert parse_anonymize_labels(None) is None
    assert parse_anonymize_labels("") is None
    assert parse_anonymize_labels("   ") is None
    assert parse_anonymize_labels("A") == ("A",)
    assert parse_anonymize_labels("A,B,C") == ("A", "B", "C")
    assert parse_anonymize_labels("  Frontier Model A , Frontier Model B  ") == (
        "Frontier Model A",
        "Frontier Model B",
    )
    # Empty entries are dropped.
    assert parse_anonymize_labels("A,,B,") == ("A", "B")


# ── anonymize_audit_stats ─────────────────────────────────────────────


def test_anonymize_audit_stats_returns_new_object_with_replaced_model() -> None:
    stats = _mk_stats(model="anthropic/claude-haiku-4.5")
    anon = anonymize_audit_stats(stats, label="Frontier Model A")

    # Returned copy carries the label.
    assert anon.model == "Frontier Model A"
    # Original is unchanged (no mutation).
    assert stats.model == "anthropic/claude-haiku-4.5"
    # Every other field is preserved.
    assert anon.audit_id == stats.audit_id
    assert anon.alpha == stats.alpha
    assert anon.per_env == stats.per_env
    assert anon.aggregate_mean_reward == stats.aggregate_mean_reward


def test_anonymize_audit_stats_strips_label_whitespace() -> None:
    stats = _mk_stats()
    anon = anonymize_audit_stats(stats, label="   Frontier Model A   ")
    assert anon.model == "Frontier Model A"


def test_anonymize_audit_stats_rejects_empty_label() -> None:
    stats = _mk_stats()
    with pytest.raises(ValueError, match="non-empty"):
        anonymize_audit_stats(stats, label="")
    with pytest.raises(ValueError, match="non-empty"):
        anonymize_audit_stats(stats, label="   ")


# ── resolve_anonymize_label priority ──────────────────────────────────


def test_resolve_anonymize_label_priority_cli_then_config_then_default() -> None:
    # CLI labels win over config + default.
    assert resolve_anonymize_label(
        explicit_labels=("CLI Label",),
        config_label="Config Label",
    ) == "CLI Label"
    # No CLI → config wins.
    assert resolve_anonymize_label(
        explicit_labels=None, config_label="Config Label"
    ) == "Config Label"
    # No CLI, blank config → default.
    assert resolve_anonymize_label(
        explicit_labels=None, config_label=None
    ) == DEFAULT_LABELS[0]
    assert resolve_anonymize_label(
        explicit_labels=None, config_label="   "
    ) == DEFAULT_LABELS[0]
    # Empty CLI tuple is treated like None — config still wins.
    assert resolve_anonymize_label(
        explicit_labels=(), config_label="Config Label"
    ) == "Config Label"


# ── end-to-end: rendered LaTeX has no real model id ───────────────────


def test_anonymized_stats_render_replaces_model_throughout_tex() -> None:
    """``render_tex`` on an anonymised stats object emits no real model id."""
    real_model = "anthropic/claude-haiku-4.5"
    stats = _mk_stats(model=real_model)
    anon = anonymize_audit_stats(stats, label="Frontier Model A")
    tex = render_tex(anon)

    # The label is present in cover, headers, citation, repro table.
    assert "Frontier Model A" in tex
    # Every form of the real id is gone.
    forbidden = [
        "claude-haiku-4.5",
        "anthropic/claude-haiku",
        "anthropic/",
        "haiku",
        "claude",
        "anthropic",
    ]
    for needle in forbidden:
        assert needle.lower() not in tex.lower(), (
            f"real-model fragment {needle!r} leaked into rendered TeX"
        )
