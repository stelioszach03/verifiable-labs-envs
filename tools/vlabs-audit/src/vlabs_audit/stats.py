"""Aggregate-statistics layer for vlabs-audit.

Reads JSONL trace files written by the SDK's ``verifiable run`` (one
:class:`verifiable_labs_envs.traces.Trace` per line) plus the corresponding
``audit_runs`` rows from :class:`vlabs_audit.storage.AuditStore`, and
produces a :class:`AuditStats` summary that the LaTeX renderer (17.E)
can consume.

Metrics
-------
Per environment:

* **mean reward** with a percentile-bootstrap 95 % CI (1 000 resamples,
  deterministic via a fixed RNG seed).
* **parse-failure rate** — fraction of episodes where the model output
  could not be parsed (DB-status ``failed`` or trace
  ``parse_success=False`` / ``failure_type=parse_error|invalid_*``).
* **format-validity rate** — fraction whose ``reward_components.format_valid``
  is truthy. When the field is absent we fall back to "parsed OK".
* **held-out coverage** — mean of the ``coverage`` field over the
  *second* half of the (seed-sorted) traces. The first half plays the
  role of an implicit calibration fold; reporting coverage on the
  held-out half is a sanity check that the SDK's per-episode conformal
  interval is still hitting target on data the audit hasn't trained on.

Across environments we report the same metrics aggregated over every
trace (cross-env mean reward, etc.).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from vlabs_audit.storage import AuditRunRecord, AuditStore

# Failure-type strings that the SDK emits when the model output could not
# be turned into a valid prediction. See
# ``verifiable_labs_envs.traces.FailureType``.
_PARSE_FAILURE_TYPES: frozenset[str] = frozenset(
    {"parse_error", "invalid_shape", "invalid_json"}
)


# ── helpers ───────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(json.loads(stripped))
    return out


def _is_parse_failure(rec: dict[str, Any]) -> bool:
    """``True`` iff the trace represents a parse-side failure."""
    if rec.get("parse_success") is False:
        return True
    ftype = rec.get("failure_type")
    return isinstance(ftype, str) and ftype.lower() in _PARSE_FAILURE_TYPES


def _format_valid(rec: dict[str, Any]) -> bool:
    """Format validity. Prefers explicit ``reward_components.format_valid``."""
    components = rec.get("reward_components") or {}
    if isinstance(components, dict) and "format_valid" in components:
        return bool(components["format_valid"])
    # Fallback: a parsed trace is treated as format-valid.
    return not _is_parse_failure(rec)


def _trace_coverage(rec: dict[str, Any]) -> float | None:
    """Empirical coverage for one trace.

    Prefer the SDK's top-level ``coverage`` field; fall back to
    ``reward_components.conformal`` (the per-episode conformal hit, 0/1)
    which the SDK populates for envs that surface conformal intervals.
    """
    cov = rec.get("coverage")
    if cov is not None:
        try:
            return float(cov)
        except (TypeError, ValueError):
            return None
    components = rec.get("reward_components") or {}
    if isinstance(components, dict) and "conformal" in components:
        try:
            return float(components["conformal"])
        except (TypeError, ValueError):
            return None
    return None


def bootstrap_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    n_resamples: int = 1_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    ``alpha`` is the *miscoverage* level — pass ``0.05`` for a 95 % CI.
    With ``len(values) == 0`` we raise; with ``len(values) == 1`` we
    return ``(values[0], values[0])`` (degenerate but well-defined).
    """
    if len(values) == 0:
        raise ValueError("bootstrap_ci: empty sample")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _holdout_mean(values: Sequence[float]) -> float | None:
    """Mean of the second half of a sequence (50/50 split)."""
    n = len(values)
    if n < 2:
        return None
    holdout = values[n // 2:]
    return float(np.mean(holdout))


# ── pydantic models ───────────────────────────────────────────────────


class EnvStats(BaseModel):
    """Per-env summary of an audit."""

    model_config = ConfigDict(extra="forbid")

    env: str
    n_episodes: int
    n_success: int
    n_failed: int
    mean_reward: float
    ci_low: float
    ci_high: float
    parse_failure_rate: float
    format_valid_rate: float
    coverage_holdout: float | None
    # Raw per-episode rewards for successful runs — used by the figures
    # module to draw distribution histograms. Empty list when n_success == 0.
    rewards: list[float] = []
    # Sum of ``estimated_cost_usd`` across this env's traces. ``0.0`` when
    # the SDK didn't surface cost data; callers must handle that as
    # "missing", not "free".
    total_cost_usd: float = 0.0


class AuditStats(BaseModel):
    """Top-level audit summary, ready to feed the LaTeX template."""

    model_config = ConfigDict(extra="forbid")

    audit_id: str
    model: str
    alpha: float
    n_episodes_per_env: int
    per_env: list[EnvStats]
    aggregate_mean_reward: float
    aggregate_ci_low: float
    aggregate_ci_high: float
    aggregate_parse_failure_rate: float
    aggregate_format_valid_rate: float
    aggregate_coverage_holdout: float | None


# ── computation ───────────────────────────────────────────────────────


def _compute_env_block(
    env: str,
    runs: Sequence[AuditRunRecord],
    traces: Sequence[dict[str, Any]],
    *,
    alpha: float,
    bootstrap_seed: int,
) -> EnvStats:
    n_total = len(runs)
    n_success = sum(1 for r in runs if r.status == "success")
    n_failed = sum(1 for r in runs if r.status == "failed")

    rewards: list[float] = [
        float(r.reward) for r in runs if r.status == "success" and r.reward is not None
    ]
    if rewards:
        mean_r = float(np.mean(rewards))
        ci_low, ci_high = bootstrap_ci(rewards, alpha=alpha, seed=bootstrap_seed)
    else:
        mean_r = ci_low = ci_high = 0.0

    # Parse-failure rate counts both DB-side failed rows (process-level
    # failures, e.g. timeout) and trace-level parse_error rows.
    trace_parse_fails = sum(1 for t in traces if _is_parse_failure(t))
    parse_fails = n_failed + trace_parse_fails
    parse_failure_rate = parse_fails / n_total if n_total else 0.0

    # Format validity is only meaningful over traces that actually exist.
    if traces:
        fmt_oks = sum(1 for t in traces if _format_valid(t))
        format_valid_rate = fmt_oks / len(traces)
    else:
        format_valid_rate = 0.0

    coverages_sorted: list[float] = []
    for t in sorted(traces, key=lambda t: t.get("seed", 0)):
        c = _trace_coverage(t)
        if c is not None:
            coverages_sorted.append(c)
    coverage_holdout = _holdout_mean(coverages_sorted)

    total_cost_usd = 0.0
    for t in traces:
        cost = t.get("estimated_cost_usd")
        if cost is None:
            continue
        try:
            total_cost_usd += float(cost)
        except (TypeError, ValueError):
            continue

    return EnvStats(
        env=env,
        n_episodes=n_total,
        n_success=n_success,
        n_failed=n_failed,
        mean_reward=mean_r,
        ci_low=ci_low,
        ci_high=ci_high,
        parse_failure_rate=parse_failure_rate,
        format_valid_rate=format_valid_rate,
        coverage_holdout=coverage_holdout,
        rewards=list(rewards),
        total_cost_usd=total_cost_usd,
    )


def _load_traces_for_env(runs: Sequence[AuditRunRecord]) -> list[dict[str, Any]]:
    """Read every JSONL trace for the successful runs in ``runs``."""
    traces: list[dict[str, Any]] = []
    for r in runs:
        if r.status != "success" or not r.jsonl_path:
            continue
        path = Path(r.jsonl_path)
        if not path.exists():
            continue
        traces.extend(_read_jsonl(path))
    return traces


def compute_audit_stats(
    store: AuditStore,
    audit_id: str,
    *,
    alpha: float = 0.1,
    bootstrap_seed: int = 42,
) -> AuditStats:
    """Compute per-env + aggregate stats for one audit.

    ``alpha`` is forwarded both to the conformal interpretation in the
    report and to the bootstrap confidence-interval miscoverage. The CI
    is reported on the mean reward; coverage is reported as plain
    fractions, no CI, since we only have one sample mean per env.
    """
    audit = store.get_audit(audit_id)
    if audit is None:
        raise ValueError(f"unknown audit_id: {audit_id}")

    runs = store.list_runs(audit_id)
    if not runs:
        raise ValueError(f"audit {audit_id} has no runs scheduled")

    by_env: dict[str, list[AuditRunRecord]] = {}
    for r in runs:
        by_env.setdefault(r.env, []).append(r)

    per_env: list[EnvStats] = []
    all_rewards: list[float] = []
    all_traces: list[dict[str, Any]] = []
    n_failed_total = 0

    for env in sorted(by_env):
        env_runs = by_env[env]
        traces = _load_traces_for_env(env_runs)
        block = _compute_env_block(
            env, env_runs, traces, alpha=alpha, bootstrap_seed=bootstrap_seed
        )
        per_env.append(block)
        all_rewards.extend(
            float(r.reward)
            for r in env_runs
            if r.status == "success" and r.reward is not None
        )
        all_traces.extend(traces)
        n_failed_total += block.n_failed

    if all_rewards:
        agg_mean = float(np.mean(all_rewards))
        agg_lo, agg_hi = bootstrap_ci(all_rewards, alpha=alpha, seed=bootstrap_seed)
    else:
        agg_mean = agg_lo = agg_hi = 0.0

    n_total = len(runs)
    trace_parse_fails = sum(1 for t in all_traces if _is_parse_failure(t))
    aggregate_parse_failure_rate = (
        (n_failed_total + trace_parse_fails) / n_total if n_total else 0.0
    )

    if all_traces:
        fmt_oks = sum(1 for t in all_traces if _format_valid(t))
        aggregate_format_valid_rate = fmt_oks / len(all_traces)
    else:
        aggregate_format_valid_rate = 0.0

    all_coverages_sorted: list[float] = []
    for t in sorted(
        all_traces, key=lambda t: (t.get("env_name", ""), t.get("seed", 0))
    ):
        c = _trace_coverage(t)
        if c is not None:
            all_coverages_sorted.append(c)
    aggregate_coverage_holdout = _holdout_mean(all_coverages_sorted)

    # Per-env episode count is informational. With uneven counts we
    # report the most common (mode) as a representative — the per-env
    # block carries the exact value.
    counts = sorted({len(v) for v in by_env.values()})
    n_episodes_per_env = counts[-1] if counts else 0

    return AuditStats(
        audit_id=audit_id,
        model=audit.model,
        alpha=alpha,
        n_episodes_per_env=n_episodes_per_env,
        per_env=per_env,
        aggregate_mean_reward=agg_mean,
        aggregate_ci_low=agg_lo,
        aggregate_ci_high=agg_hi,
        aggregate_parse_failure_rate=aggregate_parse_failure_rate,
        aggregate_format_valid_rate=aggregate_format_valid_rate,
        aggregate_coverage_holdout=aggregate_coverage_holdout,
    )


# ── CLI-friendly rendering ────────────────────────────────────────────


def format_stats_table(stats: AuditStats) -> str:
    """Pretty-print an :class:`AuditStats` for ``--print-stats`` output.

    The format mirrors ``verifiable compare`` so users moving between
    the SDK and the audit tool see consistent column layouts.
    """
    cov_target = 1.0 - stats.alpha
    lines = [
        f"Audit {stats.audit_id} — model: {stats.model}",
        (
            f"  α = {stats.alpha:.2f}  (target coverage {cov_target:.2f})  ·  "
            f"episodes/env = {stats.n_episodes_per_env}"
        ),
        "",
        f"  {'env':<32} {'mean':>8} {'95% CI':>20} {'parse-fail':>11} "
        f"{'fmt-ok':>8} {'cov-holdout':>13}",
    ]

    def _row(label: str, mean: float, lo: float, hi: float, parse: float, fmt: float, cov: float | None) -> str:
        ci = f"[{lo:.3f}, {hi:.3f}]"
        cov_s = f"{cov:.3f}" if cov is not None else "—"
        return (
            f"  {label:<32} {mean:>8.3f} {ci:>20} "
            f"{parse:>10.1%} {fmt:>7.1%} {cov_s:>13}"
        )

    for es in stats.per_env:
        lines.append(
            _row(
                es.env,
                es.mean_reward,
                es.ci_low,
                es.ci_high,
                es.parse_failure_rate,
                es.format_valid_rate,
                es.coverage_holdout,
            )
        )
    lines.append("")
    lines.append(
        _row(
            "AGGREGATE",
            stats.aggregate_mean_reward,
            stats.aggregate_ci_low,
            stats.aggregate_ci_high,
            stats.aggregate_parse_failure_rate,
            stats.aggregate_format_valid_rate,
            stats.aggregate_coverage_holdout,
        )
    )
    return "\n".join(lines)


__all__ = [
    "AuditStats",
    "EnvStats",
    "bootstrap_ci",
    "compute_audit_stats",
    "format_stats_table",
]
