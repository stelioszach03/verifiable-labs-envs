"""Unit tests for ``vlabs_audit.stats`` — bootstrap CIs + AuditStats."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlabs_audit.stats import (
    AuditStats,
    EnvStats,
    bootstrap_ci,
    compute_audit_stats,
    format_stats_table,
)
from vlabs_audit.storage import AuditStore


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _seed_complete_episodes(
    store: AuditStore,
    *,
    audit_id: str,
    env: str,
    rewards: list[float],
    traces_dir: Path,
    parse_success: bool = True,
    coverage: list[float] | None = None,
    seed_start: int = 0,
) -> None:
    """Schedule + complete N episodes with the given rewards; write JSONL."""
    n = len(rewards)
    store.schedule_episodes(audit_id, env, n, seed_start=seed_start)
    pending = [p for p in store.list_pending(audit_id) if p.env == env]
    assert len(pending) == n, "schedule_episodes did not produce expected pending rows"
    for i, run in enumerate(pending):
        store.mark_running(run.id)
        jp = traces_dir / f"{env}__seed{run.seed}.jsonl"
        rec: dict = {
            "reward": rewards[i],
            "parse_success": parse_success,
            "seed": run.seed,
            "env_name": env,
        }
        if coverage is not None:
            rec["coverage"] = coverage[i]
        _write_jsonl(jp, [rec])
        store.complete_episode(run.id, rewards[i], jp)


# ── tests ────────────────────────────────────────────────────────────


def test_synthetic_reward_mean_within_bootstrap_ci(tmp_path: Path) -> None:
    """Mean of a known reward distribution lies inside the 95 % bootstrap CI."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    rewards = [0.5 + 0.005 * i for i in range(50)]  # mean = 0.6225
    expected_mean = sum(rewards) / len(rewards)

    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {"envs": ["env-a"]})
        _seed_complete_episodes(
            store,
            audit_id=aid,
            env="env-a",
            rewards=rewards,
            traces_dir=traces_dir,
        )
        store.finish_audit(aid)
        stats = compute_audit_stats(store, aid, alpha=0.05)

    assert stats.audit_id == aid
    assert stats.model == "m"
    assert len(stats.per_env) == 1
    es = stats.per_env[0]
    assert es.env == "env-a"
    assert es.n_episodes == 50
    assert es.n_success == 50
    assert es.n_failed == 0
    assert es.mean_reward == pytest.approx(expected_mean, abs=1e-9)
    assert es.ci_low <= expected_mean <= es.ci_high
    # CI should be non-degenerate but tight on n=50.
    assert es.ci_high > es.ci_low
    assert (es.ci_high - es.ci_low) < 0.1


def test_all_zeros_has_zero_variance_ci(tmp_path: Path) -> None:
    """All-zero rewards: mean = 0 and CI collapses to [0, 0]."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {"envs": ["env-a"]})
        _seed_complete_episodes(
            store, audit_id=aid, env="env-a",
            rewards=[0.0] * 10, traces_dir=traces_dir,
        )
        stats = compute_audit_stats(store, aid)

    es = stats.per_env[0]
    assert es.mean_reward == 0.0
    assert es.ci_low == pytest.approx(0.0)
    assert es.ci_high == pytest.approx(0.0)
    assert es.parse_failure_rate == 0.0


def test_all_failed_episodes_parse_rate_one(tmp_path: Path) -> None:
    """Every episode failed at the DB layer → parse-failure rate == 1.0."""
    db = tmp_path / "x.db"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {"envs": ["env-a"]})
        store.schedule_episodes(aid, "env-a", 5, seed_start=0)
        for run in store.list_pending(aid):
            store.mark_running(run.id)
            store.fail_episode(run.id, "RuntimeError: parse_error: invalid JSON")
        stats = compute_audit_stats(store, aid)

    es = stats.per_env[0]
    assert es.n_success == 0
    assert es.n_failed == 5
    assert es.parse_failure_rate == 1.0
    assert es.mean_reward == 0.0
    assert es.ci_low == 0.0 and es.ci_high == 0.0
    # No traces on disk → format-validity rate is 0.0.
    assert es.format_valid_rate == 0.0
    # No coverage data → None.
    assert es.coverage_holdout is None


def test_bootstrap_ci_reproducible_across_seeds() -> None:
    """Same seed → identical bounds; different seed → different bounds."""
    rewards = [0.5, 0.6, 0.55, 0.7, 0.45, 0.65, 0.5, 0.6, 0.4, 0.75]
    a = bootstrap_ci(rewards, alpha=0.05, n_resamples=500, seed=42)
    b = bootstrap_ci(rewards, alpha=0.05, n_resamples=500, seed=42)
    assert a == b
    c = bootstrap_ci(rewards, alpha=0.05, n_resamples=500, seed=43)
    assert c != a
    # Sanity: empirical mean lies inside the CI for any reasonable seed.
    expected = sum(rewards) / len(rewards)
    assert a[0] <= expected <= a[1]


def test_bootstrap_ci_empty_raises_and_singleton_collapses() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])
    lo, hi = bootstrap_ci([0.42])
    assert lo == hi == 0.42


def test_coverage_holdout_matches_manual_computation(tmp_path: Path) -> None:
    """50/50 split: manual mean of the second half == coverage_holdout."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    cov = [1.0, 0.0, 1.0, 1.0]  # second half = [1.0, 1.0] → mean 1.0
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {"envs": ["env-a"]})
        _seed_complete_episodes(
            store, audit_id=aid, env="env-a",
            rewards=[0.5] * 4, traces_dir=traces_dir, coverage=cov,
        )
        stats = compute_audit_stats(store, aid)

    es = stats.per_env[0]
    assert es.coverage_holdout == pytest.approx(1.0)
    assert stats.aggregate_coverage_holdout == pytest.approx(1.0)


def test_audit_stats_round_trips_through_json(tmp_path: Path) -> None:
    """``model_dump`` and ``model_dump_json`` produce LaTeX-template-ready data."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("claude-haiku-4.5", {"envs": ["env-a"]})
        _seed_complete_episodes(
            store, audit_id=aid, env="env-a",
            rewards=[0.7, 0.6, 0.8], traces_dir=traces_dir,
            coverage=[1.0, 1.0, 0.0],
        )
        stats = compute_audit_stats(store, aid)

    payload = stats.model_dump()
    assert payload["audit_id"] == aid
    assert payload["model"] == "claude-haiku-4.5"
    assert isinstance(payload["per_env"], list)
    assert payload["per_env"][0]["env"] == "env-a"
    assert payload["per_env"][0]["n_success"] == 3
    # Pydantic round-trip via JSON.
    serialized = stats.model_dump_json()
    revived = AuditStats.model_validate_json(serialized)
    assert revived == stats
    # Each EnvStats is also independently serialisable.
    assert isinstance(stats.per_env[0], EnvStats)


def test_unknown_or_empty_audit_raises_clear_error(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with AuditStore(db_path=db) as store:
        with pytest.raises(ValueError, match="unknown audit_id"):
            compute_audit_stats(store, "aud_does_not_exist")

        aid = store.create_audit("m", {})
        # No schedule_episodes yet → no audit_runs rows.
        with pytest.raises(ValueError, match="no runs"):
            compute_audit_stats(store, aid)


def test_cross_env_aggregation_with_uneven_episode_counts(tmp_path: Path) -> None:
    """Aggregate mean is the unweighted mean over every successful episode."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {"envs": ["env-a", "env-b"]})
        # 3 episodes of reward 0.8 + 5 episodes of reward 0.4
        # → aggregate mean = (3*0.8 + 5*0.4) / 8 = 4.4/8 = 0.55
        _seed_complete_episodes(
            store, audit_id=aid, env="env-a",
            rewards=[0.8] * 3, traces_dir=traces_dir, seed_start=0,
        )
        _seed_complete_episodes(
            store, audit_id=aid, env="env-b",
            rewards=[0.4] * 5, traces_dir=traces_dir, seed_start=100,
        )
        stats = compute_audit_stats(store, aid)

    by_env = {es.env: es for es in stats.per_env}
    assert by_env["env-a"].n_episodes == 3
    assert by_env["env-b"].n_episodes == 5
    assert stats.aggregate_mean_reward == pytest.approx(0.55, abs=1e-9)
    assert stats.aggregate_ci_low <= 0.55 <= stats.aggregate_ci_high


def test_coverage_falls_back_to_reward_components_conformal(tmp_path: Path) -> None:
    """When the SDK omits top-level ``coverage`` we should read the per-episode
    conformal hit out of ``reward_components.conformal``."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {"envs": ["env-a"]})
        store.schedule_episodes(aid, "env-a", 4, seed_start=0)
        # 4 traces; second-half conformal = [1.0, 0.0] → mean 0.5
        conformal_values = [1.0, 1.0, 1.0, 0.0]
        for i, run in enumerate(store.list_pending(aid)):
            store.mark_running(run.id)
            jp = traces_dir / f"env-a__seed{run.seed}.jsonl"
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(
                json.dumps({
                    "reward": 0.5,
                    "parse_success": True,
                    "seed": run.seed,
                    "env_name": "env-a",
                    "reward_components": {"conformal": conformal_values[i]},
                }) + "\n"
            )
            store.complete_episode(run.id, 0.5, jp)
        stats = compute_audit_stats(store, aid)

    es = stats.per_env[0]
    assert es.coverage_holdout == pytest.approx(0.5)


def test_format_stats_table_renders_all_envs(tmp_path: Path) -> None:
    """``format_stats_table`` produces a header + one row per env + AGGREGATE."""
    db = tmp_path / "x.db"
    traces_dir = tmp_path / "traces"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("claude-haiku-4.5", {"envs": ["env-a", "env-b"]})
        _seed_complete_episodes(
            store, audit_id=aid, env="env-a",
            rewards=[0.7] * 3, traces_dir=traces_dir, seed_start=0,
        )
        _seed_complete_episodes(
            store, audit_id=aid, env="env-b",
            rewards=[0.3] * 3, traces_dir=traces_dir, seed_start=100,
        )
        stats = compute_audit_stats(store, aid, alpha=0.1)
        table = format_stats_table(stats)

    assert "claude-haiku-4.5" in table
    assert "env-a" in table
    assert "env-b" in table
    assert "AGGREGATE" in table
    assert "α = 0.10" in table or "alpha" in table.lower()
    # CI brackets render with three decimals.
    assert "0.700" in table or "0.7" in table
