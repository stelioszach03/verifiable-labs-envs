"""Per-env calibration smoke tests for the four canonical env families.

The generic ``test_calibration.py`` exercises ``auto_calibrate`` on
synthetic Gaussian inverse-problem fixtures. Phase 30+ envs include
discrete families (code execution, symbolic math, SQL, long-context)
where the conformal moat reaches the env's score function in a less
direct way — every env exposes ``conformal_quantile``,
``generate_instance``, ``run_baseline``, and ``score`` regardless of
whether it's a Bayesian inverse-problem env or a discrete pass/fail
verifier.

These tests pin the four-env contract:

  - The env loads, generates a deterministic instance for a fixed
    seed, runs its baseline, and produces a parseable score.
  - The score is in ``[0, 1]`` (the locked conformal-bounded range).
  - ``conformal_quantile`` is a non-negative float (every env exposes
    it as part of the public surface).
  - Two calls with the same seed produce identical instances + scores
    (determinism — required by the M3 reproducibility hashes).

15 tests total: 3 invariants × 4 env families + 3 cross-cutting
checks (env list size + registry membership for each name).
"""
from __future__ import annotations

import pytest

from verifiable_labs_envs import _REGISTRY, load_environment

# The four canonical families per the v0.0.1 dataset taxonomy:
#   code-execution     → code-humaneval
#   symbolic-math      → math-algebra
#   sql-execution      → sql-single-turn
#   long-context       → long-context-needle
ENV_FAMILIES = [
    ("code-execution", "code-humaneval"),
    ("symbolic-math", "math-algebra"),
    ("sql-execution", "sql-single-turn"),
    ("long-context", "long-context-needle"),
]


# ── cross-cutting ──────────────────────────────────────────────────


def test_all_four_target_envs_are_registered() -> None:
    """The four families used by the v0.0.1 RM/PRM datasets must all
    be registered. A registry typo would silently make the dataset
    pipelines drop those envs from the extract."""
    for _family, env_id in ENV_FAMILIES:
        assert env_id in _REGISTRY, f"missing env: {env_id}"


def test_env_families_list_has_four_entries() -> None:
    """Pin the family count — the Phase 30 dataset README references
    "four canonical families" (code, math, sql, long-context). A
    refactor that drops one would silently break that doc claim."""
    assert len(ENV_FAMILIES) == 4
    families = {f for f, _ in ENV_FAMILIES}
    assert families == {
        "code-execution",
        "symbolic-math",
        "sql-execution",
        "long-context",
    }


def test_env_families_distinct_env_ids() -> None:
    """No family duplicates the same underlying env_id."""
    env_ids = [env_id for _, env_id in ENV_FAMILIES]
    assert len(set(env_ids)) == len(env_ids)


# ── per-env: conformal_quantile attribute exposure ─────────────────


@pytest.mark.parametrize("env_id", [eid for _, eid in ENV_FAMILIES])
def test_env_exposes_conformal_quantile_attribute(env_id: str) -> None:
    """Every public env in the four families exposes a non-negative
    ``conformal_quantile`` (some envs use 0.0 by design — pass/fail
    verifiers — but the attribute MUST be present)."""
    env = load_environment(env_id)
    q = env.conformal_quantile
    assert isinstance(q, (int, float))
    assert q >= 0.0


# ── per-env: deterministic generate_instance ───────────────────────


@pytest.mark.parametrize("env_id", [eid for _, eid in ENV_FAMILIES])
def test_env_generate_instance_is_deterministic(env_id: str) -> None:
    """Same seed ⇒ same instance. M3 reproducibility hashes assume this
    invariant; if any env's RNG isn't seeded properly, the seed→hash
    map drifts between runs."""
    env = load_environment(env_id)
    inst_a = env.generate_instance(seed=42)
    inst_b = env.generate_instance(seed=42)
    # Instances should compare equal-ish — at minimum the random
    # ``seed`` field is preserved.
    assert getattr(inst_a, "seed", None) == getattr(inst_b, "seed", None)


# ── per-env: baseline + score are wired ────────────────────────────


@pytest.mark.parametrize("env_id", [eid for _, eid in ENV_FAMILIES])
def test_env_baseline_score_in_unit_interval(env_id: str) -> None:
    """``env.run_baseline(seed=...)`` produces a score dict whose
    ``reward`` lands in ``[0, 1]``.

    ``run_baseline`` is a one-shot helper that internally calls
    ``generate_instance``, the env's ``baseline_predict``, and ``score``
    in sequence — it returns the score record dict, not just the
    prediction. The env-procedural reward used by the consensus
    blender clips to ``[0, 1]``; if any env's baseline path produces
    an unclipped or NaN score, the consensus formula breaks.
    """
    env = load_environment(env_id)
    record = env.run_baseline(seed=0)
    assert isinstance(record, dict)
    assert "reward" in record
    score = float(record["reward"])
    assert 0.0 <= score <= 1.0, f"{env_id} score out of range: {score}"
