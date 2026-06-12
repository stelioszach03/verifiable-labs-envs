"""Best-of-N reranking utility (Phase 30.D, plan §5 D6-B / D11-1).

Given a set of N candidate completions for the same prompt, the PRM
scores each, then picks the highest-aggregate. This is the standard
math/code RL pipeline pattern; PRM beats outcome RM here because
per-step granularity catches mid-trace divergences that an outcome
RM misses (the final answer might still be lucky).

Public surface:

- :class:`BonCandidate` — input shape: ``(prompt, completion, env_reward)``.
  ``env_reward`` is the ground-truth outcome (when available) used to
  measure win-rate vs single-completion baseline.
- :class:`BonResult` — per-prompt outcome including the chosen index +
  the chosen aggregate reward.
- :func:`rerank_bon` — pick the best of N.
- :func:`bon_lift_metrics` — aggregate win-rate + improvement vs
  outcome-only reranking baseline (D6-B headline metric).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BonCandidate:
    """One BoN candidate — prompt + reasoning trace + ground-truth outcome.

    ``env_reward`` is the env's terminal reward for the candidate
    (when available); used by :func:`bon_lift_metrics` to compute
    win-rate vs single-completion accuracy.
    """

    prompt: str
    steps: tuple[str, ...]
    env_reward: float | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class BonResult:
    """Outcome of one BoN reranking decision."""

    chosen_index: int
    chosen_aggregate: float
    chosen_env_reward: float | None
    all_aggregates: tuple[float, ...]
    n_candidates: int


def rerank_bon(
    candidates: Sequence[BonCandidate],
    aggregate_predictor: Callable[[str, Sequence[str]], float],
) -> BonResult:
    """Pick the candidate with the highest aggregate reward.

    ``aggregate_predictor`` is the trace-level callable matching
    ``(prompt, steps) -> aggregate_reward`` (e.g.
    :func:`verifiable_labs_envs.process_reward.inference.stub_aggregate_predictor`
    for the 30.D test path). Tied candidates resolve to the lowest
    index for determinism.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    aggregates: list[float] = []
    for cand in candidates:
        agg = float(aggregate_predictor(cand.prompt, cand.steps))
        aggregates.append(agg)

    chosen = 0
    best = aggregates[0]
    for i, agg in enumerate(aggregates[1:], start=1):
        if agg > best:
            chosen = i
            best = agg
    return BonResult(
        chosen_index=chosen,
        chosen_aggregate=best,
        chosen_env_reward=candidates[chosen].env_reward,
        all_aggregates=tuple(aggregates),
        n_candidates=len(candidates),
    )


def bon_lift_metrics(
    problems: Sequence[Sequence[BonCandidate]],
    *,
    prm_aggregate_predictor: Callable[[str, Sequence[str]], float],
    rm_predictor: Callable[[str, str], float] | None = None,
    correct_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the headline D6-B BoN-lift metrics across a set of
    problems.

    For each problem (a list of N BonCandidate sharing the same
    prompt), measure:

    - **single_accuracy**: fraction of first-candidate env_rewards
      ≥ ``correct_threshold`` (single-completion baseline).
    - **prm_bon_accuracy**: fraction of PRM-reranked-best env_rewards
      ≥ ``correct_threshold``.
    - **rm_bon_accuracy** (optional, when ``rm_predictor`` supplied):
      fraction reranked by the Phase 29 RM where the joined trace is
      passed as the "completion" string.
    - **prm_bon_lift_vs_single**: PRM accuracy − single accuracy.
    - **prm_bon_lift_vs_rm**: PRM accuracy − RM accuracy. Only present
      when ``rm_predictor`` is supplied. Pass criterion: ≥ +5pp on
      held-out math-algebra (per :doc:`PHASE_30_PLAN.md` §5 D6).

    Candidates with ``env_reward=None`` are skipped from the
    aggregation; problems with all-None env_rewards contribute zero.
    """
    if not problems:
        return {
            "n_problems": 0,
            "single_accuracy": 0.0,
            "prm_bon_accuracy": 0.0,
            "prm_bon_lift_vs_single": 0.0,
        }

    single_hits = 0
    prm_hits = 0
    rm_hits = 0
    n_scored = 0

    for cands in problems:
        if not cands:
            continue
        first = cands[0]
        if first.env_reward is None:
            continue

        # Single-completion baseline = first candidate's outcome.
        if float(first.env_reward) >= correct_threshold:
            single_hits += 1

        # PRM-reranked best.
        prm_choice = rerank_bon(cands, prm_aggregate_predictor)
        if (
            prm_choice.chosen_env_reward is not None
            and float(prm_choice.chosen_env_reward) >= correct_threshold
        ):
            prm_hits += 1

        # Optional Phase 29 RM baseline reranking.
        if rm_predictor is not None:
            rm_aggregates = [
                float(rm_predictor(c.prompt, "\n".join(c.steps)))
                for c in cands
            ]
            best = max(range(len(cands)), key=lambda i: rm_aggregates[i])
            chosen_env = cands[best].env_reward
            if chosen_env is not None and float(chosen_env) >= correct_threshold:
                rm_hits += 1

        n_scored += 1

    if n_scored == 0:
        return {
            "n_problems": float(len(problems)),
            "n_scored": 0.0,
            "single_accuracy": 0.0,
            "prm_bon_accuracy": 0.0,
            "prm_bon_lift_vs_single": 0.0,
        }

    single_acc = single_hits / n_scored
    prm_acc = prm_hits / n_scored
    out = {
        "n_problems": float(len(problems)),
        "n_scored": float(n_scored),
        "single_accuracy": single_acc,
        "prm_bon_accuracy": prm_acc,
        "prm_bon_lift_vs_single": prm_acc - single_acc,
    }
    if rm_predictor is not None:
        rm_acc = rm_hits / n_scored
        out["rm_bon_accuracy"] = rm_acc
        out["prm_bon_lift_vs_rm"] = prm_acc - rm_acc
    return out


def passes_bon_lift_floor(metrics: dict[str, float], *, floor: float = 0.05) -> bool:
    """Pass criterion (Plan §5 D6): PRM BoN lift over Phase 29 RM
    baseline ≥ ``floor`` (default +5 pp). Returns False when the
    ``prm_bon_lift_vs_rm`` field is missing."""
    if "prm_bon_lift_vs_rm" not in metrics:
        return False
    return float(metrics["prm_bon_lift_vs_rm"]) >= float(floor)


def make_synthetic_bon_problems(
    n_problems: int = 10,
    *,
    n_per_problem: int = 4,
    seed: int = 0,
) -> list[list[BonCandidate]]:
    """Deterministic synthetic BoN problem generator for tests + the
    eval CLI smoke path.

    Each problem has ``n_per_problem`` candidates whose env_rewards
    span ``[0.1, 0.9]`` deterministically; the highest-env-reward
    candidate is the "correct" one. A perfect PRM ranks it first;
    a noisy PRM picks anywhere.
    """
    import numpy as np

    if n_problems < 0 or n_per_problem < 1:
        raise ValueError(
            f"n_problems must be ≥0 and n_per_problem must be ≥1; "
            f"got {n_problems}, {n_per_problem}"
        )

    rng = np.random.default_rng(seed)
    problems: list[list[BonCandidate]] = []
    for p in range(n_problems):
        prompt = f"Problem #{p:04d}: choose the correct answer."
        rewards = sorted(
            float(x) for x in rng.uniform(0.1, 0.9, size=n_per_problem)
        )
        # Shuffle the order so the "correct" candidate isn't always at idx 0.
        order = list(range(n_per_problem))
        rng.shuffle(order)
        cands: list[BonCandidate] = []
        for i in order:
            r = rewards[i]
            cands.append(
                BonCandidate(
                    prompt=prompt,
                    steps=(
                        f"Step 1 of candidate {i}",
                        f"Step 2 toward answer (reward {r:.2f})",
                    ),
                    env_reward=r,
                )
            )
        problems.append(cands)
    return problems


__all__ = [
    "BonCandidate",
    "BonResult",
    "bon_lift_metrics",
    "make_synthetic_bon_problems",
    "passes_bon_lift_floor",
    "rerank_bon",
]
