"""Helper module for ``training_proof.ipynb``.

Splits the experiment into testable, deterministic pieces so the notebook
itself stays a thin orchestration shell. The unit tests in
``tests/notebooks/`` exercise this module against ``FakeLLMSolver`` so we
can verify the prompt-search / paired-bootstrap logic without spending
OpenRouter credits.

Public surface:
- :class:`PromptCandidate` — name + system prompt + provenance label
- :func:`evaluate_prompt` — runs the env on N seeds with one prompt
- :class:`BudgetCap` — guards a hard USD cap on a session
- :func:`paired_bootstrap_ci` — paired-difference 95 % CI
- :data:`DEFAULT_CANDIDATES` — the 4 candidate system prompts the
  notebook tests by default. Hand-crafted variations of the env's
  default ``SYSTEM_PROMPT_MT`` plus one LLM-rewritten variant.

The "training" is a tournament-style prompt search: evaluate each
candidate on a small held-back validation set, pick the best, then run
the unseen held-out evaluation. This is the brief's "Plan B" simpler
optimiser path — DSPy's ``BootstrapFewShot`` requires its own client
shim against OpenRouter that we deliberately avoid.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# ── Prompt candidates ──────────────────────────────────────────


@dataclass(frozen=True)
class PromptCandidate:
    """A system-prompt variant the notebook will evaluate."""

    name: str
    system_prompt: str
    provenance: str = "hand-crafted"


# Hand-crafted variations of ``SYSTEM_PROMPT_MT``. Each prompt
# expresses the same task but emphasises a different aspect — terse
# vs. verbose, with vs. without a process hint, with vs. without an
# explicit residual-direction reminder. The "rewritten" candidate is
# how an LLM would re-phrase the brief if asked to make it more
# directive (we generated this offline and committed it verbatim so
# the notebook stays reproducible without an extra API call).
DEFAULT_CANDIDATES: list[PromptCandidate] = [
    PromptCandidate(
        name="terse-baseline",
        system_prompt=(
            "Sparse Fourier recovery. Up to 3 turns. Output one JSON "
            "object with keys 'support_idx' (k sorted ints in [0, n)) "
            "and 'support_amp_x1000' (k signed ints, same order). "
            "On turns 2-3 you receive the residual r = y - A(x_hat) "
            "of your previous answer (re/im, x1000); shrink it. "
            "No prose, no markdown."
        ),
        provenance="hand-crafted-terse",
    ),
    PromptCandidate(
        name="verbose-with-strategy",
        system_prompt=(
            "You are an expert at sparse signal recovery (compressed "
            "sensing). The forward operator is the partial Fourier "
            "transform on a known mask; you observe y = A(x) + noise.\n\n"
            "TASK: identify the k non-zero positions of x and their real "
            "amplitudes (scaled x1000 to integers).\n\n"
            "STRATEGY: use the residual to MOVE indices, not just "
            "rescale. If a residual coefficient is large at index m, "
            "consider whether x_hat is missing a support point that "
            "would zero it out.\n\n"
            "FORMAT: exactly one JSON object with keys 'support_idx' "
            "(k sorted ints in [0, n)) and 'support_amp_x1000' (k signed "
            "ints, same order). No prose."
        ),
        provenance="hand-crafted-verbose",
    ),
    PromptCandidate(
        name="step-numbered",
        system_prompt=(
            "Sparse Fourier recovery, 3 turns max. Schema: one JSON "
            "object {'support_idx': k sorted ints in [0,n), "
            "'support_amp_x1000': k signed ints, same order}.\n\n"
            "Turn 1 — use the mask + observed y to guess the k strongest "
            "frequency components.\n"
            "Turn 2 — read the residual r=y-A(x_hat). For each large "
            "|r_m|, consider swapping a support index.\n"
            "Turn 3 — fine-tune amplitudes; only swap support if "
            "the residual at that index is still large.\n\n"
            "Output: JSON only, no prose, no fences."
        ),
        provenance="hand-crafted-step-numbered",
    ),
    PromptCandidate(
        name="rewritten-directive",
        system_prompt=(
            "You solve k-sparse Fourier recovery. INPUT each turn: "
            "n (signal length), k (sparsity), sigma (noise scale), "
            "mask (observed indices), y (observed values). On turns 2-3 "
            "you ALSO receive 'residual_re_x1000' and 'residual_im_x1000' "
            "— these are y minus A(x_hat) of your previous answer, "
            "scaled by 1000. A small residual means your previous answer "
            "was close.\n\n"
            "OUTPUT one valid JSON object: {\"support_idx\": [k sorted "
            "ints in 0..n-1], \"support_amp_x1000\": [k signed ints in "
            "the same order]}. The amplitudes are real-valued and scaled "
            "by 1000; use ROUND, not floor, when converting from your "
            "internal float estimate.\n\n"
            "DO NOT output anything except this JSON object — no markdown "
            "fences, no commentary, no thinking-out-loud."
        ),
        provenance="llm-rewritten-directive",
    ),
]


# ── Budget cap ─────────────────────────────────────────────────


@dataclass
class BudgetCap:
    """Tracks cumulative LLM spend and aborts if it exceeds ``cap_usd``.

    The notebook constructs one instance and threads it through every
    call. Each ``CompletionResult`` from a real ``OpenRouterSolver``
    carries a ``usd_cost`` field; we sum those.
    """

    cap_usd: float
    spent_usd: float = 0.0

    def add(self, usd: float | None) -> None:
        if usd is None:
            return  # cost not reported (e.g. FakeLLMSolver)
        self.spent_usd += float(usd)
        if self.spent_usd > self.cap_usd:
            raise RuntimeError(
                f"Budget cap exceeded: spent ${self.spent_usd:.4f} > "
                f"cap ${self.cap_usd:.4f}. Aborting to protect spend."
            )

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)


# ── Per-seed evaluation ────────────────────────────────────────


@dataclass(frozen=True)
class SeedResult:
    """One env rollout for one (prompt, seed) pair."""

    seed: int
    prompt_name: str
    reward: float
    n_turns: int
    parse_ok: bool
    usd_cost: float


@dataclass
class CallCostTrackingSolver:
    """Wraps a base solver, accumulating ``usd_cost`` per call into ``budget``.

    We can't subclass ``OpenRouterSolver`` cheaply because ``__init__`` does
    network setup; instead we hold a reference and forward both ``complete``
    and ``complete_turns``.
    """

    base: Any  # LLMSolver
    budget: BudgetCap
    system_override: str | None = None
    _last_usd: float = 0.0

    def __post_init__(self) -> None:
        # Forward attributes the env / adapter may read.
        self.model = self.base.model
        self.temperature = self.base.temperature
        self.max_tokens = self.base.max_tokens
        self.timeout_s = self.base.timeout_s
        self.label = self.base.label

    def complete(self, system: str, user: str) -> Any:
        if self.system_override is not None:
            system = self.system_override
        result = self.base.complete(system, user)
        self.budget.add(result.usd_cost)
        self._last_usd = float(result.usd_cost or 0.0)
        return result

    def complete_turns(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        if self.system_override is not None and messages and messages[0].get("role") == "system":
            messages = [
                {**messages[0], "content": self.system_override},
                *messages[1:],
            ]
        result = self.base.complete_turns(messages, tools=tools)
        self.budget.add(result.usd_cost)
        self._last_usd = float(result.usd_cost or 0.0)
        return result


def evaluate_prompt(
    env: Any,
    base_solver: Any,
    candidate: PromptCandidate,
    seeds: Iterable[int],
    budget: BudgetCap,
    *,
    on_seed: Callable[[SeedResult], None] | None = None,
) -> list[SeedResult]:
    """Run ``env.run_rollout`` for each seed using ``candidate.system_prompt``.

    Returns one :class:`SeedResult` per seed. ``budget`` is mutated in
    place; if it's exceeded the function raises (the partial results
    are NOT returned to keep the bookkeeping simple — the caller can
    catch and inspect ``budget.spent_usd``).

    On a parse failure we record ``parse_ok=False, reward=0.0`` rather
    than propagating, so a single brittle seed doesn't kill the whole
    benchmark — this is the same convention the v2 benchmark uses.
    """
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError

    wrapped = CallCostTrackingSolver(
        base=base_solver,
        budget=budget,
        system_override=candidate.system_prompt,
    )
    out: list[SeedResult] = []
    for seed in seeds:
        instance = env.generate_instance(seed=int(seed))
        usd_before = budget.spent_usd
        try:
            scored = env.run_rollout(wrapped, instance)
            reward = float(scored["reward"])
            parse_ok = True
            n_turns = int(scored.get("meta", {}).get("n_turns", 1))
        except LLMSolverError:
            reward = 0.0
            parse_ok = False
            n_turns = 0
        result = SeedResult(
            seed=int(seed),
            prompt_name=candidate.name,
            reward=reward,
            n_turns=n_turns,
            parse_ok=parse_ok,
            usd_cost=budget.spent_usd - usd_before,
        )
        out.append(result)
        if on_seed is not None:
            on_seed(result)
    return out


# ── Statistics helpers ─────────────────────────────────────────


@dataclass(frozen=True)
class PairedBootstrapResult:
    """95 % paired-bootstrap CI for ``mean(b) - mean(a)``."""

    delta: float        # mean(b) - mean(a)
    lo: float           # 2.5th percentile of bootstrap deltas
    hi: float           # 97.5th percentile
    n: int              # number of paired samples
    n_bootstrap: int    # number of bootstrap resamples


def paired_bootstrap_ci(
    a: list[float],
    b: list[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired bootstrap on the per-seed difference ``b[i] - a[i]``.

    Returns ``mean_delta, (lo, hi)`` at 95 %. Caller decides
    significance: a CI strictly above 0 means b > a at p<0.05.
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal-length samples; got {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("paired bootstrap needs at least one paired sample")
    import random  # noqa: PLC0415 — local import keeps notebook startup quick

    rng = random.Random(seed)
    diffs = [bi - ai for ai, bi in zip(a, b, strict=True)]
    n = len(diffs)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_bootstrap)]
    hi = boot_means[int(0.975 * n_bootstrap) - 1]
    return PairedBootstrapResult(
        delta=sum(diffs) / n,
        lo=lo,
        hi=hi,
        n=n,
        n_bootstrap=n_bootstrap,
    )


# ── Reward summary ─────────────────────────────────────────────


@dataclass(frozen=True)
class RewardSummary:
    """Aggregate stats for one prompt's run on a seed pool."""

    prompt_name: str
    n: int
    mean: float
    std: float
    parse_fail_rate: float
    rewards: list[float] = field(default_factory=list)


def summarise(results: list[SeedResult]) -> RewardSummary:
    if not results:
        raise ValueError("cannot summarise an empty result list")
    rewards = [r.reward for r in results]
    n_fail = sum(1 for r in results if not r.parse_ok)
    name = results[0].prompt_name
    if not all(r.prompt_name == name for r in results):
        raise ValueError("summarise() expects all results from one prompt")
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return RewardSummary(
        prompt_name=name,
        n=len(rewards),
        mean=sum(rewards) / len(rewards),
        std=std,
        parse_fail_rate=n_fail / len(rewards),
        rewards=rewards,
    )


def best_candidate(summaries: list[RewardSummary]) -> RewardSummary:
    """Pick the highest-mean prompt; tie-break on lowest parse-fail rate."""
    if not summaries:
        raise ValueError("best_candidate() needs at least one summary")
    return max(summaries, key=lambda s: (s.mean, -s.parse_fail_rate))


__all__ = [
    "PromptCandidate",
    "DEFAULT_CANDIDATES",
    "BudgetCap",
    "SeedResult",
    "CallCostTrackingSolver",
    "evaluate_prompt",
    "PairedBootstrapResult",
    "paired_bootstrap_ci",
    "RewardSummary",
    "summarise",
    "best_candidate",
]
