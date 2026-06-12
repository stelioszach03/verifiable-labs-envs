"""long-context-synthesis — multi-needle 3-turn synthesis env (Phase 27.C).

Each instance carries 3-5 needles spread across distinct documents
(D4-B). The model's job is to read the full corpus and produce a
single free-text answer that combines all needles. Scoring uses
SQuAD-style token-F1 (D3-C) — the gold answer is the concatenation
of the needle facts.

The rollout is 3-turn (D6-A parity with `code-humaneval-multiturn`):

    Turn 1  →  context blob + question                  →  answer_v1
    Turn 2  →  feedback (F1 score + needle doc indices) →  answer_v2
                  (NO gold answer text — R10 carry-over)
    Turn 3  →  same                                     →  answer_final

Final reward is computed against ``answer_final`` and multiplied by
the standard turn-count penalty: ``final = base × (1 − 0.05 ·
(n_turns − 1))``, capped at 0.10.

Reward shape:

    reward = 0.10 · format_valid    (output is parseable JSON
                                      with an `answer` field)
           + 0.20 · parse_valid     (extracted answer is non-empty)
           + 0.70 · correctness     (token-F1 against the gold facts)

Test-default context = 4 000 tokens (D5 + R7); production callers
can scale to 128 K via the `target_tokens` hyperparameter.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.long_context_primitives import (
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEST_TOKENS,
    Corpus,
    NeedleAnchor,
    PositionMode,
    generate_corpus,
    inject_multiple_needles,
    token_f1,
)

NAME = "long-context-synthesis"

DEFAULT_MAX_TURNS: int = 3
TURN_PENALTY_PER_EXTRA: float = 0.05
TURN_PENALTY_CAP: float = 0.10

# 10 templates × 64-bit seed × 3 count modes (3 / 4 / 5 needles) ×
# ~1e6 parameter combos ≈ 5.5e23 effective instances.
EFFECTIVE_INSTANCES: int = 10 * (2**64) * 1_000_000 * 3

DEFAULT_ALPHA: float = 0.1
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correctness": 0.7,
}
DEFAULT_NEEDLE_COUNT_RANGE: tuple[int, int] = (3, 5)
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "target_tokens": DEFAULT_TEST_TOKENS,
    "document_count": DEFAULT_DOCUMENT_COUNT,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "needle_count_range": DEFAULT_NEEDLE_COUNT_RANGE,
    "max_turns": DEFAULT_MAX_TURNS,
}


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class SynthesisInstance:
    """One multi-needle synthesis problem draw."""

    question: str
    corpus: Corpus
    needle_facts: tuple[str, ...]
    needle_anchors: tuple[NeedleAnchor, ...]
    gold_answer: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.corpus.render_prompt(question=self.question)

    @property
    def needle_doc_ids(self) -> tuple[int, ...]:
        return tuple(a.document_id for a in self.needle_anchors)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "context_token_count": self.corpus.total_tokens(),
            "document_count": len(self.corpus.documents),
            "needle_count": len(self.needle_facts),
            **self.metadata,
        }


@dataclass(frozen=True)
class SynthesisPrediction:
    """Solver's free-text answer."""

    answer: str
    raw: str = ""
    confidence: float = 0.5


# ── Procedural needle generation ────────────────────────────────────


_NEEDLE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("The annual report shows {token} as the production figure",
     "production figure {token}"),
    ("The audit confirmed {token} as the verified balance",
     "verified balance {token}"),
    ("The committee assigned identifier {token} to the project",
     "identifier {token}"),
    ("The catalog records {token} as the entry code",
     "entry code {token}"),
    ("The shipment manifest lists {token} as the order reference",
     "order reference {token}"),
)


def _generate_needle_token(rng: np.random.Generator) -> str:
    """Deterministic ``XXXX-####`` token."""
    letters = "".join(
        chr(ord("A") + int(rng.integers(0, 26))) for _ in range(4)
    )
    digits = f"{int(rng.integers(0, 10000)):04d}"
    return f"{letters}-{digits}"


def _generate_question(needle_count: int) -> str:
    return (
        f"Combine the {needle_count} key facts that appear across the "
        "documents into a single concise summary, including each "
        "distinctive identifier exactly as it appears."
    )


def generate_problem(seed: int, **kwargs: Any) -> dict[str, Any]:
    """Sample a fresh multi-needle synthesis problem."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    target_tokens = int(params["target_tokens"])
    document_count = int(params["document_count"])
    n_min, n_max = params["needle_count_range"]
    n_min, n_max = int(n_min), int(n_max)
    if n_min < 1 or n_max < n_min:
        raise ValueError(
            f"needle_count_range must satisfy 1 <= n_min <= n_max; got ({n_min}, {n_max})"
        )

    rng = np.random.default_rng(int(seed))
    needle_count = int(rng.integers(n_min, n_max + 1))
    if needle_count > document_count:
        needle_count = document_count

    corpus = generate_corpus(
        seed=seed,
        target_tokens=target_tokens,
        document_count=document_count,
    )

    needles: list[str] = []
    gold_phrases: list[str] = []
    seen_tokens: set[str] = set()
    while len(needles) < needle_count:
        token = _generate_needle_token(rng)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        long_form, short_form = _NEEDLE_TEMPLATES[
            int(rng.integers(0, len(_NEEDLE_TEMPLATES)))
        ]
        needles.append(long_form.format(token=token))
        gold_phrases.append(short_form.format(token=token))

    rng_inject = np.random.default_rng(int(seed) + 1)
    position_mode: PositionMode = ("start", "middle", "end", "random")[
        int(rng_inject.integers(0, 4))
    ]
    new_corpus, anchors = inject_multiple_needles(
        corpus, needles=needles, rng=rng_inject, position=position_mode,
    )

    question = _generate_question(needle_count)
    gold_answer = "; ".join(gold_phrases) + "."

    return {
        "question": question,
        "corpus": new_corpus,
        "needle_facts": tuple(needles),
        "needle_anchors": anchors,
        "gold_answer": gold_answer,
        "needle_count": needle_count,
        "position_mode": position_mode,
    }


def generate_instance(seed: int, **kwargs: Any) -> SynthesisInstance:
    """Wrap :func:`generate_problem` in a :class:`SynthesisInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    return SynthesisInstance(
        question=problem["question"],
        corpus=problem["corpus"],
        needle_facts=problem["needle_facts"],
        needle_anchors=problem["needle_anchors"],
        gold_answer=problem["gold_answer"],
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "target_tokens": int(params["target_tokens"]),
            "needle_count": int(problem["needle_count"]),
            "position_mode": problem["position_mode"],
        },
    )


# ── Reward kernel (token-F1) ────────────────────────────────────────


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_envelope(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    candidates: list[str] = list(_FENCED_RE.findall(cleaned))
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))
    for c in candidates:
        try:
            data = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _is_format_valid(prediction: SynthesisPrediction) -> bool:
    if prediction.raw:
        data = _extract_envelope(prediction.raw)
        if not isinstance(data, dict):
            return False
        return bool(str(data.get("answer", "")).strip())
    return bool(prediction.answer.strip())


def _is_parse_valid(prediction: SynthesisPrediction) -> bool:
    return bool((prediction.answer or "").strip())


def score_components(
    prediction: SynthesisPrediction,
    instance: SynthesisInstance,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``.

    The correctness term is graded — token-F1 returns a float in
    ``[0, 1]`` (D7-A continuous scoring).
    """
    components = {"format_valid": 0.0, "parse_valid": 0.0, "correctness": 0.0}
    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components
    components["parse_valid"] = 1.0 if _is_parse_valid(prediction) else 0.0
    if components["parse_valid"] == 0.0:
        return components
    components["correctness"] = float(
        token_f1(prediction.answer, instance.gold_answer)
    )
    return components


def compute_reward(
    prediction: SynthesisPrediction,
    instance: SynthesisInstance,
    *,
    weights: dict[str, float] | None = None,
    conformal_quantile: float | None = None,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(prediction, instance)
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    completion_hash = hashlib.sha256(
        (prediction.answer or "").encode("utf-8")
    ).hexdigest()[:16]
    meta: dict[str, Any] = {
        "weights": dict(w),
        "needle_count": len(instance.needle_facts),
        "needle_doc_ids": list(instance.needle_doc_ids),
        "context_token_count": instance.corpus.total_tokens(),
        "completion_hash": completion_hash,
        "cache_key": _cache_key(NAME, int(instance.seed), completion_hash),
        "f1": float(components["correctness"]),
        "confidence": float(prediction.confidence),
    }
    if conformal_quantile is not None:
        residual = 1.0 - reward
        meta["covered"] = bool(residual <= float(conformal_quantile))
        meta["residual"] = residual
        meta["conformal_quantile"] = float(conformal_quantile)

    return {
        "reward": float(reward),
        "components": {k: float(v) for k, v in components.items()},
        "meta": meta,
    }


def _cache_key(env_id: str, seed: int, completion_hash: str) -> str:
    """Cache key on ``(env_id, seed, completion_hash)`` — D10-B."""
    payload = f"{env_id}|{seed}|{completion_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Adapter helpers ─────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are a careful long-context synthesizer. The user message "
    "contains a multi-document corpus separated by "
    "``---DOCUMENT N: <title>---`` headers, followed by a question "
    "asking you to combine multiple key facts spread across the "
    "documents.\n\n"
    "Output exactly one JSON object of the form\n"
    '    {"answer": "<combined summary>", "confidence": <float in [0, 1]>}\n\n'
    "Include each distinctive identifier verbatim. No prose outside "
    "the JSON object — JSON only."
)


def build_user_prompt(instance: SynthesisInstance) -> str:
    """Render the instance as the first-turn user message."""
    body = instance.corpus.render_prompt(question=instance.question)
    return (
        body
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"answer": "<combined summary>", "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: SynthesisInstance) -> SynthesisPrediction:
    del instance
    data = _extract_envelope(text)
    if not isinstance(data, dict):
        return SynthesisPrediction(answer="", raw=text, confidence=0.0)
    answer = str(data.get("answer", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return SynthesisPrediction(answer=answer, raw=text, confidence=confidence)


def render_synthesis_feedback(
    *,
    f1_score: float,
    needle_doc_ids: tuple[int, ...],
) -> str:
    """Build the inter-turn feedback string (R10-safe)."""
    pct = int(round(max(0.0, min(1.0, float(f1_score))) * 100))
    if f1_score < 0.50:
        return (
            f"FEEDBACK on your previous turn:\n"
            f"Your answer covers ~{pct}% of the expected content. "
            f"Review the documents at indices {list(needle_doc_ids)} "
            "for the missing facts."
        )
    if f1_score < 0.90:
        return (
            f"FEEDBACK on your previous turn:\n"
            f"Your answer covers ~{pct}% of the expected content. "
            "Refine the wording or add the missing facts."
        )
    return (
        "FEEDBACK on your previous turn:\n"
        "Your previous answer is largely correct. You may keep it "
        "for the final turn."
    )


# ── Env class + factory ─────────────────────────────────────────────


def baseline_predict(instance: SynthesisInstance) -> SynthesisPrediction:
    """Reference solver — empty answer."""
    del instance
    return SynthesisPrediction(answer="", raw="", confidence=0.0)


class LongContextSynthesisEnv:
    """RL environment handle wrapping one calibrated conformal quantile."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1; got {max_turns}")
        self.max_turns = int(max_turns)
        self.env_id: str = ""
        self.env_args: dict[str, Any] = {}

    def generate_instance(self, seed: int, **kwargs: Any) -> SynthesisInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: SynthesisPrediction,
        instance: SynthesisInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)

    def _apply_turn_penalty(
        self,
        scored: dict[str, Any],
        n_turns: int,
    ) -> dict[str, Any]:
        """Multiply the base reward by ``(1 − penalty)`` (D6-A parity)."""
        penalty = min(
            TURN_PENALTY_CAP,
            TURN_PENALTY_PER_EXTRA * max(0, n_turns - 1),
        )
        base = float(scored["reward"])
        adjusted = max(0.0, base * (1.0 - penalty))
        scored["reward"] = float(adjusted)
        scored["meta"] = {
            **scored.get("meta", {}),
            "base_reward": base,
            "turn_penalty": float(penalty),
        }
        return scored

    def build_followup_turn(
        self,
        prediction: SynthesisPrediction,
        instance: SynthesisInstance,
    ) -> str:
        """Render the inter-turn user message for turn ≥ 2."""
        components = score_components(prediction, instance)
        return render_synthesis_feedback(
            f1_score=components["correctness"],
            needle_doc_ids=instance.needle_doc_ids,
        )

    def run_rollout(
        self,
        solver: Any,
        instance: SynthesisInstance,
        *,
        adapter: Any = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        """Run up to ``max_turns`` turns of ``solver`` on ``instance``.

        Returns the final-turn :meth:`score` dict with these extras in
        ``meta``: ``turn_rewards``, ``turn_components``, ``n_turns``,
        ``max_turns``, ``base_reward``, ``turn_penalty``.
        """
        from verifiable_labs_envs.solvers.llm_solver import (  # noqa: PLC0415
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        turns = int(max_turns or self.max_turns)

        history: list[dict[str, str]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        turn_rewards: list[float] = []
        turn_components: list[dict[str, float]] = []
        last_prediction: SynthesisPrediction | None = None

        for turn_idx in range(turns):
            completion = solver.complete_turns(history)
            try:
                prediction = adapter.parse_response(completion.text, instance)
            except LLMSolverError:
                if last_prediction is None:
                    raise
                break

            scored = self.score(prediction, instance)
            turn_rewards.append(float(scored["reward"]))
            turn_components.append(dict(scored["components"]))
            last_prediction = prediction

            if turn_idx + 1 < turns:
                history.append({"role": "assistant", "content": completion.text})
                followup = self.build_followup_turn(prediction, instance)
                history.append({"role": "user", "content": followup})

        assert last_prediction is not None
        final = self.score(last_prediction, instance)
        final = self._apply_turn_penalty(final, n_turns=len(turn_rewards))
        final["meta"] = {
            **final["meta"],
            "turn_rewards": turn_rewards,
            "turn_components": turn_components,
            "n_turns": len(turn_rewards),
            "max_turns": turns,
        }
        return final


def calibrate_quantile(
    n_samples: int = 30,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals."""
    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(prediction=pred, instance=inst)
        residuals.append(1.0 - float(out["reward"]))
    return float(split_conformal_quantile(np.asarray(residuals), alpha))


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> LongContextSynthesisEnv:
    """Factory mirroring the verifiers convention."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return LongContextSynthesisEnv(
        conformal_quantile=q, max_turns=max_turns,
    )


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_MAX_TURNS",
    "TURN_PENALTY_PER_EXTRA",
    "TURN_PENALTY_CAP",
    "SYSTEM_PROMPT",
    "SynthesisInstance",
    "SynthesisPrediction",
    "LongContextSynthesisEnv",
    "baseline_predict",
    "build_user_prompt",
    "calibrate_quantile",
    "compute_reward",
    "generate_instance",
    "generate_problem",
    "load_environment",
    "parse_response",
    "render_synthesis_feedback",
    "score_components",
]
