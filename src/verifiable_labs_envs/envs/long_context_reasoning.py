"""long-context-reasoning — multi-hop chain QA env (Phase 27.D).

Three procedural templates ship in v0.0.1:

1. ``chain_two_hop``       — 2-hop fact composition (e.g., capital →
                              population).
2. ``chain_three_hop``     — 3-hop transitive composition.
3. ``arithmetic_over_facts`` — 2-hop fact retrieval + simple
                                arithmetic.

Each instance plants 2 true chain facts + 2 distractors with a
similar surface form across distinct documents (D4-C). The gold
answer is either a string (chain templates) or a number
(``arithmetic_over_facts``); scoring uses substring match for
strings and numeric tolerance (1 × 10⁻⁶) for numbers (D3-A).

Reward shape:

    reward = 0.10 · format_valid    (output is parseable JSON
                                      with an `answer` field)
           + 0.20 · parse_valid     (extracted answer is non-empty)
           + 0.70 · correctness     (exact / numeric match against
                                      gold answer)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.long_context_primitives import (
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEST_TOKENS,
    Corpus,
    build_chain_corpus,
    exact_match,
    numeric_match,
)

NAME = "long-context-reasoning"

# 3 templates × 64-bit seed × 4 distractor-position modes × ~1e6
# parameter combos ≈ 2.2e23 effective instances; well above the
# 1e15 procedural-regeneration gate.
EFFECTIVE_INSTANCES: int = 3 * (2**64) * 1_000_000 * 4

DEFAULT_ALPHA: float = 0.1
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correctness": 0.7,
}
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "target_tokens": DEFAULT_TEST_TOKENS,
    "document_count": DEFAULT_DOCUMENT_COUNT,
    "max_tokens": DEFAULT_MAX_TOKENS,
}

ChainTemplate = Literal["chain_two_hop", "chain_three_hop", "arithmetic_over_facts"]
TEMPLATE_NAMES: tuple[ChainTemplate, ...] = (
    "chain_two_hop",
    "chain_three_hop",
    "arithmetic_over_facts",
)

AnswerKind = Literal["string", "numeric"]


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningInstance:
    """One multi-hop reasoning problem draw."""

    question: str
    template_name: ChainTemplate
    seed: int
    corpus: Corpus
    gold_answer: str | float
    gold_answer_kind: AnswerKind
    gold_chain_doc_ids: tuple[int, ...]
    distractor_doc_ids: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.corpus.render_prompt(question=self.question)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "context_token_count": self.corpus.total_tokens(),
            "document_count": len(self.corpus.documents),
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class ReasoningPrediction:
    """Solver's answer (string or numeric)."""

    answer: str
    raw: str = ""
    confidence: float = 0.5


# ── Procedural chain generation ─────────────────────────────────────


_NAMES_A: tuple[str, ...] = (
    "Avalon", "Briarwood", "Caelum", "Drakemoor", "Elysian",
    "Foxfire", "Glenwood", "Halcyon", "Iolite", "Junipersend",
)
_NAMES_B: tuple[str, ...] = (
    "Region", "Province", "Territory", "Demesne", "Domain",
    "Realm", "District", "Holding", "March", "Reach",
)


def _sample_distinct(
    rng: np.random.Generator, pool: tuple[str, ...], n: int,
) -> list[str]:
    if n > len(pool):
        raise ValueError(f"requested {n} from pool of size {len(pool)}")
    idx = rng.permutation(len(pool))[:n]
    return [pool[int(i)] for i in idx]


def _two_hop_problem(rng: np.random.Generator) -> dict[str, Any]:
    """Capital → population. Gold = the population number."""
    cities = _sample_distinct(rng, _NAMES_A, 2)
    regions = _sample_distinct(rng, _NAMES_B, 2)
    true_city, distractor_city = cities[0], cities[1]
    true_region, distractor_region = regions[0], regions[1]

    true_pop = int(rng.integers(50_000, 9_999_999))
    distractor_pop = int(rng.integers(50_000, 9_999_999))
    while distractor_pop == true_pop:
        distractor_pop = int(rng.integers(50_000, 9_999_999))

    chain_facts = [
        f"The capital of {true_region} is {true_city}",
        f"The population of {true_city} is {true_pop}",
    ]
    distractor_facts = [
        f"The capital of {distractor_region} is {distractor_city}",
        f"The population of {distractor_city} is {distractor_pop}",
    ]
    question = f"What is the population of the capital of {true_region}?"
    return {
        "question": question,
        "chain_facts": chain_facts,
        "distractor_facts": distractor_facts,
        "gold_answer": float(true_pop),
        "gold_answer_kind": "numeric",
    }


def _three_hop_problem(rng: np.random.Generator) -> dict[str, Any]:
    """Region → capital → mayor. Gold = the mayor name."""
    cities = _sample_distinct(rng, _NAMES_A, 2)
    regions = _sample_distinct(rng, _NAMES_B, 2)
    mayors = _sample_distinct(rng, _NAMES_A[2:], 2)

    true_region, distractor_region = regions[0], regions[1]
    true_city, distractor_city = cities[0], cities[1]
    true_mayor, distractor_mayor = mayors[0], mayors[1]

    chain_facts = [
        f"The capital of {true_region} is {true_city}",
        f"The mayor of {true_city} is {true_mayor}",
        f"In {true_region}, the head of state is the mayor of the capital",
    ]
    distractor_facts = [
        f"The capital of {distractor_region} is {distractor_city}",
        f"The mayor of {distractor_city} is {distractor_mayor}",
    ]
    question = f"Who is the head of state of {true_region}?"
    return {
        "question": question,
        "chain_facts": chain_facts,
        "distractor_facts": distractor_facts,
        "gold_answer": true_mayor,
        "gold_answer_kind": "string",
    }


def _arithmetic_problem(rng: np.random.Generator) -> dict[str, Any]:
    """Production figures across two facilities; gold = sum."""
    cities = _sample_distinct(rng, _NAMES_A, 2)
    distractor_city = _sample_distinct(rng, _NAMES_A[2:], 1)[0]

    a = int(rng.integers(1_000, 99_999))
    b = int(rng.integers(1_000, 99_999))
    distractor_value = int(rng.integers(1_000, 99_999))
    while distractor_value in (a, b):
        distractor_value = int(rng.integers(1_000, 99_999))

    chain_facts = [
        f"The {cities[0]} facility produced {a} units last year",
        f"The {cities[1]} facility produced {b} units last year",
    ]
    distractor_facts = [
        f"The {distractor_city} facility produced {distractor_value} units last year",
        f"The {cities[0]} facility employs {distractor_value} staff",
    ]
    question = (
        f"What was the combined annual production of the {cities[0]} "
        f"and {cities[1]} facilities last year?"
    )
    return {
        "question": question,
        "chain_facts": chain_facts,
        "distractor_facts": distractor_facts,
        "gold_answer": float(a + b),
        "gold_answer_kind": "numeric",
    }


_TEMPLATES = {
    "chain_two_hop": _two_hop_problem,
    "chain_three_hop": _three_hop_problem,
    "arithmetic_over_facts": _arithmetic_problem,
}


def generate_problem(seed: int, **kwargs: Any) -> dict[str, Any]:
    """Sample a fresh multi-hop reasoning problem from the lattice."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    target_tokens = int(params["target_tokens"])
    document_count = int(params["document_count"])

    rng = np.random.default_rng(int(seed))
    template_name: ChainTemplate = TEMPLATE_NAMES[
        int(rng.integers(0, len(TEMPLATE_NAMES)))
    ]
    problem = _TEMPLATES[template_name](rng)

    chain_corpus = build_chain_corpus(
        seed=int(seed),
        chain_facts=list(problem["chain_facts"]),
        distractor_facts=list(problem["distractor_facts"]),
        document_count=document_count,
        target_tokens=target_tokens,
    )

    return {
        "template_name": template_name,
        "question": problem["question"],
        "corpus": chain_corpus.corpus,
        "gold_answer": problem["gold_answer"],
        "gold_answer_kind": problem["gold_answer_kind"],
        "gold_chain_doc_ids": chain_corpus.gold_chain_doc_ids(),
        "distractor_doc_ids": tuple(
            f.document_id for f in chain_corpus.facts if f.is_distractor
        ),
    }


def generate_instance(seed: int, **kwargs: Any) -> ReasoningInstance:
    """Wrap :func:`generate_problem` in a :class:`ReasoningInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    return ReasoningInstance(
        question=problem["question"],
        template_name=problem["template_name"],
        seed=int(seed),
        corpus=problem["corpus"],
        gold_answer=problem["gold_answer"],
        gold_answer_kind=problem["gold_answer_kind"],
        gold_chain_doc_ids=problem["gold_chain_doc_ids"],
        distractor_doc_ids=problem["distractor_doc_ids"],
        metadata={
            "alpha": float(params["alpha"]),
            "target_tokens": int(params["target_tokens"]),
        },
    )


# ── Reward kernel ───────────────────────────────────────────────────


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


def _is_format_valid(prediction: ReasoningPrediction) -> bool:
    if prediction.raw:
        data = _extract_envelope(prediction.raw)
        if not isinstance(data, dict):
            return False
        return bool(str(data.get("answer", "")).strip())
    return bool(prediction.answer.strip())


def _is_parse_valid(prediction: ReasoningPrediction) -> bool:
    return bool((prediction.answer or "").strip())


def _is_correct(prediction: ReasoningPrediction, instance: ReasoningInstance) -> bool:
    """Dispatch on ``gold_answer_kind``: substring vs numeric tolerance."""
    if instance.gold_answer_kind == "numeric":
        return numeric_match(prediction.answer, float(instance.gold_answer))
    return exact_match(prediction.answer, str(instance.gold_answer))


def score_components(
    prediction: ReasoningPrediction,
    instance: ReasoningInstance,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``."""
    components = {"format_valid": 0.0, "parse_valid": 0.0, "correctness": 0.0}
    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components
    components["parse_valid"] = 1.0 if _is_parse_valid(prediction) else 0.0
    if components["parse_valid"] == 0.0:
        return components
    components["correctness"] = 1.0 if _is_correct(prediction, instance) else 0.0
    return components


def compute_reward(
    prediction: ReasoningPrediction,
    instance: ReasoningInstance,
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
        "template": instance.template_name,
        "gold_answer_kind": instance.gold_answer_kind,
        "context_token_count": instance.corpus.total_tokens(),
        "completion_hash": completion_hash,
        "cache_key": _cache_key(NAME, int(instance.seed), completion_hash),
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
    "You are a careful long-context reasoner. The user message "
    "contains a multi-document corpus separated by "
    "``---DOCUMENT N: <title>---`` headers, followed by a multi-hop "
    "question. Some documents contain decoy facts with similar "
    "surface form — read carefully and chain the relevant facts.\n\n"
    "Output exactly one JSON object of the form\n"
    '    {"answer": "<final answer>", "confidence": <float in [0, 1]>}\n\n'
    "Numeric answers should be returned as plain numbers (no thousands "
    "separators). No prose outside the JSON — JSON only."
)


def build_user_prompt(instance: ReasoningInstance) -> str:
    body = instance.corpus.render_prompt(question=instance.question)
    return (
        body
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"answer": "<final answer>", "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: ReasoningInstance) -> ReasoningPrediction:
    del instance
    data = _extract_envelope(text)
    if not isinstance(data, dict):
        return ReasoningPrediction(answer="", raw=text, confidence=0.0)
    answer_raw = data.get("answer", "")
    # Numeric answers may come back as int/float; stringify uniformly.
    answer = str(answer_raw).strip() if answer_raw is not None else ""
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return ReasoningPrediction(answer=answer, raw=text, confidence=confidence)


# ── Env class + factory ─────────────────────────────────────────────


def baseline_predict(instance: ReasoningInstance) -> ReasoningPrediction:
    """Reference solver — empty answer."""
    del instance
    return ReasoningPrediction(answer="", raw="", confidence=0.0)


class LongContextReasoningEnv:
    """RL environment handle wrapping one calibrated conformal quantile."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.env_id: str = ""
        self.env_args: dict[str, Any] = {}

    def generate_instance(self, seed: int, **kwargs: Any) -> ReasoningInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: ReasoningPrediction,
        instance: ReasoningInstance,
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
) -> LongContextReasoningEnv:
    """Factory mirroring the verifiers convention."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return LongContextReasoningEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "TEMPLATE_NAMES",
    "SYSTEM_PROMPT",
    "ReasoningInstance",
    "ReasoningPrediction",
    "LongContextReasoningEnv",
    "baseline_predict",
    "build_user_prompt",
    "calibrate_quantile",
    "compute_reward",
    "generate_instance",
    "generate_problem",
    "load_environment",
    "parse_response",
    "score_components",
]
