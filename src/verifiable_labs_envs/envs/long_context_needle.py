"""long-context-needle — single-turn NIAH retrieval env (Phase 27.B).

Phase 27 introduces context-length scaling as a verification
primitive. The needle variant is the classic Needle-in-Haystack
shape: a procedurally generated multi-document corpus (D2-A) with a
single deterministically-positioned needle (D4-A); the model
returns the needle text and is scored by exact match (D3-A).

Reward:

    reward = 0.10 · format_valid    (output is parseable JSON
                                      with an `answer` field)
           + 0.20 · parse_valid     (extracted answer is non-empty)
           + 0.70 · correctness     (gold needle ⊂ predicted answer,
                                      case-insensitive)

Test-default context = 4 000 tokens (per D5 + R7); production
callers can scale to 128 K via the `target_tokens` hyperparameter.
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
    count_tokens,
    exact_match,
    generate_corpus,
    inject_needle,
)

NAME = "long-context-needle"

# 10 templates × 64-bit seed × ~10⁶ parameter combinations × 4
# position modes ≈ 7.4e23 effective instances; well above the 1e15
# procedural-regeneration gate.
EFFECTIVE_INSTANCES: int = 10 * (2**64) * 1_000_000 * 4

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

POSITION_MODES: tuple[PositionMode, ...] = ("start", "middle", "end", "random")


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class NeedleInstance:
    """One NIAH problem draw."""

    question: str
    corpus: Corpus
    needle_text: str
    needle_anchor: NeedleAnchor
    position_mode: PositionMode
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_answer(self) -> str:
        """The needle text (the canonical answer string)."""
        return self.needle_text

    @property
    def prompt(self) -> str:
        """Composed user prompt: documents + question."""
        return self.corpus.render_prompt(question=self.question)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "context_token_count": self.corpus.total_tokens(),
            "document_count": len(self.corpus.documents),
            **self.metadata,
        }


@dataclass(frozen=True)
class NeedlePrediction:
    """Solver's answer (the extracted needle string)."""

    answer: str
    raw: str = ""
    confidence: float = 0.5


# ── Procedural needle generation ────────────────────────────────────


# Three needle templates — each emits a distinct surface form so the
# distribution covers numeric / categorical / mixed answer shapes.
_NEEDLE_TEMPLATES: tuple[str, ...] = (
    "The secret access code is {token}",
    "The reference identifier {token} was assigned to the project",
    "The unique sequence is {token}",
)


def _generate_needle_token(rng: np.random.Generator) -> str:
    """Random alphanumeric token of the form ``ABCD-1234``."""
    letters = "".join(
        chr(ord("A") + int(rng.integers(0, 26))) for _ in range(4)
    )
    digits = f"{int(rng.integers(0, 10000)):04d}"
    return f"{letters}-{digits}"


def _generate_question(rng: np.random.Generator, needle_token: str) -> str:
    """Question phrasing — 3 templates."""
    templates = (
        "What is the secret access code mentioned in the documents?",
        "What reference identifier was assigned to the project?",
        "What unique sequence appears in the documents?",
    )
    return templates[int(rng.integers(0, len(templates)))]


def generate_problem(seed: int, **kwargs: Any) -> dict[str, Any]:
    """Sample a fresh NIAH problem from the procedural lattice."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    target_tokens = int(params["target_tokens"])
    document_count = int(params["document_count"])
    rng = np.random.default_rng(int(seed))

    # 1. Generate the corpus.
    corpus = generate_corpus(
        seed=seed,
        target_tokens=target_tokens,
        document_count=document_count,
    )

    # 2. Generate the needle token + question + position.
    needle_token = _generate_needle_token(rng)
    needle_template = _NEEDLE_TEMPLATES[int(rng.integers(0, len(_NEEDLE_TEMPLATES)))]
    needle_text = needle_template.format(token=needle_token)
    question = _generate_question(rng, needle_token)
    position_mode: PositionMode = POSITION_MODES[
        int(rng.integers(0, len(POSITION_MODES)))
    ]

    # 3. Inject the needle.
    rng_inject = np.random.default_rng(int(seed) + 1)
    new_corpus, anchor = inject_needle(
        corpus,
        needle_text=needle_text,
        position=position_mode,
        rng=rng_inject,
    )

    return {
        "question": question,
        "corpus": new_corpus,
        "needle_text": needle_text,
        "needle_token": needle_token,
        "needle_anchor": anchor,
        "position_mode": position_mode,
    }


def generate_instance(seed: int, **kwargs: Any) -> NeedleInstance:
    """Wrap :func:`generate_problem` output in a :class:`NeedleInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    return NeedleInstance(
        question=problem["question"],
        corpus=problem["corpus"],
        needle_text=problem["needle_text"],
        needle_anchor=problem["needle_anchor"],
        position_mode=problem["position_mode"],
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "target_tokens": int(params["target_tokens"]),
            "needle_token": problem["needle_token"],
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


def _is_format_valid(prediction: NeedlePrediction) -> bool:
    """``raw`` is JSON with a non-empty ``answer`` field."""
    if prediction.raw:
        data = _extract_envelope(prediction.raw)
        if not isinstance(data, dict):
            return False
        return bool(str(data.get("answer", "")).strip())
    return bool(prediction.answer.strip())


def _is_parse_valid(prediction: NeedlePrediction) -> bool:
    """Extracted answer is non-empty."""
    return bool((prediction.answer or "").strip())


def score_components(
    prediction: NeedlePrediction,
    instance: NeedleInstance,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``."""
    components = {"format_valid": 0.0, "parse_valid": 0.0, "correctness": 0.0}
    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components
    components["parse_valid"] = 1.0 if _is_parse_valid(prediction) else 0.0
    if components["parse_valid"] == 0.0:
        return components
    # The needle's distinctive token (e.g., ``ABCD-1234``) must appear
    # in the model's answer. Substring + case-insensitive (D3-A).
    needle_token = instance.metadata.get(
        "needle_token", instance.needle_text
    )
    components["correctness"] = (
        1.0 if exact_match(prediction.answer, needle_token) else 0.0
    )
    return components


def compute_reward(
    prediction: NeedlePrediction,
    instance: NeedleInstance,
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
        "position_mode": instance.position_mode,
        "context_token_count": instance.corpus.total_tokens(),
        "needle_doc_id": instance.needle_anchor.document_id,
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


# ── D10-B per-process LRU cache ─────────────────────────────────────


def _cache_key(env_id: str, seed: int, completion_hash: str) -> str:
    """Cache key on ``(env_id, seed, completion_hash)`` — D10-B."""
    payload = f"{env_id}|{seed}|{completion_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Adapter helpers ─────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are a careful long-context reader. The user message contains "
    "a multi-document corpus separated by ``---DOCUMENT N: <title>---`` "
    "headers, followed by a question. Locate the relevant fact and "
    "return it.\n\n"
    "Output exactly one JSON object of the form\n"
    '    {"answer": "<extracted text>", "confidence": <float in [0, 1]>}\n\n'
    "No prose, no markdown fences — JSON only."
)


def build_user_prompt(instance: NeedleInstance) -> str:
    """Render the env instance as the user-message text."""
    body = instance.corpus.render_prompt(question=instance.question)
    return (
        body
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"answer": "<extracted text>", "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: NeedleInstance) -> NeedlePrediction:
    """Parse the LLM's text into a :class:`NeedlePrediction`."""
    del instance
    data = _extract_envelope(text)
    if not isinstance(data, dict):
        return NeedlePrediction(answer="", raw=text, confidence=0.0)
    answer = str(data.get("answer", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return NeedlePrediction(answer=answer, raw=text, confidence=confidence)


# ── Env class + factory ─────────────────────────────────────────────


def baseline_predict(instance: NeedleInstance) -> NeedlePrediction:
    """Reference solver — empty answer."""
    del instance
    return NeedlePrediction(answer="", raw="", confidence=0.0)


class LongContextNeedleEnv:
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

    def generate_instance(self, seed: int, **kwargs: Any) -> NeedleInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: NeedlePrediction,
        instance: NeedleInstance,
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
) -> LongContextNeedleEnv:
    """Factory mirroring the verifiers convention."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return LongContextNeedleEnv(conformal_quantile=q)


# Quiet the lint warning — count_tokens is exported for downstream
# consumers who want the same tokeniser the env uses.
_ = count_tokens

__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "POSITION_MODES",
    "SYSTEM_PROMPT",
    "NeedleInstance",
    "NeedlePrediction",
    "LongContextNeedleEnv",
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
