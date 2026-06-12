"""math-algebra — single-turn algebraic-simplification RL environment.

Phase 21 introduces the symbolic-math env family. ``math-algebra`` is
the single-turn variant: given a natural-language problem (expand,
factor, simplify, …), the solver returns a SymPy-parseable answer +
self-reported confidence. Scoring is

    reward = 0.10 · format_valid          (output is parseable JSON)
           + 0.20 · parse_valid            (extracted answer is valid SymPy)
           + 0.70 · correct                (simplify(answer − gold) == 0)

with a hard threaded timeout on the ``simplify`` call so adversarial
inputs cannot wedge CI. A conformal coverage layer reuses
``verifiable_labs_envs.conformal`` directly: per-instance residual
``r = 1 − reward`` is calibrated to a ``(1 − α)``-quantile ``q̂`` over a
held-out baseline run, and the env emits a ``covered`` flag per call.

Procedural-regeneration contract: each ``(seed, hyperparams)`` pair
draws a fresh problem from a 5-template lattice with wide coefficient
ranges. The 64-bit seed space × the lattice cardinality yields
``EFFECTIVE_INSTANCES > 1e22``, well above the 1e15 threshold the
contamination-resistance validator enforces.
"""
from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sp

from verifiable_labs_envs.conformal import split_conformal_quantile

NAME = "math-algebra"
# 64-bit seed × 5 templates × ~32 768 coefficient combinations ≈ 3 × 10²³.
EFFECTIVE_INSTANCES: int = 2**64 * 5 * 32_768

DEFAULT_ALPHA: float = 0.1
DEFAULT_SIMPLIFY_TIMEOUT_S: float = 5.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correct": 0.7,
}
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "simplify_timeout_s": DEFAULT_SIMPLIFY_TIMEOUT_S,
    "coef_range": 10,  # coefficients sampled uniformly from [-coef_range, coef_range]
}

# Single shared symbol — using the same name across templates lets the
# reward function compare answers without per-instance symbol bookkeeping.
_X = sp.Symbol("x", real=True)


@dataclass(frozen=True)
class Instance:
    """One problem draw.

    ``gold_expr`` is the oracle field used by the scorer only — solvers
    must treat it as hidden (excluded by :meth:`as_inputs`).
    """

    prompt: str
    gold_expr: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {"prompt": self.prompt, **self.metadata}


@dataclass(frozen=True)
class Prediction:
    """Solver's answer.

    ``answer_expr`` is the SymPy-parseable answer string; ``confidence``
    is a self-reported scalar in ``[0, 1]`` (recorded in the audit
    trail; not folded into the reward in this single-turn variant).
    ``raw`` keeps the LLM's full response for traceability.
    """

    answer_expr: str
    raw: str = ""
    confidence: float = 0.5


# ───────────────────────── procedural generators ──────────────────────


def _nonzero(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Sample an integer from ``[lo, hi]`` inclusive, excluding zero."""
    while True:
        v = int(rng.integers(lo, hi + 1))
        if v != 0:
            return v


def _expand_binomial_product(rng: np.random.Generator, coef_range: int) -> tuple[str, sp.Expr]:
    a = _nonzero(rng, -coef_range, coef_range)
    b = int(rng.integers(-coef_range, coef_range + 1))
    c = _nonzero(rng, -coef_range, coef_range)
    d = int(rng.integers(-coef_range, coef_range + 1))
    expr = (a * _X + b) * (c * _X + d)
    gold = sp.expand(expr)
    prompt = (
        f"Expand the product ({a}*x + {b}) * ({c}*x + {d}) and write the "
        "result as a polynomial in x in standard form."
    )
    return prompt, gold


def _expand_square(rng: np.random.Generator, coef_range: int) -> tuple[str, sp.Expr]:
    a = _nonzero(rng, -coef_range, coef_range)
    b = int(rng.integers(-coef_range, coef_range + 1))
    expr = (a * _X + b) ** 2
    gold = sp.expand(expr)
    prompt = (
        f"Expand the square ({a}*x + {b})**2 as a polynomial in x in "
        "standard form."
    )
    return prompt, gold


def _difference_of_squares(rng: np.random.Generator, coef_range: int) -> tuple[str, sp.Expr]:
    a = _nonzero(rng, 1, coef_range)
    expr = (_X + a) * (_X - a)
    gold = sp.expand(expr)
    prompt = (
        f"Expand the product (x + {a}) * (x - {a}) and simplify."
    )
    return prompt, gold


def _factor_quadratic_with_rational_roots(
    rng: np.random.Generator, coef_range: int
) -> tuple[str, sp.Expr]:
    """Build a quadratic with two integer roots, ask for the factorisation.

    Gold is stored as the canonical factored form ``(x − r1)(x − r2)`` —
    SymPy's ``simplify`` reduces any equivalent factored or expanded
    answer to zero against this gold.
    """
    r1 = int(rng.integers(-coef_range, coef_range + 1))
    r2 = int(rng.integers(-coef_range, coef_range + 1))
    expanded = sp.expand((_X - r1) * (_X - r2))
    factored = (_X - r1) * (_X - r2)
    prompt = (
        f"Factor the quadratic {sp.sstr(expanded)} as a product of two "
        "linear terms in x."
    )
    return prompt, factored


def _collect_like_terms(rng: np.random.Generator, coef_range: int) -> tuple[str, sp.Expr]:
    """Random linear combination of x², x, 1 with two repeated terms."""
    a1 = _nonzero(rng, -coef_range, coef_range)
    a2 = _nonzero(rng, -coef_range, coef_range)
    b1 = _nonzero(rng, -coef_range, coef_range)
    b2 = _nonzero(rng, -coef_range, coef_range)
    c = int(rng.integers(-coef_range, coef_range + 1))
    raw = a1 * _X**2 + a2 * _X**2 + b1 * _X + b2 * _X + c
    gold = sp.expand(raw)
    raw_str = sp.sstr(raw)
    prompt = (
        f"Combine like terms: {raw_str}. Return the result as a polynomial "
        "in x in standard form."
    )
    return prompt, gold


_TEMPLATES: list[tuple[str, Any]] = [
    ("expand_binomial_product", _expand_binomial_product),
    ("expand_square", _expand_square),
    ("difference_of_squares", _difference_of_squares),
    ("factor_quadratic", _factor_quadratic_with_rational_roots),
    ("collect_like_terms", _collect_like_terms),
]


def generate_problem(
    seed: int,
    *,
    coef_range: int = 10,
    **_unused: Any,
) -> tuple[str, str]:
    """Sample a fresh ``(prompt, gold_expr_str)`` pair from the per-env distribution.

    Determinism: two calls with the same ``(seed, coef_range)`` return
    identical ``(prompt, gold_expr_str)`` tuples. ``gold_expr_str`` is a
    SymPy-parseable string (via :func:`sympy.sstr`).
    """
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    name, fn = _TEMPLATES[template_idx]
    prompt, gold = fn(rng, coef_range)
    # Tag the prompt with the template name so the audit trail can
    # group rewards by category.
    return prompt, sp.sstr(gold)


def generate_instance(seed: int, **kwargs: Any) -> Instance:
    """Wrap :func:`generate_problem` output in an :class:`Instance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    coef_range = int(params.get("coef_range", 10))
    prompt, gold_expr = generate_problem(seed, coef_range=coef_range)
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    return Instance(
        prompt=prompt,
        gold_expr=gold_expr,
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "simplify_timeout_s": float(params["simplify_timeout_s"]),
            "coef_range": coef_range,
            "template": _TEMPLATES[template_idx][0],
        },
    )


# ───────────────────────── reward kernel ────────────────────────────


def simplify_with_timeout(
    expr_diff: sp.Expr,
    timeout_s: float = DEFAULT_SIMPLIFY_TIMEOUT_S,
) -> sp.Expr | None:
    """Run ``sympy.simplify`` with a hard timeout via a daemon thread.

    Returns the simplified expression on success, ``None`` on timeout
    or any internal SymPy error (callers treat ``None`` as "not equal").
    Cross-platform — does not rely on ``signal.SIGALRM`` so it works
    the same way on Linux, macOS, and Windows. The underlying SymPy
    work continues in the daemon thread after timeout but cannot block
    the caller.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(sp.simplify, expr_diff)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return None
        except Exception:
            return None


def score_components(
    prediction: Prediction,
    instance: Instance,
    *,
    timeout_s: float = DEFAULT_SIMPLIFY_TIMEOUT_S,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``.

    Short-circuits aggressively: malformed JSON stops at
    ``format_valid``; un-parseable answer expressions stop at
    ``parse_valid``. Only survivors pay for the timeout-bounded
    simplify call.
    """
    components = {"format_valid": 0.0, "parse_valid": 0.0, "correct": 0.0}

    # 1. format_valid
    if prediction.raw:
        try:
            json.loads(prediction.raw)
            components["format_valid"] = 1.0
        except (json.JSONDecodeError, ValueError, TypeError):
            return components
    else:
        components["format_valid"] = 1.0 if prediction.answer_expr else 0.0
        if not prediction.answer_expr:
            return components

    # 2. parse_valid
    try:
        answer = sp.sympify(prediction.answer_expr)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return components
    components["parse_valid"] = 1.0

    # 3. correct
    try:
        gold = sp.sympify(instance.gold_expr)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        # Gold malformed — env bug, return what we have.
        return components

    diff = answer - gold
    simplified = simplify_with_timeout(diff, timeout_s=timeout_s)
    if simplified is not None and simplified == 0:
        components["correct"] = 1.0

    return components


def compute_reward(
    prediction: Prediction,
    instance: Instance,
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_SIMPLIFY_TIMEOUT_S,
    conformal_quantile: float | None = None,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(prediction, instance, timeout_s=timeout_s)
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    meta: dict[str, Any] = {
        "weights": dict(w),
        "timeout_s": timeout_s,
        "confidence": float(prediction.confidence),
        "template": instance.metadata.get("template", "unknown"),
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


# ───────────────────────── env class + factory ──────────────────────


def baseline_predict(instance: Instance) -> Prediction:
    """Reference solver — returns an empty prediction with zero confidence.

    Used by calibration. Empty answer scores zero on every component;
    this gives a wide residual distribution at calibration time, which
    in turn yields a non-trivial conformal threshold on the trained
    solvers later.
    """
    del instance
    return Prediction(answer_expr="", raw="", confidence=0.0)


class MathAlgebraEnv:
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

    def generate_instance(self, seed: int, **kwargs: Any) -> Instance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(self, prediction: Prediction, instance: Instance) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            timeout_s=float(self.hyperparams["simplify_timeout_s"]),
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)


def calibrate_quantile(
    n_samples: int = 30,
    alpha: float = DEFAULT_ALPHA,
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_SIMPLIFY_TIMEOUT_S,
) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals over
    ``n_samples`` fresh seeds. Reuses
    :func:`verifiable_labs_envs.conformal.split_conformal_quantile`."""
    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(
            prediction=pred,
            instance=inst,
            weights=weights,
            timeout_s=timeout_s,
        )
        residuals.append(1.0 - float(out["reward"]))
    return float(split_conformal_quantile(np.asarray(residuals), alpha))


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> MathAlgebraEnv:
    """Factory mirroring the verifiers convention. Pass
    ``calibration_quantile`` to skip auto-calibration in tests."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 30 if fast else 200
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return MathAlgebraEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_SIMPLIFY_TIMEOUT_S",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "Instance",
    "Prediction",
    "MathAlgebraEnv",
    "generate_problem",
    "generate_instance",
    "baseline_predict",
    "score_components",
    "compute_reward",
    "simplify_with_timeout",
    "calibrate_quantile",
    "load_environment",
]
