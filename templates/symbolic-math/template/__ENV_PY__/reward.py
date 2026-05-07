"""Conformal-calibrated reward for __ENV_ID__.

The reward function combines three SymPy-aware components — JSON
format validity, SymPy parse validity, and symbolic equivalence —
into a single scalar in ``[0, 1]``. The dominant term, ``correct``,
runs ``sympy.simplify(answer − gold) == 0`` inside a hard timeout so
adversarial inputs cannot wedge the env.

A conformal coverage term layers on top: at calibration time we
collect residuals ``r = 1 − reward`` over a held-out set, take the
``(1 − α)``-quantile ``q̂``, and score test-time coverage as the
fraction of test residuals ``≤ q̂``. The reward dict's ``meta`` block
includes the per-instance ``covered`` flag so downstream telemetry can
aggregate it.

The conformal kernel itself reuses
``verifiable_labs_envs.conformal.split_conformal_quantile`` /
``coverage_score``; this module imports them lazily at call time to
keep the scaffold runnable even before the parent package is on
``sys.path`` (e.g. when running unit tests directly with ``pytest``).
"""
from __future__ import annotations

import concurrent.futures
import json
from typing import Any

from __ENV_PY__.data import Instance, Prediction

DEFAULT_ALPHA: float = 0.1
DEFAULT_SIMPLIFY_TIMEOUT_S: float = 5.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correct": 0.7,
}


def _simplify_with_timeout(
    expr_diff: Any,
    timeout_s: float = DEFAULT_SIMPLIFY_TIMEOUT_S,
) -> Any:
    """Run ``sympy.simplify`` with a hard timeout via a daemon thread.

    Returns the simplified expression on success, ``None`` on timeout
    or any internal SymPy error (callers treat ``None`` as
    "not-equal"). Cross-platform — does not rely on ``signal.SIGALRM``
    so it works the same way on Linux, macOS, and Windows. The
    underlying SymPy work continues in the daemon thread after timeout
    but cannot block the caller.
    """
    import sympy as sp  # noqa: PLC0415  (lazy import keeps module load cheap)

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

    Short-circuits aggressively: if ``raw`` is not parseable JSON we
    stop at ``format_valid``; if ``answer_expr`` is not SymPy-parseable
    we stop at ``parse_valid``. Only the survivors pay for the
    timeout-bounded simplify call.
    """
    components = {"format_valid": 0.0, "parse_valid": 0.0, "correct": 0.0}

    # 1. format_valid — the raw response was JSON.
    if prediction.raw:
        try:
            json.loads(prediction.raw)
            components["format_valid"] = 1.0
        except (json.JSONDecodeError, ValueError, TypeError):
            return components
    else:
        # Empty raw is treated as "JSON not provided"; remaining
        # components still scored on the structured fields.
        components["format_valid"] = 1.0 if prediction.answer_expr else 0.0
        if not prediction.answer_expr:
            return components

    # 2. parse_valid — answer_expr is a valid SymPy expression.
    import sympy as sp  # noqa: PLC0415
    try:
        answer = sp.sympify(prediction.answer_expr)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return components
    components["parse_valid"] = 1.0

    # 3. correct — simplify(answer - gold) == 0 with timeout.
    try:
        gold = sp.sympify(instance.gold_expr)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        # Gold is malformed — env bug, not solver fault. Return what
        # we have so the scorer doesn't false-fail the prediction.
        return components

    diff = answer - gold
    simplified = _simplify_with_timeout(diff, timeout_s=timeout_s)
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
    """Combine the three components into the env reward dict.

    The optional ``conformal_quantile`` controls the per-instance
    ``covered`` flag in ``meta``: ``covered = (1 − reward) ≤ q̂``.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(prediction, instance, timeout_s=timeout_s)
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    meta: dict[str, Any] = {
        "weights": dict(w),
        "timeout_s": timeout_s,
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


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SIMPLIFY_TIMEOUT_S",
    "DEFAULT_WEIGHTS",
    "score_components",
    "compute_reward",
]
