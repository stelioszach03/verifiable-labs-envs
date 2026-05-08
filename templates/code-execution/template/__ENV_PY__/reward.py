"""Conformal-calibrated reward for __ENV_ID__.

The reward function combines three components — JSON format
validity, Python parse validity, and pytest pass-rate — into a
single scalar in ``[0, 1]``. The dominant term, ``pass_rate``, runs
the model's solution against the env's hidden test suite inside the
D5-bounded sandbox at
``__ENV_PY__.sandbox.execute_in_sandbox_sync``.

A conformal coverage term layers on top: at calibration time we
collect residuals ``r = 1 − reward`` over a held-out baseline set,
take the ``(1 − α)``-quantile ``q̂``, and score test-time coverage
as the fraction of test residuals ``≤ q̂``. The reward dict's
``meta`` block includes the per-instance ``covered`` flag so
downstream telemetry can aggregate it.
"""
from __future__ import annotations

import json
from typing import Any

from __ENV_PY__.data import CodeInstance, CodePrediction
from __ENV_PY__.sandbox import (
    DEFAULT_MEM_BYTES,
    DEFAULT_TIMEOUT_S,
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

DEFAULT_ALPHA: float = 0.1
DEFAULT_TIMEOUT_S_PER_CALL: float = DEFAULT_TIMEOUT_S
# D7-C ruling — same weights as math-algebra.
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "pass_rate": 0.7,
}


def _format_test_module(instance: CodeInstance) -> str:
    """Render the pytest module the sandbox runs.

    Each visible + hidden assertion becomes its own ``def test_NN()``
    so pytest -q counts them individually for the graded reward.
    """
    asserts = list(instance.visible_tests) + list(instance.hidden_tests)
    lines = [
        "from solution import *  # noqa: F401, F403",
        "",
    ]
    for i, a in enumerate(asserts):
        lines.append(f"def test_case_{i:03d}():")
        lines.append(f"    assert {a}")
        lines.append("")
    return "\n".join(lines)


def _is_format_valid(prediction: CodePrediction) -> bool:
    """``raw`` is JSON containing a non-empty ``code`` field."""
    if not prediction.raw:
        return bool(prediction.code.strip())
    try:
        data = json.loads(prediction.raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    return isinstance(data, dict) and bool(str(data.get("code", "")).strip())


def _is_compileable(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        compile(code, "<solution>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def score_components(
    prediction: CodePrediction,
    instance: CodeInstance,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_CALL,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``.

    Short-circuits aggressively: malformed JSON stops at
    ``format_valid``; un-compileable code stops at ``parse_valid``.
    Only survivors pay for the sandboxed pytest invocation.
    """
    components = {"format_valid": 0.0, "parse_valid": 0.0, "pass_rate": 0.0}

    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components

    code = prediction.code.strip() or prediction.raw
    if not _is_compileable(code):
        return components
    components["parse_valid"] = 1.0

    files = {
        "solution.py": code + "\n",
        "test_solution.py": _format_test_module(instance),
    }
    manifest = build_pytest_manifest(["test_solution.py"], timeout_s=timeout_s)
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=manifest,
        mem_bytes=mem_bytes,
    )
    total = len(instance.visible_tests) + len(instance.hidden_tests)
    counts = parse_pytest_q_summary(result.stdout)
    if total > 0:
        components["pass_rate"] = float(counts["passed"]) / float(total)
    return components


def compute_reward(
    prediction: CodePrediction,
    instance: CodeInstance,
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_CALL,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    conformal_quantile: float | None = None,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict.

    The optional ``conformal_quantile`` controls the per-instance
    ``covered`` flag in ``meta``: ``covered = (1 − reward) ≤ q̂``.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(
        prediction,
        instance,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
    )
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    meta: dict[str, Any] = {
        "weights": dict(w),
        "timeout_s": timeout_s,
        "confidence": float(prediction.confidence),
        "template": instance.template_name,
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
    "DEFAULT_TIMEOUT_S_PER_CALL",
    "DEFAULT_WEIGHTS",
    "score_components",
    "compute_reward",
]
