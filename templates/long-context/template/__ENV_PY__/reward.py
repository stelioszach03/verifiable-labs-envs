"""Conformal-calibrated reward for __ENV_ID__.

The reward function combines three components in ``[0, 1]``:

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing an `answer` field                    |
| `parse_valid`   | 0.20   | Extracted answer is non-empty                                            |
| `correctness`   | 0.70   | Substring match against the gold needle (case-insensitive — D3-A)        |

The conformal coverage layer reuses
``verifiable_labs_envs.conformal.split_conformal_quantile`` directly.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from __ENV_PY__.data import NeedleInstance, NeedlePrediction
from __ENV_PY__.needle import exact_match

DEFAULT_ALPHA: float = 0.1
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correctness": 0.7,
}

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
    if prediction.raw:
        data = _extract_envelope(prediction.raw)
        if not isinstance(data, dict):
            return False
        return bool(str(data.get("answer", "")).strip())
    return bool(prediction.answer.strip())


def _is_parse_valid(prediction: NeedlePrediction) -> bool:
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
    gold = instance.metadata.get("needle_token", instance.needle_text)
    components["correctness"] = (
        1.0 if exact_match(prediction.answer, gold) else 0.0
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
        "template": instance.template_name,
        "position_mode": instance.position_mode,
        "context_token_count": instance.corpus.total_tokens(),
        "needle_doc_id": instance.needle_anchor.document_id,
        "completion_hash": completion_hash,
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
    "DEFAULT_WEIGHTS",
    "score_components",
    "compute_reward",
]
