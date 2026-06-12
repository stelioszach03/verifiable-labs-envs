"""Conformal-calibrated reward for __ENV_ID__.

The reward function combines three components in ``[0, 1]``:

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Final non-tool message is parseable JSON.                                 |
| `parse_valid`   | 0.20   | Every recorded tool-call had dict args AND the final submission parses.   |
| `correctness`   | 0.70   | D2-C: 0.30 · action_validity + 0.70 · final_state_match (template).       |

The conformal coverage layer reuses
``verifiable_labs_envs.conformal.split_conformal_quantile`` directly.
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import ToolCallingInstance, ToolCallingPrediction
from __ENV_PY__.tools import canonical_action_hash

DEFAULT_ALPHA: float = 0.1
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correctness": 0.7,
}
ACTION_VALIDITY_WEIGHT: float = 0.30
STATE_MATCH_WEIGHT: float = 0.70

_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _final_text_parses(text: str) -> bool:
    if not text or not text.strip():
        return False
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
            return True
    return False


def _is_format_valid(prediction: ToolCallingPrediction) -> bool:
    return _final_text_parses(prediction.final_text)


def _is_parse_valid(prediction: ToolCallingPrediction) -> bool:
    """Every tool call had dict args AND the final submission parses."""
    if not _final_text_parses(prediction.final_text):
        return False
    for call in prediction.tool_calls:
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (json.JSONDecodeError, ValueError):
                return False
        if not isinstance(args, dict):
            return False
    return True


def _action_validity(prediction: ToolCallingPrediction) -> float:
    if not prediction.tool_calls:
        return 0.0
    ok = sum(
        1 for call in prediction.tool_calls
        if not isinstance(call.get("result"), dict) or "error" not in call["result"]
    )
    return float(ok) / float(len(prediction.tool_calls))


def _check_gold_state(
    state: Any,
    instance: ToolCallingInstance,
) -> bool:
    """Per-template predicate over the final workspace state.

    TODO: switch on ``instance.template_name`` and consume
    ``instance.gold_spec`` to build the predicate. Default
    implementation always returns False so the scaffold scores
    zero on `correctness` until customised.
    """
    del state, instance
    return False


def _correctness(
    prediction: ToolCallingPrediction,
    instance: ToolCallingInstance,
) -> float:
    av = _action_validity(prediction)
    sm = 1.0 if _check_gold_state(prediction.final_state, instance) else 0.0
    return ACTION_VALIDITY_WEIGHT * av + STATE_MATCH_WEIGHT * sm


def score_components(
    prediction: ToolCallingPrediction,
    instance: ToolCallingInstance,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``."""
    components = {
        "format_valid": 1.0 if _is_format_valid(prediction) else 0.0,
        "parse_valid": 0.0,
        "correctness": 0.0,
    }
    if components["format_valid"] == 0.0:
        return components
    components["parse_valid"] = 1.0 if _is_parse_valid(prediction) else 0.0
    if components["parse_valid"] == 0.0:
        return components
    components["correctness"] = _correctness(prediction, instance)
    return components


def compute_reward(
    prediction: ToolCallingPrediction,
    instance: ToolCallingInstance,
    *,
    weights: dict[str, float] | None = None,
    conformal_quantile: float | None = None,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(prediction, instance)
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    meta: dict[str, Any] = {
        "weights": dict(w),
        "n_tool_calls": len(prediction.tool_calls),
        "template": instance.template_name,
        "outbox_count": len(prediction.final_state.outbox),
        "files_written": sorted(prediction.final_state.files),
        "action_hash": canonical_action_hash(list(prediction.tool_calls)),
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
    "ACTION_VALIDITY_WEIGHT",
    "STATE_MATCH_WEIGHT",
    "score_components",
    "compute_reward",
]
