"""Conformal-calibrated reward for __ENV_ID__.

The reward function combines three components in ``[0, 1]``:

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing a `query` field                      |
| `parse_valid`   | 0.20   | Extracted query passes the SELECT-only gate                              |
| `correctness`   | 0.70   | Result-set equality with gold rows (ordered if gold has ORDER BY)        |

The conformal coverage layer reuses
``verifiable_labs_envs.conformal.split_conformal_quantile`` directly.
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import SqlInstance, SqlPrediction
from __ENV_PY__.sandbox import (
    DEFAULT_FLOAT_TOL,
    DEFAULT_MAX_QUERY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    compare_result_sets,
    execute_query_sync,
    is_read_only_query,
    query_canonical_hash,
)

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


def _is_format_valid(prediction: SqlPrediction) -> bool:
    if prediction.raw:
        data = _extract_envelope(prediction.raw)
        if not isinstance(data, dict):
            return False
        return bool(str(data.get("query", "")).strip())
    return bool(prediction.query.strip())


def _is_parse_valid(prediction: SqlPrediction) -> bool:
    query = prediction.query.strip() or _query_from_raw(prediction.raw)
    if not query:
        return False
    ok, _ = is_read_only_query(query)
    return ok


def _query_from_raw(raw: str) -> str:
    data = _extract_envelope(raw)
    if not isinstance(data, dict):
        return ""
    return str(data.get("query", "")).strip()


def score_components(
    prediction: SqlPrediction,
    instance: SqlInstance,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
    float_tol: float = DEFAULT_FLOAT_TOL,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``."""
    components = {"format_valid": 0.0, "parse_valid": 0.0, "correctness": 0.0}
    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components
    components["parse_valid"] = 1.0 if _is_parse_valid(prediction) else 0.0
    if components["parse_valid"] == 0.0:
        return components

    query = prediction.query.strip() or _query_from_raw(prediction.raw)
    schema_sql = list(instance.schema.create_statements + instance.schema.seed_statements)
    result = execute_query_sync(
        schema_sql=schema_sql,
        query=query,
        max_rows=max_rows,
        timeout_s=timeout_s,
        max_query_bytes=max_query_bytes,
    )
    if not result.success:
        return components
    match = compare_result_sets(
        instance.gold_result_rows,
        result.rows,
        ordered=instance.gold_query_is_ordered,
        float_tol=float_tol,
    )
    components["correctness"] = 1.0 if match else 0.0
    return components


def compute_reward(
    prediction: SqlPrediction,
    instance: SqlInstance,
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
    conformal_quantile: float | None = None,
    float_tol: float = DEFAULT_FLOAT_TOL,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(
        prediction, instance,
        timeout_s=timeout_s, max_rows=max_rows,
        max_query_bytes=max_query_bytes, float_tol=float_tol,
    )
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    query = prediction.query.strip() or _query_from_raw(prediction.raw)
    meta: dict[str, Any] = {
        "weights": dict(w),
        "template": instance.template_name,
        "schema_hash": instance.schema.schema_hash(),
        "query_hash": query_canonical_hash(query) if query else "0" * 16,
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
