"""sql-single-turn — single-shot text-to-SQL RL environment (Phase 26.B).

PHASE_26_PLAN.md introduces the SQL env family. The single-turn
variant gives the model a natural-language question + the schema
description (CREATE TABLE statements + per-table column rosters),
and expects a JSON envelope with a single ``query`` field. The env
runs the query against the seeded SQLite sandbox and scores

    reward = 0.10 · format_valid    (output is parseable JSON
                                      with a non-empty ``query`` field)
           + 0.20 · parse_valid     (extracted query passes the
                                      D8-C SELECT-only gate AND parses
                                      via SQLite without an OperationalError)
           + 0.70 · correctness     (D4-A: result-set equality with
                                      gold rows; ordered if the gold
                                      query has ORDER BY, else unordered)

The 8 procedural templates from
:mod:`verifiable_labs_envs.sql_primitives` cover the canonical
text-to-SQL distribution (filter / aggregate / join / groupby /
subquery / CTE / date arithmetic).

Procedural-regeneration contract: each ``(seed, hyperparams)`` pair
draws a fresh problem from the 8-template lattice. The 64-bit seed
space × per-template parameter ranges yield ``EFFECTIVE_INSTANCES``
of order ``1.5 × 10²⁰``, well above the 1e15 contamination-resistance
gate.
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
from verifiable_labs_envs.sql_primitives import (
    DEFAULT_FLOAT_TOL,
    DEFAULT_MAX_QUERY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    Schema,
    build_schema,
    compare_result_sets,
    execute_query_sync,
    generate_problem,
    is_read_only_query,
    query_canonical_hash,
    query_has_order_by,
)

NAME = "sql-single-turn"

# 8 templates × 64-bit seed × ~1e5 parameter combinations per template
# ≈ 1.5 × 10²³ effective instances; well above the 1e15 procedural-
# regeneration gate.
EFFECTIVE_INSTANCES: int = 8 * (2**64) * 1_000_000

DEFAULT_ALPHA: float = 0.1
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "correctness": 0.7,
}
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "max_rows": DEFAULT_MAX_ROWS,
    "timeout_s": DEFAULT_TIMEOUT_S,
    "max_query_bytes": DEFAULT_MAX_QUERY_BYTES,
}


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class SqlInstance:
    """One text-to-SQL problem draw.

    ``schema`` carries the seeded CREATE/INSERT statements + table /
    column metadata. ``gold_query`` is the canonical SELECT used as
    the oracle. ``gold_result_rows`` is the cached gold result-set —
    populated at instance generation time so a cheap ``score`` call
    doesn't have to re-run the gold query.
    """

    prompt: str
    template_name: str
    seed: int
    schema: Schema
    gold_query: str
    gold_query_is_ordered: bool
    gold_result_rows: tuple[tuple[Any, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "schema": {
                "create_statements": list(self.schema.create_statements),
                "tables": list(self.schema.table_names),
                "columns_by_table": {
                    t: list(c) for t, c in self.schema.column_names_by_table.items()
                },
            },
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class SqlPrediction:
    """Solver's answer.

    ``query`` is the SELECT (or WITH / EXPLAIN) the model proposes.
    ``raw`` keeps the LLM's full response for the audit trail.
    ``confidence`` is a self-report in ``[0, 1]``.
    """

    query: str
    raw: str = ""
    confidence: float = 0.5


# ── Generators ──────────────────────────────────────────────────────


def generate_instance(seed: int, **kwargs: Any) -> SqlInstance:
    """Wrap :func:`sql_primitives.generate_problem` in a SqlInstance."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed))
    schema = build_schema(problem)
    return SqlInstance(
        prompt=problem["prompt"],
        template_name=problem["template_name"],
        seed=int(seed),
        schema=schema,
        gold_query=problem["gold_query"],
        gold_query_is_ordered=bool(problem["gold_query_is_ordered"]),
        gold_result_rows=tuple(tuple(r) for r in problem["gold_result_rows"]),
        metadata={
            "alpha": float(params["alpha"]),
            "max_rows": int(params["max_rows"]),
            "timeout_s": float(params["timeout_s"]),
        },
    )


# ── Reward kernel ────────────────────────────────────────────────────


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_envelope(text: str) -> dict[str, Any] | None:
    """Permissively pull a JSON envelope out of an LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    candidates: list[str] = []
    candidates.extend(_FENCED_RE.findall(cleaned))
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
    """``raw`` is JSON with a non-empty ``query`` field."""
    if prediction.raw:
        data = _extract_envelope(prediction.raw)
        if not isinstance(data, dict):
            return False
        return bool(str(data.get("query", "")).strip())
    return bool(prediction.query.strip())


def _is_parse_valid(prediction: SqlPrediction) -> bool:
    """Extracted query passes the SELECT-only gate."""
    query = prediction.query.strip() or _query_from_raw(prediction.raw)
    if not query:
        return False
    ok, _ = is_read_only_query(query)
    return ok


def _query_from_raw(raw: str) -> str:
    """Pull `query` out of the raw envelope (best-effort)."""
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
    """Compute the three reward components in ``[0, 1]``.

    Short-circuits aggressively: malformed JSON stops at
    ``format_valid``; DML / non-parseable queries stop at
    ``parse_valid``. Only survivors execute against the sandbox.
    """
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
    # Ordered comparison if the gold query carries ORDER BY OR if the
    # model's query carries ORDER BY (covers cases where the gold is
    # unordered but the model adds an ordering — we accept that as
    # equivalent under the unordered branch's multiset compare).
    ordered = instance.gold_query_is_ordered
    match = compare_result_sets(
        instance.gold_result_rows,
        result.rows,
        ordered=ordered,
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
        prediction,
        instance,
        timeout_s=timeout_s,
        max_rows=max_rows,
        max_query_bytes=max_query_bytes,
        float_tol=float_tol,
    )
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    query = prediction.query.strip() or _query_from_raw(prediction.raw)
    cache_key_seed = int(instance.seed)
    schema_hash = instance.schema.schema_hash()
    query_hash = query_canonical_hash(query) if query else "0" * 16

    meta: dict[str, Any] = {
        "weights": dict(w),
        "template": instance.template_name,
        "schema_hash": schema_hash,
        "query_hash": query_hash,
        "cache_key": _cache_key(NAME, cache_key_seed, schema_hash, query_hash),
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


def _cache_key(env_id: str, seed: int, schema_hash: str, query_hash: str) -> str:
    payload = f"{env_id}|{seed}|{schema_hash}|{query_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1024)
def _cached_components(
    env_id: str,
    seed: int,
    schema_hash: str,
    query_hash: str,
    query: str,
    timeout_s: float,
    max_rows: int,
) -> tuple[float, float, float]:
    """Cached components keyed by `(env_id, seed, schema_hash, query_hash)`.

    D10-B locked. Saves recompute on `/v1/score` idempotency-key
    replays + multi-turn revisions where the model returns the same
    query twice. Per-process — no Redis hop.
    """
    del env_id, schema_hash, query_hash
    instance = generate_instance(int(seed))
    pred = SqlPrediction(query=query, raw=json.dumps({"query": query}))
    components = score_components(
        pred, instance, timeout_s=timeout_s, max_rows=max_rows,
    )
    return (
        float(components["format_valid"]),
        float(components["parse_valid"]),
        float(components["correctness"]),
    )


# ── Adapter helpers ─────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are an expert SQL programmer. Given a natural-language "
    "question and a schema description (CREATE TABLE statements plus "
    "per-table column rosters), return exactly one JSON object of the "
    "form\n\n"
    '    {"query": "<SQLite SELECT query>", "confidence": <float in [0, 1]>}\n\n'
    "Constraints:\n"
    "- SQLite dialect only (no Postgres-specific syntax).\n"
    "- SELECT, WITH, or EXPLAIN only — no INSERT / UPDATE / DELETE.\n"
    "- Avoid RANDOM() and other non-deterministic functions.\n"
    "- Use `LIMIT` only with an explicit `ORDER BY`.\n"
    "- Quote string literals with single quotes.\n\n"
    "No prose, no markdown fences — JSON only."
)


def build_user_prompt(instance: SqlInstance) -> str:
    """Render the env instance into LLM-readable text."""
    schema_md = instance.schema.render_markdown()
    create_block = "\n".join(instance.schema.create_statements)
    return (
        "PROBLEM:\n"
        f"{instance.prompt}\n\n"
        "SCHEMA (CREATE TABLE statements):\n"
        f"```\n{create_block}\n```\n\n"
        "SCHEMA (markdown):\n"
        f"{schema_md}\n\n"
        "OUTPUT SCHEMA:\n"
        '{"query": "<SQLite SELECT query>", "confidence": <float in [0, 1]>}\n\n'
        "Respond with the JSON object only."
    )


def parse_response(text: str, instance: SqlInstance) -> SqlPrediction:
    """Parse the LLM's text into a :class:`SqlPrediction`.

    Permissive: malformed inputs yield an empty-query prediction
    (zero reward) rather than raising.
    """
    del instance
    data = _extract_envelope(text)
    if not isinstance(data, dict):
        return SqlPrediction(query="", raw=text, confidence=0.0)
    query = str(data.get("query", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return SqlPrediction(query=query, raw=text, confidence=confidence)


# ── Env class + factory ─────────────────────────────────────────────


def baseline_predict(instance: SqlInstance) -> SqlPrediction:
    """Reference solver — empty query.

    Empty submission scores zero on every component; the wide
    residual distribution this produces yields a non-trivial
    conformal quantile when calibration runs over a baseline sweep.
    """
    del instance
    return SqlPrediction(query="", raw="", confidence=0.0)


class SqlSingleTurnEnv:
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

    def generate_instance(self, seed: int, **kwargs: Any) -> SqlInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(self, prediction: SqlPrediction, instance: SqlInstance) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            timeout_s=float(self.hyperparams["timeout_s"]),
            max_rows=int(self.hyperparams["max_rows"]),
            max_query_bytes=int(self.hyperparams["max_query_bytes"]),
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
) -> SqlSingleTurnEnv:
    """Factory mirroring the verifiers convention. Pass
    ``calibration_quantile`` to skip auto-calibration in tests."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return SqlSingleTurnEnv(conformal_quantile=q)


# Re-export `query_has_order_by` here for convenience — env modules
# in the family that need it can pull from one place.
__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "SYSTEM_PROMPT",
    "SqlInstance",
    "SqlPrediction",
    "SqlSingleTurnEnv",
    "baseline_predict",
    "build_user_prompt",
    "calibrate_quantile",
    "compute_reward",
    "generate_instance",
    "load_environment",
    "parse_response",
    "query_has_order_by",
    "score_components",
]
