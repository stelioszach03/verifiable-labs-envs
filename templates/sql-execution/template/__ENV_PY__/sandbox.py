"""Sandbox + comparator helpers for __ENV_ID__.

Re-exports the platform-level shared library at
``verifiable_labs_envs.sql_primitives``. Per-env scaffolds keep this
thin indirection so a customised execution policy (different
timeout / row cap, additional pre-flight checks) can be wired in
without touching the env / reward code.
"""
from __future__ import annotations

from verifiable_labs_envs.sql_primitives import (
    DEFAULT_CTE_DEPTH_CAP,
    DEFAULT_FLOAT_TOL,
    DEFAULT_MAX_QUERY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    Schema,
    SqlExecution,
    build_schema,
    canonicalise_query,
    compare_result_sets,
    cte_nesting_depth,
    execute_query_sync,
    has_forbidden_function,
    is_read_only_query,
    query_canonical_hash,
    query_has_limit_without_order_by,
    query_has_order_by,
)

__all__ = [
    "DEFAULT_CTE_DEPTH_CAP",
    "DEFAULT_FLOAT_TOL",
    "DEFAULT_MAX_QUERY_BYTES",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TIMEOUT_S",
    "Schema",
    "SqlExecution",
    "build_schema",
    "canonicalise_query",
    "compare_result_sets",
    "cte_nesting_depth",
    "execute_query_sync",
    "has_forbidden_function",
    "is_read_only_query",
    "query_canonical_hash",
    "query_has_limit_without_order_by",
    "query_has_order_by",
]
