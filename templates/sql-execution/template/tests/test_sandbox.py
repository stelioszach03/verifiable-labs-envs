"""Sandbox-primitive contract tests for __ENV_ID__.

The env's reward kernel relies on
:func:`verifiable_labs_envs.sql_primitives.execute_query_sync`. The
local re-export in ``__ENV_PY__.sandbox`` must hand back the same
surface; the platform-level isolation suite lives in the parent
repo's ``tests/test_sql_sandbox.py``.
"""
from __future__ import annotations

from __ENV_PY__.sandbox import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    SqlExecution,
    compare_result_sets,
    execute_query_sync,
    is_read_only_query,
)


def test_re_exports_match_platform_defaults():
    assert DEFAULT_MAX_ROWS == 10_000
    assert DEFAULT_TIMEOUT_S == 10.0


def test_sandbox_smoke_runs_select():
    schema = ["CREATE TABLE t (n INTEGER);", "INSERT INTO t (n) VALUES (1), (2);"]
    result = execute_query_sync(
        schema_sql=schema, query="SELECT n FROM t ORDER BY n ASC",
    )
    assert isinstance(result, SqlExecution)
    assert result.success
    assert result.rows == ((1,), (2,))


def test_sandbox_rejects_dml():
    result = execute_query_sync(
        schema_sql=["CREATE TABLE t (n INTEGER);"],
        query="INSERT INTO t VALUES (99)",
    )
    assert not result.success


def test_compare_result_sets_basic():
    assert compare_result_sets([(1,), (2,)], [(1,), (2,)], ordered=True)
    assert compare_result_sets([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not compare_result_sets([(1,)], [(1,), (2,)], ordered=True)


def test_is_read_only_query_returns_tuple():
    ok, _ = is_read_only_query("SELECT 1")
    assert ok is True
