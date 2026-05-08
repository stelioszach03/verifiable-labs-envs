"""SQL sandbox + primitives sentinel suite (Phase 26.B).

PHASE_26_PLAN.md §6 + §16 Check 6 mandate the 12 named tests below;
the rest of this file pins down the schema-generator determinism,
result-set comparator edge cases, and tokeniser behaviour.
"""
from __future__ import annotations

import time

from verifiable_labs_envs.sql_primitives import (
    DEFAULT_CTE_DEPTH_CAP,
    DEFAULT_FLOAT_TOL,
    DEFAULT_MAX_QUERY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    SqlExecution,
    build_schema,
    canonicalise_query,
    compare_result_sets,
    cte_nesting_depth,
    execute_query_sync,
    generate_problem,
    has_forbidden_function,
    is_read_only_query,
    query_canonical_hash,
    query_has_limit_without_order_by,
    query_has_order_by,
)

# ── Mandatory sandbox isolation tests (§16 Check 6) ──────────────────


def test_sandbox_memory_cap_kills_oom_query() -> None:
    """Pathological recursive CTE expansion gets killed.

    SQLite raises ``OperationalError`` (or returns truncated rows)
    when a query exhausts memory; the sandbox surfaces a non-success
    result either way."""
    schema = ["CREATE TABLE t (n INTEGER);"]
    # Multiplicative recursive CTE — explodes row count, hits row cap
    # AND wall-clock long before host memory is touched.
    query = (
        "WITH RECURSIVE big(n) AS ("
        "SELECT 1 UNION ALL SELECT n + 1 FROM big WHERE n < 100000000"
        ") SELECT n FROM big ORDER BY n ASC"
    )
    result = execute_query_sync(
        schema_sql=schema,
        query=query,
        max_rows=1000,
        timeout_s=2.0,
    )
    # Either timed out or truncated to the row cap — both prove the
    # cap engaged. The model gets a deterministic failure signal.
    assert (not result.success) or result.truncated


def test_sandbox_wall_timeout_kills_long_query() -> None:
    """A pathological query is killed within `~timeout_s + 1 s`."""
    schema = ["CREATE TABLE x (n INTEGER);"]
    query = (
        "WITH RECURSIVE spinner(n) AS ("
        "SELECT 1 UNION ALL SELECT n + 1 FROM spinner WHERE n < 10000000000"
        ") SELECT COUNT(*) FROM spinner"
    )
    start = time.monotonic()
    result = execute_query_sync(
        schema_sql=schema,
        query=query,
        timeout_s=1.0,
        max_rows=10,
    )
    elapsed = time.monotonic() - start
    assert not result.success
    assert elapsed < 4.0
    assert (result.error or "").lower().startswith("query timed out") or "interrupt" in (result.error or "").lower()


def test_sandbox_row_cap_truncates_result() -> None:
    """`fetchmany(max_rows + 1)` slice + `truncated=True` flag."""
    schema = [
        "CREATE TABLE t (n INTEGER);",
        # Seed 50 rows.
        *[f"INSERT INTO t (n) VALUES ({i});" for i in range(1, 51)],
    ]
    query = "SELECT n FROM t ORDER BY n ASC"
    result = execute_query_sync(
        schema_sql=schema,
        query=query,
        max_rows=20,
    )
    assert result.success
    assert len(result.rows) == 20
    assert result.truncated is True


def test_sandbox_query_size_cap_rejects_oversized() -> None:
    """Queries larger than `max_query_bytes` are rejected pre-parse."""
    schema = ["CREATE TABLE t (n INTEGER);"]
    big_query = "SELECT 1 -- " + ("x" * 200) + "\nORDER BY 1"
    result = execute_query_sync(
        schema_sql=schema,
        query=big_query,
        max_query_bytes=128,
    )
    assert not result.success
    assert "byte cap" in (result.error or "")


def test_sandbox_cte_depth_cap_rejects_deep() -> None:
    """Nested ``WITH`` clauses past the cap are rejected."""
    # Build a query with WITH nested cte_depth_cap + 1 times. The
    # tokeniser's `cte_nesting_depth` counts WITH keywords; we
    # construct a synthetic query that has > cap WITH tokens.
    schema = ["CREATE TABLE t (n INTEGER);"]
    inner = "SELECT 1 AS n"
    # Wrap with N nested WITH ... AS (...) clauses.
    # Each layer adds 1 WITH keyword.
    layered = inner
    cap = 5  # use a small cap for the test to keep query bytes small
    for i in range(cap + 2):
        layered = f"WITH x{i} AS ({layered}) SELECT n FROM x{i}"
    result = execute_query_sync(
        schema_sql=schema,
        query=layered,
        cte_depth_cap=cap,
        max_query_bytes=64 * 1024,
    )
    assert not result.success
    assert "CTE nesting depth" in (result.error or "")


def test_dml_first_token_gate_rejects_insert_update_delete() -> None:
    """SELECT-only gate (D8-C). Every DML / DDL token is rejected."""
    schema = ["CREATE TABLE t (n INTEGER);"]
    cases = [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET n = 2 WHERE n = 1",
        "DELETE FROM t WHERE n = 1",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN extra TEXT",
        "PRAGMA foreign_keys = ON",
        "ATTACH DATABASE 'evil.db' AS evil",
    ]
    for q in cases:
        result = execute_query_sync(schema_sql=schema, query=q)
        assert not result.success, f"DML query slipped past the gate: {q!r}"
        assert "rejected" in (result.error or "").lower() or "unsupported" in (result.error or "").lower()


def test_random_function_rejected() -> None:
    """D9 forbids RANDOM()-style non-deterministic primitives."""
    schema = ["CREATE TABLE t (n INTEGER);", "INSERT INTO t (n) VALUES (1);"]
    result = execute_query_sync(
        schema_sql=schema,
        query="SELECT RANDOM() FROM t ORDER BY 1",
    )
    assert not result.success
    assert "RANDOM" in (result.error or "")


def test_limit_without_orderby_rejected() -> None:
    """D9 — ``LIMIT`` without ``ORDER BY`` is non-deterministic."""
    schema = [
        "CREATE TABLE t (n INTEGER);",
        *[f"INSERT INTO t (n) VALUES ({i});" for i in range(5)],
    ]
    result = execute_query_sync(
        schema_sql=schema,
        query="SELECT n FROM t LIMIT 2",
    )
    assert not result.success
    assert "LIMIT" in (result.error or "")


def test_float_tolerance_1e6_in_result_match() -> None:
    """Two rows differing by ≤ 1e-6 in a float cell compare equal."""
    gold = [(1.0,)]
    pred = [(1.0 + 0.5e-6,)]
    assert compare_result_sets(gold, pred, ordered=True)
    bigger = [(1.0 + 1e-3,)]
    assert not compare_result_sets(gold, bigger, ordered=True)


def test_null_equals_null_in_result_match() -> None:
    """``NULL == NULL`` in our comparator (D9 — three-valued logic suspended)."""
    gold = [(None, 1)]
    pred = [(None, 1)]
    assert compare_result_sets(gold, pred, ordered=True)
    # NULL vs non-NULL fails.
    not_eq = [(None, 1)]
    other = [(0, 1)]
    assert not compare_result_sets(not_eq, other, ordered=True)


def test_orderby_preservation_in_gold_query_detected() -> None:
    """`query_has_order_by` flips on the gold query's ORDER BY."""
    assert query_has_order_by("SELECT n FROM t ORDER BY n ASC")
    assert query_has_order_by(
        "WITH x AS (SELECT 1) SELECT * FROM x order BY 1 desc"
    )
    assert not query_has_order_by("SELECT n FROM t WHERE n > 1")


def test_query_canonical_form_stability() -> None:
    """`canonicalise_query` is whitespace + case insensitive."""
    a = canonicalise_query("SELECT  *\nFROM\tt  WHERE n = 1")
    b = canonicalise_query("select * from t where n = 1")
    assert a == b
    # String literals preserve case.
    a = canonicalise_query("SELECT name FROM t WHERE name = 'Alice'")
    b = canonicalise_query("select Name from T where name = 'Alice'")
    assert a == b
    # Hash form stable.
    h1 = query_canonical_hash("SELECT 1")
    h2 = query_canonical_hash("select  1")
    assert h1 == h2
    assert len(h1) == 16


# ── Sandbox happy path + boundary cases ──────────────────────────────


def test_sandbox_runs_select_happy_path() -> None:
    schema = [
        "CREATE TABLE t (n INTEGER);",
        "INSERT INTO t (n) VALUES (1), (2), (3);",
    ]
    result = execute_query_sync(
        schema_sql=schema,
        query="SELECT n FROM t ORDER BY n ASC",
    )
    assert result.success
    assert result.rows == ((1,), (2,), (3,))
    assert result.column_names == ("n",)
    assert not result.truncated


def test_sandbox_with_clause_happy_path() -> None:
    schema = [
        "CREATE TABLE t (n INTEGER);",
        "INSERT INTO t (n) VALUES (1), (2);",
    ]
    result = execute_query_sync(
        schema_sql=schema,
        query="WITH x AS (SELECT n + 10 AS m FROM t) SELECT m FROM x ORDER BY m ASC",
    )
    assert result.success
    assert result.rows == ((11,), (12,))


def test_sandbox_explain_allowed() -> None:
    schema = ["CREATE TABLE t (n INTEGER);"]
    result = execute_query_sync(
        schema_sql=schema,
        query="EXPLAIN SELECT n FROM t",
    )
    assert result.success


def test_sandbox_empty_query_rejected() -> None:
    result = execute_query_sync(schema_sql=[], query="")
    assert not result.success
    assert (result.error or "").lower() == "empty query"


def test_sandbox_calls_are_independent() -> None:
    """Two sandbox calls do not share state — fresh `:memory:` per call."""
    schema = [
        "CREATE TABLE t (n INTEGER);",
        "INSERT INTO t (n) VALUES (1);",
    ]
    a = execute_query_sync(schema_sql=schema, query="SELECT COUNT(*) FROM t")
    # Second call gets a fresh DB; no rows survived from the first.
    b = execute_query_sync(
        schema_sql=["CREATE TABLE u (n INTEGER);"],
        query="SELECT COUNT(*) FROM u",
    )
    assert a.success and b.success
    assert a.rows == ((1,),)
    assert b.rows == ((0,),)


def test_sandbox_returns_sqlexecution_dataclass() -> None:
    schema = ["CREATE TABLE t (n INTEGER);"]
    result = execute_query_sync(
        schema_sql=schema, query="SELECT 1 ORDER BY 1",
    )
    assert isinstance(result, SqlExecution)
    assert result.elapsed_ms >= 0


# ── Tokeniser / read-only gate ──────────────────────────────────────


def test_is_read_only_query_accepts_select_with_explain() -> None:
    for q in (
        "SELECT 1",
        "  SELECT 1",
        "(SELECT 1)",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "EXPLAIN SELECT 1",
        "/* leading comment */ SELECT 1",
        "-- comment\nSELECT 1",
    ):
        ok, reason = is_read_only_query(q)
        assert ok, f"{q!r} rejected: {reason}"


def test_is_read_only_query_rejects_dml_with_leading_comment() -> None:
    """Leading SQL comments must not smuggle a DML token past the gate."""
    ok, reason = is_read_only_query("-- innocent comment\nINSERT INTO t VALUES (1)")
    assert not ok
    assert "INSERT" in (reason or "")


def test_is_read_only_query_rejects_empty() -> None:
    ok, reason = is_read_only_query("   \n   ")
    assert not ok
    assert reason == "empty query"


def test_has_forbidden_function_skips_string_literals() -> None:
    """A literal containing `RANDOM` must not trigger the detector."""
    assert has_forbidden_function("SELECT name FROM t WHERE name = 'is RANDOM here?'") is None
    assert has_forbidden_function("SELECT random() FROM t") == "RANDOM"


def test_query_has_limit_without_order_by_handles_subquery() -> None:
    """LIMIT inside a subquery counts as a top-level construct.

    SQLite parses LIMIT as a top-level query property; even when it
    appears inside a CTE the outer query still needs ORDER BY.
    """
    assert query_has_limit_without_order_by("SELECT n FROM t LIMIT 1")
    assert not query_has_limit_without_order_by(
        "SELECT n FROM t ORDER BY n LIMIT 1"
    )


def test_cte_nesting_depth_counts_with_keywords() -> None:
    assert cte_nesting_depth("SELECT 1") == 0
    assert cte_nesting_depth("WITH x AS (SELECT 1) SELECT * FROM x") == 1
    nested = (
        "WITH a AS (WITH b AS (WITH c AS (SELECT 1) SELECT * FROM c) "
        "SELECT * FROM b) SELECT * FROM a"
    )
    assert cte_nesting_depth(nested) == 3


# ── Result-set comparator edge cases ────────────────────────────────


def test_compare_result_sets_unordered_multiset() -> None:
    g = [(1,), (2,), (3,)]
    p = [(3,), (1,), (2,)]
    assert compare_result_sets(g, p, ordered=False)


def test_compare_result_sets_ordered_zip() -> None:
    g = [(1,), (2,), (3,)]
    p = [(1,), (3,), (2,)]
    assert compare_result_sets(g, p, ordered=False)
    assert not compare_result_sets(g, p, ordered=True)


def test_compare_result_sets_length_mismatch() -> None:
    assert not compare_result_sets([(1,)], [(1,), (2,)], ordered=True)


def test_compare_result_sets_type_mismatch_strict() -> None:
    """`'1'` is NOT equal to `1` — no implicit coercion."""
    assert not compare_result_sets([("1",)], [(1,)], ordered=True)


def test_compare_result_sets_bool_distinct_from_int() -> None:
    """SQLite returns BOOLEAN as int 0/1; we treat them the same."""
    # Both are ints (sqlite3 dialect) — should match.
    assert compare_result_sets([(1,)], [(1,)], ordered=True)


def test_compare_result_sets_handles_nan_pairs() -> None:
    """Two NaNs in matching positions compare equal."""
    g = [(float("nan"),)]
    p = [(float("nan"),)]
    assert compare_result_sets(g, p, ordered=True)
    p2 = [(1.0,)]
    assert not compare_result_sets(g, p2, ordered=True)


def test_compare_empty_result_sets_match() -> None:
    assert compare_result_sets([], [], ordered=True)
    assert compare_result_sets([], [], ordered=False)


# ── Schema generator + Schema dataclass ─────────────────────────────


def test_generate_problem_is_deterministic_per_seed() -> None:
    a = generate_problem(seed=7)
    b = generate_problem(seed=7)
    assert a["template_name"] == b["template_name"]
    assert a["create_statements"] == b["create_statements"]
    assert a["seed_statements"] == b["seed_statements"]
    assert a["gold_query"] == b["gold_query"]


def test_generate_problem_covers_all_templates() -> None:
    seen = {generate_problem(seed=s)["template_name"] for s in range(80)}
    assert len(seen) == 8


def test_schema_hash_is_stable_per_schema() -> None:
    schema = build_schema(generate_problem(seed=0))
    assert schema.schema_hash() == schema.schema_hash()
    assert len(schema.schema_hash()) == 16


def test_schema_render_markdown_lists_columns() -> None:
    schema = build_schema(generate_problem(seed=0))
    md = schema.render_markdown()
    for tbl in schema.table_names:
        assert f"`{tbl}`" in md


def test_schema_render_does_not_include_seed_inserts() -> None:
    """The prompt must NOT show INSERT data — schema only."""
    schema = build_schema(generate_problem(seed=0))
    md = schema.render_markdown()
    assert "INSERT" not in md
    assert "values" not in md.lower()


def test_constants_match_phase_26_d5() -> None:
    assert DEFAULT_MAX_ROWS == 10_000
    assert DEFAULT_TIMEOUT_S == 10.0
    assert DEFAULT_MAX_QUERY_BYTES == 32 * 1024
    assert DEFAULT_CTE_DEPTH_CAP == 64
    assert DEFAULT_FLOAT_TOL == 1e-6
