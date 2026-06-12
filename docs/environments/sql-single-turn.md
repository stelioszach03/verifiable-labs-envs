# `sql-single-turn`

**Single-turn text-to-SQL with sandboxed result-set comparison.**
Given a natural-language question + a schema description (CREATE
TABLE statements + per-table column rosters), the model returns a
SQLite SELECT in a JSON envelope. The query runs against an
ephemeral in-process SQLite sandbox seeded with the instance's
data; the result-set is compared against the gold result.

This is the first member of the **sql-execution** template family
introduced in Phase 26.

## Problem

| | |
|---|---|
| input | natural-language question + CREATE TABLE statements + column rosters |
| output | JSON `{"query": "<SQLite SELECT>", "confidence": <float in [0, 1]>}` |
| gold | hidden gold query + cached gold result-set |
| dialect | SQLite (D1-A locked); SELECT / WITH / EXPLAIN only |

Eight procedural schema templates spanning the canonical text-to-SQL
distribution:

| # | Template                  | Sample query shape                                                |
|---|---------------------------|--------------------------------------------------------------------|
| 1 | `single_table_filter`     | `SELECT * FROM products WHERE price < ? ORDER BY price ASC`      |
| 2 | `single_table_aggregate`  | `SELECT category, SUM(amount) FROM sales GROUP BY category`        |
| 3 | `two_table_join`          | `SELECT c.name FROM customers c JOIN orders o ON ... GROUP BY ...` |
| 4 | `three_table_join`        | `SELECT p.name, SUM(i.qty) FROM orders o JOIN items i ... JOIN ...` |
| 5 | `groupby_having`          | `SELECT type, COUNT(*) FROM events GROUP BY type HAVING COUNT(*) > ?` |
| 6 | `subquery_filter`         | `SELECT name FROM employees WHERE salary > (SELECT AVG(salary)...)` |
| 7 | `cte_aggregate`           | `WITH daily AS (...) SELECT date, SUM(...) FROM daily ORDER BY date` |
| 8 | `date_arithmetic`         | `SELECT STRFTIME('%H', ts), COUNT(*) FROM sessions WHERE ts BETWEEN ? AND ?` |

`EFFECTIVE_INSTANCES > 1.5 × 10²³`, well above the 1 × 10¹⁵
contamination-resistance gate.

## Variants

- [`sql-single-turn`](#) — single-turn, this page.
- [`sql-multiturn`](sql-multiturn.md) — 3-turn dialogue with
  verifier feedback (parse status + row-count diagnostics, gold
  result-set held out per R10).

## Schema

```json
{
  "query": "SELECT category, SUM(amount) FROM sales GROUP BY category ORDER BY category ASC",
  "confidence": 0.85
}
```

## Reward decomposition

```
reward = 0.10 · format_valid    (output is parseable JSON
                                  with a `query` field)
       + 0.20 · parse_valid     (extracted query passes the
                                  D8-C SELECT-only gate)
       + 0.70 · correctness     (D4-A: result-set equality with
                                  gold rows; ordered iff gold has ORDER BY)
```

Same weight structure as math / code / tool-calling — preserves
cross-env reward distribution comparability.

## Sandbox guarantees (D5 locked)

The query runs through `verifiable_labs_envs.sql_primitives.execute_query_sync`.
PHASE_26_PLAN.md §6 locks:

| Surface             | Guarantee                                                                                  | Sentinel test                                     |
|---------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------|
| DML rejection       | INSERT / UPDATE / DELETE / DROP / ALTER / PRAGMA all fail at the first-token gate          | `test_dml_first_token_gate_rejects_insert_update_delete` |
| Memory cap          | 256 MB virtual address space cap                                                           | `test_sandbox_memory_cap_kills_oom_query`         |
| Wall-clock          | 10 s timeout via `set_progress_handler` + `connection.interrupt()`                          | `test_sandbox_wall_timeout_kills_long_query`      |
| CPU                 | Implicit via wall-clock + row cap (SQLite's interrupt fires at every 1 000 VDBE ops)        | (covered by wall-clock test)                      |
| Row cap             | 10 000-row truncation; `truncated=True` flag                                               | `test_sandbox_row_cap_truncates_result`           |
| Query length        | 32 KB byte cap before parse                                                                | `test_sandbox_query_size_cap_rejects_oversized`   |
| CTE depth           | 64-deep nested `WITH` reject at parse                                                       | `test_sandbox_cte_depth_cap_rejects_deep`         |
| Determinism (RANDOM)| `RANDOM()` rejected at parse                                                               | `test_random_function_rejected`                   |
| Determinism (LIMIT) | `LIMIT` without `ORDER BY` rejected                                                         | `test_limit_without_orderby_rejected`             |
| In-process isolation| Two sandbox calls do NOT share state (fresh `:memory:` per call)                            | `test_sandbox_calls_are_independent`              |

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("sql-single-turn")
inst = env.generate_instance(seed=42)
print(inst.prompt)
```

## Tests

The repo's `tests/test_sql_sandbox.py` runs the platform-level
sandbox + primitives suite; `tests/test_sql_single_turn.py` covers
the env's templates + reward kernel.
