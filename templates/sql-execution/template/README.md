# __ENV_ID__

Verifiable Labs SQL RL environment for **__DOMAIN__**.

This package was scaffolded from
``templates/sql-execution/`` via
``scripts/create_env.py __ENV_ID__ --template sql-execution --domain "__DOMAIN__"``.

## What this env does

The env hands the solver a **natural-language question + a schema
description** (CREATE TABLE statements + per-table column rosters)
and expects a JSON envelope with a single ``query`` field
containing a SQLite SELECT. The query runs in an in-process
sandbox seeded with the instance's CREATE / INSERT statements; the
result-set is compared against the gold result-set.

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing a `query` field                      |
| `parse_valid`   | 0.20   | Extracted query passes the SELECT-only gate AND parses via SQLite        |
| `correctness`   | 0.70   | Result-set equality with gold rows (ordered if gold has ORDER BY)        |

The shared sandbox + schema generator + comparator live at
``verifiable_labs_envs.sql_primitives``; this env imports them
verbatim — D10-A locked in PHASE_26_PLAN.md.

## Filling in the scaffold

Replace the `NotImplementedError` stubs in:

- ``__ENV_PY__/data.py`` — `generate_problem(seed, **hyperparams)`
  returns a dict with `prompt`, `create_statements`,
  `seed_statements`, `gold_query`, `gold_query_is_ordered`,
  `gold_result_rows`, `template_name`. Use
  `numpy.random.default_rng(seed)` for reproducibility; ensure
  `EFFECTIVE_INSTANCES > 1e15` (procedural-regeneration gate).
- ``__ENV_PY__/env.py`` — adjust hyperparams (max_rows, timeout_s,
  alpha) if your env needs tighter or looser bounds.

The scoring kernel and adapter need no edits in most cases; the
default JSON envelope (``{"query": "...", "confidence": <float>}``)
and the result-set comparator cover the common case.

## Running

```bash
python scripts/validate_env.py environments/__ENV_PY__/   # contract checks
pytest                                                     # unit + reward + sandbox tests
```

## Why a separate template family

The inverse-problem template hard-codes forward operators + NMSE;
the symbolic-math template hard-codes SymPy equivalence; the
code-execution template hard-codes a sandboxed pytest runner; the
tool-calling template hard-codes OpenAI function-calling. None
applies to text-to-SQL. The sql-execution family swaps in:

- A `SqlInstance` shape carrying `schema` (CREATE/INSERT) +
  `gold_query` + `gold_result_rows` + `gold_query_is_ordered`.
- A reward kernel that delegates the heavy lifting to
  `verifiable_labs_envs.sql_primitives.execute_query_sync` +
  `compare_result_sets`.
- A pre-flight gate that rejects DML / non-deterministic functions
  / `LIMIT` without `ORDER BY` (D8-C + D9 locked).

## Trusted-input scope

This env's sandbox runs SQLite queries in-process under the D2-A
"sqlite3 + watchdog" mechanism (PHASE_26_PLAN.md §6). The locked
guarantee is *isolation between concurrent customer calls* via the
ephemeral `:memory:` connection per call — not *defence against a
determined attacker who has compromised an API key*. Sandbox
upgrade considerations (subprocess + rlimits) live in
``code_execution_sandbox`` from Phase 24 if a future deployment
requires them.
