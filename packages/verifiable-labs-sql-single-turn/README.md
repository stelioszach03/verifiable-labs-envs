# verifiable-labs-sql-single-turn

Single-turn text-to-SQL RL environment from the Verifiable Labs
catalogue. Each instance hands the model a natural-language question
+ schema description (CREATE TABLE statements + per-table column
rosters) and expects a JSON envelope with a `query` field. The query
runs against an in-process SQLite sandbox seeded with the instance's
INSERT data; the result-set is compared against the gold result.

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing a `query` field                      |
| `parse_valid`   | 0.20   | Extracted query passes the SELECT-only gate                              |
| `correctness`   | 0.70   | Result-set equality (ordered if the gold has ORDER BY)                   |

8 procedural schema templates spanning filter / aggregate / join /
groupby / subquery / CTE / date arithmetic — `EFFECTIVE_INSTANCES >
1.5e23`, well above the contamination-resistance gate.

## Install

```bash
pip install verifiable-labs-sql-single-turn
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
