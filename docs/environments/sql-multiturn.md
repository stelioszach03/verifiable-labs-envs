# `sql-multiturn`

**Multi-turn variant of `sql-single-turn` with verifier feedback
between turns.** Same problem distribution, same reward kernel,
same SQLite sandbox. Only the rollout protocol differs.

## Rollout

| Turn | What the model sees | What it returns |
|---|---|---|
| 1 | Question + schema description (CREATE TABLE statements + column rosters) | First SELECT |
| 2 | Parse status + row-count diagnostics (no gold rows, R10) | Revised SELECT |
| 3 | Same — final query, scored against gold result-set | Final SELECT |

The gold result-set is **never** shown to the model (R10 carry-
over). The only feedback channel between turns is a short string
of the form:

```
FEEDBACK on your previous turn:
Your query returned 7 row(s); the expected result has 5 row(s).
Review aggregation, filters, and ordering.
```

## Turn-count penalty

```
final_reward = base * (1 - min(0.05 · (n_turns - 1), 0.10))
```

Three turns scores 0.9× the equivalent single-turn reward — same
constants as `math-algebra-multiturn` (D6-A locked).

## Schema

Identical to `sql-single-turn` per turn:

```json
{
  "query": "SELECT category, SUM(amount) FROM sales GROUP BY category ORDER BY category ASC",
  "confidence": 0.85
}
```

The user message on turns 2 and 3 is verifier feedback drawn from
the diagnostic dict `{parse_error, predicted_row_count,
gold_row_count, correctness_match}`.

## Reward decomposition

Same as `sql-single-turn` (D7-A: 0.10 format + 0.20 parse + 0.70
correctness), with the multiplicative turn-count penalty applied
to the final-turn reward.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("sql-multiturn", max_turns=3)
inst = env.generate_instance(seed=42)
out = env.run_rollout(solver, inst)  # solver is your LLMSolver
print(out["meta"]["turn_rewards"], out["meta"]["turn_penalty"])
```

## See also

- [`sql-single-turn`](sql-single-turn.md) — single-turn baseline.
