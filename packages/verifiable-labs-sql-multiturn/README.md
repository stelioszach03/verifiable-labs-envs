# verifiable-labs-sql-multiturn

Multi-turn text-to-SQL RL environment from the Verifiable Labs
catalogue. Same problem distribution as `sql-single-turn`, with two
changes:

1. **Verifier feedback between turns.** After each query the user
   message echoes parse status + row-count diagnostics. The gold
   result-set is **never** serialised into a feedback message
   (R10 carry-over).
2. **Per-extra-turn penalty.** The first turn is free; each
   additional turn accrues 5% reward penalty, capped at 10%.

```
final_reward = base * (1 - min(0.05 · (n_turns - 1), 0.10))
```

3-turn cap matches the math-multiturn / code-humaneval-multiturn /
tool-calling-multiturn pattern across the platform (D6-A locked).

## Install

```bash
pip install verifiable-labs-sql-multiturn
```

Source of truth + full docs:
https://github.com/stelioszach03/verifiable-labs-envs
