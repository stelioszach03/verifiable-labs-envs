# `tool-calling-multiturn`

**Multi-turn variant of `tool-calling-single` with verifier feedback
between turns.** Same problem distribution, same reward kernel, same
mock-primitive library. Only the rollout protocol differs.

## Rollout

After every assistant tool call the user message echoes:

```
FEEDBACK on your previous turn:
Tool `calculator` succeeded; result preview: {"value": 24.0}.
Remaining budget: 27 tool call(s).
```

When a tool returns an error payload the feedback echoes the error
message + budget. When the assistant turn was a non-tool message
that didn't parse, the feedback nudges toward the JSON envelope.

Hidden `gold_spec` is **never** serialised into a feedback message
(R10 carry-over).

## Per-extra-turn penalty

```
final_reward = base * (1 - min(0.05 · (n_assistant_turns - 1), 0.10))
```

The first assistant turn is free; each additional one accrues 5%
reward penalty, capped at 10%. Three rounds (one tool + one feedback
+ one final) yields 1 extra assistant turn → 0.95× single-turn reward.

D8-C math-multiturn parity: same `TURN_PENALTY_PER_EXTRA = 0.05` and
`TURN_PENALTY_CAP = 0.10` constants as `math-algebra-multiturn`.

## Schema

Identical per turn to `tool-calling-single`:

```json
{"answer": 42, "confidence": 0.85}
```

## Reward decomposition

Same as `tool-calling-single` (D6-A: 0.10 format + 0.20 parse + 0.70
correctness; D2-C composite inside `correctness`), with the
multiplicative turn-count penalty applied to the final-turn reward.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("tool-calling-multiturn", max_tool_calls=30)
inst = env.generate_instance(seed=42)
out = env.run_rollout(solver, inst)  # solver supports OpenAI tool-calling
print(out["meta"]["turn_penalty"], out["meta"]["n_assistant_turns"])
```

## See also

- [`tool-calling-single`](tool-calling-single.md) — single-pass baseline.
- [`tool-calling-debug`](tool-calling-debug.md) — trace-debug variant.
