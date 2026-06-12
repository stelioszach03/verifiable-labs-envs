# `code-humaneval-multiturn`

**Multi-turn variant of `code-humaneval` with visible-test feedback
between turns.** Same problem distribution, same reward kernel, same
sandbox guarantees. Only the rollout protocol differs.

## Rollout

| Turn | What the model sees | What it returns |
|---|---|---|
| 1 | Function signature + docstring + visible test block | First implementation |
| 2 | Pass/fail counts on visible tests + first failure excerpt (no test source, no oracle) | Revised implementation |
| 3 | Same — final attempt, scored against **visible ∪ hidden** | Final implementation |

Hidden tests are **never** shown to the model (R10 in
PHASE_24_PLAN.md). The only feedback channel between turns is the
visible-pass count plus the first-failure line from pytest stdout —
nothing that lets the model reverse-engineer hidden assertions.

## Turn-count penalty

```
final_reward = base_reward * (1 - min(0.05 * (n_turns - 1), 0.10))
```

Three turns scores 0.9× the equivalent single-turn reward — same
constants as `math-algebra-multiturn` (D8-C ruling).

## Schema

Identical to `code-humaneval` per turn:

```json
{
  "code": "def solve_list_sum_filter(nums, threshold):\n    return sum(n for n in nums if n > threshold)",
  "confidence": 0.6
}
```

The user message on turns 2 and 3 is verifier feedback, not free
prose:

```
FEEDBACK on your previous turn:
You passed 1/2 visible test case(s).
First failure: FAILED test_case_001 - AssertionError: 6 != 7
```

## Reward decomposition

Same as `code-humaneval` (D7-C: 0.10 format + 0.20 parse + 0.70
pass_rate), with the multiplicative turn-count penalty applied to
the final-turn reward.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("code-humaneval-multiturn", max_turns=3)
inst = env.generate_instance(seed=42)
out = env.run_rollout(solver, inst)  # solver is your LLMSolver
print(out["meta"]["turn_rewards"])  # per-turn base rewards
```

## See also

- [`code-humaneval`](code-humaneval.md) — single-turn baseline.
- [`code-humaneval-tools`](code-humaneval-tools.md) — tool-use variant.
- [`code-mini-repo`](code-mini-repo.md) — repo-scale variant.
