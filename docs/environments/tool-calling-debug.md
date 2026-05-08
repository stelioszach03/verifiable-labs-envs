# `tool-calling-debug`

**Trace-debug tool-calling: prefix-conditioned trajectory completion
(D8-C).** PHASE_25_PLAN.md §9.3. Each instance ships a partial
trajectory + the `WorkspaceState` snapshot it produced, plus a goal
predicate. The model continues the rollout from that point with a
tightened budget.

This exercises a unique skill — continuation conditioning on
external state — that the single + multi-turn variants don't cover,
while reusing 100% of the existing rollout machinery.

## Templates

Three templates wrap base single-turn templates:

| Debug template       | Base                  | Prefix supplied                                     |
|----------------------|-----------------------|------------------------------------------------------|
| `partial_compute`    | `arithmetic_compute`  | First `(a+b)` step pre-computed; model finishes.    |
| `partial_search`     | `search_and_email`    | `web_search` call done; model sends the email.      |
| `partial_workspace`  | `file_concat`         | Both files read into state; model writes the merge. |

`EFFECTIVE_INSTANCES > 5 × 10²²`, well above the 1 × 10¹⁵
contamination-resistance gate.

## Instance shape

```python
@dataclass(frozen=True)
class DebugInstance:
    prompt: str
    template_name: str          # e.g. "partial_compute"
    base_template: str          # e.g. "arithmetic_compute"
    seed: int
    gold_spec: dict             # oracle, excluded from as_inputs()
    initial_files: dict
    available_tools: tuple
    prefix_messages: tuple      # canned assistant + tool messages
    prefix_state_payload: dict  # serialised WorkspaceState
    max_remaining_calls: int
```

The env's `run_rollout` seeds the conversation with:

1. The system prompt.
2. The user prompt (problem statement + workspace listing + tool list).
3. Every entry of `prefix_messages` — assistant tool-call + tool-result
   pairs (truncated to 4 KB per result).
4. A trailing user nudge: *"The trace above shows the partial
   trajectory so far. Continue from this point."*

The solver's first turn picks up from there. The rollout's initial
:class:`WorkspaceState` is a fresh copy of `prefix_state_payload`,
not a re-execution — replay is deterministic, but the env doesn't
re-spawn the prefix tools (the pre-computed dispatch happens at
instance generation time).

## Reward decomposition

Same as `tool-calling-single` (D6-A: 0.10 format + 0.20 parse +
0.70 correctness; D2-C composite inside `correctness`). The
predicate dispatch swaps `template_name` for `base_template` so
the existing per-template logic in
`tool_calling_single._check_gold_state` applies transparently.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("tool-calling-debug")
inst = env.generate_instance(seed=42)
print(inst.template_name, inst.base_template)
print(inst.prefix_state.calculator_history)  # for partial_compute
```

## See also

- [`tool-calling-single`](tool-calling-single.md) — single-pass baseline.
- [`tool-calling-multiturn`](tool-calling-multiturn.md) — multi-turn variant.
