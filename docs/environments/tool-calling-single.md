# `tool-calling-single`

**Single-pass procedural tool-calling with shared mock primitives.**
Given a natural-language task plus a subset of five mock tools
(`calculator`, `web_search`, `read_file`, `write_file`,
`send_message`), the model emits OpenAI-style function calls in a
budget-capped loop and submits a final non-tool JSON envelope.

This is the first member of the **tool-calling** template family
introduced in Phase 25.

## Problem

| | |
|---|---|
| input | natural-language task + workspace listing + available-tool list |
| output | trajectory of tool calls + final JSON `{"answer": <result>, "confidence": <float>}` |
| gold | per-template predicate over the final `WorkspaceState` |
| budget | 30 tool calls + 1 final non-tool message (D4-C) |

Ten procedural templates spanning arithmetic, search, file I/O, and
messaging:

| # | Template | Sample task |
|---|---|---|
| 1 | `arithmetic_compute` | chain calculator calls to evaluate `(a + b) * c` |
| 2 | `search_and_email` | search corpus, email the summary to fixed recipient |
| 3 | `file_concat` | read 2 files, write merged version |
| 4 | `compute_then_send` | compute, then email the digit string |
| 5 | `multi_search` | search N topics, write results to one file |
| 6 | `read_search_write` | read note, search topic, write enriched note |
| 7 | `outbox_audit` | search topic, send 2 emails about it |
| 8 | `nested_calculator` | compute `(a + b) * (c - d)` step by step |
| 9 | `search_dedup` | search twice, dedupe titles into a file |
| 10 | `compute_chain` | sequential arithmetic with tool-result feedback |

`EFFECTIVE_INSTANCES > 6 × 10²³`, well above the 1 × 10¹⁵
contamination-resistance gate.

## Variants

- [`tool-calling-single`](#) — single-pass, this page.
- [`tool-calling-multiturn`](tool-calling-multiturn.md) — same
  templates with verifier feedback between turns + 5%-per-extra-turn
  penalty (cap 10%).
- [`tool-calling-debug`](tool-calling-debug.md) — trace-debug:
  prefix-conditioned trajectory completion (D8-C).

## Schema

```json
{
  "answer": 42,
  "confidence": 0.85
}
```

The scorer reads the workspace state directly (files written, outbox
messages, calculator history) — the JSON envelope is advisory and
supplies the confidence signal.

## Reward decomposition

```
reward = 0.10 · format_valid    (final non-tool message parses as JSON)
       + 0.20 · parse_valid     (every tool-call had dict args
                                  AND the final submission parses)
       + 0.70 · correctness     (D2-C composite — see below)
```

D2-C verification (PHASE_25_PLAN.md §7):

```
correctness = 0.30 · action_validity   (fraction of tool calls
                                          returning a non-error payload)
            + 0.70 · final_state_match (per-template predicate over
                                          the WorkspaceState)
```

Action-spam attacks drop the `action_validity` term; final-state-only
trajectories never reach 1.0 because some intermediate validity is
required.

## Tool primitives

The shared library at `verifiable_labs_envs.tool_primitives` ships
five OpenAI-style schemas + a single dispatcher. Same surface across
all three tool-calling envs (D10-A locked).

```python
from verifiable_labs_envs.tool_primitives import (
    TOOL_SCHEMAS, dispatch_tool, init_state, WorkspaceState,
)

state = init_state(seed=42)
result = dispatch_tool("calculator", {"expression": "3 * (4 + 5)"}, state)
```

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("tool-calling-single")
inst = env.generate_instance(seed=42)
print(inst.prompt)
```

## Tests

The repo's `tests/test_tool_primitives.py` exercises the primitive
library; `tests/test_tool_calling_single.py` covers the env's
templates + reward kernel.
