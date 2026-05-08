# __ENV_ID__

Verifiable Labs tool-calling RL environment for **__DOMAIN__**.

This package was scaffolded from
``templates/tool-calling/`` via
``scripts/create_env.py __ENV_ID__ --template tool-calling --domain "__DOMAIN__"``.

## What this env does

The env hands the solver an OpenAI-style function-calling task: a
natural-language goal plus a subset of the platform's mock tool
primitives (`calculator`, `web_search`, `read_file`, `write_file`,
`send_message`). The model emits tool calls in a budget-capped loop
until it submits a final non-tool JSON envelope. The env scores

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Final non-tool message is parseable JSON.                                 |
| `parse_valid`   | 0.20   | Every tool-call carried valid args AND the final submission parses.       |
| `correctness`   | 0.70   | D2-C blend: 0.30 · action_validity + 0.70 · final_state_match (template). |

The shared primitives + dispatcher live at
``verifiable_labs_envs.tool_primitives``; this env imports them
verbatim — D10-A locked in PHASE_25_PLAN.md.

## Filling in the scaffold

Replace the `NotImplementedError` stubs in:

- ``__ENV_PY__/data.py`` — `generate_problem(seed, **hyperparams)`
  returns a dict with `prompt`, `gold_spec`, `initial_files`,
  `available_tools`, `template_name`. Use
  `numpy.random.default_rng(seed)` for reproducibility; ensure
  `EFFECTIVE_INSTANCES > 1e15` (procedural-regeneration gate).
- ``__ENV_PY__/env.py`` — adjust hyperparams (max_tool_calls, alpha)
  if your env needs tighter or looser bounds.
- ``__ENV_PY__/reward.py`` — implement `_check_gold_state(state,
  instance)` per template; the wider reward kernel is unchanged.

The scoring kernel and adapter need no edits in most cases; the
default JSON envelope (`{"answer": <result>, "confidence": <float>}`)
and the D2-C composite cover the common case.

## Running

```bash
python scripts/validate_env.py environments/__ENV_PY__/   # contract checks
pytest                                                     # unit + reward + tool-primitive tests
```

## Why a separate template family

The inverse-problem template hard-codes forward operators + NMSE; the
symbolic-math template hard-codes SymPy equivalence; the
code-execution template hard-codes a sandboxed pytest runner. None
applies to tool orchestration. The tool-calling family swaps in:

- A `ToolCallingInstance` shape carrying `gold_spec` + `available_tools`
  + `initial_files` for workspace seeding.
- A reward kernel that delegates the heavy lifting to
  `verifiable_labs_envs.tool_primitives.dispatch_tool` + a
  per-template `_check_gold_state` predicate.
- An OpenAI-style rollout loop (D4-C budget cap + submit terminator).
