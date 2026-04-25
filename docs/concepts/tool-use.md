# Tool use

Two v0.1 envs accept structured tool calls instead of free-form JSON
answers. This is a stepping-stone toward agentic evaluation: the
model must plan a sequence of side-effecting tool invocations, not
just emit a final solution.

## Tool-use envs

| env | tool | what the tool does |
|---|---|---|
| `sparse-fourier-recovery-tools` | `compute_fft` | computes `A(x)` for a candidate support, returns the residual |
| `phase-retrieval-tools` | `evaluate_modulus` | computes `|A x|` and returns the squared error against the observed modulus |

Both expose **one** tool — enough to demonstrate the dispatch
mechanism without making the eval surface combinatorial. Future envs
may chain tools (tool A's output feeds tool B's input).

## How dispatch works

The platform forwards the tool to the solver via the OpenAI / Anthropic
function-calling format:

```jsonc
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "compute_fft",
        "description": "Compute A(x) for the proposed support and return the residual.",
        "parameters": {
          "type": "object",
          "properties": {
            "support_idx": {"type": "array", "items": {"type": "integer"}},
            "support_amp_x1000": {"type": "array", "items": {"type": "integer"}}
          },
          "required": ["support_idx", "support_amp_x1000"]
        }
      }
    }
  ]
}
```

When the model emits a `tool_call`, the env executes the tool
locally (no model code runs anywhere except in the solver process),
sends the result back as a `tool` message, and the model continues.

The rollout halts when:

- the model emits a final assistant message without a tool call
  (interpreted as the answer), or
- `max_tool_calls` is reached (default 5; protects against infinite
  loops), or
- the tool itself raises (the env propagates the error message back
  to the model, but only once — repeat failures abort).

## Reward semantics

Tool-use envs score the **final** prediction the same way as their
non-tool siblings. The tool calls themselves are not directly
rewarded — they're a means to a better final answer. The platform
records `meta.n_tool_calls` so you can correlate tool-use frequency
with reward gains.

The headline finding in the v0.1 benchmark:
**Claude Haiku 4.5 used tools constructively** (mean reward +0.05 vs
the no-tools env on sparse-Fourier), while
**GPT-5.4-nano spammed tool calls without convergence** (mean
reward unchanged, n_tool_calls saturating the cap). This is exactly
the kind of behaviour the env is designed to surface: tools amplify
capability *and* surface failure modes.

## When NOT to add a tool to your env

- If the closed-form forward operator is cheap enough that the model
  can mentally simulate it. Tools add latency and tokens; they only
  help if computing `A(x)` is non-trivial.
- If the tool exposes the answer trivially. A tool that returns
  `argmin_x ||A(x) - y||` is a solver, not a tool. The env loses its
  signal.
- If you cannot run the tool deterministically. Non-determinism
  inside a tool breaks the `same seed → same reward` guarantee.

## Implementation pointer

The canonical reference is
`src/verifiable_labs_envs/envs/sparse_fourier_tools.py`. The minimum
surface to implement:

- `class FooToolsAdapter(EnvAdapter)` — implements
  `tool_definitions() -> list[dict]` and
  `dispatch_tool(name, args, instance) -> dict`
- the env's `run_rollout` loops `solver.complete_turns(messages, tools=tool_defs)`
  and appends `tool` messages with the dispatch results
