# `code-humaneval-tools`

**Tool-use variant of `code-humaneval`.** Same problem distribution,
same reward kernel, but the model converges via three primitives
(D9-A ruling) instead of returning a one-shot completion:

| Tool         | Effect                                                                  |
|--------------|--------------------------------------------------------------------------|
| `read_file`  | Return contents of a file in the per-call workspace                     |
| `write_file` | Overwrite a workspace file (typically `solution.py`)                     |
| `run_test`   | Run a single visible pytest case in the D5 sandbox; returns pass/fail   |

The workspace is seeded with an empty `solution.py` and a
`test_solution.py` containing the **visible tests only** — the hidden
test suite is the held-out grading signal (R10).

`DEFAULT_MAX_TOOL_CALLS = 30`, `DEFAULT_TOOL_TIMEOUT_S = 5.0` per
tool invocation.

## Rollout protocol

```
loop ≤ max_tool_calls:
    LLM emits tool_call (read_file / write_file / run_test)
    sandbox executes (per-call DEFAULT_TOOL_TIMEOUT_S)
    env appends result to context
on submit:
    read solution.py from workspace, score against visible ∪ hidden
```

Submit is signalled by a non-tool turn carrying a JSON envelope:

```json
{"code": "<final source>", "confidence": <float in [0, 1]>}
```

The scorer prefers the workspace's `solution.py` over the parsed
JSON — the JSON envelope is a fallback for models that ignore the
workspace and just return code in their final message.

## Reward decomposition

Same as `code-humaneval` (D7-C). Tool calls do not score directly;
they are the *means* by which the model converges to a solution
that the single-turn verifier evaluates.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("code-humaneval-tools", max_tool_calls=30)
inst = env.generate_instance(seed=42)
out = env.run_rollout(solver, inst)  # solver supports OpenAI tool-calling
print(out["meta"]["n_tool_calls"], out["meta"]["workspace_used"])
```

## See also

- [`code-humaneval`](code-humaneval.md) — single-turn baseline.
- [`code-humaneval-multiturn`](code-humaneval-multiturn.md) — multi-turn variant.
- [`code-mini-repo`](code-mini-repo.md) — repo-scale variant.
