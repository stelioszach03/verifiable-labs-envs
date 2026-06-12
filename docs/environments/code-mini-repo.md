# `code-mini-repo`

**Repo-scale code-execution: synthetic 3-file Python repos, multi-file
edits.** PHASE_24_PLAN.md D6-B locked — no git checkout, no clone
server. Every "repo" is procedurally generated from a 64-bit seed.

The model receives a small repo (3–5 files including a `tests/`
directory) and a task spec. It overwrites 1–3 of the files (constrained
to `editable_paths`), and the env runs the **visible + hidden** test
suite under the same D5-bounded sandbox as `code-humaneval`. Reward
weights are identical.

## Templates

| # | Template            | What the model does                                                  |
|---|---------------------|-----------------------------------------------------------------------|
| 1 | `bug_fix`           | `add(x, y)` returns `x - y` (a bug); model fixes the operator        |
| 2 | `feature_add`       | `fizzbuzz(n)` is a stub; model implements per the docstring          |
| 3 | `refactor_preserve` | Verbose `square_sum`; model refactors without breaking tests         |

Procedural-regeneration certification: 3 templates × 64-bit seed ×
~10⁶ parameter combinations ≈ 5.5 × 10²² effective instances, well
above the 1 × 10¹⁵ contamination-resistance gate.

## Schema

```json
{
  "files": {
    "calc.py": "def add(x, y):\n    return x + y\n"
  },
  "confidence": 0.9
}
```

Each entry of `files` is the **complete new content** of a file the
model wants to overwrite. Predictions touching paths outside
`editable_paths` are silently dropped — `parse_valid` only counts the
edits that hit valid targets.

## Reward decomposition

```
reward = 0.10 * format_valid    (output is JSON with non-empty `files` mapping)
       + 0.20 * parse_valid     (every edited path ∈ editable_paths AND
                                  every edited file compiles)
       + 0.70 * pass_rate       (pytest passes / total over visible ∪ hidden)
```

The repo merge order at score time:
1. Base files from the instance.
2. Editable overrides from the prediction.
3. Hidden test modules merged on top.

A `conftest.py` is auto-injected at the sandbox root so `from
math_util import …` style imports work inside `tests/`.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("code-mini-repo")
inst = env.generate_instance(seed=42)
print(inst.editable_paths)   # ('calc.py',) for bug_fix
print(inst.spec)             # natural-language task description
```

## See also

- [`code-humaneval`](code-humaneval.md) — single-file baseline.
- [`code-humaneval-multiturn`](code-humaneval-multiturn.md) — multi-turn variant.
- [`code-humaneval-tools`](code-humaneval-tools.md) — tool-use variant.
