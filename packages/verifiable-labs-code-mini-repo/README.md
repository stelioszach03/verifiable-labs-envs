# verifiable-labs-code-mini-repo

Synthetic-mini-repo code-execution RL environment from the
Verifiable Labs catalogue. The repo-scale variant: each instance
ships a procedurally generated 3-file Python repo, the model edits
1–3 of the files, and the env runs the full test suite (visible +
hidden modules) under the D5 sandbox.

Three templates (PHASE_24_PLAN.md §8.2):

| Template            | What the model does                                                       |
|---------------------|----------------------------------------------------------------------------|
| `bug_fix`           | Repo has a buggy `add` returning `x - y`; model fixes the operator.        |
| `feature_add`       | Repo has a stub `fizzbuzz`; model implements per the docstring.            |
| `refactor_preserve` | Repo has verbose `square_sum`; model refactors without breaking tests.     |

D6-B locked — no git checkout, no clone server. All 3 templates plus
their hidden test modules synthesise fresh from a 64-bit seed.
`EFFECTIVE_INSTANCES > 5e22`, well above the contamination-resistance
gate.

## Install

```bash
pip install verifiable-labs-code-mini-repo
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
