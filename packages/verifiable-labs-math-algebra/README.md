# verifiable-labs-math-algebra

Single-turn algebraic-simplification RL environment with SymPy-verified rewards and conformal coverage.

This is a thin wrapper over the monorepo `verifiable-labs-envs`.
Installing it gives you:

- A verifiers-compatible entry point: ``verifiers.environments → math-algebra``.
- A direct factory: ``from verifiable_labs_math_algebra import load_environment``.

## Install

From GitHub (subdirectory):
```
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-math-algebra"
```

Once published to the Prime Intellect Environments Hub:
```
prime env install verifiable-labs/math-algebra
```

## Use

```python
from verifiable_labs_math_algebra import load_environment

env = load_environment(calibration_quantile=0.5)
inst = env.generate_instance(seed=0)
print(inst.prompt)
# e.g. "Expand the product (3*x + 2) * (-1*x + 5) and write the result..."

# At inference time the solver returns a Prediction; here's the env's
# trivial baseline for shape demonstration:
out = env.run_baseline(seed=0)
print(out["reward"], out["components"], out["meta"]["covered"])
```

## Reward shape

A 3-component sum in `[0, 1]`:

| Component       | Weight | Rewards                                                  |
|-----------------|--------|----------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON                                 |
| `parse_valid`   | 0.20   | Extracted answer is a valid SymPy expression             |
| `correct`       | 0.70   | `simplify(answer − gold) == 0` (with hard timeout)       |

Plus a per-instance `covered` flag: `(1 − reward) ≤ q̂_α` against the
calibrated conformal threshold.

## Why a separate package

Phase 21 of the verifiable-labs roadmap introduces the symbolic-math
env family alongside the existing 10 inverse-problem envs. Math envs
do not depend on `vlabs-calibrate` — they import the conformal kernel
(`verifiable_labs_envs.conformal`) directly, same as the 10
inverse-problem envs. The only new transitive is SymPy.
