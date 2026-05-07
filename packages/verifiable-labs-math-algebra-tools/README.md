# verifiable-labs-math-algebra-tools

Tool-use algebraic-simplification RL environment with SymPy primitives, conformal coverage, and SymPy-verified rewards.

The LLM solves the standard `math-algebra` problem but must compose
its answer using primitive SymPy tools rather than reasoning end-to-end.

## Available primitives

All four are timeout-bounded (5 s default) so adversarial inputs can
never wedge a rollout:

| Primitive                                     | Action |
|-----------------------------------------------|--------|
| `sympy_simplify(expr_str)`                    | `sympy.simplify` on a parseable expression |
| `sympy_expand(expr_str)`                      | `sympy.expand` |
| `sympy_solve(equation_str, var_str)`          | solve `equation_str = 0` for `var_str` |
| `sympy_substitute(expr_str, var_str, value_str)` | substitute `var = value` |

No oracle. Tool calls do not score directly — they are the *means* by
which the model converges to the final answer. Reward is computed only
on the final parsed `answer`, identical to
[`verifiable-labs-math-algebra`](../verifiable-labs-math-algebra/).

## Install

```
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-math-algebra-tools"
```

## Use

```python
from verifiable_labs_math_algebra_tools import load_environment

env = load_environment(calibration_quantile=0.5, max_tool_calls=20)
inst = env.generate_instance(seed=0)

# At inference time, env.run_rollout() drives tool-call-then-answer
# protocol against an LLMSolver that supports OpenAI-style function calling.
```

## Why a separate tool-use variant

The single-turn variant scores the model's ability to answer
end-to-end. The multi-turn variant adds verifier feedback. The tool-use
variant adds an explicit decomposition affordance: the model can
delegate primitive operations to SymPy itself, focusing its reasoning
on which tool to call when. This isolates the model's *planning*
capability from its raw symbolic-manipulation capability.
