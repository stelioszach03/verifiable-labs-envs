# __ENV_ID__

Verifiable Labs scientific RL environment for **__DOMAIN__**.

This package was scaffolded from
``templates/symbolic-math/`` via
``scripts/create_env.py __ENV_ID__ --template symbolic-math --domain "__DOMAIN__"``.

## What this env does

The env hands the solver a **symbolic problem prompt** (e.g. an
algebraic expression to simplify or an equation to solve), expects a
**SymPy-parseable answer**, and scores the answer against a hidden
canonical-form **gold expression** using `sympy.simplify(answer −
gold) == 0`. The scoring kernel is hard-timeout-wrapped so adversarial
inputs cannot wedge the env.

The reward is a 3-component sum in `[0, 1]`:

| Component       | Weight | What it rewards                                          |
|-----------------|--------|----------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON                                 |
| `parse_valid`   | 0.20   | Extracted answer is a valid SymPy expression             |
| `correct`       | 0.70   | `simplify(answer − gold) == 0` (with timeout)            |

A **conformal coverage** term layers on top: residuals
`r = 1 − reward` over a held-out calibration set are aggregated into
the conformal quantile `q̂_α`, and the env emits a per-instance
`covered` flag that aggregates to empirical coverage at the target
`1 − α`.

## Filling in the scaffold

Replace the `NotImplementedError` stubs in:

- `__ENV_PY__/data.py` — `generate_problem(seed, **hyperparams)` returns
  a `(prompt, gold_expr)` tuple. Use `numpy.random.default_rng(seed)`
  for reproducibility; ensure the seed × pool product gives
  `EFFECTIVE_INSTANCES > 1e15` for the procedural-regeneration check.
- `__ENV_PY__/env.py` — adjust hyperparams (timeout, alpha, weights)
  if your env needs tighter or looser bounds.

The scoring kernel and adapter need no edits in most cases; the
default JSON schema (`{"answer": "...", "confidence": <float>}`) and
`simplify`-based equivalence cover the common case.

## Running

```bash
python scripts/validate_env.py environments/__ENV_PY__/   # contract checks
pytest                                                     # unit + reward tests
```

## Why a separate template family

The inverse-problem template (`templates/inverse-problem/`) hard-codes
forward operators, NMSE scoring, and per-entry σ̂ vectors — none of
which apply to symbolic algebra. This template family swaps in
SymPy-string instances, a 3-component partial-credit reward, and a
threaded `simplify` timeout so adversarial inputs cannot wedge CI.
The conformal calibration loop is unchanged from the inverse-problem
side; both families reuse `verifiable_labs_envs.conformal`.
