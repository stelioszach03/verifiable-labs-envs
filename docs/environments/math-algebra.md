# `math-algebra`

**Single-turn algebraic-simplification with SymPy-verified rewards.**
Given a natural-language problem (expand a product, factor a quadratic,
collect like terms, simplify a polynomial), the solver returns a
SymPy-parseable answer + self-reported confidence. Rewards are
computed against a hidden gold expression via SymPy's `simplify`.

This is the first non-inverse-problem env in the platform and the
first member of the **symbolic-math** template family introduced in
Phase 21.

## Problem

| | |
|---|---|
| input | natural-language algebra problem (e.g. "Expand `(3*x + 2) * (x − 5)`") |
| output | JSON `{"answer": "<sympy-parseable>", "confidence": <float in [0, 1]>}` |
| gold | hidden canonical form, compared via `simplify(answer − gold) == 0` |
| coefficient range | `[-10, 10]` (configurable via `coef_range` hyperparam) |

Five problem templates are sampled with equal weight per seed:

1. **Expand binomial product** — `(a*x + b) * (c*x + d)` → expanded polynomial.
2. **Expand square** — `(a*x + b)**2` → expanded polynomial.
3. **Difference of squares** — `(x + a) * (x − a)` → `x**2 − a**2`.
4. **Factor quadratic** — `x**2 + (r1+r2)*x + r1*r2` → `(x − r1) * (x − r2)`.
5. **Collect like terms** — random sum of `x²`, `x`, constant terms → simplified polynomial.

Procedural-regeneration certification: 64-bit seed × 5 templates ×
~32 K coefficient combinations gives `EFFECTIVE_INSTANCES > 3 × 10²³`,
well above the platform's `1e15` contamination-resistance threshold.

## Variants

- [`math-algebra`](#) — single-turn, this page.
- `math-algebra-multiturn` — 3-turn dialogue with verifier feedback
  (format → parse → equivalence diagnosis, gold expression never
  revealed).
- `math-algebra-tools` — primitive-composition tool-use with
  `sympy_simplify`, `sympy_expand`, `sympy_solve`, `sympy_substitute`.

## Schema

```json
{
  "answer": "x**2 - 1",
  "confidence": 0.85
}
```

The answer is a SymPy-parseable string — any equivalent form is
accepted. `(x-1)*(x+1)` and `x**2 − 1` both score `correct = 1.0` for
a difference-of-squares gold.

## Reward decomposition

```
reward = 0.10 * format_valid       (output is parseable JSON)
       + 0.20 * parse_valid         (extracted answer is valid SymPy)
       + 0.70 * correct             (simplify(answer − gold) == 0)
```

- The `correct` term runs SymPy's `simplify` inside a 5-second hard
  timeout via a daemon thread — adversarial inputs like
  `simplify((x**100 + sin(x))**5)` cannot wedge an episode.
- A conformal coverage layer aggregates residuals `r = 1 − reward` over
  a held-out calibration set into the `(1 − α)` quantile `q̂`. Each
  scored response carries a `covered` flag in `meta` indicating whether
  the residual passes the calibrated threshold.

## Why this env exists

Most of the platform's envs test scientific reasoning on continuous
inverse problems. `math-algebra` extends the methodology to **discrete
symbolic reasoning** while keeping the same procedural-regeneration +
calibrated-reward guarantees. It is also a deliberately easier "warm
start" for new models — the per-instance answer is a single SymPy
string, not a high-dimensional reconstruction — which makes the
scaling curve interpretable when sweeping model sizes.

## Source

[`src/verifiable_labs_envs/envs/math_algebra.py`](https://github.com/verifiablelabs/verifiable-labs-envs/blob/main/src/verifiable_labs_envs/envs/math_algebra.py).

## See also

- [Concepts → Conformal rewards](../concepts/conformal-rewards.md)
- [Concepts → Procedural regeneration](../concepts/procedural-regeneration.md)
- [Concepts → Multi-turn dialogue](../concepts/multi-turn.md) — applies
  to `math-algebra-multiturn`.
- [Concepts → Tool use](../concepts/tool-use.md) — applies to
  `math-algebra-tools`.
