# Conformal rewards

The Verifiable Labs reward function bundles three terms — a
**point-estimate** term, a **support-recovery** term (where applicable),
and a **conformal-coverage** term that scores the model's uncertainty
honesty.

The conformal term is what makes the rewards *verifiable* in the
strong sense: a model that claims certainty it doesn't have is
penalised by the reward signal directly, not just by a downstream
reviewer.

## The score, in one paragraph

For each (env, seed) we know the ground-truth `x`. The model produces
a prediction `x_hat` and a per-coordinate uncertainty `sigma_hat`. We
ask: does the **calibrated** interval `[x_hat - q · sigma_hat, x_hat + q · sigma_hat]`
contain `x`? The quantile `q` is fitted offline so that, on a
held-back calibration pool, the empirical coverage equals `1 - α`
(default `α = 0.10`, so 90 %). At test time the conformal-coverage
component of the reward is `1.0` if the truth falls inside, `0.0`
otherwise — and we *measure* the empirical coverage rate across many
test seeds. Within ±5 pp of `1 - α` the model is well-calibrated;
outside ±10 pp the alpha gate flags it.

## Why bundle uncertainty into the reward?

Standard ML benchmarks score "accuracy" alone — a model that's right
70 % of the time and confident on every answer scores the same as a
model that's right 70 % of the time and *says* it's only 70 %
confident on the wrong ones. The first model is dangerous in
production; the second is useful. Verifiable Labs rewards the second.

This matters most for:

- **scientific decision support** — wrong answer with high stated
  confidence triggers downstream errors
- **agentic systems** — a planner that knows when it doesn't know
  routes around the gap; one that doesn't propagates the error
- **regulatory readiness** — emerging frameworks (EU AI Act,
  NIST AI RMF) ask for *measured* uncertainty, not declared

## Calibration mechanics

Per-env quantile fitting is done by `verifiable_labs_envs.conformal.calibrate_quantile`:

1. Generate `n_calibration` instances on seeds far past the test pool
   (the platform's convention is seeds `0…499` for calibration,
   `60_000+` for test).
2. Run a solver — often the env's own classical baseline — on every
   one. Record per-instance residuals `|x - x_hat| / sigma_hat`.
3. Take the empirical `(1 - α)`-quantile of those residuals across
   the calibration pool. This is the value the env caches as
   `conformal_quantile`.
4. At test time the same `q` is applied: the interval is
   `[x_hat - q · sigma_hat, x_hat + q · sigma_hat]`.

Because the calibration pool is **regenerated** from the seed pool
(not stored), the published `q` does not leak any specific instance
content — only the aggregate distribution shape.

## Empirical coverage as a regression test

Every commit to the envs runs a coverage smoke test (50 seeds, 5
boots) and a 200-seed nightly. If empirical coverage drifts more than
±5 pp from target, the test fails. This guards against:

- a refactor of `forward_op` that changes the scale of `sigma_hat`
- a change to the calibration sample distribution
- a numerical bug that biases residuals

See `tests/test_calibration.py` for the assertions.

## What this is NOT

- A guarantee for *out-of-distribution* inputs. Conformal coverage
  holds in expectation over the calibration distribution; if the
  test seed shifts that distribution (different `n`, `k`, or `sigma`
  ranges), the coverage guarantee weakens.
- A substitute for downstream validation. The platform tells you
  whether a model's stated uncertainty matches the ground-truth
  spread on this benchmark. Whether that translates to your
  application is your validation problem.

## Further reading

- Repository [`docs/conformal.md`](../conformal.md) — the canonical
  internal reference, links to Vovk & Shafer's monograph and the
  split-conformal procedure.
- Repository [`docs/METHODOLOGY.md`](../METHODOLOGY.md) — full
  methodology paper drafted for OpenReview.
