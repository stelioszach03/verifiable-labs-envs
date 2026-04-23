# The conformal-coverage reward term

Every environment in this package produces a reward of the form

```
reward = w_q * score_quality + w_s * score_structure + w_c * score_conformal
```

The first two terms (PSNR / SSIM for imaging, NMSE / support-F1 for
compressed sensing) measure point-estimate quality. The third — the
**conformal-coverage term** — measures whether the solver's reported
uncertainty is *honest*. This note explains what "honest" means here,
and how the term is computed.

## The reward we want

Given a calibrated predictor that outputs a point estimate `x_hat` and a
per-entry standard-deviation estimate `sigma_hat`, we want to reward the
predictor when the conformal interval

```
[ x_hat - q_alpha * sigma_hat,  x_hat + q_alpha * sigma_hat ]
```

contains the ground truth `x*` with approximately the target probability
`1 - alpha`. Both **over**-coverage (too wide, uninformative) and
**under**-coverage (too narrow, overconfident) are undesirable.

## Calibrating `q_alpha`

We use the split-conformal procedure of Lei et al. (2018). Given a
fixed calibration set of instances `{(x*_i, x_hat_i, sigma_hat_i)}`,
compute the pooled standardized residuals

```
r_i = | x_hat_i - x*_i | / ( sigma_hat_i + eps )
```

and define `q_alpha` as the `ceil((n + 1) * (1 - alpha)) / n`
empirical quantile of `{r_i}`. Under exchangeability, an independent
test point has

```
P( x* in [ x_hat - q_alpha * sigma_hat,  x_hat + q_alpha * sigma_hat ] ) >= 1 - alpha.
```

See [`src/verifiable_labs_envs/conformal.py`](../src/verifiable_labs_envs/conformal.py)
for the implementation — `split_conformal_quantile`, `scaled_residuals`,
`interval`, `coverage`, `coverage_score`.

### A subtle choice: which entries enter the pooled residuals?

For dense signals (super-resolution, CT), every pixel carries information
and all entries are pooled.

For **sparse** signals (`sparse-fourier-recovery`), the majority of
entries are structurally zero, and pooling over all entries lets zero
entries with tiny residuals dominate the quantile — collapsing it to
zero and washing out the reward signal. The sparse-Fourier environment
therefore restricts pooling **to the true support entries** at both
calibration and scoring time. This is equivalent to saying "we reward
calibrated uncertainty on the entries that actually carry signal" and
is called out explicitly in the environment docstring.

## Reward from coverage

At score time an instance is drawn, the solver returns `(x_hat, sigma_hat)`,
and we compute the empirical coverage

```
c = mean( x* in [ x_hat - q_alpha * sigma_hat,  x_hat + q_alpha * sigma_hat ] )
```

on the relevant entry set. The reward component is

```
score_conformal = max( 0, 1 - | c - (1 - alpha) | )
```

which peaks at `1.0` when empirical coverage matches the target exactly
and declines linearly as coverage drifts in either direction.

## Why this shape

A solver that emits a tiny `sigma_hat` everywhere earns a high
`score_quality` if its point estimates are good, but its intervals
contain the ground truth at far below the target rate and
`score_conformal` collapses — a solver that *claims* certainty it does
not have is penalized.

A solver that emits a huge `sigma_hat` everywhere covers every instance
trivially, gets over-coverage `c ~ 1`, and is symmetrically penalized
for being too conservative.

The only way to win both terms is to report `sigma_hat` that tracks the
actual recovery error — i.e., to be honestly uncertain where the
problem is hard and honestly confident where it is easy. That is the
behavior we want to reward when training reasoning models on inverse
problems.

## References

- J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman,
  *Distribution-free predictive inference for regression*, JASA 2018.
- A. N. Angelopoulos, S. Bates, *A Gentle Introduction to Conformal
  Prediction*, 2022.
- RLVR-calibration background: arXiv:2509.21882, 2510.00915.
