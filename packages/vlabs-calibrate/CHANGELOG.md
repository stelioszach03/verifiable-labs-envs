# Changelog

All notable changes to `vlabs-calibrate` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a1] — 2026-04-30

First public alpha. Promoted the conformal kernel out of
`verifiable-labs-envs` into a standalone, env-agnostic package.

### Added
- `vlabs_calibrate.calibrate(fn, traces, *, alpha=0.1, nonconformity=...)`:
  one-line entry point that returns a `CalibratedRewardFn` wrapper.
- `CalibratedRewardFn` callable: drop-in reward replacement returning
  `CalibrationResult` with reward, conformal interval, σ, quantile, and
  optional coverage flag.
- `CalibratedRewardFn.evaluate(traces, *, tolerance=0.05)`: held-out
  validation returning `CoverageReport`.
- Built-in non-conformity scores: `scaled_residual` (default),
  `abs_residual`, `binary`. Custom callables also accepted.
- `vlabs_calibrate.core`: re-exports of the split-conformal primitives
  (`split_conformal_quantile`, `scaled_residuals`, `interval`,
  `coverage`, `coverage_score`).
- Coverage validation suite over five synthetic distributions (Gaussian,
  heavy-tail t₃, bimodal, sparse, structured) — each hits `(1 − α) ± 5pp`
  on held-out test sets at `n_train=500`, `n_test=2000`.
- Three external-only demos under `examples/calibrate/` (HumanEval-style
  pass/fail, MATH exact-match, GSM8K-style step-validity).

### Notes
- Single core dependency: `numpy>=1.26`. Optional `[scipy]` extra.
- Python 3.10+.
- Tests: run separately via `pytest packages/vlabs-calibrate/tests/`.
  Repo-root `pytest` does not yet pick them up — wiring into root
  `pyproject.toml` + CI is deferred to Phase 15.B.
