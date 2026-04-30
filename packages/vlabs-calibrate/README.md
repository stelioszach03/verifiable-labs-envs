# vlabs-calibrate

> Conformal coverage guarantees for any reward function. Five lines of Python.

`vlabs-calibrate` wraps any Python reward callable with a split-conformal
prediction interval providing marginal **(1 − α) coverage** under
exchangeability. Drop-in replacement for your reward function — get
calibrated intervals plus a verified-coverage flag instead of a bare scalar.

The math is the split-conformal procedure of [Lei et al. (2018)](https://arxiv.org/abs/1604.04173).
The pitch: every RL training run today ships uncalibrated rewards.
`vlabs-calibrate` is the first piece of infrastructure to fix that.

> **0.1.0a1 — alpha.** Public surface is stable for the documented use cases
> (continuous and binary reward functions). API may evolve in 0.2.0 once we
> add per-feature Mondrian conformal and exchangeability diagnostics.

## Install

```bash
pip install vlabs-calibrate
```

Python `>=3.10`, single core dependency: `numpy`.

For development inside this monorepo:

```bash
pip install -e packages/vlabs-calibrate
```

## Quickstart

```python
import numpy as np
import vlabs_calibrate as vc

# Your reward function — could be anything; signature is open.
def my_reward(*, prompt: str, completion: str, ground_truth: str) -> float:
    return float(completion.strip() == ground_truth.strip())

# Synthesise a calibration set: noisy reference labels + per-trace sigma.
rng = np.random.default_rng(0)
traces = []
for i in range(200):
    completion = "4" if rng.random() < 0.8 else "5"
    sigma = 0.2
    reward = my_reward(prompt="2+2?", completion=completion, ground_truth="4")
    reference = float(np.clip(reward + sigma * rng.standard_normal(), 0.0, 1.0))
    traces.append({
        "prompt": "2+2?",
        "completion": completion,
        "ground_truth": "4",
        "reference_reward": reference,
        "uncertainty": sigma,
    })

# Calibrate — one line.
calibrated = vc.calibrate(my_reward, traces, alpha=0.1)

# Use anywhere — drop-in replacement for `my_reward`.
result = calibrated(prompt="2+2?", completion="4", ground_truth="4", sigma=0.2)
print(result.reward, result.interval, result.target_coverage)
# → 1.0  (lo, hi)  0.9
```

## Public surface

| name | kind | purpose |
|---|---|---|
| `calibrate(fn, traces, *, alpha=0.1, ...)` | function | builds a calibrated wrapper |
| `CalibratedRewardFn` | dataclass / callable | `__call__` returns `CalibrationResult`; has `.evaluate()` |
| `CalibrationResult` | frozen dataclass | `.reward`, `.interval`, `.sigma`, `.quantile`, `.alpha`, `.covered` |
| `CoverageReport` | frozen dataclass | aggregate diagnostics from `evaluate()` |
| `Trace` | TypedDict | shape spec for calibration entries |
| `vc.core` | submodule | low-level conformal primitives |
| `vc.nonconformity` | submodule | built-in non-conformity scores + registry |
| `vc.__version__` | str | package version |

## Built-in non-conformity scores

| name | formula | when to use |
|---|---|---|
| `scaled_residual` (default) | `\|reward − reference\| / max(σ, eps)` | continuous reward + per-sample σ |
| `abs_residual` | `\|reward − reference\|` | continuous reward, no σ |
| `binary` | `0.0 if reward == reference else 1.0` | 0/1 reward; see caveat below |

> **Binary reward caveat.** For 0/1 rewards the standard split-conformal
> guarantee is degenerate: the (1 − α) quantile is either 0 or 1, producing
> a trivial covered or `[0, 1]` interval. For binary tasks consider
> Mondrian / class-conditional conformal (Vovk & Gammerman) — planned for
> 0.2.0.

## Coverage validation

`tests/test_coverage_validation.py` calibrates on `n_train = 500` traces
(`α = 0.1`) and evaluates on a held-out `n_test = 2000` set across five
synthetic distributions chosen to stress the wrapper. Empirical coverages
on a fixed seed (reproducible by running the test):

| distribution | n_train | n_test | target | empirical | quantile | width (median) | tolerance | pass |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| `gaussian` | 500 | 2000 | 0.90 | 0.9150 | 1.6717 | 1.6717 | ±0.05 | yes |
| `heavy_tail_t3` | 500 | 2000 | 0.90 | 0.9035 | 2.2679 | 2.2679 | ±0.05 | yes |
| `bimodal` | 500 | 2000 | 0.90 | 0.8960 | 2.3959 | 2.3959 | ±0.05 | yes |
| `sparse` | 500 | 2000 | 0.90 | 0.9015 | 4.5035 | 4.5035 | ±0.05 | yes |
| `structured_misspecified` | 500 | 2000 | 0.90 | 0.9120 | 2.6346 | 2.6346 | ±0.07 | yes |

The structured-misspecified case uses a 7pp tolerance because the true σ
is 2× the reported σ on half the traces — split-conformal still gives
marginal coverage under exchangeability, but finite-sample variance is
slightly larger than the well-specified cases.

## Demos

Three self-contained example scripts under [`examples/calibrate/`](./examples/calibrate/):

1. [`01_humaneval_passfail.py`](./examples/calibrate/01_humaneval_passfail.py)
   — binary 0/1 reward (HumanEval-style cheap proxy vs gold tests).
   Shows both regimes: the proxy is "verified" against gold (interval
   collapses to a point) and the degenerate `[0, 1]` regime when the
   proxy disagrees too often.
2. [`02_math_exact_match.py`](./examples/calibrate/02_math_exact_match.py)
   — judge-graded exact match with per-trace confidence as σ.
3. [`03_gsm8k_step_validity.py`](./examples/calibrate/03_gsm8k_step_validity.py)
   — continuous step-validity reward in `[0, 1]` with ensemble
   disagreement as σ. Best illustration of how interval width tracks σ
   smoothly across a test set.

Each demo synthesises its data inline (no external dataset downloads) so
you can copy a snippet straight into your own pipeline.

## Tests

```bash
pip install -e "packages/vlabs-calibrate[dev]"
pytest packages/vlabs-calibrate/tests/
```

The new package's tests are not yet wired into the repo-root `pytest` run;
that is intentional for 0.1.0a1 (Phase 15.B will add the path to root
`pyproject.toml` and CI in a separate, scoped change).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
