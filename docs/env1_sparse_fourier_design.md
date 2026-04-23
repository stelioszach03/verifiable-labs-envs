# Env 1 design — `sparse-fourier-recovery`

> Day-1 architecture sketch. Implementation target: Day 2 (2026-04-24).

## Problem

Recover a sparse signal `x* ∈ R^N` from under-sampled noisy Fourier measurements.

```
y = S · F(x*) + ε,    ε ~ N(0, σ² I)
```

- `F` — 1D discrete Fourier transform (orthonormal).
- `S` — binary subsampling mask selecting `M` of `N` frequencies, `M < N`.
- `x*` — `K`-sparse: exactly `K` nonzero entries, locations uniform random, amplitudes `~ N(0, 1)`.
- Default hyperparameters: `N = 256`, `M = 64` (4× undersampling), `K = 10`, `σ = 0.05`.

This is the canonical compressed-sensing test problem and is closest in spirit to VLBI / aperture-synthesis imaging — the bridge from the draft EHT paper to commercial RL environments.

## Instance generation

```python
@dataclass
class Instance:
    y: np.ndarray            # shape (M,), complex
    mask: np.ndarray         # shape (N,), bool — True at observed frequencies
    sigma: float             # noise std
    n: int                   # signal length
    k: int                   # true sparsity
    # Ground truth (hidden from solver; used only by scorer):
    x_true: np.ndarray       # shape (N,), real
    support_true: np.ndarray # shape (K,), int indices
    seed: int
```

`env.generate_instance(seed, n=256, m=64, k=10, sigma=0.05) -> Instance`

Every call regenerates `x*` and `ε` from the seed — no fixed dataset, so models can't memorize.

## Forward operator (JAX)

```python
def forward(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """A(x) = S · F(x).  JIT-compiled, batched."""
    return jnp.fft.fft(x, norm="ortho")[mask]
```

## Solver contract

The solver (a model under test, or a baseline) receives `y, mask, sigma, n` and must return:

```python
@dataclass
class Prediction:
    x_hat: np.ndarray        # shape (N,), real — point estimate
    sigma_hat: np.ndarray    # shape (N,), positive — per-entry std estimate
    # Optional:
    support_hat: np.ndarray | None  # shape (?,), int indices
```

`sigma_hat` is the key contribution — it forces the solver to commit to an uncertainty estimate, which is what we score with conformal coverage.

## Reward — weighted rubric

```python
reward = w_nmse * score_nmse + w_support * score_support + w_conformal * score_conformal
```

With default weights `(w_nmse, w_support, w_conformal) = (0.4, 0.3, 0.3)`.

### NMSE term

```
score_nmse = exp(- NMSE / τ_nmse),   NMSE = ||x_hat - x*||² / ||x*||²
```

Default `τ_nmse = 0.5`. Bounded in `(0, 1]`, monotone decreasing in error.

### Support-F1 term

If `support_hat` is provided, F1 of estimated support vs `support_true`. If omitted: threshold `|x_hat|` at the `K`-th largest magnitude and compute F1 against the ground truth. Bounded `[0, 1]`.

### Conformal-coverage term (the differentiator)

Pre-computed **calibration set**: fresh instances solved by the reference baseline, producing residuals ``r_i = |x_hat_i - x*_i| / (sigma_hat_i + ε)``. The pooled residuals are **restricted to the true support entries only**, because the 246-of-256 zero entries in a typical instance have near-zero residuals regardless of solver quality and would collapse the pooled quantile to 0. The ``(1 − α)``-quantile ``q_α`` is cached with the environment (α = 0.1 by default → 90% target coverage). Fast calibration (``fast=True``, default) uses 30 samples; full calibration (``fast=False``) uses 500.

At score time:

1. Build interval ``[x_hat - q_α · sigma_hat, x_hat + q_α · sigma_hat]``.
2. Empirical coverage ``c = mean(x* inside interval)`` **on the true support entries only** (matching the calibration restriction).
3. Target coverage ``c_target = 1 − α = 0.9``.
4. ``score_conformal = 1 − |c − c_target|`` (bounded ``[0, 1]``, peaks when coverage matches target, penalizes both over- and under-confidence).

**Why this is the important term:** a model that outputs ``sigma_hat = 1e-6`` everywhere gets a good ``score_nmse`` if its point estimate is good, but ``score_conformal ≈ 0`` because its intervals are empty. A model that outputs ``sigma_hat = 1e6`` everywhere gets full coverage but no information — same failure mode. The only way to win is honest uncertainty. This reward shape is the direct translation of the split-conformal method in the draft EHT paper.

## Baseline — OMP with LS-covariance σ̂ (as shipped)

The shipped reference baseline is **orthogonal matching pursuit** (OMP). It is kept under the public name `ista_baseline` for API stability — callers should read this as "the reference compressed-sensing baseline". OMP is preferred over ISTA/FISTA here because known-``k``-sparse recovery on 4× undersampling does not suffer the LASSO shrinkage bias that floors ISTA's NMSE around 0.6 regardless of iteration count.

Each OMP step: (a) compute correlation ``|A^T r|`` of the current residual with every column; (b) add the index with largest correlation to the active set; (c) refit the active-set amplitudes by real-valued least squares against the complex measurement (stacking real and imaginary parts of both the forward operator and ``y``).

**Per-entry σ̂** is the closed-form least-squares standard error on the selected support:

```
sigma_hat_S = (sigma / sqrt(2)) * sqrt(diag( (A_S^T A_S)^-1 ))
```

Outside the selected support, ``sigma_hat`` is set to the **signal-amplitude prior scale** (1.0, matching the ``N(0, 1)`` nonzero draws in ``generate_instance``). This is the honest "I have no information here" uncertainty — essential so that OMP's occasional support-selection errors do not force conformal to compensate with an enormous quantile.

The retained parameters ``lam``, ``n_bootstrap``, and ``seed`` on ``ista_baseline`` are accepted for API stability but unused by the OMP implementation.

A trivial ``zero_baseline`` (``x_hat = 0, sigma_hat = 1``) is logged alongside to establish the lower bound of the reward scale: it scores NMSE ≈ 0.135 (theoretical ``exp(-1/τ)``) and support-F1 = 0, but earns a near-perfect conformal component because a unit-wide interval legitimately covers ``N(0, 1)`` amplitudes at target 90%.

## Tests (Day-2 gate — all green)

`tests/test_sparse_fourier.py` (20 tests), `tests/test_conformal.py` (12), `tests/test_forward_ops.py` (7) — full suite 39 tests, `pytest` 0.1–0.3s. Key acceptance checks:

1. ``generate_instance`` is deterministic given a seed. ✓
2. ``forward(x_true, mask)`` equals ``y`` up to the noise level. ✓
3. ``ista_baseline`` (OMP) achieves ``score_nmse > 0.7`` on default hyperparameters. ✓
4. ``zero_baseline`` achieves ``score_nmse < 0.15`` (theoretical value ≈ 0.135). ✓
5. Empirical coverage on held-out calibration seeds is within ``[0.75, 1.0]`` of the 0.9 target. ✓ (slow-marked.)
6. ``score(prediction, instance)`` returns floats in ``[0, 1]`` for every component. ✓
7. ``A^T`` is verified self-adjoint against ``A`` on a random vector. ✓

## File layout

```
src/verifiable_labs_envs/
├── conformal.py                 # split-conformal calibration utilities (shared)
├── forward_ops.py               # JAX forward operators (shared)
└── envs/
    └── sparse_fourier.py        # Instance, load_environment, scorer, ISTA baseline
tests/
└── test_sparse_fourier.py
notebooks/
└── 01_sparse_fourier.ipynb      # walkthrough — Day 5 polish
```

## Out of scope for Day 2

- Complex-valued `x*` (keep real for the baseline; add complex variant later).
- 2D sparse Fourier. Keep 1D — much cheaper to iterate.
- Learned priors / deep unrolled networks. Use ISTA as the only baseline for the MVP; the point is the reward design, not SOTA reconstruction.
- Integration with Prime Intellect `verifiers` hub submission. Do after all three envs are stable.
