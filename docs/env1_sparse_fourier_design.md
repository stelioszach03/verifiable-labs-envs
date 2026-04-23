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

Pre-computed **calibration set**: 500 instances solved by the `ista` baseline, producing residuals `r_i = |x_hat_i - x*_i| / (sigma_hat_i + ε)`. The `(1 − α)`-quantile `q_α` is cached with the environment (α = 0.1 by default → 90% target coverage).

At score time:

1. Build interval `[x_hat - q_α · sigma_hat, x_hat + q_α · sigma_hat]`.
2. Empirical coverage `c = mean(x* inside interval)`.
3. Target coverage `c_target = 1 − α = 0.9`.
4. `score_conformal = 1 − |c − c_target|` (bounded `[0, 1]`, peaks when coverage matches target, penalizes both over- and under-confidence).

**Why this is the important term:** a model that outputs `sigma_hat = 1e-6` everywhere gets a good `score_nmse` if its point estimate is good, but `score_conformal ≈ 0` because its intervals are empty. A model that outputs `sigma_hat = 1e6` everywhere gets full coverage but no information — same failure mode. The only way to win is honest uncertainty. This reward shape is the direct translation of the split-conformal method in the draft EHT paper.

## Baseline — ISTA with σ̂ from residuals

Iterative soft-thresholding with a fixed number of iterations (default 200), step size from Lipschitz constant. For `sigma_hat`: run ISTA `B = 20` times with independent noise draws (bootstrap), take per-entry std. Crude but gives a legitimate uncertainty signal.

We will also log a trivial "zero baseline" (`x_hat = 0, sigma_hat = 1`) so the lower bound of the reward is visible in the benchmark table.

## Tests (Day-2 gate)

`tests/test_sparse_fourier.py`:

1. `generate_instance` is deterministic given a seed.
2. `forward(x_true, mask)` equals `y` up to the noise level.
3. `ista_baseline` achieves `score_nmse > 0.6` on default hyperparameters (sanity that the problem is solvable).
4. `zero_baseline` achieves `score_nmse < 0.1`.
5. Conformal coverage of the calibration set is within `[0.85, 0.95]` (the 90% target ± tolerance).
6. `score(prediction, instance)` returns floats in `[0, 1]` for every component.

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
