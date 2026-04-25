# `sparse-fourier-recovery`

**1D sparse signal recovery from a partial Fourier mask.** The
canonical compressed-sensing setup: observe `y = A(x) + ε` where `A`
is the partial DFT on a known random mask, `x` is `k`-sparse, and the
goal is to recover the support and amplitudes.

## Problem

| | |
|---|---|
| signal length `n` | 256 |
| sparsity `k` | 10 |
| mask | uniformly random, `m = 64` measurements |
| noise `σ` | 0.05 (Gaussian, applied in Fourier domain) |
| unknowns | `support_idx ∈ {0…n-1}^k`, real amplitudes |

The classical baseline is **OMP** (Orthogonal Matching Pursuit), which
the env exposes via `env.run_baseline(seed)`. OMP achieves mean reward
≈ 0.78 on this env — well above the LLM scores reported in
`paper/`.

## Variants

- `sparse-fourier-recovery` — single-turn.
- `sparse-fourier-recovery-multiturn` — up to 3 turns; turn-2/3
  feedback is the Fourier-domain residual `r = y - A(x_hat)` of the
  previous answer.
- `sparse-fourier-recovery-tools` — single-call tool
  (`compute_fft(support, amp_x1000)`) returning the residual.

## Schema

The model must return one JSON object:

```json
{
  "support_idx":      [12, 47, 91, 122, 138, 154, 177, 198, 219, 240],
  "support_amp_x1000": [800, -1200, 450, -900, 1100, -300, 700, 1400, -550, 600]
}
```

`support_idx` is `k` sorted integers in `[0, n)`; `support_amp_x1000`
is `k` signed integers (real amplitudes scaled by 1000). No prose,
no markdown fences.

## Reward decomposition

```
reward = 0.4 * nmse_score + 0.3 * support_score + 0.3 * conformal_score
```

- `nmse_score = exp(-NMSE)` where `NMSE = ||x - x_hat||² / ||x||²`.
- `support_score = |support_true ∩ support_hat| / k`.
- `conformal_score = 1.0` if every truth coordinate in the calibrated
  uncertainty interval, else fractional based on coverage.

## Why this env exists

Sparse Fourier recovery is the **simplest realistic inverse problem**:
small enough that an LLM can plausibly attempt it from the prompt
alone, structured enough that classical solvers crush LLMs by a wide
margin. It anchors the v0.1 leaderboard at the easy end.

It also lets us test the multi-turn dispatch cheaply: the residual
feedback message is small (a 64-vector of complex numbers) and
informative (zero residual means the answer is exact).

## Source

[`src/verifiable_labs_envs/envs/sparse_fourier.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/src/verifiable_labs_envs/envs/sparse_fourier.py).

## See also

- [Concepts → Multi-turn](../concepts/multi-turn.md)
- [Concepts → Tool use](../concepts/tool-use.md)
- [Training-proof tutorial](../tutorials/training-with-envs.md) — uses
  this env's multiturn variant
