# `phase-retrieval`

**Phase retrieval from intensity-only Fourier measurements.**
Observe `|F·x|²` (the modulus-squared Fourier transform of a signal)
and recover both magnitude and phase. A canonical
non-convex inverse problem from X-ray crystallography and coherent
diffraction imaging.

## Problem

| | |
|---|---|
| signal length `n` | 128 |
| support size `k` | 8 (sparse signals) |
| measurements | full Fourier modulus `|F·x|² ∈ ℝ^n` |
| noise | shot noise (Poisson-shaped) on the modulus |

The classical baseline is **HIO** (Hybrid Input-Output, Fienup's
algorithm) with random restarts. Phase retrieval is famously hard;
HIO is competitive but not always optimal, leaving room for an LLM to
make a meaningful contribution if it has the right priors.

## Variants

- `phase-retrieval` — single-turn.
- `phase-retrieval-tools` — single-call tool
  (`evaluate_modulus(support, amp_x1000)`) returning the squared error
  against the observed modulus. Tests whether the model can iterate
  intelligently with feedback rather than guess.

## Schema

```json
{
  "support_idx":      [12, 47, 91, 122, 138, 154, 177, 198],
  "support_amp_x1000": [800, -1200, 450, -900, 1100, -300, 700, 1400]
}
```

Same shape as sparse-Fourier; the difference is what's measured (here
the magnitude squared, not the linear measurement).

## Reward decomposition

```
reward = 0.5 * modulus_score + 0.3 * support_score + 0.2 * conformal_score
```

- `modulus_score = exp(- ||y - |F·x_hat|²|| / ||y||)` — direct
  fidelity to the measured intensity.
- `support_score` — same as sparse-Fourier.
- `conformal_score` — calibrated coverage on per-coordinate uncertainty.

Note: phase retrieval has a **trivial sign / shift / time-reversal
ambiguity** — `x` and `-x` produce the same modulus. The reward
function accounts for this by aligning `x_hat` and `x` (sign flip,
circular shift, reflection) before scoring; the four equivalent
solutions all score equally.

## Why this env exists

Phase retrieval is the **hardest** of the v0.1 envs and the one where
LLMs struggle most. In the paper-final benchmark, no model exceeds
mean reward 0.20 on this env — well below the OMP-on-sparse-Fourier
ceiling of 0.78 — which makes it a useful "hard-mode" stress test.

## Source

[`src/verifiable_labs_envs/envs/phase_retrieval.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/src/verifiable_labs_envs/envs/phase_retrieval.py).

## See also

- [Concepts → Tool use](../concepts/tool-use.md)
- The `phase-retrieval-tools` variant is the platform's main tool-use
  testbed.
