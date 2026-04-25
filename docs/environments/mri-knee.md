# `mri-knee-reconstruction`

**Compressed-sensing MRI reconstruction.** The model receives
under-sampled k-space data and a Cartesian sampling mask, and must
return a reconstructed image.

## Problem

| | |
|---|---|
| image size | 64 × 64 |
| sampling | Cartesian, central + random, 25 % acceleration |
| coil count | single-coil (simplified — full multi-coil is v0.2) |
| ground truth | derived from a simplified knee phantom (no patient data) |

The classical baseline is **zero-filled IFFT** followed by TV
regularisation. Mid-tier — the env's reward window is narrower than
sparse-Fourier (less room to fail catastrophically) but recoveries are
more demanding.

## Variants

- `mri-knee-reconstruction` — single-turn.
- `mri-knee-reconstruction-multiturn` — turn-2/3 feedback is the
  k-space residual `r = y - F·M·x_hat` plus a small zero-filled
  recon hint.

## Schema

```json
{
  "image_x255":      [...],  // 4096 ints
  "uncertainty_x255": [...]
}
```

Same as `lodopab-ct-simplified`.

## Reward decomposition

```
reward = 0.5 * ssim_score + 0.5 * conformal_score
```

## Why this env exists

MRI exercises a **complex-valued Fourier-based forward operator** in
2D — close enough to sparse-Fourier conceptually that we can test
whether LLMs use 1D intuition correctly in 2D. Spoiler from the
paper: most don't.

The simplified single-coil setup keeps the prompt size manageable
(64 × 64 = 4096 pixels) while preserving the qualitative challenge.
A v0.2 multi-coil variant is on the [roadmap](../company/roadmap.md).

## No patient data

Both this env and `lodopab-ct` use synthetic phantoms only. No PHI,
no fastMRI license, nothing that requires a data-use agreement. The
real-data variants (gated behind extras) wrap the public LoDoPaB and
fastMRI distributions but ship empty until the user opts in.

## Source

[`src/verifiable_labs_envs/envs/mri_knee.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/src/verifiable_labs_envs/envs/mri_knee.py).
