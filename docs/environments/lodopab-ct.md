# `lodopab-ct-simplified`

**Low-dose CT reconstruction.** A simplified variant of the
[LoDoPaB-CT](https://lodopab.grand-challenge.org/) benchmark — the
model receives a sinogram (Radon transform of a phantom image) and must
return a reconstruction.

## Problem

| | |
|---|---|
| image size | 64 × 64 |
| projection geometry | parallel-beam, 30 angles |
| dose level | low (10 % of full-dose) |
| ground truth | toy phantoms (ellipses + Shepp-Logan-shaped components) |

The classical baseline is **FBP** (Filtered Back-Projection) followed
by simple TV denoising. The env exposes `env.run_baseline(seed)` for
direct comparison.

## Variants

- `lodopab-ct-simplified` — single-turn.
- `lodopab-ct-simplified-multiturn` — turn-2/3 feedback is the
  FBP-domain residual image of the previous reconstruction.

## Schema

The model returns one JSON object containing a flattened image:

```json
{
  "image_x255":      [0, 12, 47, …, 198],   // 4096 ints in [0, 255]
  "uncertainty_x255": [10, 8, 12, …, 5]     // same length, per-pixel σ_hat
}
```

The platform multiplies by `1/255` to recover floats in `[0, 1]`.

## Reward decomposition

```
reward = 0.5 * ssim_score + 0.5 * conformal_score
```

- `ssim_score` — structural similarity index in `[0, 1]`.
- `conformal_score` — fraction of pixels where the truth falls inside
  the calibrated `[image_x255 - q · uncertainty_x255, image_x255 + q · uncertainty_x255]`
  interval.

NMSE is intentionally **not** used for CT: a uniformly grey
reconstruction can game NMSE on toy phantoms; SSIM penalises that.

## Why this env exists

CT is the canonical 2D imaging inverse problem. Including it tests
whether LLMs can generalise from 1D Fourier (sparse-Fourier) to a 2D
operator with a different forward-operator structure (Radon transform).
The benchmark in `paper/` shows substantial drops from sparse-Fourier
to CT for every model — strong evidence of weak cross-env transfer.

## Real-data variant

The optional `[ct-real]` extras install `dival` + `odl` to run the
env on actual LoDoPaB-CT challenge data (rather than the synthetic
toy phantoms). This is opt-in because the dependency footprint is
heavy. See [`docs/PRIME_INTELLECT.md`](../PRIME_INTELLECT.md) for the
real-data evaluation pipeline.

## Source

[`src/verifiable_labs_envs/envs/lodopab_ct.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/src/verifiable_labs_envs/envs/lodopab_ct.py).
