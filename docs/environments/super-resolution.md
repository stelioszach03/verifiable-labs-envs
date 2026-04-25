# `super-resolution`

**2× single-image super-resolution.** Observe a low-resolution image
(8 × 8) and recover the original 16 × 16. Classic ill-posed image
restoration setup.

## Problem

| | |
|---|---|
| HR image | 16 × 16 |
| LR image | 8 × 8 (2× bicubic downsample + Gaussian noise) |
| noise `σ` | 0.04 |
| ground truth | toy structured images (gradients, geometric primitives) |

The classical baseline is **bicubic upsampling**, which gives an
honest floor — anything beating bicubic is making non-trivial use of
the prior.

## Variants

Single-turn only in v0.1. A multi-turn variant is on the roadmap.

## Schema

```json
{
  "image_x255":      [...],  // 256 ints in [0, 255]
  "uncertainty_x255": [...]  // same shape
}
```

256 = 16 × 16 flattened.

## Reward decomposition

```
reward = 0.6 * ssim_score + 0.4 * conformal_score
```

Higher SSIM weight than CT/MRI because the conformal pool is smaller
(256 pixels) and the calibration noise is correspondingly bigger.

## Why this env exists

Super-resolution is the smallest 2D imaging env in v0.1 — useful as a
fast canary for "does the model understand 2D image structure at all".
A model that fails super-resolution but passes 1D sparse-Fourier is
likely a 1D-only solver; one that passes super-resolution and fails
CT might be limited by problem size, not problem type.

## Source

[`src/verifiable_labs_envs/envs/super_resolution.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/src/verifiable_labs_envs/envs/super_resolution.py).
