# verifiable-labs-mri-knee

MRI knee reconstruction from 4×-undersampled Cartesian k-space — the canonical accelerated-MRI problem.

## Problem

Recover a 16×16 grayscale image `x` from `y = M ⊙ F₂(x) + noise` where `F₂` is the 2D orthonormal DFT and `M` is a Cartesian undersampling mask (dense DC-centered region + random outer columns, 4× acceleration). Classical baseline: zero-filled inverse FFT.

v1 synthesizes ground truth from `skimage.data` public-domain images (resized to 16×16, grayscale, normalized). fastMRI integration is a v2 follow-up (gated by NYU application) — see `docs/MRI_DATA.md` in the monorepo.

## Install

```bash
pip install "verifiable-labs-mri-knee @ git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-mri-knee"
```

Prime Intellect Hub:

```bash
prime env install stelioszach/mri-knee-reconstruction
```

## Usage

```python
from verifiers import load_environment
env = load_environment("mri-knee-reconstruction")
out = env.run_baseline(seed=0)
print(out["reward"])
```

## v1.0 benchmark (2026-04-24)

3 models × 3 instances × 2 variants = 18 episodes, $0.10 total, 18/18 parsed:

| model | single mean | multi-turn mean |
|---|---:|---:|
| claude-haiku-4.5 | 0.682 | 0.683 |
| gpt-5.4-mini | 0.674 | 0.667 |
| gpt-5.4-nano | 0.654 | 0.589 |

Zero-filled-IFFT classical baseline mean: ~0.65. LLMs track the baseline closely and do not robustly improve over it — medical-image reconstruction is harder than it looks with a compact int-pixel representation.

Raw data: `results/mri_knee_v1_benchmark.csv` in the monorepo.

## License

Apache-2.0.
