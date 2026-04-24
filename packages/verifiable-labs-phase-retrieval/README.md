# verifiable-labs-phase-retrieval

Phase retrieval from magnitude-only Fourier measurements — the canonical test problem for X-ray crystallography, coherent diffraction imaging, semiconductor metrology, and astronomical speckle interferometry.

## Problem

Recover a k-sparse real signal `x ∈ R^n` from `y = |S·F(x)| + noise`, where `F` is the 1D orthonormal DFT and `S` selects `m` of `n` frequency positions. Classical solver: Gerchberg-Saxton iteration (alternating projection between magnitude constraint and sparse-support constraint).

Because `|F(-x)| = |F(x)|`, recovery is unique only up to a global sign flip. The scorer handles this automatically (evaluates both x_hat and -x_hat, keeps the better).

## Install

```bash
pip install "verifiable-labs-phase-retrieval @ git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-phase-retrieval"
```

Via Prime Intellect Hub (once public):

```bash
prime env install stelioszach/phase-retrieval
```

## Usage

```python
from verifiers import load_environment
env = load_environment("phase-retrieval")
out = env.run_baseline(seed=0)
print(out["reward"])
```

## v1.0 benchmark (2026-04-24)

3 models × 3 instances × 2 variants (single / 3-turn multi):

| model | single mean | multi-turn mean |
|---|---:|---:|
| claude-haiku-4.5 | 0.455 | 0.331 |
| gpt-5.4-mini | 0.365 | 0.343 |
| gpt-5.4-nano | 0.299 | 0.353 |

Classical Gerchberg-Saxton baseline: ~0.29. Phase retrieval is genuinely hard — Haiku 4.5 is the only tested model that beats the classical baseline, and multi-turn magnitude-residual feedback does not consistently help. This is a legitimate hard-RL-environment finding.

Raw data: `results/phase_retrieval_v1_benchmark.csv` in the monorepo.

## License

Apache-2.0 (see LICENSE).
