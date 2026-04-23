# verifiable-labs-envs

Reinforcement-learning environments for scientific reasoning — physics-grounded inverse problems with uncertainty-calibrated rewards.

> **Status (2026-04-23):** Day 1 of an 11-day sprint. Scaffolding only. First working environment ships by 2026-04-24.

## What this is

Frontier reasoning models are trained with verifiable rewards (RLVR). Today's RL environments are mostly text-only, saturate quickly, and miss the continuous, ill-posed reasoning that real science requires. This package provides environments where:

1. The **forward operator** is exact and JIT-compiled (JAX), so a model must actually invert physics.
2. The **reward** is a weighted sum of reconstruction quality (PSNR, SSIM, or task-appropriate metric) and **conformal-prediction coverage** — models are rewarded for honest posterior width, not overconfident point estimates.
3. Measurements are **procedurally regenerated per evaluation call**, so fixed-string memorization is structurally impossible.

## Planned environments (v0.0.1 sprint)

| # | Environment | Status |
|---|---|---|
| 1 | `sparse-fourier-recovery` | 🚧 in progress (Day 2) |
| 2 | `super-resolution-div2k-x4` | ⏳ planned (Day 3) |
| 3 | `lodopab-ct-simplified` | ⏳ planned (Day 4) |

## Install (once Day 1 is done)

```bash
git clone https://github.com/verifiable-labs/envs
cd envs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quickstart (target API — not implemented yet)

```python
from verifiable_labs_envs import load_environment

env = load_environment("sparse-fourier-recovery")
result = env.run_baseline()
print(result.reward)
```

## Author

Stelios Zacharioudakis — finishing BSc CS at the University of Athens (NKUA). Research on calibrated astronomical inverse imaging.

## License

Apache 2.0. See `LICENSE`.
