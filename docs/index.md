# Verifiable Labs

**Reinforcement-learning environments for scientific reasoning** —
physics-grounded inverse problems with **conformal-calibrated rewards**
and **procedurally regenerated** instances.

[![Status: alpha](https://img.shields.io/badge/status-v0.1.0--alpha-orange)](https://github.com/stelioszach03/verifiable-labs-envs)
[![Hosted API](https://img.shields.io/badge/hosted_API-Tier_1-blue)](api-reference/rest-api.md)
[![Python SDK](https://img.shields.io/badge/pypi-verifiable--labs-green)](api-reference/python-sdk.md)

## What this is

A benchmark **and** a reward source for evaluating frontier LLMs on
hard scientific tasks where the answer is verifiable by physics:
sparse Fourier recovery, low-dose CT reconstruction, MRI-knee
imaging, phase retrieval, super-resolution, and more.

```python
from verifiable_labs import Client

c = Client(base_url="https://api.verifiable-labs.com")
env = c.env("stelioszach/sparse-fourier-recovery")
result = env.evaluate(seed=42, answer=my_model_output)
print(result.reward, result.components, result.coverage)
```

## Why "verifiable"

Each reward is computed against ground truth that the platform owns:

1. **Procedural regeneration.** Every (env, seed) pair produces a
   fresh problem instance. Effective instance count exceeds
   `2^64` per env — model providers can't memorise the test set.
2. **Conformal calibration.** Reward signals include a
   coverage-calibrated uncertainty term, fitted offline on a
   held-back pool. The model's stated uncertainty is *measurable*,
   not vibes.
3. **Forward-operator audit trail.** Every env exposes its forward
   operator (`A`) and reward function as plain Python; the same
   instance + same prediction always yields the same reward, bit-exact.

Read the full methodology in [Concepts → Conformal rewards](concepts/conformal-rewards.md)
and [Concepts → Procedural regeneration](concepts/procedural-regeneration.md).

## Three ways to consume

| surface | use when |
|---|---|
| [REST API](api-reference/rest-api.md) | quick eval from any language; no Python install |
| [Python SDK](api-reference/python-sdk.md) | `pip install verifiable-labs`; sync + async clients |
| [Local envs](environments/index.md) | full reproducibility; integrate into a training loop |

## Project status

**v0.1.0-alpha.** 10 environments, 5 frontier models benchmarked,
337+ tests, paper at [arXiv preprint](research/paper.md). The hosted
API runs without authentication and stores sessions in memory; treat
the public endpoint as a developer playground while v0.2 (auth,
Redis-backed sessions, multi-turn turn-dispatch) lands.

See the [roadmap](company/roadmap.md) for what's planned next.
