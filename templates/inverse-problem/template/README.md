# __ENV_ID__

Verifiable Labs scientific RL environment for **__DOMAIN__**.

This is a scaffolded env. The skeleton provides:

- A `generate_ground_truth` stub for your domain's data source
  (`__ENV_PY__/data.py`).
- A `forward(x)` stub for your physics
  (`__ENV_PY__/forward_op.py`).
- A `score_point_estimate` baseline (NMSE) plus a conformal-coverage
  reward term (`__ENV_PY__/reward.py`).
- A `build_user_prompt` / `parse_response` adapter for LLM solvers
  (`__ENV_PY__/adapter.py`).
- A test suite covering instance shape, scoring range, adapter
  round-trip (`tests/`).

Replace each `NotImplementedError` with your domain logic, then run:

```bash
python ../scripts/validate_env.py environments/__ENV_PY__
```

from the repo root. The validator runs:

1. **pytest** on this env's `tests/` (must all pass).
2. **calibration coverage** within ±0.05 of the target (default α = 0.1
   → target 0.90).
3. **procedural-regeneration** check: `EFFECTIVE_INSTANCES > 1e15`.
4. **adapter compatibility** with `verifiers.load_environment`.

## Usage

```python
from __ENV_PY__ import load_environment

env = load_environment()             # auto-calibrates conformal quantile
out = env.run_baseline(seed=0)
print(out["reward"])                 # in [0, 1]
```

For tests, skip calibration:

```python
env = load_environment(calibration_quantile=2.0)
```

## Where to look in the existing repo

- `verifiable_labs_envs.envs.sparse_fourier` — reference 1D
  compressed-sensing impl.
- `verifiable_labs_envs.envs.lodopab_ct` — reference 2D Radon impl.
- `verifiable_labs_envs.envs.phase_retrieval` — magnitude-only DFT,
  classical Gerchberg-Saxton baseline.

## License

Apache-2.0.
