# verifiable-labs-sparse-fourier-multiturn

3-turn sparse Fourier recovery with residual feedback between turns

This is a thin wrapper over the monorepo `verifiable-labs-envs`. Installing it gives you:

- A verifiers-compatible entry point: ``verifiers.environments → sparse-fourier-recovery-multiturn``.
- A direct factory: ``from verifiable_labs_sparse_fourier_multiturn import load_environment``.

## Install

From GitHub (subdirectory):
```
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-sparse-fourier-multiturn"
```

Once published to the Prime Intellect Environments Hub:
```
prime env install verifiable-labs/sparse-fourier-recovery-multiturn
```

## Use

```python
from verifiable_labs_sparse_fourier_multiturn import load_environment

env = load_environment()
instance = env.generate_instance(seed=0)
out = env.run_baseline(seed=0)  # or env.run_rollout(solver, instance) for multi-turn
print(out["reward"], out["components"])
```

Full documentation of rewards, forward operators, and scoring lives in the monorepo at
`https://github.com/stelioszach03/verifiable-labs-envs`.
