# verifiable-labs-lodopab-ct-multiturn

3-turn CT reconstruction with FBP-domain residual feedback

This is a thin wrapper over the monorepo `verifiable-labs-envs`. Installing it gives you:

- A verifiers-compatible entry point: ``verifiers.environments → lodopab-ct-simplified-multiturn``.
- A direct factory: ``from verifiable_labs_lodopab_ct_multiturn import load_environment``.

## Install

From GitHub (subdirectory):
```
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-lodopab-ct-multiturn"
```

Once published to the Prime Intellect Environments Hub:
```
prime env install verifiable-labs/lodopab-ct-simplified-multiturn
```

## Use

```python
from verifiable_labs_lodopab_ct_multiturn import load_environment

env = load_environment()
instance = env.generate_instance(seed=0)
out = env.run_baseline(seed=0)  # or env.run_rollout(solver, instance) for multi-turn
print(out["reward"], out["components"])
```

Full documentation of rewards, forward operators, and scoring lives in the monorepo at
`https://github.com/stelioszach03/verifiable-labs-envs`.
