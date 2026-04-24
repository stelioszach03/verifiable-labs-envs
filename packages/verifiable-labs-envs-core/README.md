# verifiable-labs-envs-core

Shared core for the Verifiable Labs environment suite: the conformal-coverage reward math, the JAX/numpy forward operators, and the LLM-solver / adapter base classes that every env package depends on.

Installing this package transitively pulls in the monorepo `verifiable-labs-envs` so every env is available via the ``load_environment()`` factory; it also gives downstream code a stable import path for the shared primitives:

```python
from verifiable_labs_envs_core import (
    conformal, forward_ops,
    LLMSolver, OpenRouterSolver, FakeLLMSolver,
    EnvAdapter, register_adapter,
)
```

## Install

From source:
```
pip install git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-envs-core
```

Once published to Prime Intellect Environments Hub:
```
prime env install verifiable-labs/envs-core
```
