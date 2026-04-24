#!/usr/bin/env python3
"""Generate the 7 verifiers-compatible package directories under packages/.

Each package is a thin wrapper over the monorepo ``verifiable-labs-envs``:
- ``verifiable-labs-envs-core`` re-exports the shared utilities (conformal,
  forward_ops, solvers base + adapters base + shared tools).
- Six env packages (sparse-fourier, sparse-fourier-multiturn,
  sparse-fourier-tools, super-resolution, lodopab-ct, lodopab-ct-multiturn)
  re-export ``load_environment`` and declare the verifiers entry-point.

Idempotent: overwrites existing packages/<name>/pyproject.toml + README.md +
module on every run.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO / "packages"

CORE_PKG = "verifiable-labs-envs-core"
CORE_MOD = "verifiable_labs_envs_core"

# (pypi-name, python-module-name, registry-name, env-module, tagline)
ENV_SPECS = [
    ("verifiable-labs-sparse-fourier", "verifiable_labs_sparse_fourier",
     "sparse-fourier-recovery", "sparse_fourier",
     "1D sparse Fourier recovery with OMP baseline + conformal σ̂"),
    ("verifiable-labs-sparse-fourier-multiturn", "verifiable_labs_sparse_fourier_multiturn",
     "sparse-fourier-recovery-multiturn", "sparse_fourier_multiturn",
     "3-turn sparse Fourier recovery with residual feedback between turns"),
    ("verifiable-labs-sparse-fourier-tools", "verifiable_labs_sparse_fourier_tools",
     "sparse-fourier-recovery-tools", "sparse_fourier_tools",
     "Tool-use sparse Fourier recovery (fft, ifft, ista, check_residual)"),
    ("verifiable-labs-super-resolution", "verifiable_labs_super_resolution",
     "super-resolution-div2k-x4", "super_resolution",
     "4× single-image super-resolution with bicubic baseline"),
    ("verifiable-labs-lodopab-ct", "verifiable_labs_lodopab_ct",
     "lodopab-ct-simplified", "lodopab_ct",
     "2D parallel-beam CT (phantom or real LoDoPaB-CT slices via use_real_data)"),
    ("verifiable-labs-lodopab-ct-multiturn", "verifiable_labs_lodopab_ct_multiturn",
     "lodopab-ct-simplified-multiturn", "lodopab_ct_multiturn",
     "3-turn CT reconstruction with FBP-domain residual feedback"),
]

CORE_PYPROJECT = """[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "verifiable-labs-envs-core"
version = "0.1.0"
description = "Shared core for the Verifiable Labs environment suite (conformal rewards, forward operators, solver + adapter base classes)."
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
authors = [{ name = "Stelios Zacharioudakis" }]
keywords = ["reinforcement-learning", "inverse-problems", "rlvr", "evaluation", "scientific-ml", "conformal-prediction"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "verifiable-labs-envs @ git+https://github.com/stelioszach03/verifiable-labs-envs.git@main",
]

[project.urls]
Homepage = "https://github.com/stelioszach03/verifiable-labs-envs"
Issues = "https://github.com/stelioszach03/verifiable-labs-envs/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/verifiable_labs_envs_core"]
"""

CORE_INIT = """\"\"\"Verifiable Labs core — shared utilities re-exported from the monorepo.

This is a thin wrapper around the source-of-truth ``verifiable_labs_envs``
package. Installing ``verifiable-labs-envs-core`` pulls in the full repo
and exposes the shared modules:
- ``conformal``        — split-conformal calibration + coverage score
- ``forward_ops``      — shared forward operators (Fourier, blur, Radon)
- ``solvers``          — LLMSolver base + OpenRouterSolver + FakeLLMSolver
- ``solvers.adapters`` — per-env prompt/parser adapters

Use this package as a dependency when building downstream env packages
that want a stable API for the shared primitives.
\"\"\"
from verifiable_labs_envs import conformal, forward_ops
from verifiable_labs_envs.solvers import (
    CompletionResult,
    EnvAdapter,
    FakeLLMSolver,
    HAS_OPENROUTER_KEY,
    LLMSolver,
    LLMSolverError,
    OpenRouterSolver,
    register_adapter,
)

__all__ = [
    "conformal",
    "forward_ops",
    "CompletionResult",
    "EnvAdapter",
    "FakeLLMSolver",
    "HAS_OPENROUTER_KEY",
    "LLMSolver",
    "LLMSolverError",
    "OpenRouterSolver",
    "register_adapter",
]
"""

CORE_README = """# verifiable-labs-envs-core

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
"""


def env_pyproject(pypi: str, mod: str, reg: str, tagline: str) -> str:
    return f"""[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "{pypi}"
version = "0.1.0"
description = "{tagline}"
readme = "README.md"
requires-python = ">=3.11"
license = {{ file = "LICENSE" }}
authors = [{{ name = "Stelios Zacharioudakis" }}]
keywords = ["reinforcement-learning", "inverse-problems", "rlvr", "evaluation", "scientific-ml"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "verifiable-labs-envs-core @ git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-envs-core",
]

[project.urls]
Homepage = "https://github.com/stelioszach03/verifiable-labs-envs"
Issues = "https://github.com/stelioszach03/verifiable-labs-envs/issues"

# Verifiers-compatible entry point so `prime env install {pypi}` / the
# Prime Intellect Hub can discover load_environment() automatically.
[project.entry-points."verifiers.environments"]
{reg} = "{mod}:load_environment"

[tool.hatch.build.targets.wheel]
packages = ["src/{mod}"]
"""


def env_init(mod: str, reg: str, env_mod: str, tagline: str) -> str:
    return f'''"""{pypi_tagline_header(mod, reg, tagline)}"""
from verifiable_labs_envs.envs.{env_mod} import load_environment as _load_environment_base

ENV_NAME = "{reg}"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``{reg}`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
'''


def pypi_tagline_header(mod: str, reg: str, tagline: str) -> str:
    return (
        f"Verifiable Labs env wrapper: {reg}.\\n\\n{tagline}\\n\\n"
        f"Thin re-export over ``verifiable_labs_envs.envs``; the monorepo is the "
        f"source of truth. This package exists so the env can be installed and "
        f"discovered independently via the verifiers / Prime Intellect Hub "
        f"entry-point mechanism."
    )


def env_readme(pypi: str, reg: str, env_mod: str, tagline: str) -> str:
    return f"""# {pypi}

{tagline}

This is a thin wrapper over the monorepo `verifiable-labs-envs`. Installing it gives you:

- A verifiers-compatible entry point: ``verifiers.environments → {reg}``.
- A direct factory: ``from {pypi.replace('-', '_')} import load_environment``.

## Install

From GitHub (subdirectory):
```
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/{pypi}"
```

Once published to the Prime Intellect Environments Hub:
```
prime env install verifiable-labs/{reg}
```

## Use

```python
from {pypi.replace('-', '_')} import load_environment

env = load_environment()
instance = env.generate_instance(seed=0)
out = env.run_baseline(seed=0)  # or env.run_rollout(solver, instance) for multi-turn
print(out["reward"], out["components"])
```

Full documentation of rewards, forward operators, and scoring lives in the monorepo at
`https://github.com/stelioszach03/verifiable-labs-envs`.
"""


LICENSE_REF = """Apache License 2.0. See the LICENSE file at the monorepo root:
https://github.com/stelioszach03/verifiable-labs-envs/blob/main/LICENSE
"""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def gen_core() -> None:
    root = PKG_ROOT / CORE_PKG
    write(root / "pyproject.toml", CORE_PYPROJECT)
    write(root / "README.md", CORE_README)
    write(root / "LICENSE", LICENSE_REF)
    write(root / "src" / CORE_MOD / "__init__.py", CORE_INIT)


def gen_env(pypi: str, mod: str, reg: str, env_mod: str, tagline: str) -> None:
    root = PKG_ROOT / pypi
    write(root / "pyproject.toml", env_pyproject(pypi, mod, reg, tagline))
    write(root / "README.md", env_readme(pypi, reg, env_mod, tagline))
    write(root / "LICENSE", LICENSE_REF)
    write(root / "src" / mod / "__init__.py", env_init(mod, reg, env_mod, tagline))


def main() -> None:
    print(f"Generating packages under {PKG_ROOT} ...")
    gen_core()
    print(f"  + {CORE_PKG}")
    for pypi, mod, reg, env_mod, tagline in ENV_SPECS:
        gen_env(pypi, mod, reg, env_mod, tagline)
        print(f"  + {pypi}")
    print(f"Done. {1 + len(ENV_SPECS)} package directories created.")


if __name__ == "__main__":
    main()
