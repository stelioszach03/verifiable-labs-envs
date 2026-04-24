#!/usr/bin/env python3
"""Populate the six environments/ directories created by `prime env init` with
real wrapper modules that re-export load_environment from the monorepo.

Each env directory gets:
- ``<env_id>.py``      — replaces the NotImplementedError stub
- ``pyproject.toml``   — keeps prime-compatible metadata but points the
                          dependency at our monorepo so the wrapper can import
                          the real env code at runtime.
- ``README.md``         — per-env description + install snippet.

Run from the repo root: python scripts/populate_prime_envs.py
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO / "environments"

# (prime-env-id, python-module-name-with-underscores, monorepo-env-module, tagline)
SPECS = [
    ("sparse-fourier-recovery", "sparse_fourier_recovery", "sparse_fourier",
     "1D sparse Fourier recovery with OMP baseline and conformal σ̂."),
    ("sparse-fourier-recovery-multiturn", "sparse_fourier_recovery_multiturn",
     "sparse_fourier_multiturn",
     "3-turn sparse Fourier recovery with residual feedback between turns."),
    ("sparse-fourier-recovery-tools", "sparse_fourier_recovery_tools",
     "sparse_fourier_tools",
     "Tool-use sparse Fourier recovery (fft, ifft, ista, check_residual)."),
    ("super-resolution-div2k-x4", "super_resolution_div2k_x4", "super_resolution",
     "4× single-image super-resolution with a bicubic baseline."),
    ("lodopab-ct-simplified", "lodopab_ct_simplified", "lodopab_ct",
     "2D parallel-beam CT (phantom or real LoDoPaB slices via use_real_data)."),
    ("lodopab-ct-simplified-multiturn", "lodopab_ct_simplified_multiturn",
     "lodopab_ct_multiturn",
     "3-turn CT reconstruction with FBP-domain residual feedback."),
]


def _py_body(mod_name: str, monorepo_mod: str, tagline: str) -> str:
    return f'''"""{tagline}

Prime Intellect Hub wrapper around ``verifiable_labs_envs.envs.{monorepo_mod}``.
The monorepo at https://github.com/stelioszach03/verifiable-labs-envs is the
source of truth; this file is a thin re-export so the env can be installed
and discovered via the Prime Intellect Environments Hub.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.{monorepo_mod} import load_environment as _le


def load_environment(**kwargs: Any):
    """Factory for the ``{mod_name.replace('_', '-')}`` environment.

    Passes kwargs through to the monorepo's ``load_environment`` (accepts
    ``calibration_quantile``, ``fast``, and env-specific options like
    ``max_turns`` or ``use_real_data`` where applicable).
    """
    return _le(**kwargs)
'''


def _pyproject(env_id: str, mod_name: str, tagline: str, version: str = "0.2.0") -> str:
    return f'''[project]
name = "{env_id}"
description = "{tagline}"
tags = ["scientific-reasoning", "inverse-problems", "rlvr", "conformal", "eval"]
version = "{version}"
requires-python = ">=3.11"
dependencies = [
    # verifiers>=0.1.12 (not >=0.1.13): the 0.1.13 line is dev-only at
    # publish time, and PEP 440 treats 0.1.13.devN as < 0.1.13, so
    # `>=0.1.13` would reject dev releases even with pip --pre. The
    # widened pin lets Hub consumers install from stable releases. See
    # docs/PRIME_INTELLECT_VERIFICATION.md for the reproducer.
    "verifiers>=0.1.12",
    "verifiable-labs-envs @ git+https://github.com/stelioszach03/verifiable-labs-envs.git@main",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.metadata]
# The monorepo isn't on PyPI yet, so the verifiable-labs-envs dep is a
# Git URL. Hatchling refuses direct references by default; this flag
# whitelists them so the wheel can build.
allow-direct-references = true

[tool.hatch.build]
include = ["{mod_name}.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 5
rollouts_per_example = 3
'''


def _readme(env_id: str, monorepo_mod: str, tagline: str) -> str:
    return f"""# {env_id}

{tagline}

Verifiable Labs Scientific-RL environment. Published as a thin wrapper around the monorepo at https://github.com/stelioszach03/verifiable-labs-envs — the wrapper pulls the monorepo as a Git dependency so the full source of truth (rewards, forward operators, LLM adapter, conformal calibration) stays in one place.

## Install

```bash
prime env install verifiable-labs/{env_id}
```

## Use

```python
from verifiers import load_environment    # or Prime SDK equivalent
env = load_environment("{env_id}")
out = env.run_baseline(seed=0)
print(out["reward"])
```

See the monorepo README + docs for the reward spec, contamination story, and benchmark data.
"""


def main() -> None:
    for env_id, mod_name, monorepo_mod, tagline in SPECS:
        d = ENVS_DIR / mod_name
        if not d.exists():
            print(f"skip {env_id}: directory {d} not initialized (run `prime env init` first)")
            continue
        (d / f"{mod_name}.py").write_text(_py_body(mod_name, monorepo_mod, tagline))
        (d / "pyproject.toml").write_text(_pyproject(env_id, mod_name, tagline))
        (d / "README.md").write_text(_readme(env_id, monorepo_mod, tagline))
        print(f"wrote {d.relative_to(REPO)}")


if __name__ == "__main__":
    main()
