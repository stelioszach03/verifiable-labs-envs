# Writing a custom environment

End goal: scaffold a new inverse-problem env, fill in the forward
operator + reward, and pass the validator. Estimated time: 2-4 hours
for a 1D problem; longer for 2D imaging or non-trivial priors.

## TL;DR

```bash
# 1. Scaffold.
verifiable init-env seismic-fwi --domain "geophysics-1d-FWI"

# 2. Fill in the NotImplementedError stubs in environments/seismic-fwi/.
#    See "What you need to write" below.

# 3. Validate.
verifiable validate-env environments/seismic-fwi
```

## What you need to write

The scaffold creates a Python package with these files. You fill in
the marked stubs; everything else stays as-is.

```
environments/seismic-fwi/
├── pyproject.toml                 (auto-filled)
├── README.md                      (auto-filled; edit prose)
├── seismic_fwi/
│   ├── __init__.py                (auto-filled — exports)
│   ├── data.py                    ← FILL: ground-truth sampler
│   ├── forward_op.py              ← FILL: forward + adjoint operators
│   ├── reward.py                  ← FILL (optional): custom reward
│   ├── adapter.py                 ← FILL: prompt + parse_response
│   └── env.py                     (auto-filled — orchestrates everything)
├── conftest.py                    (auto-filled)
└── tests/
    ├── test_env.py                (skeleton; add your assertions)
    ├── test_reward.py             (skeleton)
    └── test_adapter.py            (skeleton)
```

### `data.py` — ground-truth sampler

```python
import numpy as np

def generate_ground_truth(rng: np.random.Generator) -> np.ndarray:
    """Sample one ground-truth instance from your prior. The sampler
    must consume *only* `rng` — no system time, no global state — so
    `seed → instance` stays deterministic."""
    raise NotImplementedError
```

For seismic FWI you might sample 5-10 layer thicknesses from a
log-normal distribution and 5-10 layer velocities from a problem-
specific prior.

### `forward_op.py` — forward operator

```python
def forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Apply your forward operator A to signal x; return the
    measurement y the model will see."""
    raise NotImplementedError
```

### `adapter.py` — LLM adapter

```python
SYSTEM_PROMPT = """You are an expert at 1D seismic full-waveform
inversion. Given a synthetic seismogram, recover the layered
velocity model.

Output exactly one JSON object: {"layer_velocities": [...]}.
No prose, no markdown.
"""

def build_user_prompt(instance) -> str:
    return f"Seismogram: {list(instance.y)}, n_layers: {instance.n_layers}"

def parse_response(text: str, instance) -> Prediction:
    # Strip ``` fences, parse JSON, validate types/ranges, return Prediction.
    ...
```

## Validator checks

`verifiable validate-env <path>` runs four independent checks:

1. **`pytest`** on the env's `tests/` directory. Skipped tests (e.g.
   on `NotImplementedError`) count as a pass while you're iterating.
2. **Calibration coverage.** Runs `env.run_baseline(seed)` on
   `--n-cal` (default 50) fresh seeds, reads `meta["coverage"]` from
   each, asserts the mean is within `--tolerance` (default 0.05) of
   `1 - α`.
3. **Procedural-regeneration count.** Reads the env package's
   `EFFECTIVE_INSTANCES` constant, asserts `> 1e15`. Smaller pools
   are vulnerable to memorisation.
4. **Adapter compatibility.** Tries `verifiers.load_environment(env_id)`
   first (Prime Intellect Hub path), falls back to
   `verifiable_labs_envs.load_environment`, then asserts
   `env.generate_instance(0)` succeeds.

The validator exits non-zero if any check fails.

## Iterating

The validator is fast — run it after every change:

```bash
verifiable validate-env environments/seismic-fwi
```

While you're filling in the stubs the calibration + adapter checks
will fail (expected — they need a working forward operator). You can
skip them temporarily:

```bash
verifiable validate-env environments/seismic-fwi --skip-adapter-check --n-cal 5
```

Once your forward operator + reward are correct, re-enable the full
checks and confirm green before opening a PR.

## Submitting your env

Two paths:

1. **As a plugin.** The scaffold's `pyproject.toml` already declares
   the entry point `[project.entry-points."verifiers.environments"]`.
   Anyone who `pip install`s your env package can
   `load_environment("<env-id>")`. Publish to PyPI under any name.
2. **Into the v0.1 monorepo.** Open a PR adding the env id to
   `src/verifiable_labs_envs/__init__.py::ENV_MODULES` and submit.
   The platform tests it on every CI run. The validator is the merge
   gate.

## Concrete env ideas

The scaffold's top-level [README.md](../templates/inverse-problem/README.md)
lists five concrete inverse-problem starting points:

- holographic 3D reconstruction (3D scattering operator)
- electron-microscope tomography (3D Radon transform)
- seismic full-waveform inversion 1D (wave-equation forward op)
- inverse rendering (differentiable scene → image)
- protein residue distogram (graph-structured prior)

Pick one matching your domain expertise. The scaffold handles the
boilerplate — the physics belongs to you.

## See also

- [`docs/ONBOARD_AGENT_5_MIN.md`](ONBOARD_AGENT_5_MIN.md) — onboard your agent
- [`docs/concepts/conformal-rewards.md`](concepts/conformal-rewards.md) — what
  the calibration check is verifying
- [`docs/concepts/procedural-regeneration.md`](concepts/procedural-regeneration.md)
  — why `EFFECTIVE_INSTANCES > 1e15` is the floor
- [`templates/inverse-problem/`](../templates/inverse-problem) — the canonical
  scaffold reference
