# Tutorial: creating a custom env

End goal: scaffold a new inverse-problem env, fill in the forward
operator + reward, and pass `scripts/validate_env.py`. Estimated time:
2-4 hours for a 1D problem; longer for 2D/3D imaging or non-trivial
priors.

## Prerequisites

- Editable install: `pip install -e ".[dev]"`
- A clear problem statement: forward operator `A`, what the model
  observes, what ground truth looks like, what a good reward is.

## Step 1 — scaffold

```bash
python scripts/create_env.py seismic-fwi --domain "geophysics-1d-FWI"
```

This creates `environments/seismic-fwi/` with:

```
seismic-fwi/
├── pyproject.toml
├── README.md
├── seismic_fwi/
│   ├── __init__.py
│   ├── data.py        # Instance, Prediction dataclasses + ground-truth gen
│   ├── forward_op.py  # forward() and adjoint() — NotImplementedError stubs
│   ├── reward.py      # NMSE + conformal-coverage reward
│   ├── adapter.py     # SYSTEM_PROMPT, build_user_prompt, parse_response
│   └── env.py         # the env class with generate_instance / score / run_baseline
├── conftest.py
└── tests/
    ├── test_env.py
    ├── test_reward.py
    └── test_adapter.py
```

The validator passes immediately on the unfilled scaffold (tests skip
on `NotImplementedError`; calibration check warns; adapter check
warns). That's intentional — you can commit early and fill in the
problem-specific parts incrementally.

## Step 2 — implement the forward operator

Open `seismic_fwi/forward_op.py`:

```python
def forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Apply the forward operator A to signal x."""
    # Your problem-specific code here.
    raise NotImplementedError
```

For `seismic-fwi` (1D full-waveform inversion), `forward` solves the
1D wave equation for a layered velocity model `x` and returns a
synthetic seismogram. Use `numpy` / `scipy` only — no ML
dependencies.

## Step 3 — implement ground-truth generation

`seismic_fwi/data.py`:

```python
def generate_ground_truth(rng: np.random.Generator) -> np.ndarray:
    """Sample a ground-truth instance from the problem's prior."""
    raise NotImplementedError
```

For seismic FWI: sample 5-10 layer thicknesses from a log-normal
distribution and 5-10 layer velocities from a problem-specific
prior. The sampler must consume *only* `rng` so determinism is
preserved — no system time, no global state.

## Step 4 — fill in the env class

`seismic_fwi/env.py` already has the structure:

```python
def generate_instance(self, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    x_true = generate_ground_truth(rng)            # from data.py
    y = forward(x_true, self.hyperparams) + noise  # from forward_op.py
    return Instance(x_true=x_true, y=y, ...)
```

Plus `score(prediction, instance) -> dict` (NMSE + conformal_score)
and `run_baseline(seed) -> dict` (a classical solver — for FWI: a
gradient-descent inversion).

## Step 5 — write the LLM adapter

`seismic_fwi/adapter.py`:

```python
SYSTEM_PROMPT = """You are an expert at 1D seismic full-waveform
inversion. Given a synthetic seismogram, recover the layered velocity
model. ..."""

def build_user_prompt(instance) -> str:
    return f"Seismogram: {instance.y.tolist()} ..."

def parse_response(text: str, instance) -> Prediction:
    parsed = extract_json_block(text)
    # Validate, coerce types, return Prediction.
    ...
```

## Step 6 — run the validator

```bash
python scripts/validate_env.py environments/seismic-fwi
```

Output (filled-in env):

```
[1/4] tests pass: 6 passed in 0.42s
[2/4] empirical coverage on 50 seeds = 0.92 (target 0.90 ± 0.05) ✓
[3/4] EFFECTIVE_INSTANCES = 2.1e+18 > 1e15 ✓
[4/4] adapter round-trip on seed=0 ✓
```

If any check fails the validator exits non-zero with a specific
remediation. See [`scripts/validate_env.py`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/scripts/validate_env.py)
for the exact assertions.

## Step 7 — register in the platform

Two ways:

1. **Plugin** (zero-touch in the monorepo): the env's
   `pyproject.toml` already declares the entry point
   `[project.entry-points."verifiers.environments"]`. Anyone who
   `pip install`-s your env package can `load_environment("seismic-fwi")`.
2. **Monorepo merge** (for v0.1 envs): add the env id to
   `src/verifiable_labs_envs/__init__.py::ENV_MODULES` and submit a
   PR. The platform tests it on every CI run.

## What to put in your env's README

The scaffold's `README.md` template asks for:

- problem statement (what is the forward operator? what is observed?)
- ground-truth distribution (what does a typical instance look like?)
- baseline (what classical solver should the LLM beat?)
- known limitations (where the env's reward is brittle)

A concise, accurate README is what makes the env reusable. Resist
the temptation to overclaim difficulty or generality.

## See also

- [Concepts → Procedural regeneration](../concepts/procedural-regeneration.md) —
  why the validator enforces `EFFECTIVE_INSTANCES > 1e15`.
- [Concepts → Conformal rewards](../concepts/conformal-rewards.md) —
  what the calibration check is verifying.
- [`templates/inverse-problem/README.md`](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/templates/inverse-problem/README.md) —
  five concrete env ideas (holographic 3D, EM tomography, seismic
  FWI, inverse rendering, protein distogram).
