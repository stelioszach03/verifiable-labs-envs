# Verifiable Labs — inverse-problem env scaffold

Code-first template + scaffold scripts for adding a new scientific
inverse-problem RL environment to the Verifiable Labs catalogue. The
scaffold is the same one used to bootstrap every shipped env in
`src/verifiable_labs_envs/envs/`; the template files live in
`templates/inverse-problem/template/` and get rendered into the
target directory by `scripts/create_env.py`.

> **What this is not:** a multi-domain visual builder, an
> auto-generator from natural-language specs, or a UI. It's a
> deliberate code-first scaffold for researchers who already have a
> domain in mind and want to drop into the same patterns the existing
> ten envs follow. Visual / spec-driven builders are explicitly Tier-2.

## Quickstart — three commands

```bash
# 1. Scaffold a new env
python scripts/create_env.py seismic-fwi --domain "geophysics"

# 2. Edit the four NotImplementedError stubs:
#    environments/seismic_fwi/{forward_op,data,reward,env}.py
$EDITOR environments/seismic_fwi/

# 3. Validate
python scripts/validate_env.py environments/seismic_fwi
```

The validator runs four checks the brief specifies:

1. **All tests pass** — `pytest environments/seismic_fwi/tests`.
2. **Calibration coverage** within ±0.05 of the env's target — uses
   the env's own `run_baseline` over 50 fresh seeds and reads
   `meta["coverage"]`.
3. **Procedural-regeneration count** > 10¹⁵ unique measurement
   strings — the env exposes `EFFECTIVE_INSTANCES` in
   `<env>/__init__.py` and the validator just reads that constant
   (the user is expected to set `|seed_space| × |ground_truth_pool|`
   honestly there).
4. **Adapter compatibility** — `verifiers.load_environment(env_id)`
   resolves and `.generate_instance(0)` returns an `Instance`. Falls
   back to `verifiable_labs_envs.load_environment` if the env was
   added to the in-tree registry.

If any check fails, the validator exits non-zero and prints a clear
diagnosis pointing at the file/function to fix.

## What the scaffold gives you

```
environments/<env_py>/                  ← the rendered scaffold
├── pyproject.toml                       ← package metadata + verifiers entry-point
├── README.md
├── <env_py>/
│   ├── __init__.py                      ← ENV_ID, EFFECTIVE_INSTANCES, public re-exports
│   ├── env.py                           ← top-level <Env> class + load_environment
│   ├── forward_op.py                    ← TODO: your physics (forward, adjoint)
│   ├── data.py                          ← TODO: your ground-truth generator
│   ├── reward.py                        ← NMSE + conformal coverage default
│   └── adapter.py                       ← TODO: your LLM prompt / response shapes
└── tests/
    ├── test_env.py                      ← instance shape, determinism, scoring range
    ├── test_reward.py                   ← reward bounds, perfect-prediction = 1.0
    └── test_adapter.py                  ← adapter round-trip
```

Files marked **TODO** raise `NotImplementedError` at import-time
deliberately — until you replace them, `pytest` will report
*skipped* (not failing) so the scaffold can land in source control
without false-green coverage.

## Substitution map

`scripts/create_env.py` replaces these literal markers in every
template file (and renames `__ENV_PY__/` to your Python module name):

| marker | example | derivation |
|---|---|---|
| `__ENV_ID__` | `seismic-fwi` | the kebab-case slug you pass on CLI |
| `__ENV_PY__` | `seismic_fwi` | env id with hyphens → underscores |
| `__ENV_CLASS__` | `SeismicFwiEnv` | CamelCase class name |
| `__DOMAIN__` | `geophysics` | the human-readable `--domain` you pass |
| `__DOMAIN_TAG__` | `geophysics` | kebab-case domain (slugified) |

You can keep adding markers by editing the `_SUBSTITUTIONS` dict in
`scripts/create_env.py`.

## Five example domain ideas

These are deliberately scoped to fit the scaffold's pattern (real
forward operator, calibrated reward, contamination-resistant data
source). Pick one and run the scaffold — all five have been
sanity-checked against the existing envs for plumbing fit.

### 1. Holographic 3D reconstruction (`holography-volumetric`)

- **Forward:** sequential 2D Fresnel propagators stacked along z.
- **Ground truth:** 3D point-cloud / voxel grid of fluorescent
  sources, generated procedurally as random Poisson clusters.
- **Reward:** voxel-NMSE + per-source localisation error within a
  conformal interval.
- **Why now:** super-resolved fluorescence microscopy is hot in
  systems biology; no public RL env exists.

### 2. Electron-microscope tomography (`em-tomography-tilt-series`)

- **Forward:** Radon transform restricted to a missing-wedge
  acquisition (±60° rather than full 180°).
- **Ground truth:** synthetic biological volumes (membrane shells,
  ribosome-shaped lattices).
- **Reward:** NMSE + per-voxel uncertainty calibration; SSIM as a
  secondary domain metric.
- **Why now:** missing-wedge cryo-ET reconstruction is an active
  ML research area with no benchmarked RL signal.

### 3. Seismic FWI (`seismic-fwi-1d`)

- **Forward:** 1D acoustic wave equation with a sparse layered
  velocity model and a single source-receiver pair.
- **Ground truth:** procedurally generated layered velocity profiles
  (2-5 layers, 1500-4500 m/s).
- **Reward:** velocity-model NMSE + travel-time consistency +
  conformal coverage of layer boundaries.
- **Why now:** the geophysics community has open data (OpenFWI) but
  no calibrated RL benchmark; the 1D restriction sidesteps the
  devito-on-macOS install pain documented in `docs/BLOCKERS.md`.

### 4. Inverse rendering / radiance fields (`inverse-rendering-2d`)

- **Forward:** ray-traced rendering of a low-poly scene with known
  camera intrinsics.
- **Ground truth:** Procedurally generated 2D scenes (random
  rectangles + texture maps).
- **Reward:** per-pixel reconstruction NMSE + parameter recovery
  (camera pose, light position) under conformal calibration.
- **Why now:** every NeRF / Gaussian-splatting paper is an inverse
  problem in disguise; an RL env that scores reconstructions of
  known scenes is the missing benchmarking primitive.

### 5. Protein structure inference (`protein-residue-distogram`)

- **Forward:** residue-pair distance histogram from a known 3D fold.
- **Ground truth:** small protein fragments (≤30 residues) from PDB
  with synthetic permutations of side-chain rotamers.
- **Reward:** per-pair-bin NLL + secondary-structure recovery + a
  domain-specific Ramachandran-plausibility term.
- **Why now:** AlphaFold-style problems are the highest-value RLVR
  signal in computational biology and currently have no
  Verifiable-Labs-style env.

Each idea has been written so the four scaffold checks (tests +
calibration + 10¹⁵ regeneration + adapter compat) are achievable in a
focused weekend. Treat them as starter ideas; the framework doesn't
constrain you to compressed-sensing-shaped problems.

## Manual override hooks

- **Custom calibration**: replace `calibrate_quantile` in `env.py` if
  your domain's non-conformity score isn't `max |x - x_hat| / σ`
  (e.g. SSIM-derived for images, KL for probabilistic outputs).
- **Tighter `EFFECTIVE_INSTANCES`**: the default `2**64 *
  1024` assumes a 1024-element ground-truth pool. Set the second
  factor honestly — `validate_env.py` checks the literal value.
- **Multi-turn variant**: copy your env's `env.py` to
  `env_multiturn.py` and add a `run_rollout(solver, instance)` method
  that feeds residuals back between turns. See
  `verifiable_labs_envs.envs.sparse_fourier_multiturn` for the
  reference shape.

## Where to ship the env

After the validator passes:

1. Add the env to `src/verifiable_labs_envs/__init__.py:_REGISTRY`
   so `verifiers.load_environment` works in-tree.
2. Add an entry to `packages/` if you want a Hub-installable
   distribution.
3. Push to the Prime Intellect Hub (`prime env push` from
   `environments/<env_py>/`).
4. Update the Hosted Evaluation API's `registry.py` so
   `GET /v1/environments` advertises your new env.

The brief's quality bar — pass tests, calibrate, certify
contamination, register adapter — is exactly the four scaffold checks
the validator runs.
