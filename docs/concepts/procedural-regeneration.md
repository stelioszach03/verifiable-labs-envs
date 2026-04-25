# Procedural regeneration

Every (env, seed) pair produces a fresh problem instance from a
deterministic generator. The Verifiable Labs envs do not store any
test set — they store the **distribution** the test set is drawn from,
and re-generate on demand.

## Effective instance count

Each env reports an `effective_instance_count` in its metadata. This
is the **product** of:

- the seed space addressable by the platform (typically `2^64`)
- the per-env ground-truth pool size (e.g. for sparse-Fourier:
  `C(n=256, k=10) ≈ 2^57` choices of support × continuous amplitudes;
  reported as `~10^17` distinct floor-quantised instances)

For all v0.1 envs the effective count exceeds `1e15`. The
[`scripts/validate_env.py`](../tutorials/creating-custom-env.md)
validator enforces this floor for newly contributed envs.

## Why this matters for evaluation

The evaluator's nightmare is **dataset memorisation**: a model that's
seen the test set during training scores artificially high. Standard
benchmarks (MMLU, HumanEval, GSM-8K) live on the public web and have
been ingested by every frontier-model training run.

Procedural regeneration breaks the memorisation channel:

1. The seed pool the platform tests on (`60_000…` and beyond) was
   never published before the platform launched; the calibration pool
   (`0…499`) was published as code, not as instances.
2. Even if a seed value leaks, regenerating that seed on a model
   provider's training data still requires the env's source code to
   be present in the training corpus *and* the provider to have run
   the generator at the right git commit. The platform pins commits.
3. The contamination check at `docs/CONTAMINATION.md` documents the
   audit: we benchmarked frontier models and confirmed that none
   showed disproportionate gains on previously-published seeds vs.
   freshly-generated ones.

## Determinism

`env.generate_instance(seed=k)` is bit-exact: same seed, same env,
same git commit → same instance, every time. This is what makes the
benchmark *re-runnable*: a v0.1 score from 2026-04 is reproducible
in 2027.

The platform achieves this by:

- routing every random-number draw through `numpy.random.default_rng(seed)`
  (Philox 4×64; portable across platforms)
- avoiding system-time / system-random sources
- pinning numpy / scipy / jax versions in `pyproject.toml`
- testing equality on a fixed (env, seed) → reward fingerprint in
  `tests/test_seed_determinism.py`

## What the seed encodes

For most envs the seed feeds:

- a **support set** (which coordinates of `x` are non-zero)
- **amplitudes** for those coordinates (from a problem-specific prior)
- **noise** (Gaussian, scaled by the env's `sigma`)
- in some envs, **forward-operator parameters** (e.g. CT projection
  geometry, MRI mask pattern)

Per-env details live in the env-specific docs page. The point of
this page is to assert the property: every test problem is generated,
not recalled.

## When procedural regeneration is *not* enough

It still doesn't help if:

- the **forward operator** itself was the published artefact and is
  small enough to memorise (e.g. a fixed 32×32 sensing matrix)
- the **prior** the seed draws from is so narrow that all instances
  look alike

The v0.1 envs avoid both: forward operators are large enough to be
non-memorisable (e.g. partial Fourier on `n = 256`), and priors have
broad amplitude / support distributions. The
`docs/CONTAMINATION.md` audit confirms empirically.

## Audit hooks

Three hooks let third parties verify the regeneration claim:

1. `effective_instance_count` is a public field on every env's
   metadata, surfaced via the REST `/v1/environments` endpoint.
2. The seed → instance map is a single Python function; running the
   same seed in a fresh clone reproduces the instance bit-for-bit.
3. The `tests/test_seed_determinism.py` suite asserts the fingerprint
   on every CI run.
