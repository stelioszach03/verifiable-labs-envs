# Blockers & deferred work

Honest log of envs planned but not shipped in the sprint-giga round, with
rationale so the follow-up work can pick up cold.

## Deferred from sprint-giga (2026-04-24)

### Task 3 — Seismic full-waveform inversion

**Planned**: 2D acoustic FWI env with wave-equation forward operator and
travel-time tomography baseline. $15 B commercial TAM, no existing RL env
competitor.

**Deferred because**:

1. **Wave-equation solver install risk**. The three candidate libraries all
   have known issues on macOS arm64:
   - `jaxwave` — surface of JAX primitives not currently wrapped for
     variable-coefficient PDEs; would require custom assembly.
   - `devito` — builds OpenMP-parallelized C kernels at runtime; historically
     fragile on macOS ARM without a Homebrew gcc install and an
     `OMP_NUM_THREADS` override. A failed attempt in a scratch venv would
     burn an hour without guaranteed success.
   - From-scratch FD solver — works but needs careful stability analysis
     (CFL condition) and PML absorbing boundaries. ~500 lines of careful
     code to avoid numerical artifacts being mistaken for physics.
2. **Data footprint**. OpenFWI FlatVel-A is ~500 MB. Procedural regeneration
   from random-layer geology (2–5 layers, velocities 1500–4500 m/s) is the
   fallback but adds another 200 lines of generator code.
3. **Budget**. At the observed per-episode cost of $0.01–0.05 for vision/
   signal envs with 2 k tokens, seismic at 64×128 velocity grids would
   average $0.06–0.12 per episode before tool use — the $0.80 cap would be
   tight even for a 3×3×2 matrix.

**Follow-up plan**: implement a from-scratch 2D acoustic FD solver in
`src/verifiable_labs_envs/forward_ops/wave_equation.py` once we have a
dedicated morning (2–3 hour block). Start with 1D layered media + travel-
time baseline; expand to 2D only after the 1D version scores sensibly.

### Task 4 — Retrosynthesis (USPTO-50K)

**Planned**: given a product SMILES, propose reactant SMILES, with RDKit-
based validity / atom-balance / template-match verifier.

**Deferred because**:

1. **Different domain, higher verifier complexity**. All existing envs are
   real-valued-array inverse problems with continuous reward functions.
   Retrosynthesis is discrete (SMILES strings) with a rule-based verifier;
   the reward-component design (validity × atom-balance × template-match)
   needs a day of iteration to avoid reward hacking.
2. **RDKit + pytorch-geometric dep footprint** (~400 MB) is unfriendly to
   the monorepo's current tight-venv assumption. It would need an optional
   `[retrosynthesis]` extras group.
3. **Baseline is template-based** (e.g. LocalRetro) — another non-trivial
   install and reference-answer pipeline.

**Follow-up plan**: ship a minimal retrosynthesis env with just the
SMILES-validity + atom-balance verifier (skip template matching) as v0.1.
Use the HuggingFace `sagawa/ReactionT5v2-retrosynthesis-USPTO_50k` dataset
(CC0). Template-match verifier + proper reward design as v0.2 follow-up.

## Why ship partial now instead of stretching

The user's explicit constraint was "MAXIMUM execution, no mediocrity". Shipping
2 of 4 new env families at production quality (tests + benchmark + Hub push
+ CITATION + doc) beats shipping 4 mediocre envs that might have reward-
function bugs (like the `ista_tool` oracle artifact from Sprint 0/1). The
deferred work has clear follow-up plans that aren't blocked on anything the
current sprint changed.

## Cumulative state after sprint-giga

- **10 envs live on Prime Intellect Hub** (was 6).
  - New: `phase-retrieval`, `phase-retrieval-multiturn`,
    `mri-knee-reconstruction`, `mri-knee-reconstruction-multiturn`.
- **Pipeline infrastructure shipped**: `forward_ops/` package with
  `ForwardOperator` ABC + two new operators (`FFTMask2DOp`,
  `MagnitudeOnlyOp`) + `auto_calibrate()` utility.
- **Spend this sprint**: $0.13 LLM API out of the $4.00 cap.
- **Tests**: 254 green, 1 skipped (was 184 before the sprint).
- **Ruff**: clean.
