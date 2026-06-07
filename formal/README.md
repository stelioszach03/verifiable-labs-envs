# `formal/` — machine-verified Lean 4 proofs

This directory contains a standalone Lake project with the formal proofs behind Verifiable Labs's
reward-calibration stack. **The mathematics here is machine-verified.** The Python implementation in
`src/verifiable_labs_envs/formal_spec/` is property-tested against this specification — it is itself
**not** formally verified. The honest claim is described in the project root `README.md` under
"Formally verified guarantees".

All proofs in this directory are **`sorry`-free** and depend only on the three standard Lean
axioms: `propext`, `Classical.choice`, `Quot.sound`. You can verify any individual theorem with
`#print axioms <theorem_name>` inside a Lean session.

## Module map

| File | Headline theorems | Plain-English meaning |
|---|---|---|
| `VerifiableLabsFormal/CalibratedReward.lean` | `calibratedReward_bounded`, `calibratedReward_mono_V`, `calibratedReward_mono_C`, `calibratedReward_anti_H`, `calibratedReward_strict_anti_H` | The calibrated reward `R* = V·C − λ·H` lies in `[−λ, 1]`, increases in the value `V` and confidence `C`, and decreases in hackability `H` (strictly when `λ > 0`). |
| `VerifiableLabsFormal/VGS.lean` | `VGS_bounded`, `VGS_mono_{G,C,R,D}`, `VGS_anti_{H,K,L}`, `VGS_strict_mono_G` | The Verifiable Generalization Score `VGS = G·C·R·D − λH − μK − νL` is bounded in `[−(λ+μ+ν), 1]`, increases in every quality term, and decreases in every penalty term. |
| `VerifiableLabsFormal/AdaptiveDifficulty.lean` | `fixedPoint_iff_solve_rate_eq`, `exists_fixedPoint`, `stability_nonexpansive`, `stability_strict` | The difficulty update `d' = d + η(s − s*)` has a fixed point exactly when `s(d) = s*`; under antitone, `L`-Lipschitz solve-rate with `η·L < 1` the iteration is non-expansive around the fixed point, and strictly contracting away from it. |
| `VerifiableLabsFormal/VerifierInvariance.lean` | `invariant_preserves_correct`, `shortcut_violates_invariance`, `invariantSubgroup`, `invariant_of_generators` | Invariance of a verifier under a transformation pair `(T_X, T_A)` preserves correctness; a verifier that flips under an invariance is a shortcut; the set of invariant transformations forms a subgroup. |
| `VerifiableLabsFormal/ConformalCoverage.lean` ⭐ | `split_conformal_coverage`, `split_conformal_reward_coverage` | The split-conformal calibration set produces a residual interval that contains the true reward with probability at least `1 − α`, proved via order statistics and a leave-one-out exchangeability argument. *This is the proof anchoring our public reward-interval guarantee.* |
| `VerifiableLabsFormal/ModelRouting.lean` | `selected_model_optimal`, `cheaper_model_preferred`, `near_optimal_under_error` | The argmax of the utility `U = Q − γ·Cost − δ·Latency − ρ·Risk` is optimal; under `ε`-bounded utility estimation error the routed model is `2ε`-near-optimal. |
| `VerifiableLabsFormal/VerifiablePipeline.lean` | `pipeline_reward_bounded`, `pipeline_conformal_coverage`, `pipeline_difficulty_stable`, `pipeline_routing_near_optimal`, `pipeline_generalization_strict_mono`, 4 others | Composition theorem: the bundled pipeline output preserves every guarantee proved in the six modules above. |
| `VerifiableLabsFormal/SelfImprovementGate.lean` | `AcceptUpdate` definition, `accepted_sequence_mono_VGS`, `accepted_sequence_VGS_lower_bound` | A 7-condition checkpoint-acceptance predicate; any accepted sequence has VGS monotone non-decreasing, with `VGS_n ≥ VGS_0 + n·τ`. |
| `VerifiableLabsFormal/Main.lean` | — (no theorems) | Top-level option/scope file — heartbeat limits, `BigOperators` / `Classical` open, pretty-printer settings. Lake builds this to validate the configuration. |

## Provenance

Proofs authored and discharged by **[Aristotle](https://aristotle.harmonic.fun)** (Harmonic AI's
interactive theorem-proving system). Export checked into this repo on **2026-05-21**. Aristotle's
original Lake package was named `RequestProject`; the package and directory were renamed to
`VerifiableLabsFormal` on import. Proof content (the `.lean` files) is otherwise byte-identical to
the export — the only mechanical edits were the six `import RequestProject.X` →
`import VerifiableLabsFormal.X` lines inside `VerifiablePipeline.lean`. See `ARISTOTLE_SUMMARY.md`
for Aristotle's run-by-run authoring notes.

To credit Aristotle on PRs touching `formal/`, tag `@Aristotle-Harmonic` in the PR description.

## Toolchain pin

The Lean toolchain and Mathlib revision are pinned in `lean-toolchain` and `lake-manifest.json`
respectively:

| Component | Version |
|---|---|
| Lean | `leanprover/lean4:v4.28.0` |
| Mathlib | `v4.28.0`, git rev `8f9d9cff6bd728b17a24e163c9402775d9e6a365` |

**Do not bump versions without re-verifying every proof.** Mathlib bumps routinely refactor
identifiers used in proof scripts; a single rename inside `Mathlib.Probability` can break
`split_conformal_coverage` overnight. The CI workflow `formal-verification.yml` enforces a green
`lake build` on every push that touches `formal/**`.

## Local verification

```bash
# One-time: install elan (Lean version manager)
curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none
source ~/.elan/env

# Build (first run downloads ~1 GB of Mathlib oleans via the official cache)
cd formal/
lake exe cache get        # mandatory — without this, the build takes ~1 h instead of ~5 min
lake build

# Sorry-free check
! grep -rn '\bsorry\b' . --include='*.lean'

# Inspect axioms used by any theorem
lake env lean --run <(cat <<'EOF'
import VerifiableLabsFormal.ConformalCoverage
#print axioms split_conformal_coverage
EOF
)
# expected output: `split_conformal_coverage` depends on axioms: [propext, Classical.choice, Quot.sound]
```

## What this directory is and isn't

|   |   |
|---|---|
| ✅ Is | A machine-verified mathematical specification of the public guarantees Verifiable Labs claims (calibrated-reward bounds, conformal coverage, gate monotonicity, etc.). |
| ✅ Is | The source of truth that the Python module `src/verifiable_labs_envs/formal_spec/` mirrors and property-tests against. |
| ❌ Isn't | A proof of correctness of the Python code or the hosted API. The implementation is property-tested for parity with this spec; the implementation itself is not formally verified. |
| ❌ Isn't | A licence to write the phrases *"formally verified code"*, *"formally verified system"*, or *"formally verified API"* in any other documentation. Those claims are prohibited. The only approved wording is in the project `README.md`. |
