# Contamination resistance

**Status (2026-04-23):** this document covers the three shipped environments:
`sparse-fourier-recovery`, `super-resolution-div2k-x4`, `lodopab-ct-simplified`.
Multi-turn, tool-use, and real-data CT environments inherit the same
contamination-resistance properties by construction; they will be covered as
they ship.

## 1. Why contamination is the core problem for text benchmarks

"Contamination" is the catch-all name for any way an evaluation score can be
gamed without solving the problem the score claims to measure. The three
attack surfaces that have hollowed out static text benchmarks during 2024–2025
are:

1. **Train-set leakage.** The test instances leak into pre-training or SFT
   data. The model has seen the exact question before and recalls the answer.
   Well-documented for MATH, HumanEval, GPQA, and MMLU across multiple
   frontier models.
2. **Answer-string matching.** Reward functions compare a stringified model
   output to a ground-truth string. Small paraphrases, Unicode spoofing, or
   deterministic prompt-engineering hacks let a model clear the bar without
   solving the underlying task. The 2025 RLVR-reward-hacking literature
   explicitly calls this out — see arXiv:2509.21882 and arXiv:2510.00915,
   both of which argue that scalar-correctness rewards without uncertainty
   calibration are gameable in exactly this shape.
3. **Distribution creep.** Once a benchmark is popular, model providers
   train on problems that "look like" it (same domain, same prose style).
   The score goes up even though test-set leakage in the strict sense
   hasn't happened. Every saturation plot on Papers With Code from 2022–2025
   shows this drift.

A commercial RL-training environment has to defeat all three. Otherwise the
RLVR pipeline that consumes it trains a model that appears to improve on the
target capability but has actually just learned a prompt-level shortcut.

## 2. How this repo structurally defeats each attack

| Attack | Text-benchmark vulnerability | This repo's defense |
|---|---|---|
| Train-set leakage | Fixed `(question, answer)` pairs can be scraped or memorized. | **Procedural measurement regeneration.** Every call to `generate_instance(seed)` draws a fresh ground truth `x*` and a fresh noise vector `η`. No `(y, x*)` tuple exists as a durable artifact. |
| Answer-string matching | Reward is `answer == expected_string`. | **Continuous, physics-grounded rewards.** PSNR / SSIM / NMSE / support-F1 / conformal coverage — all numeric, all computed from the ground-truth array `x*` and the solver's `x_hat`. No string comparison anywhere in the scoring pipeline. |
| Distribution creep | "Problems like MATH" can be generated and trained on. | **Exact forward operators.** To solve a new instance the model must invert the operator `A` on the specific `y` for that seed. Training-time paraphrases don't help because there is no prose surface; the input is a vector of integers. |

Concretely, the procedural-regeneration property is enforced by these lines:

- `src/verifiable_labs_envs/envs/sparse_fourier.py:generate_instance` — fresh
  support `{i_1, ..., i_k}`, fresh amplitudes `~ N(0, 1)`, fresh mask
  `~ Uniform(C(n, m))`, fresh complex noise `~ CN(0, σ²)` per seed.
- `src/verifiable_labs_envs/envs/super_resolution.py:generate_instance` —
  image is deterministic given the seed-selected slot in
  `CALIBRATION_IMAGES`, but the additive measurement noise is regenerated
  from the seed on every call.
- `src/verifiable_labs_envs/envs/lodopab_ct.py:generate_instance` — same
  pattern: deterministic phantom + fresh sinogram noise per seed.

## 3. Effective instance count per environment

The relevant question is not "how many distinct ground truths exist?" but
"how many distinct `(y, x*)` tuples exist?" because that is what a
hypothetical memorizing adversary would need to store.

### `sparse-fourier-recovery`

Defaults: `n=256`, `k=10`, `m=64`, `σ=0.05`.

- Support positions: `C(256, 10) ≈ 2.7 × 10¹⁵` distinct index sets.
- Non-zero amplitudes: 10 independent draws from a continuous density
  `N(0, 1)` → uncountable.
- Mask positions: `C(256, 64) ≈ 4.0 × 10⁵⁷` distinct mask sets.
- Measurement noise: 64 complex draws from `CN(0, σ²)` → uncountable
  (128 independent real coordinates per instance).

Effective instance space: continuous and unbounded along the amplitude and
noise axes; combinatorially large along the support and mask axes. A
memorizing adversary would need at minimum a look-up table of order
`2.7 × 10¹⁵ × 4 × 10⁵⁷ ≈ 10⁷³` entries before the real-valued axes are even
counted.

### `super-resolution-div2k-x4`

Defaults: `shape=(128, 128)`, `factor=4`, `noise_sigma=0.01`, 6 rotation
images (`camera`, `moon`, `astronaut`, `coffee`, `chelsea`,
`immunohistochemistry`).

- HR images: 6 fixed sources in v0.0.1 — a real weakness.
- Measurement noise: 1024 real draws from `N(0, σ²)` per instance,
  resampled per seed.

The image-count weakness is acknowledged. The mitigation is twofold:
(a) the measurement `y` differs per seed even for the same image, so
pattern-matching the HR image and returning it verbatim does not produce
a reconstruction that beats bicubic — the solver still has to denoise and
de-blur. (b) The v0.1 roadmap adds the DIV2K validation set (800 images)
as the default rotation; the current 6-image rotation is a CI-safety
compromise that stays fully reproducible without a ~5 GB download.

### `lodopab-ct-simplified`

Defaults: `shape=(128, 128)`, `n_angles=60`, `noise_sigma=0.5`, 5 rotation
phantoms (Shepp-Logan + 4 skimage images acting as attenuation stand-ins,
each circularly masked).

- Phantom sources: 5 fixed in v0.0.1.
- Sinogram noise: 60 angles × 128 detector bins = 7680 real draws from
  `N(0, σ²)` per instance, resampled per seed.

Same weakness as super-res on the phantom side, same mitigation via noise.
Phase 2 of the Sprint 1 plan replaces this with 100 real LoDoPaB-CT
validation slices (LIDC/IDRI lung CT), raising the discrete count
by 20× and adding real patient geometry.

## 4. Empirical probe (see `scripts/memorization_probe.py`)

Abstract arguments are worth less than measurement. The memorization probe
runs each environment's reference baseline + a target LLM under two
conditions:

**Condition A — pipeline determinism.** Calls `env.generate_instance(seed=42)`
twice and verifies the two scores are bit-identical. Any difference flags a
non-deterministic pipeline (bad for reproducibility but, perversely, evidence
against memorization since the memorizer would have to reconcile the drift).

**Condition B — cross-seed variance.** Runs the solver on 10 distinct random
seeds. Expected `std(reward) > 0.01`. A suspiciously low variance (near zero,
far from the typical reference-baseline spread) would indicate that the
solver's output does not depend on the concrete `y` — i.e. either a trivial
constant predictor or a memorized average. The probe flags variance below
0.01 as a risk signal and logs it to `results/memorization_probe.csv`.

The probe is **not** a cryptographic guarantee. It is a sanity check that a
shipped evaluation actually responds to its inputs. The structural properties
in §2 do the real work; §4 is the evidence.

## 5. Honest limitations

- The 6-image and 5-phantom rotations in super-res and CT are small discrete
  sets. An adversary who knew the source images could, in principle, train
  a per-image de-blurring network offline, and that network would do well
  on this benchmark without doing anything you could call reasoning. Phase 2
  (real LoDoPaB-CT) and a post-sprint DIV2K upgrade fix this for both
  image envs; `sparse-fourier-recovery` is immune because it has no discrete
  image source.
- Procedural regeneration protects against *byte-level* memorization of
  fixed `(y, x*)` pairs. It does **not** protect against a model that has
  internalized the physics of the forward operator and simply inverts it
  competently. That is exactly what we want: high rewards on this benchmark
  mean the model has learned to solve inverse problems, which is the
  capability we claim to measure.
- The conformal-calibration set is reused across models (so q_α is the same
  for all), but the calibration set itself is generated from fresh seeds
  distinct from evaluation seeds. See `docs/conformal.md` for details.

## 6. References

- `arXiv:2509.21882` — "Reward hacking in RLVR: miscalibration, sycophancy,
  and verifier gameability" (2025).
- `arXiv:2510.00915` — "Calibrated rewards for reinforcement learning from
  verifiable feedback" (2025).
- Lei et al. 2018 — "Distribution-free predictive inference for regression,"
  *JASA*.

See also `docs/conformal.md` for the companion write-up on how the
uncertainty-calibrated reward term is computed and why it additionally
penalizes a specific family of over-confident hacks.
