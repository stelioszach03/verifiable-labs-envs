# verifiable-labs-envs

Reinforcement-learning environments for scientific reasoning — physics-grounded inverse problems with uncertainty-calibrated rewards.

> **Status (2026-04-24, post Sprint 1):** Six environments (3 single-turn + 2 multi-turn + 1 tool-use) live on Prime Intellect Environments Hub. Static leaderboard live on HuggingFace Spaces. 176 tests green, full suite under 2 s.
>
> - 🔗 Leaderboard: https://huggingface.co/spaces/stelioszach03/scientific-rl-benchmark
> - 🔗 Prime Intellect Hub envs: [`stelioszach/sparse-fourier-recovery`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery), [`-multiturn`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery-multiturn), [`-tools`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery-tools), [`super-resolution-div2k-x4`](https://app.primeintellect.ai/dashboard/environments/stelioszach/super-resolution-div2k-x4), [`lodopab-ct-simplified`](https://app.primeintellect.ai/dashboard/environments/stelioszach/lodopab-ct-simplified), [`-multiturn`](https://app.primeintellect.ai/dashboard/environments/stelioszach/lodopab-ct-simplified-multiturn).
> - 🔗 Full Sprint 1 summary: [`docs/SPRINT_1_COMPLETE.md`](docs/SPRINT_1_COMPLETE.md).

## What this is

Frontier reasoning models are trained with verifiable rewards (RLVR). Today's RL environments are mostly text-only, saturate quickly, and miss the continuous, ill-posed reasoning that real science requires. This package provides environments where:

1. The **forward operator** is exact and JIT-compiled (JAX), so a model must actually invert physics.
2. The **reward** is a weighted sum of reconstruction quality (PSNR, SSIM, or task-appropriate metric) and **conformal-prediction coverage** — models are rewarded for honest posterior width, not overconfident point estimates.
3. Measurements are **procedurally regenerated per evaluation call**, so fixed-string memorization is structurally impossible.

## Environments (v0.0.1)

| # | Environment | Status | Forward operator | Baseline |
|---|---|---|---|---|
| 1 | `sparse-fourier-recovery` | ✅ | subsampled orthonormal 1D DFT | OMP with LS-covariance σ̂ |
| 2 | `super-resolution-div2k-x4` | ✅ | Gaussian blur + 4× decimation | bicubic with edge-weighted σ̂ |
| 3 | `lodopab-ct-simplified` | ✅ | 2D parallel-beam Radon (60-angle) | FBP with edge-weighted σ̂ (phantom default; real-patient LoDoPaB-CT slices via `use_real_data=True`) |

## Classical-baseline benchmark (5 seeds each, default hyperparameters)

| environment | reference reward | zero reward | gap | conformal q |
|---|---:|---:|---:|---:|
| `lodopab-ct-simplified` | 0.712 | 0.151 | +0.561 | 0.241 |
| `sparse-fourier-recovery` | 0.869 | 0.336 | +0.533 | 1.587 |
| `super-resolution-div2k-x4` | 0.629 | 0.425 | +0.203 | 2.167 |

Reproduce with `python benchmarks/run_all.py --seeds 5`.

## Multi-turn rollouts (`sparse-fourier-recovery-multiturn`)

Ships a 3-turn conversation variant of `sparse-fourier-recovery`: turn 1 is the full problem, turns 2–3 show the Fourier-domain residual `r = y - A(x_hat)` of the previous answer and ask for a correction.

Async benchmark (3 models × 3 instances × 3 turns = 27 calls, $0.09 total, 33.6 s wall-clock with `Semaphore(10)`):

| Model | Turn 0 → Turn 1 → Turn 2 | Final | Episodes failed |
|---|---|---:|---|
| Claude Haiku 4.5 | 0.371 → 0.380 → 0.363 | 0.363 | 0/3 |
| Claude Sonnet 4.6 | 0.348 → 0.348 → 0.347 | 0.347 | 2/3 (turn-1 parse) |
| GPT-5.4 mini | 0.353 → 0.331 → 0.331 | 0.331 | 0/3 |

Headline finding: **frontier LLMs do not yet know how to use residual feedback constructively on sparse-Fourier recovery.** Scores plateau or regress at turns 2–3. This is itself the most actionable signal in the entire benchmark — it's exactly the surface RLVR post-training on these environments would be expected to improve.

Raw data: [`results/multiturn_sparse_fourier_recovery_multiturn.csv`](results/multiturn_sparse_fourier_recovery_multiturn.csv). Plot: [`results/multiturn_sparse_fourier_recovery_multiturn_curves.png`](results/multiturn_sparse_fourier_recovery_multiturn_curves.png).

## Tool-use rollouts (`sparse-fourier-recovery-tools`)

Same underlying problem as `sparse-fourier-recovery`, but the LLM is given **5 Python primitive tools** it must compose itself over ISTA-like iterations before committing to a final answer. No tool returns a full reconstruction on its own — the model has to iterate `forward → residual → adjoint → threshold` to converge.

- `fft_tool(signal_x1000)` → apply ``A = S·F`` to a length-n dense candidate.
- `ifft_tool(spectrum_re_x1000, spectrum_im_x1000)` → adjoint of A (zero-fill at mask + inverse DFT).
- `threshold_tool(signal_x1000, tau_x1000)` → elementwise soft-threshold (the ISTA proximal step).
- `compute_residual_tool(signal_x1000)` → returns `r = y − A(x)` + L2 / max-abs.
- `sparsity_norm_tool(signal_x1000)` → returns ‖x‖₁, ‖x‖₂, nonzero count.

Cap: 30 tool calls per episode (rebench used 5–15). Tools reference instance-bound state so call payloads stay small.

> **History — v0.1 was an oracle-delegation artifact.** The original
> tool-use env exposed an `ista_tool()` that returned the OMP oracle's
> answer. In the Task-4.1 benchmark all three tested models called it
> once and scored a byte-identical **0.858** per seed — the fingerprint
> of oracle adoption, not reasoning. v0.3 (2026-04-24 polish) removes
> `ista_tool` and replaces it with the five primitives above. A
> regression test (`test_no_single_tool_call_leaks_the_answer`) verifies
> no primitive transmits the target to the model.

v0.3 rebench (3 cheap models × 3 seeds, **$0.64 total** under $1 cap):

| Model | Mean reward (parsed) | Parse fails | Best episode |
|---|---:|---|---:|
| Claude Haiku 4.5 | 0.404 (n=1) | 1/2 seeds | 0.404 |
| GPT-5.4 mini | 0.403 (n=3) | 0/3 | 0.408 |
| GPT-5.4 nano | — | 3/3 | FAIL |

Empty-answer floor ≈ **0.354**, classical OMP baseline ≈ **0.931**.
All parsed rewards cluster just above the empty-answer floor — the
primitive tool set is genuinely hard, cheap LLMs cannot yet compose
ISTA from primitives. Tool sequences differ across models (no
byte-identical v0.1-style pattern). Full analysis:
[`results/sparse_fourier_reconciliation.md`](results/sparse_fourier_reconciliation.md)
("v0.3 follow-up"). Raw data:
[`results/llm_benchmark_tools_v2.csv`](results/llm_benchmark_tools_v2.csv). Reproduce:

```bash
python benchmarks/run_tools_v2_rebench.py \
  --models anthropic/claude-haiku-4.5,openai/gpt-5.4-mini,openai/gpt-5.4-nano \
  --n-instances 3 --max-tool-calls 5 --max-cost 0.30 --conformal-quantile 1.587
```

### Multi-turn CT (`lodopab-ct-simplified-multiturn`, phantom mode, 3 models × 3 instances × 3 turns, $0.56, 141 s)

Same 3-turn design: turn 1 takes a 32×32 FBP, turn 2–3 take the sinogram-residual back-projection (downsampled to 32×32, encoded as signed int8 + scale factor).

| Model | Turn 0 → Turn 1 → Turn 2 | Final mean | Episodes failed |
|---|---|---:|---|
| Claude Sonnet 4.6 | 0.618 → 0.645 → **0.657** | 0.657 | 1/3 (turn-1 parse) |
| GPT-5.4 mini | 0.622 → 0.642 → 0.641 | 0.622 | 1/3 (turn-2 parse) |
| Claude Haiku 4.5 | 0.626 → 0.488 → **0.344** | 0.550 | 2/3 (turn-2 parse) |

**Key finding (different from sparse-F!)**: Sonnet 4.6 improves **monotonically** across turns (+3.9 pp), GPT-5.4 mini plateaus after the first-turn bump, and Claude Haiku 4.5 **regresses severely** (−28.2 pp turn 2 → turn 3) — the residual-image feedback actively confuses it. Multi-turn rollouts surface a differential capability that single-turn scores completely mask.

Raw data: [`results/multiturn_lodopab_ct_simplified_multiturn.csv`](results/multiturn_lodopab_ct_simplified_multiturn.csv). Plot: [`results/multiturn_lodopab_ct_simplified_multiturn_curves.png`](results/multiturn_lodopab_ct_simplified_multiturn_curves.png).

Reproduce with:

```bash
python benchmarks/run_multiturn_benchmark.py \
  --env sparse-fourier-recovery-multiturn \
  --models anthropic/claude-haiku-4.5,anthropic/claude-sonnet-4.6,openai/gpt-5.4-mini \
  --n 3 --max-turns 3 --max-cost 2.0 --conformal-quantile 1.587
```

## Real-data CT (LoDoPaB-CT validation, opt-in via `use_real_data=True`)

Phase 2 adds a real-patient-geometry path on `lodopab-ct-simplified`: 3552 validation slices from the LoDoPaB-CT dataset (Leuschner et al. 2021, Nature Scientific Data) drawn from the LIDC-IDRI clinical chest-CT cohort. CI defaults stay on the phantom rotation so no download is required. One-shot activation:

```bash
bash scripts/download_lodopab_validation.sh      # ~1.5 GB zip, 28 HDF5 chunks
python -c "from verifiable_labs_envs.envs import lodopab_ct as ct; print(ct.load_environment(use_real_data=True).run_baseline(seed=0))"
```

Spot-check numbers (this repo, Apr 2026):

| Solver | Mode | Mean reward | Notes |
|---|---|---:|---|
| Classical FBP | phantom (5 seeds) | 0.712 | Sprint 0 baseline |
| Classical FBP | **real (10 seeds)** | **0.731** | mean PSNR 0.62, SSIM 0.64 — real CT is structurally cleaner than the synthetic phantoms |
| Claude Haiku 4.5 | phantom (5 seeds) | 0.615 | Sprint 0 0/5 parse-fail |
| Claude Haiku 4.5 | **real (3 seeds)** | 0.694 on 1/3 success | 2/3 parse-fails — "expected 32 entries, got 31" on seeds 0 and 2. Real CT grids are harder for the model to transcribe without losing count than the phantom pattern. |

Raw data: [`results/ct_real_spotcheck.csv`](results/ct_real_spotcheck.csv).

## v2 benchmark — 4 models × 6 envs (Sprint 1)

Full 6-environment sweep including multi-turn and tool-use variants. Opus 4.7 dropped from this sweep because Sonnet ≈ Opus within noise in Sprint 0 and keeping it would have blown the $3 cap.

| model | SparseF | SparseF-MT | SparseF-Tools | SuperRes | CT | CT-MT | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-4.5 | 0.364 | 0.351 | 0.334 | 0.726 | 0.640 | 0.527 | **0.490** |
| claude-sonnet-4.6 | 0.305 | 0.328 | 0.337 | 0.726 | 0.580 | 0.640 | **0.486** |
| gpt-5.4 | 0.293 | 0.365 | 0.306 | 0.721 | 0.601 | 0.654 | **0.490** |
| gpt-5.4-mini | 0.338 | 0.363 | 0.354 | 0.534 | 0.505 | 0.371 | **0.411** |
| **env mean** | 0.325 | 0.352 | 0.333 | 0.677 | 0.581 | 0.548 | |

Three new findings the v2 sweep surfaces:

- **Multi-turn helps *frontier* models on CT, hurts *small* models** — GPT-5.4 CT 0.60 → CT-MT 0.65, Sonnet 0.58 → 0.64; Haiku CT 0.64 → CT-MT 0.53, mini 0.51 → 0.37. Budget models can't maintain coherence across the residual-feedback protocol; frontier models use the extra turns productively.
- **Sparse-Fourier stays flat across single-turn / multi-turn / tool-use** (all 0.29–0.37). No rollout format unlocks compressed sensing for any tested model. The `SparseF-Tools` column in the v2 table above was a v0.1 run where the tool-use env still shipped the `ista_tool` oracle; after the v0.3 rebench with primitive-only tools (see the tool-use section above), cheap LLMs cluster right at the empty-answer floor — reinforcing this finding, not contradicting it.
- **SuperRes saturates for the Claude-Sonnet / Claude-Haiku / GPT-5.4 cluster** at ~0.72–0.73, with GPT-5.4-mini trailing at 0.53. Compression-style image denoising is the easiest task in the battery; all frontier models converge.

Heatmap: [`results/benchmark_v2_heatmap.png`](results/benchmark_v2_heatmap.png). Raw data: [`results/llm_benchmark_v2.csv`](results/llm_benchmark_v2.csv). Full summary with caveats: [`results/benchmark_v2_summary.md`](results/benchmark_v2_summary.md).

## LLM benchmark v1 (OpenRouter, 5 seeds each, total spend $1.89)

| Model | SparseFourier | SuperRes | LoDoPaB-CT | Mean (3 envs) |
|---|---:|---:|---:|---:|
| **Reference baseline (OMP / bicubic / FBP)** | **0.869** | **0.629** | **0.712** | **0.737** |
| Claude Opus 4.7 | 0.300 | 0.628 | 0.625 | 0.518 |
| Claude Sonnet 4.6 | 0.316 | 0.629 | 0.595 | 0.513 |
| **Claude Haiku 4.5** | 0.361 | 0.625 | 0.615 | **0.534** |
| GPT-5.4 | 0.311 | 0.601 | 0.571 | 0.494 |
| GPT-5.4 mini | 0.340 | 0.464 *(1/5 fail)* | 0.578 *(1/5 fail)* | 0.460 |
| GPT-5.4 nano | 0.350 | 0.528 *(2/6 fail)* | 0.197 *(4/6 fail)* | 0.358 |
| Zero baseline | 0.336 | 0.425 | 0.151 | 0.304 |

Clean discrimination across model tiers and clean rank-ordering against the expert classical baselines. The environments measure capability, not chance:

- **Classical expert algorithms (mean 0.737) beat every general-purpose LLM** on these inverse problems.
- **Sparse-Fourier is a weak LLM discriminator** (all models 0.30–0.36, barely above zero baseline 0.336) — compressed sensing is not yet a text-completion task.
- **Super-resolution and CT produce a useful ranking** (Haiku / Sonnet / Opus / GPT-5.4 cluster at ~0.60, small models drop off).
- **JSON-count parse-failure rate scales inversely with model size**: `gpt-5.4-nano` fails 33% of grid outputs, `gpt-5.4-mini` 11%, everything Haiku-and-above 0% — a legitimate discrimination axis on its own.
- **Cross-env correlation matrix** (Spearman, n=6 models): SuperRes ↔ CT = +0.66 (same structural task); SparseF ↔ image envs = −0.26 to −0.37 (different capabilities). The three envs measure different things. Full methodology in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md); heatmap in [`results/env_correlation_heatmap.png`](results/env_correlation_heatmap.png).

![Cross-env Spearman correlation heatmap](results/env_correlation_heatmap.png)

Reproduce with `python benchmarks/run_llm_benchmark.py --preset paid-full`. See [`results/llm_benchmark.md`](results/llm_benchmark.md) for the full analysis and [`results/llm_benchmark.csv`](results/llm_benchmark.csv) for per-call raw data.

## Install

### Full monorepo (developers + research use)

```bash
git clone https://github.com/stelioszach03/verifiable-labs-envs
cd verifiable-labs-envs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                  # 176+ tests green
```

### Single environment via Prime Intellect Hub (now live)

All six envs are published on the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments):

```bash
pip install prime
prime login
prime env install stelioszach/sparse-fourier-recovery
# or any of: sparse-fourier-recovery-multiturn, sparse-fourier-recovery-tools,
#            super-resolution-div2k-x4, lodopab-ct-simplified, lodopab-ct-simplified-multiturn
```

### Single environment via GitHub subdirectory

```bash
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-sparse-fourier"
# or: verifiable-labs-sparse-fourier-multiturn, -tools, super-resolution,
#     lodopab-ct, lodopab-ct-multiturn, envs-core
```

Then:

```python
from verifiable_labs_sparse_fourier import load_environment
env = load_environment()
out = env.run_baseline(seed=0)
```

## Quickstart

```python
from verifiable_labs_envs import load_environment

env = load_environment("sparse-fourier-recovery")
result = env.run_baseline(seed=0)
print(result["reward"])            # e.g. 0.931
print(result["components"])        # {"nmse": 0.977, "support": 0.900, "conformal": 0.900}
print(result["meta"]["coverage"])  # 0.80 — fraction of support entries inside the conformal interval
```

Any custom solver can be scored by returning a `Prediction(x_hat, sigma_hat, support_hat=...)`
and passing it to `env.score(prediction, instance)`.

Walkthrough across all three environments:

```bash
python examples/quickstart.py
```

## Contamination resistance

Every environment in this repo is structurally resistant to the three attacks that have hollowed out static text benchmarks: train-set leakage, answer-string matching, and distribution creep. Full analysis in [`docs/CONTAMINATION.md`](docs/CONTAMINATION.md). Headline numbers:

- `sparse-fourier-recovery` — the per-instance state space is continuous (10 real-valued amplitudes + 128 real-valued complex-noise coordinates), on top of `C(256, 10) × C(256, 64) ≈ 10⁷³` combinatorial arrangements of support and mask.
- `super-resolution-div2k-x4` and `lodopab-ct-simplified` — the discrete image / phantom set is small (6 and 5 respectively, a known v0.0.1 weakness flagged in the doc), but measurement noise is regenerated per call so memorizing the HR image doesn't reproduce the measurement.
- An empirical memorization probe at [`scripts/memorization_probe.py`](scripts/memorization_probe.py) confirms: across Haiku 4.5, GPT-5.4 mini, and GPT-5.4 nano on `sparse-fourier-recovery`, all three models show cross-seed reward std ≥ 0.02 (no constant-output signatures). Raw data: [`results/memorization_probe.csv`](results/memorization_probe.csv).

## Documentation

- [`docs/conformal.md`](docs/conformal.md) — the conformal-coverage reward term: why it's there, how it's calibrated, what it rewards.
- [`docs/CONTAMINATION.md`](docs/CONTAMINATION.md) — contamination resistance analysis, per-env effective instance count, empirical probe methodology.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — benchmark aggregation, cross-env correlation interpretation, failure taxonomy.
- [`docs/PRIME_INTELLECT.md`](docs/PRIME_INTELLECT.md) — package split + Hub publish instructions.
- [`docs/LEADERBOARD.md`](docs/LEADERBOARD.md) — Gradio leaderboard + HF Spaces deploy.
- [`docs/env1_sparse_fourier_design.md`](docs/env1_sparse_fourier_design.md) — Env 1 architecture and reward specification.

## Leaderboard

Static [Gradio leaderboard](leaderboard/app.py) backed by the v2 benchmark CSV — three tabs (Overview, Methodology, Submit). Run locally:

```bash
cd leaderboard && pip install -r requirements.txt && python app.py
```

HF Spaces deploy pending `HF_TOKEN` setup; see [`docs/LEADERBOARD.md`](docs/LEADERBOARD.md) for the exact deploy command.

## Author

Stelios Zacharioudakis — finishing BSc CS at the University of Athens (NKUA). Research on calibrated astronomical inverse imaging.

## License

Apache 2.0. See `LICENSE`.
