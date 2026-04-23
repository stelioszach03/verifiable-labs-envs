# Methodology — what the benchmark actually measures

Companion note to [`docs/conformal.md`](conformal.md) (the reward-term
math) and [`docs/CONTAMINATION.md`](CONTAMINATION.md) (the contamination-
resistance argument).

## 1. How scores are aggregated

Every result in `results/llm_benchmark.csv` is a single-call row with:

- `model` — the OpenRouter model id.
- `env` — one of `sparse-fourier-recovery`, `super-resolution-div2k-x4`, `lodopab-ct-simplified`.
- `seed` — the instance seed. Fresh noise and (for sparse Fourier) fresh support are drawn from this seed.
- `reward`, `components` — output of `env.score(prediction, instance)`.
- `failure_mode`, `failure_message` — populated when the call did not produce a scorable prediction.
- `latency_s`, `usd_cost`, `prompt_tokens`, `completion_tokens` — transport metadata.

Per-(model, env) aggregates average over all **successful** rows for that pair, irrespective of which sweep they came from. Rows with `failure_mode` set are excluded from the mean-reward numbers but are counted in the failure taxonomy (see §4).

## 2. Model-mean table and bar chart

The per-model mean reward across the three envs is computed by `analysis/correlation_analysis.py` and visualized in `results/model_mean_scores.png`.

As of the 2026-04-23 run (n=6 models with full coverage):

| Model | Mean reward |
|---|---:|
| Classical baseline (OMP / bicubic / FBP) | 0.737 |
| Claude Haiku 4.5 | 0.534 |
| Claude Opus 4.7 | 0.518 |
| Claude Sonnet 4.6 | 0.513 |
| GPT-5.4 | 0.494 |
| GPT-5.4 mini | 0.460 |
| GPT-5.4 nano | 0.358 |
| Zero baseline | 0.304 |

Two things are interesting and should be read carefully:

1. **Classical expert algorithms beat every LLM** by ~0.20 mean reward. This is the expected 2026 result for physics-grounded inverse problems — the envs are not yet saturated.
2. **Haiku 4.5 outranks its larger Anthropic siblings** on the 3-env mean. The gap (0.534 vs 0.518 vs 0.513) is within variance, but it persists across all three envs individually, which suggests it is not pure noise. The working hypothesis is that Haiku's JSON-output discipline is especially tight for this family of structured tasks.

## 3. Cross-env correlation

`analysis/correlation_analysis.py` computes pairwise Spearman rank correlations between envs across the 6 models. The result is a 3×3 symmetric matrix in `results/env_correlation_matrix.csv`, plotted in `results/env_correlation_heatmap.png`.

|            | SparseF | SuperRes | CT |
|---|---:|---:|---:|
| **SparseF**  | 1.00    | −0.37    | −0.26 |
| **SuperRes** | −0.37   | 1.00     | **+0.66** |
| **CT**       | −0.26   | +0.66    | 1.00 |

Reading these:

- **SuperRes ↔ CT (+0.66)** — same structural task shape (coarse-grid image output, server-side bicubic upsample), so models that handle one handle the other. A useful sanity check that the two image envs are not redundant — +0.66 is a moderate, not identical, correlation.
- **SparseF ↔ image envs (−0.26 to −0.37)** — negative, but small. The compact-JSON sparse-Fourier task rewards different failure modes than the long-JSON image tasks (support-picking vs 1024-cell grid transcription). We therefore interpret the three envs as measuring **different underlying capabilities**, which is the point of a battery.

**Honest caveat**: with n=6 observations (models), the nominal p-value for ρ = +0.66 is roughly 0.15, well outside any conventional significance threshold. These numbers are **descriptive** — "how do models rank-order across envs in this sample" — not statistical claims. With the Phase 6 v2 benchmark (larger model list) these estimates will tighten.

## 4. Failure taxonomy

`analysis/failure_taxonomy.py` labels each row of the CSV with exactly one category:

| Category | Definition |
|---|---|
| `api_failure` | Transport-level error (bad model id, rate limit, 5xx). |
| `parse_failure` | The response text was returned but JSON extraction / schema validation failed. |
| `support_error` | Sparse-Fourier only: `nmse_component ≥ 0.6` but `support_F1 ≤ 0.2`. The solver knows there's signal but can't locate it. |
| `magnitude_error` | Sparse-Fourier only: `support_F1 ≥ 0.6` but `nmse_component ≤ 0.3`. The solver locates support but gets amplitudes wrong. |
| `over_smoothing` | Image envs only: `ssim + 0.15 < psnr`. The point estimate has correct global pixel averages but has lost structure. |
| `ok` | None of the above; successful, in-distribution response. |

Distribution as of 2026-04-23 (see `results/failure_taxonomy.csv`):

- **api_failure** dominates only for bad model ids (`google/gemini-3.1-pro`, `deepseek/deepseek-chat-v3:free`, `meta-llama/llama-3.3-70b-instruct:free`) — 100% of their calls, 0% of the paid-models' calls.
- **parse_failure** is exclusively a small-model phenomenon: `gpt-5.4-nano` 67% on CT, `gpt-5.4-mini` 20% on CT and super-res, Haiku / Sonnet / Opus / GPT-5.4 at 0%.
- **support_error, magnitude_error, over_smoothing** fire 0 times in the v0.0.1 sample. None of the shipped models are good enough at sparse Fourier to cleanly hit either of the first two categories (both NMSE and support-F1 are low together), and the image envs currently show PSNR < SSIM across the board which keeps over_smoothing from triggering. These categories will become useful with the multi-turn and tool-use envs in Phase 3–4, where we expect richer intermediate behavior.

## 5. What this tells us about RL training priorities

Reading the numbers above through the lens of "which env does which RL post-training signal come from?":

- **Sparse-Fourier** is our strictest low-bandwidth reasoning task: tight input, tight output, unforgiving support metric. It currently rewards *picking the right 10 of 256* — a skill today's LLMs do not have out of the box and would have to learn via RLVR.
- **Super-resolution** is the most forgiving of the three because SSIM credits even crude denoising; it's the easiest env to get non-zero reward on, useful as a training-signal warm-up.
- **LoDoPaB-CT** lies between. Structured FBP-like priors help, but getting the transitions from bone to soft tissue right needs better uncertainty calibration — which is what the conformal reward term tracks.

The correlation table says these three signals are not redundant, so a post-training loop that incorporates all three gets meaningfully richer gradient information than a loop that only samples one.

## 6. Limitations acknowledged up front

- **n=6 models** is a small sample. Phase 6 will widen this.
- **Free-tier models dropped** after rate-limit / 404 issues in Sprint 0. They're in the CSV only as api_failure rows.
- **Haiku-beats-Opus** could be seed sensitivity; it needs to hold on the larger v2 sweep.
- **over_smoothing threshold** was set conservatively (`ssim + 0.15 < psnr`) and produced zero hits in v0.0.1 — may need recalibration once we have Phase 3–4 data.
- **Aggregation across Sprint 0 sweeps**: rows from Step 3 through Step 5b are mixed. They share the same environment code but are from different days and different benchmark-run contexts. This is acceptable for descriptive statistics; the Phase 6 v2 run will be single-session.
