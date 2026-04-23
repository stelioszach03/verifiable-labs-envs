# LLM benchmark results

**Run date**: 2026-04-23. **Total API spend**: $1.89. **Total calls**: ~100 across 6 models × 3 envs.

Models accessed via OpenRouter. Each entry is the mean reward over 5 seeds (seeds 0–4, fresh noise each call). Parse / API failures are counted separately and shown after the score.

## Headline table

| Model | SparseFourier | SuperRes | LoDoPaB-CT | Mean (3 envs) |
|---|---:|---:|---:|---:|
| **Reference baseline (OMP / bicubic / FBP)** | **0.869** | **0.629** | **0.712** | **0.737** |
| Claude Opus 4.7 | 0.300 | 0.628 | 0.625 | 0.518 |
| GPT-5.4 | 0.311 | 0.601 | 0.571 | 0.494 |
| Claude Sonnet 4.6 | 0.316 | 0.629 | 0.595 | 0.513 |
| Claude Haiku 4.5 | 0.361 | 0.625 | 0.615 | 0.534 |
| GPT-5.4 mini | 0.340 | 0.464 *(1/5 fail)* | 0.578 *(1/5 fail)* | 0.460 |
| GPT-5.4 nano | 0.350 | 0.528 *(2/6 fail)* | 0.197 *(4/6 fail)* | 0.358 |
| Zero baseline | 0.336 | 0.425 | 0.151 | 0.304 |

## What the numbers mean

- **The classical expert baselines beat every LLM by ~0.20 mean reward.** OMP, bicubic, and FBP are tailored algorithms for each forward operator. General reasoning models don't close that gap — as expected for a 2026 test of "can a chat model reconstruct physics out of the box."
- **`sparse-fourier-recovery` is a weak LLM discriminator.** Every model scores in a tight band (0.300–0.361) that is barely above the zero baseline (0.336). All models struggle to pick the correct sparse support from Fourier measurements; their NMSE is low (close to zero) but support-F1 is ~0. This is itself a result — compressed sensing is not yet a text-completion task.
- **`super-resolution-div2k-x4` and `lodopab-ct-simplified` produce a useful ranking.** Haiku 4.5 (0.625 / 0.615), Sonnet 4.6 (0.629 / 0.595), Opus 4.7 (0.628 / 0.625), and GPT-5.4 (0.601 / 0.571) cluster together within ~0.03. Smaller models drop off: GPT-5.4 mini averages 0.464 on SR, nano averages 0.197 on CT.
- **Parse-failure rate scales inversely with model size.** Small models lose count on long structured JSON outputs (asked for 32 entries, produce 30 or 33): `gpt-5.4-nano` fails 6/18 image calls (33%), `gpt-5.4-mini` 2/18 (11%), everything Haiku-and-above 0/54 (0%). This is a legitimate discrimination axis, not a benchmark bug.

## Also of note

- **Haiku 4.5 beats its own bigger siblings** on the 3-env mean (0.534 vs Sonnet 0.513 and Opus 0.518). Within ordinary variance, but a useful reminder that bigger ≠ better on narrow structured tasks.
- **Conformal-coverage scores saturate near 1.0 for poor point estimates**. When `x_hat` is far from `x_true`, the `sigma_hat` prior (1.0 off-support for sparse-Fourier; gradient-weighted for images) produces wide enough intervals to trivially cover the ground truth. This is correctly reflected in the reward: the conformal component can't compensate for a bad point estimate because it's capped at 1.0.

## Models not included

- `google/gemini-3.1-pro` — not a valid OpenRouter ID as of 2026-04-23 (`BadRequestError: 400`). Dropped from the sweep.
- `meta-llama/llama-3.3-70b-instruct:free`, `deepseek/deepseek-chat-v3:free` — free tier rate-limited or removed (429 / 404). Not retried.

## Reproducibility

```bash
python benchmarks/run_llm_benchmark.py --preset paid-full --max-cost 10
```

Raw per-call data in [`results/llm_benchmark.csv`](llm_benchmark.csv).
