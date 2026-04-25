# Benchmarks

The platform ships canonical benchmark results in `results/`. This
page summarises what's there and how to read it.

## v0.1 paper-final benchmarks

Five models × ten environments × three seeds = ~150 episodes per
matrix. Two matrices: single-turn and multi-turn.

| CSV | rows | scope |
|---|---|---|
| `results/complete_matrix_single_turn.csv` | 90 | 5 models × 6 single-turn envs × 3 seeds |
| `results/complete_matrix_multi_turn.csv` | ~75 | 5 models × 3 multi-turn envs × 5 seeds |
| `results/llm_benchmark_v2.csv` | larger | the full v2 sweep used for the leaderboard |
| `results/llm_benchmark_tools_v2.csv` | smaller | tool-use envs, separate sweep |

All CSVs share the same schema (`env`, `model`, `seed`, `turn`,
`reward`, `components`, `parse_ok`, plus solver-side metadata like
`usd_cost` and `latency_s`).

## Headline findings

From the paper:

1. **Cross-env transfer is poor.** No single model is best on more
   than 3 / 10 envs. Sparse-Fourier and super-resolution scores do
   not predict CT or phase-retrieval scores.
2. **Calibration is universally weak.** Empirical conformal coverage
   sits in the 40–70 % band for most (model, env) pairs vs. the
   90 % target — models are over-confident across the board.
3. **Multi-turn helps unevenly.** Sparse-Fourier benefits clearly
   (+0.04 mean reward across models); CT and MRI are flat. The
   informative-feedback hypothesis is supported only for envs where
   the residual lives in a space the LLM can reason about (Fourier
   coefficients, not 64×64 image residuals).
4. **Tool use amplifies model differences.** Claude Haiku 4.5 used
   tools constructively (+0.05 on sparse-Fourier-tools); GPT-5.4-nano
   spammed without convergence.

## Regenerating

```bash
# Full re-run — ~$5 on OpenRouter, ~30 min wall time.
python benchmarks/run_v2_benchmark.py

# Single model, single env (for spot checks):
python benchmarks/run_llm_benchmark.py \
    --model anthropic/claude-haiku-4.5 \
    --env sparse-fourier-recovery \
    --seeds 0 1 2 3 4
```

The script writes incrementally to the CSV so a partial run still
produces usable data on KeyboardInterrupt.

## How to read a row

```csv
timestamp,env,model,seed,turn,reward,components,parse_ok,usd_cost,prompt_tokens,completion_tokens,latency_s,error,meta
2026-04-24T13:43:20.067252+00:00,sparse-fourier-recovery,openai/gpt-5.4-mini,2,1,0.3841,"nmse=0.135, support=0.200, conformal=0.900",True,0.00094875,905,60,1.603,,
```

- `env`, `model`, `seed`, `turn` — the (env, model, seed) tuple. For
  multi-turn envs, the same (env, seed, model) appears once per turn;
  the `--keep-only-final-turn` filter in `scripts/generate_report.py`
  collapses to the latest turn.
- `reward` — the bundled scalar in `[0, 1]`.
- `components` — comma-separated `key=value` pairs decomposing
  `reward`.
- `parse_ok` — whether the env's adapter could parse the model
  output. Parse failures count as `reward = 0`.
- `usd_cost` — OpenRouter's reported cost for this call.
- `error` — non-empty if the call itself raised (timeout, 5xx, etc.).

## Related results files

- `results/coverage_validation_n200.csv` — 200-seed conformal
  coverage validation; the regression test that catches reward-
  function drift.
- `results/env_correlation_matrix.csv` — pairwise correlation of
  per-(env, seed) rewards across models. Helps identify env clusters.
- `results/failure_taxonomy.csv` — classified failure modes (parse,
  off-support, miscalibration, etc.) with example prompts.
- `results/memorization_probe.csv` — the contamination-check audit
  trail; see [`docs/CONTAMINATION.md`](../CONTAMINATION.md).

## Live leaderboard

The hosted API exposes a leaderboard endpoint that reads from these
same CSVs:

```bash
curl https://api.verifiable-labs.com/v1/leaderboard?env_id=sparse-fourier-recovery | jq
```

The Hugging Face Space at
[verifiable-labs/leaderboard](https://huggingface.co/spaces/verifiable-labs/leaderboard)
is the public dashboard; same data, different surface.
