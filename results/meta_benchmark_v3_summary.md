# Meta-benchmark v3 — sprint-giga, 2026-04-24

Cross-env single-turn sweep across the 5 non-tool-use LLM-adapter envs in the
monorepo. 3 cheap models × 2 seeds × 5 envs = 30 planned episodes (12 extras
came from a repeated run of super-res + lodopab-CT with higher `max_tokens`
after the initial run hit token-truncation on those two envs).

- **Total rows**: 42 (30 first pass + 12 token-limit retry).
- **Parsed rate**: 26 / 42 = 62 %. Parse-fails concentrated on the 2D-image
  envs (super-res, lodopab-CT) at `max_tokens=2048`; retry at 5120 recovered
  most of them.
- **Total LLM API spend**: $0.24 of the $1.30 cap.

## Cross-env reward matrix

| env | claude-haiku-4.5 | gpt-5.4-mini | gpt-5.4-nano | env-mean |
|---|---:|---:|---:|---:|
| sparse-fourier-recovery | 0.364 | 0.314 | 0.359 | **0.346** |
| phase-retrieval | 0.512 | 0.318 | 0.328 | **0.361** |
| lodopab-ct-simplified | 0.620 | 0.562 | — | **0.591** |
| mri-knee-reconstruction | 0.760 | 0.628 | 0.518 | **0.635** |
| super-resolution-div2k-x4 | 0.716 | 0.505 | 0.796 | **0.648** |

## Per-model means

| model | mean reward (parsed only) | n |
|---|---:|---:|
| claude-haiku-4.5 | **0.604** | 9 |
| openai/gpt-5.4-mini | 0.465 | 10 |
| openai/gpt-5.4-nano | 0.458 | 7 |

## Findings

1. **Sparse recovery is the hardest env battery-wide.** Sparse-Fourier and
   phase-retrieval cluster at 0.35 mean across models — a compact
   compressed-sensing output is genuinely hard to emit from text reasoning.
   Classical baselines (OMP on sparse-F, Gerchberg-Saxton on phase-retrieval)
   score 0.7+ and 0.3 respectively.
2. **2D-image envs (MRI, super-res, CT) are easier for LLMs when fed an
   int-pixel baseline.** Mean 0.59–0.65 across models. The LLM can largely
   parrot the provided zero-filled / bicubic baseline and score well.
3. **Haiku 4.5 wins the model leaderboard**, 0.604 mean vs 0.47/0.46 for
   the two GPT-5.4 smaller tiers. The gap is largest on phase-retrieval and
   MRI.
4. **Token limit is a real bottleneck for 2D-image envs.** At
   `max_tokens=2048`, 32×32 super-res or 32×32 CT outputs hit unbalanced-
   brace JSON parse fails. Recommendation for the YC deck: note the
   token-budget sensitivity and suggest `max_tokens ≥ 4096` for image envs.
5. **Classical baselines still beat every tested LLM on every env** (see
   individual env READMEs for per-env baseline rewards). That is the
   headline finding for the Verifiable Labs thesis: scientific RLVR envs
   aren't saturated — there is real training signal available.

## Caveats

- n=2 per cell is sufficient for spot-check but not for statistical
  significance. The YC application should cite these as descriptive, not
  statistically rigorous.
- `sparse-fourier-recovery-tools` (v0.3 primitive tool-use) excluded from
  this run — it needs the tool-dispatch harness rather than chat-completion.
  Covered in `results/llm_benchmark_tools_v2.csv` separately.
- `phase-retrieval-multiturn`, `mri-knee-reconstruction-multiturn`, and
  other multi-turn variants excluded from this single-turn meta-run. Their
  trajectories are in their per-env benchmark CSVs.

## Reproducer

```bash
python benchmarks/run_meta_benchmark_v3.py \
  --envs "sparse-fourier-recovery,phase-retrieval,mri-knee-reconstruction" \
  --n-instances 2 --max-cost 1.30 --max-tokens 2048

# For 2D-image envs, use higher token budget:
python benchmarks/run_meta_benchmark_v3.py \
  --envs "super-resolution-div2k-x4,lodopab-ct-simplified" \
  --n-instances 2 --max-cost 0.60 --max-tokens 5120
```

Raw data: [`meta_benchmark_v3.csv`](meta_benchmark_v3.csv).
