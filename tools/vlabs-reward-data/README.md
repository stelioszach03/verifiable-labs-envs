# vlabs-reward-data

Phase 29.B reward-distillation dataset extraction CLI for the
Verifiable Labs platform. Builds the JSONL training set for the
distilled reward model service (D2-A Qwen2.5-1.5B student + LoRA
fine-tune); see [`PHASE_29_PLAN.md`](../../PHASE_29_PLAN.md) for the
locked architectural rulings.

## Pipelines

The CLI surfaces five commands, each independently runnable:

| Command            | Source                              | Output                                                     |
|--------------------|-------------------------------------|------------------------------------------------------------|
| `extract`          | 22-env catalogue (D4-A primary)     | JSONL of `RewardTrainingRow` rows tagged `source="env"`    |
| `extract-external` | UltraFeedback HF subset (D4-C)      | JSONL rows tagged `source="external"`                      |
| `judge`            | OpenRouter frontier-model slice     | JSONL rows tagged `source="judgment"` (in-place merge)     |
| `merge`            | List of JSONL shards                | One concatenated JSONL                                     |
| `summary`          | One JSONL                           | Aggregate stats printed to stdout                          |

## Quickstart (no API keys required)

```bash
# 1. Smoke extraction — 5 rows from each of 3 envs
vlabs-reward-data extract \
    --envs math-algebra,sql-single-turn,code-humaneval \
    --n-per-env 5 \
    --output reports/reward_distillation/smoke.jsonl

# 2. UltraFeedback synthetic stand-in (no datasets dep needed)
vlabs-reward-data extract-external \
    --n 100 --seed 42 \
    --output reports/reward_distillation/external.jsonl

# 3. Merge shards
vlabs-reward-data merge \
    --inputs reports/reward_distillation/smoke.jsonl,reports/reward_distillation/external.jsonl \
    --output reports/reward_distillation/v0.0.1_train.jsonl

# 4. Inspect
vlabs-reward-data summary --input reports/reward_distillation/v0.0.1_train.jsonl
```

## Live frontier-judge slice (gated)

Setting `OPENROUTER_API_KEY` AND `VLABS_PHASE29_COLLECT_FRONTIER=1`
unlocks the live judge call against `anthropic/claude-sonnet-4.6`.
The CLI enforces a **$30 cost cap** per [§5 D1-D](../../PHASE_29_PLAN.md)
of the plan and refuses to proceed past that ceiling.

```bash
export OPENROUTER_API_KEY=sk-or-...
export VLABS_PHASE29_COLLECT_FRONTIER=1

vlabs-reward-data judge \
    --input reports/reward_distillation/v0.0.1_train.jsonl \
    --output reports/reward_distillation/v0.0.1_train.judged.jsonl \
    --fraction 0.10 \
    --max-rows 1500
```

Without those gates the command falls back to the deterministic
`stub_judge_caller` (uniform 0.5 score) so the harness shape is
exercised without billing the OpenRouter account.

## Determinism contract

Every row carries a SHA-256 `row_id` derived from
`(env_id, seed, prompt, completion)`. Re-extracting at the same
seed produces bit-identical rows; this is the §7 reproducibility
guarantee that lets the audit trail tie back to a `dataset_jobs`
row carrying the corresponding hash.

## Phase status

This package ships in **Phase 29.B** as no-GPU scaffolding. The
trained student model arrives in 29.G when AWS Activate / Google
for Startups / NVIDIA Inception GPU credits resolve; until then
the data side is fully usable for offline analysis.
