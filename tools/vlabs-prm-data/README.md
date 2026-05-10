# vlabs-prm-data

Phase 30.B process-reward dataset extraction CLI. Builds the JSONL
training set for the distilled PRM service (D3-A Qwen2.5-1.5B
student + per-step regression head); see
[`PHASE_30_PLAN.md`](../../PHASE_30_PLAN.md) for the locked
architectural rulings.

## Pipelines

The CLI surfaces five commands, each independently runnable:

| Command            | Source                                          | Output                                                          |
|--------------------|-------------------------------------------------|-----------------------------------------------------------------|
| `extract`          | 22-env catalogue (D5-A primary)                 | JSONL of `ProcessRewardTraceRow` rows tagged `source="env"`     |
| `extend-from-rm`   | Phase 29 `RewardTrainingRow` JSONL              | JSONL rows with segmentation + per-step labels added            |
| `judge-steps`      | OpenRouter per-step frontier slice (D2-C)       | JSONL rows tagged `source="judgment"` (in-place merge)          |
| `merge`            | List of JSONL shards                            | One concatenated JSONL                                          |
| `summary`          | One JSONL                                       | Aggregate stats printed to stdout                               |

## Quickstart (no API keys required)

```bash
# 1. Smoke extraction — 3 traces from each of 3 text envs
vlabs-prm-data extract \
    --envs math-algebra,sql-single-turn,code-humaneval \
    --n-per-env 3 \
    --output reports/process_reward/smoke.jsonl

# 2. Augment a Phase 29 JSONL into PRM trace rows (segment + label)
vlabs-prm-data extend-from-rm \
    --input reports/reward_distillation/v0.0.1_train.jsonl \
    --output reports/process_reward/from_rm.jsonl

# 3. Merge shards
vlabs-prm-data merge \
    --inputs reports/process_reward/smoke.jsonl,reports/process_reward/from_rm.jsonl \
    --output reports/process_reward/v0.0.1_train.jsonl

# 4. Inspect
vlabs-prm-data summary --input reports/process_reward/v0.0.1_train.jsonl
```

## Live per-step frontier-judge slice (gated)

Setting `OPENROUTER_API_KEY` AND `VLABS_PHASE30_COLLECT_FRONTIER=1`
unlocks the live judge call against `anthropic/claude-sonnet-4.6`.
The CLI enforces a **$50 cost cap** per [§19](../../PHASE_30_PLAN.md)
of the plan and refuses to proceed past that ceiling.

```bash
export OPENROUTER_API_KEY=sk-or-...
export VLABS_PHASE30_COLLECT_FRONTIER=1

vlabs-prm-data judge-steps \
    --input reports/process_reward/v0.0.1_train.jsonl \
    --output reports/process_reward/v0.0.1_train.judged.jsonl \
    --fraction 0.10 \
    --max-steps 1500
```

Without those gates the command falls back to the deterministic
`stub_step_judge_caller` (uniform 0.5 score per step) so the harness
shape is exercised without billing the OpenRouter account.

## Determinism contract

Every row carries a SHA-256 `row_id` derived from
`(env_id, seed, prompt, joined_steps)`. Re-extracting at the same
seed produces bit-identical rows; this is the §7 reproducibility
guarantee that lets the audit trail tie back to a `dataset_jobs`
row carrying the corresponding hash.

The segmenter is **deterministic** — no `random`, no seeded RNG
inside :func:`segment_trace`. R10 invariant.

## Phase status

This package ships in **Phase 30.B** as no-GPU scaffolding. The
trained student model arrives in 30.G when AWS Activate / Google
for Startups / NVIDIA Inception GPU credits resolve and Phase 29.F
clean completion is recorded; until then the data side is fully
usable for offline analysis.
