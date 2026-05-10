# vlabs-prm-train

Phase 30.C training scaffolding for the process reward model service.
Wraps TRL's per-step regression loop (D3-A Qwen2.5-1.5B + per-step
head, optionally D4-D multi-task with the Phase 29 outcome head)
plus W&B integration, calibration, and checkpointing helpers per
[`PHASE_30_PLAN.md`](../../PHASE_30_PLAN.md) §3 D3 / D4.

> **Status:** scaffolding only. Real training runs land in **Phase
> 30.F** when GPU credits resolve (gate inherited from 29.F per §19).
> Until then the CLI's `train` command performs a dry-run dependency
> check and refuses to proceed.

## Dependencies

Reuses Phase 29's training toolchain verbatim where possible:

- `vlabs_reward_train.lora_config.LoraSpec` — locked LoRA r=16 / α=32.
- `vlabs_reward_train.checkpointing` — pattern for checkpoint manifest
  + R2 upload (PRM has its own
  `verifiable_labs_envs.process_reward.checkpoint` extension that
  adds per-step quantiles + base_rm_id link).
- `vlabs_reward_train.wandb_callback` — W&B init + lifecycle helpers
  (PRM logs additional per-step + multi-task metrics via
  `verifiable_labs_envs.process_reward.wandb_integration`).

## Quickstart (CPU dry-run; no training)

```bash
# Verify dependencies
vlabs-prm-train dependencies

# Print resolved training config without running training
vlabs-prm-train dry-run \
    --dataset reports/process_reward/v0.0.1_train.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir runs/prm-train/exp_001/
```

## Full GPU training command (Phase 30.F)

```bash
vlabs-prm-train train \
    --dataset reports/process_reward/v0.0.1_train.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --base-rm-checkpoint runs/reward-train/exp_004/      # D13-B/C optional
    --output-dir runs/prm-train/exp_001/ \
    --lora-r 16 --lora-alpha 32 \
    --lr 1e-4 --epochs 3 --batch-size 8 \
    --grad-accum 8 \
    --multi-task \
    --multi-task-outcome-weight 0.3 \
    --eval-set reports/process_reward/v0.0.1_eval.jsonl \
    --calib-set reports/process_reward/v0.0.1_calib.jsonl \
    --wandb-project vlabs-prm-distillation
```

## D13-C hybrid path semantics

- **D13-A (default in v0.0.1):** omit `--base-rm-checkpoint`. The
  trainer initialises fresh LoRA adapters; the PRM and Phase 29 RM
  serve from independent backbones.
- **D13-B (shared backbone, opt-in):** pass
  `--base-rm-checkpoint <path/to/phase-29-checkpoint>`. The trainer
  loads the Phase 29 RM LoRA adapters + outcome head as the starting
  point and joint-trains both heads.
- **D13-C (hybrid v0.0.2 path):** add `--multi-task
  --freeze-outcome-head`. The outcome head is frozen during PRM
  training so the joint loss only updates the per-step head + shared
  backbone adapters. Eliminates R16 (outcome regression).

## What 30.C ships (and what it doesn't)

| ✓ Lands in 30.C                                | ✗ Deferred to 30.F                          |
|------------------------------------------------|---------------------------------------------|
| PrmTrainingConfig dataclass + JSON round-trip  | Real LoRA fine-tune on a real GPU            |
| Per-step target tensor packing + mask          | bf16 mixed-precision training loop           |
| MultiTaskConfig + joint-loss builder           | Real W&B run logging                         |
| PRM checkpoint manifest + (mock) R2 upload     | R2 upload of trained checkpoints             |
| CLI surface + dry-run + dependency check       | Production deploy                            |
