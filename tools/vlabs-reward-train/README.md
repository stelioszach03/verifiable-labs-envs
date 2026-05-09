# vlabs-reward-train

Phase 29.C training scaffolding for the distilled reward model service.
Wraps TRL's `GRPOTrainer` (D6-A regression-on-consensus_reward via LoRA
adapters per [`PHASE_29_PLAN.md`](../../PHASE_29_PLAN.md) §3 D3) plus
W&B integration, calibration, and checkpointing helpers.

> **Status:** scaffolding only. Real training runs land in **Phase
> 29.F** when GPU credits resolve (AWS Activate / Google for Startups
> / NVIDIA Inception). Until then the CLI's `train` command performs
> a dry-run dependency check and refuses to proceed.

## Modules

| Module                | Purpose                                                          |
|-----------------------|------------------------------------------------------------------|
| `lora_config.py`      | PEFT LoraConfig defaults (rank 16, alpha 32, q/k/v/o targets)    |
| `data_loader.py`      | JSONL → torch Dataset + DataLoader (CPU-friendly)                |
| `calibration.py`      | Conformal calibration step on the trained student's residuals    |
| `checkpointing.py`    | Local + (mocked) R2 upload + manifest                            |
| `wandb_callback.py`   | W&B integration via `wandb.init(mode="offline")` for CI safety   |
| `trainer.py`          | TRL `GRPOTrainer` wrapper + dependency-check                     |
| `eval.py`             | In-loop held-out env scoring on every checkpoint                 |
| `cli.py`              | Typer CLI: `train`, `dry-run`, `version`, `dependencies`         |

## Quickstart (CPU dry-run; no training)

```bash
# Verify dependencies
vlabs-reward-train dependencies

# Print resolved training config without running training
vlabs-reward-train dry-run \
    --dataset reports/reward_distillation/v0.0.1_train.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir runs/reward-train/exp_001/
```

The full GRPO training command (Phase 29.F):

```bash
vlabs-reward-train train \
    --dataset reports/reward_distillation/v0.0.1_train.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir runs/reward-train/exp_001/ \
    --lora-r 16 --lora-alpha 32 \
    --lr 2e-4 --epochs 3 --batch-size 16 \
    --grad-accum 4 \
    --eval-set reports/reward_distillation/v0.0.1_eval.jsonl \
    --calib-set reports/reward_distillation/v0.0.1_calib.jsonl \
    --wandb-project vlabs-reward-distillation
```

## What 29.C ships (and what it doesn't)

| ✓ Lands in 29.C                              | ✗ Deferred to 29.F                             |
|----------------------------------------------|------------------------------------------------|
| LoraConfig dict + adapter shape              | Real LoRA fine-tune on a real GPU              |
| JSONL → Dataset / DataLoader on CPU          | bf16 mixed-precision training loop             |
| Conformal calibration step (split-conformal) | Real W&B run logging                           |
| Mock-friendly checkpoint IO                  | R2 upload of trained checkpoints                |
| CLI surface + dry-run + `dependencies` check | Production deploy                               |

## Dependency contract

Heavy deps (`torch`, `transformers`, `peft`, `trl`, `wandb`) are
imported **lazily** so the CLI's help text + dry-run path work in a
minimal environment. The `train` command guards on
`validate_dependencies()` which surfaces a friendly error listing
missing packages instead of failing inside the trainer.
