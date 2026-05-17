"""``vlabs-reward-train`` Typer CLI — Phase 29.F unlock.

Subcommands:

- ``train`` — full training run. As of 29.F, runs the live GRPO
  loop with TRL 1.4 + vLLM 0.21 colocate. Refuses to proceed if
  required deps are missing or the dataset doesn't exist.
- ``dry-run`` — print the resolved config + dependency status without
  running training. Useful for asserting CLI plumbing in CI.
- ``dependencies`` — list which heavy deps are present + which W&B
  credentials are wired up.
- ``version`` — print the package version.
- ``checkpoints`` — list local checkpoints + their manifests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from vlabs_reward_train import __version__

app = typer.Typer(
    name="vlabs-reward-train",
    help="Train the distilled reward model (Phase 29.F — TRL 1.4 + vLLM 0.21 colocate).",
    no_args_is_help=True,
    add_completion=False,
)


def _build_config_from_args(
    *,
    dataset: Path,
    base_model: str,
    output_dir: Path,
    eval_set: Path | None,
    calib_set: Path | None,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lora_r: int,
    lora_alpha: int,
    wandb_project: str,
    wandb_mode: str,
    seed: int,
    max_steps: int,
    num_generations: int,
    env_id: str,
    vllm_mode: str,
    vllm_gpu_memory_utilization: float,
    vllm_tensor_parallel_size: int,
    vllm_max_model_length: int,
    max_completion_length: int,
    beta: float,
):
    from vlabs_reward_train.trainer import TrainingConfig

    return TrainingConfig(
        base_model=base_model,
        output_dir=str(output_dir),
        dataset_path=str(dataset),
        eval_dataset_path=str(eval_set) if eval_set else None,
        calib_dataset_path=str(calib_set) if calib_set else None,
        lr=float(lr),
        epochs=int(epochs),
        batch_size=int(batch_size),
        grad_accum=int(grad_accum),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        seed=int(seed),
        max_steps=int(max_steps),
        num_generations=int(num_generations),
        env_id=env_id,
        vllm_mode=vllm_mode,
        vllm_gpu_memory_utilization=float(vllm_gpu_memory_utilization),
        vllm_tensor_parallel_size=int(vllm_tensor_parallel_size),
        vllm_max_model_length=int(vllm_max_model_length),
        max_completion_length=int(max_completion_length),
        beta=float(beta),
    )


@app.command()
def version() -> None:
    """Print the CLI version and exit."""
    typer.echo(f"vlabs-reward-train v{__version__}")


@app.command()
def dependencies() -> None:
    """Probe for the GPU-training toolchain."""
    from vlabs_reward_train.trainer import validate_dependencies
    from vlabs_reward_train.wandb_callback import (
        has_wandb_credentials,
        is_wandb_available,
    )

    status = validate_dependencies()
    payload = {
        **status.to_dict(),
        "wandb_installed": is_wandb_available(),
        "wandb_credentials": has_wandb_credentials(),
        "gpu_path_ready": status.is_satisfied,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("dry-run")
def dry_run(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Training JSONL path.")
    ] = Path("reports/reward_distillation/v0.0.1_train.jsonl"),
    base_model: Annotated[
        str, typer.Option("--base-model")
    ] = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = Path("runs/reward-train/exp_001"),
    eval_set: Annotated[Path | None, typer.Option("--eval-set")] = None,
    calib_set: Annotated[Path | None, typer.Option("--calib-set")] = None,
    lr: Annotated[float, typer.Option("--lr", "--learning-rate")] = 2e-4,
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum", min=1)] = 4,
    lora_r: Annotated[int, typer.Option("--lora-r", min=1)] = 16,
    lora_alpha: Annotated[int, typer.Option("--lora-alpha", min=1)] = 32,
    wandb_project: Annotated[
        str, typer.Option("--wandb-project")
    ] = "vlabs-reward-distillation",
    wandb_mode: Annotated[
        str, typer.Option("--wandb-mode", help="online / offline / disabled")
    ] = "offline",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Hard step cap (-1 = run to epoch end).",
        ),
    ] = -1,
    num_generations: Annotated[
        int, typer.Option("--num-generations", min=2)
    ] = 4,
    env_id: Annotated[
        str,
        typer.Option(
            "--env-id",
            help="verifiable-labs-envs registry id used by the reward fn.",
        ),
    ] = "sparse-fourier-recovery",
    vllm_mode: Annotated[
        str,
        typer.Option(
            "--vllm-mode",
            help="vLLM execution mode: 'colocate' or 'server'.",
        ),
    ] = "colocate",
    vllm_gpu_memory_utilization: Annotated[
        float,
        typer.Option(
            "--vllm-gpu-memory-utilization",
            min=0.05,
            max=0.95,
            help="Fraction of GPU memory vLLM may claim.",
        ),
    ] = 0.3,
    vllm_tensor_parallel_size: Annotated[
        int, typer.Option("--vllm-tensor-parallel-size", min=1)
    ] = 1,
    vllm_max_model_length: Annotated[
        int,
        typer.Option(
            "--vllm-max-model-length",
            min=128,
            help="Total prompt+completion length budget for vLLM.",
        ),
    ] = 3072,
    max_completion_length: Annotated[
        int, typer.Option("--max-completion-length", min=16)
    ] = 1024,
    beta: Annotated[
        float,
        typer.Option(
            "--beta",
            help="KL coefficient (TRL 1.4 rename of kl_coefficient).",
        ),
    ] = 0.04,
    write_run_card_to_disk: Annotated[
        bool,
        typer.Option(
            "--write-run-card/--no-write-run-card",
            help="Persist run_card.json under output_dir.",
        ),
    ] = False,
) -> None:
    """Print the resolved training config + dependency status."""
    from vlabs_reward_train.lora_config import lora_summary
    from vlabs_reward_train.trainer import (
        build_training_args,
        validate_dependencies,
        write_run_card,
    )

    config = _build_config_from_args(
        dataset=dataset,
        base_model=base_model,
        output_dir=output_dir,
        eval_set=eval_set,
        calib_set=calib_set,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        seed=seed,
        max_steps=max_steps,
        num_generations=num_generations,
        env_id=env_id,
        vllm_mode=vllm_mode,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_max_model_length=vllm_max_model_length,
        max_completion_length=max_completion_length,
        beta=beta,
    )
    status = validate_dependencies()
    args = build_training_args(config)
    summary = lora_summary(config.lora_spec)

    payload = {
        "config": config.to_dict(),
        "training_args": args,
        "lora_summary": summary,
        "dependencies": status.to_dict(),
        "ready_to_train": status.is_satisfied,
    }
    if write_run_card_to_disk:
        target = write_run_card(output_dir, config, status)
        payload["run_card_path"] = str(target)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def train(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Training JSONL path.")
    ],
    base_model: Annotated[
        str, typer.Option("--base-model")
    ] = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = Path("runs/reward-train/exp_001"),
    eval_set: Annotated[Path | None, typer.Option("--eval-set")] = None,
    calib_set: Annotated[Path | None, typer.Option("--calib-set")] = None,
    lr: Annotated[float, typer.Option("--lr", "--learning-rate")] = 2e-4,
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 16,
    grad_accum: Annotated[int, typer.Option("--grad-accum", min=1)] = 4,
    lora_r: Annotated[int, typer.Option("--lora-r", min=1)] = 16,
    lora_alpha: Annotated[int, typer.Option("--lora-alpha", min=1)] = 32,
    wandb_project: Annotated[
        str, typer.Option("--wandb-project")
    ] = "vlabs-reward-distillation",
    wandb_mode: Annotated[
        str, typer.Option("--wandb-mode")
    ] = "offline",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Hard step cap (-1 = run to epoch end).",
        ),
    ] = -1,
    num_generations: Annotated[
        int, typer.Option("--num-generations", min=2)
    ] = 4,
    env_id: Annotated[
        str,
        typer.Option(
            "--env-id",
            help="verifiable-labs-envs registry id used by the reward fn.",
        ),
    ] = "sparse-fourier-recovery",
    vllm_mode: Annotated[
        str,
        typer.Option(
            "--vllm-mode", help="vLLM execution mode: 'colocate' or 'server'."
        ),
    ] = "colocate",
    vllm_gpu_memory_utilization: Annotated[
        float,
        typer.Option(
            "--vllm-gpu-memory-utilization",
            min=0.05,
            max=0.95,
        ),
    ] = 0.3,
    vllm_tensor_parallel_size: Annotated[
        int, typer.Option("--vllm-tensor-parallel-size", min=1)
    ] = 1,
    vllm_max_model_length: Annotated[
        int, typer.Option("--vllm-max-model-length", min=128)
    ] = 3072,
    max_completion_length: Annotated[
        int, typer.Option("--max-completion-length", min=16)
    ] = 1024,
    beta: Annotated[
        float, typer.Option("--beta")
    ] = 0.04,
) -> None:
    """Run the live GRPO training loop (29.F unlock).

    Exit codes:
      * 0 — training completed.
      * 2 — missing dependencies (install ``vlabs-reward-train[gpu]``).
      * 4 — dataset path does not exist.
      * non-zero — any other error from the trainer (re-raised).
    """
    from vlabs_reward_train.trainer import (
        build_grpo_trainer,
        validate_dependencies,
    )

    config = _build_config_from_args(
        dataset=dataset,
        base_model=base_model,
        output_dir=output_dir,
        eval_set=eval_set,
        calib_set=calib_set,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        seed=seed,
        max_steps=max_steps,
        num_generations=num_generations,
        env_id=env_id,
        vllm_mode=vllm_mode,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_max_model_length=vllm_max_model_length,
        max_completion_length=max_completion_length,
        beta=beta,
    )
    status = validate_dependencies()
    if not status.is_satisfied:
        typer.echo(
            "ABORT: missing dependencies — install vlabs-reward-train[gpu] "
            f"to pull: {', '.join(status.missing)}",
            err=True,
        )
        raise typer.Exit(code=2)

    if not Path(config.dataset_path).exists():
        typer.echo(
            f"ABORT: dataset not found at {config.dataset_path}", err=True
        )
        raise typer.Exit(code=4)

    typer.echo(
        f"Constructing GRPOTrainer (env_id={config.env_id}, "
        f"max_steps={config.max_steps}, base_model={config.base_model})…",
        err=True,
    )
    trainer = build_grpo_trainer(config)

    typer.echo(
        f"Starting training. Output → {config.output_dir}", err=True
    )
    trainer.train()
    typer.echo(
        f"Training complete. Checkpoints / logs under {config.output_dir}",
        err=True,
    )


@app.command()
def checkpoints(
    parent: Annotated[
        Path,
        typer.Option("--parent", help="Parent dir holding manifest.json files."),
    ] = Path("runs/reward-train"),
) -> None:
    """List local checkpoints + their manifests."""
    from vlabs_reward_train.checkpointing import list_local_checkpoints, manifest_table

    manifests = list_local_checkpoints(parent)
    typer.echo(json.dumps(manifest_table(manifests), indent=2, sort_keys=True))


__all__ = ["app"]
