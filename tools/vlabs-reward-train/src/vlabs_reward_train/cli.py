"""``vlabs-reward-train`` Typer CLI — Phase 29.C scaffold.

Subcommands:

- ``train`` — full training run. Refuses to proceed in 29.C; the GPU
  path lights up in 29.F when credits resolve.
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
    help="Train the distilled reward model (Phase 29.C scaffolding; GPU runs gated to 29.F).",
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
    lr: Annotated[float, typer.Option("--lr")] = 2e-4,
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
    lr: Annotated[float, typer.Option("--lr")] = 2e-4,
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
) -> None:
    """Run the full GRPO training loop. **Gated to 29.F.**"""
    from vlabs_reward_train.trainer import (
        GpuPathNotImplemented,
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
    )
    status = validate_dependencies()
    if not status.is_satisfied:
        typer.echo(
            "ABORT: missing dependencies — install vlabs-reward-train[gpu] "
            f"to pull: {', '.join(status.missing)}",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        build_grpo_trainer(config)
    except GpuPathNotImplemented as exc:
        typer.echo(f"ABORT: {exc}", err=True)
        raise typer.Exit(code=3) from exc


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
