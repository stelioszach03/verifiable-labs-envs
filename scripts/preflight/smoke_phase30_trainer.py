"""scripts/preflight/smoke_phase30_trainer.py — Phase 30 PRM trainer smoke.

CPU-only end-to-end exercise of the Phase 30 PRM training scaffold:

1. ``validate_dependencies`` reports which heavy deps are present.
2. ``PrmTrainingConfig`` round-trips + multi-task config (D4-D)
   surfaces the per-step / outcome weight blend correctly.
3. ``build_training_args`` produces a serialisable args dict.
4. ``write_run_card`` emits a markdown run card under tmp/.
5. PRM checkpoint manifest write+read+list round-trip.
6. Fake R2 upload of the manifest.

Output: reports/preflight/phase30_trainer_smoke.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "preflight" / "phase30_trainer_smoke.json"


def run_smoke(out: Path | str = DEFAULT_OUT) -> dict:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from verifiable_labs_envs.process_reward import (
        checkpoint as prm_checkpoint,
    )
    from verifiable_labs_envs.process_reward import trainer as prm_trainer
    from verifiable_labs_envs.process_reward import (
        wandb_integration as prm_wandb,
    )

    # 1. Dependencies.
    deps = prm_trainer.validate_dependencies()
    deps_payload = {
        "available": list(deps.available),
        "missing": list(deps.missing),
        "is_satisfied": deps.is_satisfied,
    }

    # 2. Config round-trip + multi-task blend.
    cfg = prm_trainer.PrmTrainingConfig(
        dataset_path="/tmp/preflight/fake-prm-train.jsonl",
        output_dir=str(out_path.parent / "phase30_trainer_run"),
        multi_task=True,
        multi_task_outcome_weight=0.3,
    )
    cfg_dict = cfg.to_dict()
    cfg_back = prm_trainer.PrmTrainingConfig.from_dict(cfg_dict)
    cfg_payload = {
        "round_trip_ok": cfg_back == cfg,
        "shared_backbone": cfg.shared_backbone,
        "per_step_loss_weight": cfg.per_step_loss_weight,
        "outcome_weight": cfg.multi_task_outcome_weight,
        "weights_sum_to_one": (
            abs(
                cfg.per_step_loss_weight + cfg.multi_task_outcome_weight - 1.0
            )
            < 1e-9
        ),
        "num_keys": len(cfg_dict),
    }

    # 3. Build training args.
    args = prm_trainer.build_training_args(cfg)
    args_payload = {
        "n_keys": len(args),
        "has_lr": "learning_rate" in args or "lr" in args,
    }

    # 4. Run card emission. ``write_run_card(output_dir, cfg, status)``
    #    writes ``run_card.json`` into ``output_dir`` (sandboxed under
    #    the smoke output tree).
    run_card_dir = out_path.parent / "phase30_trainer_run"
    run_card_path = prm_trainer.write_run_card(run_card_dir, cfg, deps)
    run_card_payload = {
        "path": str(run_card_path),
        "exists": run_card_path.is_file(),
        "size_bytes": (
            run_card_path.stat().st_size if run_card_path.is_file() else 0
        ),
    }

    # 5. PRM checkpoint manifest.
    ckpt_dir = out_path.parent / "phase30_trainer_run" / "ckpt"
    manifest = prm_checkpoint.PrmCheckpointManifest(
        model_id=prm_checkpoint.model_id_for(
            "distilled-qwen-1-5b", "0.0.0-smoke"
        ),
        version="0.0.0-smoke",
        base_model=cfg.base_model,
        step_granularity="per_step",
        base_rm_id=None,
        lora_config={"r": cfg.lora_r, "alpha": cfg.lora_alpha},
        training_config=cfg.to_dict(),
        multi_task={
            "enabled": cfg.multi_task,
            "outcome_weight": cfg.multi_task_outcome_weight,
            "per_step_weight": cfg.per_step_loss_weight,
        },
        metrics={"smoke_step": 5, "loss_proxy": 0.5},
        checkpoint_files=(),
        step_conformal_quantiles=None,
        aggregate_conformal_quantile=0.087,
        created_at_unix=float(int(time.time())),
        schema_version=prm_checkpoint.SCHEMA_VERSION,
    )
    manifest_path = prm_checkpoint.write_manifest(ckpt_dir, manifest)
    read_back = prm_checkpoint.read_manifest(manifest_path)
    manifest_payload = {
        "path": str(manifest_path),
        "model_id": read_back.model_id,
        "round_trip_ok": read_back == manifest,
    }

    # 6. Fake R2 upload.
    uploader = prm_checkpoint.fake_r2_uploader(
        out_path.parent / "phase30_trainer_run" / "fake-r2"
    )
    uploaded = prm_checkpoint.upload_checkpoint(
        ckpt_dir, manifest, uploader=uploader
    )
    uploader_payload = {
        "uploaded_keys": list(uploaded.keys()),
        "n_uploaded": len(uploaded),
        "is_real_r2_configured": prm_checkpoint.is_real_r2_configured(),
    }

    # 7. Local checkpoint discovery.
    found = prm_checkpoint.list_local_checkpoints(ckpt_dir.parent)
    list_payload = {
        "n_checkpoints": len(found),
        "first_model_id": found[0].model_id if found else None,
    }

    # 8. Wandb integration probe.
    wandb_payload = {
        "is_wandb_available": prm_wandb.is_wandb_available(),
        "has_wandb_credentials": prm_wandb.has_wandb_credentials(),
    }

    report = {
        "phase": "30",
        "track": "trainer",
        "ok": True,
        "dependencies": deps_payload,
        "config": cfg_payload,
        "training_args": args_payload,
        "run_card": run_card_payload,
        "manifest": manifest_payload,
        "upload": uploader_payload,
        "list_local": list_payload,
        "wandb": wandb_payload,
        "fake_steps_executed": 5,
        "loss_proxy": 0.5,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--quiet", action="store_true")
    ns = parser.parse_args(argv)

    try:
        report = run_smoke(out=ns.out)
    except Exception as exc:  # noqa: BLE001
        if not ns.quiet:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not ns.quiet:
        print(f"OK -> {ns.out}")
        print(
            f"  deps_satisfied={report['dependencies']['is_satisfied']} "
            f"weights_blend_ok={report['config']['weights_sum_to_one']} "
            f"manifest_round_trip={report['manifest']['round_trip_ok']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
