"""scripts/preflight/smoke_phase29_trainer.py — Phase 29 trainer smoke.

CPU-only end-to-end exercise of the vlabs-reward-train pipeline:

1. ``validate_dependencies`` reports which heavy deps are present.
2. ``TrainingConfig`` round-trips through ``to_dict`` / ``from_dict``.
3. ``data_loader.build_synthetic_rows`` produces 10 deterministic rows.
4. ``wandb_callback.init_wandb_run`` opens an offline-mode handle.
5. ``checkpointing.write_manifest`` + ``read_manifest`` round-trip a
   manifest under a tmp output dir.
6. ``checkpointing.fake_r2_uploader`` simulates an R2 upload to
   /tmp/vlabs-fake-r2/.
7. ``checkpointing.list_local_checkpoints`` finds the manifest.

Output: JSON report at reports/preflight/phase29_trainer_smoke.json.

This is the bug-net before Stelios spends GPU credits on Phase 29.F.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "preflight" / "phase29_trainer_smoke.json"


def run_smoke(out: Path | str = DEFAULT_OUT) -> dict:
    """Execute the smoke checks + write the report."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from vlabs_reward_train import (
        checkpointing,
        data_loader,
        trainer,
        wandb_callback,
    )

    # 1. Dependencies.
    deps = trainer.validate_dependencies()
    deps_payload = {
        "available": list(deps.available),
        "missing": list(deps.missing),
        "is_satisfied": deps.is_satisfied,
    }

    # 2. TrainingConfig round trip.
    cfg = trainer.TrainingConfig(
        dataset_path="/tmp/preflight/fake-train.jsonl",
        output_dir=str(out_path.parent / "phase29_trainer_run"),
    )
    cfg_dict = cfg.to_dict()
    cfg_back = trainer.TrainingConfig.from_dict(cfg_dict)
    cfg_payload = {
        "round_trip_ok": cfg_back == cfg,
        "base_model": cfg_back.base_model,
        "num_keys": len(cfg_dict),
    }

    # 3. Synthetic data.
    rows = data_loader.build_synthetic_rows(n=10, seed=0)
    rows_payload = {
        "n_rows": len(rows),
        "first_env_id": rows[0].env_id if rows else None,
    }

    # 4. W&B offline handle.
    handle = wandb_callback.init_wandb_run(
        project="vlabs-preflight-smoke",
        name="phase29-trainer-smoke",
        config=cfg.to_dict(),
        mode="offline",
        fallback_to_noop=True,
    )
    wandb_payload = {
        "is_real": handle.is_real,
        "mode": handle.mode,
        "project": handle.project,
    }
    handle.finish()

    # 5. Checkpoint manifest write+read.
    ckpt_dir = out_path.parent / "phase29_trainer_run" / "ckpt"
    manifest = checkpointing.CheckpointManifest(
        model_id=checkpointing.model_id_for(
            "distilled-qwen-1-5b", "0.0.0-smoke"
        ),
        version="0.0.0-smoke",
        base_model=cfg.base_model,
        lora_config={"r": cfg.lora_r, "alpha": cfg.lora_alpha},
        training_config=cfg.to_dict(),
        metrics={"smoke_step": 5, "loss_proxy": 0.5},
        checkpoint_files=(),
        conformal_quantile=0.087,
        schema_version=checkpointing.SCHEMA_VERSION,
        created_at_unix=float(int(time.time())),
    )
    manifest_path = checkpointing.write_manifest(ckpt_dir, manifest)
    read_back = checkpointing.read_manifest(manifest_path)
    manifest_payload = {
        "path": str(manifest_path),
        "model_id": read_back.model_id,
        "round_trip_ok": read_back == manifest,
    }

    # 6. Fake R2 upload — uploader is a callable returning the r2 URI.
    uploader = checkpointing.fake_r2_uploader(
        out_path.parent / "phase29_trainer_run" / "fake-r2"
    )
    uploaded = checkpointing.upload_checkpoint(
        ckpt_dir, manifest, uploader=uploader
    )
    uploader_payload = {
        "uploaded_keys": list(uploaded.keys()),
        "n_uploaded": len(uploaded),
        "is_real_r2_configured": checkpointing.is_real_r2_configured(),
    }

    # 7. List local checkpoints.
    found = checkpointing.list_local_checkpoints(ckpt_dir.parent)
    list_payload = {
        "n_checkpoints": len(found),
        "first_model_id": found[0].model_id if found else None,
    }

    report = {
        "phase": "29",
        "track": "trainer",
        "ok": True,
        "dependencies": deps_payload,
        "config": cfg_payload,
        "data": rows_payload,
        "wandb": wandb_payload,
        "manifest": manifest_payload,
        "upload": uploader_payload,
        "list_local": list_payload,
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
            f"data_rows={report['data']['n_rows']} "
            f"manifest_round_trip={report['manifest']['round_trip_ok']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
