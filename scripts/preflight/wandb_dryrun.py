"""scripts/preflight/wandb_dryrun.py — W&B logging dry-run.

Initialises a W&B run in offline mode (or no-op fallback when wandb
is not installed), logs:

- 100 fake training metrics (loss + grad_norm + lr) following an
  exponential-decay loss curve so the offline page renders something
  recognisable.
- A fake confusion matrix as a 4×4 list-of-lists.
- A fake plot artifact (matplotlib not required — we log a JSON
  summary instead so the script works without optional deps).

Output: reports/preflight/wandb_dryrun_smoke.json (script-local
report) plus the W&B offline run directory under ``wandb/offline/``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "preflight" / "wandb_dryrun_smoke.json"

DEFAULT_N_STEPS = 100


def fake_metric_curve(step: int, total: int) -> dict[str, float]:
    """Exponential-decay loss + linear lr-decay + clipped grad norm."""
    progress = step / max(total - 1, 1)
    loss = 2.0 * math.exp(-3.0 * progress) + 0.1
    return {
        "loss": loss,
        "grad_norm": 1.0 + 0.5 * math.sin(step / 7.0),
        "lr": 2e-4 * (1.0 - 0.9 * progress),
    }


def fake_confusion_matrix() -> list[list[int]]:
    """4×4 confusion matrix with a strong diagonal — easy to eyeball."""
    return [
        [40, 2, 1, 0],
        [3, 38, 1, 1],
        [1, 2, 36, 4],
        [0, 1, 3, 39],
    ]


def run_smoke(
    out: Path | str = DEFAULT_OUT,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    project: str = "vlabs-preflight-smoke",
    run_name: str = "wandb-dryrun",
) -> dict:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the Phase 29 wandb_callback shim so we get the no-op
    # fallback for free when wandb isn't installed.
    from vlabs_reward_train import wandb_callback

    handle = wandb_callback.init_wandb_run(
        project=project,
        name=run_name,
        config={"n_steps": n_steps, "smoke": True},
        mode="offline",
        fallback_to_noop=True,
    )

    metrics_logged: list[dict[str, float]] = []
    for step in range(n_steps):
        m = fake_metric_curve(step, n_steps)
        wandb_callback.log_metrics(handle, step, m)
        metrics_logged.append(m)

    confusion = fake_confusion_matrix()
    # Log a calibration card payload — exercises the second helper.
    wandb_callback.log_calibration_card(
        handle,
        {
            "schema_version": "v0.1.0",
            "buckets": [
                {"low": 0.0, "high": 0.5, "n": 100, "coverage": 0.93},
                {"low": 0.5, "high": 1.0, "n": 100, "coverage": 0.91},
            ],
            "confusion_matrix": confusion,
        },
    )
    handle.finish()

    final_loss = metrics_logged[-1]["loss"] if metrics_logged else None
    initial_loss = metrics_logged[0]["loss"] if metrics_logged else None
    loss_dropped = (
        final_loss < initial_loss
        if (final_loss is not None and initial_loss is not None)
        else False
    )

    report = {
        "ok": True,
        "n_steps": n_steps,
        "is_real_wandb": handle.is_real,
        "mode": handle.mode,
        "project": handle.project,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_dropped,
        "confusion_matrix_shape": [len(confusion), len(confusion[0])],
        "n_metrics_logged": len(metrics_logged),
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--project", default="vlabs-preflight-smoke")
    parser.add_argument("--run-name", default="wandb-dryrun")
    parser.add_argument("--quiet", action="store_true")
    ns = parser.parse_args(argv)

    try:
        report = run_smoke(
            out=ns.out,
            n_steps=ns.n_steps,
            project=ns.project,
            run_name=ns.run_name,
        )
    except Exception as exc:  # noqa: BLE001
        if not ns.quiet:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not ns.quiet:
        print(f"OK -> {ns.out}")
        print(
            f"  is_real_wandb={report['is_real_wandb']} "
            f"mode={report['mode']} steps={report['n_metrics_logged']} "
            f"loss {report['initial_loss']:.3f} -> {report['final_loss']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
