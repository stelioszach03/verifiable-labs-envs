"""scripts/preflight/smoke_phase29_eval.py — Phase 29 eval smoke.

Exercises the held-out env eval pipeline + the rank correlation +
calibration helpers, all with a CPU-only stub student that returns
0.5 + uniform noise.

Output: reports/preflight/phase29_eval_smoke.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "preflight" / "phase29_eval_smoke.json"


def stub_student(_prompt: str, _completion: str) -> float:
    """Deterministic 0.5 baseline. Real student lands in 29.F."""
    return 0.5


def run_smoke(out: Path | str = DEFAULT_OUT) -> dict:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from vlabs_reward_train import data_loader
    from vlabs_reward_train import eval as rt_eval

    # 1. Held-out eval over 1 env, 5 episodes (synthetic via
    #    rows_by_env so we don't hit the real env runtime).
    rows = data_loader.build_synthetic_rows(n=5, seed=2024)
    rows_by_env = {"math-algebra": rows}
    report = rt_eval.evaluate_held_out_envs(
        student_predict=stub_student,
        env_ids=("math-algebra",),
        n_per_env=5,
        rows_by_env=rows_by_env,
    )
    report_payload = rt_eval.report_to_dict(report)

    # 2. Rank correlation on stub predictions.
    truth = np.array([0.1, 0.4, 0.7, 0.9, 0.5])
    pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # constant student → rho=0
    rho = rt_eval.spearman_rho(truth, pred)

    # 3. Calibration MSE — predicted vs empirical CI widths
    #    (D10-A moat metric; both arrays of the same shape).
    predicted_widths = np.array([0.10, 0.12, 0.09, 0.11, 0.10])
    empirical_widths = np.array([0.11, 0.13, 0.10, 0.12, 0.09])
    cal_mse = rt_eval.calibration_mse(predicted_widths, empirical_widths)

    output = {
        "phase": "29",
        "track": "eval",
        "ok": True,
        "held_out_eval": report_payload,
        "rank_correlation": {
            "rho_constant_student": float(rho),
            "expected_rho_for_constant": 0.0,
        },
        "calibration_mse": {"value": float(cal_mse)},
        "stub_baseline_value": 0.5,
        "n_envs": 1,
        "n_episodes_per_env": 5,
    }
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    return output


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
            f"  rho={report['rank_correlation']['rho_constant_student']:.3f} "
            f"cal_mse={report['calibration_mse']['value']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
