"""scripts/preflight/smoke_phase30_eval.py — Phase 30 PRM eval smoke.

CPU-only exercise of the PRM eval pipeline:

1. Synthetic ProcessBench dataset construction.
2. Stub PRM step predictor that returns 0.5 → ProcessBench scoring.
3. Synthetic BoN problems → BoN reranking with the same stub.
4. ``bon_lift_metrics`` computes the lift-vs-single-completion +
   lift-vs-RM headline metrics. The 30.F gating floor is +5 pp
   vs Phase 29 RM on math-algebra (D6-B); we don't enforce it here
   because the stub returns identical scores everywhere.

Output: reports/preflight/phase30_eval_smoke.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "preflight" / "phase30_eval_smoke.json"

BASELINE_STUB_VALUE = 0.5


def stub_step_predictor(_problem: str, _steps, _step_idx: int) -> float:
    """ProcessBench step-level stub. Returns 0.5 for every step.

    Because the threshold is 0.5 inclusive, the stub never marks any
    step as "below threshold" → the model behaves as if every trace
    is fully correct."""
    return BASELINE_STUB_VALUE


def stub_aggregate_predictor(_prompt: str, _steps) -> float:
    """Trace-level stub used by the BoN reranker. Constant baseline
    so the reranker resolves ties to the lowest index → equivalent
    to picking the first candidate."""
    return BASELINE_STUB_VALUE


def stub_rm_predictor(_prompt: str, _completion: str) -> float:
    """Phase 29 RM stub for the comparative BoN baseline."""
    return BASELINE_STUB_VALUE


def run_smoke(out: Path | str = DEFAULT_OUT) -> dict:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from verifiable_labs_envs.process_reward import bon_rerank
    from verifiable_labs_envs.process_reward import eval as prm_eval

    # 1. ProcessBench scaffold + scoring.
    pb_traces = prm_eval.build_synthetic_processbench(n_traces=10, seed=2024)
    pb_report = prm_eval.evaluate_processbench(
        traces=pb_traces,
        step_predictor=stub_step_predictor,
        error_threshold=0.5,
    )
    processbench_payload = {
        "n_traces": pb_report.n_traces,
        "overall_accuracy": pb_report.overall_accuracy,
        "per_subset": dict(pb_report.per_subset),
        "per_subset_count": dict(pb_report.per_subset_count),
        "n_correct_traces": pb_report.n_correct_traces,
        "n_error_traces": pb_report.n_error_traces,
        "passes": pb_report.passes(),
    }

    # 2. Synthetic BoN problems, 5 problems × 3 candidates.
    problems = bon_rerank.make_synthetic_bon_problems(
        n_problems=5, n_per_problem=3, seed=2024
    )

    # Per-problem reranking so we can inspect chosen indices.
    bon_choices = []
    for cands in problems:
        result = bon_rerank.rerank_bon(cands, stub_aggregate_predictor)
        bon_choices.append(
            {
                "n_candidates": result.n_candidates,
                "chosen_index": result.chosen_index,
                "chosen_aggregate": result.chosen_aggregate,
                "chosen_env_reward": result.chosen_env_reward,
            }
        )

    # 3. Headline lift metrics across all problems.
    lift_metrics = bon_rerank.bon_lift_metrics(
        problems,
        prm_aggregate_predictor=stub_aggregate_predictor,
        rm_predictor=stub_rm_predictor,
        correct_threshold=0.5,
    )
    bon_payload = {
        "n_problems": len(problems),
        "n_candidates_per_problem": 3,
        "metrics": dict(lift_metrics),
        "passes_lift_floor_5pct": bon_rerank.passes_bon_lift_floor(
            lift_metrics, floor=0.05
        ),
        "per_problem_choices": bon_choices,
    }

    # 4. evaluate_bon convenience wrapper sanity check.
    bon_eval = prm_eval.evaluate_bon(
        problems=problems,
        aggregate_predictor=stub_aggregate_predictor,
        rm_predictor=stub_rm_predictor,
    )
    bon_eval_payload = dict(bon_eval)

    report = {
        "phase": "30",
        "track": "eval",
        "ok": True,
        "stub_baseline_value": BASELINE_STUB_VALUE,
        "processbench": processbench_payload,
        "bon": bon_payload,
        "evaluate_bon_summary": bon_eval_payload,
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
            f"  pb_acc={report['processbench']['overall_accuracy']:.3f} "
            f"bon_n={report['bon']['n_problems']} "
            f"prm_acc={report['bon']['metrics'].get('prm_bon_accuracy', 0):.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
