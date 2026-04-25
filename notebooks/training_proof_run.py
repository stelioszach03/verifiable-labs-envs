"""CLI runner for the training-proof experiment.

Produces ``results/training_proof.csv`` and a JSON summary at
``results/training_proof_summary.json``. Used both as a standalone
script and by the notebook (which invokes ``run_experiment`` directly).

Usage::

    # Tiny smoke run (3 train, 2 val, 3 held-out seeds) — under $0.10:
    python notebooks/training_proof_run.py --smoke

    # Full run from the brief (30 baseline, 5 val, 30 held-out — ~$1.10):
    python notebooks/training_proof_run.py

    # Cap the spend at $0.50 instead of the default $2:
    python notebooks/training_proof_run.py --cap-usd 0.50

The script:
  1. Loads ``sparse-fourier-recovery-multiturn`` (calibrated, fast=True).
  2. Evaluates the env's default ``SYSTEM_PROMPT_MT`` on the validation
     pool and on each ``DEFAULT_CANDIDATES`` prompt.
  3. Picks the highest-mean prompt as ``best``.
  4. Runs both default and ``best`` on the held-out seed pool.
  5. Computes paired-bootstrap CI on the held-out delta.
  6. Writes everything to disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

import training_proof_lib as lib  # noqa: E402

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# Seed pools — chosen to sit far past the calibration seed range so we
# don't accidentally evaluate on calibration data.
BASELINE_SEEDS = list(range(60_000, 60_030))   # 30 seeds
VAL_SEEDS = list(range(60_500, 60_505))        # 5 seeds for prompt selection
HELDOUT_SEEDS = list(range(60_100, 60_130))    # 30 unseen seeds

# Smoke-run subsets (fast, cheap).
SMOKE_BASELINE = list(range(60_000, 60_003))
SMOKE_VAL = list(range(60_500, 60_502))
SMOKE_HELDOUT = list(range(60_100, 60_103))


def _build_solver(model: str):
    """Construct the OpenRouterSolver. Raises RuntimeError if no key."""
    from verifiable_labs_envs.solvers import OpenRouterSolver
    return OpenRouterSolver(model=model)


def _default_candidate(env_adapter_module: str) -> lib.PromptCandidate:
    """Wrap the env's own ``SYSTEM_PROMPT_MT`` as a candidate so the
    baseline run goes through the same code path as the variants."""
    from importlib import import_module
    mod = import_module(env_adapter_module)
    return lib.PromptCandidate(
        name="default-systemprompt",
        system_prompt=mod.SYSTEM_PROMPT_MT,
        provenance="env-default",
    )


def run_experiment(
    *,
    model: str = DEFAULT_MODEL,
    cap_usd: float = 2.0,
    smoke: bool = False,
    out_dir: Path | None = None,
    log: bool = True,
) -> dict:
    """Run the full experiment and return a summary dict.

    Side effects: writes ``training_proof.csv`` (per-seed rewards) and
    ``training_proof_summary.json`` (aggregates) under ``out_dir``.
    """
    if out_dir is None:
        out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    from verifiable_labs_envs import load_environment  # noqa: PLC0415

    env = load_environment("sparse-fourier-recovery-multiturn")  # default fast calibration
    solver = _build_solver(model)
    budget = lib.BudgetCap(cap_usd=cap_usd)

    baseline_seeds = SMOKE_BASELINE if smoke else BASELINE_SEEDS
    val_seeds = SMOKE_VAL if smoke else VAL_SEEDS
    heldout_seeds = SMOKE_HELDOUT if smoke else HELDOUT_SEEDS

    default_cand = _default_candidate(
        "verifiable_labs_envs.solvers.adapters.sparse_fourier_multiturn"
    )

    def _say(msg: str) -> None:
        if log:
            print(msg, flush=True)

    t0 = time.perf_counter()
    _say(f"== model: {model}")
    _say(f"== cap:   ${cap_usd:.2f}")
    _say(f"== smoke: {smoke}  (seeds: {len(baseline_seeds)} baseline / "
         f"{len(val_seeds)} val / {len(heldout_seeds)} held-out)")

    # 1. Baseline distribution on baseline_seeds.
    _say("\n[1/4] Baseline distribution on default prompt:")
    baseline_results = lib.evaluate_prompt(
        env, solver, default_cand, baseline_seeds, budget,
        on_seed=lambda r: _say(
            f"  seed={r.seed:>5d} reward={r.reward:.3f} "
            f"parse={'ok' if r.parse_ok else 'FAIL'} "
            f"turns={r.n_turns} spent=${budget.spent_usd:.4f}"
        ),
    )
    baseline_summary = lib.summarise(baseline_results)
    _say(f"  → mean={baseline_summary.mean:.4f} ± {baseline_summary.std:.4f} "
         f"(parse-fail {baseline_summary.parse_fail_rate:.2f})")

    # 2. Candidate validation: each candidate is evaluated on val_seeds.
    _say("\n[2/4] Candidate validation:")
    candidate_summaries: list[lib.RewardSummary] = []
    val_results_by_candidate: dict[str, list[lib.SeedResult]] = {}
    for cand in lib.DEFAULT_CANDIDATES:
        _say(f"  candidate={cand.name}")
        results = lib.evaluate_prompt(env, solver, cand, val_seeds, budget)
        val_results_by_candidate[cand.name] = results
        s = lib.summarise(results)
        candidate_summaries.append(s)
        _say(f"    mean={s.mean:.4f} parse-fail={s.parse_fail_rate:.2f} "
             f"spent=${budget.spent_usd:.4f}")

    best = lib.best_candidate(candidate_summaries)
    best_cand = next(c for c in lib.DEFAULT_CANDIDATES if c.name == best.prompt_name)
    _say(f"\n[3/4] Best candidate by val: {best.prompt_name} (mean={best.mean:.4f})")

    # 3. Held-out evaluation: default + best on the same seeds.
    _say("\n[4/4] Held-out evaluation (default vs best):")
    heldout_default = lib.evaluate_prompt(env, solver, default_cand, heldout_seeds, budget)
    heldout_best = lib.evaluate_prompt(env, solver, best_cand, heldout_seeds, budget)

    summary_default = lib.summarise(heldout_default)
    summary_best = lib.summarise(heldout_best)

    bootstrap = lib.paired_bootstrap_ci(
        summary_default.rewards, summary_best.rewards, n_bootstrap=5000, seed=0,
    )
    _say(f"\n  default mean={summary_default.mean:.4f}")
    _say(f"  best    mean={summary_best.mean:.4f}")
    _say(f"  Δ = {bootstrap.delta:+.4f} (95% CI [{bootstrap.lo:+.4f}, {bootstrap.hi:+.4f}])")

    elapsed = time.perf_counter() - t0
    _say(f"\nTotal spent: ${budget.spent_usd:.4f} of ${cap_usd:.2f} cap "
         f"({elapsed:.1f}s wall)")

    # 4. Persist raw + summary.
    csv_path = out_dir / ("training_proof_smoke.csv" if smoke else "training_proof.csv")
    json_path = out_dir / (
        "training_proof_smoke_summary.json" if smoke else "training_proof_summary.json"
    )
    rows: list[dict] = []
    rows.extend({**asdict(r), "split": "baseline"} for r in baseline_results)
    for cand_name, results in val_results_by_candidate.items():
        rows.extend({**asdict(r), "split": f"val:{cand_name}"} for r in results)
    rows.extend({**asdict(r), "split": "heldout-default"} for r in heldout_default)
    rows.extend({**asdict(r), "split": "heldout-best"} for r in heldout_best)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "seed", "prompt_name", "reward", "n_turns", "parse_ok", "usd_cost"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    _say(f"\nWrote per-seed CSV → {csv_path}")

    summary_doc = {
        "model": model,
        "cap_usd": cap_usd,
        "spent_usd": budget.spent_usd,
        "elapsed_s": elapsed,
        "smoke": smoke,
        "baseline": {
            "seeds": baseline_seeds,
            "mean": baseline_summary.mean,
            "std": baseline_summary.std,
            "parse_fail_rate": baseline_summary.parse_fail_rate,
        },
        "candidates": [
            {"name": s.prompt_name, "mean": s.mean, "parse_fail_rate": s.parse_fail_rate}
            for s in candidate_summaries
        ],
        "best": {"name": best.prompt_name, "mean_on_val": best.mean},
        "heldout": {
            "seeds": heldout_seeds,
            "default_mean": summary_default.mean,
            "best_mean": summary_best.mean,
            "delta": bootstrap.delta,
            "ci_lo": bootstrap.lo,
            "ci_hi": bootstrap.hi,
            "n_bootstrap": bootstrap.n_bootstrap,
        },
    }
    json_path.write_text(json.dumps(summary_doc, indent=2))
    _say(f"Wrote summary    → {json_path}")
    return summary_doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cap-usd", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true", help="tiny seed pools, ~$0.05")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY is not set. "
              "Add it to .env or export it before running.", file=sys.stderr)
        return 2

    run_experiment(
        model=args.model,
        cap_usd=args.cap_usd,
        smoke=args.smoke,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
