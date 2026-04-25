#!/usr/bin/env python3
"""Training-signal proof — random search over a parameterised OMP solver.

Demonstrates that the env's reward function is dense enough to drive
**parameter optimisation**: the simplest reproducible RLVR proxy.

This is **not** a full RL training loop. It uses:

- a parameterised version of the env's classical OMP solver
  (knobs: ``damping`` ∈ [0.5, 1.5], ``shrink_threshold`` ∈ [0, 0.2])
- random search over the (damping, shrink_threshold) space
- a baseline / val / held-out split so the reported gain isn't an
  overfit to the training pool

No paid LLM calls. No GPU. Runs in ~30 s with ``--quick`` (used by CI),
~5 min on the full configuration. Outputs:

- ``results/training_signal_demo.csv`` — one row per (split, params, seed)
- ``results/training_signal_demo.md`` — Markdown summary + paired-bootstrap CI

Honest framing: this proves the env *exposes* a reward signal that can
optimise a solver. Frontier-model RLVR is future work.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from verifiable_labs_envs import load_environment
from verifiable_labs_envs.envs.sparse_fourier import Prediction, _omp_single
from verifiable_labs_envs.solvers.adapters.sparse_fourier import _ls_sigma_hat_on_support

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Seed pools — chosen far past the calibration range.
BASELINE_SEEDS = list(range(70_000, 70_050))
VAL_SEEDS = list(range(70_500, 70_530))
HELDOUT_SEEDS = list(range(70_100, 70_150))

# Smoke-run subsets (CI-friendly).
SMOKE_BASELINE = list(range(70_000, 70_005))
SMOKE_VAL = list(range(70_500, 70_503))
SMOKE_HELDOUT = list(range(70_100, 70_105))


@dataclass
class SolverParams:
    damping: float          # multiplicative shrinkage on recovered amplitudes
    shrink_threshold: float  # zero-out coefficients below this magnitude (in standardised units)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeResult:
    split: str
    seed: int
    damping: float
    shrink_threshold: float
    reward: float
    nmse: float | None
    support: float | None
    conformal: float | None


def parameterised_omp_predict(instance, params: SolverParams) -> Prediction:
    """OMP recovery + tunable post-processing.

    Two knobs:

    1. ``damping`` — multiplicatively scales recovered amplitudes. The
       optimum is around 1.0 in the absence of noise; with the env's
       Gaussian noise, the best damping is slightly below 1.0
       (Stein-style shrinkage).
    2. ``shrink_threshold`` — zero out amplitudes whose magnitude is
       below this fraction of the maximum recovered amplitude. Helps
       on instances where OMP picks a spurious support entry.
    """
    inputs = instance.as_inputs()
    y, mask, sigma, n, k = (
        inputs["y"], inputs["mask"], inputs["sigma"], inputs["n"], inputs["k"],
    )
    x_hat, support = _omp_single(y, mask, n, k)

    # 1. Damping.
    x_hat = x_hat * float(params.damping)
    # 2. Shrinkage.
    if params.shrink_threshold > 0:
        max_abs = float(np.max(np.abs(x_hat)) or 1.0)
        mask_zero = np.abs(x_hat) < params.shrink_threshold * max_abs
        x_hat = np.where(mask_zero, 0.0, x_hat)

    sigma_hat = np.full(n, 1.0, dtype=np.float64)
    sigma_hat[support] = _ls_sigma_hat_on_support(mask, support, n, sigma)
    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat, support_hat=support)


def evaluate(env, params: SolverParams, seeds: Iterable[int], split: str) -> list[EpisodeResult]:
    out: list[EpisodeResult] = []
    for seed in seeds:
        instance = env.generate_instance(seed=int(seed))
        prediction = parameterised_omp_predict(instance, params)
        score = env.score(prediction, instance)
        components = score.get("components", {}) or {}
        out.append(EpisodeResult(
            split=split,
            seed=int(seed),
            damping=params.damping,
            shrink_threshold=params.shrink_threshold,
            reward=float(score.get("reward", 0.0)),
            nmse=_opt_float(components.get("nmse")),
            support=_opt_float(components.get("support")),
            conformal=_opt_float(components.get("conformal")),
        ))
    return out


def _opt_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _mean(rs: list[EpisodeResult]) -> float:
    return statistics.fmean([r.reward for r in rs]) if rs else 0.0


def _paired_bootstrap_ci(
    a: list[float], b: list[float], n_boot: int = 5000, seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (mean_delta, lo, hi) at 95 %."""
    if not a or not b or len(a) != len(b):
        return 0.0, 0.0, 0.0
    diffs = [bi - ai for ai, bi in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    boot = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(s) / n)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot) - 1]
    return sum(diffs) / n, lo, hi


def random_search(
    env, val_seeds: list[int], n_trials: int, *, seed: int = 0,
) -> tuple[SolverParams, list[tuple[SolverParams, float]]]:
    """Sample ``n_trials`` random (damping, shrink) candidates, evaluate
    each on ``val_seeds``, return the best by mean reward and the full
    history."""
    rng = random.Random(seed)
    history: list[tuple[SolverParams, float]] = []
    for _ in range(n_trials):
        params = SolverParams(
            damping=rng.uniform(0.5, 1.5),
            shrink_threshold=rng.uniform(0.0, 0.2),
        )
        rs = evaluate(env, params, val_seeds, split="val")
        history.append((params, _mean(rs)))
    best_params, best_mean = max(history, key=lambda t: t[1])
    return best_params, history


def write_csv(results: list[EpisodeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "seed", "damping", "shrink_threshold", "reward",
                        "nmse", "support", "conformal"],
        )
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def write_markdown(
    *,
    out_path: Path,
    baseline_results: list[EpisodeResult],
    best_params: SolverParams,
    best_history: list[tuple[SolverParams, float]],
    heldout_baseline: list[EpisodeResult],
    heldout_best: list[EpisodeResult],
    elapsed_s: float,
) -> None:
    baseline_mean = _mean(baseline_results)
    heldout_baseline_mean = _mean(heldout_baseline)
    heldout_best_mean = _mean(heldout_best)
    delta, lo, hi = _paired_bootstrap_ci(
        [r.reward for r in heldout_baseline], [r.reward for r in heldout_best],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""# Training-signal demo — sparse-fourier-recovery

_v0.1.0-alpha · generated by `examples/training_signal_demo.py`_

This is a **minimal proof** that the Verifiable Labs env's reward is
dense enough to drive parameter optimisation. It is **not** a full
RL training loop. The optimised "policy" is a parameterised
classical solver (OMP with two scalar knobs); the optimiser is plain
random search.

## Setup

- Env: `sparse-fourier-recovery`
- Solver: parameterised OMP (knobs: `damping`, `shrink_threshold`)
- Baseline: `damping=1.0, shrink_threshold=0.0` (matches the env's
  default classical solver)
- Search: random over `damping ∈ [0.5, 1.5]`,
  `shrink_threshold ∈ [0, 0.2]` for {len(best_history)} trials
- Splits: {len(baseline_results)} baseline / {len(best_history)} val candidates
  × N val seeds / {len(heldout_baseline)} held-out

## Headline

| metric | value |
|---|---|
| baseline mean reward (no tuning) | **{baseline_mean:.4f}** |
| best params on val | `damping={best_params.damping:.3f}`, `shrink_threshold={best_params.shrink_threshold:.3f}` |
| held-out mean (baseline) | {heldout_baseline_mean:.4f} |
| held-out mean (best) | **{heldout_best_mean:.4f}** |
| held-out Δ (best − baseline) | **{delta:+.4f}** (95 % CI [{lo:+.4f}, {hi:+.4f}]) |
| total wall time | {elapsed_s:.1f} s |

## Reading

A 95 % CI strictly above zero would mean the random search found a
parameter setting that **generalises** beyond the val pool —
training-signal evidence at p < 0.05.

A CI that includes zero means: either the parameter space doesn't
contain a meaningfully better setting (the env's default OMP is
already near-optimal in the explored ranges), or the search budget /
val pool was too small. Both are honest outcomes; we report whatever
falls out.

## Top {min(5, len(best_history))} val parameters

| rank | damping | shrink_threshold | mean reward (val) |
|---:|---:|---:|---:|
"""
    )
    sorted_hist = sorted(best_history, key=lambda t: -t[1])[:5]
    rows = []
    for i, (p, m) in enumerate(sorted_hist, start=1):
        rows.append(f"| {i} | {p.damping:.3f} | {p.shrink_threshold:.3f} | {m:.4f} |\n")
    out_path.write_text(
        out_path.read_text()
        + "".join(rows)
        + """
## Honest scope

- Single env, single seed pool, single optimiser. No claim about
  generalisation to other envs or search algorithms.
- The "policy" here is a 2-parameter classical solver. Real RLVR on
  frontier LLMs would consume the same `env.score(prediction, instance)`
  signal but require gradient-style updates on a much larger
  parameter set.
- v0.1.0-alpha. The platform makes no claim of regulatory compliance.

## Reproducing

```bash
# Quick (~30 s, used by CI):
python examples/training_signal_demo.py --quick

# Full (~5 min):
python examples/training_signal_demo.py
```

Outputs:

- `results/training_signal_demo.csv` — per-(split, params, seed) rows
- `results/training_signal_demo.md` — this file
"""
    )


def run_demo(*, quick: bool, n_trials: int, seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = load_environment("sparse-fourier-recovery", calibration_quantile=2.0)

    baseline_seeds = SMOKE_BASELINE if quick else BASELINE_SEEDS
    val_seeds = SMOKE_VAL if quick else VAL_SEEDS
    heldout_seeds = SMOKE_HELDOUT if quick else HELDOUT_SEEDS

    t0 = time.perf_counter()

    # 1. Baseline at default parameters.
    baseline_params = SolverParams(damping=1.0, shrink_threshold=0.0)
    baseline_results = evaluate(env, baseline_params, baseline_seeds, split="baseline")

    # 2. Random search on val.
    best_params, history = random_search(env, val_seeds, n_trials=n_trials, seed=seed)

    # 3. Held-out evaluation: baseline vs best.
    heldout_baseline = evaluate(env, baseline_params, heldout_seeds, split="heldout-baseline")
    heldout_best = evaluate(env, best_params, heldout_seeds, split="heldout-best")

    elapsed_s = time.perf_counter() - t0

    # Persist.
    csv_path = out_dir / "training_signal_demo.csv"
    md_path = out_dir / "training_signal_demo.md"
    write_csv(
        baseline_results + heldout_baseline + heldout_best,
        csv_path,
    )
    write_markdown(
        out_path=md_path,
        baseline_results=baseline_results,
        best_params=best_params,
        best_history=history,
        heldout_baseline=heldout_baseline,
        heldout_best=heldout_best,
        elapsed_s=elapsed_s,
    )

    summary = {
        "quick": quick,
        "n_trials": n_trials,
        "baseline_mean": _mean(baseline_results),
        "best_params": best_params.as_dict(),
        "best_val_mean": max(m for _, m in history),
        "heldout_baseline_mean": _mean(heldout_baseline),
        "heldout_best_mean": _mean(heldout_best),
        "elapsed_s": elapsed_s,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="tiny seed pools (CI-friendly, ~30 s)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="random-search trial count (default: 8 quick, 30 full)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    n_trials = args.n_trials if args.n_trials is not None else (8 if args.quick else 30)
    run_demo(
        quick=args.quick, n_trials=n_trials, seed=args.seed, out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
