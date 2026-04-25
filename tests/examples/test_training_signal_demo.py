"""Smoke test for ``examples/training_signal_demo.py``.

Runs the demo with ``--quick`` so CI exercises the full
baseline → search → held-out flow in <1 s without any LLM calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import training_signal_demo as demo  # noqa: E402


def test_run_demo_quick_writes_csv_and_md(tmp_path):
    summary = demo.run_demo(
        quick=True, n_trials=4, seed=0, out_dir=tmp_path,
    )
    csv_path = tmp_path / "training_signal_demo.csv"
    md_path = tmp_path / "training_signal_demo.md"
    assert csv_path.exists()
    assert md_path.exists()
    assert "Training-signal demo" in md_path.read_text()
    assert summary["best_params"]["damping"] is not None
    assert "elapsed_s" in summary


def test_parameterised_omp_runs_against_real_instance():
    from verifiable_labs_envs import load_environment
    env = load_environment("sparse-fourier-recovery", calibration_quantile=2.0)
    instance = env.generate_instance(seed=0)
    params = demo.SolverParams(damping=1.0, shrink_threshold=0.0)
    pred = demo.parameterised_omp_predict(instance, params)
    assert pred.x_hat.shape == (instance.n,)
    assert pred.support_hat is not None


def test_random_search_picks_best_by_val_mean(tmp_path):
    """Sanity-check that the search returns the highest-val candidate."""
    from verifiable_labs_envs import load_environment
    env = load_environment("sparse-fourier-recovery", calibration_quantile=2.0)
    best, history = demo.random_search(env, demo.SMOKE_VAL, n_trials=4, seed=0)
    means = [m for _, m in history]
    assert max(means) == max(m for p, m in history if p == best)


def test_paired_bootstrap_returns_zero_when_lengths_mismatch():
    delta, lo, hi = demo._paired_bootstrap_ci([1.0, 2.0], [3.0])
    assert (delta, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_detects_a_real_difference():
    a = [0.40, 0.42, 0.41, 0.43, 0.40]
    b = [v + 0.10 for v in a]
    delta, lo, hi = demo._paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert delta == 0.10
    assert lo > 0.0
