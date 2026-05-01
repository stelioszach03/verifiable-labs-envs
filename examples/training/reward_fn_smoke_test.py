"""Smoke demo for ``make_reward_fn``.

Builds 10 dummy completions × 5 instances (seeds 0–4) for the
``sparse-fourier-recovery`` env and prints a per-call components table
plus an aggregate summary. CPU-only — no LLM, no GPU.

The 10 completion strategies cover the three failure modes plus several
"good" cases of varying quality, exercising every branch of
``reward_fn``'s three-stage gating.

Usage:
    python examples/training/reward_fn_smoke_test.py
"""
from __future__ import annotations

import json

import numpy as np

from verifiable_labs_envs.training import make_reward_fn

ENV_ID = "sparse-fourier-recovery"
N_INSTANCES = 5
SEEDS = list(range(N_INSTANCES))


def _strategy_zero() -> str:
    return json.dumps({"support_idx": list(range(10)), "support_amp_x1000": [0] * 10})


def _strategy_ones() -> str:
    return json.dumps(
        {"support_idx": list(range(10)), "support_amp_x1000": [1000] * 10}
    )


def _strategy_random_amps(seed: int) -> str:
    rng = np.random.default_rng(seed)
    amps = (rng.standard_normal(10) * 1000).round().astype(int).tolist()
    return json.dumps({"support_idx": list(range(10)), "support_amp_x1000": amps})


def _strategy_oracle(seed: int) -> str:
    """Cheat: regenerate the instance, read x_true, build the perfect completion.

    Used here only to verify the reward function gives near-1.0 reward
    when fed the ground truth.
    """
    from verifiable_labs_envs import load_environment

    env = load_environment(ENV_ID, calibration_quantile=2.0)
    inst = env.generate_instance(seed=seed)
    support = sorted(int(i) for i in inst.support_true)
    # Read the amplitudes at those support positions from x_true.
    amps = [int(round(float(inst.x_true[i]) * 1000)) for i in support]
    return json.dumps({"support_idx": support, "support_amp_x1000": amps})


def _strategy_garbage() -> str:
    return "this is not json at all, just words"


def _strategy_empty() -> str:
    return ""


def _strategy_wrong_keys() -> str:
    return json.dumps({"foo": "bar", "answer": [1, 2, 3]})


def _strategy_wrong_length() -> str:
    return json.dumps({"support_idx": [0, 1, 2], "support_amp_x1000": [100, 200, 300]})


def _strategy_out_of_range() -> str:
    return json.dumps({"support_idx": [9999] * 10, "support_amp_x1000": [100] * 10})


def _strategy_fenced_json() -> str:
    """Wrapped in markdown fences — adapter should still accept."""
    inner = json.dumps(
        {"support_idx": list(range(10)), "support_amp_x1000": [500] * 10}
    )
    return f"Sure, here is my answer:\n```json\n{inner}\n```"


def _build_completions_grid() -> list[tuple[str, str, int]]:
    """Return list of (strategy_label, completion_text, seed) tuples.

    10 strategies × 5 instances = 50 rows. Strategies exercise every
    failure path plus "good" cases.
    """
    strategies = [
        ("zero", lambda s: _strategy_zero()),
        ("ones", lambda s: _strategy_ones()),
        ("random_amps", _strategy_random_amps),
        ("oracle", _strategy_oracle),
        ("garbage", lambda s: _strategy_garbage()),
        ("empty", lambda s: _strategy_empty()),
        ("wrong_keys", lambda s: _strategy_wrong_keys()),
        ("wrong_length", lambda s: _strategy_wrong_length()),
        ("out_of_range", lambda s: _strategy_out_of_range()),
        ("fenced_json", lambda s: _strategy_fenced_json()),
    ]
    rows: list[tuple[str, str, int]] = []
    for label, gen in strategies:
        for seed in SEEDS:
            rows.append((label, gen(seed), seed))
    return rows


def main() -> None:
    rows = _build_completions_grid()
    completions = [c for _, c, _ in rows]
    seeds = [s for _, _, s in rows]
    labels = [line for line, _, _ in rows]

    reward_fn = make_reward_fn(ENV_ID)
    rewards = reward_fn(
        prompts=[""] * len(rows),
        completions=completions,
        instance_seed=seeds,
    )

    # Per-call table grouped by strategy.
    print(
        f"{'strategy':<14}{'seed':>5}{'reward':>10}{'pv':>5}{'fv':>5}{'nmse':>9}{'support':>10}{'conformal':>11}"
    )
    print("-" * 73)
    for (label, _, seed), reward, rec in zip(
        rows, rewards, reward_fn.stats.per_call, strict=True
    ):
        c = rec["components"]
        print(
            f"{label:<14}{seed:>5}{reward:>10.4f}{int(c['parse_valid']):>5}"
            f"{int(c['format_valid']):>5}{c['nmse']:>9.4f}"
            f"{c['support']:>10.4f}{c['conformal']:>11.4f}"
        )

    # Aggregate per strategy (helps eyeball that "oracle" >> "zero" >> "wrong_*").
    print()
    print(f"{'strategy':<14}{'mean_reward':>14}{'parse_ok%':>11}{'format_ok%':>12}")
    print("-" * 51)
    for label in dict.fromkeys(labels):
        per = [
            (rec["reward"], rec["components"]["parse_valid"], rec["components"]["format_valid"])
            for rec, line in zip(reward_fn.stats.per_call, labels, strict=True)
            if line == label
        ]
        n = len(per)
        mean_r = sum(p[0] for p in per) / n
        pv_pct = 100.0 * sum(p[1] for p in per) / n
        fv_pct = 100.0 * sum(p[2] for p in per) / n
        print(f"{label:<14}{mean_r:>14.4f}{pv_pct:>10.0f}%{fv_pct:>11.0f}%")

    # Final aggregate.
    agg = reward_fn.stats.aggregate()
    print()
    print("=== aggregate over all 50 calls ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
