"""Leaderboard aggregation from existing benchmark CSVs.

Reads the per-(env, model, seed) benchmark CSVs already in ``results/``
and computes per-model summary stats for a given env. The same set of
CSVs the paper figure-generator uses; the loader is tolerant to both
v2 schema (``seed`` + ``turn``) and giga-sprint schema (``instance_id``).

This is read-only at request time — no LLM calls, no recomputation.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

# Resolve REPO root once at module load — assume this file lives at
# ``<repo>/src/verifiable_labs_api/leaderboard.py``.
REPO = Path(__file__).resolve().parent.parent.parent
_RESULTS = REPO / "results"

# All benchmark CSVs we know how to consume. Missing files are tolerated.
_BENCHMARK_CSVS: tuple[Path, ...] = (
    _RESULTS / "llm_benchmark_v2.csv",
    _RESULTS / "meta_benchmark_v3.csv",
    _RESULTS / "phase_retrieval_v1_benchmark.csv",
    _RESULTS / "mri_knee_v1_benchmark.csv",
    _RESULTS / "opus_nano_fill_v2.csv",
    _RESULTS / "complete_matrix_single_turn.csv",
    _RESULTS / "complete_matrix_multi_turn.csv",
    _RESULTS / "tools_v2_complete.csv",
)


def _row_seed(row: dict[str, str]) -> int | None:
    raw = row.get("seed") or row.get("instance_id")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _row_turn(row: dict[str, str]) -> int:
    raw = row.get("turn")
    try:
        return int(raw) if raw else 1
    except (TypeError, ValueError):
        return 1


def _final_turn_rewards(env_id: str) -> tuple[
    dict[str, list[float]],          # model → list of final-turn rewards
    dict[str, int],                  # model → total attempted episodes
    list[str],                       # source CSVs that contributed rows
]:
    """Aggregate final-turn rewards per model for ``env_id``.

    Tolerates the qualified form ``stelioszach/sparse-fourier-recovery``
    by stripping the prefix.
    """
    bare = env_id.split("/", 1)[-1]

    # (model, seed) → (best_turn_seen, reward, parse_ok)
    best: dict[tuple[str, int], tuple[int, float | None, bool]] = {}
    attempts: dict[str, int] = defaultdict(int)
    contributing: list[str] = []

    for path in _BENCHMARK_CSVS:
        if not path.exists():
            continue
        added_from_this = False
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("env") != bare:
                    continue
                model = row.get("model", "")
                if not model:
                    continue
                added_from_this = True
                seed = _row_seed(row)
                if seed is None:
                    continue
                turn = _row_turn(row)
                parse_ok = row.get("parse_ok") == "True"
                reward: float | None
                try:
                    reward = float(row["reward"]) if parse_ok else None
                except (TypeError, ValueError):
                    reward = None
                key = (model, seed)
                # Each (model, seed) counted once towards attempts on its
                # earliest appearance. Final-turn reward keeps the
                # highest-turn parse-ok row.
                if key not in best:
                    attempts[model] += 1
                cur_turn, cur_reward, cur_ok = best.get(key, (-1, None, False))
                if turn > cur_turn or (turn == cur_turn and parse_ok and not cur_ok):
                    best[key] = (turn, reward, parse_ok)
        if added_from_this:
            contributing.append(path.name)

    rewards: dict[str, list[float]] = defaultdict(list)
    for (model, _seed), (_turn, reward, parse_ok) in best.items():
        if parse_ok and reward is not None:
            rewards[model].append(reward)

    return dict(rewards), dict(attempts), contributing


def aggregate_for_env(env_id: str) -> dict:
    """Return ``{"env_id": ..., "rows": [...], "sources": [...]}``.

    Each row is ``{model, n, mean_reward, std_reward, parse_fail_rate}``,
    sorted by ``mean_reward`` descending.
    """
    rewards, attempts, sources = _final_turn_rewards(env_id)
    rows = []
    for model in sorted(set(rewards) | set(attempts)):
        n_parsed = len(rewards.get(model, []))
        n_total = attempts.get(model, n_parsed)
        if n_parsed == 0:
            mean = 0.0
            std = 0.0
        else:
            vals = rewards[model]
            mean = float(fmean(vals))
            std = float(pstdev(vals)) if len(vals) > 1 else 0.0
        parse_fail = 0.0 if n_total == 0 else 1.0 - n_parsed / n_total
        rows.append({
            "model": model,
            "n": n_parsed,
            "mean_reward": round(mean, 4),
            "std_reward": round(std, 4),
            "parse_fail_rate": round(parse_fail, 4),
        })
    rows.sort(key=lambda r: -r["mean_reward"])
    return {
        "env_id": env_id,
        "rows": rows,
        "sources": sources,
    }
