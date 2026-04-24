#!/usr/bin/env python3
"""Plot per-turn reward trajectories for a multi-turn benchmark CSV.

Reads ``results/multiturn_<env>.csv`` (written by ``benchmarks/run_multiturn_benchmark.py``)
and emits ``results/multiturn_<env>_curves.png`` with one line per model.

Each marker on a line is the mean reward at that turn across the model's
successful (parse_ok) episodes. Parse failures are dropped from the mean
but counted in an annotation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(csv_path: Path) -> tuple[dict, dict, dict]:
    """Returns (per_turn_rewards[model][turn] -> list[float],
                episode_finals[model] -> list[float],
                fail_counts[model] -> (parse_fail_count, total_episodes)."""
    per_turn: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    episode_last: dict[tuple[str, int], float] = {}
    episode_seen: dict[str, set[int]] = defaultdict(set)
    episode_failed: dict[str, set[int]] = defaultdict(set)

    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            try:
                turn = int(row["turn"])
                instance_id = int(row["instance_id"])
            except (TypeError, ValueError):
                continue
            parse_ok = row["parse_ok"] in ("True", "true", "1")
            episode_seen[model].add(instance_id)
            if not parse_ok:
                episode_failed[model].add(instance_id)
                continue
            try:
                reward = float(row["reward"])
            except (TypeError, ValueError):
                continue
            per_turn[model][turn].append(reward)
            episode_last[(model, instance_id)] = reward

    episode_finals: dict[str, list[float]] = defaultdict(list)
    for (m, _), r in episode_last.items():
        episode_finals[m].append(r)

    fail_counts = {m: (len(episode_failed[m]), len(episode_seen[m])) for m in episode_seen}
    return dict(per_turn), dict(episode_finals), fail_counts


def _short(name: str) -> str:
    return name.split("/")[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 2

    per_turn, episode_finals, fail_counts = _load(args.csv)
    if args.out is None:
        args.out = args.csv.with_suffix(".png").with_name(args.csv.stem + "_curves.png")

    models = sorted(per_turn)
    if not models:
        print("ERROR: no successful rows to plot", file=sys.stderr)
        return 3
    all_turns = sorted({t for m in models for t in per_turn[m]})

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.get_cmap("tab10").colors
    for i, model in enumerate(models):
        xs = []
        ys = []
        ns = []
        for t in all_turns:
            vals = per_turn[model].get(t, [])
            if vals:
                xs.append(t)
                ys.append(fmean(vals))
                ns.append(len(vals))
        label = f"{_short(model)}  (final mean {fmean(episode_finals[model]):.3f}"
        parse_fail, total = fail_counts.get(model, (0, 0))
        if parse_fail:
            label += f", {parse_fail}/{total} episode fail"
        label += ")"
        ax.plot(xs, ys, marker="o", color=colors[i % len(colors)], label=label)

    ax.set_xlabel("Turn index")
    ax.set_ylabel("Mean reward across successful episodes")
    ax.set_xticks(all_turns)
    ax.set_ylim(0.2, 0.55)
    ax.set_title(f"Multi-turn reward trajectory  ({args.csv.stem})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
