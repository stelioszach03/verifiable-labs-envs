#!/usr/bin/env python3
"""Merge all benchmark CSVs into a unified leaderboard file.

Reads:
- results/llm_benchmark_v2.csv           (Sprint 1 — 6 envs, single + multi + tools)
- results/phase_retrieval_v1_benchmark.csv (sprint-giga task 1)
- results/mri_knee_v1_benchmark.csv        (sprint-giga task 2)
- results/meta_benchmark_v3.csv            (sprint-giga task 5)

Writes: leaderboard/data/llm_benchmark_all.csv (schema-compatible with the
Gradio app's _load_final_turn helper; columns: env, model, seed, turn,
reward, parse_ok, + a source tag so the UI can show which sprint the row
came from).

Normalizes the row schema across the different per-sprint CSVs:
- v2 already has (seed, turn); pass through.
- v3 + phase + mri have (instance_id); rename → seed and set turn=1 for
  `variant=single`, turn=3 for `variant=multiturn` (we only keep the
  final-turn score on MT rows, matching the Gradio app's "final turn" logic).
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V2 = REPO / "results" / "llm_benchmark_v2.csv"
PHASE = REPO / "results" / "phase_retrieval_v1_benchmark.csv"
MRI = REPO / "results" / "mri_knee_v1_benchmark.csv"
META = REPO / "results" / "meta_benchmark_v3.csv"
OUT = REPO / "leaderboard" / "data" / "llm_benchmark_all.csv"

OUT_COLUMNS = [
    "source",      # "v2" | "phase-v1" | "mri-v1" | "meta-v3"
    "env",
    "model",
    "seed",
    "turn",
    "reward",
    "components",
    "parse_ok",
    "usd_cost",
]


def _row(source: str, env: str, model: str, seed: int | str,
         turn: int, reward, components: str, parse_ok: bool, cost) -> dict:
    return {
        "source": source,
        "env": env,
        "model": model,
        "seed": str(seed),
        "turn": int(turn),
        "reward": "" if reward is None or reward == "" else f"{float(reward):.4f}",
        "components": components or "",
        "parse_ok": "True" if parse_ok else "False",
        "usd_cost": "" if cost in (None, "") else f"{float(cost):.6f}",
    }


def main() -> None:
    rows_out: list[dict] = []

    # v2 — already has seed + turn
    if V2.exists():
        with V2.open() as fh:
            for r in csv.DictReader(fh):
                rows_out.append(_row(
                    source="v2",
                    env=r["env"],
                    model=r["model"],
                    seed=r["seed"],
                    turn=int(r["turn"]) if r["turn"] else 1,
                    reward=r["reward"] or None,
                    components=r.get("components", ""),
                    parse_ok=(r.get("parse_ok") == "True"),
                    cost=r.get("usd_cost"),
                ))

    # phase / mri — (variant ∈ {single, multiturn}), instance_id → seed
    for src_label, csv_path in [("phase-v1", PHASE), ("mri-v1", MRI)]:
        if not csv_path.exists():
            continue
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                variant = r.get("variant", "single")
                turn = 3 if variant == "multiturn" else 1
                rows_out.append(_row(
                    source=src_label,
                    env=r["env"],
                    model=r["model"],
                    seed=r["instance_id"],
                    turn=turn,
                    reward=r["reward"] or None,
                    components=r.get("components", ""),
                    parse_ok=(r.get("parse_ok") == "True"),
                    cost=r.get("usd_cost"),
                ))

    # meta-v3 — single-turn only, instance_id → seed
    if META.exists():
        with META.open() as fh:
            for r in csv.DictReader(fh):
                rows_out.append(_row(
                    source="meta-v3",
                    env=r["env"],
                    model=r["model"],
                    seed=r["instance_id"],
                    turn=1,
                    reward=r["reward"] or None,
                    components=r.get("components", ""),
                    parse_ok=(r.get("parse_ok") == "True"),
                    cost=r.get("usd_cost"),
                ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows_out)

    print(f"wrote {len(rows_out)} rows → {OUT.relative_to(REPO)}")
    # Summary
    by_source: dict[str, int] = {}
    by_env: dict[str, int] = {}
    for r in rows_out:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_env[r["env"]] = by_env.get(r["env"], 0) + 1
    print("by source:", dict(sorted(by_source.items())))
    print("by env:   ", dict(sorted(by_env.items())))


if __name__ == "__main__":
    main()
