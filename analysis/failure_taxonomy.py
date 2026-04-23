#!/usr/bin/env python3
"""Classify LLM failures by type and emit a per-(model, env, category) count.

Categories (mutually exclusive, first match wins):

- `api_failure`            — transport / upstream error (HTTP 4xx/5xx, timeout).
- `parse_failure`          — valid response text but JSON extraction or schema validation failed.
- `support_error`          — sparse-Fourier only: point estimate is close (NMSE score ≥ 0.6) but the
                             support is mostly wrong (support F1 ≤ 0.2). The solver knows "there's
                             signal somewhere" but can't locate it.
- `magnitude_error`        — sparse-Fourier only: support largely correct (F1 ≥ 0.6) but NMSE score
                             low (≤ 0.3). The solver can locate the signal but gets the amplitudes
                             wrong.
- `over_smoothing`         — image envs (super-res, CT): SSIM meaningfully worse than PSNR on the
                             same instance (ssim_score < psnr_score − 0.15). The solver produced a
                             correct-ish global structure but blurred the edges.
- `ok`                     — none of the above; treated as a successful call for the purpose of this
                             taxonomy.

Output: `results/failure_taxonomy.csv` with columns
`model, env, category, count, share`.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPARSE_F = "sparse-fourier-recovery"
IMAGE_ENVS = ("super-resolution-div2k-x4", "lodopab-ct-simplified")

CATEGORIES = (
    "api_failure",
    "parse_failure",
    "support_error",
    "magnitude_error",
    "over_smoothing",
    "ok",
)

_COMPONENT_RE = re.compile(r"(\w+)=([0-9]+\.[0-9]+)")


def _parse_components(blob: str) -> dict[str, float]:
    return {m.group(1): float(m.group(2)) for m in _COMPONENT_RE.finditer(blob)}


def _categorize(row: dict[str, str]) -> str:
    mode = row.get("failure_mode", "")
    if mode == "api":
        return "api_failure"
    if mode in ("parse", "unexpected", "score"):
        return "parse_failure"
    # successful row — inspect components
    components = _parse_components(row.get("components", "") or "")
    env = row["env"]
    if env == SPARSE_F:
        nmse = components.get("nmse", 0.0)
        support = components.get("support", 0.0)
        if nmse >= 0.6 and support <= 0.2:
            return "support_error"
        if support >= 0.6 and nmse <= 0.3:
            return "magnitude_error"
    if env in IMAGE_ENVS:
        psnr = components.get("psnr", 0.0)
        ssim = components.get("ssim", 0.0)
        if ssim + 0.15 < psnr:
            return "over_smoothing"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=_PROJECT_ROOT / "results" / "llm_benchmark.csv")
    parser.add_argument("--out", type=Path,
                        default=_PROJECT_ROOT / "results" / "failure_taxonomy.csv")
    args = parser.parse_args()

    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)

    with args.csv.open() as fh:
        for row in csv.DictReader(fh):
            category = _categorize(row)
            counts[(row["model"], row["env"], category)] += 1
            totals[(row["model"], row["env"])] += 1

    # Pretty-print summary to stdout.
    print(f"{'model':40s}  {'env':28s}  {'category':18s}  count  share")
    print("-" * 110)
    rows = []
    for (model, env, category), count in sorted(counts.items()):
        total = totals[(model, env)]
        share = count / total if total else 0.0
        print(f"{model:40s}  {env:28s}  {category:18s}  {count:>5d}  {share:>5.2%}")
        rows.append({"model": model, "env": env, "category": category,
                     "count": count, "share": round(share, 4)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "env", "category", "count", "share"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
