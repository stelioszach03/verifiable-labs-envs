#!/usr/bin/env python3
"""Generate a compliance report from a Verifiable Labs benchmark CSV.

Reads a per-episode benchmark CSV (one row per (env, model, seed,
turn) combination) and emits a Markdown report — and, optionally, a
PDF — using ``templates/compliance-report/report_template.md``.

The template uses ``string.Template`` ($var) substitution. There are
no Jinja-style conditionals; all branching lives in this script.

Usage::

    # Produce Markdown only:
    python scripts/generate_report.py \\
        --benchmark-csv results/complete_matrix_single_turn.csv \\
        --model anthropic/claude-haiku-4.5 \\
        --output report.md

    # Produce Markdown + PDF (needs pandoc OR weasyprint):
    python scripts/generate_report.py \\
        --benchmark-csv results/complete_matrix_single_turn.csv \\
        --model anthropic/claude-haiku-4.5 \\
        --output report.md --pdf report.pdf

The script exits non-zero on:
- missing CSV file
- model not present in the CSV
- empty filtered dataset (no rows survive after model + parse filters)
- PDF requested but neither pandoc nor weasyprint is available
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import statistics
import string
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "templates" / "compliance-report" / "report_template.md"
DEFAULT_ALPHA = 0.10
DEFAULT_TARGET_COVERAGE = 0.90


# ── data model ────────────────────────────────────────────


@dataclass
class Episode:
    """One scored (env, seed, turn) episode for a single model."""
    env: str
    model: str
    seed: int
    turn: int
    reward: float
    parse_ok: bool
    coverage: float | None  # parsed from components string when present


def _parse_components_str(s: str) -> dict[str, float]:
    """Parse the platform's ``components`` column shape, e.g.
    ``'nmse=0.135, support=0.200, conformal=0.900'``."""
    out: dict[str, float] = {}
    if not s or s == "{}":
        return out
    for part in s.split(","):
        if "=" not in part:
            continue
        k, _, v = part.strip().partition("=")
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def load_episodes(csv_path: Path, model: str) -> list[Episode]:
    """Load all episodes for ``model`` from a benchmark CSV.

    Filters to the latest-turn-per-(env, seed) row (so multi-turn envs
    contribute one final reward per episode, not one per turn).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"benchmark CSV not found: {csv_path}")
    rows: list[dict] = []
    with csv_path.open() as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        raise ValueError(f"benchmark CSV is empty: {csv_path}")

    filtered = [r for r in rows if r.get("model") == model]
    if not filtered:
        models = sorted({r.get("model", "") for r in rows if r.get("model")})
        raise ValueError(
            f"model {model!r} not found in {csv_path}; available: {models}"
        )

    # Group by (env, seed) and keep only the highest-turn row.
    by_episode: dict[tuple[str, int], dict] = {}
    for r in filtered:
        try:
            seed = int(r["seed"])
            turn = int(r.get("turn", 1))
        except (KeyError, ValueError):
            continue
        key = (r["env"], seed)
        existing = by_episode.get(key)
        if existing is None or int(existing.get("turn", 1)) < turn:
            by_episode[key] = r

    episodes: list[Episode] = []
    for r in by_episode.values():
        comps = _parse_components_str(r.get("components", "") or "")
        try:
            reward = float(r.get("reward", 0.0) or 0.0)
        except ValueError:
            reward = 0.0
        parse_ok = str(r.get("parse_ok", "True")).strip().lower() in {"true", "1"}
        episodes.append(Episode(
            env=r["env"],
            model=r["model"],
            seed=int(r["seed"]),
            turn=int(r.get("turn", 1)),
            reward=reward,
            parse_ok=parse_ok,
            coverage=comps.get("conformal"),
        ))
    if not episodes:
        raise ValueError(f"No usable episodes for model {model!r} after filtering")
    return episodes


# ── aggregation ───────────────────────────────────────────


@dataclass
class EnvStats:
    env: str
    n: int
    mean_reward: float
    std_reward: float
    parse_fail_rate: float
    coverage: float | None  # mean of per-episode conformal scores when available


def per_env_stats(episodes: list[Episode]) -> list[EnvStats]:
    by_env: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_env[ep.env].append(ep)
    out: list[EnvStats] = []
    for env, eps in sorted(by_env.items()):
        rewards = [e.reward for e in eps]
        parse_fail = sum(1 for e in eps if not e.parse_ok) / len(eps)
        cov = [e.coverage for e in eps if e.coverage is not None]
        out.append(EnvStats(
            env=env,
            n=len(eps),
            mean_reward=statistics.fmean(rewards),
            std_reward=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            parse_fail_rate=parse_fail,
            coverage=(statistics.fmean(cov) if cov else None),
        ))
    return out


def make_per_env_table(stats: list[EnvStats]) -> str:
    """Markdown table — one row per env, sorted by mean reward (descending)."""
    rows = sorted(stats, key=lambda s: -s.mean_reward)
    lines = [
        "| env | n | mean reward | std | parse-fail | coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in rows:
        cov = f"{s.coverage * 100:.1f} %" if s.coverage is not None else "—"
        lines.append(
            f"| `{s.env}` | {s.n} | {s.mean_reward:.3f} | {s.std_reward:.3f} | "
            f"{s.parse_fail_rate * 100:.1f} % | {cov} |"
        )
    return "\n".join(lines)


def make_low_reward_table(stats: list[EnvStats], threshold: float = 0.3) -> str:
    rows = [s for s in sorted(stats, key=lambda s: s.mean_reward) if s.mean_reward < threshold]
    if not rows:
        return f"_No env scored below {threshold:.2f}._"
    lines = [
        f"Envs with mean reward below **{threshold:.2f}** (suggesting structural failure):",
        "",
        "| env | mean reward | parse-fail |",
        "|---|---:|---:|",
    ]
    for s in rows:
        lines.append(
            f"| `{s.env}` | {s.mean_reward:.3f} | {s.parse_fail_rate * 100:.1f} % |"
        )
    return "\n".join(lines)


def make_env_list(stats: list[EnvStats]) -> str:
    return "\n".join(f"- `{s.env}` ({s.n} episodes)" for s in stats)


def domains_summary(envs: list[str]) -> str:
    """Best-effort domain inference from env names."""
    keywords = {
        "compressed-sensing": ["sparse-fourier"],
        "imaging": ["lodopab", "mri", "super-resolution"],
        "physics-inverse": ["phase-retrieval", "tomography", "fwi"],
        "rendering-inverse": ["inverse-rendering", "holographic"],
    }
    matched = []
    for domain, kws in keywords.items():
        if any(any(kw in e for e in envs) for kw in kws):
            matched.append(domain)
    if not matched:
        return "scientific reasoning"
    if len(matched) == 1:
        return matched[0]
    return ", ".join(matched[:-1]) + ", and " + matched[-1]


def headline_findings(stats: list[EnvStats], overall_mean: float, parse_fail_rate: float) -> str:
    bullets: list[str] = []
    best = max(stats, key=lambda s: s.mean_reward)
    worst = min(stats, key=lambda s: s.mean_reward)
    bullets.append(
        f"- **Strongest env:** `{best.env}` (mean reward {best.mean_reward:.3f}, "
        f"n = {best.n})."
    )
    bullets.append(
        f"- **Weakest env:** `{worst.env}` (mean reward {worst.mean_reward:.3f}, "
        f"n = {worst.n})."
    )
    if parse_fail_rate > 0.10:
        bullets.append(
            f"- **High parse-fail rate ({parse_fail_rate * 100:.1f} %)** — formatting "
            f"brittleness; consider tightening the system prompt."
        )
    elif parse_fail_rate > 0.0:
        bullets.append(
            f"- **Parse-fail rate {parse_fail_rate * 100:.1f} %** — within v0.1 "
            f"acceptance band (<5 %)."
        )
    spread = max(s.mean_reward for s in stats) - min(s.mean_reward for s in stats)
    if spread > 0.30:
        bullets.append(
            f"- **Capability spread of {spread:.2f}** across envs — model is "
            f"specialised; cross-env transfer is poor."
        )
    overall_band = "above" if overall_mean >= 0.50 else "at" if overall_mean >= 0.30 else "below"
    bullets.append(f"- **Aggregate mean {overall_mean:.3f}** — {overall_band} v0.1 baseline.")
    return "\n".join(bullets)


def recommendations(
    stats: list[EnvStats],
    overall_mean: float,
    parse_fail_rate: float,
    coverage_pct: float | None,
    target_coverage_pct: float,
) -> str:
    bullets: list[str] = []
    if parse_fail_rate > 0.05:
        bullets.append(
            "1. **Tighten output formatting.** Parse-fail rate exceeds 5 %; rewrite "
            "the system prompt to forbid markdown fences and emit a strict JSON "
            "schema reminder. The Tier-1 SDK exposes `client.env(...).adapter` to "
            "inspect the canonical prompt."
        )
    if coverage_pct is not None and abs(coverage_pct - target_coverage_pct) > 10:
        direction = "under" if coverage_pct < target_coverage_pct else "over"
        bullets.append(
            f"2. **Recalibrate uncertainty.** Empirical coverage is {direction}-target "
            f"by {abs(coverage_pct - target_coverage_pct):.1f} pp. The model's "
            f"stated uncertainty bounds are unreliable for risk decisions."
        )
    weak = [s for s in stats if s.mean_reward < 0.30]
    if weak:
        names = ", ".join(f"`{s.env}`" for s in weak)
        bullets.append(
            f"3. **Inspect failures on {names}.** Mean reward below 0.30 typically "
            f"indicates a structural misunderstanding of the problem, not a "
            f"prompt-engineering issue. Pull representative seeds from the per-"
            f"episode CSV and review the model's reasoning."
        )
    if overall_mean >= 0.50 and parse_fail_rate < 0.05 and (
        coverage_pct is None or abs(coverage_pct - target_coverage_pct) <= 10
    ):
        bullets.append(
            "4. **Suitable for downstream evaluation.** The model meets v0.1 "
            "acceptance bands (mean ≥ 0.50, parse-fail < 5 %, coverage within "
            "±10 pp of target). Pair with a domain-specific test set before "
            "production sign-off."
        )
    if not bullets:
        bullets.append(
            "1. **No critical issues flagged.** Aggregate metrics fall within v0.1 "
            "acceptance bands."
        )
    return "\n".join(bullets)


def recommended_next_step(
    overall_mean: float,
    parse_fail_rate: float,
) -> str:
    if parse_fail_rate > 0.10:
        return (
            "Address parse failures first — formatting issues mask capability. "
            "Then re-run this report against the same CSV column."
        )
    if overall_mean < 0.30:
        return (
            "Manually review the lowest-reward env to determine whether the "
            "failure is structural (capability gap) or surface-level (prompt-"
            "engineering)."
        )
    if overall_mean < 0.50:
        return (
            "Compare against peer models on the same CSV; isolated low scores "
            "may reflect env difficulty rather than model weakness."
        )
    return (
        "Pair this aggregate report with a domain-specific test set before "
        "production sign-off. The platform makes no claim of regulatory "
        "compliance on its own."
    )


# ── PDF rendering ─────────────────────────────────────────


def render_pdf(md_path: Path, pdf_path: Path) -> str:
    """Render Markdown to PDF using pandoc; fall back to weasyprint.

    Returns the name of the renderer used. Raises ``RuntimeError`` if
    neither is available.
    """
    if shutil.which("pandoc"):
        # Pandoc with pdf-engine=tectonic if available, otherwise pdflatex.
        engine_args: list[str] = []
        for engine in ("tectonic", "xelatex", "pdflatex"):
            if shutil.which(engine):
                engine_args = [f"--pdf-engine={engine}"]
                break
        cmd = ["pandoc", str(md_path), "-o", str(pdf_path), *engine_args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"pandoc failed: {result.stderr}")
        return "pandoc"

    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "neither pandoc nor weasyprint is available; install one to render PDF"
        ) from e

    import markdown
    html_text = markdown.markdown(md_path.read_text(), extensions=["tables"])
    HTML(string=html_text).write_pdf(str(pdf_path))
    return "weasyprint"


# ── main ──────────────────────────────────────────────────


def render_report(
    *,
    benchmark_csv: Path,
    model: str,
    output_md: Path,
    output_pdf: Path | None = None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    alpha: float = DEFAULT_ALPHA,
    template_path: Path = TEMPLATE_PATH,
) -> dict[str, str]:
    """Render the report, returning the substitution dict for inspection."""
    episodes = load_episodes(benchmark_csv, model)
    stats = per_env_stats(episodes)

    rewards = [e.reward for e in episodes]
    parse_fail_rate = sum(1 for e in episodes if not e.parse_ok) / len(episodes)
    coverages = [e.coverage for e in episodes if e.coverage is not None]
    overall_coverage = statistics.fmean(coverages) if coverages else None

    over_count = sum(
        1 for s in stats
        if s.coverage is not None and s.coverage > target_coverage + 0.05
    )
    under_count = sum(
        1 for s in stats
        if s.coverage is not None and s.coverage < target_coverage - 0.05
    )

    seeds_per_env = Counter(e.env for e in episodes).most_common(1)[0][1]

    today = dt.date.today().isoformat()
    iso_now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    subs: dict[str, str] = {
        "model": model,
        "date": today,
        "date_iso": iso_now,
        "benchmark_csv_basename": benchmark_csv.name,
        "n_envs": str(len(stats)),
        "n_seeds_total": str(len(episodes)),
        "seeds_per_env": str(seeds_per_env),
        "domains": domains_summary([s.env for s in stats]),
        "mean_reward": f"{statistics.fmean(rewards):.3f}",
        "median_reward": f"{statistics.median(rewards):.3f}",
        "std_reward": f"{statistics.pstdev(rewards) if len(rewards) > 1 else 0:.3f}",
        "min_reward": f"{min(rewards):.3f}",
        "max_reward": f"{max(rewards):.3f}",
        "parse_fail_count": str(sum(1 for e in episodes if not e.parse_ok)),
        "parse_fail_rate_pct": f"{parse_fail_rate * 100:.1f}",
        "coverage_pct": (
            f"{overall_coverage * 100:.1f}" if overall_coverage is not None else "n/a"
        ),
        "target_coverage_pct": f"{target_coverage * 100:.0f}",
        "alpha": f"{alpha:.2f}",
        "over_coverage_count": str(over_count),
        "under_coverage_count": str(under_count),
        "per_env_table": make_per_env_table(stats),
        "low_reward_envs_table": make_low_reward_table(stats),
        "env_list": make_env_list(stats),
        "headline_findings": headline_findings(
            stats, statistics.fmean(rewards), parse_fail_rate,
        ),
        "recommendations": recommendations(
            stats,
            statistics.fmean(rewards),
            parse_fail_rate,
            (overall_coverage * 100 if overall_coverage is not None else None),
            target_coverage * 100,
        ),
        "recommended_next_step": recommended_next_step(
            statistics.fmean(rewards), parse_fail_rate,
        ),
    }
    template_text = template_path.read_text()
    rendered = string.Template(template_text).safe_substitute(subs)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(rendered)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        render_pdf(output_md, output_pdf)
    return subs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-csv", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Output Markdown path")
    ap.add_argument("--pdf", type=Path, default=None, help="Optional PDF path")
    ap.add_argument(
        "--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE,
        help="Conformal coverage target (default 0.90)",
    )
    ap.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Calibration miscoverage budget (default 0.10)",
    )
    args = ap.parse_args(argv)

    try:
        render_report(
            benchmark_csv=args.benchmark_csv,
            model=args.model,
            output_md=args.output,
            output_pdf=args.pdf,
            target_coverage=args.target_coverage,
            alpha=args.alpha,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}", file=sys.stderr)
    if args.pdf:
        print(f"Wrote {args.pdf}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
