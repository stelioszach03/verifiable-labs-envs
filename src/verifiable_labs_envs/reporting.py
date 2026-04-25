"""Run-driven Markdown report generator.

Consumed by ``verifiable report --run <jsonl> --out <md>``. Reads a
JSONL run file written by ``verifiable run``, computes summary
statistics, and emits a Markdown document with the 12 sections from
the YC-sprint brief.

The CSV-driven aggregate compliance report at
``scripts/generate_report.py`` is *separate*: that one consumes the
multi-model paper-final benchmark CSVs. This module consumes a single
agent's run on a single env.
"""
from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from verifiable_labs_envs.traces import FailureType, Trace

REQUIRED_SECTIONS = (
    "# ",                                 # title
    "## Run summary",                     # core stats
    "## Reward distribution",             # mean / std / hist
    "## Reward components",               # per-component averages
    "## Classical baseline",              # baseline mean + gap
    "## Failure modes",                   # parse failures + types
    "## Latency + cost",                  # timing / cost
    "## Best episodes",                   # top 3
    "## Worst episodes",                  # bottom 3
    "## Limitations",                     # honest caveats
    "## Recommended next actions",        # what to do
    "## Appendix",                        # source file + schema notes
)


def render_run_report(traces: Iterable[Trace], out_path: str | Path) -> Path:
    """Render the report and write it. Returns the output path."""
    traces_list = list(traces)
    if not traces_list:
        raise ValueError("render_run_report needs at least one trace")
    text = _render_markdown(traces_list)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# ── stats helpers ──────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _fmt_optional_float(v: float | None, fmt: str = "{:.3f}") -> str:
    return fmt.format(v) if v is not None else "—"


def _component_means(traces: list[Trace]) -> dict[str, float]:
    sums: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for t in traces:
        for k, v in (t.reward_components or {}).items():
            sums[k] += v
            counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def _failure_breakdown(traces: list[Trace]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for t in traces:
        if not t.parse_success:
            out[t.failure_type.value] += 1
    return dict(out.most_common())


# ── section writers ────────────────────────────────────────────


def _render_markdown(traces: list[Trace]) -> str:
    rewards = [t.reward for t in traces if t.parse_success]
    parse_ok = sum(1 for t in traces if t.parse_success)
    n = len(traces)
    env_name = traces[0].env_name
    agent_name = traces[0].agent_name
    model = traces[0].model_name
    baseline = [t.classical_baseline_reward for t in traces if t.classical_baseline_reward is not None]
    gaps = [t.gap_to_classical for t in traces if t.gap_to_classical is not None]
    coverages = [t.coverage for t in traces if t.coverage is not None]
    latencies = [t.latency_ms for t in traces if t.latency_ms is not None]
    costs = [t.estimated_cost_usd for t in traces if t.estimated_cost_usd is not None]
    parse_fail_pct = (1 - parse_ok / n) * 100 if n else 0.0

    parts: list[str] = []

    # Title.
    title = f"# Verifiable Labs run report — `{env_name}` × `{agent_name}`"
    parts.append(title)
    parts.append("")
    parts.append(f"_Generated {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}_")
    parts.append("")

    # 1. Run summary.
    parts.append("## Run summary")
    parts.append("")
    parts.append("| key | value |")
    parts.append("|---|---|")
    parts.append(f"| env | `{env_name}` |")
    parts.append(f"| agent | `{agent_name}` |")
    if model:
        parts.append(f"| model | `{model}` |")
    parts.append(f"| episodes (n) | {n} |")
    parts.append(f"| parse_ok | {parse_ok} / {n} ({parse_ok / n * 100:.0f} %)")
    parts.append(f"| schema_version | {traces[0].schema_version} |")
    parts.append("")

    # 2. Reward distribution.
    parts.append("## Reward distribution")
    parts.append("")
    if rewards:
        parts.append(f"- mean: **{_mean(rewards):.4f}**")
        parts.append(f"- std (population): {_std(rewards):.4f}")
        parts.append(f"- median: {statistics.median(rewards):.4f}")
        parts.append(f"- min / max: {min(rewards):.4f} / {max(rewards):.4f}")
    else:
        parts.append("_All episodes failed to parse; no reward distribution._")
    parts.append("")

    # 3. Reward components.
    parts.append("## Reward components")
    parts.append("")
    component_means = _component_means(traces)
    if component_means:
        parts.append("| component | mean |")
        parts.append("|---|---:|")
        for k in sorted(component_means):
            parts.append(f"| `{k}` | {component_means[k]:.4f} |")
    else:
        parts.append("_No reward-component decomposition available — env does not expose it, "
                     "or all episodes parse-failed._")
    parts.append("")

    # 4. Classical baseline.
    parts.append("## Classical baseline")
    parts.append("")
    if baseline:
        parts.append(f"- baseline mean reward: **{_mean(baseline):.4f}** (n={len(baseline)})")
        if gaps:
            parts.append(f"- mean gap (agent − baseline): **{_mean(gaps):+.4f}**")
            parts.append(f"- gap range: [{min(gaps):+.4f}, {max(gaps):+.4f}]")
        if coverages:
            parts.append(f"- mean conformal coverage: {_mean(coverages):.3f}")
    else:
        parts.append("_Classical baseline not recorded for this run._ "
                     "Re-run with `--with-baseline` to populate.")
    parts.append("")

    # 5. Failure modes.
    parts.append("## Failure modes")
    parts.append("")
    parts.append(f"- parse-failure rate: **{parse_fail_pct:.1f} %** "
                 f"({n - parse_ok} of {n})")
    breakdown = _failure_breakdown(traces)
    if breakdown:
        parts.append("- failure-type breakdown:")
        for ft, count in breakdown.items():
            parts.append(f"  - `{ft}`: {count}")
    else:
        parts.append("- no failures recorded — every episode produced a parseable prediction.")
    parts.append("")

    # 6. Latency + cost.
    parts.append("## Latency + cost")
    parts.append("")
    if latencies:
        parts.append(f"- mean latency: {_mean(latencies):.0f} ms")
        parts.append(f"- p95 latency: {_percentile(latencies, 95):.0f} ms")
        parts.append(f"- max latency: {max(latencies):.0f} ms")
    else:
        parts.append("- latency not recorded for this run.")
    if costs:
        parts.append(f"- total cost: ${sum(costs):.4f}")
        parts.append(f"- mean cost / episode: ${_mean(costs):.4f}")
    else:
        parts.append("- cost not recorded (agent did not report `estimated_cost_usd`).")
    parts.append("")

    # 7-8. Best / worst episodes.
    parts.append("## Best episodes")
    parts.append("")
    parts.extend(_episode_table(sorted(traces, key=lambda t: -t.reward)[:3]))
    parts.append("")

    parts.append("## Worst episodes")
    parts.append("")
    parts.extend(_episode_table(sorted(traces, key=lambda t: t.reward)[:3]))
    parts.append("")

    # 9. Limitations.
    parts.append("## Limitations")
    parts.append("")
    parts.append(f"- single-env, single-agent run on n={n} episodes. Generalisation "
                 "to other envs or model families is **not** asserted by this report.")
    parts.append("- v0.1.0-alpha. The platform makes no claim of regulatory compliance "
                 "with NIST AI RMF, EU AI Act, ISO 42001, or any other framework.")
    parts.append("- statistical precision on a single env scales with √n; treat the "
                 "mean reward above as a point estimate, not a verdict.")
    parts.append("")

    # 10. Recommended next actions.
    parts.append("## Recommended next actions")
    parts.append("")
    parts.extend(_recommendations(traces, parse_fail_pct, gaps))
    parts.append("")

    # 11. Appendix.
    parts.append("## Appendix")
    parts.append("")
    parts.append(f"- trace schema version: {traces[0].schema_version}")
    parts.append("- generated by `verifiable_labs_envs.reporting.render_run_report`")
    parts.append("- per-episode JSONL is the canonical artefact; this Markdown is a "
                 "human-readable summary.")
    parts.append("")
    return "\n".join(parts) + "\n"


def _episode_table(traces: list[Trace]) -> list[str]:
    if not traces:
        return ["_No episodes._"]
    lines = ["| seed | reward | parse_ok | failure | latency (ms) |",
             "|---:|---:|:---:|:---:|---:|"]
    for t in traces:
        seed = "—" if t.seed is None else str(t.seed)
        ok = "✓" if t.parse_success else "✗"
        ftype = t.failure_type.value if t.failure_type != FailureType.NONE else "—"
        latency = f"{t.latency_ms:.0f}" if t.latency_ms is not None else "—"
        lines.append(f"| {seed} | {t.reward:.3f} | {ok} | {ftype} | {latency} |")
    return lines


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def _recommendations(
    traces: list[Trace], parse_fail_pct: float, gaps: list[float],
) -> list[str]:
    out: list[str] = []
    if parse_fail_pct > 20:
        out.append(f"1. **High parse-fail rate ({parse_fail_pct:.1f} %) — fix output formatting.** "
                   "Inspect the agent's raw output for the failed episodes "
                   "(check the `metadata.error` field in each trace). Most parse "
                   "failures are markdown fences, prose preambles, or mis-typed "
                   "numeric fields.")
    elif parse_fail_pct > 5:
        out.append(f"1. Parse-fail rate is {parse_fail_pct:.1f} %; tighten the system "
                   "prompt or add an output schema reminder. <5 % is the v0.1 acceptance band.")
    if gaps and statistics.fmean(gaps) < -0.05:
        out.append(
            f"2. Agent under-performs the classical baseline by "
            f"{abs(statistics.fmean(gaps)):.3f} on average. Investigate whether "
            "the agent has access to the same problem inputs as the baseline; "
            "if so, this is a real capability gap."
        )
    if not gaps:
        out.append("3. Re-run with `--with-baseline` to populate the "
                   "`classical_baseline_reward` and `gap_to_classical` fields.")
    if not out:
        out.append("1. No critical issues flagged. Run on more seeds for tighter "
                   "confidence intervals, or compare against another agent with "
                   "`verifiable compare --runs <a> <b>`.")
    return out


__all__ = ["render_run_report", "REQUIRED_SECTIONS"]
