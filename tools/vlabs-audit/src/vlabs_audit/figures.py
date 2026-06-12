"""Matplotlib PDF figures for the LaTeX audit report.

Produces four figure types from a populated :class:`vlabs_audit.stats.AuditStats`:

* ``reward_distribution_<env>.pdf`` — one per env: histogram + 95 % CI band
  + mean line.
* ``coverage_calibration.pdf`` — empirical coverage vs target per env, bars
  coloured green when at-or-above target, red below.
* ``score_breakdown.pdf`` — stacked horizontal bar (format pass / format
  fail / parse error) per env.
* ``cost_per_correct.pdf`` — USD per successful episode, log-scaled when
  the per-env range exceeds 100×. Renders a "no cost data" plate when
  the underlying traces don't carry ``estimated_cost_usd``.

Styling is deliberately minimal: serif font (matches a LaTeX body), no
colour gradients, no chartjunk. Output is always vector PDF, sized to
the LaTeX default text width (6.5 in), saved with metadata
``Subject = "audit_id:<id>"`` so the embedded provenance survives the
upload-to-customer round-trip.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Use the non-interactive backend BEFORE pyplot imports — figures must
# render in headless CI / SSH environments without a display.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 — backend must be set first
import numpy as np  # noqa: E402

from vlabs_audit.stats import AuditStats, EnvStats  # noqa: E402

# ── styling ──────────────────────────────────────────────────────────

_FIGURE_WIDTH_IN = 6.5  # LaTeX default \textwidth in inches
_FIGURE_HEIGHT_IN = 4.0
_FONT_FAMILY = "serif"
_FONT_SIZE = 10

_COLOR_PASS = "#2ca02c"      # green
_COLOR_WARN = "#ffbb33"      # amber
_COLOR_FAIL = "#d62728"      # red
_COLOR_NEUTRAL = "#1f77b4"   # blue
_COLOR_MISSING = "#9aa1a8"   # grey for unknown/no-data bars


# ── helpers ──────────────────────────────────────────────────────────


def _new_axes(
    *,
    width: float = _FIGURE_WIDTH_IN,
    height: float = _FIGURE_HEIGHT_IN,
) -> tuple[plt.Figure, plt.Axes]:
    """Construct a fresh ``(figure, axes)`` with shared styling applied."""
    plt.rc("font", family=_FONT_FAMILY, size=_FONT_SIZE)
    plt.rc("axes", titlesize=_FONT_SIZE + 1, labelsize=_FONT_SIZE)
    plt.rc("xtick", labelsize=_FONT_SIZE - 1)
    plt.rc("ytick", labelsize=_FONT_SIZE - 1)
    plt.rc("legend", fontsize=_FONT_SIZE - 1)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save_pdf(
    fig: plt.Figure,
    path: Path,
    *,
    audit_id: str,
    title: str,
) -> Path:
    """Write a PDF with embedded metadata and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "vlabs-audit",
            "Subject": f"audit_id:{audit_id}",
            "Keywords": "verifiable-labs,audit,calibration,conformal-prediction",
        },
    )
    plt.close(fig)
    return path


def _slugify(name: str) -> str:
    """Filename-safe slug: replace path-unfriendly chars."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _coverage_color(coverage: float | None, target: float) -> str:
    """Bar colour for an empirical coverage value vs ``target``.

    * Missing or NaN → neutral grey (we cannot judge calibration).
    * ``coverage >= target`` → green (calibrated or over-coverage).
    * ``coverage < target`` → red (under-coverage = miscalibrated).
    """
    if coverage is None:
        return _COLOR_MISSING
    try:
        c = float(coverage)
    except (TypeError, ValueError):
        return _COLOR_MISSING
    if np.isnan(c):
        return _COLOR_MISSING
    return _COLOR_PASS if c >= target else _COLOR_FAIL


def _render_text_summary(ax: plt.Axes, lines: list[tuple[str, str]]) -> None:
    """Render ``[(label, value), ...]`` as a centred two-column text card.

    Used by single-env audits (and zero-env edge cases) where a single
    bar / single stacked-row chart would look broken.
    """
    ax.set_axis_off()
    if not lines:
        ax.text(
            0.5, 0.5, "no data",
            ha="center", va="center", transform=ax.transAxes,
            color=_COLOR_MISSING, fontsize=_FONT_SIZE + 1,
        )
        return

    n = len(lines)
    # Vertical layout: rows packed in the middle 80 % of the axes.
    top, bottom = 0.85, 0.15
    ys = (
        [0.5] if n == 1
        else [top - i * (top - bottom) / (n - 1) for i in range(n)]
    )

    for (label, value), y in zip(lines, ys, strict=True):
        ax.text(
            0.45, y, label,
            ha="right", va="center", transform=ax.transAxes,
            color=_COLOR_MISSING, fontsize=_FONT_SIZE + 1,
        )
        ax.text(
            0.55, y, value,
            ha="left", va="center", transform=ax.transAxes,
            color="black", fontsize=_FONT_SIZE + 2, fontweight="bold",
        )


# ── individual figures ──────────────────────────────────────────────


def reward_distribution(
    env_stats: EnvStats,
    *,
    audit_id: str,
    out: Path,
) -> Path:
    """Histogram of per-episode rewards with mean + 95 % CI band."""
    rewards = list(env_stats.rewards)
    fig, ax = _new_axes()
    title = f"Reward Distribution — {env_stats.env}"

    if not rewards:
        ax.text(
            0.5, 0.5,
            "no successful episodes",
            ha="center", va="center", transform=ax.transAxes,
            color=_COLOR_MISSING, fontsize=_FONT_SIZE + 1,
        )
        ax.set_axis_off()
        ax.set_title(title)
        return _save_pdf(fig, out, audit_id=audit_id, title=title)

    bins = max(5, min(20, len(rewards) // 2 + 1))
    ax.hist(
        rewards, bins=bins, density=True,
        alpha=0.55, color=_COLOR_NEUTRAL, edgecolor="white",
    )
    ax.axvspan(
        env_stats.ci_low, env_stats.ci_high,
        color="0.85", zorder=0,
        label=f"95% CI [{env_stats.ci_low:.3f}, {env_stats.ci_high:.3f}]",
    )
    ax.axvline(
        env_stats.mean_reward, color="black", linestyle="--", linewidth=1.2,
        label=f"mean = {env_stats.mean_reward:.3f}",
    )
    ax.set_xlabel("reward")
    ax.set_ylabel("density")
    upper = max(1.0, max(rewards) * 1.05)
    lower = min(0.0, min(rewards) * 1.05)
    ax.set_xlim(lower, upper)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title)
    return _save_pdf(fig, out, audit_id=audit_id, title=title)


def coverage_calibration(stats: AuditStats, out: Path) -> Path:
    """Bar chart: empirical (held-out) coverage vs target per env.

    With a single env we render a centred numeric card instead of a
    one-bar chart (which reads as "broken" against a target line).
    """
    target = 1.0 - stats.alpha
    fig, ax = _new_axes()
    title = f"Empirical Coverage vs Target ({target * 100:.0f}%)"

    if not stats.per_env:
        ax.text(
            0.5, 0.5, "no environments",
            ha="center", va="center", transform=ax.transAxes,
            color=_COLOR_MISSING,
        )
        ax.set_axis_off()
        ax.set_title(title)
        return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)

    if len(stats.per_env) == 1:
        es = stats.per_env[0]
        cov = es.coverage_holdout
        cov_str = "n/a" if cov is None else f"{cov:.3f}"
        if cov is None:
            verdict = "no coverage data"
        elif cov >= target:
            verdict = "within target"
        else:
            verdict = "below target"
        _render_text_summary(
            ax,
            [
                ("environment:", es.env),
                ("empirical coverage:", cov_str),
                (f"target ({target:.2f}):", verdict),
            ],
        )
        ax.set_title(title)
        return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)

    envs = [es.env for es in stats.per_env]
    coverages: list[float] = [
        es.coverage_holdout if es.coverage_holdout is not None else float("nan")
        for es in stats.per_env
    ]
    colors = [_coverage_color(es.coverage_holdout, target) for es in stats.per_env]
    x = np.arange(len(envs))
    # NaN bars render as zero-height; replace with 0 explicitly so
    # matplotlib is happy and the missing-data colour still applies.
    heights = [0.0 if np.isnan(c) else c for c in coverages]
    ax.bar(x, heights, color=colors, alpha=0.85)
    ax.axhline(
        target, linestyle="--", color="black", linewidth=1.0,
        label=f"target = {target:.2f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("empirical coverage")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(title)
    return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)


def score_breakdown(stats: AuditStats, out: Path) -> Path:
    """Horizontal stacked bar per env: format-pass / format-fail / parse-error.

    Single-env audits render a numeric card rather than a one-row stack
    that would read as "broken" with no comparison context.
    """
    title = "Quality Breakdown by Environment"

    if not stats.per_env:
        fig, ax = _new_axes()
        ax.text(
            0.5, 0.5, "no environments",
            ha="center", va="center", transform=ax.transAxes,
            color=_COLOR_MISSING,
        )
        ax.set_axis_off()
        ax.set_title(title)
        return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)

    envs = [es.env for es in stats.per_env]
    pass_frac: list[float] = []
    fail_frac: list[float] = []
    parse_err_frac: list[float] = []
    for es in stats.per_env:
        parse_err = max(0.0, min(1.0, es.parse_failure_rate))
        success = max(0.0, 1.0 - parse_err)
        fmt_valid = max(0.0, min(1.0, es.format_valid_rate))
        pass_frac.append(success * fmt_valid)
        fail_frac.append(success * (1.0 - fmt_valid))
        parse_err_frac.append(parse_err)

    if len(envs) == 1:
        fig, ax = _new_axes()
        _render_text_summary(
            ax,
            [
                ("environment:", envs[0]),
                ("format pass:", f"{pass_frac[0] * 100:.1f}%"),
                ("format fail:", f"{fail_frac[0] * 100:.1f}%"),
                ("parse error:", f"{parse_err_frac[0] * 100:.1f}%"),
            ],
        )
        ax.set_title(title)
        return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)

    height = max(2.5, 0.55 * len(envs) + 1.5)
    fig, ax = _new_axes(height=height)
    y = np.arange(len(envs))
    ax.barh(y, pass_frac, color=_COLOR_PASS, label="format pass")
    ax.barh(y, fail_frac, left=pass_frac, color=_COLOR_WARN, label="format fail")
    left = np.array(pass_frac) + np.array(fail_frac)
    ax.barh(y, parse_err_frac, left=left, color=_COLOR_FAIL, label="parse error")
    ax.set_yticks(y)
    ax.set_yticklabels(envs)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("fraction of episodes")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.set_title(title)
    return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)


def cost_per_correct(stats: AuditStats, out: Path) -> Path:
    """Bar chart: USD per successful episode per env (log-scaled if needed)."""
    envs: list[str] = []
    costs: list[float] = []
    for es in stats.per_env:
        if es.n_success == 0 or es.total_cost_usd <= 0.0:
            continue
        envs.append(es.env)
        costs.append(es.total_cost_usd / es.n_success)

    fig, ax = _new_axes()
    title = "Cost per Correct Answer"

    if not envs:
        ax.text(
            0.5, 0.5,
            "Cost data not available\n"
            "(traces did not include estimated_cost_usd)",
            ha="center", va="center", transform=ax.transAxes,
            color=_COLOR_MISSING, fontsize=_FONT_SIZE + 1,
        )
        ax.set_axis_off()
        ax.set_title(title)
        return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)

    x = np.arange(len(envs))
    ax.bar(x, costs, color=_COLOR_NEUTRAL, alpha=0.85)
    if min(costs) > 0.0 and max(costs) / min(costs) > 100.0:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha="right")
    ax.set_ylabel("USD per correct answer")
    ax.set_title(title)
    return _save_pdf(fig, out, audit_id=stats.audit_id, title=title)


# ── public entry point ──────────────────────────────────────────────


def render_all_figures(stats: AuditStats, figures_dir: Path) -> list[Path]:
    """Render every figure for ``stats`` into ``figures_dir``.

    Returns the list of written paths, in render order: per-env reward
    distributions first, then ``coverage_calibration``, ``score_breakdown``,
    ``cost_per_correct``. Raises :class:`ValueError` when ``stats`` has
    no ``per_env`` entries (nothing meaningful to plot).
    """
    if not stats.per_env:
        raise ValueError(
            "AuditStats has no per-env data; nothing to render — "
            "did the audit complete with at least one scheduled env?"
        )
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for es in stats.per_env:
        slug = _slugify(es.env)
        paths.append(
            reward_distribution(
                es,
                audit_id=stats.audit_id,
                out=figures_dir / f"reward_distribution_{slug}.pdf",
            )
        )
    paths.append(
        coverage_calibration(stats, figures_dir / "coverage_calibration.pdf")
    )
    paths.append(
        score_breakdown(stats, figures_dir / "score_breakdown.pdf")
    )
    paths.append(
        cost_per_correct(stats, figures_dir / "cost_per_correct.pdf")
    )
    return paths


__all__: Iterable[str] = [
    "coverage_calibration",
    "cost_per_correct",
    "render_all_figures",
    "reward_distribution",
    "score_breakdown",
]
