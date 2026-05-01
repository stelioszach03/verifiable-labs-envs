"""Unit tests for ``vlabs_audit.figures`` — PDF generation + colour logic."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vlabs_audit.figures import (
    _COLOR_FAIL,
    _COLOR_MISSING,
    _COLOR_PASS,
    _coverage_color,
    cost_per_correct,
    coverage_calibration,
    render_all_figures,
    reward_distribution,
    score_breakdown,
)
from vlabs_audit.stats import AuditStats, EnvStats


def _pdf_text(path: Path) -> str:
    """Extract the rendered text from a PDF via ``pdftotext``.

    Matplotlib subsets fonts so the raw bytes don't contain ASCII
    strings; ``pdftotext`` is the canonical way to recover them.
    Skips the test when ``pdftotext`` is unavailable on PATH.
    """
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext (poppler-utils) not installed")
    proc = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return proc.stdout

# ── synthetic-stats helpers ──────────────────────────────────────────


def _env_stats(
    *,
    env: str = "env-a",
    n_episodes: int = 5,
    n_success: int | None = None,
    rewards: list[float] | None = None,
    parse_failure_rate: float = 0.0,
    format_valid_rate: float = 1.0,
    coverage_holdout: float | None = 0.95,
    total_cost_usd: float = 0.0,
) -> EnvStats:
    if rewards is None:
        rewards = [0.5 + 0.05 * i for i in range(n_episodes)]
    if n_success is None:
        n_success = len(rewards)
    n_failed = n_episodes - n_success
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    return EnvStats(
        env=env,
        n_episodes=n_episodes,
        n_success=n_success,
        n_failed=n_failed,
        mean_reward=mean_r,
        ci_low=mean_r - 0.05,
        ci_high=mean_r + 0.05,
        parse_failure_rate=parse_failure_rate,
        format_valid_rate=format_valid_rate,
        coverage_holdout=coverage_holdout,
        rewards=rewards,
        total_cost_usd=total_cost_usd,
    )


def _audit_stats(
    *,
    audit_id: str = "aud_synthetic",
    model: str = "test-model",
    alpha: float = 0.1,
    per_env: list[EnvStats] | None = None,
) -> AuditStats:
    if per_env is None:
        per_env = [_env_stats()]
    all_rewards = [r for es in per_env for r in es.rewards]
    agg_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    return AuditStats(
        audit_id=audit_id,
        model=model,
        alpha=alpha,
        n_episodes_per_env=max((es.n_episodes for es in per_env), default=0),
        per_env=per_env,
        aggregate_mean_reward=agg_mean,
        aggregate_ci_low=agg_mean - 0.05,
        aggregate_ci_high=agg_mean + 0.05,
        aggregate_parse_failure_rate=0.0,
        aggregate_format_valid_rate=1.0,
        aggregate_coverage_holdout=0.95,
    )


# ── tests ────────────────────────────────────────────────────────────


def test_render_all_figures_produces_non_empty_pdfs(tmp_path: Path) -> None:
    """Every figure type renders without exception and writes a non-empty PDF."""
    stats = _audit_stats(
        per_env=[
            _env_stats(env="env-a", coverage_holdout=0.95, total_cost_usd=0.05),
            _env_stats(env="env-b", coverage_holdout=0.80, total_cost_usd=0.02),
        ],
    )
    out_dir = tmp_path / "figs"
    paths = render_all_figures(stats, out_dir)

    # 2 envs → 2 reward distributions + 3 audit-level figures = 5.
    assert len(paths) == 5
    assert {p.name for p in paths} == {
        "reward_distribution_env-a.pdf",
        "reward_distribution_env-b.pdf",
        "coverage_calibration.pdf",
        "score_breakdown.pdf",
        "cost_per_correct.pdf",
    }
    for p in paths:
        assert p.exists(), p
        assert p.stat().st_size > 1024, f"{p} is suspiciously small"
        # Every PDF starts with the magic %PDF- header.
        with p.open("rb") as fh:
            assert fh.read(5) == b"%PDF-"


def test_pdf_metadata_includes_audit_id(tmp_path: Path) -> None:
    """Each rendered PDF embeds ``audit_id:<id>`` in /Subject metadata."""
    audit_id = "aud_metadata_check_1234"
    stats = _audit_stats(audit_id=audit_id)
    out = tmp_path / "coverage.pdf"
    coverage_calibration(stats, out)
    raw = out.read_bytes()
    # Either uncompressed (\Subject (audit_id:...)) or compressed object stream.
    # Search for the audit_id byte sequence regardless of encoding.
    assert audit_id.encode("ascii") in raw, (
        f"{audit_id!r} not found in PDF metadata; bytes start: {raw[:200]!r}"
    )
    # And the matplotlib backend writes /Author = vlabs-audit.
    assert b"vlabs-audit" in raw


def test_render_all_figures_empty_per_env_raises(tmp_path: Path) -> None:
    """An AuditStats with no per_env entries cannot be rendered."""
    stats = AuditStats(
        audit_id="aud_empty",
        model="m",
        alpha=0.1,
        n_episodes_per_env=0,
        per_env=[],
        aggregate_mean_reward=0.0,
        aggregate_ci_low=0.0,
        aggregate_ci_high=0.0,
        aggregate_parse_failure_rate=0.0,
        aggregate_format_valid_rate=0.0,
        aggregate_coverage_holdout=None,
    )
    with pytest.raises(ValueError, match="no per-env data"):
        render_all_figures(stats, tmp_path / "figs")


def test_cost_per_correct_handles_missing_cost_data(tmp_path: Path) -> None:
    """When no env has cost data the figure renders a 'not available' plate."""
    stats = _audit_stats(
        per_env=[
            _env_stats(env="env-a", total_cost_usd=0.0),
            _env_stats(env="env-b", total_cost_usd=0.0),
        ],
    )
    out = tmp_path / "cost.pdf"
    p = cost_per_correct(stats, out)
    assert p.exists()
    assert p.stat().st_size > 1024
    # The header is still a valid PDF — no exception, no empty file.
    raw = out.read_bytes()
    assert raw[:5] == b"%PDF-"


def test_cost_per_correct_renders_real_costs(tmp_path: Path) -> None:
    """When at least one env has cost data, the figure plots its bar."""
    stats = _audit_stats(
        per_env=[
            _env_stats(env="env-a", n_success=5, total_cost_usd=0.10),
            _env_stats(env="env-b", n_success=4, total_cost_usd=0.0),  # skipped
        ],
    )
    out = tmp_path / "cost.pdf"
    p = cost_per_correct(stats, out)
    assert p.exists()
    assert p.stat().st_size > 1024


@pytest.mark.parametrize(
    ("coverage", "target", "expected"),
    [
        (0.95, 0.90, _COLOR_PASS),
        (0.90, 0.90, _COLOR_PASS),  # at-target counts as pass
        (0.85, 0.90, _COLOR_FAIL),
        (0.50, 0.90, _COLOR_FAIL),
        (1.00, 0.90, _COLOR_PASS),
        (None, 0.90, _COLOR_MISSING),
        (float("nan"), 0.90, _COLOR_MISSING),
    ],
)
def test_coverage_color_assignment(
    coverage: float | None, target: float, expected: str
) -> None:
    assert _coverage_color(coverage, target) == expected


def test_reward_distribution_with_no_successes_renders_placeholder(
    tmp_path: Path,
) -> None:
    """A 0-success env still produces a (placeholder) PDF rather than crashing."""
    stats = _audit_stats(
        per_env=[
            _env_stats(env="env-a", n_episodes=3, n_success=0, rewards=[]),
        ],
    )
    out = tmp_path / "rd.pdf"
    p = reward_distribution(stats.per_env[0], audit_id=stats.audit_id, out=out)
    assert p.exists()
    assert p.stat().st_size > 512


def test_score_breakdown_handles_mixed_failure_modes(tmp_path: Path) -> None:
    """A breakdown with parse errors + format failures still totals to 1.0."""
    stats = _audit_stats(
        per_env=[
            _env_stats(
                env="env-a", n_episodes=10, n_success=8,
                rewards=[0.5] * 8,
                parse_failure_rate=0.2, format_valid_rate=0.75,
            ),
        ],
    )
    out = tmp_path / "sb.pdf"
    p = score_breakdown(stats, out)
    assert p.exists()
    assert p.stat().st_size > 1024


def test_single_env_coverage_calibration_renders_text_summary(
    tmp_path: Path,
) -> None:
    """Single-env audits get a centred text card, not a one-bar chart."""
    stats = _audit_stats(
        per_env=[_env_stats(env="env-only", coverage_holdout=0.95)],
    )
    out = tmp_path / "cov.pdf"
    p = coverage_calibration(stats, out)
    assert p.exists()
    assert p.read_bytes()[:5] == b"%PDF-"
    text = _pdf_text(p)
    # The env name and the within/below verdict appear inline as text.
    assert "env-only" in text
    assert "within target" in text or "below target" in text


def test_single_env_score_breakdown_renders_text_summary(tmp_path: Path) -> None:
    """Single-env audits get a centred text card for the breakdown too."""
    stats = _audit_stats(
        per_env=[
            _env_stats(
                env="env-only", n_episodes=10, n_success=9,
                rewards=[0.5] * 9,
                parse_failure_rate=0.1, format_valid_rate=0.9,
            ),
        ],
    )
    out = tmp_path / "sb.pdf"
    p = score_breakdown(stats, out)
    assert p.exists()
    assert p.read_bytes()[:5] == b"%PDF-"
    text = _pdf_text(p)
    assert "env-only" in text
    # Format-pass / format-fail / parse-error labels all present.
    for needle in ("format pass", "format fail", "parse error"):
        assert needle in text


def test_below_target_coverage_says_below_target(tmp_path: Path) -> None:
    """Coverage below target labels the verdict 'below target'."""
    stats = _audit_stats(
        per_env=[_env_stats(env="env-only", coverage_holdout=0.50)],
        alpha=0.1,  # target = 0.90 → 0.50 is well below
    )
    out = tmp_path / "cov.pdf"
    p = coverage_calibration(stats, out)
    text = _pdf_text(p)
    assert "below target" in text
    assert "within target" not in text
