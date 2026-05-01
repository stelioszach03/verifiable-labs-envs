"""Unit tests for ``vlabs_audit.latex`` — escape filter + render_tex + render_report."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from vlabs_audit.latex import latex_escape, render_report, render_tex
from vlabs_audit.stats import AuditStats, EnvStats

# ── synthetic stats helper ───────────────────────────────────────────


def _mk_env_stats(
    *,
    env: str = "sparse-fourier-recovery",
    n_episodes: int = 5,
    n_success: int = 5,
    rewards: list[float] | None = None,
    parse_failure_rate: float = 0.0,
    format_valid_rate: float = 1.0,
    coverage_holdout: float | None = 0.93,
    total_cost_usd: float = 0.0,
) -> EnvStats:
    rewards = rewards or [0.32, 0.35, 0.38, 0.34, 0.38][:n_success]
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    return EnvStats(
        env=env,
        n_episodes=n_episodes,
        n_success=n_success,
        n_failed=n_episodes - n_success,
        mean_reward=mean_r,
        ci_low=mean_r - 0.02,
        ci_high=mean_r + 0.02,
        parse_failure_rate=parse_failure_rate,
        format_valid_rate=format_valid_rate,
        coverage_holdout=coverage_holdout,
        rewards=rewards,
        total_cost_usd=total_cost_usd,
    )


def _mk_audit_stats(
    *,
    audit_id: str = "aud_test_1234",
    model: str = "anthropic/claude-haiku-4.5",
    alpha: float = 0.1,
    per_env: list[EnvStats] | None = None,
) -> AuditStats:
    per_env = per_env or [_mk_env_stats()]
    rewards = [r for es in per_env for r in es.rewards]
    agg_mean = sum(rewards) / len(rewards) if rewards else 0.0
    return AuditStats(
        audit_id=audit_id,
        model=model,
        alpha=alpha,
        n_episodes_per_env=max((es.n_episodes for es in per_env), default=0),
        per_env=per_env,
        aggregate_mean_reward=agg_mean,
        aggregate_ci_low=agg_mean - 0.02,
        aggregate_ci_high=agg_mean + 0.02,
        aggregate_parse_failure_rate=0.0,
        aggregate_format_valid_rate=1.0,
        aggregate_coverage_holdout=0.93,
    )


# ── (a) escape filter handles all 11 special characters ──────────────


def test_latex_escape_handles_every_special_character() -> None:
    """Every one of the 11 special chars round-trips through latex_escape."""
    cases: list[tuple[str, str]] = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\^{}"),
        ("~", r"\~{}"),
        ("<", r"\textless{}"),
        (">", r"\textgreater{}"),
    ]
    for src, dst in cases:
        assert latex_escape(src) == dst, f"escape({src!r}) gave {latex_escape(src)!r}"

    # Backslash must be processed first so we don't double-escape its
    # replacement sequence.
    assert latex_escape("\\&") == r"\textbackslash{}\&"
    assert latex_escape("100% & rising") == r"100\% \& rising"
    assert latex_escape("path/to/file_v1.pdf") == r"path/to/file\_v1.pdf"
    # Non-string inputs are coerced via str(); None becomes empty.
    assert latex_escape(None) == ""
    assert latex_escape(42) == "42"
    assert latex_escape(3.14) == "3.14"


def test_latex_escape_preserves_safe_chars() -> None:
    """Non-special chars (letters, digits, hyphens, slashes) pass through."""
    safe = "Sparse-Fourier 2026 anthropic/claude-haiku-4.5"
    assert latex_escape(safe) == safe


# ── (b)+(e) full render produces all 5 sections without exception ────


def test_render_tex_produces_full_document_with_all_sections(tmp_path: Path) -> None:
    """Rendering full TeX runs every section template without raising."""
    stats = _mk_audit_stats(
        per_env=[
            _mk_env_stats(env="sparse-fourier-recovery"),
            _mk_env_stats(env="phase-retrieval", coverage_holdout=0.85),
        ],
    )
    tex = render_tex(stats)
    assert isinstance(tex, str)
    assert len(tex) > 1000  # full document is several KB

    for header in (
        r"\section{Executive Summary}",
        r"\section{Methodology}",
        r"\section{Results}",
        r"\section{Recommendations}",
        r"\section{Appendix}",
    ):
        assert header in tex, f"missing section: {header!r}"


def test_render_tex_matches_snapshot_substrings(tmp_path: Path) -> None:
    """Rendered TeX contains every required substring listed in fixtures."""
    fixture = (
        Path(__file__).parent / "fixtures" / "expected_sections.txt"
    ).read_text(encoding="utf-8")
    needles = [
        line for line in fixture.splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    stats = _mk_audit_stats()
    tex = render_tex(stats)
    missing = [n for n in needles if n not in tex]
    assert not missing, f"snapshot needles missing from rendered TeX: {missing}"


# ── (g) cover macro receives correct args ───────────────────────────


def test_cover_macro_receives_three_correct_args() -> None:
    """\\makevlabscover must be called with title, model, and date."""
    stats = _mk_audit_stats(
        audit_id="aud_cover_test",
        model="some/model_name_with_underscore",
    )
    tex = render_tex(stats)

    # Title is always the brand "Verifiable Labs"; model is escaped;
    # date is YYYY-MM-DD.
    assert r"\makevlabscover{Verifiable Labs}{some/model\_name\_with\_underscore}{" in tex
    # The audit id is set via setvlabsaudit on its own line.
    assert r"\setvlabsaudit{aud\_cover\_test}" in tex


# ── (f) figure paths are correct + relative ──────────────────────────


def test_figure_paths_in_latex_are_relative_basenames() -> None:
    """Figures are referenced by basename (no slashes, no absolute paths)."""
    stats = _mk_audit_stats(
        per_env=[
            _mk_env_stats(env="sparse-fourier-recovery"),
            _mk_env_stats(env="phase-retrieval"),
        ],
    )
    tex = render_tex(stats)
    # Per-env reward distribution figures use the env slug directly.
    assert "reward_distribution_sparse-fourier-recovery.pdf" in tex
    assert "reward_distribution_phase-retrieval.pdf" in tex
    # Audit-level figures use canonical filenames.
    assert "coverage_calibration.pdf" in tex
    assert "score_breakdown.pdf" in tex
    assert "cost_per_correct.pdf" in tex
    # No \includegraphics{...} target should ever start with a "/" — that
    # would break tectonic's working-dir-relative resolution.
    bad = [
        line for line in tex.splitlines()
        if r"\includegraphics" in line and "{/" in line
    ]
    assert not bad, f"absolute include paths found: {bad}"


# ── (c) missing tectonic raises with install URL ─────────────────────


def test_render_report_missing_tectonic_raises_with_install_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vlabs_audit.latex as latex_mod

    monkeypatch.setattr(latex_mod.shutil, "which", lambda _name: None)
    stats = _mk_audit_stats()
    with pytest.raises(RuntimeError) as exc_info:
        render_report(stats, tmp_path / "figs", tmp_path / "out.pdf")
    msg = str(exc_info.value)
    assert "tectonic" in msg
    assert "tectonic-typesetting.github.io" in msg


# ── (d) tectonic compile failure surfaces the log tail ───────────────


@dataclass
class _FakeProc:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def test_render_report_compile_failure_surfaces_log_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vlabs_audit.latex as latex_mod

    monkeypatch.setattr(latex_mod.shutil, "which", lambda _name: "/fake/tectonic")
    monkeypatch.setattr(
        latex_mod.subprocess, "run",
        lambda *a, **kw: _FakeProc(
            returncode=2,
            stderr="error: Undefined control sequence \\fakecommand at line 42\n",
        ),
    )
    figs = tmp_path / "figs"
    figs.mkdir()
    stats = _mk_audit_stats()
    with pytest.raises(RuntimeError) as exc_info:
        render_report(stats, figs, tmp_path / "out.pdf")
    msg = str(exc_info.value)
    assert "exit 2" in msg
    assert "fakecommand" in msg


# ── (b) full pipeline with tectonic stubbed ──────────────────────────


def test_render_report_full_pipeline_with_stubbed_tectonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end render_report with subprocess.run stubbed.

    The stub writes a minimal valid PDF to ``main.pdf`` in the cwd it
    receives, mimicking what tectonic would do.
    """
    import vlabs_audit.latex as latex_mod

    monkeypatch.setattr(latex_mod.shutil, "which", lambda _name: "/fake/tectonic")

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = Path(kwargs["cwd"])
        # Verify the working dir contains main.tex + the class file + figures.
        assert (captured["cwd"] / "main.tex").exists()
        assert (captured["cwd"] / "vlabs_report.cls").exists()
        # Write a fake PDF where tectonic would have.
        (captured["cwd"] / "main.pdf").write_bytes(
            b"%PDF-1.4\n%fake\n%%EOF\n"
        )
        return _FakeProc(returncode=0, stdout="ok\n")

    monkeypatch.setattr(latex_mod.subprocess, "run", fake_run)

    # Feed real figures so the cwd-copy step has something to do.
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "reward_distribution_sparse-fourier-recovery.pdf").write_bytes(b"%PDF-1.4\nfake\n")
    (figs / "coverage_calibration.pdf").write_bytes(b"%PDF-1.4\nfake\n")
    (figs / "score_breakdown.pdf").write_bytes(b"%PDF-1.4\nfake\n")
    (figs / "cost_per_correct.pdf").write_bytes(b"%PDF-1.4\nfake\n")

    out = tmp_path / "out" / "report.pdf"
    stats = _mk_audit_stats()
    pdf_path = render_report(stats, figs, out)

    assert pdf_path == out
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")
    # CLI invocation is the documented one.
    assert captured["cmd"] == [
        "tectonic", "-X", "compile", "main.tex",
    ]


def test_render_report_handles_tectonic_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``subprocess.TimeoutExpired`` propagates so the caller can decide."""
    import vlabs_audit.latex as latex_mod

    monkeypatch.setattr(latex_mod.shutil, "which", lambda _name: "/fake/tectonic")

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(latex_mod.subprocess, "run", boom)
    figs = tmp_path / "figs"
    figs.mkdir()
    with pytest.raises(subprocess.TimeoutExpired):
        render_report(_mk_audit_stats(), figs, tmp_path / "out.pdf", timeout_s=1.0)


def test_render_tex_includes_per_env_stats_table_rows() -> None:
    """Each env contributes a stats table row with mean + CI + parse rate."""
    stats = _mk_audit_stats(
        per_env=[
            _mk_env_stats(env="sparse-fourier-recovery", parse_failure_rate=0.0),
            _mk_env_stats(env="phase-retrieval", parse_failure_rate=0.10),
        ],
    )
    tex = render_tex(stats)
    # Both env display names appear in tex.
    assert "Sparse Fourier Recovery" in tex
    assert "Phase Retrieval" in tex
    # The 10 % parse rate is rendered as a percent.
    assert "10.0\\%" in tex
