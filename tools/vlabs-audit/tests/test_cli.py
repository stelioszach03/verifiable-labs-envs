"""Unit tests for ``vlabs_audit.cli`` — Typer command wiring + dry-run output."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vlabs_audit import __version__
from vlabs_audit.cli import app
from vlabs_audit.runner import EpisodeOutput

runner = CliRunner()


def test_help_includes_audit_and_version() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout
    assert "version" in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
    assert "vlabs-audit" in result.stdout


def test_dry_run_prints_resolved_config(tmp_yaml_config: Path) -> None:
    result = runner.invoke(
        app, ["audit", "--config", str(tmp_yaml_config), "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "claude-haiku-4.5" in out
    assert "sparse-fourier-recovery" in out
    assert "dry run" in out.lower()
    # Total episodes is shown alongside per-env count.
    assert "episodes" in out.lower()


def test_dry_run_with_overrides(tmp_yaml_config: Path) -> None:
    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--episodes", "100",
            "--alpha", "0.05",
            "--parallel", "4",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "100" in result.stdout
    assert "0.05" in result.stdout
    assert "parallel:        4" in result.stdout


def test_dry_run_envs_override(tmp_yaml_config: Path) -> None:
    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--envs", "sparse-fourier-recovery,phase-retrieval",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "sparse-fourier-recovery" in result.stdout
    assert "phase-retrieval" in result.stdout


def test_audit_without_config_or_required_args_fails() -> None:
    result = runner.invoke(app, ["audit"])
    # No config + no overrides → AuditConfig validation fails → exit 2.
    assert result.exit_code == 2


def test_audit_runs_with_fake_runner(
    tmp_yaml_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI run with the subprocess runner stubbed via monkeypatch."""
    from vlabs_audit import runner as runner_mod

    def fake_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        jp = output_dir / f"{env}__seed{seed}.jsonl"
        jp.write_text('{"reward": 0.5}\n')
        return EpisodeOutput(reward=0.5, jsonl_path=jp)

    monkeypatch.setattr(runner_mod, "default_episode_run", fake_run)
    monkeypatch.setenv("VLABS_AUDIT_HOME", str(tmp_path / "audit_home"))

    result = runner.invoke(
        app,
        ["audit", "--config", str(tmp_yaml_config), "--episodes", "2"],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    out = result.stdout
    assert "complete" in out.lower()
    assert "success" in out.lower()
    # 1 env × 2 episodes = 2 successful rows
    assert "'success': 2" in out


def test_audit_resume_unknown_id_fails(
    tmp_yaml_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLABS_AUDIT_HOME", str(tmp_path / "audit_home"))
    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--resume", "aud_does_not_exist",
        ],
    )
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "unknown audit_id" in combined.lower()


def test_dry_run_with_print_stats_shows_stub(tmp_yaml_config: Path) -> None:
    """``--print-stats --dry-run`` doesn't run anything, but flags the table."""
    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--print-stats",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "dry run" in out.lower()
    assert "print-stats" in out.lower()


def test_audit_with_print_stats_renders_table(
    tmp_yaml_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: fake runner_fn writes traces, ``--print-stats`` prints AGGREGATE."""
    from vlabs_audit import runner as runner_mod
    from vlabs_audit.runner import EpisodeOutput

    def fake_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        jp = output_dir / f"{env}__seed{seed}.jsonl"
        # Realistic-ish trace content matching the SDK's Trace shape.
        jp.write_text(
            '{"reward": 0.6, "parse_success": true, "coverage": 1.0, '
            f'"seed": {seed}, "env_name": "{env}"}}\n'
        )
        return EpisodeOutput(reward=0.6, jsonl_path=jp)

    monkeypatch.setattr(runner_mod, "default_episode_run", fake_run)
    monkeypatch.setenv("VLABS_AUDIT_HOME", str(tmp_path / "audit_home"))

    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--episodes", "3",
            "--print-stats",
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    out = result.stdout
    assert "AGGREGATE" in out
    assert "sparse-fourier-recovery" in out
    assert "0.600" in out  # mean reward column


def test_dry_run_with_figures_dir_shows_stub(
    tmp_yaml_config: Path, tmp_path: Path
) -> None:
    """``--figures-dir`` in dry-run prints a stub line; nothing is rendered."""
    figs = tmp_path / "figs"
    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--figures-dir", str(figs),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "figures-dir" in result.stdout.lower()
    # Auto-implies stats stub too.
    assert "print-stats" in result.stdout.lower()
    # Dry run never creates the figures dir.
    assert not figs.exists()


def test_audit_with_figures_dir_writes_pdfs(
    tmp_yaml_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--figures-dir`` after a real run writes PDF files and reports the count."""
    from vlabs_audit import runner as runner_mod
    from vlabs_audit.runner import EpisodeOutput

    def fake_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        jp = output_dir / f"{env}__seed{seed}.jsonl"
        jp.write_text(
            '{"reward": 0.55, "parse_success": true, "coverage": 1.0, '
            f'"seed": {seed}, "env_name": "{env}"}}\n'
        )
        return EpisodeOutput(reward=0.55, jsonl_path=jp)

    monkeypatch.setattr(runner_mod, "default_episode_run", fake_run)
    monkeypatch.setenv("VLABS_AUDIT_HOME", str(tmp_path / "audit_home"))
    figs = tmp_path / "figs"

    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--episodes", "3",
            "--figures-dir", str(figs),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "Wrote" in result.stdout and "figures" in result.stdout
    # 1 env × 3 episodes → 1 reward distribution + 3 audit-level figs = 4 PDFs.
    pdfs = sorted(figs.glob("*.pdf"))
    assert len(pdfs) == 4
    for p in pdfs:
        assert p.stat().st_size > 1024
        with p.open("rb") as fh:
            assert fh.read(5) == b"%PDF-"


def test_dry_run_anonymize_flag(tmp_yaml_config: Path) -> None:
    result = runner.invoke(
        app,
        [
            "audit",
            "--config", str(tmp_yaml_config),
            "--anonymize",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "anonymize:       True" in result.stdout
