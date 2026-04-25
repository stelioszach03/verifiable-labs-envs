"""Tests for the ``verifiable`` CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verifiable_labs_envs.cli import main as cli_main
from verifiable_labs_envs.traces import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "agents"


def _run_cli(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Spawn the CLI as a subprocess so we get the real argparse exit codes."""
    return subprocess.run(
        [sys.executable, "-m", "verifiable_labs_envs", *argv],
        capture_output=True, text=True, cwd=cwd or REPO_ROOT,
    )


# ── argparse plumbing ──────────────────────────────────────


def test_root_help_lists_all_subcommands():
    proc = _run_cli("--help")
    assert proc.returncode == 0
    out = proc.stdout
    for sub in ("envs", "run", "compare", "report", "init-env", "validate-env"):
        assert sub in out


@pytest.mark.parametrize("sub", ["envs", "run", "compare", "report", "init-env", "validate-env"])
def test_subcommand_help_works(sub):
    proc = _run_cli(sub, "--help")
    assert proc.returncode == 0, proc.stderr
    assert sub.replace("-", "") in proc.stdout.lower() or "usage:" in proc.stdout


def test_no_command_errors_cleanly():
    proc = _run_cli()
    assert proc.returncode != 0


# ── envs ───────────────────────────────────────────────────


def test_envs_text_lists_at_least_10():
    proc = _run_cli("envs")
    assert proc.returncode == 0
    assert "environments available" in proc.stdout
    # Each env id should appear; check the canonical sparse-fourier one.
    assert "sparse-fourier-recovery" in proc.stdout


def test_envs_json_format():
    proc = _run_cli("envs", "--format", "json")
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert isinstance(parsed, list)
    assert len(parsed) >= 10
    assert "sparse-fourier-recovery" in parsed


# ── run ────────────────────────────────────────────────────


def test_run_zero_agent_writes_valid_jsonl(tmp_path):
    out = tmp_path / "out.jsonl"
    rc = cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "zero_agent.py"),
        "--n", "2", "--out", str(out), "--quiet",
    ])
    assert rc == 0
    assert out.exists()
    traces = read_jsonl(out)
    assert len(traces) == 2
    for t in traces:
        assert t.env_name == "sparse-fourier-recovery"
        assert t.agent_name == "zero"
        assert t.parse_success is True
        assert t.seed in (0, 1)
        assert "nmse" in t.reward_components or t.reward_components  # populated


def test_run_random_agent_produces_distinct_seeds(tmp_path):
    out = tmp_path / "random.jsonl"
    rc = cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "random_agent.py"),
        "--n", "3", "--start-seed", "100", "--out", str(out), "--quiet",
    ])
    assert rc == 0
    traces = read_jsonl(out)
    assert [t.seed for t in traces] == [100, 101, 102]
    # Predictions hash differently across seeds because random_agent
    # seeds from the observation.
    hashes = {t.prediction_hash for t in traces}
    assert len(hashes) == 3


def test_run_with_baseline_populates_gap(tmp_path):
    out = tmp_path / "baseline_run.jsonl"
    rc = cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "zero_agent.py"),
        "--n", "2", "--out", str(out), "--quiet", "--with-baseline",
    ])
    assert rc == 0
    traces = read_jsonl(out)
    for t in traces:
        assert t.classical_baseline_reward is not None
        assert t.gap_to_classical is not None


def test_run_simple_baseline_agent_uses_classical(tmp_path):
    out = tmp_path / "classical.jsonl"
    rc = cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "simple_baseline_agent.py"),
        "--n", "2", "--out", str(out), "--quiet",
    ])
    assert rc == 0
    traces = read_jsonl(out)
    for t in traces:
        assert t.parse_success is True
        # The metadata records the classical-baseline path.
        assert t.metadata.get("_classical_baseline") is True
        # Classical baseline should beat the zero agent's ~0.34 floor.
        assert t.reward >= 0.30


def test_run_unknown_env_errors(tmp_path):
    out = tmp_path / "x.jsonl"
    rc = cli_main([
        "run", "--env", "definitely-not-an-env",
        "--agent", str(EXAMPLES / "zero_agent.py"),
        "--n", "1", "--out", str(out), "--quiet",
    ])
    assert rc == 2


def test_run_missing_agent_errors(tmp_path):
    out = tmp_path / "x.jsonl"
    rc = cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(tmp_path / "nope.py"),
        "--n", "1", "--out", str(out), "--quiet",
    ])
    assert rc == 2


def test_run_env_kwargs_forwarded(tmp_path):
    out = tmp_path / "kw.jsonl"
    rc = cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "zero_agent.py"),
        "--n", "1", "--out", str(out), "--quiet",
        "--env-kwarg", "calibration_quantile=2.5",
    ])
    assert rc == 0
    traces = read_jsonl(out)
    assert traces[0].metadata["env_kwargs"] == {"calibration_quantile": 2.5}


# ── report ──────────────────────────────────────────────────


def test_report_renders_all_required_sections(tmp_path):
    # Produce a run, then report on it.
    run_out = tmp_path / "run.jsonl"
    cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "zero_agent.py"),
        "--n", "3", "--out", str(run_out), "--quiet",
    ])
    md_out = tmp_path / "report.md"
    rc = cli_main([
        "report", "--run", str(run_out), "--out", str(md_out), "--quiet",
    ])
    assert rc == 0
    text = md_out.read_text()
    from verifiable_labs_envs.reporting import REQUIRED_SECTIONS
    for section in REQUIRED_SECTIONS:
        assert section in text, f"missing section: {section}"


def test_report_errors_on_empty_run(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    md = tmp_path / "out.md"
    rc = cli_main(["report", "--run", str(empty), "--out", str(md)])
    assert rc == 2


# ── compare ─────────────────────────────────────────────────


def test_compare_prints_table_for_two_runs(tmp_path):
    a = tmp_path / "zero.jsonl"
    b = tmp_path / "random.jsonl"
    cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "zero_agent.py"),
        "--n", "2", "--out", str(a), "--quiet",
    ])
    cli_main([
        "run", "--env", "sparse-fourier-recovery",
        "--agent", str(EXAMPLES / "random_agent.py"),
        "--n", "2", "--out", str(b), "--quiet",
    ])
    proc = _run_cli("compare", "--runs", str(a), str(b))
    assert proc.returncode == 0
    out = proc.stdout
    assert "zero" in out and "random" in out
    assert "mean" in out and "parse_ok%" in out


def test_compare_errors_on_missing_run(tmp_path):
    proc = _run_cli("compare", "--runs", str(tmp_path / "no.jsonl"))
    assert proc.returncode != 0


# ── init-env / validate-env ──────────────────────────────────


def test_init_env_scaffolds_to_target(tmp_path):
    target = tmp_path / "demo-env-cli"
    rc = cli_main([
        "init-env", "demo-env-cli", "--domain", "demo",
        "--target", str(target),
    ])
    assert rc == 0
    assert (target / "pyproject.toml").exists()
    # Template directory was renamed via __ENV_PY__ substitution.
    assert (target / "demo_env_cli").is_dir()


def test_validate_env_runs_against_fresh_scaffold(tmp_path, capsys):
    """Fresh scaffold's tests + procedural-regen pass; calibration + adapter
    fail because the forward operator stub raises NotImplementedError. The
    validator correctly reports a non-zero exit code under that condition —
    we assert the validator *runs* and emits its [N/4] check headers."""
    target = tmp_path / "demo-env-validate"
    rc = cli_main([
        "init-env", "demo-env-validate", "--domain", "demo",
        "--target", str(target),
    ])
    assert rc == 0
    rc = cli_main([
        "validate-env", str(target), "--skip-adapter-check",
    ])
    # Calibration still runs and fails on the unfilled forward operator.
    # That's expected — the validator's contract is "non-zero on any failed
    # check", and unfilled scaffolds always fail calibration.
    assert rc != 0


def test_validate_env_fails_on_malformed(tmp_path):
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "not_a_python_package.txt").write_text("nope")
    rc = cli_main(["validate-env", str(bad)])
    assert rc != 0
