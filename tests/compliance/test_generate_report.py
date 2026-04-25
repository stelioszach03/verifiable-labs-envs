"""Tests for ``scripts/generate_report.py``."""
from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_report.py"
TEMPLATE = REPO_ROOT / "templates" / "compliance-report" / "report_template.md"


def _load_module():
    name = "_generate_report"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # required for @dataclass to resolve forward refs
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _synthetic_rows(model: str = "test-model") -> list[dict]:
    """A 3-env × 5-seed synthetic benchmark with realistic shapes."""
    out = []
    base = [
        ("env-a", 0.7, 0.85),
        ("env-b", 0.4, 0.90),
        ("env-c", 0.15, 0.65),  # below 0.30 → should appear in low-reward table
    ]
    for env, mean, cov in base:
        for seed in range(5):
            reward = mean + 0.01 * (seed - 2)  # tiny per-seed variation
            comps = f"nmse={mean:.3f}, support={mean - 0.05:.3f}, conformal={cov:.3f}"
            out.append({
                "timestamp": "2026-04-25T12:00:00+00:00",
                "env": env,
                "model": model,
                "seed": seed,
                "turn": 1,
                "reward": reward,
                "components": comps,
                "parse_ok": "True" if seed != 4 or env != "env-a" else "False",
                "usd_cost": 0.001,
                "prompt_tokens": 800,
                "completion_tokens": 50,
                "latency_s": 1.5,
                "error": "",
                "meta": "",
            })
    return out


# ── unit-level tests ────────────────────────────────────


def test_parse_components_roundtrip():
    m = _load_module()
    parsed = m._parse_components_str("nmse=0.135, support=0.200, conformal=0.900")
    assert parsed == {"nmse": pytest.approx(0.135), "support": pytest.approx(0.200), "conformal": pytest.approx(0.900)}


def test_parse_components_handles_empty():
    m = _load_module()
    assert m._parse_components_str("") == {}
    assert m._parse_components_str("{}") == {}


def test_load_episodes_filters_to_model(tmp_path):
    m = _load_module()
    rows = _synthetic_rows("model-a") + _synthetic_rows("model-b")
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    eps = m.load_episodes(csv_path, "model-a")
    assert len(eps) == 15  # 3 envs × 5 seeds
    assert {e.model for e in eps} == {"model-a"}


def test_load_episodes_keeps_only_latest_turn(tmp_path):
    m = _load_module()
    rows = _synthetic_rows()
    # Add a turn-2 row for one episode that should win.
    rows.append({**rows[0], "turn": 2, "reward": 0.99})
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    eps = m.load_episodes(csv_path, "test-model")
    matching = [e for e in eps if e.env == rows[0]["env"] and e.seed == int(rows[0]["seed"])]
    assert len(matching) == 1
    assert matching[0].turn == 2
    assert matching[0].reward == pytest.approx(0.99)


def test_load_episodes_raises_on_missing_model(tmp_path):
    m = _load_module()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, _synthetic_rows())
    with pytest.raises(ValueError, match="not found"):
        m.load_episodes(csv_path, "nonexistent-model")


def test_load_episodes_raises_on_missing_csv():
    m = _load_module()
    with pytest.raises(FileNotFoundError):
        m.load_episodes(Path("/no/such/path.csv"), "any-model")


def test_per_env_stats_aggregates_correctly(tmp_path):
    m = _load_module()
    rows = _synthetic_rows()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    eps = m.load_episodes(csv_path, "test-model")
    stats = m.per_env_stats(eps)
    assert {s.env for s in stats} == {"env-a", "env-b", "env-c"}
    by_name = {s.env: s for s in stats}
    assert by_name["env-a"].n == 5
    assert by_name["env-a"].parse_fail_rate == pytest.approx(0.2)  # 1/5
    assert 0.6 <= by_name["env-a"].mean_reward <= 0.75


def test_make_per_env_table_sorts_by_reward_descending(tmp_path):
    m = _load_module()
    rows = _synthetic_rows()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    stats = m.per_env_stats(m.load_episodes(csv_path, "test-model"))
    table = m.make_per_env_table(stats)
    # env-a (mean ~0.7) should appear before env-b (~0.4) before env-c (~0.15)
    pos_a = table.index("`env-a`")
    pos_b = table.index("`env-b`")
    pos_c = table.index("`env-c`")
    assert pos_a < pos_b < pos_c


def test_make_low_reward_table_skips_when_all_pass(tmp_path):
    m = _load_module()
    rows = [r for r in _synthetic_rows() if r["env"] != "env-c"]  # drop low-scoring env
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    stats = m.per_env_stats(m.load_episodes(csv_path, "test-model"))
    table = m.make_low_reward_table(stats, threshold=0.3)
    assert "No env scored below" in table


def test_make_low_reward_table_lists_underperformers(tmp_path):
    m = _load_module()
    rows = _synthetic_rows()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    stats = m.per_env_stats(m.load_episodes(csv_path, "test-model"))
    table = m.make_low_reward_table(stats, threshold=0.3)
    assert "`env-c`" in table
    assert "`env-a`" not in table
    assert "`env-b`" not in table


def test_recommendations_flag_high_parse_fail():
    m = _load_module()
    fake_stats = [
        m.EnvStats(env="env-x", n=10, mean_reward=0.6, std_reward=0.05, parse_fail_rate=0.2, coverage=0.92),
    ]
    out = m.recommendations(fake_stats, overall_mean=0.6, parse_fail_rate=0.2,
                            coverage_pct=92.0, target_coverage_pct=90.0)
    assert "Tighten output formatting" in out


def test_recommendations_flag_miscalibration():
    m = _load_module()
    fake_stats = [m.EnvStats(env="env-x", n=10, mean_reward=0.7, std_reward=0.0, parse_fail_rate=0.0, coverage=0.55)]
    out = m.recommendations(fake_stats, overall_mean=0.7, parse_fail_rate=0.0,
                            coverage_pct=55.0, target_coverage_pct=90.0)
    assert "Recalibrate" in out


def test_recommendations_clean_run_acceptance():
    m = _load_module()
    fake_stats = [m.EnvStats(env="env-x", n=10, mean_reward=0.6, std_reward=0.05, parse_fail_rate=0.0, coverage=0.91)]
    out = m.recommendations(fake_stats, overall_mean=0.6, parse_fail_rate=0.0,
                            coverage_pct=91.0, target_coverage_pct=90.0)
    assert "Suitable for downstream evaluation" in out


# ── end-to-end render ───────────────────────────────────


def test_render_report_produces_all_seven_sections(tmp_path):
    m = _load_module()
    rows = _synthetic_rows()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    out = tmp_path / "report.md"
    m.render_report(
        benchmark_csv=csv_path, model="test-model",
        output_md=out, output_pdf=None,
        template_path=TEMPLATE,
    )
    text = out.read_text()
    for header in (
        "## 1. Executive Summary",
        "## 2. Methodology",
        "## 3. Capability Assessment",
        "## 4. Failure Modes",
        "## 5. Calibration",
        "## 6. Recommendations",
        "## 7. Appendix",
    ):
        assert header in text, f"missing section: {header}"
    # No unfilled placeholders.
    assert "${" not in text


def test_render_report_mentions_target_model(tmp_path):
    m = _load_module()
    rows = _synthetic_rows("very-specific-model-name")
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    out = tmp_path / "report.md"
    m.render_report(
        benchmark_csv=csv_path, model="very-specific-model-name",
        output_md=out, output_pdf=None,
        template_path=TEMPLATE,
    )
    assert "very-specific-model-name" in out.read_text()


def test_cli_writes_markdown(tmp_path):
    rows = _synthetic_rows()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark-csv", str(csv_path),
            "--model", "test-model",
            "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "Verifiable Labs Compliance Report" in out.read_text()


def test_cli_errors_when_model_missing(tmp_path):
    rows = _synthetic_rows("present-model")
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    out = tmp_path / "report.md"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--benchmark-csv", str(csv_path),
            "--model", "missing-model",
            "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "not found" in proc.stderr or "not found" in proc.stdout


@pytest.mark.skipif(not shutil.which("pandoc"), reason="pandoc not installed")
def test_render_report_writes_pdf_when_requested(tmp_path):
    m = _load_module()
    rows = _synthetic_rows()
    csv_path = tmp_path / "bench.csv"
    _write_csv(csv_path, rows)
    md = tmp_path / "report.md"
    pdf = tmp_path / "report.pdf"
    m.render_report(
        benchmark_csv=csv_path, model="test-model",
        output_md=md, output_pdf=pdf,
        template_path=TEMPLATE,
    )
    assert pdf.exists() and pdf.stat().st_size > 1000  # non-empty PDF


def test_render_pdf_raises_when_no_renderer_available(tmp_path, monkeypatch):
    m = _load_module()
    md = tmp_path / "tmp.md"
    md.write_text("# hello")
    pdf = tmp_path / "tmp.pdf"
    monkeypatch.setattr(m.shutil, "which", lambda _: None)

    def _no_weasy(name, *a, **kw):
        raise ImportError("weasyprint not installed")

    monkeypatch.setattr(
        sys.modules["builtins"],
        "__import__",
        lambda name, *a, **kw: (
            _no_weasy(name) if name == "weasyprint" else __builtins__["__import__"](name, *a, **kw)
        ),
    )
    with pytest.raises(RuntimeError, match="neither pandoc nor weasyprint"):
        m.render_pdf(md, pdf)
