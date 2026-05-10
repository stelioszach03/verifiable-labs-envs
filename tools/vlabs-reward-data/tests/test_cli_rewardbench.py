"""Unit tests for the ``extract-rewardbench`` Typer subcommand.

Pinning the wire from the dataset CLI through to
:func:`load_rewardbench_subset`: covers the synthetic-fallback path
(no network), the explicit ``--no-synthetic-fallback`` gate, JSONL
shape, the per-category summary, and seed determinism. The 18-test
``test_reward_distillation_rewardbench.py`` suite covers the adapter
itself; these tests cover the CLI plumbing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vlabs_reward_data.cli import app

runner = CliRunner()


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_extract_rewardbench_help_lists_command() -> None:
    """The new subcommand is wired into the top-level CLI help text."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "extract-rewardbench" in result.stdout.lower()


def test_extract_rewardbench_help_returns_zero() -> None:
    """``--help`` for the subcommand executes cleanly.

    We deliberately do NOT regex-match help-text option names —
    Typer's rich renderer wraps long flag names mid-token at narrow
    widths, and the ``CliRunner`` doesn't expose a stable terminal-
    width override. The command shape is covered by every other test
    in this file (each invokes the subcommand with explicit flags),
    so this test is a pure smoke of ``--help`` itself.
    """
    result = runner.invoke(app, ["extract-rewardbench", "--help"])
    assert result.exit_code == 0
    # Sanity: the command name must at least appear somewhere.
    assert "extract-rewardbench" in result.stdout.lower()


def test_extract_rewardbench_synthetic_fallback_writes_n_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the synthetic fallback (no network); verify N rows + JSONL shape.

    We monkey-patch ``importlib.import_module`` to raise
    ``ImportError("datasets")`` so the adapter falls back to
    :func:`build_synthetic_rewardbench`. This isolates the CLI from
    network conditions.
    """
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "rb.jsonl"
    result = runner.invoke(
        app,
        [
            "extract-rewardbench",
            "--n", "8",
            "--seed", "0",
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = _read_jsonl(output)
    assert len(rows) == 8


def test_extract_rewardbench_no_synthetic_fallback_hard_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-synthetic-fallback`` must raise instead of silently
    sliding into the synthetic generator when the dataset is missing."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "rb.jsonl"
    result = runner.invoke(
        app,
        [
            "extract-rewardbench",
            "--n", "5",
            "--no-synthetic-fallback",
            "--output", str(output),
        ],
    )
    # Typer wraps the RuntimeError in a non-zero exit; capturing the
    # exception handler chain is enough.
    assert result.exit_code != 0


def test_extract_rewardbench_jsonl_row_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each row carries the canonical (prompt, chosen, rejected,
    category, pair_id, source) fields."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "rb.jsonl"
    result = runner.invoke(
        app, ["extract-rewardbench", "--n", "4", "--output", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    rows = _read_jsonl(output)
    assert len(rows) == 4
    for row in rows:
        for key in ("prompt", "chosen", "rejected", "category", "pair_id", "source"):
            assert key in row, f"missing {key} in row: {row!r}"
        assert row["source"] == "rewardbench"


def test_extract_rewardbench_emits_per_category_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Per-category counts land in stdout for downstream tooling."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "rb.jsonl"
    result = runner.invoke(
        app, ["extract-rewardbench", "--n", "8", "--output", str(output)]
    )
    assert result.exit_code == 0
    assert "per_category" in result.stdout


def test_extract_rewardbench_seed_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same seed → identical row ordering."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    runner.invoke(
        app, ["extract-rewardbench", "--n", "6", "--seed", "42", "--output", str(out_a)]
    )
    runner.invoke(
        app, ["extract-rewardbench", "--n", "6", "--seed", "42", "--output", str(out_b)]
    )
    assert _read_jsonl(out_a) == _read_jsonl(out_b)


def test_extract_rewardbench_zero_n_writes_empty_file(
    tmp_path: Path,
) -> None:
    """``--n 0`` is valid; produces an empty JSONL (header-less)."""
    output = tmp_path / "empty.jsonl"
    result = runner.invoke(
        app, ["extract-rewardbench", "--n", "0", "--output", str(output)]
    )
    assert result.exit_code == 0
    assert output.exists()
    assert output.read_text() == ""


def test_extract_rewardbench_creates_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nested output path's parent gets ``mkdir -p``ed by the CLI."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "deeply" / "nested" / "rb.jsonl"
    assert not output.parent.exists()
    result = runner.invoke(
        app, ["extract-rewardbench", "--n", "3", "--output", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()


def test_extract_rewardbench_subset_filter_kwarg_is_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``--subset`` flag reaches load_rewardbench_subset."""
    captured: dict[str, Any] = {}

    def _fake_load(*, n: int, seed: int = 0, subset: str = "all", fallback_to_synthetic: bool = True):
        captured.update(
            n=n, seed=seed, subset=subset, fallback_to_synthetic=fallback_to_synthetic
        )
        return []

    from verifiable_labs_envs.reward_distillation import rewardbench_adapter

    monkeypatch.setattr(
        rewardbench_adapter, "load_rewardbench_subset", _fake_load
    )

    output = tmp_path / "rb.jsonl"
    result = runner.invoke(
        app,
        [
            "extract-rewardbench",
            "--n", "5",
            "--subset", "safety",
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0
    assert captured["subset"] == "safety"
    assert captured["n"] == 5
