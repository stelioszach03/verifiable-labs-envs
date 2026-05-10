"""Unit tests for the ``extract-processbench`` Typer subcommand.

Pinning the CLI wire from ``vlabs-prm-data`` through to
:func:`load_processbench_subset` (Qwen/ProcessBench). The 30-test
suite in ``tests/test_process_reward_eval.py`` covers the adapter
internals; these tests exercise the CLI plumbing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vlabs_prm_data.cli import app

runner = CliRunner()


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_extract_processbench_help_returns_zero() -> None:
    """``--help`` for the subcommand executes cleanly. We don't grep
    flag names because Typer's rich help wraps long flags mid-token at
    narrow widths and the runner doesn't expose terminal-width
    overrides."""
    result = runner.invoke(app, ["extract-processbench", "--help"])
    assert result.exit_code == 0
    assert "extract-processbench" in result.stdout.lower()


def test_extract_processbench_top_level_help_lists_command() -> None:
    """The new subcommand is discoverable from the top-level CLI."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "extract-processbench" in result.stdout.lower()


def test_extract_processbench_synthetic_fallback_writes_n_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the synthetic fallback (no network); verify N rows + JSONL shape."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "pb.jsonl"
    result = runner.invoke(
        app,
        [
            "extract-processbench",
            "--n", "5",
            "--seed", "0",
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    rows = _read_jsonl(output)
    assert len(rows) == 5


def test_extract_processbench_no_synthetic_fallback_hard_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-synthetic-fallback`` raises rather than sliding into the
    synthetic generator when the dataset is missing."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "pb.jsonl"
    result = runner.invoke(
        app,
        [
            "extract-processbench",
            "--n", "3",
            "--no-synthetic-fallback",
            "--output", str(output),
        ],
    )
    assert result.exit_code != 0


def test_extract_processbench_jsonl_row_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each row carries the canonical ProcessBench fields."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "pb.jsonl"
    result = runner.invoke(
        app, ["extract-processbench", "--n", "3", "--output", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    rows = _read_jsonl(output)
    assert len(rows) == 3
    for row in rows:
        for key in (
            "problem",
            "steps",
            "first_error_step",
            "subset",
            "trace_id",
            "source",
        ):
            assert key in row, f"missing {key} in row: {row!r}"
        assert row["source"] == "processbench"
        assert isinstance(row["steps"], list)


def test_extract_processbench_emits_per_subset_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-subset counts land in stdout for downstream tooling."""
    import importlib

    real_import = importlib.import_module

    def _stub(name: str, package: str | None = None):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("datasets")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)

    output = tmp_path / "pb.jsonl"
    result = runner.invoke(
        app, ["extract-processbench", "--n", "5", "--output", str(output)]
    )
    assert result.exit_code == 0
    assert "per_subset" in result.stdout


def test_extract_processbench_seed_determinism(
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
        app,
        [
            "extract-processbench",
            "--n", "4",
            "--seed", "42",
            "--output", str(out_a),
        ],
    )
    runner.invoke(
        app,
        [
            "extract-processbench",
            "--n", "4",
            "--seed", "42",
            "--output", str(out_b),
        ],
    )
    assert _read_jsonl(out_a) == _read_jsonl(out_b)


def test_extract_processbench_zero_n_writes_empty_file(tmp_path: Path) -> None:
    """``--n 0`` is valid; produces an empty JSONL."""
    output = tmp_path / "empty.jsonl"
    result = runner.invoke(
        app, ["extract-processbench", "--n", "0", "--output", str(output)]
    )
    assert result.exit_code == 0
    assert output.exists()
    assert output.read_text() == ""


def test_extract_processbench_creates_parent_directory(
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

    output = tmp_path / "deep" / "nest" / "pb.jsonl"
    assert not output.parent.exists()
    result = runner.invoke(
        app, ["extract-processbench", "--n", "2", "--output", str(output)]
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()


def test_extract_processbench_subset_filter_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``--subset`` flag reaches load_processbench_subset."""
    captured: dict[str, Any] = {}

    def _fake_load(*, n: int, seed: int = 0, subset: str = "all", fallback_to_synthetic: bool = True):
        captured.update(
            n=n, seed=seed, subset=subset, fallback_to_synthetic=fallback_to_synthetic
        )
        return []

    from verifiable_labs_envs.process_reward import eval as pb_eval

    monkeypatch.setattr(pb_eval, "load_processbench_subset", _fake_load)

    output = tmp_path / "pb.jsonl"
    result = runner.invoke(
        app,
        [
            "extract-processbench",
            "--n", "5",
            "--subset", "gsm8k",
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["subset"] == "gsm8k"
    assert captured["n"] == 5
