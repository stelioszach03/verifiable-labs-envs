"""Tests for scripts/preflight/wandb_dryrun.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "preflight" / "wandb_dryrun.py"


def _load():
    spec = importlib.util.spec_from_file_location("wandb_dryrun", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wb():
    return _load()


# ── fake-metric curve sanity ──────────────────────────────────────


def test_fake_metric_curve_decreases_loss(wb) -> None:
    first = wb.fake_metric_curve(0, 100)
    last = wb.fake_metric_curve(99, 100)
    assert last["loss"] < first["loss"]


def test_fake_metric_curve_lr_decays(wb) -> None:
    first = wb.fake_metric_curve(0, 100)
    last = wb.fake_metric_curve(99, 100)
    assert last["lr"] < first["lr"]


def test_fake_confusion_matrix_is_4x4(wb) -> None:
    cm = wb.fake_confusion_matrix()
    assert len(cm) == 4
    assert all(len(row) == 4 for row in cm)


# ── run_smoke + report shape ──────────────────────────────────────


def test_run_smoke_writes_json(wb, tmp_path) -> None:
    out = tmp_path / "wandb.json"
    report = wb.run_smoke(out=out, n_steps=20)
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_run_smoke_logs_all_steps(wb, tmp_path) -> None:
    report = wb.run_smoke(out=tmp_path / "wb.json", n_steps=50)
    assert report["n_metrics_logged"] == 50
    assert report["n_steps"] == 50


def test_run_smoke_offline_mode_default(wb, tmp_path) -> None:
    report = wb.run_smoke(out=tmp_path / "wb.json", n_steps=5)
    assert report["mode"] == "offline"


def test_run_smoke_loss_decreased_flag(wb, tmp_path) -> None:
    report = wb.run_smoke(out=tmp_path / "wb.json", n_steps=20)
    assert report["loss_decreased"] is True
    assert report["initial_loss"] > report["final_loss"]


def test_run_smoke_records_confusion_shape(wb, tmp_path) -> None:
    report = wb.run_smoke(out=tmp_path / "wb.json", n_steps=5)
    assert report["confusion_matrix_shape"] == [4, 4]


def test_main_cli_exits_zero(wb, tmp_path) -> None:
    rc = wb.main(
        [
            "--out",
            str(tmp_path / "main.json"),
            "--n-steps",
            "10",
            "--quiet",
        ]
    )
    assert rc == 0
    assert (tmp_path / "main.json").is_file()


def test_run_smoke_project_name_propagates(wb, tmp_path) -> None:
    report = wb.run_smoke(
        out=tmp_path / "wb.json", n_steps=5, project="custom-project"
    )
    assert report["project"] == "custom-project"


def test_run_smoke_short_run_is_short(wb, tmp_path) -> None:
    """Edge case: 1-step run still produces a valid report."""
    report = wb.run_smoke(out=tmp_path / "wb.json", n_steps=1)
    assert report["n_metrics_logged"] == 1
    assert report["initial_loss"] == report["final_loss"]
