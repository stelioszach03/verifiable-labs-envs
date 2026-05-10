"""Tests for scripts/preflight/smoke_phase29_*.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SCRIPT = REPO_ROOT / "scripts" / "preflight" / "smoke_phase29_trainer.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "preflight" / "smoke_phase29_eval.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer_mod():
    return _load(TRAINER_SCRIPT, "smoke_phase29_trainer")


@pytest.fixture(scope="module")
def eval_mod():
    return _load(EVAL_SCRIPT, "smoke_phase29_eval")


# ── trainer smoke ──────────────────────────────────────────────────


def test_trainer_run_smoke_writes_json(trainer_mod, tmp_path) -> None:
    out = tmp_path / "phase29_trainer.json"
    report = trainer_mod.run_smoke(out=out)
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_trainer_report_top_level_shape(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    assert report["phase"] == "29"
    assert report["track"] == "trainer"
    assert report["ok"] is True
    expected_keys = {
        "dependencies",
        "config",
        "data",
        "wandb",
        "manifest",
        "upload",
        "list_local",
        "fake_steps_executed",
        "loss_proxy",
    }
    assert expected_keys.issubset(report.keys())


def test_trainer_dependencies_section_has_lists(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    deps = report["dependencies"]
    assert isinstance(deps["available"], list)
    assert isinstance(deps["missing"], list)
    assert isinstance(deps["is_satisfied"], bool)


def test_trainer_config_round_trip_succeeds(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    cfg = report["config"]
    assert cfg["round_trip_ok"] is True
    assert cfg["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["num_keys"] >= 15


def test_trainer_synthetic_data_yields_10_rows(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    assert report["data"]["n_rows"] == 10
    assert report["data"]["first_env_id"] == "math-algebra"


def test_trainer_wandb_offline_mode(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    w = report["wandb"]
    assert w["mode"] == "offline"
    assert w["project"] == "vlabs-preflight-smoke"


def test_trainer_manifest_round_trip(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    m = report["manifest"]
    assert m["round_trip_ok"] is True
    assert m["model_id"].startswith("vlabs-reward-distilled-qwen-1-5b-v0.0.0-smoke")
    assert Path(m["path"]).is_file()


def test_trainer_upload_picks_up_manifest(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    up = report["upload"]
    assert up["n_uploaded"] >= 1
    assert any("manifest" in k for k in up["uploaded_keys"])
    assert up["is_real_r2_configured"] is False


def test_trainer_list_local_finds_manifest(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    listed = report["list_local"]
    assert listed["n_checkpoints"] >= 1
    assert listed["first_model_id"].startswith("vlabs-reward")


def test_trainer_fake_steps_recorded(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p29t.json")
    assert report["fake_steps_executed"] == 5
    assert report["loss_proxy"] == 0.5


def test_trainer_main_returns_zero(trainer_mod, tmp_path, capsys) -> None:
    rc = trainer_mod.main(
        ["--out", str(tmp_path / "main.json"), "--quiet"]
    )
    assert rc == 0
    assert (tmp_path / "main.json").is_file()


# ── eval smoke ─────────────────────────────────────────────────────


def test_eval_run_smoke_writes_json(eval_mod, tmp_path) -> None:
    out = tmp_path / "phase29_eval.json"
    report = eval_mod.run_smoke(out=out)
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_eval_report_top_level_shape(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p29e.json")
    assert report["phase"] == "29"
    assert report["track"] == "eval"
    assert report["ok"] is True
    assert {"held_out_eval", "rank_correlation", "calibration_mse"}.issubset(
        report.keys()
    )


def test_eval_stub_student_returns_constant(eval_mod) -> None:
    """Sanity: the stub student is locked at 0.5 (used as the smoke
    baseline value across the report)."""
    assert eval_mod.stub_student("p", "c") == 0.5


def test_eval_constant_student_yields_zero_rho(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p29e.json")
    rho = report["rank_correlation"]["rho_constant_student"]
    # Constant predictions → rho is exactly 0 (stable for our seed).
    assert abs(rho) < 1e-9


def test_eval_calibration_mse_is_small(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p29e.json")
    assert report["calibration_mse"]["value"] >= 0.0
    # Synthetic widths are within 0.02 of each other → MSE ~ 1e-4 at most.
    assert report["calibration_mse"]["value"] < 1e-2


def test_eval_held_out_eval_section_present(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p29e.json")
    held = report["held_out_eval"]
    assert isinstance(held, dict)
    # Whatever fields the upstream report exports are surfaced as a
    # dict — the smoke just confirms the conversion didn't drop them.
    assert len(held) > 0


def test_eval_n_envs_and_episodes(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p29e.json")
    assert report["n_envs"] == 1
    assert report["n_episodes_per_env"] == 5


def test_eval_main_returns_zero(eval_mod, tmp_path) -> None:
    rc = eval_mod.main(["--out", str(tmp_path / "main.json"), "--quiet"])
    assert rc == 0
    assert (tmp_path / "main.json").is_file()


def test_eval_stub_baseline_value(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p29e.json")
    assert report["stub_baseline_value"] == 0.5
