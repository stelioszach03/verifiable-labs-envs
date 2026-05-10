"""Tests for scripts/preflight/smoke_phase30_*.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SCRIPT = REPO_ROOT / "scripts" / "preflight" / "smoke_phase30_trainer.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "preflight" / "smoke_phase30_eval.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer_mod():
    return _load(TRAINER_SCRIPT, "smoke_phase30_trainer")


@pytest.fixture(scope="module")
def eval_mod():
    return _load(EVAL_SCRIPT, "smoke_phase30_eval")


# ── trainer smoke ──────────────────────────────────────────────────


def test_p30_trainer_run_smoke_writes_json(trainer_mod, tmp_path) -> None:
    out = tmp_path / "phase30_trainer.json"
    report = trainer_mod.run_smoke(out=out)
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_p30_trainer_report_top_level_shape(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    assert report["phase"] == "30"
    assert report["track"] == "trainer"
    assert report["ok"] is True
    expected = {
        "dependencies",
        "config",
        "training_args",
        "run_card",
        "manifest",
        "upload",
        "list_local",
        "wandb",
    }
    assert expected.issubset(report.keys())


def test_p30_trainer_dependencies_section(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    deps = report["dependencies"]
    assert isinstance(deps["available"], list)
    assert isinstance(deps["missing"], list)
    assert isinstance(deps["is_satisfied"], bool)


def test_p30_trainer_multi_task_blend_sums_to_one(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    cfg = report["config"]
    assert cfg["weights_sum_to_one"] is True
    assert cfg["per_step_loss_weight"] == pytest.approx(0.7)
    assert cfg["outcome_weight"] == pytest.approx(0.3)


def test_p30_trainer_config_round_trip_succeeds(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    cfg = report["config"]
    assert cfg["round_trip_ok"] is True
    assert cfg["shared_backbone"] is False  # base_rm_checkpoint=None
    assert cfg["num_keys"] >= 18


def test_p30_trainer_training_args_emit(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    args = report["training_args"]
    assert args["n_keys"] > 0
    assert args["has_lr"] is True


def test_p30_trainer_run_card_written(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    rc = report["run_card"]
    assert rc["exists"] is True
    assert rc["size_bytes"] > 0
    assert Path(rc["path"]).is_file()


def test_p30_trainer_manifest_round_trip(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    m = report["manifest"]
    assert m["round_trip_ok"] is True
    assert m["model_id"].startswith("vlabs-prm-distilled-qwen-1-5b-v0.0.0-smoke")
    assert Path(m["path"]).is_file()


def test_p30_trainer_upload_uploads_manifest(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    up = report["upload"]
    assert up["n_uploaded"] >= 1
    assert up["is_real_r2_configured"] is False


def test_p30_trainer_list_local_finds_manifest(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    listed = report["list_local"]
    assert listed["n_checkpoints"] >= 1
    assert listed["first_model_id"].startswith("vlabs-prm")


def test_p30_trainer_wandb_section_has_flags(trainer_mod, tmp_path) -> None:
    report = trainer_mod.run_smoke(out=tmp_path / "p30t.json")
    w = report["wandb"]
    assert "is_wandb_available" in w
    assert "has_wandb_credentials" in w


def test_p30_trainer_main_returns_zero(trainer_mod, tmp_path) -> None:
    rc = trainer_mod.main(
        ["--out", str(tmp_path / "main.json"), "--quiet"]
    )
    assert rc == 0
    assert (tmp_path / "main.json").is_file()


# ── eval smoke ─────────────────────────────────────────────────────


def test_p30_eval_run_smoke_writes_json(eval_mod, tmp_path) -> None:
    out = tmp_path / "phase30_eval.json"
    report = eval_mod.run_smoke(out=out)
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_p30_eval_report_top_level_shape(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    assert report["phase"] == "30"
    assert report["track"] == "eval"
    assert {
        "stub_baseline_value",
        "processbench",
        "bon",
        "evaluate_bon_summary",
    }.issubset(report.keys())


def test_p30_eval_stubs_are_constant(eval_mod) -> None:
    assert eval_mod.stub_step_predictor("p", ["s"], 0) == 0.5
    assert eval_mod.stub_aggregate_predictor("p", ["s"]) == 0.5
    assert eval_mod.stub_rm_predictor("p", "c") == 0.5


def test_p30_eval_processbench_traces_match_n(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    pb = report["processbench"]
    assert pb["n_traces"] == 10
    # The stub PRM never marks any step as below threshold → it acts
    # as a "predict every trace fully correct" model. Accuracy ends up
    # equal to the fraction of fully-correct traces, which is roughly
    # half by the synthetic builder's design.
    assert 0.0 <= pb["overall_accuracy"] <= 1.0


def test_p30_eval_processbench_passes_flag_present(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    assert isinstance(report["processbench"]["passes"], bool)


def test_p30_eval_bon_problems_count(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    bon = report["bon"]
    assert bon["n_problems"] == 5
    assert bon["n_candidates_per_problem"] == 3
    assert len(bon["per_problem_choices"]) == 5


def test_p30_eval_bon_constant_aggregator_picks_index_zero(
    eval_mod, tmp_path
) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    # rerank_bon resolves ties to the lowest index → with a constant
    # aggregator every chosen_index must be 0.
    for choice in report["bon"]["per_problem_choices"]:
        assert choice["chosen_index"] == 0
        assert choice["chosen_aggregate"] == 0.5


def test_p30_eval_lift_metrics_keys(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    metrics = report["bon"]["metrics"]
    assert "single_accuracy" in metrics
    assert "prm_bon_accuracy" in metrics


def test_p30_eval_evaluate_bon_summary_present(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    assert isinstance(report["evaluate_bon_summary"], dict)


def test_p30_eval_main_returns_zero(eval_mod, tmp_path) -> None:
    rc = eval_mod.main(["--out", str(tmp_path / "main.json"), "--quiet"])
    assert rc == 0
    assert (tmp_path / "main.json").is_file()


def test_p30_eval_stub_baseline_locked(eval_mod, tmp_path) -> None:
    report = eval_mod.run_smoke(out=tmp_path / "p30e.json")
    assert report["stub_baseline_value"] == 0.5
