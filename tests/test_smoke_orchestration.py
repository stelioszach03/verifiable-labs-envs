"""Tests for the Phase-18 smoke-test orchestration scripts.

Covers:
  - scripts/smoke_test_experiment.py
  - scripts/check_pipeline_health.py
  - scripts/run_all_experiments.sh (resume / state-file behaviour)

The scripts are loaded as modules via ``importlib`` (mirroring the existing
``tests/preflight/test_smoke_phase29.py`` style) so we can exercise their
internal functions without invoking heavy training paths.

External heavy deps (``torch``, ``vllm`` …) are stubbed via ``types.SimpleNamespace``
or ``unittest.mock.MagicMock`` so the suite stays GPU-free + import-light.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_test_experiment.py"
HEALTH_SCRIPT = REPO_ROOT / "scripts" / "check_pipeline_health.py"
ORCHESTRATOR = REPO_ROOT / "scripts" / "run_all_experiments.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke_mod():
    return _load(SMOKE_SCRIPT, "phase18_smoke_test_experiment")


@pytest.fixture(scope="module")
def health_mod():
    return _load(HEALTH_SCRIPT, "phase18_check_pipeline_health")


def _fake_torch_cuda(
    *,
    available: bool = True,
    vram_bytes: int = 32 * 1024 ** 3,
    name: str = "NVIDIA RTX 5090",
    memory_allocated: int = 0,
) -> types.SimpleNamespace:
    """Build a minimal stub of the ``torch`` module sufficient for the checks."""
    props = types.SimpleNamespace(total_memory=vram_bytes, name=name)
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_properties=lambda _idx: props,
        memory_allocated=lambda _idx=0: memory_allocated,
    )
    return types.SimpleNamespace(cuda=cuda)


# ---------------------------------------------------------------------------
# Smoke module — registry shape
# ---------------------------------------------------------------------------
def test_registry_has_phase18_redo(smoke_mod) -> None:
    assert "phase18-redo" in smoke_mod.EXPERIMENT_REGISTRY


def test_registry_has_E1_through_E10(smoke_mod) -> None:
    keys = set(smoke_mod.EXPERIMENT_REGISTRY)
    expected = {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E10"}
    missing = expected - keys
    assert not missing, f"missing experiments: {sorted(missing)}"


def test_registry_does_not_include_E9(smoke_mod) -> None:
    # E9 is deliberately omitted from the protocol — guard against silent drift.
    assert "E9" not in smoke_mod.EXPERIMENT_REGISTRY


def test_registry_specs_are_frozen(smoke_mod) -> None:
    spec = smoke_mod.EXPERIMENT_REGISTRY["E1"]
    with pytest.raises(Exception):
        spec.min_vram_gb = 99.0  # type: ignore[misc]


@pytest.mark.parametrize("exp_id", ["phase18-redo", "E1", "E4", "E7", "E10"])
def test_registry_specs_have_required_fields(smoke_mod, exp_id) -> None:
    spec = smoke_mod.EXPERIMENT_REGISTRY[exp_id]
    assert spec.experiment_id == exp_id
    assert isinstance(spec.description, str) and spec.description
    assert isinstance(spec.required_deps, tuple) and "torch" in spec.required_deps
    assert spec.min_vram_gb > 0
    assert spec.min_disk_gb > 0
    assert spec.estimated_full_cost_usd > 0


def test_allowed_experiments_sorted_tuple(smoke_mod) -> None:
    assert smoke_mod.ALLOWED_EXPERIMENTS == tuple(
        sorted(smoke_mod.EXPERIMENT_REGISTRY)
    )
    assert isinstance(smoke_mod.ALLOWED_EXPERIMENTS, tuple)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
def test_exit_codes_are_0_through_4(smoke_mod) -> None:
    codes = {
        smoke_mod.EXIT_OK,
        smoke_mod.EXIT_SETUP,
        smoke_mod.EXIT_REWARD_ZERO,
        smoke_mod.EXIT_OOM,
        smoke_mod.EXIT_JSON_FAIL,
    }
    assert codes == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# check_experiment_registered
# ---------------------------------------------------------------------------
def test_check_experiment_registered_valid(smoke_mod) -> None:
    result = smoke_mod.check_experiment_registered("phase18-redo")
    assert result.ok is True
    assert "phase18-redo" in result.detail


def test_check_experiment_registered_invalid(smoke_mod) -> None:
    result = smoke_mod.check_experiment_registered("not-a-real-experiment")
    assert result.ok is False
    assert "unknown" in result.detail.lower()


def test_check_experiment_registered_custom_registry(smoke_mod) -> None:
    custom = {"X1": smoke_mod.EXPERIMENT_REGISTRY["phase18-redo"]}
    result = smoke_mod.check_experiment_registered("X1", registry=custom)
    assert result.ok is True


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------
def test_check_dependencies_empty_passes(smoke_mod) -> None:
    result = smoke_mod.check_dependencies(())
    assert result.ok is True


def test_check_dependencies_stdlib_passes(smoke_mod) -> None:
    # sys + json are always importable.
    result = smoke_mod.check_dependencies(("sys", "json"))
    assert result.ok is True


def test_check_dependencies_single_missing(smoke_mod) -> None:
    result = smoke_mod.check_dependencies(("nonexistent_pkg_zzz_phase18",))
    assert result.ok is False
    assert "nonexistent_pkg_zzz_phase18" in result.detail


def test_check_dependencies_partial_missing(smoke_mod) -> None:
    result = smoke_mod.check_dependencies(("sys", "nonexistent_pkg_zzz_phase18"))
    assert result.ok is False
    assert "nonexistent_pkg_zzz_phase18" in result.detail


# ---------------------------------------------------------------------------
# check_disk_free
# ---------------------------------------------------------------------------
def test_check_disk_free_ok(smoke_mod, tmp_path) -> None:
    result = smoke_mod.check_disk_free(tmp_path, min_gb=0.0001)
    assert result.ok is True


def test_check_disk_free_low(smoke_mod, monkeypatch, tmp_path) -> None:
    fake = types.SimpleNamespace(total=100, used=100, free=0)
    monkeypatch.setattr(smoke_mod.shutil, "disk_usage", lambda _p: fake)
    result = smoke_mod.check_disk_free(tmp_path, min_gb=1.0)
    assert result.ok is False
    assert "required" in result.detail


def test_check_disk_free_missing_path(smoke_mod, monkeypatch, tmp_path) -> None:
    def boom(_p):
        raise FileNotFoundError("no such path")

    monkeypatch.setattr(smoke_mod.shutil, "disk_usage", boom)
    result = smoke_mod.check_disk_free(tmp_path / "no-such", min_gb=0.0)
    assert result.ok is False
    assert "does not exist" in result.detail


# ---------------------------------------------------------------------------
# check_gpu_available
# ---------------------------------------------------------------------------
def test_check_gpu_available_no_cuda(smoke_mod) -> None:
    fake_torch = _fake_torch_cuda(available=False)
    result = smoke_mod.check_gpu_available(24.0, torch_=fake_torch)
    assert result.ok is False
    assert "cuda" in result.detail.lower()


def test_check_gpu_available_low_vram(smoke_mod) -> None:
    fake_torch = _fake_torch_cuda(vram_bytes=8 * 1024 ** 3, name="GTX 1080")
    result = smoke_mod.check_gpu_available(24.0, torch_=fake_torch)
    assert result.ok is False
    assert "VRAM" in result.detail


def test_check_gpu_available_ok(smoke_mod) -> None:
    fake_torch = _fake_torch_cuda(vram_bytes=32 * 1024 ** 3, name="RTX 5090")
    result = smoke_mod.check_gpu_available(24.0, torch_=fake_torch)
    assert result.ok is True
    assert "RTX 5090" in result.detail


def test_check_gpu_available_uses_lazy_import_when_torch_param_none(smoke_mod) -> None:
    # When torch_=None and real torch isn't available, the check fails
    # cleanly rather than raising ImportError.
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("real torch is installed — lazy-import path not exercisable here")
    result = smoke_mod.check_gpu_available(24.0)
    assert result.ok is False
    assert "torch" in result.detail.lower()


# ---------------------------------------------------------------------------
# check_runner_importable
# ---------------------------------------------------------------------------
def test_check_runner_importable_deferred_when_none(smoke_mod) -> None:
    result = smoke_mod.check_runner_importable(None, None)
    assert result.ok is True
    assert "deferred" in result.detail.lower()


def test_check_runner_importable_module_only_present(smoke_mod) -> None:
    result = smoke_mod.check_runner_importable("json", None)
    assert result.ok is True
    assert "json" in result.detail


def test_check_runner_importable_callable_present(smoke_mod) -> None:
    result = smoke_mod.check_runner_importable("json", "dumps")
    assert result.ok is True
    assert "json.dumps" in result.detail


def test_check_runner_importable_module_missing(smoke_mod) -> None:
    result = smoke_mod.check_runner_importable("nope_module_zzz_phase18", "run")
    assert result.ok is False
    assert "not importable" in result.detail


def test_check_runner_importable_callable_missing(smoke_mod) -> None:
    result = smoke_mod.check_runner_importable(
        "json", "nope_callable_zzz_phase18"
    )
    assert result.ok is False
    assert "nope_callable_zzz_phase18" in result.detail


# ---------------------------------------------------------------------------
# estimate_cost_usd
# ---------------------------------------------------------------------------
def test_estimate_cost_usd_zero(smoke_mod) -> None:
    assert smoke_mod.estimate_cost_usd(0.0) == 0.0


def test_estimate_cost_usd_one_hour(smoke_mod) -> None:
    assert smoke_mod.estimate_cost_usd(3600.0, hourly_rate_usd=0.99) == pytest.approx(0.99)


def test_estimate_cost_usd_negative_raises(smoke_mod) -> None:
    with pytest.raises(ValueError):
        smoke_mod.estimate_cost_usd(-1.0)


# ---------------------------------------------------------------------------
# run_smoke
# ---------------------------------------------------------------------------
def test_run_smoke_unknown_experiment_returns_exit_setup(smoke_mod, tmp_path) -> None:
    report = smoke_mod.run_smoke("not-real-zzz", disk_check_path=tmp_path)
    assert report.ok is False
    assert report.exit_code == smoke_mod.EXIT_SETUP
    assert report.checks[0].name == "experiment_registered"
    assert report.checks[0].ok is False


def test_run_smoke_max_steps_zero_raises(smoke_mod) -> None:
    with pytest.raises(ValueError):
        smoke_mod.run_smoke("phase18-redo", max_steps=0)


def test_run_smoke_happy_path_with_mocked_gpu(smoke_mod, tmp_path) -> None:
    fake_torch = _fake_torch_cuda(vram_bytes=32 * 1024 ** 3)
    # Custom registry with an experiment whose only dep is the stdlib
    # `json` package so the dependency check succeeds without torch/vllm.
    custom = {
        "T1": smoke_mod.ExperimentSpec(
            experiment_id="T1",
            description="test-only experiment",
            runner_module=None,
            runner_callable=None,
            required_deps=("json",),
            min_vram_gb=1.0,
            min_disk_gb=0.0001,
            estimated_full_cost_usd=0.1,
        )
    }
    report = smoke_mod.run_smoke(
        "T1",
        disk_check_path=tmp_path,
        registry=custom,
        torch_=fake_torch,
    )
    assert report.ok is True
    assert report.exit_code == smoke_mod.EXIT_OK
    assert all(c.ok for c in report.checks)
    # Without dry-run-after, runner check is not in the list.
    names = [c.name for c in report.checks]
    assert "runner_importable" not in names


def test_run_smoke_includes_runner_check_with_dry_run_after(smoke_mod, tmp_path) -> None:
    fake_torch = _fake_torch_cuda()
    custom = {
        "T1": smoke_mod.ExperimentSpec(
            experiment_id="T1",
            description="test-only experiment",
            runner_module="json",
            runner_callable="dumps",
            required_deps=("json",),
            min_vram_gb=1.0,
            min_disk_gb=0.0001,
            estimated_full_cost_usd=0.1,
        )
    }
    report = smoke_mod.run_smoke(
        "T1",
        disk_check_path=tmp_path,
        registry=custom,
        torch_=fake_torch,
        dry_run_after=True,
    )
    assert report.ok is True
    assert any(c.name == "runner_importable" for c in report.checks)


def test_run_smoke_dependency_failure_returns_exit_setup(smoke_mod, tmp_path) -> None:
    fake_torch = _fake_torch_cuda()
    custom = {
        "T1": smoke_mod.ExperimentSpec(
            experiment_id="T1",
            description="dep-fail test",
            runner_module=None,
            runner_callable=None,
            required_deps=("nonexistent_pkg_zzz_phase18",),
            min_vram_gb=1.0,
            min_disk_gb=0.0001,
            estimated_full_cost_usd=0.1,
        )
    }
    report = smoke_mod.run_smoke(
        "T1",
        disk_check_path=tmp_path,
        registry=custom,
        torch_=fake_torch,
    )
    assert report.ok is False
    assert report.exit_code == smoke_mod.EXIT_SETUP


def test_run_smoke_to_dict_shape(smoke_mod, tmp_path) -> None:
    fake_torch = _fake_torch_cuda()
    custom = {
        "T1": smoke_mod.ExperimentSpec(
            experiment_id="T1",
            description="shape test",
            runner_module=None,
            runner_callable=None,
            required_deps=("json",),
            min_vram_gb=1.0,
            min_disk_gb=0.0001,
            estimated_full_cost_usd=0.1,
        )
    }
    report = smoke_mod.run_smoke(
        "T1",
        disk_check_path=tmp_path,
        registry=custom,
        torch_=fake_torch,
    )
    d = report.to_dict()
    expected = {
        "experiment_id",
        "started_at",
        "finished_at",
        "duration_s",
        "ok",
        "exit_code",
        "estimated_cost_usd",
        "checks",
    }
    assert expected <= set(d)
    assert isinstance(d["checks"], list)
    assert all({"name", "ok", "detail"} <= set(c) for c in d["checks"])


def test_report_output_path_default(smoke_mod) -> None:
    p = smoke_mod.report_output_path("E1")
    assert p.name == "smoke_E1.json"
    assert "preflight" in str(p)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def test_cli_rejects_unknown_experiment(smoke_mod) -> None:
    parser = smoke_mod._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--experiment", "not-real-zzz"])


def test_cli_accepts_phase18_redo(smoke_mod) -> None:
    parser = smoke_mod._build_parser()
    args = parser.parse_args(["--experiment", "phase18-redo"])
    assert args.experiment == "phase18-redo"
    assert args.max_steps == 5
    assert args.dry_run_after is False
    assert args.hourly_rate_usd == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# health module — disk
# ---------------------------------------------------------------------------
def test_health_disk_free_ok(health_mod, tmp_path) -> None:
    result = health_mod.check_disk_free(tmp_path, min_gb=0.0001)
    assert result.ok is True
    assert result.status == "ok"


def test_health_disk_free_low(health_mod, monkeypatch, tmp_path) -> None:
    fake = types.SimpleNamespace(total=100, used=100, free=0)
    monkeypatch.setattr(health_mod.shutil, "disk_usage", lambda _p: fake)
    result = health_mod.check_disk_free(tmp_path, min_gb=1.0)
    assert result.ok is False
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# health module — gpu memory
# ---------------------------------------------------------------------------
def test_health_gpu_memory_skipped_when_no_cuda(health_mod) -> None:
    fake_torch = _fake_torch_cuda(available=False)
    result = health_mod.check_gpu_memory_recovered(1.0, torch_=fake_torch)
    assert result.ok is True
    assert result.status == "skipped"


def test_health_gpu_memory_ok(health_mod) -> None:
    fake_torch = _fake_torch_cuda(available=True, memory_allocated=0)
    result = health_mod.check_gpu_memory_recovered(1.0, torch_=fake_torch)
    assert result.ok is True
    assert result.status == "ok"


def test_health_gpu_memory_over_limit(health_mod) -> None:
    fake_torch = _fake_torch_cuda(
        available=True, memory_allocated=4 * 1024 ** 3
    )
    result = health_mod.check_gpu_memory_recovered(1.0, torch_=fake_torch)
    assert result.ok is False
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# health module — coverage drift
# ---------------------------------------------------------------------------
def test_health_coverage_drift_skipped_if_no_files(health_mod, tmp_path) -> None:
    result = health_mod.check_coverage_drift(
        tmp_path / "baseline.json", tmp_path / "latest.json", 0.05
    )
    assert result.ok is True
    assert result.status == "skipped"


def test_health_coverage_drift_within_limit(health_mod, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    latest = tmp_path / "latest.json"
    baseline.write_text(json.dumps({"sparse-fourier-recovery": 0.901}))
    latest.write_text(json.dumps({"sparse-fourier-recovery": 0.910}))
    result = health_mod.check_coverage_drift(baseline, latest, max_drift_frac=0.05)
    assert result.ok is True


def test_health_coverage_drift_over_limit(health_mod, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    latest = tmp_path / "latest.json"
    baseline.write_text(json.dumps({"phase-retrieval": 0.901}))
    latest.write_text(json.dumps({"phase-retrieval": 0.800}))
    result = health_mod.check_coverage_drift(baseline, latest, max_drift_frac=0.05)
    assert result.ok is False
    assert "phase-retrieval" in result.detail


def test_health_coverage_drift_malformed_json(health_mod, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    latest = tmp_path / "latest.json"
    baseline.write_text("not-json{")
    latest.write_text("{}")
    result = health_mod.check_coverage_drift(baseline, latest, max_drift_frac=0.05)
    assert result.ok is False


def test_health_coverage_drift_skipped_when_no_overlap(health_mod, tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    latest = tmp_path / "latest.json"
    baseline.write_text(json.dumps({"env-a": 0.9}))
    latest.write_text(json.dumps({"env-b": 0.9}))
    result = health_mod.check_coverage_drift(baseline, latest, max_drift_frac=0.05)
    assert result.ok is True
    assert result.status == "skipped"


# ---------------------------------------------------------------------------
# health module — checkpoint
# ---------------------------------------------------------------------------
def test_health_checkpoint_skipped_when_missing(health_mod, tmp_path) -> None:
    result = health_mod.check_checkpoint_loadable(tmp_path / "no-such")
    assert result.ok is True
    assert result.status == "skipped"


def test_health_checkpoint_skipped_when_empty(health_mod, tmp_path) -> None:
    result = health_mod.check_checkpoint_loadable(tmp_path)
    assert result.ok is True
    assert result.status == "skipped"


def test_health_checkpoint_fails_with_no_weights(health_mod, tmp_path) -> None:
    ckpt = tmp_path / "ckpt_001"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    result = health_mod.check_checkpoint_loadable(tmp_path)
    assert result.ok is False
    assert "no recognised weight file" in result.detail


def test_health_checkpoint_ok_with_safetensors(health_mod, tmp_path) -> None:
    ckpt = tmp_path / "ckpt_001"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"fake-weights-but-non-empty")
    result = health_mod.check_checkpoint_loadable(tmp_path)
    assert result.ok is True
    assert "model.safetensors" in result.detail


# ---------------------------------------------------------------------------
# health module — orchestration
# ---------------------------------------------------------------------------
def test_run_health_smoke(health_mod, tmp_path) -> None:
    fake_torch = _fake_torch_cuda(available=False)
    report = health_mod.run_health(
        disk_path=tmp_path,
        min_disk_gb=0.0001,
        checkpoint_dir=tmp_path / "no-ckpts",
        coverage_baseline=tmp_path / "no-baseline.json",
        coverage_latest=tmp_path / "no-latest.json",
        torch_=fake_torch,
    )
    assert report.ok is True
    assert report.exit_code == health_mod.EXIT_OK
    names = [c.name for c in report.checks]
    assert "disk_free" in names
    assert "gpu_memory_recovered" in names
    assert "coverage_drift" in names
    assert "checkpoint_loadable" in names


def test_run_health_propagates_disk_failure(health_mod, monkeypatch, tmp_path) -> None:
    fake = types.SimpleNamespace(total=100, used=100, free=0)
    monkeypatch.setattr(health_mod.shutil, "disk_usage", lambda _p: fake)
    fake_torch = _fake_torch_cuda(available=False)
    report = health_mod.run_health(
        disk_path=tmp_path,
        min_disk_gb=10.0,
        checkpoint_dir=tmp_path / "no-ckpts",
        coverage_baseline=tmp_path / "no-baseline.json",
        coverage_latest=tmp_path / "no-latest.json",
        torch_=fake_torch,
    )
    assert report.ok is False
    assert report.exit_code == health_mod.EXIT_FAIL


# ---------------------------------------------------------------------------
# Orchestrator shell script — surface contract.
# ---------------------------------------------------------------------------
def test_orchestrator_script_exists() -> None:
    assert ORCHESTRATOR.is_file()


def test_orchestrator_lists_all_experiments() -> None:
    body = ORCHESTRATOR.read_text()
    # Order matters: phase18-redo must come first.
    idx_p18 = body.find('"phase18-redo"')
    assert idx_p18 != -1
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E10"):
        idx = body.find(f'"{exp}"')
        assert idx != -1, f"orchestrator missing {exp}"
        assert idx > idx_p18, f"{exp} listed before phase18-redo"
    # E9 must be absent.
    assert '"E9"' not in body


def test_orchestrator_sets_safe_bash_options() -> None:
    body = ORCHESTRATOR.read_text()
    assert "set -euo pipefail" in body


def test_orchestrator_calls_smoke_then_runner() -> None:
    body = ORCHESTRATOR.read_text()
    # Must invoke the smoke script, then the runner per experiment.
    assert "smoke_test_experiment.py" in body
    assert "run_${exp}.sh" in body


def test_orchestrator_records_done_state() -> None:
    body = ORCHESTRATOR.read_text()
    # The :done sentinel is required by the resume contract.
    assert "record_state" in body
    assert ':done' in body


@pytest.mark.parametrize(
    "exp",
    ["phase18-redo", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E10"],
)
def test_orchestrator_each_experiment_listed(exp) -> None:
    body = ORCHESTRATOR.read_text()
    assert f'"{exp}"' in body
