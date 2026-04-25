"""Tests for ``scripts/create_env.py`` — the inverse-problem scaffold."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "create_env.py"


def _load_create_env_module():
    """Import scripts/create_env.py as a module so we can unit-test
    its helpers without invoking the CLI."""
    spec = importlib.util.spec_from_file_location("_create_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# ── helper-function unit tests ─────────────────────────────────


def test_kebab_to_snake():
    m = _load_create_env_module()
    assert m._kebab_to_snake("seismic-fwi") == "seismic_fwi"
    assert m._kebab_to_snake("phase-retrieval-multiturn") == "phase_retrieval_multiturn"


def test_kebab_to_camel():
    m = _load_create_env_module()
    assert m._kebab_to_camel("seismic-fwi") == "SeismicFwi"
    assert m._kebab_to_camel("phase-retrieval-multiturn") == "PhaseRetrievalMultiturn"


def test_slugify_strips_punctuation_and_spaces():
    m = _load_create_env_module()
    assert m._slugify("Medical Imaging (CT)") == "medical-imaging-ct"
    assert m._slugify("  Geophysics  ") == "geophysics"
    assert m._slugify("") == "general"
    assert m._slugify("!!") == "general"


def test_substitution_dict_contains_all_markers():
    m = _load_create_env_module()
    subs = m._build_substitutions("phase-retrieval", "Coherent diffraction imaging")
    assert subs["__ENV_ID__"] == "phase-retrieval"
    assert subs["__ENV_PY__"] == "phase_retrieval"
    assert subs["__ENV_CLASS__"] == "PhaseRetrievalEnv"
    assert subs["__DOMAIN__"] == "Coherent diffraction imaging"
    assert subs["__DOMAIN_TAG__"] == "coherent-diffraction-imaging"


def test_validate_env_id_rejects_underscores():
    m = _load_create_env_module()
    with pytest.raises(SystemExit):
        m._validate_env_id("seismic_fwi")


def test_validate_env_id_rejects_uppercase():
    m = _load_create_env_module()
    with pytest.raises(SystemExit):
        m._validate_env_id("Seismic-FWI")


def test_validate_env_id_rejects_consecutive_hyphens():
    m = _load_create_env_module()
    with pytest.raises(SystemExit):
        m._validate_env_id("seismic--fwi")


def test_validate_env_id_accepts_kebab_case():
    m = _load_create_env_module()
    # No exception
    m._validate_env_id("seismic-fwi")
    m._validate_env_id("phase-retrieval-multiturn")
    m._validate_env_id("a")


# ── end-to-end CLI tests ───────────────────────────────────────


def test_scaffold_writes_expected_tree(tmp_path):
    target = tmp_path / "scratch_env"
    proc = _run_script("demo-thing", "--domain", "test", "--target", str(target))
    assert proc.returncode == 0, proc.stderr
    expected = {
        "pyproject.toml",
        "README.md",
        "conftest.py",
        "demo_thing/__init__.py",
        "demo_thing/env.py",
        "demo_thing/forward_op.py",
        "demo_thing/reward.py",
        "demo_thing/data.py",
        "demo_thing/adapter.py",
        "tests/test_env.py",
        "tests/test_reward.py",
        "tests/test_adapter.py",
    }
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"


def test_scaffold_substitutes_all_placeholders(tmp_path):
    target = tmp_path / "x"
    _run_script("foo-bar-baz", "--domain", "demo", "--target", str(target))
    text_files = [p for p in target.rglob("*") if p.is_file() and p.suffix in {".py", ".toml", ".md"}]
    for f in text_files:
        text = f.read_text(encoding="utf-8")
        for marker in ("__ENV_ID__", "__ENV_PY__", "__ENV_CLASS__",
                       "__DOMAIN__", "__DOMAIN_TAG__"):
            assert marker not in text, f"{marker} not substituted in {f}"


def test_scaffold_renames_template_directory(tmp_path):
    target = tmp_path / "y"
    _run_script("alpha-beta", "--domain", "test", "--target", str(target))
    assert (target / "alpha_beta").is_dir()
    assert not (target / "__ENV_PY__").exists()


def test_scaffold_imports_cleanly(tmp_path):
    """Render + import the rendered package without invoking the env."""
    target = tmp_path / "imports"
    _run_script("alpha-beta", "--domain", "test", "--target", str(target))
    sys.path.insert(0, str(target))
    try:
        if "alpha_beta" in sys.modules:
            del sys.modules["alpha_beta"]
        mod = importlib.import_module("alpha_beta")
        assert mod.ENV_ID == "alpha-beta"
        assert mod.DOMAIN == "test"
        assert mod.EFFECTIVE_INSTANCES > 1e15
    finally:
        sys.path.remove(str(target))
        for k in list(sys.modules):
            if k.startswith("alpha_beta"):
                del sys.modules[k]


def test_scaffold_refuses_existing_target_without_force(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    proc = _run_script("foo", "--domain", "x", "--target", str(target))
    assert proc.returncode != 0
    assert "already exists" in proc.stderr or "already exists" in proc.stdout


def test_scaffold_force_overwrites(tmp_path):
    target = tmp_path / "force"
    target.mkdir()
    (target / "stale.txt").write_text("delete me")
    proc = _run_script("foo", "--domain", "x", "--target", str(target), "--force")
    assert proc.returncode == 0
    assert not (target / "stale.txt").exists()
    assert (target / "foo" / "__init__.py").exists()


def test_scaffold_rejects_invalid_env_id(tmp_path):
    target = tmp_path / "bad"
    proc = _run_script("Bad_Name", "--domain", "x", "--target", str(target))
    assert proc.returncode != 0
    assert "kebab-case" in proc.stderr or "kebab-case" in proc.stdout
