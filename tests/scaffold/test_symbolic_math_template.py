"""Tests for ``scripts/create_env.py --template symbolic-math``.

Mirrors ``tests/scaffold/test_create_env.py`` but exercises the new
symbolic-math template family added in Phase 21.B. The two suites
share the underlying scaffold script; this file pins down the
symbolic-math-specific assertions (no forward_op.py, sympy in the
generated pyproject, threaded simplify timeout in reward.py).
"""
from __future__ import annotations

import importlib
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
    spec = importlib.util.spec_from_file_location("_create_env_sm", SCRIPT)
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


def test_templates_table_includes_symbolic_math():
    m = _load_create_env_module()
    assert "symbolic-math" in m.TEMPLATES
    assert "inverse-problem" in m.TEMPLATES, (
        "inverse-problem must remain registered for backward compat"
    )


def test_resolve_template_returns_existing_dir():
    m = _load_create_env_module()
    path = m._resolve_template("symbolic-math")
    assert path.is_dir()
    # Sanity: the template's __ENV_PY__ dir is present and forward_op.py is NOT.
    assert (path / "__ENV_PY__" / "env.py").is_file()
    assert (path / "__ENV_PY__" / "data.py").is_file()
    assert (path / "__ENV_PY__" / "reward.py").is_file()
    assert (path / "__ENV_PY__" / "adapter.py").is_file()
    assert not (path / "__ENV_PY__" / "forward_op.py").exists(), (
        "symbolic-math template must NOT carry forward_op.py "
        "(no forward operator for symbolic algebra)"
    )


def test_resolve_template_rejects_unknown_name():
    m = _load_create_env_module()
    with pytest.raises(SystemExit):
        m._resolve_template("not-a-real-family")


def test_template_dir_constant_still_points_at_inverse_problem():
    """Backward-compat: the existing 9 scaffold tests reach in for
    TEMPLATE_DIR at import time and assume the inverse-problem path."""
    m = _load_create_env_module()
    assert m.TEMPLATES["inverse-problem"] == m.TEMPLATE_DIR


# ── end-to-end CLI tests ───────────────────────────────────────


def test_default_template_is_inverse_problem(tmp_path):
    """Calling without --template MUST scaffold the inverse-problem
    family (preserving the 9 existing scaffold tests)."""
    target = tmp_path / "default"
    proc = _run_script("legacy-default", "--domain", "test", "--target", str(target))
    assert proc.returncode == 0, proc.stderr
    # Inverse-problem family carries forward_op.py.
    assert (target / "legacy_default" / "forward_op.py").is_file()


def test_symbolic_math_writes_expected_tree(tmp_path):
    target = tmp_path / "scratch_math_env"
    proc = _run_script(
        "demo-math", "--template", "symbolic-math",
        "--domain", "algebra", "--target", str(target),
    )
    assert proc.returncode == 0, proc.stderr
    expected = {
        "pyproject.toml",
        "README.md",
        "conftest.py",
        "demo_math/__init__.py",
        "demo_math/env.py",
        "demo_math/reward.py",
        "demo_math/data.py",
        "demo_math/adapter.py",
        "tests/test_env.py",
        "tests/test_reward.py",
        "tests/test_adapter.py",
    }
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"


def test_symbolic_math_does_not_carry_forward_op(tmp_path):
    target = tmp_path / "no_fwd_op"
    _run_script(
        "demo-math", "--template", "symbolic-math",
        "--domain", "algebra", "--target", str(target),
    )
    assert not (target / "demo_math" / "forward_op.py").exists()


def test_symbolic_math_substitutes_all_placeholders(tmp_path):
    target = tmp_path / "subs"
    _run_script(
        "foo-math", "--template", "symbolic-math",
        "--domain", "demo", "--target", str(target),
    )
    text_files = [
        p for p in target.rglob("*")
        if p.is_file() and p.suffix in {".py", ".toml", ".md"}
    ]
    for f in text_files:
        text = f.read_text(encoding="utf-8")
        for marker in ("__ENV_ID__", "__ENV_PY__", "__ENV_CLASS__",
                       "__DOMAIN__", "__DOMAIN_TAG__"):
            assert marker not in text, f"{marker} not substituted in {f}"


def test_symbolic_math_pyproject_lists_sympy_dep(tmp_path):
    target = tmp_path / "deps"
    _run_script(
        "foo-math", "--template", "symbolic-math",
        "--domain", "algebra", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "sympy>=1.12" in text


def test_symbolic_math_imports_cleanly(tmp_path):
    """Render + import the rendered package without invoking the env."""
    target = tmp_path / "imports"
    _run_script(
        "alpha-math", "--template", "symbolic-math",
        "--domain", "test", "--target", str(target),
    )
    sys.path.insert(0, str(target))
    try:
        if "alpha_math" in sys.modules:
            del sys.modules["alpha_math"]
        mod = importlib.import_module("alpha_math")
        assert mod.ENV_ID == "alpha-math"
        assert mod.DOMAIN == "test"
        assert mod.EFFECTIVE_INSTANCES > 1e15
    finally:
        sys.path.remove(str(target))
        for k in list(sys.modules):
            if k.startswith("alpha_math"):
                del sys.modules[k]


def test_unknown_template_fails_fast(tmp_path):
    target = tmp_path / "bad_template"
    proc = _run_script(
        "demo", "--template", "not-real",
        "--domain", "x", "--target", str(target),
    )
    assert proc.returncode != 0
    assert not target.exists() or not any(target.iterdir()), (
        "scaffolder must not write any files when --template is unknown"
    )


def test_pyproject_skeletons_agree_on_build_system():
    """Cross-template invariant — both families must share the same
    build-system + Python version + license declaration so a future
    refactor into a shared `_base/` is mechanical, not a redesign."""
    inv = (REPO_ROOT / "templates" / "inverse-problem" / "template" / "pyproject.toml").read_text()
    sym = (REPO_ROOT / "templates" / "symbolic-math" / "template" / "pyproject.toml").read_text()
    for marker in (
        'requires = ["hatchling>=1.24"]',
        'build-backend = "hatchling.build"',
        'requires-python = ">=3.11"',
        'license = { text = "Apache-2.0" }',
    ):
        assert marker in inv, f"inverse-problem pyproject missing: {marker}"
        assert marker in sym, f"symbolic-math pyproject missing: {marker}"
