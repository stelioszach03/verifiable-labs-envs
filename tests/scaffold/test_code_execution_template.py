"""Tests for ``scripts/create_env.py --template code-execution``.

Mirrors ``test_symbolic_math_template.py`` but exercises the
code-execution template family added in Phase 24.B. The two suites
share the underlying scaffold script; this file pins down the
code-execution-specific assertions (no forward_op.py, no sympy in the
generated pyproject, sandbox.py present, sandbox dep wiring).
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
    spec = importlib.util.spec_from_file_location("_create_env_ce", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# ── helper-function unit tests ───────────────────────────────────────


def test_templates_table_includes_code_execution():
    m = _load_create_env_module()
    assert "code-execution" in m.TEMPLATES
    assert "inverse-problem" in m.TEMPLATES, (
        "inverse-problem must remain registered for backward compat"
    )
    assert "symbolic-math" in m.TEMPLATES, (
        "symbolic-math must remain registered (Phase 21)"
    )


def test_resolve_code_execution_returns_existing_dir():
    m = _load_create_env_module()
    path = m._resolve_template("code-execution")
    assert path.is_dir()
    # Sanity: the template's __ENV_PY__ dir is present and forward_op.py is NOT.
    assert (path / "__ENV_PY__" / "env.py").is_file()
    assert (path / "__ENV_PY__" / "data.py").is_file()
    assert (path / "__ENV_PY__" / "reward.py").is_file()
    assert (path / "__ENV_PY__" / "adapter.py").is_file()
    assert (path / "__ENV_PY__" / "sandbox.py").is_file(), (
        "code-execution template MUST ship a sandbox.py thin re-export"
    )
    assert not (path / "__ENV_PY__" / "forward_op.py").exists(), (
        "code-execution template must NOT carry forward_op.py "
        "(no forward operator for code execution)"
    )


# ── end-to-end CLI tests ─────────────────────────────────────────────


def test_code_execution_writes_expected_tree(tmp_path):
    target = tmp_path / "scratch_code_env"
    proc = _run_script(
        "demo-code", "--template", "code-execution",
        "--domain", "general programming", "--target", str(target),
    )
    assert proc.returncode == 0, proc.stderr
    expected = {
        "pyproject.toml",
        "README.md",
        "conftest.py",
        "demo_code/__init__.py",
        "demo_code/env.py",
        "demo_code/reward.py",
        "demo_code/data.py",
        "demo_code/adapter.py",
        "demo_code/sandbox.py",
        "tests/test_env.py",
        "tests/test_reward.py",
        "tests/test_adapter.py",
        "tests/test_sandbox.py",
    }
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"


def test_code_execution_does_not_carry_forward_op(tmp_path):
    target = tmp_path / "no_fwd_op"
    _run_script(
        "demo-code", "--template", "code-execution",
        "--domain", "general programming", "--target", str(target),
    )
    assert not (target / "demo_code" / "forward_op.py").exists()


def test_code_execution_substitutes_all_placeholders(tmp_path):
    target = tmp_path / "subs"
    _run_script(
        "foo-code", "--template", "code-execution",
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


def test_code_execution_pyproject_does_not_pull_sympy(tmp_path):
    """Sandbox-based scoring doesn't need SymPy. The pyproject must
    NOT silently pull sympy through the symbolic-math template."""
    target = tmp_path / "deps"
    _run_script(
        "foo-code", "--template", "code-execution",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "sympy" not in text


def test_code_execution_pyproject_pulls_verifiable_labs_envs(tmp_path):
    """The sandbox primitive lives in verifiable-labs-envs; the
    scaffolded pyproject must depend on it."""
    target = tmp_path / "sandbox_dep"
    _run_script(
        "foo-code", "--template", "code-execution",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "verifiable-labs-envs" in text


def test_code_execution_imports_cleanly(tmp_path):
    """Render + import the rendered package without invoking the env."""
    target = tmp_path / "imports"
    _run_script(
        "alpha-code", "--template", "code-execution",
        "--domain", "test", "--target", str(target),
    )
    sys.path.insert(0, str(target))
    try:
        if "alpha_code" in sys.modules:
            del sys.modules["alpha_code"]
        mod = importlib.import_module("alpha_code")
        assert mod.ENV_ID == "alpha-code"
        assert mod.DOMAIN == "test"
        assert mod.EFFECTIVE_INSTANCES > 1e15
    finally:
        sys.path.remove(str(target))
        for k in list(sys.modules):
            if k.startswith("alpha_code"):
                del sys.modules[k]


def test_code_execution_pyproject_skeleton_matches_other_templates():
    """Cross-template invariant — every family must share the same
    build-system + Python version + license declaration."""
    code_exec = (
        REPO_ROOT / "templates" / "code-execution" / "template" / "pyproject.toml"
    ).read_text()
    for marker in (
        'requires = ["hatchling>=1.24"]',
        'build-backend = "hatchling.build"',
        'requires-python = ">=3.11"',
        'license = { text = "Apache-2.0" }',
    ):
        assert marker in code_exec, f"code-execution pyproject missing: {marker}"


def test_code_execution_unknown_template_still_fails_fast(tmp_path):
    target = tmp_path / "bad_template"
    proc = _run_script(
        "demo", "--template", "not-real",
        "--domain", "x", "--target", str(target),
    )
    assert proc.returncode != 0
    assert not target.exists() or not any(target.iterdir()), (
        "scaffolder must not write any files when --template is unknown"
    )


def test_unknown_template_error_message_lists_code_execution():
    """The error message for an unknown template lists the available
    families. The list must include ``code-execution`` (Phase 24)."""
    m = _load_create_env_module()
    with pytest.raises(SystemExit) as exc:
        m._resolve_template("not-real")
    assert "code-execution" in str(exc.value)
