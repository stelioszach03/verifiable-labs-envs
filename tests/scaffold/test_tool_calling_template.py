"""Tests for ``scripts/create_env.py --template tool-calling``.

Mirrors ``test_code_execution_template.py`` but exercises the
tool-calling template family added in Phase 25.B. Pins down the
tool-calling-specific assertions: tools.py thin re-export, no
forward_op.py, no sympy in the generated pyproject, dependency on
the parent ``verifiable-labs-envs`` package for the shared
primitives library.
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
    spec = importlib.util.spec_from_file_location("_create_env_tc", SCRIPT)
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


def test_templates_table_includes_tool_calling():
    m = _load_create_env_module()
    assert "tool-calling" in m.TEMPLATES
    # Older families remain registered.
    for older in ("inverse-problem", "symbolic-math", "code-execution"):
        assert older in m.TEMPLATES


def test_resolve_tool_calling_returns_existing_dir():
    m = _load_create_env_module()
    path = m._resolve_template("tool-calling")
    assert path.is_dir()
    for sub in ("__ENV_PY__/env.py", "__ENV_PY__/data.py",
                "__ENV_PY__/reward.py", "__ENV_PY__/adapter.py",
                "__ENV_PY__/tools.py"):
        assert (path / sub).is_file(), f"missing {sub}"
    assert not (path / "__ENV_PY__" / "forward_op.py").exists(), (
        "tool-calling template must NOT carry forward_op.py"
    )


# ── end-to-end CLI tests ─────────────────────────────────────────────


def test_tool_calling_writes_expected_tree(tmp_path):
    target = tmp_path / "scratch_tool_env"
    proc = _run_script(
        "demo-tool", "--template", "tool-calling",
        "--domain", "tool orchestration", "--target", str(target),
    )
    assert proc.returncode == 0, proc.stderr
    expected = {
        "pyproject.toml",
        "README.md",
        "conftest.py",
        "demo_tool/__init__.py",
        "demo_tool/env.py",
        "demo_tool/reward.py",
        "demo_tool/data.py",
        "demo_tool/adapter.py",
        "demo_tool/tools.py",
        "tests/test_env.py",
        "tests/test_reward.py",
        "tests/test_adapter.py",
        "tests/test_tools.py",
    }
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"


def test_tool_calling_does_not_carry_forward_op(tmp_path):
    target = tmp_path / "no_fwd_op"
    _run_script(
        "demo-tool", "--template", "tool-calling",
        "--domain", "tool orchestration", "--target", str(target),
    )
    assert not (target / "demo_tool" / "forward_op.py").exists()


def test_tool_calling_substitutes_all_placeholders(tmp_path):
    target = tmp_path / "subs"
    _run_script(
        "foo-tool", "--template", "tool-calling",
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


def test_tool_calling_pyproject_does_not_pull_sympy(tmp_path):
    target = tmp_path / "deps"
    _run_script(
        "foo-tool", "--template", "tool-calling",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "sympy" not in text


def test_tool_calling_pyproject_pulls_verifiable_labs_envs(tmp_path):
    """The shared tool primitives live in verifiable-labs-envs."""
    target = tmp_path / "primitives_dep"
    _run_script(
        "foo-tool", "--template", "tool-calling",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "verifiable-labs-envs" in text


def test_tool_calling_imports_cleanly(tmp_path):
    target = tmp_path / "imports"
    _run_script(
        "alpha-tool", "--template", "tool-calling",
        "--domain", "test", "--target", str(target),
    )
    sys.path.insert(0, str(target))
    try:
        if "alpha_tool" in sys.modules:
            del sys.modules["alpha_tool"]
        mod = importlib.import_module("alpha_tool")
        assert mod.ENV_ID == "alpha-tool"
        assert mod.DOMAIN == "test"
        assert mod.EFFECTIVE_INSTANCES > 1e15
    finally:
        sys.path.remove(str(target))
        for k in list(sys.modules):
            if k.startswith("alpha_tool"):
                del sys.modules[k]


def test_tool_calling_pyproject_skeleton_matches_other_families():
    text = (
        REPO_ROOT / "templates" / "tool-calling" / "template" / "pyproject.toml"
    ).read_text()
    for marker in (
        'requires = ["hatchling>=1.24"]',
        'build-backend = "hatchling.build"',
        'requires-python = ">=3.11"',
        'license = { text = "Apache-2.0" }',
    ):
        assert marker in text, f"tool-calling pyproject missing: {marker}"


def test_unknown_template_error_message_lists_tool_calling():
    m = _load_create_env_module()
    with pytest.raises(SystemExit) as exc:
        m._resolve_template("not-real")
    assert "tool-calling" in str(exc.value)
