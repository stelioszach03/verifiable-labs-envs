"""Tests for ``scripts/create_env.py --template sql-execution``."""
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
    spec = importlib.util.spec_from_file_location("_create_env_sql", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_templates_table_includes_sql_execution() -> None:
    m = _load_create_env_module()
    assert "sql-execution" in m.TEMPLATES
    for older in ("inverse-problem", "symbolic-math", "code-execution", "tool-calling"):
        assert older in m.TEMPLATES


def test_resolve_sql_execution_returns_existing_dir() -> None:
    m = _load_create_env_module()
    path = m._resolve_template("sql-execution")
    assert path.is_dir()
    for sub in (
        "__ENV_PY__/env.py", "__ENV_PY__/data.py",
        "__ENV_PY__/reward.py", "__ENV_PY__/adapter.py",
        "__ENV_PY__/sandbox.py",
    ):
        assert (path / sub).is_file(), f"missing {sub}"
    assert not (path / "__ENV_PY__" / "forward_op.py").exists()


def test_sql_execution_writes_expected_tree(tmp_path) -> None:
    target = tmp_path / "scratch_sql_env"
    proc = _run_script(
        "demo-sql", "--template", "sql-execution",
        "--domain", "text-to-sql", "--target", str(target),
    )
    assert proc.returncode == 0, proc.stderr
    expected = {
        "pyproject.toml",
        "README.md",
        "conftest.py",
        "demo_sql/__init__.py",
        "demo_sql/env.py",
        "demo_sql/reward.py",
        "demo_sql/data.py",
        "demo_sql/adapter.py",
        "demo_sql/sandbox.py",
        "tests/test_env.py",
        "tests/test_reward.py",
        "tests/test_adapter.py",
        "tests/test_sandbox.py",
    }
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"


def test_sql_execution_does_not_carry_forward_op(tmp_path) -> None:
    target = tmp_path / "no_fwd_op"
    _run_script(
        "demo-sql", "--template", "sql-execution",
        "--domain", "demo", "--target", str(target),
    )
    assert not (target / "demo_sql" / "forward_op.py").exists()


def test_sql_execution_substitutes_all_placeholders(tmp_path) -> None:
    target = tmp_path / "subs"
    _run_script(
        "foo-sql", "--template", "sql-execution",
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


def test_sql_execution_pyproject_does_not_pull_sympy(tmp_path) -> None:
    target = tmp_path / "deps"
    _run_script(
        "foo-sql", "--template", "sql-execution",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "sympy" not in text


def test_sql_execution_pyproject_pulls_verifiable_labs_envs(tmp_path) -> None:
    """The shared sql_primitives lives in verifiable-labs-envs."""
    target = tmp_path / "primitives_dep"
    _run_script(
        "foo-sql", "--template", "sql-execution",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "verifiable-labs-envs" in text


def test_sql_execution_imports_cleanly(tmp_path) -> None:
    target = tmp_path / "imports"
    _run_script(
        "alpha-sql", "--template", "sql-execution",
        "--domain", "test", "--target", str(target),
    )
    sys.path.insert(0, str(target))
    try:
        if "alpha_sql" in sys.modules:
            del sys.modules["alpha_sql"]
        mod = importlib.import_module("alpha_sql")
        assert mod.ENV_ID == "alpha-sql"
        assert mod.DOMAIN == "test"
        assert mod.EFFECTIVE_INSTANCES > 1e15
    finally:
        sys.path.remove(str(target))
        for k in list(sys.modules):
            if k.startswith("alpha_sql"):
                del sys.modules[k]


def test_sql_execution_pyproject_skeleton_matches_other_families() -> None:
    text = (
        REPO_ROOT / "templates" / "sql-execution" / "template" / "pyproject.toml"
    ).read_text()
    for marker in (
        'requires = ["hatchling>=1.24"]',
        'build-backend = "hatchling.build"',
        'requires-python = ">=3.11"',
        'license = { text = "Apache-2.0" }',
    ):
        assert marker in text, f"sql-execution pyproject missing: {marker}"


def test_unknown_template_error_message_lists_sql_execution() -> None:
    m = _load_create_env_module()
    with pytest.raises(SystemExit) as exc:
        m._resolve_template("not-real")
    assert "sql-execution" in str(exc.value)
