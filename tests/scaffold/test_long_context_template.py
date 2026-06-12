"""Tests for ``scripts/create_env.py --template long-context``."""
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
    spec = importlib.util.spec_from_file_location("_create_env_lc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_templates_table_includes_long_context() -> None:
    m = _load_create_env_module()
    assert "long-context" in m.TEMPLATES
    for older in (
        "inverse-problem", "symbolic-math", "code-execution",
        "tool-calling", "sql-execution",
    ):
        assert older in m.TEMPLATES


def test_resolve_long_context_returns_existing_dir() -> None:
    m = _load_create_env_module()
    path = m._resolve_template("long-context")
    assert path.is_dir()
    for sub in (
        "__ENV_PY__/env.py", "__ENV_PY__/data.py",
        "__ENV_PY__/reward.py", "__ENV_PY__/adapter.py",
        "__ENV_PY__/corpus.py", "__ENV_PY__/needle.py",
    ):
        assert (path / sub).is_file(), f"missing {sub}"
    # Long-context envs do NOT carry a forward operator (inverse-problem
    # only) or a SQLite sandbox (sql-execution only).
    assert not (path / "__ENV_PY__" / "forward_op.py").exists()
    assert not (path / "__ENV_PY__" / "sandbox.py").exists()


def test_long_context_writes_expected_tree(tmp_path) -> None:
    target = tmp_path / "scratch_lc_env"
    proc = _run_script(
        "demo-lc", "--template", "long-context",
        "--domain", "long-context retrieval", "--target", str(target),
    )
    assert proc.returncode == 0, proc.stderr
    expected = {
        "pyproject.toml",
        "README.md",
        "conftest.py",
        "demo_lc/__init__.py",
        "demo_lc/env.py",
        "demo_lc/reward.py",
        "demo_lc/data.py",
        "demo_lc/adapter.py",
        "demo_lc/corpus.py",
        "demo_lc/needle.py",
        "tests/test_env.py",
        "tests/test_reward.py",
        "tests/test_adapter.py",
        "tests/test_corpus.py",
        "tests/test_needle.py",
    }
    actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"


def test_long_context_does_not_carry_forward_op_or_sandbox(tmp_path) -> None:
    target = tmp_path / "no_inv_no_sandbox"
    _run_script(
        "demo-lc", "--template", "long-context",
        "--domain", "demo", "--target", str(target),
    )
    assert not (target / "demo_lc" / "forward_op.py").exists()
    assert not (target / "demo_lc" / "sandbox.py").exists()


def test_long_context_substitutes_all_placeholders(tmp_path) -> None:
    target = tmp_path / "subs"
    _run_script(
        "foo-lc", "--template", "long-context",
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


def test_long_context_pyproject_does_not_pull_sympy_or_sqlite(tmp_path) -> None:
    target = tmp_path / "deps"
    _run_script(
        "foo-lc", "--template", "long-context",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "sympy" not in text
    # sqlite3 ships in stdlib so it isn't a dep, but explicit DB libs
    # should not appear in the long-context family.
    assert "sqlalchemy" not in text.lower()


def test_long_context_pyproject_pulls_tiktoken(tmp_path) -> None:
    """Long-context envs need tiktoken for the cl100k_base tokeniser."""
    target = tmp_path / "tiktoken_dep"
    _run_script(
        "foo-lc", "--template", "long-context",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "tiktoken" in text


def test_long_context_pyproject_pulls_verifiable_labs_envs(tmp_path) -> None:
    """The shared long_context_primitives lives in verifiable-labs-envs."""
    target = tmp_path / "primitives_dep"
    _run_script(
        "foo-lc", "--template", "long-context",
        "--domain", "demo", "--target", str(target),
    )
    text = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "verifiable-labs-envs" in text


def test_long_context_imports_cleanly(tmp_path) -> None:
    target = tmp_path / "imports"
    _run_script(
        "alpha-lc", "--template", "long-context",
        "--domain", "test", "--target", str(target),
    )
    sys.path.insert(0, str(target))
    try:
        if "alpha_lc" in sys.modules:
            del sys.modules["alpha_lc"]
        mod = importlib.import_module("alpha_lc")
        assert mod.ENV_ID == "alpha-lc"
        assert mod.DOMAIN == "test"
        assert mod.EFFECTIVE_INSTANCES > 1e15
    finally:
        sys.path.remove(str(target))
        for k in list(sys.modules):
            if k.startswith("alpha_lc"):
                del sys.modules[k]


def test_long_context_pyproject_skeleton_matches_other_families() -> None:
    text = (
        REPO_ROOT / "templates" / "long-context" / "template" / "pyproject.toml"
    ).read_text()
    for marker in (
        'requires = ["hatchling>=1.24"]',
        'build-backend = "hatchling.build"',
        'requires-python = ">=3.11"',
        'license = { text = "Apache-2.0" }',
    ):
        assert marker in text, f"long-context pyproject missing: {marker}"


def test_unknown_template_error_message_lists_long_context() -> None:
    m = _load_create_env_module()
    with pytest.raises(SystemExit) as exc:
        m._resolve_template("not-real")
    assert "long-context" in str(exc.value)
