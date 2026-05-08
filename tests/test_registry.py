"""Smoke tests for the environment registry — Day 1 sanity only."""

import pytest

import verifiable_labs_envs as vle


def test_version() -> None:
    assert vle.__version__ == "0.0.1"


def test_registry_lists_expected_envs() -> None:
    envs = vle.list_environments()
    assert envs == [
        "code-humaneval",
        "code-humaneval-multiturn",
        "code-humaneval-tools",
        "code-mini-repo",
        "lodopab-ct-simplified",
        "lodopab-ct-simplified-multiturn",
        "long-context-needle",
        "long-context-reasoning",
        "long-context-synthesis",
        "math-algebra",
        "math-algebra-multiturn",
        "math-algebra-tools",
        "mri-knee-reconstruction",
        "mri-knee-reconstruction-multiturn",
        "phase-retrieval",
        "phase-retrieval-multiturn",
        "sparse-fourier-recovery",
        "sparse-fourier-recovery-multiturn",
        "sparse-fourier-recovery-tools",
        "sql-multiturn",
        "sql-single-turn",
        "super-resolution-div2k-x4",
        "tool-calling-debug",
        "tool-calling-multiturn",
        "tool-calling-single",
    ]


def test_registry_carries_twentyfive_envs() -> None:
    """Phase 27.E — registry grows from 22 → 25 with the long-context family.
    25 envs across 7 template families closes the env-catalogue track per §19.
    """
    assert len(vle.list_environments()) == 25


def test_load_code_humaneval_via_registry() -> None:
    env = vle.load_environment("code-humaneval", calibration_quantile=0.5)
    assert env.name == "code-humaneval"


def test_load_code_humaneval_multiturn_via_registry() -> None:
    env = vle.load_environment("code-humaneval-multiturn", calibration_quantile=0.5)
    assert env.name == "code-humaneval-multiturn"


def test_load_code_humaneval_tools_via_registry() -> None:
    env = vle.load_environment("code-humaneval-tools", calibration_quantile=0.5)
    assert env.name == "code-humaneval-tools"


def test_load_code_mini_repo_via_registry() -> None:
    env = vle.load_environment("code-mini-repo", calibration_quantile=0.5)
    assert env.name == "code-mini-repo"


def test_load_tool_calling_single_via_registry() -> None:
    env = vle.load_environment("tool-calling-single", calibration_quantile=0.5)
    assert env.name == "tool-calling-single"


def test_load_tool_calling_multiturn_via_registry() -> None:
    env = vle.load_environment("tool-calling-multiturn", calibration_quantile=0.5)
    assert env.name == "tool-calling-multiturn"


def test_load_tool_calling_debug_via_registry() -> None:
    env = vle.load_environment("tool-calling-debug", calibration_quantile=0.5)
    assert env.name == "tool-calling-debug"


def test_load_sql_single_turn_via_registry() -> None:
    env = vle.load_environment("sql-single-turn", calibration_quantile=0.5)
    assert env.name == "sql-single-turn"


def test_load_sql_multiturn_via_registry() -> None:
    env = vle.load_environment("sql-multiturn", calibration_quantile=0.5)
    assert env.name == "sql-multiturn"


def test_load_long_context_needle_via_registry() -> None:
    env = vle.load_environment("long-context-needle", calibration_quantile=0.5)
    assert env.name == "long-context-needle"


def test_load_long_context_synthesis_via_registry() -> None:
    env = vle.load_environment("long-context-synthesis", calibration_quantile=0.5)
    assert env.name == "long-context-synthesis"


def test_load_long_context_reasoning_via_registry() -> None:
    env = vle.load_environment("long-context-reasoning", calibration_quantile=0.5)
    assert env.name == "long-context-reasoning"


def test_unknown_environment_raises() -> None:
    with pytest.raises(KeyError, match="Unknown environment"):
        vle.load_environment("does-not-exist")
