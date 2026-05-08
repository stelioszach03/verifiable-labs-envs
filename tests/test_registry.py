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
        "super-resolution-div2k-x4",
    ]


def test_registry_carries_seventeen_envs() -> None:
    """Phase 24.F — registry grows from 13 → 17 with the code-* family."""
    assert len(vle.list_environments()) == 17


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


def test_unknown_environment_raises() -> None:
    with pytest.raises(KeyError, match="Unknown environment"):
        vle.load_environment("does-not-exist")
