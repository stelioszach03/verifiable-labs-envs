"""Smoke tests for the environment registry — Day 1 sanity only."""

import pytest

import verifiable_labs_envs as vle


def test_version() -> None:
    assert vle.__version__ == "0.0.1"


def test_registry_lists_expected_envs() -> None:
    envs = vle.list_environments()
    assert envs == [
        "lodopab-ct-simplified",
        "lodopab-ct-simplified-multiturn",
        "sparse-fourier-recovery",
        "sparse-fourier-recovery-multiturn",
        "sparse-fourier-recovery-tools",
        "super-resolution-div2k-x4",
    ]


def test_unknown_environment_raises() -> None:
    with pytest.raises(KeyError, match="Unknown environment"):
        vle.load_environment("does-not-exist")
