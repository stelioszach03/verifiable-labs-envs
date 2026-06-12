"""Smoke tests for the __ENV_ID__ env scaffold.

These tests verify the env's contract — ENV_ID shape, effective
instance count, scoring range, baseline runnability — *after* the user
has filled in the ``NotImplementedError`` stub in ``data.py``. While
the stub is still in place the tests skip with a clear marker so the
scaffold is checked in without false-green coverage.
"""
from __future__ import annotations

import pytest

from __ENV_PY__ import EFFECTIVE_INSTANCES, ENV_ID
from __ENV_PY__.env import (
    __ENV_CLASS__,
    DEFAULT_HYPERPARAMS,
    baseline_predict,
    generate_instance,
    load_environment,
)


def test_env_id_is_kebab_case():
    assert ENV_ID == "__ENV_ID__"
    assert "_" not in ENV_ID, "env id should be kebab-case (use - not _)"


def test_effective_instances_above_procedural_threshold():
    """Procedural-regeneration check: > 1e15 unique problem strings.
    The validator script enforces this same bound."""
    assert EFFECTIVE_INSTANCES > 1e15, (
        f"EFFECTIVE_INSTANCES = {EFFECTIVE_INSTANCES:.2e} is below the "
        "1e15 threshold for contamination-resistance certification. "
        "Adjust the |problem_pool| factor in __ENV_PY__/__init__.py."
    )


def test_load_environment_returns_class_instance():
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, __ENV_CLASS__)
    assert env.name == ENV_ID


def test_default_hyperparams_carry_alpha_and_timeout():
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert "simplify_timeout_s" in DEFAULT_HYPERPARAMS
    assert DEFAULT_HYPERPARAMS["simplify_timeout_s"] > 0


def test_generate_instance_runs():
    """Skipped while problem generator is unimplemented."""
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    assert inst.seed == 0
    assert inst.prompt
    assert inst.gold_expr


def test_generate_instance_seed_determinism():
    try:
        a = generate_instance(seed=42)
        b = generate_instance(seed=42)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    assert a.prompt == b.prompt
    assert a.gold_expr == b.gold_expr


def test_score_returns_well_shaped_dict():
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    pred = baseline_predict(inst)
    env = __ENV_CLASS__(conformal_quantile=0.5)
    out = env.score(pred, inst)
    assert "reward" in out
    assert 0.0 <= out["reward"] <= 1.0
    assert "components" in out
    for k, v in out["components"].items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
    assert "meta" in out
    assert "covered" in out["meta"]


def test_run_baseline_produces_finite_reward():
    env = __ENV_CLASS__(conformal_quantile=0.5)
    try:
        out = env.run_baseline(seed=0)
    except NotImplementedError:
        pytest.skip("baseline pipeline contains a NotImplementedError stub")
    assert isinstance(out["reward"], float)
    assert 0.0 <= out["reward"] <= 1.0


def test_as_inputs_excludes_gold_expr():
    """Solvers must not see the oracle field."""
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    inputs = inst.as_inputs()
    assert "gold_expr" not in inputs
    assert "prompt" in inputs
