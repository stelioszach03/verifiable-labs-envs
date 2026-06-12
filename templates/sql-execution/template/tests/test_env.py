"""Smoke tests for the __ENV_ID__ env scaffold."""
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
    assert "_" not in ENV_ID, "env id should be kebab-case"


def test_effective_instances_above_procedural_threshold():
    assert EFFECTIVE_INSTANCES > 1e15


def test_load_environment_returns_class_instance():
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, __ENV_CLASS__)
    assert env.name == ENV_ID


def test_default_hyperparams_carry_alpha_and_caps():
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert "max_rows" in DEFAULT_HYPERPARAMS
    assert "timeout_s" in DEFAULT_HYPERPARAMS


def test_generate_instance_runs():
    """Skipped while problem generator is unimplemented."""
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    assert inst.seed == 0
    assert inst.prompt
    assert inst.gold_query
    assert inst.schema.create_statements
    assert inst.schema.seed_statements


def test_generate_instance_seed_determinism():
    try:
        a = generate_instance(seed=42)
        b = generate_instance(seed=42)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    assert a.gold_query == b.gold_query
    assert a.gold_result_rows == b.gold_result_rows


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
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"
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


def test_as_inputs_excludes_oracle_fields():
    """Solvers must not see gold_query or gold_result_rows."""
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_problem still NotImplemented")
    inputs = inst.as_inputs()
    assert "gold_query" not in inputs
    assert "gold_result_rows" not in inputs
    assert "prompt" in inputs
    assert "schema" in inputs
