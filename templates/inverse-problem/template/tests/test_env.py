"""Smoke tests for the __ENV_ID__ env scaffold.

These tests verify the env's contract — instance shape, scoring
range, baseline runnability — *after* the user has filled in the
``NotImplementedError`` stubs. While the stubs are still in place the
tests will skip with a clear marker so the scaffold is checked in
without false-green coverage.
"""
from __future__ import annotations

import numpy as np
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
    """Procedural-regeneration check: > 1e15 unique measurement strings.
    The validator script enforces this same bound."""
    assert EFFECTIVE_INSTANCES > 1e15, (
        f"EFFECTIVE_INSTANCES = {EFFECTIVE_INSTANCES:.2e} is below the "
        "1e15 threshold for contamination-resistance certification. "
        "Adjust the |ground_truth_pool| factor in __ENV_PY__/__init__.py."
    )


def test_load_environment_returns_class_instance():
    env = load_environment(calibration_quantile=2.0)
    assert isinstance(env, __ENV_CLASS__)
    assert env.name == ENV_ID


def test_default_hyperparams_carry_alpha():
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0


def test_generate_instance_runs():
    """Skipped while ground-truth is unimplemented."""
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_ground_truth or forward_op still NotImplemented")
    assert inst.seed == 0
    assert inst.x_true.size > 0
    assert inst.y.size > 0


def test_generate_instance_seed_determinism():
    try:
        a = generate_instance(seed=42)
        b = generate_instance(seed=42)
    except NotImplementedError:
        pytest.skip("generate_ground_truth or forward_op still NotImplemented")
    assert np.array_equal(a.x_true, b.x_true)
    assert np.allclose(a.y, b.y)


def test_score_returns_well_shaped_dict():
    try:
        inst = generate_instance(seed=0)
        pred = baseline_predict(inst)
    except NotImplementedError:
        pytest.skip("generate_ground_truth, forward_op, or baseline_predict NotImplemented")
    env = __ENV_CLASS__(conformal_quantile=2.0)
    out = env.score(pred, inst)
    assert "reward" in out
    assert 0.0 <= out["reward"] <= 1.0
    assert "components" in out
    for k, v in out["components"].items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_run_baseline_produces_finite_reward():
    env = __ENV_CLASS__(conformal_quantile=2.0)
    try:
        out = env.run_baseline(seed=0)
    except NotImplementedError:
        pytest.skip("baseline pipeline contains a NotImplementedError stub")
    assert np.isfinite(out["reward"])
