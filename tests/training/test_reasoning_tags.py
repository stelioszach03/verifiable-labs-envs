"""Tests for reasoning-tags parsing (Logic-RL / DeepSeek-R1 chat-template style).

Covers the pure tag extractor (``extract_tagged_answer``), the
adapter-backed parser (``parse_with_tags``), the ``use_tags`` flag on
both reward-function factories, and end-to-end backward compatibility
with the existing test fixtures from M2.

References:
    * Logic-RL: Xie et al. 2025, arXiv:2502.14768
    * DeepSeek-R1 distilled chat template (Jan 2025)
"""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs import load_environment
from verifiable_labs_envs.solvers.llm_solver import LLMSolverError, get_adapter
from verifiable_labs_envs.training import (
    extract_tagged_answer,
    make_reward_fn,
    make_reward_fn_posterior,
    parse_with_tags,
)


ENV_ID = "sparse-fourier-recovery"


# ── helpers ───────────────────────────────────────────────────────────


def _zero_json(k: int = 10, n: int = 256) -> str:
    return json.dumps({
        "support_idx": list(range(min(k, n))),
        "support_amp_x1000": [0] * k,
    })


def _wrap_with_tags(answer_json: str, *, think: str = "Considering the inputs.") -> str:
    return f"<think>{think}</think>\n<answer>{answer_json}</answer>"


@pytest.fixture(scope="module")
def adapter():
    return get_adapter(ENV_ID)


@pytest.fixture(scope="module")
def env_instance():
    env = load_environment(ENV_ID, calibration_quantile=2.0)
    return env, env.generate_instance(seed=0)


# ── extract_tagged_answer (pure) ─────────────────────────────────────


def test_both_tags_present_extracts_answer() -> None:
    text = "<think>step by step</think><answer>final result</answer>"
    assert extract_tagged_answer(text) == "final result"


def test_only_answer_no_think_raises() -> None:
    with pytest.raises(LLMSolverError, match="<think>"):
        extract_tagged_answer("<answer>only</answer>")


def test_only_think_no_answer_raises() -> None:
    with pytest.raises(LLMSolverError, match="<answer>"):
        extract_tagged_answer("<think>thinking...</think>")


def test_tags_in_wrong_order_raises() -> None:
    text = "<answer>preliminary</answer><think>now thinking</think>"
    with pytest.raises(LLMSolverError, match="must appear before"):
        extract_tagged_answer(text)


def test_whitespace_around_tags_tolerated() -> None:
    text = "  \n<think>\n  step 1  \n</think>\n\n<answer>\n  result \n</answer>\n"
    assert extract_tagged_answer(text) == "result"


def test_multiple_answer_tags_first_only() -> None:
    text = (
        "<think>analysis</think>"
        "<answer>first</answer>"
        "<answer>second</answer>"
    )
    assert extract_tagged_answer(text) == "first"


def test_empty_text_raises() -> None:
    with pytest.raises(LLMSolverError, match="empty"):
        extract_tagged_answer("")


def test_case_insensitive_tag_matching() -> None:
    text = "<THINK>analysis</THINK><ANSWER>upper</ANSWER>"
    assert extract_tagged_answer(text) == "upper"


# ── parse_with_tags (adapter integration) ────────────────────────────


def test_parse_with_tags_valid_json(adapter, env_instance) -> None:
    _env, inst = env_instance
    text = _wrap_with_tags(_zero_json(k=inst.k, n=inst.n))
    pred = parse_with_tags(text, inst, adapter)
    assert hasattr(pred, "x_hat")
    assert hasattr(pred, "support_hat")


def test_parse_with_tags_malformed_json_inside_answer(adapter, env_instance) -> None:
    _env, inst = env_instance
    text = "<think>analyzing</think><answer>{this is not json}</answer>"
    with pytest.raises(LLMSolverError):
        parse_with_tags(text, inst, adapter)


def test_parse_with_tags_correct_keys_wrong_length(adapter, env_instance) -> None:
    _env, inst = env_instance
    # k=3 instead of inst.k
    bad_payload = json.dumps({"support_idx": [0, 1, 2], "support_amp_x1000": [0, 0, 0]})
    text = _wrap_with_tags(bad_payload)
    with pytest.raises(LLMSolverError):
        parse_with_tags(text, inst, adapter)


# ── reward-fn flag wiring ────────────────────────────────────────────


def test_make_reward_fn_default_no_tags() -> None:
    """Backward compat: default use_tags=False; legacy parse path."""
    fn = make_reward_fn(ENV_ID)
    assert fn.use_tags is False


def test_make_reward_fn_posterior_default_use_tags() -> None:
    """Phase 14 default: posterior factory turns tags ON by default."""
    fn = make_reward_fn_posterior(ENV_ID)
    assert fn.use_tags is True


def test_use_tags_true_rejects_naked_json() -> None:
    """A completion that's just raw JSON (no tags) format-fails when
    ``use_tags=True``, even though the same completion passes when
    ``use_tags=False``."""
    fn_no_tags = make_reward_fn(ENV_ID, use_tags=False)
    fn_with_tags = make_reward_fn(ENV_ID, use_tags=True)

    naked = _zero_json()
    rewards_no_tags = fn_no_tags(prompts=[""], completions=[naked], instance_seed=[0])
    rewards_with_tags = fn_with_tags(prompts=[""], completions=[naked], instance_seed=[0])

    assert rewards_no_tags[0] > 0.0  # legacy: format passes, scores > 0
    assert rewards_with_tags[0] == 0.0  # strict: missing tags → format_fail
    rec = fn_with_tags.stats.per_call[-1]
    assert rec["components"]["parse_valid"] == 1.0
    assert rec["components"]["format_valid"] == 0.0
    assert rec["failure_type"] == "format_error"


def test_use_tags_true_accepts_well_tagged() -> None:
    fn = make_reward_fn(ENV_ID, use_tags=True)
    text = _wrap_with_tags(_zero_json())
    rewards = fn(prompts=[""], completions=[text], instance_seed=[0])
    rec = fn.stats.per_call[-1]
    assert rewards[0] > 0.0
    assert rec["components"]["parse_valid"] == 1.0
    assert rec["components"]["format_valid"] == 1.0


def test_realistic_qwen_style_completion_passes() -> None:
    """A representative completion mimicking what a fine-tuned Qwen would
    emit: prose preamble, tagged reasoning, tagged JSON answer."""
    fn = make_reward_fn(ENV_ID, use_tags=True)
    completion = (
        "Sure, here is my analysis.\n"
        "<think>\n"
        "Looking at the measurement vector y, the support is likely\n"
        "concentrated near low frequencies. I'll guess indices 0..9.\n"
        "</think>\n"
        "<answer>\n"
        + _zero_json() + "\n"
        "</answer>\n"
    )
    rewards = fn(prompts=[""], completions=[completion], instance_seed=[0])
    rec = fn.stats.per_call[-1]
    assert rewards[0] > 0.0
    assert rec["components"]["format_valid"] == 1.0


def test_use_tags_false_existing_fixtures_still_pass() -> None:
    """Backward compat critical test: with ``use_tags=False`` (default
    on the legacy factory), the M2 fixtures (raw JSON, no tags) still
    score above zero — no behaviour change for existing pipelines."""
    fn = make_reward_fn(ENV_ID)
    assert fn.use_tags is False
    rewards = fn(prompts=[""], completions=[_zero_json()], instance_seed=[0])
    # M2 baseline expectation: zero-prediction lands in [0.32, 0.36].
    assert 0.30 <= rewards[0] <= 0.40
    rec = fn.stats.per_call[-1]
    assert rec["components"]["format_valid"] == 1.0


def test_posterior_with_tags_perfect_oracle_passes() -> None:
    """Posterior factory + tags + oracle wrapped → high reward."""
    from verifiable_labs_envs import load_environment
    env = load_environment(ENV_ID, calibration_quantile=2.0)
    inst = env.generate_instance(seed=42)
    support = sorted(int(i) for i in inst.support_true)
    amps = [int(round(float(inst.x_true[i]) * 1000)) for i in support]
    payload = json.dumps({"support_idx": support, "support_amp_x1000": amps})
    text = _wrap_with_tags(payload)

    fn = make_reward_fn_posterior(ENV_ID)  # use_tags=True default
    assert fn.use_tags is True
    rewards = fn(prompts=[""], completions=[text], instance_seed=[42])
    rec = fn.stats.per_call[-1]
    assert rewards[0] > 0.95
    assert rec["components"]["r_format"] == 1.0
    assert rec["components"]["r_outcome"] == 1.0
