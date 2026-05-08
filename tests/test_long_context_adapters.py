"""Tests for the three long-context env adapters (Phase 27.E)."""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.envs.long_context_needle import (
    generate_instance as generate_needle_instance,
)
from verifiable_labs_envs.envs.long_context_reasoning import (
    generate_instance as generate_reasoning_instance,
)
from verifiable_labs_envs.envs.long_context_synthesis import (
    SynthesisPrediction,
)
from verifiable_labs_envs.envs.long_context_synthesis import (
    generate_instance as generate_synthesis_instance,
)
from verifiable_labs_envs.solvers.adapters.long_context_needle import (
    LongContextNeedleAdapter,
)
from verifiable_labs_envs.solvers.adapters.long_context_reasoning import (
    LongContextReasoningAdapter,
)
from verifiable_labs_envs.solvers.adapters.long_context_synthesis import (
    LongContextSynthesisAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

# ── Adapters auto-registered ────────────────────────────────────────


def test_needle_adapter_registered() -> None:
    assert "long-context-needle" in _ADAPTERS
    assert isinstance(_ADAPTERS["long-context-needle"], LongContextNeedleAdapter)


def test_synthesis_adapter_registered() -> None:
    assert "long-context-synthesis" in _ADAPTERS
    assert isinstance(_ADAPTERS["long-context-synthesis"], LongContextSynthesisAdapter)


def test_reasoning_adapter_registered() -> None:
    assert "long-context-reasoning" in _ADAPTERS
    assert isinstance(_ADAPTERS["long-context-reasoning"], LongContextReasoningAdapter)


# ── Needle adapter ─────────────────────────────────────────────────


def test_needle_adapter_round_trip() -> None:
    adapter = LongContextNeedleAdapter()
    inst = generate_needle_instance(seed=0)
    prompt = adapter.build_user_prompt(inst)
    assert "QUESTION:" in prompt
    payload = json.dumps({"answer": "ABCD-1234", "confidence": 0.6})
    pred = adapter.parse_response(payload, inst)
    assert pred.answer == "ABCD-1234"
    assert pred.confidence == pytest.approx(0.6)


# ── Synthesis adapter ──────────────────────────────────────────────


def test_synthesis_adapter_round_trip() -> None:
    adapter = LongContextSynthesisAdapter()
    inst = generate_synthesis_instance(seed=0)
    prompt = adapter.build_user_prompt(inst)
    assert "QUESTION:" in prompt
    payload = json.dumps({"answer": "summary text", "confidence": 0.5})
    pred = adapter.parse_response(payload, inst)
    assert pred.answer == "summary text"


def test_synthesis_adapter_followup_does_not_leak_gold() -> None:
    """R10: inter-turn feedback must NOT carry the gold answer."""
    adapter = LongContextSynthesisAdapter()
    inst = generate_synthesis_instance(seed=0)
    last_pred = SynthesisPrediction(answer="", raw="", confidence=0.0)
    feedback = adapter.build_followup_turn(history=[], last_prediction=last_pred, instance=inst)
    assert "FEEDBACK" in feedback
    assert inst.gold_answer not in feedback


def test_synthesis_adapter_followup_with_no_prediction() -> None:
    """The adapter must handle ``last_prediction=None`` (parse failure)."""
    adapter = LongContextSynthesisAdapter()
    inst = generate_synthesis_instance(seed=0)
    feedback = adapter.build_followup_turn(history=[], last_prediction=None, instance=inst)
    assert "FEEDBACK" in feedback


# ── Reasoning adapter ─────────────────────────────────────────────


def test_reasoning_adapter_round_trip() -> None:
    adapter = LongContextReasoningAdapter()
    inst = generate_reasoning_instance(seed=0)
    prompt = adapter.build_user_prompt(inst)
    assert "QUESTION:" in prompt
    payload = json.dumps({"answer": "12345", "confidence": 0.4})
    pred = adapter.parse_response(payload, inst)
    assert pred.answer == "12345"
