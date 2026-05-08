"""Reward-function tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.corpus import Corpus, Document, NeedleAnchor
from __ENV_PY__.data import NeedleInstance, NeedlePrediction
from __ENV_PY__.reward import (
    DEFAULT_WEIGHTS,
    compute_reward,
    score_components,
)


def _toy_instance(needle: str = "secret-XYZ-1234") -> NeedleInstance:
    body_a = "Lorem ipsum dolor sit amet."
    body_b = f"The reference identifier is {needle}."
    return NeedleInstance(
        question="What identifier appears in the documents?",
        template_name="toy",
        seed=0,
        corpus=Corpus(
            documents=(
                Document(id=0, title="Doc A", body=body_a),
                Document(id=1, title="Doc B", body=body_b),
            ),
            seed=0,
        ),
        needle_text=body_b,
        needle_anchor=NeedleAnchor(
            document_id=1, char_offset=0, needle_text=body_b, is_distractor=False,
        ),
        position_mode="middle",
        metadata={"alpha": 0.1, "target_tokens": 200, "needle_token": needle},
    )


def _toy_prediction(answer: str) -> NeedlePrediction:
    return NeedlePrediction(
        answer=answer,
        raw=json.dumps({"answer": answer, "confidence": 0.5}),
        confidence=0.5,
    )


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_27_d7_a():
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_score_components_garbage_short_circuits():
    inst = _toy_instance()
    pred = NeedlePrediction(answer="", raw="not json", confidence=0.0)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_correct_needle_clears_correctness():
    inst = _toy_instance("FOO-9999")
    pred = _toy_prediction("FOO-9999")
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == pytest.approx(1.0)


def test_score_components_wrong_answer_zeroes_correctness():
    inst = _toy_instance("AAA-0001")
    pred = _toy_prediction("BBB-0002")
    components = score_components(pred, inst)
    assert components["correctness"] == 0.0


def test_compute_reward_with_conformal_emits_covered_flag():
    inst = _toy_instance("FOO-1111")
    pred = _toy_prediction("FOO-1111")
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    meta = out["meta"]
    assert "covered" in meta
    assert isinstance(meta["covered"], bool)
    assert "residual" in meta
    assert 0.0 <= meta["residual"] <= 1.0


def test_compute_reward_carries_completion_hash_and_template():
    inst = _toy_instance("ZZZ-0000")
    pred = _toy_prediction("ZZZ-0000")
    out = compute_reward(prediction=pred, instance=inst)
    assert "completion_hash" in out["meta"]
    assert out["meta"]["template"] == "toy"
    assert out["meta"]["position_mode"] == "middle"
