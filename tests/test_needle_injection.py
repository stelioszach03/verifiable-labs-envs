"""Needle-injection + chain-corpus + verification-helper tests (Phase 27.B).

PHASE_27_PLAN.md §16 Check 6 mandates several named tests below;
the rest of this file pins down position semantics, distractor
shape, and per-comparator boundary cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs.long_context_primitives import (
    NeedleAnchor,
    build_chain_corpus,
    exact_match,
    generate_corpus,
    inject_distractors,
    inject_multiple_needles,
    inject_needle,
    numeric_match,
    token_f1,
)

# ── Mandatory determinism / chain tests ──────────────────────────────


def test_needle_position_seed_determinism() -> None:
    """Same seed + same params → same NeedleAnchor."""
    c = generate_corpus(seed=0, target_tokens=2000, document_count=8)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    _, anchor_a = inject_needle(c, needle_text="The token is X-7421", position="random", rng=rng_a)
    _, anchor_b = inject_needle(c, needle_text="The token is X-7421", position="random", rng=rng_b)
    assert anchor_a == anchor_b


def test_distractor_seed_determinism() -> None:
    """Same seed → same distractor placement."""
    c = generate_corpus(seed=0, target_tokens=2000, document_count=8)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    _, anchors_a = inject_distractors(
        c,
        true_needles=["fact A is 1", "fact B is 2"],
        distractor_needles=["fact A is 99", "fact B is 88"],
        rng=rng_a,
    )
    _, anchors_b = inject_distractors(
        c,
        true_needles=["fact A is 1", "fact B is 2"],
        distractor_needles=["fact A is 99", "fact B is 88"],
        rng=rng_b,
    )
    assert anchors_a == anchors_b


def test_multi_hop_chain_gold_extractable() -> None:
    """``ChainCorpus.gold_chain_doc_ids`` returns the chain in order,
    excluding distractor doc ids."""
    chain = build_chain_corpus(
        seed=42,
        chain_facts=["X is the capital of Y", "Y has population 12345"],
        distractor_facts=["Z has population 99999", "W is unrelated"],
        document_count=8,
        target_tokens=2000,
    )
    doc_ids = chain.gold_chain_doc_ids()
    assert len(doc_ids) == 2
    # No overlap with distractor doc ids.
    distractor_ids = {f.document_id for f in chain.facts if f.is_distractor}
    for did in doc_ids:
        assert did not in distractor_ids
    # Each chain fact's text actually lands in its recorded document.
    for fact in chain.facts:
        if fact.is_distractor:
            continue
        body = chain.corpus.documents[fact.document_id].body
        assert fact.text in body, f"chain fact missing from doc body: {fact.text!r}"


# ── Single-needle injection ──────────────────────────────────────────


def test_inject_needle_writes_into_target_doc() -> None:
    c = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    rng = np.random.default_rng(0)
    new_corpus, anchor = inject_needle(
        c, needle_text="Sentinel-Abc-1234", position="middle", rng=rng,
        document_id=2,
    )
    assert anchor.document_id == 2
    assert "Sentinel-Abc-1234" in new_corpus.documents[2].body
    # Other documents are unchanged.
    for i, (orig, new) in enumerate(zip(c.documents, new_corpus.documents, strict=True)):
        if i == 2:
            continue
        assert orig.body == new.body


def test_inject_needle_position_start_lands_near_beginning() -> None:
    c = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    rng = np.random.default_rng(0)
    new_corpus, anchor = inject_needle(
        c, needle_text="START-MARKER", position="start", rng=rng,
        document_id=0,
    )
    body = new_corpus.documents[0].body
    # The needle must appear in the first ~10% of the body.
    needle_idx = body.index("START-MARKER")
    assert needle_idx <= max(50, len(body) * 0.20), (
        f"start-position needle landed at idx {needle_idx} of body len {len(body)}"
    )


def test_inject_needle_position_end_lands_near_end() -> None:
    c = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    rng = np.random.default_rng(0)
    new_corpus, anchor = inject_needle(
        c, needle_text="END-MARKER", position="end", rng=rng,
        document_id=0,
    )
    body = new_corpus.documents[0].body
    needle_idx = body.index("END-MARKER")
    assert needle_idx >= len(body) * 0.80, (
        f"end-position needle landed at idx {needle_idx} of body len {len(body)}"
    )


def test_inject_needle_rejects_empty_text() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="needle_text"):
        inject_needle(c, needle_text="", position="random", rng=rng)


def test_inject_needle_rejects_invalid_doc_id() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="document_id"):
        inject_needle(c, needle_text="x", position="random", rng=rng, document_id=99)


def test_inject_needle_returns_anchor_with_needle_text() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    _, anchor = inject_needle(
        c, needle_text="The combination is 4-1-7", position="random", rng=rng,
    )
    assert anchor.needle_text == "The combination is 4-1-7"
    assert anchor.is_distractor is False


# ── Multi-needle injection ──────────────────────────────────────────


def test_inject_multiple_needles_uses_distinct_documents() -> None:
    c = generate_corpus(seed=0, target_tokens=2000, document_count=6)
    rng = np.random.default_rng(0)
    _, anchors = inject_multiple_needles(
        c, needles=["A", "B", "C", "D"], rng=rng,
    )
    assert len(anchors) == 4
    doc_ids = {a.document_id for a in anchors}
    # Distinct documents (matches the spec's "one needle per doc").
    assert len(doc_ids) == 4


def test_inject_multiple_needles_rejects_excess() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too many needles"):
        inject_multiple_needles(c, needles=["a", "b", "c", "d"], rng=rng)


def test_inject_multiple_needles_empty_returns_unchanged() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    new_corpus, anchors = inject_multiple_needles(c, needles=[], rng=rng)
    assert anchors == ()
    assert new_corpus.documents == c.documents


# ── Distractor injection ────────────────────────────────────────────


def test_inject_distractors_separates_classes() -> None:
    c = generate_corpus(seed=0, target_tokens=2000, document_count=8)
    rng = np.random.default_rng(0)
    _, anchors = inject_distractors(
        c,
        true_needles=["true 1", "true 2"],
        distractor_needles=["decoy A", "decoy B"],
        rng=rng,
    )
    true_anchors = [a for a in anchors if not a.is_distractor]
    distractor_anchors = [a for a in anchors if a.is_distractor]
    assert len(true_anchors) == 2
    assert len(distractor_anchors) == 2
    # No overlap in document ids.
    true_docs = {a.document_id for a in true_anchors}
    distractor_docs = {a.document_id for a in distractor_anchors}
    assert true_docs.isdisjoint(distractor_docs)


def test_inject_distractors_rejects_overflow() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too many needles"):
        inject_distractors(
            c,
            true_needles=["a", "b"],
            distractor_needles=["c", "d"],
            rng=rng,
        )


def test_inject_distractors_empty_returns_unchanged() -> None:
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    rng = np.random.default_rng(0)
    new_corpus, anchors = inject_distractors(
        c, true_needles=[], distractor_needles=[], rng=rng,
    )
    assert anchors == ()
    assert new_corpus.documents == c.documents


# ── Chain corpus ────────────────────────────────────────────────────


def test_build_chain_corpus_seed_determinism() -> None:
    a = build_chain_corpus(
        seed=42, chain_facts=["fact 1", "fact 2"],
        distractor_facts=["decoy A"], document_count=6, target_tokens=2000,
    )
    b = build_chain_corpus(
        seed=42, chain_facts=["fact 1", "fact 2"],
        distractor_facts=["decoy A"], document_count=6, target_tokens=2000,
    )
    assert a.gold_chain_doc_ids() == b.gold_chain_doc_ids()
    assert a.facts == b.facts


def test_build_chain_corpus_rejects_empty_chain() -> None:
    with pytest.raises(ValueError, match="chain_facts"):
        build_chain_corpus(seed=0, chain_facts=[], document_count=4, target_tokens=1000)


def test_build_chain_corpus_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="too many facts"):
        build_chain_corpus(
            seed=0,
            chain_facts=["a", "b"],
            distractor_facts=["c", "d"],
            document_count=3,
            target_tokens=1000,
        )


def test_chain_corpus_returns_chain_in_order() -> None:
    """The gold chain doc-id list should reflect the order facts were
    passed in (so the verifier can index into the chain by step)."""
    chain = build_chain_corpus(
        seed=0,
        chain_facts=["step A", "step B", "step C"],
        document_count=6,
        target_tokens=2000,
    )
    facts = [f for f in chain.facts if not f.is_distractor]
    assert [f.text for f in facts] == ["step A", "step B", "step C"]


# ── Verification helpers ────────────────────────────────────────────


def test_exact_match_substring_lowercase_default() -> None:
    assert exact_match("the answer is HERMES-7421 indeed", "hermes-7421") is True
    assert exact_match("HERMES-7421", "hermes-7421") is True


def test_exact_match_case_sensitive_when_requested() -> None:
    assert exact_match("hermes-7421", "HERMES-7421", case_sensitive=True) is False
    assert exact_match("HERMES-7421", "HERMES-7421", case_sensitive=True) is True


def test_exact_match_handles_empty_inputs() -> None:
    assert exact_match("", "x") is False
    assert exact_match("x", "") is False
    assert exact_match("", "") is False


def test_numeric_match_with_tolerance() -> None:
    assert numeric_match("the answer is 12345", 12345.0)
    assert numeric_match("12345", 12345.0)
    assert numeric_match("12345.0000001", 12345.0, tol=1e-6)
    assert not numeric_match("12345.5", 12345.0, tol=1e-6)


def test_numeric_match_handles_negative_and_decimal() -> None:
    assert numeric_match("the result was -3.14", -3.14)
    assert numeric_match("approx 0.5 magnitude", 0.5)


def test_numeric_match_returns_false_on_no_number() -> None:
    assert not numeric_match("no number here", 42.0)
    assert not numeric_match("", 42.0)


def test_token_f1_perfect_match() -> None:
    assert token_f1("hello world", "hello world") == pytest.approx(1.0)


def test_token_f1_no_overlap() -> None:
    assert token_f1("hello world", "good morning") == pytest.approx(0.0)


def test_token_f1_partial_overlap() -> None:
    f1 = token_f1("the cat sat", "the cat ran")
    assert 0.0 < f1 < 1.0


def test_token_f1_normalises_punctuation_and_case() -> None:
    """Strips punctuation, lowercases, collapses whitespace."""
    assert token_f1("Hello, World!", "hello world") == pytest.approx(1.0)
    assert token_f1("  HELLO   WORLD  ", "hello world") == pytest.approx(1.0)


def test_token_f1_empty_inputs_zero() -> None:
    assert token_f1("", "anything") == 0.0
    assert token_f1("anything", "") == 0.0
    assert token_f1("", "") == 0.0


def test_needle_anchor_distractor_flag_default_false() -> None:
    a = NeedleAnchor(document_id=0, char_offset=10, needle_text="x")
    assert a.is_distractor is False
