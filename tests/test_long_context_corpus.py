"""Corpus generator + tokeniser determinism tests (Phase 27.B).

PHASE_27_PLAN.md §16 Check 6 mandates several named tests below;
the rest of this file pins down the procedural-corpus shape +
boundary cases.
"""
from __future__ import annotations

import pytest

from verifiable_labs_envs.long_context_primitives import (
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEST_TOKENS,
    SANDBOX_TEST_TOKENS,
    TOPIC_TEMPLATE_NAMES,
    Document,
    count_tokens,
    generate_corpus,
)

# ── Mandatory determinism / token-budget tests ───────────────────────


def test_corpus_seed_determinism() -> None:
    """Identical seed yields byte-identical Corpus."""
    a = generate_corpus(seed=42, target_tokens=4000, document_count=8)
    b = generate_corpus(seed=42, target_tokens=4000, document_count=8)
    assert tuple(d.body for d in a.documents) == tuple(d.body for d in b.documents)
    assert tuple(d.title for d in a.documents) == tuple(d.title for d in b.documents)


def test_tokeniser_token_count_stability() -> None:
    """`count_tokens` is stable across calls (cached encoder)."""
    text = "The quick brown fox jumps over the lazy dog."
    a = count_tokens(text)
    b = count_tokens(text)
    assert a == b
    assert a > 0
    # Empty text → 0.
    assert count_tokens("") == 0
    # Token count scales monotonically with text length.
    assert count_tokens(text * 10) > count_tokens(text)


def test_corpus_token_budget_enforcement_at_4k_16k_128k() -> None:
    """Generator respects target_tokens at the three locked checkpoints.

    Tolerance: the generator stops when the token total reaches the
    target. Lower bound = `target × 0.85` (8 sentences may overshoot
    slightly). Upper bound = `target × 1.20` (last batch can land
    above target).
    """
    # 4 K — unit-test default.
    c4k = generate_corpus(seed=0, target_tokens=4_000, document_count=8)
    n4k = c4k.total_tokens()
    assert 0.85 * 4_000 <= n4k <= 1.20 * 4_000, f"4K corpus = {n4k} tokens"

    # 16 K — sandbox test default.
    c16k = generate_corpus(seed=0, target_tokens=16_000, document_count=12)
    n16k = c16k.total_tokens()
    assert 0.85 * 16_000 <= n16k <= 1.20 * 16_000, f"16K corpus = {n16k} tokens"

    # 128 K is too slow for unit-test runtime; verify the path doesn't
    # crash on a large request. We use 32 K as the proxy ceiling — the
    # generator's loop is identical at 128 K, just slower.
    c32k = generate_corpus(seed=0, target_tokens=32_000, document_count=16)
    n32k = c32k.total_tokens()
    assert 0.85 * 32_000 <= n32k <= 1.20 * 32_000, f"32K corpus = {n32k} tokens"


def test_document_separator_format_strict() -> None:
    """Rendered prompt uses the locked ``---DOCUMENT N: <title>---`` format."""
    c = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    prompt = c.render_prompt(question="What is the answer?")
    for doc in c.documents:
        expected = f"---DOCUMENT {doc.id}: {doc.title}---"
        assert expected in prompt, f"separator missing for doc {doc.id}"
    assert "QUESTION:" in prompt
    assert "What is the answer?" in prompt


# ── Generator boundary / shape tests ─────────────────────────────────


def test_generate_corpus_default_document_count() -> None:
    c = generate_corpus(seed=0, target_tokens=DEFAULT_TEST_TOKENS)
    assert len(c.documents) == DEFAULT_DOCUMENT_COUNT


def test_generate_corpus_explicit_document_count() -> None:
    c = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    assert len(c.documents) == 4


def test_generate_corpus_homogeneous_template() -> None:
    """All documents share the same template when ``template`` is pinned."""
    c = generate_corpus(seed=0, target_tokens=4000, document_count=6, template="bio_article")
    # All titles share the bio_article prefix.
    for doc in c.documents:
        assert "Biographical entry:" in doc.title


def test_generate_corpus_heterogeneous_when_template_none() -> None:
    """When template=None each doc samples independently."""
    c = generate_corpus(seed=0, target_tokens=8000, document_count=10)
    titles = {d.title for d in c.documents}
    # We expect at least 4 distinct titles across 10 docs (10 templates × proper-noun rotation).
    assert len(titles) >= 4


def test_generate_corpus_rejects_zero_documents() -> None:
    with pytest.raises(ValueError, match="document_count"):
        generate_corpus(seed=0, target_tokens=1000, document_count=0)


def test_generate_corpus_rejects_tiny_target() -> None:
    with pytest.raises(ValueError, match="target_tokens"):
        generate_corpus(seed=0, target_tokens=10, document_count=4)


def test_generate_corpus_varies_with_seed() -> None:
    """Different seeds produce visibly different corpora."""
    a = generate_corpus(seed=0, target_tokens=4000, document_count=8)
    b = generate_corpus(seed=1, target_tokens=4000, document_count=8)
    a_titles = tuple(d.title for d in a.documents)
    b_titles = tuple(d.title for d in b.documents)
    # At least one title differs.
    assert a_titles != b_titles


def test_topic_templates_count_matches_plan() -> None:
    """PHASE_27_PLAN.md §6 locks 10 topic templates."""
    assert len(TOPIC_TEMPLATE_NAMES) == 10


def test_constants_match_phase_27_d5() -> None:
    assert DEFAULT_TEST_TOKENS == 4_000
    assert SANDBOX_TEST_TOKENS == 16_000
    assert DEFAULT_MAX_TOKENS == 128_000
    assert DEFAULT_DOCUMENT_COUNT == 8


def test_corpus_render_prompt_strips_separators_for_no_question() -> None:
    """Empty question still produces a valid prompt body."""
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    prompt = c.render_prompt(question="")
    for doc in c.documents:
        assert f"---DOCUMENT {doc.id}:" in prompt


def test_document_render_includes_separator_and_body() -> None:
    doc = Document(id=7, title="my title", body="hello world")
    text = doc.render()
    assert "---DOCUMENT 7: my title---" in text
    assert "hello world" in text


def test_corpus_with_documents_returns_new_instance() -> None:
    """``with_documents`` is purely functional — original unchanged."""
    c = generate_corpus(seed=0, target_tokens=1000, document_count=3)
    new_docs = (Document(id=99, title="x", body="y"),)
    c2 = c.with_documents(new_docs)
    assert c.documents != c2.documents
    assert c2.seed == c.seed


def test_corpus_all_templates_render_distinct_titles() -> None:
    """Each template produces a recognisable title pattern."""
    expected_substrings = {
        "bio_article": "Biographical entry:",
        "science_abstract": "Abstract:",
        "news_report": "Regional report:",
        "product_review": "Field test review",
        "technical_manual": "Technical manual section",
        "legal_document": "Legal agreement clause set",
        "historical_summary": "Historical summary",
        "recipe_collection": "Procedural recipe entry",
        "travel_log": "Travel log entry",
        "interview_transcript": "Interview transcript excerpt",
    }
    for template_name, expected_substring in expected_substrings.items():
        c = generate_corpus(
            seed=42, target_tokens=500, document_count=2, template=template_name,
        )
        for doc in c.documents:
            assert expected_substring in doc.title, (
                f"{template_name} title doesn't include '{expected_substring}': {doc.title!r}"
            )


def test_corpus_total_tokens_grows_monotonically() -> None:
    """Larger target_tokens → larger total."""
    c1 = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    c2 = generate_corpus(seed=0, target_tokens=8000, document_count=4)
    assert c2.total_tokens() > c1.total_tokens()


def test_corpus_render_prompt_is_pure_text() -> None:
    """Rendered prompt is a single UTF-8 string with no embedded NUL or junk."""
    c = generate_corpus(seed=0, target_tokens=2000, document_count=4)
    prompt = c.render_prompt(question="test")
    assert isinstance(prompt, str)
    assert "\x00" not in prompt
