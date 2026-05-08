"""Corpus-primitive contract tests for __ENV_ID__.

The env's reward kernel relies on
:func:`verifiable_labs_envs.long_context_primitives.generate_corpus`
and the needle-injection helpers. The local re-export in
``__ENV_PY__.corpus`` must hand back the same surface; the
platform-level token-budget + determinism suite lives in the
parent repo's ``tests/test_long_context_corpus.py``.
"""
from __future__ import annotations

import numpy as np

from __ENV_PY__.corpus import (
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_TEST_TOKENS,
    Corpus,
    Document,
    count_tokens,
    generate_corpus,
    inject_needle,
)


def test_re_exports_match_platform_defaults():
    assert DEFAULT_TEST_TOKENS == 4_000
    assert DEFAULT_DOCUMENT_COUNT == 8


def test_corpus_smoke_generates():
    corpus = generate_corpus(seed=0, target_tokens=512, document_count=2)
    assert isinstance(corpus, Corpus)
    assert len(corpus.documents) == 2
    for doc in corpus.documents:
        assert isinstance(doc, Document)
        assert doc.body.strip()


def test_count_tokens_returns_int():
    n = count_tokens("hello world")
    assert isinstance(n, int)
    assert n > 0


def test_inject_needle_smoke():
    corpus = generate_corpus(seed=1, target_tokens=512, document_count=2)
    rng = np.random.default_rng(0)
    new_corpus, anchor = inject_needle(
        corpus, needle_text="The secret is FOO-1234", position="middle", rng=rng,
    )
    assert anchor.document_id in (0, 1)
    assert "FOO-1234" in new_corpus.documents[anchor.document_id].body
