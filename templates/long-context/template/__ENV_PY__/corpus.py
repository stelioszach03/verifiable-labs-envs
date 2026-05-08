"""Corpus + needle-injection helpers for __ENV_ID__.

Re-exports the platform-level shared library at
``verifiable_labs_envs.long_context_primitives``. Per-env scaffolds
keep this thin indirection so a customised topic template pool or
needle-injection policy can be wired in without touching the env /
reward code.
"""
from __future__ import annotations

from verifiable_labs_envs.long_context_primitives import (
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_MAX_CORPUS_BYTES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEST_TOKENS,
    SANDBOX_TEST_TOKENS,
    TOPIC_TEMPLATE_NAMES,
    ChainCorpus,
    ChainFact,
    Corpus,
    Document,
    NeedleAnchor,
    PositionMode,
    build_chain_corpus,
    count_tokens,
    generate_corpus,
    inject_distractors,
    inject_multiple_needles,
    inject_needle,
)

__all__ = [
    "DEFAULT_DOCUMENT_COUNT",
    "DEFAULT_MAX_CORPUS_BYTES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEST_TOKENS",
    "SANDBOX_TEST_TOKENS",
    "TOPIC_TEMPLATE_NAMES",
    "ChainCorpus",
    "ChainFact",
    "Corpus",
    "Document",
    "NeedleAnchor",
    "PositionMode",
    "build_chain_corpus",
    "count_tokens",
    "generate_corpus",
    "inject_distractors",
    "inject_multiple_needles",
    "inject_needle",
]
