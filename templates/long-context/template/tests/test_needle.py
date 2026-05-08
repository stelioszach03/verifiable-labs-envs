"""Verification-helper contract tests for __ENV_ID__."""
from __future__ import annotations

from __ENV_PY__.needle import exact_match, numeric_match, token_f1


def test_exact_match_substring_case_insensitive():
    assert exact_match("the answer is FOO-1234", "FOO-1234")
    assert exact_match("the answer is foo-1234", "FOO-1234")  # case-insensitive
    assert not exact_match("the answer is BAR-9999", "FOO-1234")


def test_exact_match_requires_non_empty():
    assert not exact_match("", "FOO-1234")
    assert not exact_match("foo", "")


def test_numeric_match_with_tolerance():
    assert numeric_match("42", 42.0)
    assert numeric_match("the answer is 42", 42.0)
    assert numeric_match("3.14159", 3.14, tol=1e-2)
    assert not numeric_match("99", 42.0)


def test_token_f1_basic():
    # Identical strings → F1 = 1.0.
    assert token_f1("the quick brown fox", "the quick brown fox") == 1.0
    # Disjoint strings → F1 = 0.0.
    assert token_f1("apple", "banana") == 0.0
    # Partial overlap → 0 < F1 < 1.
    f1 = token_f1("the quick brown fox", "the slow brown dog")
    assert 0.0 < f1 < 1.0
