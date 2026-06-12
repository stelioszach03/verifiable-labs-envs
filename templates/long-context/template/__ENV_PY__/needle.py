"""Verification helpers for __ENV_ID__.

Re-exports the platform-level comparators from
``verifiable_labs_envs.long_context_primitives`` so a per-env reward
kernel can swap between substring (``exact_match``), token-F1
(``token_f1``), and numeric (``numeric_match``) without restating
the underlying logic.
"""
from __future__ import annotations

from verifiable_labs_envs.long_context_primitives import (
    exact_match,
    numeric_match,
    token_f1,
)

__all__ = [
    "exact_match",
    "numeric_match",
    "token_f1",
]
