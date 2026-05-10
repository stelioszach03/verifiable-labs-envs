"""Tests for ``vlabs_api.storage._build_key`` and ``_ext_for_format``.

Phase 28 validation surfaced a cosmetic bug: any ``output_format``
other than ``"parquet"`` was forced to a ``.jsonl`` extension on disk,
so the monitor PDF artefacts ended up named ``pdf.jsonl``. The fix
introduced ``_ext_for_format`` with a small format→extension map and
a sensible fallback. These tests pin the new behaviour so the bug
can't silently regress.
"""
from __future__ import annotations

import pytest

from vlabs_api import storage


# ── _ext_for_format unit tests ─────────────────────────────────────


def test_ext_for_format_parquet() -> None:
    assert storage._ext_for_format("parquet") == "parquet"


def test_ext_for_format_jsonl() -> None:
    assert storage._ext_for_format("jsonl") == "jsonl"


def test_ext_for_format_pdf_no_longer_falls_back_to_jsonl() -> None:
    """Phase 28 regression: PDF outputs were saved as ``pdf.jsonl``."""
    assert storage._ext_for_format("pdf") == "pdf"


def test_ext_for_format_csv() -> None:
    assert storage._ext_for_format("csv") == "csv"


def test_ext_for_format_unknown_format_falls_through_to_format_string() -> None:
    """A future format that's not in the map should still produce
    a sensible filename — using the format string as the extension."""
    assert storage._ext_for_format("avro") == "avro"


def test_ext_for_format_strips_leading_dot_and_lowercases() -> None:
    assert storage._ext_for_format(".PDF") == "pdf"


def test_ext_for_format_blank_input_falls_back_to_bin() -> None:
    """Defensive: an empty / whitespace format should not produce a
    filename ending in a bare ``.`` — return ``bin`` so the on-disk
    artefact is still distinguishable."""
    assert storage._ext_for_format("") == "bin"
    assert storage._ext_for_format("   ") == "bin"


# ── _build_key integration ─────────────────────────────────────────


def test_build_key_pdf_filename_now_has_pdf_ext() -> None:
    key = storage._build_key("user42", "ds_abc", "pdf")
    assert key == "user42/ds_abc/pdf.pdf"


def test_build_key_parquet_unchanged() -> None:
    key = storage._build_key("u", "d", "parquet")
    assert key == "u/d/parquet.parquet"


def test_build_key_jsonl_unchanged() -> None:
    key = storage._build_key("u", "d", "jsonl")
    assert key == "u/d/jsonl.jsonl"


def test_build_key_csv_now_has_csv_ext() -> None:
    key = storage._build_key("u", "d", "csv")
    assert key == "u/d/csv.csv"
