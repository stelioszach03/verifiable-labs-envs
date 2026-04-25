"""JSON-safe serialisation of env ``Instance`` payloads.

The envs' ``Instance.as_inputs()`` dicts contain NumPy arrays and
complex numbers, neither of which FastAPI can serialise out of the
box. This module converts them to plain-Python primitives suitable
for ``json.dumps``.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def to_json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` to JSON-safe Python primitives.

    - NumPy arrays → nested lists.
    - Complex scalars → ``{"re": float, "im": float}`` dict.
    - NumPy scalars → equivalent Python int / float / bool.
    - Bytes → UTF-8 string (best-effort) or hex.
    - Sets → sorted lists.
    - Anything else falls through unchanged (FastAPI will then handle
      or raise).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "re": [to_json_safe(v) for v in obj.real.tolist()],
                "im": [to_json_safe(v) for v in obj.imag.tolist()],
            }
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        # numpy scalar — defer to .item()
        return to_json_safe(obj.item())
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_json_safe(v) for v in obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()
    # Last-resort fallback: stringify unknown types so FastAPI doesn't
    # blow up during serialisation. Rare and surfaced in logs.
    return str(obj)
