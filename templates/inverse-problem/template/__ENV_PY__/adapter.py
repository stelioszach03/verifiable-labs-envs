"""LLM adapter for __ENV_ID__.

The adapter shapes how the env exposes itself to text-driven solvers:

- ``system_prompt`` — sets the LLM's role.
- ``build_user_prompt(instance)`` — turns the env's ``Instance`` into
  text the LLM reads.
- ``parse_response(text, instance)`` — turns the LLM's text reply into
  a ``Prediction`` that can be scored.

The adapter is registered with the ``verifiers`` framework so that
``verifiers.load_environment("__ENV_ID__")`` discovers it
automatically; the entry point in ``pyproject.toml`` does the wiring.

TODO: replace ``build_user_prompt`` and ``parse_response`` with the
JSON shape your domain needs.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from __ENV_PY__.data import Instance, Prediction


SYSTEM_PROMPT = (
    "You are an expert solver for the __ENV_ID__ inverse problem "
    "(__DOMAIN__). Given the measurements, return a JSON object with "
    "your reconstruction."
    "\n\nNo prose, no markdown fences — output only the JSON."
)


def build_user_prompt(instance: Instance) -> str:
    """Render the env instance into LLM-readable text.

    TODO: swap the placeholder body for a real serialisation of your
    instance. Examples:
    - ``sparse_fourier``: `(n, k, sigma, mask, y_re, y_im)` packed as
      integer fields scaled x1000.
    - ``mri_knee``: 16x16 zero-filled image as int[0, 255] grid.
    """
    payload = {
        "n": int(np.prod(instance.x_true.shape)),
        "y_x1000": [int(round(float(v) * 1000)) for v in
                    np.asarray(instance.y).ravel().tolist()],
    }
    return (
        "INPUTS:\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"x_hat_x1000": [<n integers, real signal x1000>],\n'
        + ' "sigma_hat_x1000": [<n integers, per-entry uncertainty x1000>]}'
        + "\n\nRespond with the JSON object only."
    )


def parse_response(text: str, instance: Instance) -> Prediction:
    """Parse the LLM's text into a ``Prediction``.

    TODO: tighten the schema — current implementation is permissive
    (returns zero predictions if parsing fails) so the scaffold
    doesn't crash before the user's first edit.
    """
    n = int(np.prod(instance.x_true.shape))
    try:
        # Strip code-fence wrappers if present.
        cleaned = text.strip().lstrip("`").rstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
        data = json.loads(cleaned)
        x_hat = np.array(data.get("x_hat_x1000", [0.0] * n), dtype=np.float64) / 1000.0
        sigma_hat = np.array(
            data.get("sigma_hat_x1000", [1000.0] * n), dtype=np.float64
        ) / 1000.0
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        x_hat = np.zeros(n, dtype=np.float64)
        sigma_hat = np.ones(n, dtype=np.float64)
    return Prediction(
        x_hat=x_hat.reshape(instance.x_true.shape),
        sigma_hat=sigma_hat.reshape(instance.x_true.shape),
    )


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "parse_response"]
