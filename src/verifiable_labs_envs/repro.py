"""Reproducibility metadata helpers.

Provides three deterministic identifiers that go into every
:class:`~verifiable_labs_envs.traces.Trace`'s ``metadata`` dict (no
schema change to ``traces.py`` — they live as free-form keys):

* ``config_hash``   — pinpoints the run-level configuration (model, seed
  range, env, hyperparameters). Excludes cosmetic / runtime-only fields
  (``timestamp``, ``wandb_run_id``, ``host``) so two runs with different
  wall clocks but the same experimental setup hash to the same value.
* ``instance_hash`` — pinpoints the per-episode env state (env id +
  version + seed + prior parameters). Same inputs across machines /
  Python versions produce the same hash.
* ``reward_hash``   — a 6-decimal quantization of the float reward,
  serialised as a string. Not cryptographic; the name preserves
  symmetry with the other two and makes per-trace equality checks
  robust against floating-point noise.

The two cryptographic hashes delegate to
:func:`verifiable_labs_envs.traces.hash_payload` so this module never
diverges from the rest of the codebase's canonical-JSON serialisation.
``hash_payload`` returns ``"sha256:" + sha256(canonical_json)[:16]``.
"""
from __future__ import annotations

import json
from typing import Any

from verifiable_labs_envs.traces import hash_payload

# Fields that must NOT contribute to ``config_hash`` — they vary across
# otherwise-identical runs and would defeat the dedup property.
EXCLUDED_FROM_CONFIG_HASH: frozenset[str] = frozenset(
    {"timestamp", "wandb_run_id", "host"}
)


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to a deterministic JSON string.

    Sorted keys, no whitespace, ASCII-safe. Matches the encoding used
    by :func:`verifiable_labs_envs.traces.hash_payload` so independent
    callers produce byte-identical canonical forms.

    Parameters
    ----------
    obj : Any
        Anything ``json.dumps`` can handle, including objects with a
        ``__str__`` fallback (delegated via ``default=str``).

    Returns
    -------
    str
        Canonical JSON string. Always ASCII; non-ASCII chars are
        escaped as ``\\uXXXX``.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of a run config, excluding cosmetic / runtime-only fields.

    Parameters
    ----------
    cfg : dict[str, Any]
        Run-level configuration. Keys in
        :data:`EXCLUDED_FROM_CONFIG_HASH` (``timestamp``,
        ``wandb_run_id``, ``host``) are dropped before hashing.

    Returns
    -------
    str
        Format ``"sha256:<16-hex>"``.
    """
    filtered = {
        k: v for k, v in cfg.items() if k not in EXCLUDED_FROM_CONFIG_HASH
    }
    return hash_payload(filtered)


def instance_hash(
    env_id: str,
    env_version: str | None,
    seed: int,
    prior_params: dict[str, Any] | None,
) -> str:
    """Stable hash of an episode's env-instance inputs.

    Two invocations with the same arguments — across machines, Python
    versions, package re-installs — produce the same hash, by virtue
    of the canonical-JSON encoding. The intended use is paired
    pre/post comparison: pair traces by ``instance_hash`` rather than
    ``seed`` so silent env-version drift can't masquerade as
    same-seed evaluation.

    Parameters
    ----------
    env_id : str
        Registered env id, e.g. ``"sparse-fourier-recovery"``.
    env_version : str | None
        Env semantic-version label. ``None`` is allowed but should be
        avoided in production (means "version unknown").
    seed : int
        Per-episode RNG seed used by the env's ``generate_instance``.
    prior_params : dict[str, Any] | None
        Env hyperparameters passed through ``load_environment``
        (``n``, ``m``, ``k``, ``sigma``, etc., for sparse-fourier).
        ``None`` is normalised to ``{}``.

    Returns
    -------
    str
        Format ``"sha256:<16-hex>"``.
    """
    payload = {
        "env_id": env_id,
        "env_version": env_version,
        "seed": int(seed),
        "prior_params": prior_params or {},
    }
    return hash_payload(payload)


def reward_hash(reward: float) -> str:
    """Six-decimal quantisation of ``reward`` as a string.

    Despite the name this is not a cryptographic hash — it's a stable
    string representation of the reward that ignores float noise below
    1e-6. Use it to dedupe / compare rewards between runs without
    being bitten by ``0.30000000000000004``-style artefacts.

    Examples
    --------
    >>> reward_hash(0.5)
    '0.500000'
    >>> reward_hash(0.123456789)
    '0.123457'
    >>> reward_hash(-0.5)
    '-0.500000'
    """
    return f"{float(reward):.6f}"


__all__ = [
    "EXCLUDED_FROM_CONFIG_HASH",
    "canonical_json",
    "config_hash",
    "instance_hash",
    "reward_hash",
]
