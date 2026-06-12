"""Model-id anonymisation for customer-facing audit reports.

Sales artefacts (the LaTeX/PDF report) need to omit the real model id
under cold-outreach branding ("Frontier Model A"). The simplest design
puts the substitution at the boundary between :class:`AuditStats` and
the renderer: replace ``stats.model`` with the desired label, and every
downstream reader (LaTeX context, figure captions, citation block,
reproducibility table) automatically sees the anonymous name.

For single-model audits (the v0.0.1 surface area) we use a single label
— defaulting to ``"Frontier Model A"`` — overrideable via the
``--anonymize-labels`` CLI flag (CSV, first element wins) or the
``anonymize_label`` field in the YAML config (preserved from 17.A).
The CSV plural form is forward-compatible with v0.0.2's multi-model
audits.
"""
from __future__ import annotations

from collections.abc import Iterable

from vlabs_audit.stats import AuditStats

DEFAULT_LABELS: tuple[str, ...] = (
    "Frontier Model A",
    "Frontier Model B",
    "Frontier Model C",
    "Frontier Model D",
    "Frontier Model E",
)


def parse_anonymize_labels(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated label list, trimming whitespace.

    Returns ``None`` when the input is ``None`` (caller falls back to
    config default) or when the input contains no non-empty labels.
    """
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    parts = [p for p in parts if p]
    return tuple(parts) if parts else None


def anonymize_audit_stats(
    stats: AuditStats,
    *,
    label: str = DEFAULT_LABELS[0],
) -> AuditStats:
    """Return a copy of ``stats`` with ``stats.model`` replaced by ``label``.

    The original :class:`AuditStats` is not mutated — pydantic's
    ``model_copy`` returns a fresh instance, leaving the on-disk
    ``audits.config_json`` and the original object untouched.

    Raises :class:`ValueError` for an empty / whitespace-only label;
    otherwise the label is taken verbatim (LaTeX escaping happens later
    in the renderer pipeline).
    """
    cleaned = (label or "").strip()
    if not cleaned:
        raise ValueError("anonymize_audit_stats: label must be non-empty")
    return stats.model_copy(update={"model": cleaned})


def resolve_anonymize_label(
    *,
    explicit_labels: tuple[str, ...] | None,
    config_label: str | None,
    fallback: str = DEFAULT_LABELS[0],
) -> str:
    """Pick the label for a single-model audit.

    Priority: CLI ``--anonymize-labels`` (first element)
    → YAML ``anonymize_label`` → built-in ``DEFAULT_LABELS[0]``.
    """
    if explicit_labels:
        return explicit_labels[0]
    if config_label:
        stripped = config_label.strip()
        if stripped:
            return stripped
    return fallback


__all__: Iterable[str] = [
    "DEFAULT_LABELS",
    "anonymize_audit_stats",
    "parse_anonymize_labels",
    "resolve_anonymize_label",
]
