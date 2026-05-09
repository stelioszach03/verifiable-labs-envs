"""Cadence math for continuous-capability monitors (Phase 28.B).

PHASE_28_PLAN.md §5 D3-A locks three cadences: ``daily``, ``weekly``,
``monthly``. The corresponding "runs per month" multiplier is used by
the projected-cost calculator (§12) and the next-run-at scheduler tick
(§7); the ``compute_next_run_at`` helper produces the next anchor given
the current cadence + a reference timestamp.

Catch-up semantics (§7): after extended downtime, missed runs fire
**once** on recovery — `compute_next_run_at(anchor=now)` always returns
a *future* timestamp, never iteratively forwards through skipped slots.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

CadenceName = Literal["daily", "weekly", "monthly"]

# Runs-per-month constants used by the projected-cost calculator.
# Monthly = 1 (single fire); weekly = 4 (4.34 runs/mo rounded down,
# matches the simpler customer-facing figure); daily = 30.
RUNS_PER_MONTH: dict[str, int] = {
    "daily": 30,
    "weekly": 4,
    "monthly": 1,
}

_CADENCE_DELTAS: dict[str, timedelta] = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    # Approximate month — 30 days. The plan locks this for v0.0.1
    # alpha; the next iteration will shift to calendar-month math.
    "monthly": timedelta(days=30),
}


def runs_per_month(cadence: CadenceName) -> int:
    """Number of monitor runs the given cadence implies per 30 days."""
    if cadence not in RUNS_PER_MONTH:
        raise ValueError(f"unknown cadence: {cadence!r}")
    return RUNS_PER_MONTH[cadence]


def compute_next_run_at(
    cadence: CadenceName,
    *,
    anchor: datetime | None = None,
) -> datetime:
    """Return the next firing time for a monitor with this cadence.

    The result is always strictly in the future relative to ``anchor``
    (default: ``datetime.now(UTC)``). After extended downtime the
    catch-up loop sets ``next_run_at`` to ``anchor + delta`` rather
    than iterating through every missed slot.
    """
    if cadence not in _CADENCE_DELTAS:
        raise ValueError(f"unknown cadence: {cadence!r}")
    base = anchor or datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return base + _CADENCE_DELTAS[cadence]


def projected_monthly_episodes(
    cadence: CadenceName,
    n_envs: int,
    episodes_per_env: int,
) -> int:
    """Approximate per-month episode count for a monitor configuration.

    Used by the projected-cost calculator at create-time to enforce
    tier ceilings (PHASE_28_PLAN.md §5 D8-C / §12).
    """
    return runs_per_month(cadence) * int(n_envs) * int(episodes_per_env)


__all__ = [
    "CadenceName",
    "RUNS_PER_MONTH",
    "compute_next_run_at",
    "projected_monthly_episodes",
    "runs_per_month",
]
