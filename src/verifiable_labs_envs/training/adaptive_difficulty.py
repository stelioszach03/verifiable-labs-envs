"""RLVE-style adaptive-difficulty tracker for curriculum GRPO training.

Faithful to Zeng et al. 2025 (arXiv:2511.07317), Algorithm 1: maintain a
window ``[l, h]`` of admissible difficulty levels per env, sample uniformly
inside the window per rollout, and advance ``h`` whenever the recent
success rate at the current ceiling clears a threshold.

The tracker itself does not know about env shapes — it only manages
integer difficulty levels. The translation from a difficulty integer to a
concrete env-kwargs dict (``n``, ``m``, ``k`` for ``sparse-fourier-recovery``)
is done by :func:`difficulty_to_kwargs`, which looks up an anchor table.
This keeps the env logic untouched (Hard Rule #7): we just pass the
returned kwargs into the env's existing ``generate_instance(seed, **kwargs)``.

Design notes:

* Anchors are sparse (every 5 levels). Difficulty integers between
  anchors snap *down* to the nearest defined anchor — i.e. crossing
  an anchor boundary triggers a real shape change. This means
  ``h += 1`` within the same anchor interval keeps the env identical;
  the algorithm only steps "up" when an anchor boundary is crossed.
  RLVE-faithful and easy to reason about for a 5-tier curriculum.
* The tracker is JSON-serialisable via :meth:`to_dict` / :meth:`from_dict`
  so a multi-env joint run can checkpoint each per-env tracker alongside
  the model state and resume after VM death.
"""
from __future__ import annotations

import dataclasses
import logging
import random
from typing import Any

logger = logging.getLogger("verifiable_labs_envs.training.adaptive_difficulty")


# ── difficulty schedule ────────────────────────────────────────────────


_SPARSE_FOURIER_ANCHORS: list[tuple[int, dict[str, Any]]] = [
    # (difficulty_threshold, env_kwargs)
    # m is set to roughly n // 4 (matches the env's default 4× subsampling).
    (0,  {"n": 64,  "m": 16, "k": 3,  "sigma": 0.05, "alpha": 0.1}),
    (5,  {"n": 96,  "m": 24, "k": 5,  "sigma": 0.05, "alpha": 0.1}),
    (10, {"n": 128, "m": 32, "k": 8,  "sigma": 0.05, "alpha": 0.1}),
    (15, {"n": 192, "m": 48, "k": 12, "sigma": 0.05, "alpha": 0.1}),
    (20, {"n": 256, "m": 64, "k": 20, "sigma": 0.05, "alpha": 0.1}),
]

# Phase-retrieval defaults: n=32, m=24, k=4 (3× oversampling).
_PHASE_RETRIEVAL_ANCHORS: list[tuple[int, dict[str, Any]]] = [
    (0,  {"n": 16, "m": 12, "k": 2, "sigma": 0.02, "alpha": 0.1}),
    (5,  {"n": 24, "m": 18, "k": 3, "sigma": 0.02, "alpha": 0.1}),
    (10, {"n": 32, "m": 24, "k": 4, "sigma": 0.02, "alpha": 0.1}),  # ≈default
    (15, {"n": 48, "m": 32, "k": 6, "sigma": 0.02, "alpha": 0.1}),
    (20, {"n": 64, "m": 48, "k": 8, "sigma": 0.02, "alpha": 0.1}),
]

# Super-resolution: shape × factor controls image difficulty.
# Smaller images @ lower factor are easier; defaults are (128,128) @ 4×.
_SUPER_RESOLUTION_ANCHORS: list[tuple[int, dict[str, Any]]] = [
    (0,  {"shape": (32, 32),   "factor": 2}),
    (5,  {"shape": (48, 48),   "factor": 2}),
    (10, {"shape": (64, 64),   "factor": 4}),
    (15, {"shape": (96, 96),   "factor": 4}),
    (20, {"shape": (128, 128), "factor": 4}),  # default
]

#: Mapping ``env_id`` → list of (threshold, kwargs) anchors. Anchors
#: must be sorted by threshold ascending. Add new entries here when
#: extending the curriculum to additional envs.
ANCHOR_TABLES: dict[str, list[tuple[int, dict[str, Any]]]] = {
    "sparse-fourier-recovery": _SPARSE_FOURIER_ANCHORS,
    "phase-retrieval": _PHASE_RETRIEVAL_ANCHORS,
    "super-resolution-div2k-x4": _SUPER_RESOLUTION_ANCHORS,
}


def difficulty_to_kwargs(env_id: str, difficulty: int) -> dict[str, Any]:
    """Map an integer difficulty level to env kwargs (anchor snap-down).

    Parameters
    ----------
    env_id : str
        Registered env id, e.g. ``"sparse-fourier-recovery"``.
    difficulty : int
        Non-negative integer difficulty. Negative values are clamped
        to 0; values above the maximum anchor are clamped to that
        anchor.

    Returns
    -------
    dict[str, Any]
        kwargs dict to pass into ``env.generate_instance(seed, **kwargs)``.
        Empty dict if ``env_id`` has no anchor table — the caller should
        fall back to env defaults.
    """
    anchors = ANCHOR_TABLES.get(env_id)
    if not anchors:
        return {}
    d = max(int(difficulty), 0)
    selected = anchors[0][1]
    for thresh, kwargs in anchors:
        if d >= thresh:
            selected = kwargs
        else:
            break
    return dict(selected)


def max_difficulty(env_id: str) -> int:
    """Highest defined anchor threshold for ``env_id``. 0 if unknown."""
    anchors = ANCHOR_TABLES.get(env_id)
    if not anchors:
        return 0
    return int(anchors[-1][0])


# ── tracker ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class AdaptiveDifficultyTracker:
    """RLVE Algorithm-1 tracker: bookkeeping of [l, h] difficulty window.

    Parameters
    ----------
    env_id : str
        Env this tracker manages.
    tau_acc : float, default 0.9
        Success-rate threshold at the current ceiling ``h`` to trigger
        advancement.
    tau_num : int, default 32
        Minimum number of rollouts at ``h`` before checking the
        success-rate threshold.
    d_delta : int, default 4
        Maximum width of the active window ``[l, h]`` — i.e.
        ``h - l + 1 <= d_delta``. Once exceeded, ``l`` is bumped up.
    success_threshold : float, default 0.5
        Reward at or above this counts as a "success" for the
        success-rate calculation.
    """

    env_id: str
    tau_acc: float = 0.9
    tau_num: int = 32
    d_delta: int = 4
    success_threshold: float = 0.5

    # State
    l: int = 0  # noqa: E741
    h: int = 0
    a: int = 0  # successes at h since last check
    b: int = 0  # rollouts at h since last check
    advances: int = 0  # total times we've advanced h
    rollouts_total: int = 0  # all-time rollouts recorded

    def __post_init__(self) -> None:
        if not (0.0 < self.tau_acc <= 1.0):
            raise ValueError(f"tau_acc must be in (0, 1], got {self.tau_acc}")
        if self.tau_num <= 0:
            raise ValueError(f"tau_num must be positive, got {self.tau_num}")
        if self.d_delta <= 0:
            raise ValueError(f"d_delta must be positive, got {self.d_delta}")

    def sample_difficulty(self, rng: random.Random | None = None) -> int:
        """Sample uniformly from ``[l, h]`` inclusive."""
        r = rng if rng is not None else random
        return r.randint(self.l, self.h)

    def record_rollout(self, difficulty: int, reward: float) -> None:
        """Record a rollout result. Counters at the ceiling ``h`` only.

        Rollouts at difficulty < ``h`` don't contribute to the
        advancement decision (they're "easy" by construction); we only
        care whether the model has saturated the current ceiling.
        """
        self.rollouts_total += 1
        if int(difficulty) == self.h:
            self.b += 1
            if float(reward) >= self.success_threshold:
                self.a += 1

    def maybe_advance(self) -> bool:
        """Check the advancement condition. Returns True iff ``h`` was bumped.

        After the check (whether or not advancement occurred), the local
        success/total counters are reset so the next window is clean.
        """
        if self.b < self.tau_num:
            return False
        advanced = False
        if self.b > 0 and (self.a / self.b) >= self.tau_acc:
            self.h += 1
            self.advances += 1
            # Slide the window if it exceeds the budget width.
            if self.h - self.l + 1 > self.d_delta:
                self.l = self.h - self.d_delta + 1
            advanced = True
            logger.info(
                "tracker[%s] advanced to h=%d (l=%d). a/b=%d/%d",
                self.env_id, self.h, self.l, self.a, self.b,
            )
        # Reset window counters whether or not we advanced.
        self.a = 0
        self.b = 0
        return advanced

    def stats(self) -> dict[str, Any]:
        """Snapshot of internal state for logging or debugging."""
        return {
            "env_id": self.env_id,
            "l": self.l,
            "h": self.h,
            "a": self.a,
            "b": self.b,
            "advances": self.advances,
            "rollouts_total": self.rollouts_total,
            "current_success_rate": (self.a / self.b) if self.b > 0 else None,
            "tau_acc": self.tau_acc,
            "tau_num": self.tau_num,
            "d_delta": self.d_delta,
        }

    # JSON round-trip ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdaptiveDifficultyTracker:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


__all__ = [
    "ANCHOR_TABLES",
    "AdaptiveDifficultyTracker",
    "difficulty_to_kwargs",
    "max_difficulty",
]
