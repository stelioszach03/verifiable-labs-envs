"""Training-time utilities for Verifiable Labs environments.

Currently exposes :func:`make_reward_fn`, a TRL-compatible reward
wrapper that converts environment scoring into the
``Callable[[list, list], list[float]]`` signature expected by
:class:`trl.GRPOTrainer`.
"""
from __future__ import annotations

from verifiable_labs_envs.training.adaptive_difficulty import (
    ANCHOR_TABLES,
    AdaptiveDifficultyTracker,
    difficulty_to_kwargs,
    max_difficulty,
)
from verifiable_labs_envs.training.reward_fn import (
    OUTCOME_THRESHOLDS_REGISTRY,
    PosteriorRewardStats,
    RewardStats,
    extract_tagged_answer,
    make_reward_fn,
    make_reward_fn_multienv,
    make_reward_fn_posterior,
    parse_with_tags,
    posterior_reward,
    posterior_reward_image,
    validate_env_schema,
)

__all__ = [
    "ANCHOR_TABLES",
    "AdaptiveDifficultyTracker",
    "OUTCOME_THRESHOLDS_REGISTRY",
    "PosteriorRewardStats",
    "RewardStats",
    "difficulty_to_kwargs",
    "extract_tagged_answer",
    "make_reward_fn",
    "make_reward_fn_multienv",
    "make_reward_fn_posterior",
    "max_difficulty",
    "parse_with_tags",
    "posterior_reward",
    "posterior_reward_image",
    "validate_env_schema",
]
