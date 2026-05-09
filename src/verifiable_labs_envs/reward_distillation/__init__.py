"""Distilled reward model — dataset, training, eval primitives (Phase 29).

This package builds on the locked Layer 1 conformal moat (Phase 22)
to produce a small student reward model that:

1. Consumes (prompt, completion) inputs.
2. Emits a calibrated scalar reward in ``[0, 1]`` plus a finite-sample
   90 % conformal interval (per D10-A).
3. Serves at $0.001-0.005 per call on a 1.5B-parameter Qwen base
   (per D2-A).

29.B ships dataset construction; 29.C ships training scaffolding;
29.D ships eval; 29.E ships the service endpoint surface (still
backed by stub inference until 29.F-G land trained weights).
"""
from __future__ import annotations

from verifiable_labs_envs.reward_distillation.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    consensus_reward,
    disagreement_metrics,
)
from verifiable_labs_envs.reward_distillation.dataset import (
    DEFAULT_HELD_OUT_ENVS,
    RewardTrainingRow,
    collect_env_rows,
    read_jsonl,
    write_jsonl,
)
from verifiable_labs_envs.reward_distillation.frontier_judge import (
    FrontierJudgeResult,
    is_borderline,
    sample_frontier_judgments,
)
from verifiable_labs_envs.reward_distillation.ultrafeedback import (
    collect_ultrafeedback_subset,
)

__all__ = [
    "DEFAULT_ENV_WEIGHT",
    "DEFAULT_FRONTIER_WEIGHT",
    "DEFAULT_HELD_OUT_ENVS",
    "FrontierJudgeResult",
    "RewardTrainingRow",
    "collect_env_rows",
    "collect_ultrafeedback_subset",
    "consensus_reward",
    "disagreement_metrics",
    "is_borderline",
    "read_jsonl",
    "sample_frontier_judgments",
    "write_jsonl",
]
