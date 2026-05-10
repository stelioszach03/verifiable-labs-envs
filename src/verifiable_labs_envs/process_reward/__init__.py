"""Process reward model — step segmentation, labeling, training,
eval, and stub serving (Phase 30).

This package builds on the locked Layer 1 conformal moat (Phase 22)
and the Phase 29 distilled outcome reward model to produce a
**per-step** reward sequence + per-step calibrated confidence
intervals + an aggregate trace score for any
``(prompt, reasoning_trace)`` pair.

Public surface (Phase 30.B):

1. :mod:`segmentation` — D14-D hybrid step segmentation
   (newline + `Step N:` + sentence-boundary fallback).
2. :mod:`step_labeling` — D2-D per-step env partial scores via
   procedural decomposition.
3. :mod:`consensus` — per-step ensemble (extends Phase 29 D5-D
   blend to step granularity).
4. :mod:`dataset` — :class:`ProcessRewardTraceRow` shape +
   :func:`collect_env_traces` extraction + JSONL IO.
5. :mod:`frontier_judge` — per-step LLM-as-judge harness for D2-C
   borderline backfill. Gated on ``OPENROUTER_API_KEY`` +
   ``VLABS_PHASE30_COLLECT_FRONTIER=1``.

30.C ships training scaffolding; 30.D ships eval; 30.E ships the
service endpoint surface. 30.F-G arrive when GPU credits resolve
(gate inherited from 29.F per :doc:`PHASE_30_PLAN.md` §19).

The trained student is a Qwen2.5-1.5B-Instruct LoRA fine-tune with
a per-step regression head (D3-A); v0.0.1 ships under D13-C hybrid
(independent serving with shared-backbone scaffold for v0.0.2+).
"""
from __future__ import annotations

from verifiable_labs_envs.process_reward.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    per_step_consensus,
    per_step_disagreement_metrics,
    trace_aggregate_consensus,
)
from verifiable_labs_envs.process_reward.dataset import (
    DEFAULT_HELD_OUT_ENVS,
    DEFAULT_TRAINING_ENVS,
    ProcessRewardTraceRow,
    collect_env_traces,
    read_jsonl,
    write_jsonl,
)
from verifiable_labs_envs.process_reward.frontier_judge import (
    PerStepFrontierResult,
    is_borderline_step,
    sample_per_step_judgments,
)
from verifiable_labs_envs.process_reward.segmentation import (
    DEFAULT_MAX_STEPS,
    SegmentationOutcome,
    segment_trace,
)
from verifiable_labs_envs.process_reward.step_labeling import (
    StepLabelOutcome,
    label_steps,
)

__all__ = [
    "DEFAULT_ENV_WEIGHT",
    "DEFAULT_FRONTIER_WEIGHT",
    "DEFAULT_HELD_OUT_ENVS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TRAINING_ENVS",
    "PerStepFrontierResult",
    "ProcessRewardTraceRow",
    "SegmentationOutcome",
    "StepLabelOutcome",
    "collect_env_traces",
    "is_borderline_step",
    "label_steps",
    "per_step_consensus",
    "per_step_disagreement_metrics",
    "read_jsonl",
    "sample_per_step_judgments",
    "segment_trace",
    "trace_aggregate_consensus",
    "write_jsonl",
]
