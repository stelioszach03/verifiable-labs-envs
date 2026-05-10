"""TRL-compatible reward function wrapper for Verifiable Labs envs.

Bridges the env API (``env.score(prediction, instance) -> dict``) to TRL
0.17's ``GRPOTrainer.reward_funcs`` contract:

    reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]

TRL passes every dataset column except ``prompt`` through ``**kwargs``,
each value being a list with one entry per item in the batch. The
wrapper requires a per-row seed column (default name
``"instance_seed"``) so that the original env instance can be
regenerated and scored against the model's completion.

Three-stage gating per (completion, seed) pair:

1. **parse_valid**: ``extract_json_block(completion)`` succeeds
   (i.e. the completion contains a syntactically valid JSON object).
2. **format_valid**: ``adapter.parse_response(completion, instance)``
   succeeds — keys present, types/lengths/ranges all valid for the env.
3. **score**: ``env.score(prediction, instance)`` returns a reward dict.

If any stage fails the reward is 0.0 and the failure mode is recorded
on :class:`RewardStats`. The reward function never raises on
completion-side errors; only programming errors (missing seed column,
mismatched batch lengths) raise.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("verifiable_labs_envs.training.reward_fn")


# ── reasoning tags (Logic-RL / DeepSeek-R1 chat-template style) ──────


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_tagged_answer(text: str) -> str:
    """Strict reasoning-tags extractor.

    Requires the text to contain BOTH a ``<think>...</think>`` block
    and a ``<answer>...</answer>`` block, with ``<think>`` opening
    strictly before ``<answer>``. Returns the contents of the first
    ``<answer>`` block (whitespace-stripped).

    References
    ----------
    * Logic-RL: Xie et al. 2025, arXiv:2502.14768.
    * DeepSeek-R1 distilled chat template (Jan 2025).

    Raises
    ------
    LLMSolverError
        If either tag is missing, or if ``<think>`` does not precede
        ``<answer>`` in the source text.
    """
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError  # local import

    if not text:
        raise LLMSolverError("empty response text")

    answer_match = _ANSWER_RE.search(text)
    if not answer_match:
        raise LLMSolverError("missing <answer>...</answer> block")

    think_match = _THINK_RE.search(text)
    if not think_match:
        raise LLMSolverError("missing <think>...</think> block")

    if think_match.start() >= answer_match.start():
        raise LLMSolverError(
            "<think>...</think> must appear before <answer>...</answer>"
        )

    return answer_match.group(1).strip()


def parse_with_tags(text: str, instance: Any, adapter: Any) -> Any:
    """Parse a tagged completion via the env's adapter.

    Extracts the contents of ``<answer>...</answer>`` (after asserting
    ``<think>...</think>`` precedes it), then delegates JSON / schema
    validation to ``adapter.parse_response``. This gives us the strict
    reasoning-tags format gate from Logic-RL while reusing every
    existing per-env validation rule (k indices, range, deduplication,
    int/float coercion, etc.).
    """
    answer_text = extract_tagged_answer(text)
    return adapter.parse_response(answer_text, instance)


@dataclasses.dataclass
class RewardStats:
    """Running totals plus per-call records for downstream analysis.

    ``per_call`` is appended to in batch order; M7 paired comparison
    consumes it directly to compute per-instance deltas.
    """

    n_calls: int = 0
    n_parse_valid: int = 0
    n_format_valid: int = 0
    n_score_exceptions: int = 0
    sum_reward: float = 0.0
    sum_nmse: float = 0.0
    sum_support: float = 0.0
    sum_conformal: float = 0.0
    per_call: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def aggregate(self) -> dict[str, float]:
        """Aggregate metrics derived from running totals."""
        n = max(self.n_calls, 1)
        n_fmt = max(self.n_format_valid, 1)
        return {
            "n_calls": float(self.n_calls),
            "mean_reward": self.sum_reward / n,
            "parse_valid_rate": self.n_parse_valid / n,
            "format_valid_rate": self.n_format_valid / n,
            "score_exception_rate": self.n_score_exceptions / n,
            "mean_nmse": self.sum_nmse / n_fmt,
            "mean_support": self.sum_support / n_fmt,
            "mean_conformal": self.sum_conformal / n_fmt,
        }

    def reset(self) -> None:
        """Zero out totals and clear per-call history."""
        self.n_calls = 0
        self.n_parse_valid = 0
        self.n_format_valid = 0
        self.n_score_exceptions = 0
        self.sum_reward = 0.0
        self.sum_nmse = 0.0
        self.sum_support = 0.0
        self.sum_conformal = 0.0
        self.per_call.clear()


def _zero_components() -> dict[str, float]:
    return {
        "nmse": 0.0,
        "support": 0.0,
        "conformal": 0.0,
        "parse_valid": 0.0,
        "format_valid": 0.0,
    }


def make_reward_fn(
    env_id: str,
    *,
    seed_kwarg: str = "instance_seed",
    env_kwargs: dict[str, Any] | None = None,
    calibration_quantile: float | None = 2.0,
    use_tags: bool = False,
) -> Callable[..., list[float]]:
    """Construct a TRL-compatible reward function for the named env.

    Parameters
    ----------
    env_id : str
        Environment id from the verifiable-labs-envs registry, e.g.
        ``"sparse-fourier-recovery"``.
    seed_kwarg : str, optional
        Name of the per-row dataset column carrying the integer seed
        used to regenerate the env instance. Defaults to
        ``"instance_seed"``.
    env_kwargs : dict[str, Any] | None, optional
        Extra kwargs forwarded to ``load_environment(env_id, ...)``.
    calibration_quantile : float | None, optional
        Default conformal calibration shortcut for sparse-fourier-style
        envs (matches the CLI default of ``2.0``). Set to ``None`` to
        omit. Image envs that don't accept this kwarg fall back
        gracefully via the ``TypeError`` retry pattern used in the CLI.
    use_tags : bool, optional, default ``False``
        When True, the format-validation step requires reasoning tags
        (``<think>...</think>`` followed by ``<answer>...</answer>``).
        JSON is parsed from inside the answer tag only. When False
        (default — backward-compatible with M2/M5/M6 fixtures), the
        adapter's existing parser is used directly on the raw
        completion. References: Logic-RL (arXiv:2502.14768),
        DeepSeek-R1 distilled chat template.

    Returns
    -------
    reward_fn : Callable
        Signature ``reward_fn(prompts: list[str], completions: list[str],
        **kwargs) -> list[float]``. Conforms to TRL 0.17's
        ``GRPOTrainer.reward_funcs`` contract. Robust to malformed
        completions (returns 0.0, never raises). Per-call statistics
        accumulate on ``reward_fn.stats`` (a :class:`RewardStats`).

    Notes
    -----
    The returned callable carries four attributes for inspection:
    ``stats`` (RewardStats), ``env_id`` (str), ``seed_kwarg`` (str),
    and ``use_tags`` (bool).
    """
    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.solvers.adapters._common import extract_json_block
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError, get_adapter

    extra = dict(env_kwargs or {})
    if calibration_quantile is not None and "calibration_quantile" not in extra:
        extra["calibration_quantile"] = calibration_quantile
    try:
        env = load_environment(env_id, **extra)
    except TypeError:
        extra.pop("calibration_quantile", None)
        env = load_environment(env_id, **extra)

    adapter = get_adapter(env_id)
    stats = RewardStats()

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        seeds = kwargs.get(seed_kwarg)
        if seeds is None:
            raise ValueError(
                f"reward_fn requires dataset column {seed_kwarg!r}; "
                f"got kwargs keys={sorted(kwargs)}"
            )
        if len(seeds) != len(completions):
            raise ValueError(
                f"len({seed_kwarg})={len(seeds)} != "
                f"len(completions)={len(completions)}"
            )

        rewards: list[float] = []
        for completion, seed in zip(completions, seeds, strict=True):
            stats.n_calls += 1
            seed_int = int(seed)
            record: dict[str, Any] = {
                "seed": seed_int,
                "reward": 0.0,
                "components": _zero_components(),
                "failure_type": None,
            }

            text = completion if isinstance(completion, str) else ""

            # Stage 1 — parse_valid: is the completion a JSON object at all?
            try:
                extract_json_block(text)
            except LLMSolverError:
                record["failure_type"] = "parse_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            record["components"]["parse_valid"] = 1.0
            stats.n_parse_valid += 1

            # Stage 2 — format_valid: does adapter.parse_response accept it?
            try:
                instance = env.generate_instance(seed=seed_int)
            except Exception as e:  # noqa: BLE001 — env-loading boundary
                logger.warning(
                    "generate_instance failed for seed=%d: %s",
                    seed_int,
                    str(e)[:200],
                )
                stats.n_score_exceptions += 1
                record["failure_type"] = "instance_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue

            try:
                if use_tags:
                    prediction = parse_with_tags(text, instance, adapter)
                else:
                    prediction = adapter.parse_response(text, instance)
            except LLMSolverError:
                record["failure_type"] = "format_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            except Exception as e:  # noqa: BLE001 — adapter boundary
                logger.warning(
                    "adapter.parse_response unexpected exception for seed=%d: %s",
                    seed_int,
                    str(e)[:200],
                )
                stats.n_score_exceptions += 1
                record["failure_type"] = "adapter_exception"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            record["components"]["format_valid"] = 1.0
            stats.n_format_valid += 1

            # Stage 3 — env.score
            try:
                score = env.score(prediction, instance)
            except Exception as e:  # noqa: BLE001 — scoring boundary
                logger.warning(
                    "env.score failed for seed=%d: %s",
                    seed_int,
                    str(e)[:200],
                )
                stats.n_score_exceptions += 1
                record["failure_type"] = "score_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue

            reward = float(score.get("reward", 0.0))
            env_components = score.get("components", {}) or {}
            nmse = float(env_components.get("nmse", 0.0))
            support = float(env_components.get("support", 0.0))
            conformal = float(env_components.get("conformal", 0.0))

            record["reward"] = reward
            record["components"]["nmse"] = nmse
            record["components"]["support"] = support
            record["components"]["conformal"] = conformal

            stats.sum_reward += reward
            stats.sum_nmse += nmse
            stats.sum_support += support
            stats.sum_conformal += conformal

            rewards.append(reward)
            stats.per_call.append(record)

        return rewards

    reward_fn.stats = stats  # type: ignore[attr-defined]
    reward_fn.env_id = env_id  # type: ignore[attr-defined]
    reward_fn.seed_kwarg = seed_kwarg  # type: ignore[attr-defined]
    reward_fn.use_tags = use_tags  # type: ignore[attr-defined]
    return reward_fn


# ── posterior reward gating (P-GRPO; Fan et al. 2025) ──────────────────


# Outcome-pass thresholds for sparse-fourier-recovery (raw NMSE = squared
# reconstruction error normalised by signal energy; support score is F1 on
# the recovered top-k indices).
DEFAULT_OUTCOME_NMSE_MAX = 0.10
DEFAULT_OUTCOME_SUPPORT_MIN = 0.5

# Quality blend (used only when the outcome gate passes).
QUALITY_NMSE_WEIGHT = 0.5
QUALITY_CONFORMAL_WEIGHT = 0.5

# Normalisation: R_format ∈ {0,1} + R_outcome ∈ {0,1} + R_outcome × R_quality
# is bounded by 0+1+1 = 3, so we divide by 3 to get [0, 1].
POSTERIOR_NORM = 3.0


def posterior_reward(
    *,
    parse_valid: float,
    format_valid: float,
    nmse_raw: float,
    support: float,
    conformal: float,
    outcome_nmse_max: float = DEFAULT_OUTCOME_NMSE_MAX,
    outcome_support_min: float = DEFAULT_OUTCOME_SUPPORT_MIN,
) -> tuple[float, dict[str, float]]:
    """Compute a P-GRPO-style posterior-gated reward.

    Formula (Fan et al. 2025, arXiv:2508.05170, Eq. 1; R_format gates all
    downstream terms so a malformed answer earns zero):

        R = R_format · ( 1 + R_outcome + R_outcome · R_quality ) / 3

    Where:

    * ``R_format`` ∈ {0, 1}: ``parse_valid`` AND ``format_valid``.
    * ``R_outcome`` ∈ {0, 1}: ``support >= outcome_support_min`` AND
      ``nmse_raw <= outcome_nmse_max``.
    * ``R_quality`` ∈ [0, 1]: blended reconstruction × calibration —
      ``(1 - min(nmse_raw, 1)) · 0.5 + conformal · 0.5``.

    Two posterior-gating properties:

    1. ``R_format`` gates everything — a malformed completion earns 0,
       even if the underlying numbers happen to be perfect.
    2. ``R_quality`` only contributes when ``R_outcome = 1`` — a
       well-formed but structurally wrong answer earns *no* quality
       credit even if its nmse / conformal numbers look good. This
       prevents reward hacking via the conformal floor that emerged
       in M7 (where 60% format compliance translated into a
       ~+0.10 free conformal reward without any structural learning).

    Returns
    -------
    (reward, components) : tuple
        ``reward`` is the normalised scalar in [0, 1]. ``components``
        is a dict with keys: ``r_format``, ``r_outcome``, ``r_quality``,
        ``r_unnormalised``, ``r_normalised``, plus the input fields
        ``nmse_raw``, ``support``, ``conformal`` for trace fidelity.
    """
    r_format = 1.0 if (float(parse_valid) >= 0.5 and float(format_valid) >= 0.5) else 0.0
    outcome_pass = (
        float(support) >= outcome_support_min
        and float(nmse_raw) <= outcome_nmse_max
    )
    r_outcome = 1.0 if outcome_pass else 0.0
    nmse_clipped = min(max(float(nmse_raw), 0.0), 1.0)
    r_quality = (1.0 - nmse_clipped) * QUALITY_NMSE_WEIGHT + float(conformal) * QUALITY_CONFORMAL_WEIGHT
    r_quality = float(max(0.0, min(1.0, r_quality)))

    # Posterior gate: r_format gates everything; r_outcome gates r_quality.
    r_unnorm = r_format * (1.0 + r_outcome + r_outcome * r_quality)
    r_norm = r_unnorm / POSTERIOR_NORM

    components = {
        "r_format": r_format,
        "r_outcome": r_outcome,
        "r_quality": r_quality,
        "r_unnormalised": r_unnorm,
        "r_normalised": r_norm,
        "nmse_raw": float(nmse_raw),
        "support": float(support),
        "conformal": float(conformal),
    }
    return r_norm, components


@dataclasses.dataclass
class PosteriorRewardStats:
    """Running totals + per-call records for the posterior reward function."""

    n_calls: int = 0
    n_parse_valid: int = 0
    n_format_valid: int = 0
    n_outcome_correct: int = 0
    n_score_exceptions: int = 0
    sum_reward: float = 0.0
    sum_quality_when_outcome: float = 0.0
    per_call: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def aggregate(self) -> dict[str, float]:
        n = max(self.n_calls, 1)
        n_oc = max(self.n_outcome_correct, 1)
        return {
            "n_calls": float(self.n_calls),
            "mean_reward": self.sum_reward / n,
            "parse_valid_rate": self.n_parse_valid / n,
            "format_valid_rate": self.n_format_valid / n,
            "outcome_correct_rate": self.n_outcome_correct / n,
            "score_exception_rate": self.n_score_exceptions / n,
            "mean_quality_when_outcome": self.sum_quality_when_outcome / n_oc,
        }

    def reset(self) -> None:
        self.n_calls = 0
        self.n_parse_valid = 0
        self.n_format_valid = 0
        self.n_outcome_correct = 0
        self.n_score_exceptions = 0
        self.sum_reward = 0.0
        self.sum_quality_when_outcome = 0.0
        self.per_call.clear()


def make_reward_fn_posterior(
    env_id: str,
    *,
    seed_kwarg: str = "instance_seed",
    env_kwargs: dict[str, Any] | None = None,
    calibration_quantile: float | None = 2.0,
    outcome_nmse_max: float = DEFAULT_OUTCOME_NMSE_MAX,
    outcome_support_min: float = DEFAULT_OUTCOME_SUPPORT_MIN,
    use_tags: bool = True,
) -> Callable[..., list[float]]:
    """TRL-compatible reward function with P-GRPO posterior gating.

    Same TRL signature as :func:`make_reward_fn` and same three-stage
    failure handling. Differs in the reward computation: scores are
    composed via :func:`posterior_reward` so that ``R_quality`` only
    contributes when ``R_outcome = 1``.

    Parameters
    ----------
    env_id, seed_kwarg, env_kwargs, calibration_quantile : see
        :func:`make_reward_fn` for the same semantics.
    outcome_nmse_max, outcome_support_min : tuneable thresholds for
        the binary outcome gate. Defaults match the values surfaced in
        the M7 writeup as the realistic post-format-learning targets.
    use_tags : bool, optional, default ``True``
        When True (Phase 14 default), format validation requires
        ``<think>...</think>`` followed by ``<answer>...</answer>``
        with valid JSON inside the answer tag. When False, falls back
        to the adapter's existing parser (no tags). The Phase 14
        training run is intended to teach explicit reasoning, so
        tags are on by default in this constructor.
    """
    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.solvers.adapters._common import extract_json_block
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError, get_adapter

    extra = dict(env_kwargs or {})
    if calibration_quantile is not None and "calibration_quantile" not in extra:
        extra["calibration_quantile"] = calibration_quantile
    try:
        env = load_environment(env_id, **extra)
    except TypeError:
        extra.pop("calibration_quantile", None)
        env = load_environment(env_id, **extra)

    adapter = get_adapter(env_id)
    stats = PosteriorRewardStats()

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        seeds = kwargs.get(seed_kwarg)
        if seeds is None:
            raise ValueError(
                f"reward_fn requires dataset column {seed_kwarg!r}; "
                f"got kwargs keys={sorted(kwargs)}"
            )
        if len(seeds) != len(completions):
            raise ValueError(
                f"len({seed_kwarg})={len(seeds)} != "
                f"len(completions)={len(completions)}"
            )

        rewards: list[float] = []
        for completion, seed in zip(completions, seeds, strict=True):
            stats.n_calls += 1
            seed_int = int(seed)
            text = completion if isinstance(completion, str) else ""

            record: dict[str, Any] = {
                "seed": seed_int,
                "reward": 0.0,
                "components": {
                    "r_format": 0.0,
                    "r_outcome": 0.0,
                    "r_quality": 0.0,
                    "r_unnormalised": 0.0,
                    "r_normalised": 0.0,
                    "nmse_raw": 0.0,
                    "support": 0.0,
                    "conformal": 0.0,
                    "parse_valid": 0.0,
                    "format_valid": 0.0,
                },
                "failure_type": None,
            }

            # Stage 1: parse_valid (well-formed JSON object).
            try:
                extract_json_block(text)
            except LLMSolverError:
                record["failure_type"] = "parse_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            record["components"]["parse_valid"] = 1.0
            stats.n_parse_valid += 1

            # Stage 2: format_valid via env adapter.
            try:
                instance = env.generate_instance(seed=seed_int)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "generate_instance failed for seed=%d: %s",
                    seed_int, str(e)[:200],
                )
                stats.n_score_exceptions += 1
                record["failure_type"] = "instance_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue

            try:
                if use_tags:
                    prediction = parse_with_tags(text, instance, adapter)
                else:
                    prediction = adapter.parse_response(text, instance)
            except LLMSolverError:
                record["failure_type"] = "format_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "adapter.parse_response unexpected exception for seed=%d: %s",
                    seed_int, str(e)[:200],
                )
                stats.n_score_exceptions += 1
                record["failure_type"] = "adapter_exception"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            record["components"]["format_valid"] = 1.0
            stats.n_format_valid += 1

            # Stage 3: env scoring → pull raw NMSE from meta for the gate.
            try:
                score = env.score(prediction, instance)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "env.score failed for seed=%d: %s", seed_int, str(e)[:200],
                )
                stats.n_score_exceptions += 1
                record["failure_type"] = "score_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue

            env_components = score.get("components", {}) or {}
            env_meta = score.get("meta", {}) or {}
            support = float(env_components.get("support", 0.0))
            conformal = float(env_components.get("conformal", 0.0))
            # nmse_raw is the unsquashed reconstruction error (∈ [0, ∞));
            # the env's components.nmse is exp(-nmse_raw / NMSE_TAU) so
            # we MUST use meta.nmse_raw for the gate threshold.
            nmse_raw = float(env_meta.get("nmse_raw", 0.0))

            r, comp = posterior_reward(
                parse_valid=1.0,
                format_valid=1.0,
                nmse_raw=nmse_raw,
                support=support,
                conformal=conformal,
                outcome_nmse_max=outcome_nmse_max,
                outcome_support_min=outcome_support_min,
            )
            record["reward"] = r
            record["components"].update(comp)
            stats.sum_reward += r
            if comp["r_outcome"] >= 0.5:
                stats.n_outcome_correct += 1
                stats.sum_quality_when_outcome += comp["r_quality"]
            rewards.append(r)
            stats.per_call.append(record)

        return rewards

    reward_fn.stats = stats  # type: ignore[attr-defined]
    reward_fn.env_id = env_id  # type: ignore[attr-defined]
    reward_fn.seed_kwarg = seed_kwarg  # type: ignore[attr-defined]
    reward_fn.use_tags = use_tags  # type: ignore[attr-defined]
    reward_fn.outcome_nmse_max = outcome_nmse_max  # type: ignore[attr-defined]
    reward_fn.outcome_support_min = outcome_support_min  # type: ignore[attr-defined]
    return reward_fn


# ── posterior reward (image-recovery family: super-resolution / mri-knee) ──


DEFAULT_OUTCOME_PSNR_DB_MIN = 22.0
DEFAULT_OUTCOME_SSIM_MIN = 0.70
DEFAULT_PSNR_DB_NORMALISER = 30.0  # PSNR ≥ 30 dB → quality contribution caps at 0.5


def posterior_reward_image(
    *,
    parse_valid: float,
    format_valid: float,
    psnr_db: float,
    ssim: float,
    conformal: float,
    outcome_psnr_db_min: float = DEFAULT_OUTCOME_PSNR_DB_MIN,
    outcome_ssim_min: float = DEFAULT_OUTCOME_SSIM_MIN,
    psnr_db_normaliser: float = DEFAULT_PSNR_DB_NORMALISER,
) -> tuple[float, dict[str, float]]:
    """Image-family posterior gating (super-resolution / mri-knee).

    Same structural shape as :func:`posterior_reward` but with PSNR/SSIM
    in place of nmse_raw/support:

        R_outcome = (psnr_db ≥ psnr_db_min) AND (ssim ≥ ssim_min)
        R_quality = clip(psnr_db / psnr_db_normaliser, 0, 1) · 0.5
                    + clip(ssim, 0, 1) · 0.5
        R         = R_format · (1 + R_outcome + R_outcome · R_quality) / 3
    """
    r_format = 1.0 if (float(parse_valid) >= 0.5 and float(format_valid) >= 0.5) else 0.0
    psnr = float(psnr_db)
    ssim_v = float(ssim)
    outcome_pass = (psnr >= outcome_psnr_db_min) and (ssim_v >= outcome_ssim_min)
    r_outcome = 1.0 if outcome_pass else 0.0
    psnr_norm = max(0.0, min(1.0, psnr / float(psnr_db_normaliser)))
    ssim_clipped = max(0.0, min(1.0, ssim_v))
    r_quality = psnr_norm * 0.5 + ssim_clipped * 0.5
    r_unnorm = r_format * (1.0 + r_outcome + r_outcome * r_quality)
    r_norm = r_unnorm / POSTERIOR_NORM
    return r_norm, {
        "r_format": r_format,
        "r_outcome": r_outcome,
        "r_quality": r_quality,
        "r_unnormalised": r_unnorm,
        "r_normalised": r_norm,
        "psnr_db": psnr,
        "ssim": ssim_clipped,
        "conformal": float(conformal),
    }


# ── Per-env posterior thresholds + schema expectations ────────────────


#: Registry keyed by env_id; consumed by :func:`make_reward_fn_multienv`
#: to dispatch to the right posterior formula. Each entry declares:
#:   * ``kind`` — "sparse" or "image"
#:   * ``expected_components`` — keys we expect ``score()['components']``
#:     to contain (used by pre-flight schema check)
#:   * ``expected_meta`` — keys we expect in ``score()['meta']`` (e.g.
#:     ``"nmse_raw"`` for sparse, ``"psnr_db"`` for image)
#:   * threshold fields specific to ``kind``
OUTCOME_THRESHOLDS_REGISTRY: dict[str, dict[str, Any]] = {
    "sparse-fourier-recovery": {
        "kind": "sparse",
        "expected_components": ("nmse", "support", "conformal"),
        "expected_meta": ("nmse_raw",),
        "support_min": 0.5,
        "nmse_max": 0.10,
    },
    "phase-retrieval": {
        "kind": "sparse",
        "expected_components": ("nmse", "support", "conformal"),
        "expected_meta": ("nmse_raw",),
        "support_min": 0.5,
        "nmse_max": 0.10,
    },
    "super-resolution-div2k-x4": {
        "kind": "image",
        "expected_components": ("psnr", "ssim", "conformal"),
        "expected_meta": ("psnr_db",),
        "psnr_db_min": 22.0,
        "ssim_min": 0.70,
    },
    "mri-knee-reconstruction": {
        "kind": "image",
        "expected_components": ("psnr", "ssim", "conformal"),
        "expected_meta": ("psnr_db",),
        "psnr_db_min": 25.0,
        "ssim_min": 0.70,
    },
}


def validate_env_schema(env_id: str, score_dict: dict[str, Any]) -> list[str]:
    """Check ``env.score(...)`` output against the registry.

    Returns a list of human-readable error strings (empty list = OK).
    """
    cfg = OUTCOME_THRESHOLDS_REGISTRY.get(env_id)
    if cfg is None:
        return [f"env_id={env_id!r} not in OUTCOME_THRESHOLDS_REGISTRY"]
    errs: list[str] = []
    components = score_dict.get("components") or {}
    meta = score_dict.get("meta") or {}
    for k in cfg["expected_components"]:
        if k not in components:
            errs.append(f"score()['components'] missing key {k!r}")
    for k in cfg["expected_meta"]:
        if k not in meta:
            errs.append(f"score()['meta'] missing key {k!r}")
    return errs


# ── Multi-env reward function (P-GRPO + dispatch by env_id) ───────────


def make_reward_fn_multienv(
    env_ids: list[str],
    *,
    seed_kwarg: str = "instance_seed",
    env_id_kwarg: str = "env_id",
    use_tags: bool = True,
    calibration_quantile: float | None = 2.0,
) -> Callable[..., list[float]]:
    """Build a TRL-compatible reward function that dispatches per-env.

    Each dataset row must include ``env_id`` and ``instance_seed``. The
    reward function regenerates the env-specific instance, parses the
    completion via the env's adapter (with reasoning tags if
    ``use_tags=True``), scores via ``env.score(...)``, then routes the
    result through the appropriate posterior formula:

    * "sparse" envs (sparse-fourier-recovery, phase-retrieval) →
      :func:`posterior_reward`
    * "image" envs (super-resolution-div2k-x4, mri-knee-reconstruction)
      → :func:`posterior_reward_image`

    Per-env thresholds come from :data:`OUTCOME_THRESHOLDS_REGISTRY`.
    Aggregate stats (``reward_fn.stats``) tracks per-env counters.

    Raises
    ------
    ValueError
        If any ``env_id`` in ``env_ids`` lacks an
        :data:`OUTCOME_THRESHOLDS_REGISTRY` entry — the multi-env
        reward function refuses to silently fall back to a default.
    """
    from verifiable_labs_envs import load_environment
    from verifiable_labs_envs.solvers.adapters._common import extract_json_block
    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError, get_adapter
    from verifiable_labs_envs.training.adaptive_difficulty import difficulty_to_kwargs

    # Validate each env has a registry entry (no silent fallbacks).
    for eid in env_ids:
        if eid not in OUTCOME_THRESHOLDS_REGISTRY:
            raise ValueError(
                f"env_id={eid!r} has no OUTCOME_THRESHOLDS_REGISTRY entry; "
                f"add one before calling make_reward_fn_multienv"
            )

    # Load envs + adapters once.
    envs: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    for eid in env_ids:
        try:
            envs[eid] = load_environment(eid, calibration_quantile=calibration_quantile)
        except TypeError:
            envs[eid] = load_environment(eid)
        adapters[eid] = get_adapter(eid)

    stats = PosteriorRewardStats()
    per_env_counts: dict[str, int] = {eid: 0 for eid in env_ids}

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        seeds = kwargs.get(seed_kwarg)
        env_id_list = kwargs.get(env_id_kwarg)
        # Difficulty column is optional; if absent, we use the env's defaults.
        # When present (multi-env training with adaptive difficulty), we MUST
        # use it so the reward fn's instance matches the dataset's instance.
        difficulties = kwargs.get("difficulty")
        if seeds is None or env_id_list is None:
            raise ValueError(
                f"reward_fn requires kwargs {seed_kwarg!r} and {env_id_kwarg!r}; "
                f"got keys={sorted(kwargs)}"
            )
        if not (len(seeds) == len(env_id_list) == len(completions)):
            raise ValueError("seeds, env_ids, completions length mismatch")
        if difficulties is not None and len(difficulties) != len(completions):
            raise ValueError(
                "difficulty length must match completions when provided"
            )

        rewards: list[float] = []
        for i, (completion, seed, eid) in enumerate(
            zip(completions, seeds, env_id_list, strict=True)
        ):
            stats.n_calls += 1
            per_env_counts[eid] = per_env_counts.get(eid, 0) + 1
            seed_int = int(seed)
            text = completion if isinstance(completion, str) else ""
            cfg = OUTCOME_THRESHOLDS_REGISTRY[eid]
            env = envs[eid]
            adapter = adapters[eid]
            diff = int(difficulties[i]) if difficulties is not None else None

            record: dict[str, Any] = {
                "seed": seed_int,
                "env_id": eid,
                "difficulty": diff,
                "reward": 0.0,
                "components": {"r_format": 0.0, "r_outcome": 0.0, "r_quality": 0.0,
                               "parse_valid": 0.0, "format_valid": 0.0},
                "failure_type": None,
            }

            # Stage 1: parse_valid.
            try:
                extract_json_block(text)
            except LLMSolverError:
                record["failure_type"] = "parse_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            record["components"]["parse_valid"] = 1.0
            stats.n_parse_valid += 1

            # Stage 2: instance + format_valid. Crucial: pass the per-row
            # difficulty kwargs so the reward-side instance matches the
            # dataset-side prompt (built via the same kwargs in
            # build_multienv_dataset). Otherwise n/k/shape mismatches make
            # every parse fail length validation.
            diff_kwargs = (
                difficulty_to_kwargs(eid, diff) if diff is not None else {}
            )
            try:
                instance = env.generate_instance(seed=seed_int, **diff_kwargs)
            except Exception as e:  # noqa: BLE001
                logger.warning("multienv generate_instance failed env=%s seed=%d: %s",
                               eid, seed_int, str(e)[:200])
                stats.n_score_exceptions += 1
                record["failure_type"] = "instance_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue

            try:
                if use_tags:
                    prediction = parse_with_tags(text, instance, adapter)
                else:
                    prediction = adapter.parse_response(text, instance)
            except LLMSolverError:
                record["failure_type"] = "format_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning("multienv parse_response failed env=%s seed=%d: %s",
                               eid, seed_int, str(e)[:200])
                stats.n_score_exceptions += 1
                record["failure_type"] = "adapter_exception"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue
            record["components"]["format_valid"] = 1.0
            stats.n_format_valid += 1

            # Stage 3: env scoring.
            try:
                score = env.score(prediction, instance)
            except Exception as e:  # noqa: BLE001
                logger.warning("multienv env.score failed env=%s seed=%d: %s",
                               eid, seed_int, str(e)[:200])
                stats.n_score_exceptions += 1
                record["failure_type"] = "score_error"
                stats.per_call.append(record)
                rewards.append(0.0)
                continue

            comps = score.get("components", {}) or {}
            meta = score.get("meta", {}) or {}

            # Dispatch on env-family kind.
            if cfg["kind"] == "sparse":
                r, comp = posterior_reward(
                    parse_valid=1.0, format_valid=1.0,
                    nmse_raw=float(meta.get("nmse_raw", 0.0)),
                    support=float(comps.get("support", 0.0)),
                    conformal=float(comps.get("conformal", 0.0)),
                    outcome_nmse_max=cfg["nmse_max"],
                    outcome_support_min=cfg["support_min"],
                )
            elif cfg["kind"] == "image":
                r, comp = posterior_reward_image(
                    parse_valid=1.0, format_valid=1.0,
                    psnr_db=float(meta.get("psnr_db", 0.0)),
                    ssim=float(comps.get("ssim", 0.0)),
                    conformal=float(comps.get("conformal", 0.0)),
                    outcome_psnr_db_min=cfg["psnr_db_min"],
                    outcome_ssim_min=cfg["ssim_min"],
                )
            else:
                raise ValueError(f"unknown posterior kind {cfg['kind']!r}")

            record["reward"] = r
            record["components"].update(comp)
            stats.sum_reward += r
            if comp["r_outcome"] >= 0.5:
                stats.n_outcome_correct += 1
                stats.sum_quality_when_outcome += comp["r_quality"]
            rewards.append(r)
            stats.per_call.append(record)

        return rewards

    reward_fn.stats = stats  # type: ignore[attr-defined]
    reward_fn.env_ids = list(env_ids)  # type: ignore[attr-defined]
    reward_fn.seed_kwarg = seed_kwarg  # type: ignore[attr-defined]
    reward_fn.env_id_kwarg = env_id_kwarg  # type: ignore[attr-defined]
    reward_fn.use_tags = use_tags  # type: ignore[attr-defined]
    reward_fn.per_env_counts = per_env_counts  # type: ignore[attr-defined]
    return reward_fn
