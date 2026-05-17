"""Tests for ``vlabs_reward_train.trainer`` (Phase 29.F unlock)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlabs_reward_train.trainer import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BETA,
    DEFAULT_ENV_ID,
    DEFAULT_LR,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_MAX_MODEL_LENGTH,
    DEFAULT_VLLM_MODE,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    REQUIRED_DEPS,
    DependencyStatus,
    GpuPathNotImplemented,
    TrainingConfig,
    _load_grpo_prompts_dataset,
    build_grpo_trainer,
    build_training_args,
    validate_dependencies,
    write_run_card,
)


# ─────────────────────────── helpers ──────────────────────────────


class _FakeTokenizer:
    """Stand-in for HF AutoTokenizer that mimics
    ``apply_chat_template(messages, tokenize=False, add_generation_prompt=True)``
    with a predictable, inspectable output. Used to keep
    ``_load_grpo_prompts_dataset`` tests fast and offline (no model
    download, no real tokenizer init)."""

    name_or_path = "fake-tokenizer"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert tokenize is False, "tests rely on tokenize=False"
        parts: list[str] = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


# ───────────────────── locked defaults / config ───────────────────


def test_default_base_model_locked_per_plan() -> None:
    """Plan §5 D2-A: Qwen2.5-1.5B-Instruct."""
    assert DEFAULT_BASE_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert pytest.approx(2e-4) == DEFAULT_LR


def test_required_deps_includes_torch_trl_peft() -> None:
    assert "torch" in REQUIRED_DEPS
    assert "trl" in REQUIRED_DEPS
    assert "peft" in REQUIRED_DEPS
    assert "transformers" in REQUIRED_DEPS


def test_training_config_round_trip() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    payload = cfg.to_dict()
    restored = TrainingConfig.from_dict(payload)
    assert restored == cfg


def test_training_config_with_overrides_creates_new() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    new = cfg.with_overrides(lr=1e-5, epochs=10)
    assert new.lr == pytest.approx(1e-5)
    assert new.epochs == 10
    assert cfg.lr != new.lr  # original untouched


def test_training_config_lora_spec_pulls_through() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", lora_r=8, lora_alpha=16)
    spec = cfg.lora_spec
    assert spec.r == 8
    assert spec.alpha == 16


def test_training_config_vllm_defaults_present() -> None:
    """29.F: TrainingConfig carries first-class vLLM + env_id fields."""
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    assert cfg.vllm_mode == DEFAULT_VLLM_MODE == "colocate"
    assert cfg.vllm_gpu_memory_utilization == pytest.approx(
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    )
    assert cfg.vllm_tensor_parallel_size == DEFAULT_VLLM_TENSOR_PARALLEL_SIZE == 1
    assert cfg.vllm_max_model_length == DEFAULT_VLLM_MAX_MODEL_LENGTH == 3072
    assert cfg.num_generations == DEFAULT_NUM_GENERATIONS
    assert cfg.env_id == DEFAULT_ENV_ID == "sparse-fourier-recovery"
    assert cfg.max_steps == DEFAULT_MAX_STEPS == -1
    assert cfg.beta == pytest.approx(DEFAULT_BETA)


def test_training_config_vllm_overrides() -> None:
    cfg = TrainingConfig(
        dataset_path="/tmp/x.jsonl",
        vllm_gpu_memory_utilization=0.45,
        vllm_max_model_length=2048,
        env_id="math-algebra",
        max_steps=10,
    )
    assert cfg.vllm_gpu_memory_utilization == pytest.approx(0.45)
    assert cfg.vllm_max_model_length == 2048
    assert cfg.env_id == "math-algebra"
    assert cfg.max_steps == 10


# ───────────────────── validate_dependencies ──────────────────────


def test_validate_dependencies_returns_status() -> None:
    status = validate_dependencies()
    assert isinstance(status, DependencyStatus)
    # In CI without GPU deps, missing is non-empty.
    assert isinstance(status.missing, tuple)
    assert isinstance(status.available, tuple)
    assert status.is_satisfied == (not status.missing)


def test_validate_dependencies_with_minimal_required() -> None:
    """Probe with a known-installed dep so we hit the available branch."""
    status = validate_dependencies(required=("json",))
    assert status.is_satisfied
    assert "json" in status.available


def test_validate_dependencies_status_to_dict() -> None:
    status = DependencyStatus(available=("torch",), missing=("trl",))
    assert status.to_dict() == {
        "available": ["torch"],
        "missing": ["trl"],
        "is_satisfied": False,
    }


# ─────────────────────── build_training_args ──────────────────────


def test_build_training_args_basic() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    args = build_training_args(cfg)
    assert args["learning_rate"] == pytest.approx(DEFAULT_LR)
    assert args["num_train_epochs"] == cfg.epochs
    assert args["per_device_train_batch_size"] == cfg.batch_size
    assert args["bf16"] is True
    assert args["report_to"] == ["wandb"]


def test_build_training_args_carries_29f_fields() -> None:
    """29.F: max_steps, num_generations, beta, vllm_max_model_length
    surface as TRL 1.4 GRPOConfig kwargs."""
    cfg = TrainingConfig(
        dataset_path="/tmp/x.jsonl",
        max_steps=10,
        num_generations=4,
        beta=0.04,
    )
    args = build_training_args(cfg)
    assert args["max_steps"] == 10
    assert args["num_generations"] == 4
    assert args["beta"] == pytest.approx(0.04)
    assert args["vllm_max_model_length"] == DEFAULT_VLLM_MAX_MODEL_LENGTH
    # Old kwargs MUST NOT appear:
    assert "kl_coefficient" not in args
    assert "max_prompt_length" not in args


def test_build_training_args_disabled_wandb_drops_report_to() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", wandb_mode="disabled")
    args = build_training_args(cfg)
    assert args["report_to"] == []


def test_build_training_args_rejects_missing_dataset() -> None:
    cfg = TrainingConfig(dataset_path="")
    with pytest.raises(ValueError, match="dataset_path"):
        build_training_args(cfg)


def test_build_training_args_rejects_invalid_lr() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", lr=1.5)
    with pytest.raises(ValueError, match="lr"):
        build_training_args(cfg)


def test_build_training_args_rejects_invalid_epochs() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        build_training_args(cfg)


def test_gpu_path_not_implemented_class_still_exists() -> None:
    """Phase 29.F unlock no longer raises this on a happy path,
    but downstream code that catches the exception keeps compiling."""
    exc = GpuPathNotImplemented("legacy message")
    assert isinstance(exc, RuntimeError)


# ────────────────── build_grpo_trainer / dataset loader ────────────────


def test_build_grpo_trainer_rejects_missing_dataset(tmp_path: Path) -> None:
    """29.F unlock: build_grpo_trainer now does real work; the very
    first failure mode is the dataset not existing — verify it raises
    a clean FileNotFoundError, NOT GpuPathNotImplemented."""
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    cfg = TrainingConfig(
        dataset_path=str(tmp_path / "does-not-exist.jsonl"),
        env_id="sparse-fourier-recovery",
    )
    with pytest.raises(FileNotFoundError, match="dataset JSONL not found"):
        build_grpo_trainer(cfg)


def test_load_grpo_prompts_dataset_missing_file(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    with pytest.raises(FileNotFoundError):
        _load_grpo_prompts_dataset(
            tmp_path / "does-not-exist.jsonl",
            env_id="sparse-fourier-recovery",
            tokenizer=_FakeTokenizer(),
        )


def _write_mini_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


def test_load_grpo_prompts_dataset_filters_by_env_id(tmp_path: Path) -> None:
    """Filter by env_id; rebuild prompts via adapter + chat template
    (NOT pass-through of dataset's raw prompt field)."""
    pytest.importorskip("datasets")
    pytest.importorskip("verifiable_labs_envs.reward_distillation.dataset")

    fp = tmp_path / "mini.jsonl"
    _write_mini_jsonl(
        fp,
        [
            {
                "row_id": "rwd_a",
                "env_id": "sparse-fourier-recovery",
                "prompt": "STALE-PROMPT-A",  # must be ignored
                "completion": "c",
                "env_reward": 0.5,
                "env_components": {},
                "conformal_interval": [0.4, 0.6],
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.5,
                "disagreement": None,
                "source": "env",
                "metadata": {"seed": 42},
            },
            {
                "row_id": "rwd_b",
                "env_id": "math-algebra",  # wrong env: dropped
                "prompt": "STALE-PROMPT-B",
                "completion": "c",
                "env_reward": 0.1,
                "env_components": {},
                "conformal_interval": [0.0, 0.2],
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.1,
                "disagreement": None,
                "source": "env",
                "metadata": {"seed": 7},
            },
            {
                "row_id": "rwd_c",
                "env_id": "sparse-fourier-recovery",
                "prompt": "STALE-PROMPT-C",
                "completion": "c",
                "env_reward": 0.9,
                "env_components": {},
                "conformal_interval": [0.85, 0.95],
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.9,
                "disagreement": None,
                "source": "env",
                "metadata": {"seed": 100},
            },
        ],
    )

    tok = _FakeTokenizer()
    ds = _load_grpo_prompts_dataset(
        fp, env_id="sparse-fourier-recovery", tokenizer=tok
    )
    assert len(ds) == 2  # math-algebra row dropped
    assert set(ds.column_names) == {"prompt", "instance_seed"}
    seeds = sorted(int(s) for s in ds["instance_seed"])
    assert seeds == [42, 100]

    # Every prompt is the chat-templated assembly, NOT the dataset's raw prompt.
    for prompt in ds["prompt"]:
        assert "STALE-PROMPT" not in prompt, (
            "expected the adapter-rebuilt prompt, "
            "not the dataset's stale `prompt` field"
        )
        # _FakeTokenizer wraps in <|im_start|>...<|im_end|> markers.
        assert "<|im_start|>system" in prompt or "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt  # add_generation_prompt=True
        # SparseFourierLLMAdapter.system_prompt mentions "sparse signal recovery".
        assert "sparse signal recovery" in prompt.lower()
        # user-message half includes the OUTPUT SCHEMA description.
        assert "support_idx" in prompt
        assert "support_amp_x1000" in prompt


def test_load_grpo_prompts_dataset_no_matches(tmp_path: Path) -> None:
    """Empty match should raise ValueError, not silently produce
    an empty dataset (GRPO would crash much later with an opaque
    error otherwise)."""
    pytest.importorskip("datasets")
    pytest.importorskip("verifiable_labs_envs.reward_distillation.dataset")

    fp = tmp_path / "mini.jsonl"
    _write_mini_jsonl(
        fp,
        [
            {
                "row_id": "rwd_a",
                "env_id": "math-algebra",
                "prompt": "p",
                "completion": "c",
                "env_reward": 0.5,
                "env_components": {},
                "conformal_interval": [0.4, 0.6],
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.5,
                "disagreement": None,
                "source": "env",
                "metadata": {"seed": 7},
            }
        ],
    )

    with pytest.raises(ValueError, match="no rows with env_id"):
        _load_grpo_prompts_dataset(
            fp, env_id="sparse-fourier-recovery", tokenizer=_FakeTokenizer()
        )


def test_load_grpo_prompts_dataset_dedupes_repeated_seeds(
    tmp_path: Path,
) -> None:
    """If the JSONL contains two rows with the same (env_id, seed),
    the loader keeps only the first to avoid duplicate prompts."""
    pytest.importorskip("datasets")
    pytest.importorskip("verifiable_labs_envs.reward_distillation.dataset")

    fp = tmp_path / "mini.jsonl"
    _write_mini_jsonl(
        fp,
        [
            {
                "row_id": "rwd_a",
                "env_id": "sparse-fourier-recovery",
                "prompt": "p",
                "completion": "c",
                "env_reward": 0.5,
                "env_components": {},
                "conformal_interval": [0.4, 0.6],
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.5,
                "disagreement": None,
                "source": "env",
                "metadata": {"seed": 1},
            },
            {
                "row_id": "rwd_a_dup",
                "env_id": "sparse-fourier-recovery",
                "prompt": "p",
                "completion": "c",
                "env_reward": 0.6,
                "env_components": {},
                "conformal_interval": [0.5, 0.7],
                "frontier_judgment": None,
                "frontier_rationale": None,
                "consensus_reward": 0.6,
                "disagreement": None,
                "source": "judge",  # same seed, different source
                "metadata": {"seed": 1},
            },
        ],
    )

    ds = _load_grpo_prompts_dataset(
        fp, env_id="sparse-fourier-recovery", tokenizer=_FakeTokenizer()
    )
    assert len(ds) == 1
    assert int(ds[0]["instance_seed"]) == 1


# ──────────────────────────── write_run_card ──────────────────────────


def test_write_run_card_persists_payload(tmp_path: Path) -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    status = DependencyStatus(available=("torch",), missing=("trl",))
    target = write_run_card(tmp_path, cfg, status)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["config"]["base_model"] == DEFAULT_BASE_MODEL
    assert payload["dependencies"]["is_satisfied"] is False
    assert payload["schema_version"] == "v0.1.0"
    # 29.F field surface persists round-trip:
    assert "vllm_gpu_memory_utilization" in payload["config"]
    assert "env_id" in payload["config"]
    assert "max_steps" in payload["config"]
    assert "beta" in payload["config"]
