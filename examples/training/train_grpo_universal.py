"""Universal GRPO training entry point — works across base-model families.

Wraps :mod:`train_grpo_qwen` (single-env) and :mod:`train_multienv_grpo`
(multi-env) with a ``--base-model-id`` flag and per-model registry. The
default invocation is a *dry-run* validator: tokenizer + chat template +
HF cache check + VRAM budget warning, without ever launching training
or downloading model weights.

Supported base models (Phase D — implementation only)
-----------------------------------------------------
* ``Qwen/Qwen2.5-1.5B-Instruct``      — current default; tested in M6/M7.
* ``meta-llama/Llama-3.2-1B-Instruct`` — gated; needs ``HF_TOKEN``.
* ``microsoft/Phi-3.5-mini-instruct``  — 3.5B; needs A100-80GB or
  ``gradient_checkpointing=True`` on smaller GPUs.
* ``google/gemma-2-2b-it``             — gated; needs ``HF_TOKEN``.

Phase 14 will run actual multi-model training; this script only
exercises the wiring.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Drive HF cache before any HF imports.
DEFAULT_CACHE = "/content/drive/MyDrive/verifiable-labs/hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(DEFAULT_CACHE) / "hub"))


# ── Model registry ─────────────────────────────────────────────────────


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "size_b": 1.5,
        "vram_peak_gb": 15,
        "needs_hf_token": False,
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "chat_template_family": "chatml",
        "notes": "Default; tested in M6/M7.",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "size_b": 1.0,
        "vram_peak_gb": 12,
        "needs_hf_token": True,
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "chat_template_family": "llama3",
        "notes": "Smallest base. Gated — set HF_TOKEN env var.",
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "size_b": 3.5,
        "vram_peak_gb": 35,
        "needs_hf_token": False,
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "chat_template_family": "phi3",
        "notes": "Largest. Use gradient_checkpointing on <80GB GPUs.",
    },
    "google/gemma-2-2b-it": {
        "size_b": 2.0,
        "vram_peak_gb": 20,
        "needs_hf_token": True,
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "chat_template_family": "gemma",
        "notes": "Gemma-2 instruct. Gated — set HF_TOKEN env var.",
    },
}


# ── Dry-run validator ─────────────────────────────────────────────────


def _hf_token_available() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder
        return bool(HfFolder.get_token())
    except Exception:  # noqa: BLE001
        return False


def _model_already_cached(model_id: str) -> bool:
    """Check both HF_HOME and TRANSFORMERS_CACHE locations for the model."""
    name = "models--" + model_id.replace("/", "--")
    candidate_dirs = [
        Path(DEFAULT_CACHE) / name,
        Path(DEFAULT_CACHE) / "hub" / name,
        Path(DEFAULT_CACHE) / "transformers" / name,
    ]
    return any(d.exists() for d in candidate_dirs)


def dry_run_check(
    model_id: str,
    *,
    require_token_present: bool = False,
) -> dict[str, Any]:
    """Validate one model's tokenizer + chat template wiring without
    downloading weights.

    Returns
    -------
    report : dict
        ``{"model_id", "config", "tokenizer_loaded", "chat_template_ok",
           "weights_cached", "hf_token_present", "errors", "warnings"}``.
    """
    report: dict[str, Any] = {
        "model_id": model_id,
        "config": MODEL_REGISTRY.get(model_id),
        "tokenizer_loaded": False,
        "chat_template_ok": False,
        "weights_cached": False,
        "hf_token_present": _hf_token_available(),
        "errors": [],
        "warnings": [],
    }
    cfg = MODEL_REGISTRY.get(model_id)
    if cfg is None:
        report["errors"].append(
            f"model_id={model_id!r} not in MODEL_REGISTRY; "
            f"add an entry (size_b, vram_peak_gb, ...) before training"
        )
        return report

    # 1) HF token presence for gated models.
    if cfg["needs_hf_token"] and not report["hf_token_present"]:
        msg = (
            f"{model_id} is gated; set HF_TOKEN before downloading weights"
        )
        if require_token_present:
            report["errors"].append(msg)
        else:
            report["warnings"].append(msg)

    # 2) Tokenizer load + chat template check (small download, ~MB).
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=cfg["trust_remote_code"],
            use_fast=cfg["use_fast_tokenizer"],
            cache_dir=DEFAULT_CACHE,
        )
        report["tokenizer_loaded"] = True
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        sample_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "What is 2 + 2?"},
        ]
        try:
            templated = tokenizer.apply_chat_template(
                sample_messages, tokenize=False, add_generation_prompt=True,
            )
            report["chat_template_ok"] = bool(templated and len(templated) > 0)
            report["chat_template_sample_chars"] = len(templated)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(
                f"apply_chat_template failed: {type(e).__name__}: {e}"
            )

        # 3) Tokenize a batch to confirm vocabulary works.
        try:
            tokenizer(["hello world", "another test"], return_tensors=None,
                      padding=True, truncation=True, max_length=64)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(
                f"tokenizer batch encode failed: {type(e).__name__}: {e}"
            )
    except Exception as e:  # noqa: BLE001
        report["errors"].append(
            f"AutoTokenizer.from_pretrained failed "
            f"({type(e).__name__}): {str(e)[:200]}"
        )

    # 4) Weights-cached check (no download).
    report["weights_cached"] = _model_already_cached(model_id)

    return report


def _print_report(report: dict[str, Any]) -> None:
    cfg = report["config"]
    mid = report["model_id"]
    print(f"  [{mid}]")
    if cfg is None:
        for er in report["errors"]:
            print(f"    ❌ {er}")
        return
    print(f"    size:        {cfg['size_b']} B")
    print(f"    vram_peak:   ~{cfg['vram_peak_gb']} GB")
    print(f"    chat family: {cfg['chat_template_family']}")
    print(f"    needs token: {cfg['needs_hf_token']}  "
          f"(HF_TOKEN present: {report['hf_token_present']})")
    print(f"    weights cached: {report['weights_cached']}")
    print(f"    tokenizer loaded: {report['tokenizer_loaded']}")
    print(f"    chat template:  {report['chat_template_ok']}"
          + (f" ({report.get('chat_template_sample_chars', 0)} chars)"
             if report["chat_template_ok"] else ""))
    print(f"    notes: {cfg['notes']}")
    for w in report["warnings"]:
        print(f"    ⚠️  {w}")
    for e in report["errors"]:
        print(f"    ❌ {e}")


def vram_budget_warning(model_id: str, available_gb: float = 80.0) -> str | None:
    cfg = MODEL_REGISTRY.get(model_id)
    if cfg is None:
        return None
    peak = cfg["vram_peak_gb"]
    if peak >= available_gb * 0.9:  # within 10% of capacity
        return (
            f"⚠️  {model_id}: peak ~{peak} GB is close to available "
            f"~{available_gb:.0f} GB. Enable gradient_checkpointing or "
            f"reduce batch size."
        )
    return None


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model-id", default=None,
                    help="single model_id to validate; mutually exclusive with --all")
    ap.add_argument("--all", action="store_true",
                    help="dry-run check for every model in MODEL_REGISTRY")
    ap.add_argument("--multi-env", action="store_true",
                    help="(intent only — see --launch) train multi-env via "
                         "train_multienv_grpo logic")
    ap.add_argument("--launch", action="store_true",
                    help="actually run training; implementation-only by default")
    ap.add_argument("--require-token", action="store_true",
                    help="fail dry-run if HF_TOKEN is missing for a gated model")
    args = ap.parse_args()

    if not args.base_model_id and not args.all:
        ap.error("Pass --base-model-id <id> or --all")
    if args.base_model_id and args.all:
        ap.error("--base-model-id and --all are mutually exclusive")

    targets = [args.base_model_id] if args.base_model_id else list(MODEL_REGISTRY.keys())

    print("=== Universal GRPO trainer — DRY-RUN model validation ===\n")
    reports: list[dict[str, Any]] = []
    for mid in targets:
        rep = dry_run_check(mid, require_token_present=args.require_token)
        _print_report(rep)
        reports.append(rep)
        warn = vram_budget_warning(mid)
        if warn:
            print(f"    {warn}")
        print()

    fatal = [r for r in reports if r["errors"]]
    if fatal:
        print(f"❌ Dry-run failed for {len(fatal)} model(s).")
        return 2

    print(f"✅ All {len(reports)} model(s) passed dry-run validation.")
    if not args.launch:
        print("[DRY-RUN] no training launched. Pass --launch to start a real run.")
        return 0

    # Launch path — single-env (default) or multi-env. Both delegate to
    # the existing scripts via subprocess so this entry point stays thin.
    if len(targets) != 1:
        print("ERROR: --launch requires --base-model-id (one model at a time).",
              file=sys.stderr)
        return 2
    mid = targets[0]
    if args.multi_env:
        print(f"[LAUNCH] multi-env training is gated through "
              f"examples/training/train_multienv_grpo.py with model = {mid}")
        print("        Run that script with --launch to train; the multi-env "
              "model_id override is wired in Phase 14 follow-up.")
    else:
        print(f"[LAUNCH] single-env training requested for {mid}; this "
              f"functionality is gated through examples/training/train_grpo_qwen.py")
        print("        Phase 14 follow-up will thread --base-model-id "
              "into that script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
