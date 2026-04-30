"""Tests for Phase D — multi-model validation setup (impl only)."""
from __future__ import annotations

import importlib.util as _il
import sys
from pathlib import Path

# Import the universal trainer + report generator as modules.
_REPO = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, name: str):
    p = _REPO / rel_path
    spec = _il.spec_from_file_location(name, p)
    mod = _il.module_from_spec(spec)
    assert spec.loader is not None
    # Register in sys.modules BEFORE exec so @dataclass introspection works
    # (it does sys.modules.get(cls.__module__).__dict__).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_uni = _load_module("examples/training/train_grpo_universal.py",
                    "train_grpo_universal")
_rep = _load_module("examples/reports/generate_capability_report.py",
                    "generate_capability_report")


# ── 1. MODEL_REGISTRY contains all 4 expected entries ────────────────


def test_model_registry_has_four_models() -> None:
    expected = {
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "google/gemma-2-2b-it",
    }
    assert set(_uni.MODEL_REGISTRY) == expected
    for _mid, cfg in _uni.MODEL_REGISTRY.items():
        assert {"size_b", "vram_peak_gb", "needs_hf_token",
                "trust_remote_code", "use_fast_tokenizer",
                "chat_template_family", "notes"} <= set(cfg)


# ── 2. dry_run_check on Qwen (cached) succeeds ───────────────────────


def test_dry_run_check_qwen_passes() -> None:
    rep = _uni.dry_run_check("Qwen/Qwen2.5-1.5B-Instruct")
    assert rep["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert rep["config"] is not None
    assert rep["tokenizer_loaded"] is True
    assert rep["chat_template_ok"] is True
    assert rep["weights_cached"] is True  # cached from M5/M6
    assert rep["errors"] == []


# ── 3. dry_run_check on unknown model emits errors ───────────────────


def test_dry_run_check_unknown_model_errors() -> None:
    rep = _uni.dry_run_check("totally-fake/Model-9999")
    assert rep["config"] is None
    assert any("not in MODEL_REGISTRY" in e for e in rep["errors"])


# ── 4. Capability report renders from mock data ──────────────────────


def test_capability_report_mock_renders(tmp_path: Path) -> None:
    results = [_rep._from_dict(d) for d in _rep.MOCK_RESULTS]
    md, plots = _rep.render_capability_report(
        results, plots_dir=tmp_path / "plots",
    )
    assert "# Capability Improvement Report" in md
    # All 4 sections from the spec must be present.
    for header in (
        "## 1. Executive Summary",
        "## 2. Δ improvement per model",
        "## 3. Best-performing base model",
        "## 4. Sample efficiency comparison",
        "## 5. Cost per improvement point",
        "## 6. Recommendations",
        "## 7. Reproducibility checklist",
        "## 8. Pricing tier recommendations",
    ):
        assert header in md
    # All 4 model_ids appear in the table.
    for d in _rep.MOCK_RESULTS:
        assert d["model_id"] in md
    # Two PNGs were rendered.
    assert len(plots) == 2
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 0


# ── 5. ModelResult cost_per_point calc ───────────────────────────────


def test_model_result_cost_per_point() -> None:
    r = _rep._from_dict(_rep.MOCK_RESULTS[0])  # Qwen
    # delta_pp = 0.1237 × 100 = 12.37 pp; gpu_hours = 2.0; $/hr = 1.0
    # cost_per_point = 2.0 × 1.0 / 12.37 ≈ $0.1617 / pp
    assert abs(r.cost_per_point_usd - 2.0 / 12.37) < 0.005


# ── 6. Cheap model identification (smallest $/pp) ────────────────────


def test_cheapest_model_is_one_with_lowest_cost_per_pp() -> None:
    results = [_rep._from_dict(d) for d in _rep.MOCK_RESULTS]
    cheapest = min(results, key=lambda r: r.cost_per_point_usd)
    # In MOCK_RESULTS, Llama (smallest model, fewest GPU hours) should win.
    assert cheapest.model_id == "meta-llama/Llama-3.2-1B-Instruct"
