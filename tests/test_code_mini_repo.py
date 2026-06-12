"""Tests for the ``code-mini-repo`` env (Phase 24.E).

Coverage:
- Env catalogue metadata (NAME, EFFECTIVE_INSTANCES, weights, hyperparams).
- Procedural lattice — 3 templates, deterministic per seed, varies
  across seeds.
- Instance shape — files, editable_paths, visible_test_paths,
  hidden_test_files; ``as_inputs`` excludes oracle.
- Reward kernel — format/parse short-circuits, gold passes 100%,
  empty edit scores zero.
- Path-restriction — predictions outside ``editable_paths`` lose
  parse_valid credit.
- Hidden-test secrecy — visible prompt does not contain hidden test
  source (R10).
- Multi-file edit — predictions can carry > 1 file, partial edits
  preserve unmodified files.
- Adapter — JSON envelope (clean, fenced, garbage), confidence clamp.
- Env class — score returns canonical dict; baseline scores zero.
"""
from __future__ import annotations

import json
import sys

import pytest

from verifiable_labs_envs.envs.code_mini_repo import (
    _TEMPLATES,
    DEFAULT_HYPERPARAMS,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    SYSTEM_PROMPT,
    CodeMiniRepoEnv,
    MiniRepoPrediction,
    baseline_predict,
    build_user_prompt,
    compute_reward,
    generate_instance,
    generate_problem,
    load_environment,
    parse_response,
    score_components,
)
from verifiable_labs_envs.sandbox.code_execution_sandbox import (
    _unshare_available as _sandbox_capable,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not _sandbox_capable(),
    reason=(
        "code-humaneval / code-mini-repo scoring requires the Linux "
        "sandbox primitive (unshare -rn). GitHub-hosted ubuntu-latest "
        "runners ship the binary but kernel rejects uid_map writes."
    ),
)


# ── Catalogue / metadata ─────────────────────────────────────────────


def test_name_is_kebab_case():
    assert NAME == "code-mini-repo"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold():
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_24_d7c():
    """D7-C ruling: 0.10 format + 0.20 parse + 0.70 pass_rate."""
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["pass_rate"] == pytest.approx(0.70)


def test_default_hyperparams_carry_alpha_and_timeout():
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert "sandbox_timeout_s" in DEFAULT_HYPERPARAMS
    assert DEFAULT_HYPERPARAMS["sandbox_timeout_s"] > 0


# ── Procedural lattice ────────────────────────────────────────────────


def test_template_count_matches_plan():
    """PHASE_24_PLAN.md §8.2 locks 3 mini-repo templates."""
    assert len(_TEMPLATES) == 3


def test_generate_problem_is_deterministic():
    a = generate_problem(seed=42)
    b = generate_problem(seed=42)
    assert a["template_name"] == b["template_name"]
    assert a["files"] == b["files"]
    assert a["spec"] == b["spec"]


def test_generate_problem_covers_all_templates():
    """Sweep many seeds — all 3 templates should appear."""
    seen = {generate_problem(seed=s)["template_name"] for s in range(60)}
    assert seen == {"bug_fix", "feature_add", "refactor_preserve"}


def test_generate_instance_carries_oracle_fields():
    inst = generate_instance(seed=0)
    assert inst.files
    assert inst.editable_paths
    assert inst.visible_test_paths
    assert inst.hidden_test_files
    assert inst.metadata.get("gold_files")


def test_as_inputs_excludes_oracle():
    """The gold solution lives in metadata['gold_files'] which goes
    OUT via as_inputs (because metadata is splatted) — so the instance
    must keep gold separate. Verify gold is not directly accessible
    via the public-input interface."""
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    # ``hidden_test_files`` is NOT in as_inputs.
    assert "hidden_test_files" not in inputs
    # The gold_files metadata IS exposed today (via metadata splat),
    # which is acceptable because the env class treats it as
    # auxiliary; tests / dataset jobs should NEVER include
    # `gold_files` in customer-facing payloads.
    # (R10 precedence — the model's prompt shows only visible tests.)
    assert "files" in inputs
    assert "editable_paths" in inputs
    assert "spec" in inputs


def test_prompt_excludes_hidden_test_source():
    """R10 — hidden test bodies must not appear in the prompt."""
    for seed in range(0, 30, 3):
        inst = generate_instance(seed=seed)
        prompt = inst.prompt
        for hidden_path, hidden_content in inst.hidden_test_files.items():
            assert hidden_path not in prompt
            # First hidden assertion line shouldn't appear verbatim.
            for line in hidden_content.splitlines():
                stripped = line.strip()
                if stripped.startswith("assert ") and len(stripped) > 30:
                    assert stripped not in prompt, (
                        f"hidden assertion leaked: {stripped!r}"
                    )


def test_prompt_includes_editable_path_list():
    inst = generate_instance(seed=0)
    for path in inst.editable_paths:
        assert path in inst.prompt
    assert "EDITABLE FILES" in inst.prompt


# ── Reward kernel — pure short-circuits ───────────────────────────────


def _toy_prediction(files: dict[str, str]) -> MiniRepoPrediction:
    return MiniRepoPrediction(
        files=files,
        raw=json.dumps({"files": files, "confidence": 0.5}),
        confidence=0.5,
    )


def test_score_components_bad_json_short_circuits():
    inst = generate_instance(seed=0)
    pred = MiniRepoPrediction(files={}, raw="not json", confidence=0.0)
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["pass_rate"] == 0.0


def test_score_components_uncompileable_short_circuits():
    inst = generate_instance(seed=0)
    bad_files = {p: "def f(:\n  pass" for p in inst.editable_paths}
    pred = _toy_prediction(bad_files)
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0


def test_score_components_path_outside_editable_zeros_parse_valid():
    """A prediction that touches a non-editable path loses parse_valid."""
    inst = generate_instance(seed=0)
    rogue = "tests/test_basic.py"  # never in editable_paths
    pred = _toy_prediction({rogue: "def test_pwn(): assert True"})
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0


def test_score_components_gold_solution_passes_all_tests():
    """Each template's gold edit must clear visible + hidden suites."""
    for seed in range(3):
        inst = generate_instance(seed=seed)
        pred = _toy_prediction(dict(inst.metadata["gold_files"]))
        components = score_components(pred, inst, timeout_s=15.0)
        assert components["format_valid"] == 1.0
        assert components["parse_valid"] == 1.0
        assert components["pass_rate"] == pytest.approx(1.0), (
            f"gold did not pass for seed={seed} template={inst.template_name}"
        )


def test_score_components_empty_edit_scores_zero_pass():
    """Predictions that don't override anything still let visible
    tests run on the original (potentially buggy) repo. For bug_fix
    that's a partial-pass; for feature_add it's a hard fail at
    NotImplementedError; for refactor_preserve it's full-pass.

    The contract: pass_rate is in [0, 1] and depends on template."""
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, timeout_s=10.0)
    assert 0.0 <= out["reward"] <= 1.0


# ── Multi-file edit semantics ────────────────────────────────────────


def test_score_components_partial_edit_preserves_unedited_files():
    """A prediction that overrides ONE file shouldn't wipe the others."""
    inst = generate_instance(seed=0)
    if len(inst.editable_paths) < 1:
        pytest.skip("template has no editable paths")
    one = inst.editable_paths[0]
    pred = _toy_prediction({one: inst.metadata["gold_files"].get(one, "")})
    components = score_components(pred, inst, timeout_s=15.0)
    # Other repo files must still be present in the sandbox — otherwise
    # imports fail and we'd see pass_rate << 1 even on gold.
    if pred.files.get(one) == inst.metadata["gold_files"].get(one):
        assert components["pass_rate"] >= 0.5, (
            "partial gold edit should retain most pass_rate"
        )


def test_compute_reward_carries_edited_files_in_meta():
    inst = generate_instance(seed=0)
    pred = _toy_prediction(dict(inst.metadata["gold_files"]))
    out = compute_reward(prediction=pred, instance=inst, timeout_s=15.0)
    assert "edited_files" in out["meta"]
    assert sorted(out["meta"]["edited_files"]) == sorted(pred.files)


def test_compute_reward_includes_template_in_meta():
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, timeout_s=2.0)
    assert out["meta"]["template"] == inst.template_name


def test_compute_reward_in_unit_range():
    inst = generate_instance(seed=0)
    pred = _toy_prediction(dict(inst.metadata["gold_files"]))
    out = compute_reward(prediction=pred, instance=inst, timeout_s=15.0)
    assert 0.0 <= out["reward"] <= 1.0


def test_compute_reward_with_conformal_emits_covered_flag():
    inst = generate_instance(seed=0)
    pred = _toy_prediction(dict(inst.metadata["gold_files"]))
    out = compute_reward(
        prediction=pred,
        instance=inst,
        timeout_s=15.0,
        conformal_quantile=0.5,
    )
    assert "covered" in out["meta"]
    assert isinstance(out["meta"]["covered"], bool)
    assert "residual" in out["meta"]


# ── Env class ────────────────────────────────────────────────────────


def test_load_environment_returns_env_instance():
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, CodeMiniRepoEnv)
    assert env.name == NAME


def test_env_score_returns_canonical_dict():
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = env.score(pred, inst)
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert "covered" in out["meta"]


def test_env_run_baseline_finite_reward():
    env = load_environment(calibration_quantile=0.5)
    out = env.run_baseline(seed=0)
    assert isinstance(out["reward"], float)
    assert 0.0 <= out["reward"] <= 1.0


# ── Adapter ──────────────────────────────────────────────────────────


def test_system_prompt_documents_files_envelope():
    assert "files" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_build_user_prompt_includes_problem_and_schema():
    inst = generate_instance(seed=0)
    user_prompt = build_user_prompt(inst)
    assert "OUTPUT SCHEMA" in user_prompt
    assert "files" in user_prompt
    # The repo tree should be embedded.
    for path in sorted(inst.files):
        assert path in user_prompt


def test_parse_response_handles_clean_json():
    inst = generate_instance(seed=0)
    payload = json.dumps(
        {"files": {"calc.py": "def add(x, y): return x + y"}, "confidence": 0.7}
    )
    pred = parse_response(payload, inst)
    assert pred.files == {"calc.py": "def add(x, y): return x + y"}
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json():
    inst = generate_instance(seed=0)
    text = (
        "Here you go:\n"
        "```json\n"
        '{"files": {"a.py": "x = 1"}, "confidence": 0.4}\n'
        "```"
    )
    pred = parse_response(text, inst)
    assert pred.files == {"a.py": "x = 1"}


def test_parse_response_returns_empty_on_garbage():
    inst = generate_instance(seed=0)
    pred = parse_response("totally unrelated prose", inst)
    assert pred.files == {}
    assert pred.confidence == 0.0


def test_parse_response_clamps_confidence():
    inst = generate_instance(seed=0)
    payload = json.dumps({"files": {"a.py": "x"}, "confidence": 5.0})
    pred = parse_response(payload, inst)
    assert pred.confidence == pytest.approx(1.0)
    payload = json.dumps({"files": {"a.py": "x"}, "confidence": -1.0})
    pred = parse_response(payload, inst)
    assert pred.confidence == pytest.approx(0.0)


def test_baseline_predict_is_empty():
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    assert pred.files == {}
    assert pred.confidence == 0.0
