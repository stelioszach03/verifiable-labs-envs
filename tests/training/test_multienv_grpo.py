"""Tests for the multi-env GRPO scaffolding (Phase C.4 — IMPL ONLY).

Covers schema validation, sampler distribution, per-env tracker
isolation, seed-pool disjointness, JSON round-trip of tracker states,
held-out env never appearing in training samples, and a 1-batch
non-NaN reward smoke test (no gradient update).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

# Import the C.4 script as a module to test its public surface.
import importlib.util as _il
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "examples" / "training" / "train_multienv_grpo.py"
_spec = _il.spec_from_file_location("train_multienv_grpo", _SCRIPT_PATH)
_mod = _il.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

from verifiable_labs_envs.training import (
    OUTCOME_THRESHOLDS_REGISTRY,
    AdaptiveDifficultyTracker,
    make_reward_fn_multienv,
    validate_env_schema,
)


TRAIN_ENVS = _mod.TRAIN_ENVS
HELDOUT_ENV = _mod.HELDOUT_ENV
SEED_POOLS = _mod.SEED_POOLS


# ── 1. Pre-flight passes for all 3 train envs ────────────────────────


def test_preflight_all_three_train_envs() -> None:
    report = _mod.run_preflight(TRAIN_ENVS)
    for eid in TRAIN_ENVS:
        assert eid in report, f"missing report entry for {eid}"
        rec = report[eid]
        assert rec.get("errors") == [], f"{eid} pre-flight errors: {rec['errors']}"
        # Each env should produce 5 anchor instances successfully.
        assert len(rec["anchor_results"]) == 5
        # Each env must declare a posterior kind.
        assert rec["kind"] in {"sparse", "image"}


# ── 2. Sampler distribution is approximately uniform ─────────────────


def test_sampler_env_distribution_uniform() -> None:
    trackers = _mod.make_per_env_trackers(TRAIN_ENVS)
    rng = random.Random(0)
    rows = _mod.sample_batch_rows(trackers, n_rows=900, rng=rng)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["env_id"]] = counts.get(r["env_id"], 0) + 1
    # 900 / 3 envs = 300 each, allow generous slack.
    for eid in TRAIN_ENVS:
        assert 200 <= counts.get(eid, 0) <= 400, (
            f"{eid} count {counts.get(eid, 0)} outside [200, 400]"
        )


# ── 3. Per-env tracker state isolation ───────────────────────────────


def test_per_env_trackers_isolated() -> None:
    trackers = _mod.make_per_env_trackers(
        TRAIN_ENVS, tau_num=4, tau_acc=0.75
    )
    # Advance only sparse-fourier-recovery's tracker.
    sf = trackers["sparse-fourier-recovery"]
    for _ in range(4):
        sf.record_rollout(difficulty=0, reward=1.0)
    sf.maybe_advance()
    assert sf.h == 1
    # Other trackers must be untouched.
    for eid in TRAIN_ENVS:
        if eid != "sparse-fourier-recovery":
            assert trackers[eid].h == 0
            assert trackers[eid].advances == 0


# ── 4. Seed pools disjoint within and across envs ────────────────────


def test_seed_pools_disjoint() -> None:
    _mod.assert_seeds_disjoint()  # raises if not


def test_no_overlap_with_m5_test_seeds() -> None:
    """The M5 sparse-fourier TEST seeds (2000-2099) must NOT appear in
    any env's TRAIN/VAL pools — so paired BEFORE/AFTER eval stays
    untouched. (They live exactly at sparse-fourier-recovery/test.)"""
    m5_test = set(range(2000, 2100))
    for eid, pools in SEED_POOLS.items():
        for split_name, pool in pools.items():
            if (eid, split_name) == ("sparse-fourier-recovery", "test"):
                continue  # this is the M5 set itself
            assert not (m5_test & set(pool)), (
                f"{eid}/{split_name} overlaps M5 TEST seeds: "
                f"{sorted(m5_test & set(pool))[:5]}..."
            )


# ── 5. Tracker state JSON round-trip ─────────────────────────────────


def test_tracker_state_roundtrip(tmp_path: Path) -> None:
    trackers_a = _mod.make_per_env_trackers(TRAIN_ENVS, tau_num=4, tau_acc=0.75)
    # Mutate one tracker.
    sf = trackers_a["sparse-fourier-recovery"]
    for _ in range(4):
        sf.record_rollout(difficulty=0, reward=1.0)
    sf.maybe_advance()
    # Save and reload.
    state_path = tmp_path / "tracker_state.json"
    _mod.save_tracker_states(trackers_a, state_path)
    trackers_b = _mod.load_tracker_states(state_path)
    assert set(trackers_b.keys()) == set(TRAIN_ENVS)
    for eid in TRAIN_ENVS:
        a = trackers_a[eid]
        b = trackers_b[eid]
        assert a.l == b.l
        assert a.h == b.h
        assert a.advances == b.advances
        assert a.rollouts_total == b.rollouts_total
        assert a.tau_acc == b.tau_acc
        assert a.tau_num == b.tau_num


# ── 6. Resume: load_tracker_states reconstructs functional trackers ──


def test_load_trackers_are_usable(tmp_path: Path) -> None:
    """A loaded tracker must accept further record/advance calls."""
    trackers_a = _mod.make_per_env_trackers(TRAIN_ENVS, tau_num=2, tau_acc=0.5)
    state_path = tmp_path / "tracker_state.json"
    _mod.save_tracker_states(trackers_a, state_path)
    trackers_b = _mod.load_tracker_states(state_path)
    sf = trackers_b["sparse-fourier-recovery"]
    sf.record_rollout(difficulty=0, reward=1.0)
    sf.record_rollout(difficulty=0, reward=1.0)
    advanced = sf.maybe_advance()
    assert advanced is True
    assert sf.h == 1


# ── 7. Schema validation catches incompatible env ────────────────────


def test_validate_env_schema_catches_incompatible() -> None:
    """A score() dict missing a required component / meta key must produce
    schema errors for the matching env_id."""
    bad_sparse = {"reward": 0.0, "components": {"nmse": 0.0, "conformal": 0.0},
                  "meta": {}}  # missing 'support' + 'nmse_raw'
    errs = validate_env_schema("sparse-fourier-recovery", bad_sparse)
    assert any("support" in e for e in errs)
    assert any("nmse_raw" in e for e in errs)

    bad_image = {"reward": 0.0, "components": {"psnr": 0.0, "conformal": 0.0},
                 "meta": {}}  # missing 'ssim' + 'psnr_db'
    errs2 = validate_env_schema("super-resolution-div2k-x4", bad_image)
    assert any("ssim" in e for e in errs2)
    assert any("psnr_db" in e for e in errs2)

    # An unregistered env id should also surface as a schema error.
    errs3 = validate_env_schema("totally-fake-env", {})
    assert any("not in OUTCOME_THRESHOLDS_REGISTRY" in e for e in errs3)


# ── 8. Held-out env never sampled during training ────────────────────


def test_heldout_never_sampled() -> None:
    trackers = _mod.make_per_env_trackers(TRAIN_ENVS)  # only train envs!
    rng = random.Random(0)
    rows = _mod.sample_batch_rows(trackers, n_rows=300, rng=rng)
    sampled_envs = {r["env_id"] for r in rows}
    assert HELDOUT_ENV not in sampled_envs
    # And the held-out env's TRAIN pool is empty by construction.
    assert SEED_POOLS[HELDOUT_ENV]["train"] == range(0, 0)


# ── 9. Posterior reward fn loads for all 3 train envs ────────────────


def test_make_reward_fn_multienv_loads() -> None:
    fn = make_reward_fn_multienv(TRAIN_ENVS, use_tags=True)
    assert fn.env_ids == TRAIN_ENVS
    assert fn.use_tags is True
    # Stats start at zero.
    assert fn.stats.n_calls == 0
    # Per-env counts initialised.
    assert all(eid in fn.per_env_counts for eid in TRAIN_ENVS)


def test_make_reward_fn_multienv_rejects_unknown_env() -> None:
    with pytest.raises(ValueError, match="OUTCOME_THRESHOLDS_REGISTRY"):
        make_reward_fn_multienv(["totally-fake-env"])


# ── 10. Mock 1-batch reward call produces non-NaN finite floats ──────


def test_one_batch_reward_smoke() -> None:
    """Run the multi-env reward fn on a minimal batch with one row per
    train env. All three rewards must be finite floats. Bad completions
    → 0.0, not NaN."""
    fn = make_reward_fn_multienv(TRAIN_ENVS, use_tags=False)  # plain JSON in tests
    completions = [
        "totally not json",          # parse_fail for sparse-fourier
        "<not json either>",         # parse_fail for phase-retrieval
        "{not even json}",           # parse_fail for super-resolution
    ]
    seeds = [
        SEED_POOLS["sparse-fourier-recovery"]["train"].start,
        SEED_POOLS["phase-retrieval"]["train"].start,
        SEED_POOLS["super-resolution-div2k-x4"]["train"].start,
    ]
    env_ids = list(TRAIN_ENVS)
    rewards = fn(prompts=[""] * 3, completions=completions,
                 instance_seed=seeds, env_id=env_ids)
    import math
    assert len(rewards) == 3
    assert all(math.isfinite(r) for r in rewards)
    assert all(r == 0.0 for r in rewards)  # all parse_fail
    assert fn.stats.n_calls == 3
    assert all(fn.per_env_counts[eid] == 1 for eid in TRAIN_ENVS)


# ── 11. Pre-flight report serialisable as JSON (for milestone summary) ─


def test_preflight_report_json_serialisable() -> None:
    report = _mod.run_preflight(TRAIN_ENVS)
    blob = json.dumps(report, default=str)
    assert "sparse-fourier-recovery" in blob
    assert "score_components" in blob


# ── 12. D.0 fix: difficulty kwargs actually thread through to env ────


def test_difficulty_kwargs_thread_through_to_instance() -> None:
    """build_multienv_dataset must pass difficulty_to_kwargs(eid, diff)
    into env.generate_instance, so that different difficulty levels
    produce instances with different shapes (n, k for sparse;
    shape for image)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir="/content/drive/MyDrive/verifiable-labs/hf_cache",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Two rows for sparse-fourier-recovery at different difficulty anchors.
    rows = [
        {"env_id": "sparse-fourier-recovery", "difficulty": 0,
         "instance_seed": SEED_POOLS["sparse-fourier-recovery"]["train"].start},
        {"env_id": "sparse-fourier-recovery", "difficulty": 20,
         "instance_seed": SEED_POOLS["sparse-fourier-recovery"]["train"].start},
    ]
    ds = _mod.build_multienv_dataset(rows, tokenizer=tokenizer, use_tags=False)
    # Prompt at diff=0 (n=64, k=3) is shorter than at diff=20 (n=256, k=20).
    prompt_easy = ds[0]["prompt"]
    prompt_hard = ds[1]["prompt"]
    assert len(prompt_hard) > len(prompt_easy), (
        f"prompt at diff=20 should be longer than at diff=0; got {len(prompt_hard)} vs {len(prompt_easy)}"
    )
    # The easy prompt should mention k=3, the hard one k=20.
    assert '"k": 3' in prompt_easy
    assert '"k": 20' in prompt_hard
    assert '"n": 64' in prompt_easy
    assert '"n": 256' in prompt_hard
