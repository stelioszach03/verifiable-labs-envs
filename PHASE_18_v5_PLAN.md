# Phase 18 — v5 PLAN (post-mortem of v4 multi-model GRPO failure)

**Status:** v4 (4 simultaneous A100 pods on RunPod) HALTED 2026-05-02 14:57 UTC+3
after ~6 hours of training-with-zero-reward across all 4 models.
Pods stopped via RunPod UI to halt billing. v5 is the next attempt
once root cause is properly understood.

## TL;DR

All 4 pods (Qwen2.5-1.5B-Instruct, Phi-3.5-mini-instruct,
gemma-2-2b-it, Llama-3.2-1B-Instruct) ran GRPO on `sparse-fourier-recovery`
with reward=0 across all training steps. **Zero gradient → zero
weight updates → effectively no training**, despite 50–130 reported
optimization steps per pod.

Five remediation attempts on the Gemma pod did not produce
reward > 0. The other 3 pods stayed in a clip_ratio=1.0 (no EOS)
regime throughout.

We do not yet have a verified fix. v5 must redesign before any
new pod billing.

## What was tried — chronological diary

### Attempt 1 — temperature 0.9 → 0.5 (BUG FIX, partial)

**Hypothesis:** TRL 0.17's `GRPOConfig.temperature` defaults to 0.9.
At T=0.9, sampling rarely picks the EOS token, so every completion
runs to the 512-token cap. This corrupts the JSON output and
parse fails.

**Action:** patched `temperature=0.5` into all 4 `<slug>_grpo.py`
scripts (line 242 originally; later normalized). Cleaned all
incomplete checkpoints (rationale: pre-fix gradient was zero,
so checkpoints ≡ base weights). Restarted all 4 fresh.

**Result:**
- Gemma DID start terminating (37.5% EOS rate, mean_term_len=280
  tokens) — partial success on the EOS dimension.
- Qwen / Phi / Llama: clip_ratio remained 1.0 across all logged
  steps — they STILL never EOS-terminate at T=0.5.
- Reward stayed at 0 across all 4. So even when termination
  worked, the content was unparseable.

**Verdict:** necessary but not sufficient. Did not actually
fix the underlying problem.

### Attempt 2 — debug instrumentation (DIAGNOSTIC, success)

**Hypothesis:** the reward_fn might be receiving inputs differently
in the training loop vs. our standalone smoke tests (which had
shown rewards 0.27–0.30 on a checkpoint).

**Action:** patched `reward_fn` in `gemma2_2b_grpo.py` to write
`/tmp/reward_debug.jsonl` per call, recording per-completion:
  - `completion_len` (chars)
  - `completion_head` (first 160 chars)
  - `seed_type` and `seed_repr`
  - `outcome` ("OK reward=…" or "ExceptionType: msg")
  - all rewards for the call
  - kwargs keys

**Result:** completions were pure backtick spam:

```
"````````````````````````````````````````````````````````````````````````````````"
"````````````\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n..."
```

All 8 completions per call, every call. `len(completion)` ranged
360–2412 chars. Outcome on every one: `LLMSolverError: no JSON
block found in response (no '{' character)`.

`seed_type=int`, `seed_repr=542` etc. — seed handling was correct.
kwargs_keys=['seed'] — only `seed` is passed by TRL. So reward_fn
was being called correctly. **The model output was the problem,
not the reward function.**

### Attempt 3 — aggressive sampling (FAILED)

**Hypothesis:** the backtick spam is a sampling-noise attractor.
With tighter sampling, the model would escape into more sensible
output.

**Action:** rewrote the GRPOConfig sampling block to:
```
temperature=0.5,
top_p=0.9,
top_k=20,
min_p=0.1,
repetition_penalty=1.5,
```

**Result:** completions were STILL pure backtick spam, length
1536 chars exactly = 384 tokens × 4 chars/token of the Gemma
"\`\`\`\`" multi-backtick token (id 91901).

This is the smoking gun: with `repetition_penalty=1.5` actually
applied, the logit for token 91901 should be divided by 1.5^k
after k repeats. After 5 repeats it should be 7.6× lower; after
10 repeats 58×; after 20 repeats 3,300×. The fact that the
model still picked it 384 times in a row means **the repetition
penalty is NOT being applied** in TRL 0.17's regular generation
path (`use_vllm=False`).

This is a TRL bug or a config-passthrough issue. Source inspection
showed `GenerationConfig` is built correctly at `trl.GRPOTrainer:650`
with `repetition_penalty=self.repetition_penalty`, and `model.generate()`
is called with that config at line 974. So the param IS in the
GenerationConfig. Either HF's generate() is not honoring it for
some reason, or TRL is overriding it later.

**Verdict:** sampling param tweaks are useless until the
TRL/HF integration is fixed.

### Attempt 4 — bad_words_ids backtick ban (PARTIAL SUCCESS)

**Hypothesis:** if we can't penalize, we ban. Inject
`generation_config.bad_words_ids` for every token whose decode
contains a backtick.

**Action:** added a post-trainer-init block:
```python
_backtick_ids = []
for _tid in range(tokenizer.vocab_size):
    if '`' in tokenizer.decode([_tid]):
        _backtick_ids.append([_tid])
trainer.generation_config.bad_words_ids = _backtick_ids
```

For Gemma, this banned **77 tokens**.

**Result:** the backtick attractor was eliminated. Model now
opens with `{` (good — JSON character) but immediately repeats
`}` 8–9 times, then fire emojis, then newlines:

```
"{}}}}}}}}}🔥🔥🔥🔥🔥🔥 🔥 🔥🔥  \n\n\n\n\n\n..."
```

Outcome: `LLMSolverError: missing key 'support_idx' in response`.
So parse_response found a `{` (the prefix), tried to extract JSON,
got `{}` (empty object), and failed for missing keys. Reward still 0.

This confirms the model is so distribution-shifted on this prompt
that it has multiple repeating attractors (backticks, braces,
emojis). Banning one just routes the probability mass to the next.

### Attempt 5 — JSON `{` prefill (UNVERIFIED)

**Hypothesis:** instead of fighting the model's bad attractors,
force the JSON to start by appending `{` to the prompt itself.
The model continues from `{` with the JSON contents.

**Action:**
- `_build_prompt` returns `chat_template_output + '{'`.
- `reward_fn` prepends `{` to the completion before
  `adapter.parse_response`.

**Result:** UNKNOWN. Gemma was relaunched, reached step 3/1500
(mid inductor compile, 22 s/it), then user halted before any
reward_debug.jsonl entry was written.

This is the most promising attempt and was not given a chance
to be evaluated.

## Summary of failure modes by pod

| pod | last verified state | failure mode |
|---|---|---|
| qwen25_15b | step 53/1500 | clip_ratio=1.0, mean_term_len=0, reward=0 (never EOS) |
| phi35_mini | step 74/1500 | clip_ratio=1.0, mean_term_len=0, reward=0 (never EOS) |
| gemma2_2b  | attempt 5 step 3 | various attractors (backticks → braces → emojis) |
| llama32_1b | step 128/1500 | clip_ratio=1.0, mean_term_len=0, reward=0 (never EOS) |

The Qwen/Phi/Llama failure mode (never EOS) was never debugged
with reward_fn instrumentation — we don't know if their content
is also garbage or if it's coherent JSON that just runs over the
512-token cap.

## Underlying issues (ranked by certainty)

1. **(certain) TRL 0.17 default temperature=0.9 + `use_vllm=False` causes
   max_new_tokens-clip on every generation.** Setting temperature=0.5
   helped Gemma slightly; doesn't help Qwen/Phi/Llama at all.
2. **(strongly suspected) `repetition_penalty` is silently a no-op
   in the `use_vllm=False` regular generation path.** rep_penalty=1.5
   should have killed the backtick loop after 5–10 repeats; it didn't
   over 384 repeats. TRL upstream bug or HF generate quirk.
3. **(strongly suspected) Small base models on this prompt format
   collapse to repeating-token attractors.** Confirmed for Gemma-2-2b-it
   (4 different attractors observed). Likely for the others too —
   verifying requires debug instrumentation we never deployed on them.
4. **(possible) The `sparse-fourier-recovery` prompt is too OOD
   for 1–2B param instruct models.** The schematic-only example in
   the system prompt may be insufficient. The model needs concrete
   in-context examples or fine-tuning before it understands the
   target format.
5. **(possible) Gradient checkpointing + `use_cache=False`** inside
   Gemma2DecoderLayer produces a forward-pass numerical regime
   different from inference and degrades generation quality.

## Hypotheses + concrete options for v5

Ranked by expected payoff vs. effort.

### Option A — Constrained (guided) decoding [BEST]

TRL 0.17 has `guided_decoding_regex` in the vLLM path. With
`use_vllm=True` and a regex like
`\{"support_idx":\s*\[[\d,\s]+\],\s*"support_amp_x1000":\s*\[[\-\d,\s]+\]\}`,
the FSM mask physically prevents the model from emitting any
non-conforming token. Reward becomes a function of the *content*
of the JSON, not whether JSON parses at all.

- Effort: 30–60 min for vLLM install + regex tuning.
- Payoff: removes the entire "model produces gibberish"
  failure mode at a stroke. Gives GRPO a chance to actually learn.
- Risk: vLLM install on RunPod was historically tricky; need to
  verify on one pod before fan-out.

### Option B — SFT warmup before GRPO

Generate ~200–500 (prompt, gold-JSON) pairs from `zero_agent.py`
or an oracle. Run 1–2 epochs of supervised fine-tuning to teach
the format. Then start GRPO from the SFT checkpoint.

- Effort: 1–2 hours for SFT pipeline + ~30 min training per model.
- Payoff: cold-start problem solved. Model will not be in
  the backtick attractor.
- Risk: more code, more moving parts. Need to verify SFT loss
  curve looks sane.

### Option C — JSON `{` prefill (Attempt 5, finished)

Re-run the v4 v5-attempt-5 fix on a single pod. If it works,
fan out. Cheap to test.

- Effort: 0 (patches are written, just re-deploy).
- Payoff: if model outputs valid JSON given `{` start, problem
  solved.
- Risk: model might still pick `}` immediately after the prefilled
  `{`, giving `{}` (empty object). The Attempt 4 result strongly
  suggests this is what happens.

### Option D — Debug what Qwen/Phi/Llama are actually emitting

We never deployed `patch_reward_debug.py` to the non-Gemma pods.
We don't know if their reward=0 is from the same garbage-attractor
problem or something else (could be valid JSON that just doesn't
score). Cheap and informative.

- Effort: 15 min per pod to redeploy patch + read first 20 entries.
- Payoff: tells us if the problem is universal or Gemma-specific.
- Risk: zero.

### Option E — Verify rep_penalty actually applies

Run a tiny standalone test that constructs a TRL `GenerationConfig`
the same way the trainer does, calls `model.generate()` with it,
and checks if rep_penalty is honored. If TRL has a bug, file
upstream.

- Effort: 30 min.
- Payoff: ground truth on whether sampling-param fixes can
  ever work, or whether bad_words_ids is the only viable lever.
- Risk: zero.

### Option F — Drop the smallest models

Llama-3.2-1B-Instruct may simply lack capacity. Qwen2.5-0.5B
was already excluded for the same reason. Drop Llama-1B from
the v5 fleet, focus on Phi-3.5-mini (3.8B) and Qwen2.5-1.5B.

- Effort: trivial — comment out Llama from the launcher.
- Payoff: smaller billing surface, less debugging.
- Risk: weakens the "small-model RL works" claim. Negative
  result on Llama-1B is itself a finding, but only worth
  reporting if we got it to actually train (which we didn't).

### Option G — Simpler env first

Phase 18 hits multiple risks at once: (i) JSON output format,
(ii) sparse-fourier-recovery being a complex env, (iii) GRPO
on small models, (iv) multi-model fan-out. Test the GRPO
pipeline first on `basic-addition` or any env where we know
zero_agent already gets non-trivial reward at base. Once that
works end-to-end, swap in sparse-fourier-recovery.

- Effort: 1–2 hours for env swap + smoke test.
- Payoff: isolates which of the 4 risks is the actual blocker.
- Risk: schedule slip.

## Recommended v5 sequence

1. **D + E first (90 min, no pods)** — read the 3 unobserved
   pods' likely failure mode by running a debug patch on a
   *single* pod (e.g. Phi). Verify rep_penalty bug locally
   if reproducible.
2. **A (next, single pod, 1–2 hours)** — vLLM + guided
   decoding on Phi-3.5-mini. If valid JSON is emitted with
   nontrivial reward variance, this is the path forward.
3. **If A works, fan out to Qwen + Phi + Gemma.** Skip Llama-1B.
   Run for 4–6 hours, evaluate at step 200 instead of waiting
   for full 1500.
4. **If A doesn't work, fall back to B (SFT warmup)**.

## Estimated time to next properly-evaluated training result

| stage | time |
|---|---|
| Decide A vs B (D+E investigation) | 90 min |
| Implement chosen path on 1 pod | 1–3 hours |
| Smoke verify (10 steps) | 15 min |
| Production run on 3 pods | 4–6 hours |
| Eval + audit | 1 hour |
| **Total wall-clock** | **8–12 hours** |

This presumes no further surprises. Prior attempts have shown
each phase has roughly 1× hidden-issue cost, so 1.5× of the
above (12–18 hours) is a more honest estimate.

## What is preserved on the network volume

If pods are restarted later (without wiping `/workspace`):
- `/workspace/checkpoints/phase18/<slug>_grpo_sf/` — incomplete
  checkpoints, all bit-identical to base (zero gradient → no
  weight updates). Safe to rm -rf for v5.
- `/workspace/<slug>_train_v3.log` — full training stdout
  for forensic analysis.
- `/workspace/hf_cache/` — model weights cached. Keep.
- `/workspace/verifiable-labs-envs/` — repo with all v4 patches
  applied to `examples/training/phase18/<slug>_grpo.py`. Either
  reset to clean origin/main or pick the patches we want
  retained.
- `/tmp/reward_debug.jsonl` (Gemma only) — in-RAM tmpfs, lost
  on pod stop. Cannot be recovered.

## What is in this repo (uncommitted, modified)

```
M src/verifiable_labs_envs/cli.py            (Phase 13 leftover, not Phase 18)
?? examples/eval/                             (Phase 13)
?? examples/reports/                          (Phase 13)
?? examples/training/                         (Phase 18 notebooks + scripts)
?? runs_local/                                (Phase 13 eval outputs)
?? src/verifiable_labs_envs/repro.py          (Phase 13)
?? src/verifiable_labs_envs/training/         (Phase 18 helpers)
?? tests/fixtures/                            (Phase 13)
?? tests/test_reproducibility.py              (Phase 13)
?? tests/test_timeout.py                      (Phase 13)
?? tests/training/                            (Phase 18 tests)
```

This `PHASE_18_v5_PLAN.md` is the only addition for v5.

## Local artifacts saved

`~/phase18_runpod_logs/` contains:
- `PHASE_18_PLAN_rev4.md` (original v4 plan)
- All `patch_*.py` and `*.sh` scripts used in v4 fixes
- `README.md` documenting the contents

Per-pod `*_train_v3.log`, `trainer_state.json`, and
`reward_debug.jsonl` could NOT be copied (pods unreachable).
