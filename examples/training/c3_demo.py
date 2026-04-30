"""C.3 demo — reasoning tags validation matrix."""
from __future__ import annotations

import json

from verifiable_labs_envs.training import (
    extract_tagged_answer,
    make_reward_fn,
)

print("=== Sample tagged completion (well-formed) ===")
sample = (
    "Sure, here is my analysis.\n"
    "<think>\n"
    "Looking at y, the support is concentrated near low frequencies.\n"
    "I'll guess indices 0..9 with zero amplitudes.\n"
    "</think>\n"
    "<answer>\n"
    + json.dumps({"support_idx": list(range(10)), "support_amp_x1000": [0]*10})
    + "\n</answer>"
)
print(sample)
print("---extracted from <answer>:---")
print(extract_tagged_answer(sample))
print()

zero_json = json.dumps({"support_idx": list(range(10)), "support_amp_x1000": [0]*10})

print("=== Validation matrix (use_tags=True) ===")
fn = make_reward_fn("sparse-fourier-recovery", use_tags=True)
cases = [
    ("both tags + valid JSON",
     sample),
    ("missing <think>",
     f"<answer>{zero_json}</answer>"),
    ("missing <answer>",
     "<think>thinking step by step...</think>"),
    ("wrong order (<answer> first)",
     f"<answer>{zero_json}</answer><think>after</think>"),
    ("both tags + bad JSON",
     "<think>x</think><answer>{not valid}</answer>"),
    ("multiple <answer>, first valid",
     f"<think>x</think><answer>{zero_json}</answer><answer>{{decoy}}</answer>"),
    ("naked JSON (no tags)",
     zero_json),
    ("case-insensitive (<THINK><ANSWER>)",
     f"<THINK>thinking</THINK><ANSWER>{zero_json}</ANSWER>"),
    ("whitespace-tolerant",
     f"\n<think>\nfoo\n</think>\n\n<answer>\n{zero_json}\n</answer>\n"),
]
prompts = [""] * len(cases)
completions = [c for _, c in cases]
seeds = [0] * len(cases)
fn.stats.reset()
rewards = fn(prompts=prompts, completions=completions, instance_seed=seeds)

W = 38
print(f"{'case':<{W}} {'reward':>7} {'parse':>6} {'fmt':>4} {'failure_type':<16}")
print("-" * (W + 7 + 6 + 4 + 17 + 4))
for (name, _), r, rec in zip(cases, rewards, fn.stats.per_call):
    pv = int(rec["components"]["parse_valid"])
    fv = int(rec["components"]["format_valid"])
    ft = rec["failure_type"] or "-"
    print(f"{name:<{W}} {r:>7.4f} {pv:>6} {fv:>4} {ft:<16}")

print()
print("=== Backward compatibility (use_tags=False, M2 fixture) ===")
fn_legacy = make_reward_fn("sparse-fourier-recovery", use_tags=False)
fn_legacy.stats.reset()
r = fn_legacy(prompts=[""], completions=[zero_json], instance_seed=[0])
rec = fn_legacy.stats.per_call[-1]
print(f"naked JSON via use_tags=False:  reward={r[0]:.4f}  format_valid={int(rec['components']['format_valid'])}")
