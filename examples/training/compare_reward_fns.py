"""Side-by-side demo of `make_reward_fn` (env-native sum) vs
`make_reward_fn_posterior` (P-GRPO gating) on identical inputs.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/training"))

from test_reward_fn_posterior import _oracle_completion, _zero_completion  # type: ignore[import-not-found]

from verifiable_labs_envs.training import make_reward_fn, make_reward_fn_posterior

ENV = "sparse-fourier-recovery"
SEED = 42

fn_v1 = make_reward_fn(ENV)
fn_v2 = make_reward_fn_posterior(ENV)

garbage = "this is not json at all"
zero = _zero_completion()

# Random JSON: right schema, garbage indices.
rng = random.Random(0)
random_idx = sorted(rng.sample(range(256), 10))
random_amps = [rng.randint(-1000, 1000) for _ in range(10)]
random_pred = json.dumps(
    {"support_idx": random_idx, "support_amp_x1000": random_amps}
)

# Partial-match: keep 5 of the oracle's true indices, replace the other 5
# with off-support indices.
oracle = _oracle_completion(SEED)
o = json.loads(oracle)
rng2 = random.Random(1)
off_support = [i for i in range(256) if i not in o["support_idx"]]
partial_idx = sorted(o["support_idx"][:5] + rng2.sample(off_support, 5))
partial_pred = json.dumps(
    {"support_idx": partial_idx, "support_amp_x1000": o["support_amp_x1000"]}
)

strategies: list[tuple[str, str]] = [
    ("garbage (parse fail)", garbage),
    ("zero @ idx 0..9 (format ok, outcome fail)", zero),
    ("random JSON (format ok, outcome fail)", random_pred),
    ("5/10 support match (format ok, partial outcome)", partial_pred),
    ("oracle (perfect)", oracle),
]

prompts = [""] * len(strategies)
completions = [c for _, c in strategies]
seeds = [SEED] * len(strategies)

fn_v1.stats.reset()
fn_v2.stats.reset()
r_v1 = fn_v1(prompts=prompts, completions=completions, instance_seed=seeds)
r_v2 = fn_v2(prompts=prompts, completions=completions, instance_seed=seeds)

NAME_W = 50
print(
    f"{'strategy':<{NAME_W}} {'env-native v1':>13} {'posterior v2':>13} "
    f"{'r_fmt':>6} {'r_out':>6} {'r_qual':>7}"
)
print("-" * (NAME_W + 13 + 13 + 6 + 6 + 7 + 8))
for i, (name, _) in enumerate(strategies):
    rec2 = fn_v2.stats.per_call[i]
    c2 = rec2["components"]
    print(
        f"{name:<{NAME_W}} {r_v1[i]:>13.4f} {r_v2[i]:>13.4f} "
        f"{int(c2['r_format']):>6d} {int(c2['r_outcome']):>6d} {c2['r_quality']:>7.4f}"
    )

print()
print("Aggregates:")
print(f"  env-native v1: {fn_v1.stats.aggregate()}")
print(f"  posterior v2:  {fn_v2.stats.aggregate()}")
