#!/usr/bin/env python3
"""Quickstart: GRPO post-training on a Verifiable Labs environment.

Minimal, runnable end-to-end loop that points TRL's GRPOTrainer at
`sparse-fourier-recovery`. Episode latency is essentially zero (pure
NumPy DFT — no image, no subprocess), the gap between zero baseline
(~0.35) and OMP classical baseline (~0.93) gives reward signal room
to show up, and the strict-JSON output is the canonical case for the
low-temperature pattern in main() below.

Runtime (defaults: Qwen 0.5B, 10 GRPO steps, 8 prompts): ~12-15 min on
a free Colab T4; ~6-8 min on a paid A100 (bump --max-steps to 500 and
--model to Qwen/Qwen2.5-1.5B-Instruct for a real research run); ~30 s
for `--dry-run` on CPU once the model is cached locally.

Expect mean reward to climb from ~0 (parse failures dominate) toward
the ~0.35 zero-solution baseline as the model learns to emit valid
{"x_hat": [n floats]} JSON. The bulk of measurable GRPO gain on this
config is format compliance, not structural inversion — the ~0.93 OMP
classical-baseline number above is a specialised-solver upper bound,
not something a 0.5B is expected to approach in 10 or even 500 steps.
A flat-zero curve almost always means a generation-format bug rather
than a learning failure (see the temperature comment).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from verifiable_labs_envs import load_environment
from verifiable_labs_envs.envs.sparse_fourier import Prediction

ENV_ID = "sparse-fourier-recovery"

_PROMPT = """\
Reconstruct a length-{n} real signal that is k={k}-sparse in the
standard basis from m={m} subsampled DFT measurements.

Mask (DFT indices, 0-based): {mask}
Measurements y[i] = (DFT[x])[mask[i]] as (real, imag) pairs: {y}

Reply with ONLY this JSON object on a single line, no prose:
{{"x_hat": [<{n} floats>]}}
"""

_JSON_RE = re.compile(r"\{\s*\"x_hat\"\s*:\s*\[[^\[\]]*\]\s*\}", re.DOTALL)


def format_prompt(instance) -> str:
    inp = instance.as_inputs()
    return _PROMPT.format(
        n=int(inp["n"]),
        k=int(inp["k"]),
        m=int(inp["mask"].size),
        mask=inp["mask"].tolist(),
        y=[(float(z.real), float(z.imag)) for z in inp["y"]],
    )


def make_reward_fn(env_id: str):
    """TRL-compatible reward closure. Parse / shape failures score 0.0
    so a bad completion never crashes the loop."""
    env = load_environment(env_id)

    def reward_fn(prompts=None, completions=None, **kwargs):
        seeds = kwargs.get("seed") or [0] * len(completions)
        rewards = []
        for completion, seed in zip(completions, seeds, strict=False):
            text = completion if isinstance(completion, str) else str(completion)
            try:
                m = _JSON_RE.search(text)
                if m is None:
                    rewards.append(0.0)
                    continue
                instance = env.generate_instance(seed=int(seed))
                n = int(instance.as_inputs()["n"])
                x_hat = np.asarray(json.loads(m.group(0))["x_hat"], dtype=np.float64)
                if x_hat.shape != (n,):
                    rewards.append(0.0)
                    continue
                pred = Prediction(x_hat=x_hat, sigma_hat=np.ones(n))
                rewards.append(float(env.score(pred, instance)["reward"]))
            except Exception:
                rewards.append(0.0)
        return rewards

    return reward_fn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF causal-LM (default fits free Colab T4).")
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="Build env+reward+config+trainer, skip .train().")
    args = p.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    env = load_environment(ENV_ID)
    seeds = list(range(args.num_seeds))
    ref = env.run_baseline(seed=seeds[0])["reward"]
    print(f"env={ENV_ID}  model={args.model}  "
          f"conformal_q={env.conformal_quantile:.3f}  "
          f"reference-baseline-reward(seed=0)={ref:.3f}")

    train_dataset = Dataset.from_list([
        {"prompt": format_prompt(env.generate_instance(seed=s)), "seed": s}
        for s in seeds
    ])

    # Temperature 0.5 is the single most important knob in this file for
    # strict-JSON envs. At T >= 0.7, small Qwens routinely hit EOS before
    # closing the array — those completions parse-fail, the reward fn
    # returns 0.0, and the curve looks like training never started. Cap
    # T low while the schema is strict; raise it for open-ended envs.
    config = GRPOConfig(
        output_dir="./grpo_quickstart_out",
        learning_rate=1e-6, num_generations=4, beta=0.04,
        max_prompt_length=2048, max_completion_length=2048,
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        max_steps=args.max_steps, temperature=0.5,
        logging_steps=1, save_strategy="no", report_to=[],
    )
    print(f"config: max_steps={config.max_steps}, lr={config.learning_rate}, "
          f"G={config.num_generations}, T={config.temperature}, beta={config.beta}")
    print("building trainer (downloads model on first run)...")

    trainer = GRPOTrainer(
        model=args.model, reward_funcs=make_reward_fn(ENV_ID),
        args=config, train_dataset=train_dataset,
    )
    print(f"trainer ready: model_cls={type(trainer.model).__name__}, "
          f"dataset_rows={len(train_dataset)}")

    if args.dry_run:
        print("DRY-RUN OK — env, reward fn, dataset, config, trainer all built.")
        return
    trainer.train()
    print("training complete.")


if __name__ == "__main__":
    main()
