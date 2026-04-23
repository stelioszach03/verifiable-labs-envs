#!/usr/bin/env python3
"""Quickstart for verifiable-labs-envs.

Walks through the three shipped environments end-to-end: load, generate
an instance, run the reference baseline, score the reconstruction, print
the full reward / component / meta dict. Run me with::

    python examples/quickstart.py
"""
from __future__ import annotations

from verifiable_labs_envs import list_environments, load_environment


def main() -> None:
    print("Available environments:", list_environments())
    print()

    for name in sorted(list_environments()):
        print("=" * 72)
        print(f"  {name}")
        print("=" * 72)

        env = load_environment(name)
        print(f"conformal_quantile (calibrated): {env.conformal_quantile:.3f}")
        print(f"hyperparameters: {env.hyperparams}")

        out = env.run_baseline(seed=0)
        print(f"seed=0 reference-baseline reward: {out['reward']:.3f}")
        print("  components:", out["components"])
        print("  meta:", {k: v for k, v in out["meta"].items() if k != "weights"})
        print()


if __name__ == "__main__":
    main()
