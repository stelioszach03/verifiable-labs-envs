"""Demo 3: GSM8K-style step-validity reward (continuous, ensemble σ).

Scenario
--------
The user's reward function returns the *fraction of reasoning steps* that
were judged valid by an automatic step-checker, in ``[0, 1]``. The σ for
each completion is the *standard deviation across an ensemble* of three
independent step-checkers — small σ = checkers agree, large σ = checkers
disagree.

This is the bread-and-butter case for ``nonconformity="scaled_residual"``:
continuous reward + per-trace σ that reflects model uncertainty. Compared
to demo 2, the reward here is genuinely continuous (not 0/1), so
interval widths track σ smoothly across the test set.

Run::

    python examples/calibrate/03_gsm8k_step_validity.py
"""
from __future__ import annotations

import numpy as np

import vlabs_calibrate as vc

ALPHA = 0.1
N_CALIBRATION = 500
N_TEST = 1000


def step_validity_reward(*, problem: str, reasoning_steps: tuple[float, ...]) -> float:
    """Mean step validity in ``[0, 1]``.

    ``reasoning_steps`` is a per-step probability-of-validity in [0, 1]; the
    reward is their mean. This stands in for "fraction of steps that the
    automatic checker accepted" in real GSM8K pipelines.
    """
    del problem
    return float(np.mean(reasoning_steps))


def synthesize_traces(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        # Each problem has 3-7 steps with skill-level mean probability ~ 0.7.
        n_steps = int(rng.integers(3, 8))
        skill = float(np.clip(rng.normal(0.7, 0.15), 0.0, 1.0))
        steps = tuple(float(np.clip(rng.normal(skill, 0.1), 0.0, 1.0)) for _ in range(n_steps))
        # Ensemble σ: stand-in for std across 3 checkers — bigger when steps are uncertain.
        sigma = float(np.clip(0.05 + 0.4 * np.std(steps), 0.02, 0.5))
        # Reference reward = noisy gold around skill, scaled by σ.
        ref = float(np.clip(skill + sigma * rng.standard_normal(), 0.0, 1.0))
        out.append(
            {
                "problem": f"gsm_{i}",
                "reasoning_steps": steps,
                "reference_reward": ref,
                "uncertainty": sigma,
            }
        )
    return out


def main() -> int:
    print("vlabs-calibrate demo 03 — GSM8K-style step-validity with ensemble σ")
    print("=" * 70)
    train = synthesize_traces(N_CALIBRATION, seed=40)
    test = synthesize_traces(N_TEST, seed=41)

    cal = vc.calibrate(step_validity_reward, train, alpha=ALPHA, nonconformity="scaled_residual")
    print(f"calibrated n={cal.n_calibration}  α={cal.alpha}  q={cal.quantile:.4f}")
    print(f"non-conformity stats: {dict(cal.nonconformity_stats)}")

    # Show how interval width tracks σ across a few representative test points.
    sigmas = sorted(test, key=lambda t: t["uncertainty"])
    samples = [sigmas[0], sigmas[len(sigmas) // 2], sigmas[-1]]
    print("\n[interval width vs σ — three representative samples]")
    print("| label | σ | reward | lower | upper | width | covered |")
    print("|---|---:|---:|---:|---:|---:|:-:|")
    for tag, t in zip(("narrow", "median", "widest"), samples, strict=True):
        r = cal(
            problem=t["problem"],
            reasoning_steps=t["reasoning_steps"],
            sigma=t["uncertainty"],
            reference=t["reference_reward"],
        )
        width = r.interval[1] - r.interval[0]
        print(
            f"| {tag} | {r.sigma:.3f} | {r.reward:.4f} | "
            f"{r.interval[0]:.4f} | {r.interval[1]:.4f} | {width:.4f} | "
            f"{'✅' if r.covered else '❌'} |"
        )

    report = cal.evaluate(test)
    print(
        f"\nheld-out eval: n={report.n}  empirical={report.empirical_coverage:.4f}  "
        f"target={report.target_coverage:.2f}  passes={report.passes}\n"
        f"  interval width: median={report.interval_width_median:.4f}  "
        f"mean={report.interval_width_mean:.4f}\n"
        f"  non-conformity: mean={report.nonconformity['mean']:.4f}  "
        f"max={report.nonconformity['max']:.4f}"
    )
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
