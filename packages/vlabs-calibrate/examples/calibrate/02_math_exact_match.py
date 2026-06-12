"""Demo 2: MATH-style exact-match reward with judge confidence as σ.

Scenario
--------
The user's reward function is a *judge-graded* exact-match score: an LLM
judge extracts the final answer from the completion, normalizes it, and
compares it to the gold answer. The judge also emits a confidence
``σ ∈ (0, 1]`` for its extraction (small σ = confident, large σ =
ambiguous formatting / plausible alternative answers).

Because the reward is a function of the judge — a noisy estimator of the
true correctness — we calibrate with ``nonconformity="scaled_residual"``.
Test-time intervals widen with σ so the confidence flag is honest.

Run::

    python examples/calibrate/02_math_exact_match.py
"""
from __future__ import annotations

import numpy as np

import vlabs_calibrate as vc

ALPHA = 0.1
N_CALIBRATION = 500
N_TEST = 1000


def judge_reward(*, problem: str, completion: str, gold_answer: str) -> float:
    """Toy judge: returns 1.0 if the gold appears in the completion, else 0.0."""
    del problem  # not used by this toy judge
    return 1.0 if gold_answer.strip() in completion else 0.0


def synthesize_traces(n: int, *, base_acc: float, seed: int) -> list[dict]:
    """Generate ``n`` calibration traces.

    Each trace contains a (problem, completion, gold_answer, σ) tuple plus
    a noisy ``reference_reward`` ∈ [0, 1]. ``σ`` is sampled to reflect the
    judge's per-trace confidence; the noise added to the reference scales
    with σ so the synthesis is consistent with the calibration assumption.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        gold = str(int(rng.integers(0, 100)))
        # ~base_acc of the time the model gets it right; otherwise noise.
        if rng.random() < base_acc:
            completion = f"Therefore the answer is {gold}."
        else:
            wrong = str(int(rng.integers(0, 100)))
            completion = f"Therefore the answer is {wrong}."
        sigma = float(rng.uniform(0.05, 0.30))
        proxy = judge_reward(problem=f"prob_{i}", completion=completion, gold_answer=gold)
        reference = float(np.clip(proxy + sigma * rng.standard_normal(), 0.0, 1.0))
        out.append(
            {
                "problem": f"prob_{i}",
                "completion": completion,
                "gold_answer": gold,
                "reference_reward": reference,
                "uncertainty": sigma,
            }
        )
    return out


def main() -> int:
    print("vlabs-calibrate demo 02 — MATH-style exact-match with judge confidence")
    print("=" * 70)
    train = synthesize_traces(N_CALIBRATION, base_acc=0.7, seed=30)
    test = synthesize_traces(N_TEST, base_acc=0.7, seed=31)

    cal = vc.calibrate(judge_reward, train, alpha=ALPHA, nonconformity="scaled_residual")
    print(f"calibrated n={cal.n_calibration}  α={cal.alpha}  q={cal.quantile:.4f}")
    print(f"non-conformity stats: {dict(cal.nonconformity_stats)}")

    # Two illustrative test calls — narrow vs wide σ.
    narrow = test[0]
    wide_idx = max(range(len(test)), key=lambda i: test[i]["uncertainty"])
    wide = test[wide_idx]
    for tag, t in (("narrow σ", narrow), ("widest σ", wide)):
        result = cal(
            problem=t["problem"],
            completion=t["completion"],
            gold_answer=t["gold_answer"],
            sigma=t["uncertainty"],
            reference=t["reference_reward"],
        )
        print(
            f"\n[{tag}] σ={t['uncertainty']:.3f}  reward={result.reward:.1f}  "
            f"interval=({result.interval[0]:.3f}, {result.interval[1]:.3f})  "
            f"covered={result.covered}"
        )

    report = cal.evaluate(test)
    print(
        f"\nheld-out eval: n={report.n}  empirical={report.empirical_coverage:.4f}  "
        f"target={report.target_coverage:.2f}  passes={report.passes}  "
        f"width_median={report.interval_width_median:.4f}  width_mean={report.interval_width_mean:.4f}"
    )
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
