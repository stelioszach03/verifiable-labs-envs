"""Demo 1: HumanEval-style binary pass/fail reward.

Scenario
--------
The user's reward function is a *cheap proxy* — it runs a small subset of
test cases and returns ``1.0`` if all of them pass, ``0.0`` otherwise.
The reference reward is the *gold standard* — running the full test suite,
which is too expensive to do on every rollout.

We calibrate the proxy against the gold using ``nonconformity="binary"``.
Because the standard split-conformal guarantee is degenerate for 0/1
rewards, the resulting interval is either a singleton ``{predicted}``
(the proxy is "verified" against the gold by the calibration set) or
the trivial ``[0, 1]`` (the calibration set saw too much disagreement to
guarantee anything). This demo shows both regimes.

Run::

    python examples/calibrate/01_humaneval_passfail.py
"""
from __future__ import annotations

import numpy as np

import vlabs_calibrate as vc

ALPHA = 0.1
N_CALIBRATION = 500
N_TEST = 1000


def cheap_proxy_reward(*, prompt: str, completion: str, ground_truth: str) -> float:
    """Toy proxy: agrees with ground_truth based on the parity of len(completion).

    A real proxy would run a small subset of unit tests; the choice of toy
    rule here is arbitrary — what matters is that the proxy is *cheap* and
    *imperfect* relative to the gold-standard label.
    """
    del prompt, ground_truth  # unused — proxy decides on completion alone
    return 1.0 if len(completion) % 2 == 0 else 0.0


def synthesize_traces(n: int, *, agreement_rate: float, seed: int) -> list[dict]:
    """Generate ``n`` calibration traces.

    Each trace has prompt + completion + ground_truth + the proxy's reward
    plus a ``reference_reward`` (the gold). The agreement rate is the
    probability that ``reference_reward == proxy_reward``.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        completion = "x" * int(rng.integers(1, 30))  # arbitrary length
        prompt = f"task_{i}"
        ground_truth = "expected"
        proxy = cheap_proxy_reward(
            prompt=prompt, completion=completion, ground_truth=ground_truth
        )
        reference = float(proxy if rng.random() < agreement_rate else 1.0 - proxy)
        out.append(
            {
                "prompt": prompt,
                "completion": completion,
                "ground_truth": ground_truth,
                "reference_reward": reference,
            }
        )
    return out


def run_scenario(label: str, agreement_rate: float, seed_train: int, seed_test: int) -> None:
    train = synthesize_traces(N_CALIBRATION, agreement_rate=agreement_rate, seed=seed_train)
    test = synthesize_traces(N_TEST, agreement_rate=agreement_rate, seed=seed_test)

    cal = vc.calibrate(cheap_proxy_reward, train, alpha=ALPHA, nonconformity="binary")
    report = cal.evaluate(test)

    sample = test[0]
    point = cal(
        prompt=sample["prompt"],
        completion=sample["completion"],
        ground_truth=sample["ground_truth"],
    )

    print(f"\n--- {label} (agreement_rate={agreement_rate:.2f}) ---")
    print(f"calibration n={cal.n_calibration}  α={cal.alpha}  q={cal.quantile:.4f}")
    print(
        f"sample call: reward={point.reward}  interval={point.interval}  "
        f"target_coverage={point.target_coverage}"
    )
    print(
        f"eval n={report.n}  empirical_coverage={report.empirical_coverage:.4f}  "
        f"target={report.target_coverage:.2f}  passes={report.passes}"
    )

    if report.quantile < 1.0:
        print("  -> proxy is verified against gold within (1-α) coverage on this set.")
    else:
        print("  -> degenerate regime: interval is the trivial [0, 1].")
        print("     For non-trivial conditional coverage on binary tasks, see Mondrian conformal.")


def main() -> int:
    print("vlabs-calibrate demo 01 — HumanEval-style binary pass/fail proxy")
    print("=" * 70)
    # High-agreement regime: proxy and gold mostly agree -> q collapses to 0.
    run_scenario("high-agreement", agreement_rate=0.95, seed_train=10, seed_test=20)
    # Low-agreement regime: proxy diverges from gold often enough that the
    # split-conformal guarantee becomes vacuous.
    run_scenario("low-agreement", agreement_rate=0.55, seed_train=11, seed_test=21)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
