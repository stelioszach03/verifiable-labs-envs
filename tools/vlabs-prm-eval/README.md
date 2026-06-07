# vlabs-prm-eval

Process-reward-model eval-card tooling for Verifiable Labs.

Today the package ships a single command — **`vlabs-prm-eval gate`** — which
evaluates the seven-condition checkpoint-promotion gate from
[`formal/VerifiableLabsFormal/SelfImprovementGate.lean`](../../formal/VerifiableLabsFormal/SelfImprovementGate.lean)
against a pair of eval cards (old vs candidate) and exits `0` on **ACCEPT**
or `1` on **REJECT**. The gate predicate itself is machine-verified in Lean 4
(`accepted_sequence_mono_VGS`, `accepted_sequence_VGS_lower_bound`) and
mirrored in Python under `verifiable_labs_envs.formal_spec.gate`; this CLI
adds only the eval-card → `ModelMetrics` schema bridge and a user-facing
report.

Planned subcommands (`card`, `aggregate`, `promote`) follow the same
pattern but are intentionally out of scope here.

## Install

From the monorepo root:

```bash
pip install -e tools/vlabs-prm-eval
```

The package depends on `verifiable-labs-envs` (for the verified gate predicate)
and `typer`.

## `vlabs-prm-eval gate`

```text
Usage: vlabs-prm-eval gate [OPTIONS]

  Evaluate the 7-condition checkpoint promotion gate
  (SelfImprovementGate.lean) against two eval cards. Exit 0 on ACCEPT,
  1 on REJECT.

Options:
  -O, --old PATH           Path to the previous-checkpoint eval card (JSON).
                           [required]
  -N, --new PATH           Path to the candidate-checkpoint eval card (JSON).
                           [required]
  --metrics-map PATH       Optional JSON file overriding DEFAULT_METRICS_MAP
                           (vgs_weights, regression_rules).
  --tau FLOAT              Minimum VGS gain required for an accept.  [default: 0.01]
  --eps-h FLOAT            Allowed increase in hack_risk.  [default: 0.02]
  --eps-c FLOAT            Allowed drop in calibration (empirical coverage).
                           [default: 0.02]
  --eps-o FLOAT            Allowed drop in OOD (mean held-out env metric).
                           [default: 0.02]
  --eps-k FLOAT            Allowed cost increase ($/audit).  [default: 5.0]
  --eps-l FLOAT            Allowed latency increase (seconds/audit).  [default: 0.5]
  --cost-old FLOAT         Override old-card cost if not present in JSON.
  --cost-new FLOAT         Override new-card cost if not present in JSON.
  --latency-old FLOAT      Override old-card latency if not present in JSON.
  --latency-new FLOAT      Override new-card latency if not present in JSON.
  --help                   Show this message and exit.
```

### Eval-card schema

```json
{
  "model_id":                  "qwen-2.5-1.5b-grpo-step-500",
  "processbench_overall":      0.78,
  "mean_held_out_env_metric":  0.71,
  "bon_accuracy":              0.69,
  "coverage":                  0.91,
  "held_out_envs": {
    "sparse-fourier-recovery": 0.74,
    "kalman-filter-tuning":    0.68,
    "ode-residual":            0.71
  },
  "invariance_violation_rate": 0.08,
  "cost":                      0.62,
  "latency":                   0.94
}
```

All fields except `model_id` are optional. Missing numeric fields default to
`0.0`, except for `invariance_violation_rate`: when absent, `hack_risk`
defaults to `0.0` and a `WARNING:` line is emitted on stderr noting that
the gate cannot detect shortcut-taking without that field. Run
`vlabs-prm-eval`'s invariance harness (see
`verifiable_labs_envs.formal_spec.invariance.check_invariance`) to populate
it.

### Eval-card → `ModelMetrics` mapping

The gate consumes `ModelMetrics` (`vgs`, `hack_risk`, `calibration`, `ood`,
`cost`, `latency`, `regression`). The default mapping — overridable via
`--metrics-map` — is:

| `ModelMetrics` field | Source                                                                                    |
|----------------------|-------------------------------------------------------------------------------------------|
| `vgs`                | `0.40·processbench_overall + 0.30·mean_held_out_env_metric + 0.30·bon_accuracy`           |
| `hack_risk`          | `invariance_violation_rate` (else 0.0 + stderr warning)                                   |
| `calibration`        | `coverage`                                                                                |
| `ood`                | `mean(held_out_envs.values())` if present, else `mean_held_out_env_metric`                |
| `cost`               | `cost` (or `--cost-old` / `--cost-new` override)                                          |
| `latency`            | `latency` (or `--latency-old` / `--latency-new` override)                                 |
| `regression`         | `True` if any of: `processbench_overall < 0.60`, `\|coverage − 0.90\| > 0.05`, or `bon_accuracy(new) − bon_accuracy(old) < 0.05`. |

To override the weights or thresholds, pass a `--metrics-map <map.json>`:

```json
{
  "vgs_weights": {
    "processbench_overall":      0.50,
    "mean_held_out_env_metric":  0.30,
    "bon_accuracy":              0.20
  },
  "regression_rules": {
    "processbench_min":  0.65,
    "bon_lift_min":      0.03,
    "coverage_target":   0.90,
    "coverage_tol":      0.05
  }
}
```

### Output

`vlabs-prm-eval gate` prints an `== ACCEPT ==` / `== REJECT ==` header
followed by a per-condition table with the old and new values, the budget,
and an `OK`/`!!` marker (ASCII-only for cross-platform stdout safety).
On reject, the canonical reason names from
`verifiable_labs_envs.formal_spec.gate` are listed:

```text
== REJECT ==

condition                              old        new     budget  OK
-------------------------------- ---------- ---------- ----------  --
vgs_gain >= tau                      0.7050     0.7080     0.0100  !!
hack_risk <= eps_h                   0.1000     0.1000     0.0200  OK
calibration >= -eps_c                0.9000     0.9000     0.0200  OK
ood >= -eps_o                        0.7000     0.7000     0.0200  OK
cost <= eps_k                        1.0000     1.0000     5.0000  OK
latency <= eps_l                     1.0000     1.0000     0.5000  OK
regression flag                       False      False      False  OK

Reasons:
  - vgs_gain_below_tau
```

Exit code: `0` on accept, `1` on reject — suitable for use in CI gates and
in the Phase 30.F/G checkpoint-promotion scripts.

## Provenance

The `accept_update` predicate this CLI invokes is the Python mirror of the
Lean 4 definition in
[`formal/VerifiableLabsFormal/SelfImprovementGate.lean`](../../formal/VerifiableLabsFormal/SelfImprovementGate.lean),
machine-verified sorry-free under the three standard Lean axioms only
(`propext`, `Classical.choice`, `Quot.sound`). The Python mirror is
property-tested against the formal specification in
`tests/formal_spec/test_gate_properties.py`.
