"""vlabs-prm-eval — process-reward-model eval card tooling.

Current commands:

* ``vlabs-prm-eval gate`` — checkpoint promotion gate. Reads two eval
  cards (old / new), maps each onto ``ModelMetrics``, evaluates the
  7-condition ``AcceptUpdate`` predicate from
  ``formal/VerifiableLabsFormal/SelfImprovementGate.lean`` via the
  Python mirror in ``verifiable_labs_envs.formal_spec.gate``, and exits
  0/1 accordingly.

The PRM eval-card schema is defined in ``vlabs_prm_eval.gate`` (see
``EvalCard`` + ``DEFAULT_METRICS_MAP``).
"""
