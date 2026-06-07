"""Python mirror of the machine-verified Lean 4 specification in ``formal/``.

This package implements, in pure Python, the formulas, the 7-condition
self-improvement gate, and the invariance-violation harness whose
mathematical properties are proved in
``formal/VerifiableLabsFormal/``.

The Python implementation is *property-tested* against the Lean
specification (see ``tests/formal_spec/``); it is **not** itself
formally verified. When documenting downstream behaviour, do not
describe this code, the SDK, or the hosted API as "formally verified".
The only verified artefact is the Lean source. The approved wording is
in the project ``README.md`` under "Formally verified guarantees".

Lean cross-reference:

* ``formulas``  ↔ ``CalibratedReward.lean`` + ``VGS.lean`` +
  ``AdaptiveDifficulty.lean`` + ``ModelRouting.lean``
* ``gate``      ↔ ``SelfImprovementGate.lean``
* ``invariance``↔ ``VerifierInvariance.lean``
"""

from .formulas import (
    calibrated_reward,
    vgs,
    difficulty_update,
    routing_utility,
    select_model,
)
from .gate import (
    ModelMetrics,
    Tolerances,
    GateDecision,
    accept_update,
)
from .invariance import (
    InvarianceReport,
    check_invariance,
)

__all__ = [
    "calibrated_reward",
    "vgs",
    "difficulty_update",
    "routing_utility",
    "select_model",
    "ModelMetrics",
    "Tolerances",
    "GateDecision",
    "accept_update",
    "InvarianceReport",
    "check_invariance",
]
