"""Problem generator for __ENV_ID__.

Each call must produce a fresh ``CodeInstance`` from the supplied
seed. The instance carries:

- ``function_signature`` — the `def name(args) -> type:` line the
  solver must implement.
- ``docstring`` — natural-language spec the solver reads.
- ``visible_tests`` — small set of assertions the prompt shows
  (a few input/output examples). Visible to the model.
- ``hidden_tests`` — full assertion battery used for scoring.
  **Never** included in :meth:`CodeInstance.as_inputs`.
- ``gold_solution`` — reference implementation; passes 100% of
  hidden_tests by construction.
- ``template_name`` — categorical tag for telemetry.

Procedural regeneration from a 64-bit seed plus a finite template
pool gives the contamination-resistance guarantee that makes the env
safe to use as an RLVR training signal.

TODO: replace the body of :func:`generate_problem` with your domain's
generator. Examples shipped under ``src/verifiable_labs_envs/envs/``:

- ``code-humaneval``: 12 procedural templates spanning lists, strings,
  dicts, ints, trees, graphs.
- ``code-humaneval-multiturn``: same generator, multi-turn dialogue
  with test-feedback between turns.
- ``code-humaneval-tools``: same generator, model composes
  read_file / write_file / run_test primitives.
- ``code-mini-repo``: synthetic 3–5-file mini-repo, multi-file edits.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeInstance:
    """One coding-problem draw.

    ``hidden_tests`` is the oracle field excluded from
    :meth:`as_inputs`. ``gold_solution`` is also kept hidden so a
    cheating solver can't dump it back as its prediction.
    """

    function_signature: str
    docstring: str
    visible_tests: tuple[str, ...]
    hidden_tests: tuple[str, ...]
    gold_solution: str
    template_name: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """Composed natural-language problem statement.

        Same shape as a HumanEval prompt: signature + indented
        docstring + visible test block.
        """
        body = textwrap.indent(self.docstring.strip(), "    ")
        visible_block = "\n".join(f"    >>> {t}" for t in self.visible_tests)
        return (
            f"{self.function_signature}\n"
            f'    """\n{body}\n\n'
            f"    Examples:\n{visible_block}\n"
            f'    """'
        )

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "function_signature": self.function_signature,
            "visible_tests": list(self.visible_tests),
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class CodePrediction:
    """Solver's answer.

    ``code`` is the Python source the LLM proposes; ``raw`` keeps the
    full LLM response for traceability; ``confidence`` is a scalar
    self-report in ``[0, 1]``.
    """

    code: str
    raw: str = ""
    confidence: float = 0.5


def generate_problem(seed: int, **hyperparams: Any) -> dict[str, Any]:
    """Sample a fresh problem dict from the per-env distribution.

    Determinism: two calls with the same seed and hyperparameters must
    return identical dicts. The returned dict has keys
    ``function_signature``, ``docstring``, ``visible_tests``,
    ``hidden_tests``, ``gold_solution``, ``template_name``.
    """
    raise NotImplementedError(
        "TODO: implement problem generator for __ENV_ID__. "
        "Use np.random.default_rng(seed) for reproducibility, sample "
        "templates from a pool whose size × seed-space gives "
        "EFFECTIVE_INSTANCES > 1e15. See "
        "src/verifiable_labs_envs/envs/code_humaneval.py for a worked "
        "example with 12 templates."
    )


__all__ = ["CodeInstance", "CodePrediction", "generate_problem"]
