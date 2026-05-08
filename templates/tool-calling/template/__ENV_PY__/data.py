"""Problem generator for __ENV_ID__.

Each call must produce a fresh problem dict with these keys:

- ``template_name`` — categorical tag for telemetry.
- ``prompt`` — natural-language task the LLM reads.
- ``gold_spec`` — oracle data the env's
  ``_check_gold_state(state, instance)`` consumes to build the
  per-template predicate.
- ``initial_files`` — workspace seed (passed into
  ``verifiable_labs_envs.tool_primitives.init_state``).
- ``available_tools`` — tuple of tool names visible to the model;
  must be a subset of ``TOOL_DISPATCH`` keys
  (``calculator``, ``web_search``, ``read_file``, ``write_file``,
  ``send_message``).

Procedural regeneration from a 64-bit seed plus a finite template
pool gives the contamination-resistance guarantee that makes the env
safe to use as an RLVR training signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verifiable_labs_envs.tool_primitives import WorkspaceState


@dataclass(frozen=True)
class ToolCallingInstance:
    """One tool-calling problem draw.

    ``gold_spec`` is the oracle excluded from
    :meth:`as_inputs`. ``initial_files`` is the workspace seed.
    """

    prompt: str
    template_name: str
    seed: int
    gold_spec: dict[str, Any]
    initial_files: dict[str, str]
    available_tools: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "available_tools": list(self.available_tools),
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class ToolCallingPrediction:
    """Solver's recorded trajectory.

    ``tool_calls`` is the ordered list of ``{name, arguments, result}``
    triples produced during the rollout. ``final_text`` is the LLM's
    terminating non-tool message; ``final_state`` is the workspace
    state at rollout end.
    """

    tool_calls: tuple[dict[str, Any], ...]
    final_text: str
    final_state: WorkspaceState
    raw: str = ""
    confidence: float = 0.5


def generate_problem(seed: int, **hyperparams: Any) -> dict[str, Any]:
    """Sample a fresh tool-calling problem from the per-env distribution.

    Determinism: identical ``seed`` returns the byte-identical dict.
    """
    raise NotImplementedError(
        "TODO: implement problem generator for __ENV_ID__. "
        "Use np.random.default_rng(seed) for reproducibility, sample "
        "templates from a pool whose size × seed-space × parameter "
        "range gives EFFECTIVE_INSTANCES > 1e15. See "
        "src/verifiable_labs_envs/envs/tool_calling_single.py for a "
        "worked example with 10 templates."
    )


__all__ = ["ToolCallingInstance", "ToolCallingPrediction", "generate_problem"]
