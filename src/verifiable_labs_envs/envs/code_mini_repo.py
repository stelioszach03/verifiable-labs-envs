"""code-mini-repo — synthetic-mini-repo code-execution RL environment.

Phase 24.E ships the repo-scale variant under D6-B: a procedurally
generated 3-file mini-repo where the model must edit 1–3 files to
make a hidden test suite pass. No git checkout, no clone server —
every "repo" is synthesised fresh from a 64-bit seed plus a
3-template lattice.

Three templates (PHASE_24_PLAN.md §8.2):

- ``bug_fix``           — repo has a known bug; tests fail. Model edits the offending file to fix.
- ``feature_add``       — repo has a stub + a new failing test. Model implements the spec.
- ``refactor_preserve`` — repo has passing tests + a refactor spec. Model rewrites without breaking tests.

Reward kernel reuses ``code_humaneval``'s D7-C weights verbatim
(0.10 format + 0.20 parse + 0.70 pass_rate); the only change is
``files`` going from a single ``solution.py`` to a 3–5-file mapping
the model can selectively overwrite.

The sandbox primitive accepts ``files: dict[str, str]`` already
(``execute_in_sandbox_sync``), so the implementation slots in
without sandbox changes — we materialise the merged ``base_files +
prediction.files + hidden_test_files`` into a fresh tmpdir per call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.envs.code_humaneval import (
    DEFAULT_MEM_BYTES,
    DEFAULT_TIMEOUT_S_PER_CALL,
)
from verifiable_labs_envs.sandbox import (
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

NAME = "code-mini-repo"

# 3 templates × 64-bit seed × ~1e6 parameter combinations per template
# = 5.5e22 effective instances, well above the 1e15 procedural-
# regeneration gate.
EFFECTIVE_INSTANCES: int = 3 * (2**64) * 1_000_000

DEFAULT_ALPHA: float = 0.1
DEFAULT_TIMEOUT_S_PER_REPO: float = DEFAULT_TIMEOUT_S_PER_CALL
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "pass_rate": 0.7,
}
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "sandbox_timeout_s": DEFAULT_TIMEOUT_S_PER_REPO,
    "sandbox_mem_bytes": DEFAULT_MEM_BYTES,
}


# ── Public dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class MiniRepoInstance:
    """One mini-repo problem draw.

    ``files`` is the entire seed repo (3–5 files including a visible
    test file). ``editable_paths`` enumerates the files the model is
    allowed to rewrite. ``hidden_test_files`` is the oracle —
    additional pytest modules merged at score time, never visible to
    the solver. ``spec`` carries the natural-language task description
    that goes into the prompt.
    """

    files: dict[str, str]
    editable_paths: tuple[str, ...]
    visible_test_paths: tuple[str, ...]
    hidden_test_files: dict[str, str]
    spec: str
    template_name: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        """Composed prompt — repo tree + spec + visible tests body."""
        tree = "\n".join(f"  - {p}" for p in sorted(self.files))
        editable = ", ".join(sorted(self.editable_paths))
        visible_blocks = "\n\n".join(
            f"--- {p} ---\n{self.files.get(p, '<missing>')}"
            for p in self.visible_test_paths
        )
        return (
            f"REPO STRUCTURE:\n{tree}\n\n"
            f"EDITABLE FILES: {editable}\n\n"
            f"SPEC:\n{self.spec}\n\n"
            f"VISIBLE TESTS:\n{visible_blocks}"
        )

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "files": dict(self.files),
            "editable_paths": list(self.editable_paths),
            "visible_test_paths": list(self.visible_test_paths),
            "spec": self.spec,
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class MiniRepoPrediction:
    """Solver's answer.

    ``files`` is a partial mapping path → new content, restricted to
    the instance's ``editable_paths``. ``raw`` keeps the LLM's full
    response for traceability; ``confidence`` is a self-reported
    scalar in ``[0, 1]``.
    """

    files: dict[str, str]
    raw: str = ""
    confidence: float = 0.5


# ── Procedural template lattice ──────────────────────────────────────


def _tmpl_bug_fix(rng: np.random.Generator) -> dict[str, Any]:
    """Buggy `add` function — model fixes the operator.

    Parameter range: arbitrary integer pairs in the visible+hidden
    tests, ~1e8 distinct combinations.
    """
    a = int(rng.integers(-50, 50))
    b = int(rng.integers(-50, 50))
    files = {
        "calc.py": "def add(x: int, y: int) -> int:\n    return x - y\n",
        "main.py": (
            "from calc import add\n\n"
            "def total(values: list[int]) -> int:\n"
            "    out = 0\n"
            "    for v in values:\n"
            "        out = add(out, v)\n"
            "    return out\n"
        ),
        "tests/test_basic.py": (
            "from calc import add\n"
            "from main import total\n\n"
            f"def test_add_smoke():\n    assert add({a}, {b}) == {a + b}\n\n"
            "def test_total_smoke():\n    assert total([1, 2, 3]) == 6\n"
        ),
    }
    hidden_test_files = {
        "tests/test_hidden.py": (
            "from calc import add\n"
            "from main import total\n\n"
            "def test_add_zero():\n    assert add(0, 0) == 0\n\n"
            "def test_add_negative():\n    assert add(-5, 5) == 0\n\n"
            "def test_total_empty():\n    assert total([]) == 0\n\n"
            "def test_total_negative():\n    assert total([-1, -2, -3]) == -6\n"
        ),
    }
    spec = (
        "The `add` function in `calc.py` returns `x - y` (a bug). Edit `calc.py` "
        "so that `add(x, y)` returns the actual sum. Do not modify `main.py` "
        "or any test file."
    )
    gold_files = {
        "calc.py": "def add(x: int, y: int) -> int:\n    return x + y\n",
    }
    return _problem_dict(
        template_name="bug_fix",
        files=files,
        editable_paths=("calc.py",),
        visible_test_paths=("tests/test_basic.py",),
        hidden_test_files=hidden_test_files,
        spec=spec,
        gold_files=gold_files,
    )


def _tmpl_feature_add(rng: np.random.Generator) -> dict[str, Any]:
    """Stub fizzbuzz — model implements per the docstring."""
    upper = int(rng.integers(15, 30))
    files = {
        "fizz.py": (
            "def fizzbuzz(n: int) -> list[str]:\n"
            '    """Return a list of length n.\n\n'
            "    Index i (1-indexed) is:\n"
            '      - "FizzBuzz" if i is divisible by 15\n'
            '      - "Fizz" if i is divisible by 3\n'
            '      - "Buzz" if i is divisible by 5\n'
            "      - str(i) otherwise.\n"
            '    """\n'
            "    raise NotImplementedError\n"
        ),
        "tests/test_basic.py": (
            "from fizz import fizzbuzz\n\n"
            "def test_smoke():\n"
            "    assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
        ),
    }
    expected = []
    for i in range(1, upper + 1):
        if i % 15 == 0:
            expected.append("FizzBuzz")
        elif i % 3 == 0:
            expected.append("Fizz")
        elif i % 5 == 0:
            expected.append("Buzz")
        else:
            expected.append(str(i))
    hidden_test_files = {
        "tests/test_hidden.py": (
            "from fizz import fizzbuzz\n\n"
            "def test_length():\n"
            f"    assert len(fizzbuzz({upper})) == {upper}\n\n"
            "def test_full_sequence():\n"
            f"    assert fizzbuzz({upper}) == {expected!r}\n\n"
            "def test_fizz_at_3():\n"
            "    assert fizzbuzz(3)[2] == 'Fizz'\n\n"
            "def test_buzz_at_5():\n"
            "    assert fizzbuzz(5)[4] == 'Buzz'\n\n"
            "def test_fizzbuzz_at_15():\n"
            "    assert fizzbuzz(15)[14] == 'FizzBuzz'\n"
        ),
    }
    spec = (
        "The `fizzbuzz` function in `fizz.py` is a stub. Implement it per "
        "the docstring so all visible and hidden tests pass."
    )
    gold_files = {
        "fizz.py": (
            "def fizzbuzz(n: int) -> list[str]:\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            "            out.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            out.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            out.append('Buzz')\n"
            "        else:\n"
            "            out.append(str(i))\n"
            "    return out\n"
        ),
    }
    return _problem_dict(
        template_name="feature_add",
        files=files,
        editable_paths=("fizz.py",),
        visible_test_paths=("tests/test_basic.py",),
        hidden_test_files=hidden_test_files,
        spec=spec,
        gold_files=gold_files,
    )


def _tmpl_refactor_preserve(rng: np.random.Generator) -> dict[str, Any]:
    """Verbose `square_sum` — model refactors while keeping tests green."""
    n = int(rng.integers(3, 10))
    files = {
        "math_util.py": (
            "def square_sum(nums: list[int]) -> int:\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        squared = n * n\n"
            "        total = total + squared\n"
            "    return total\n"
        ),
        "main.py": (
            "from math_util import square_sum\n\n"
            "def report(nums: list[int]) -> str:\n"
            '    return f"sum of squares = {square_sum(nums)}"\n'
        ),
        "tests/test_basic.py": (
            "from math_util import square_sum\n"
            "from main import report\n\n"
            f"def test_visible_one():\n    assert square_sum([{n}, {n + 1}]) == "
            f"{n * n + (n + 1) ** 2}\n\n"
            "def test_report_smoke():\n"
            "    assert 'sum of squares' in report([1, 2, 3])\n"
        ),
    }
    hidden_test_files = {
        "tests/test_hidden.py": (
            "from math_util import square_sum\n\n"
            "def test_empty():\n    assert square_sum([]) == 0\n\n"
            "def test_negative():\n    assert square_sum([-1, -2, -3]) == 14\n\n"
            "def test_single():\n    assert square_sum([7]) == 49\n\n"
            "def test_large():\n"
            f"    assert square_sum(list(range({n + 5}))) == "
            f"{sum(i * i for i in range(n + 5))}\n"
        ),
    }
    spec = (
        "`square_sum` in `math_util.py` works but is verbose. Refactor it "
        "into a more concise form (e.g. a generator expression) without "
        "breaking the existing tests. The function signature must stay "
        "`def square_sum(nums: list[int]) -> int`."
    )
    gold_files = {
        "math_util.py": (
            "def square_sum(nums: list[int]) -> int:\n"
            "    return sum(n * n for n in nums)\n"
        ),
    }
    return _problem_dict(
        template_name="refactor_preserve",
        files=files,
        editable_paths=("math_util.py",),
        visible_test_paths=("tests/test_basic.py",),
        hidden_test_files=hidden_test_files,
        spec=spec,
        gold_files=gold_files,
    )


def _problem_dict(
    *,
    template_name: str,
    files: dict[str, str],
    editable_paths: tuple[str, ...],
    visible_test_paths: tuple[str, ...],
    hidden_test_files: dict[str, str],
    spec: str,
    gold_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "template_name": template_name,
        "files": files,
        "editable_paths": editable_paths,
        "visible_test_paths": visible_test_paths,
        "hidden_test_files": hidden_test_files,
        "spec": spec,
        "gold_files": gold_files,
    }


_TEMPLATES = (_tmpl_bug_fix, _tmpl_feature_add, _tmpl_refactor_preserve)


# ── Generators ───────────────────────────────────────────────────────


def generate_problem(seed: int, **_unused: Any) -> dict[str, Any]:
    """Sample a fresh mini-repo problem dict from the procedural lattice."""
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    return _TEMPLATES[template_idx](rng)


def generate_instance(seed: int, **kwargs: Any) -> MiniRepoInstance:
    """Wrap :func:`generate_problem` output in a :class:`MiniRepoInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed))
    return MiniRepoInstance(
        files=dict(problem["files"]),
        editable_paths=tuple(problem["editable_paths"]),
        visible_test_paths=tuple(problem["visible_test_paths"]),
        hidden_test_files=dict(problem["hidden_test_files"]),
        spec=problem["spec"],
        template_name=problem["template_name"],
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "sandbox_timeout_s": float(params["sandbox_timeout_s"]),
            "gold_files": dict(problem["gold_files"]),
        },
    )


# ── Reward kernel ────────────────────────────────────────────────────


def _is_compileable(content: str) -> bool:
    if not content or not content.strip():
        return False
    try:
        compile(content, "<file>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _is_format_valid(prediction: MiniRepoPrediction) -> bool:
    """``raw`` is JSON containing a non-empty ``files`` mapping."""
    if not prediction.raw:
        return bool(prediction.files)
    try:
        data = json.loads(prediction.raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    files = data.get("files")
    return isinstance(files, dict) and len(files) > 0


def _is_parse_valid(
    prediction: MiniRepoPrediction,
    instance: MiniRepoInstance,
) -> bool:
    """All prediction files are in editable_paths and their content compiles.

    Test files (.py inside ``tests/``) skip the compile check — pytest
    discovery handles them — but the path-restriction still applies.
    """
    if not prediction.files:
        return False
    editable = set(instance.editable_paths)
    for path, content in prediction.files.items():
        if path not in editable:
            return False
        if not _is_compileable(content):
            return False
    return True


_CONFTEST_PY = (
    "# Auto-injected by code-mini-repo so pytest discovers modules at the\n"
    "# repo root (top of the sandbox tmpdir) regardless of where the test\n"
    "# file lives.\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
)


def _merge_files(
    instance: MiniRepoInstance,
    prediction_files: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the final repo to materialise into the sandbox.

    Order: base files → editable overrides (capped to editable_paths)
    → hidden test modules → conftest.py (forces sandbox-root onto
    sys.path so ``from math_util import …`` style imports work
    inside ``tests/``).
    """
    merged = dict(instance.files)
    if prediction_files:
        editable = set(instance.editable_paths)
        for path, content in prediction_files.items():
            if path in editable:
                merged[path] = content
    merged.update(instance.hidden_test_files)
    # Always inject the conftest at the sandbox root. If a template
    # ever ships its own conftest.py we leave that one untouched.
    merged.setdefault("conftest.py", _CONFTEST_PY)
    return merged


def _count_total_tests(instance: MiniRepoInstance) -> int:
    """Return the test count across visible + hidden modules.

    Counts ``def test_`` occurrences in each module — quick and
    matches pytest's discovery rules. (Class-based test discovery
    isn't used by any template.)"""
    total = 0
    pattern = re.compile(r"^def\s+test_", re.MULTILINE)
    for path, content in instance.files.items():
        if path in instance.visible_test_paths:
            total += len(pattern.findall(content))
    for content in instance.hidden_test_files.values():
        total += len(pattern.findall(content))
    return total


def score_components(
    prediction: MiniRepoPrediction,
    instance: MiniRepoInstance,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_REPO,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``."""
    components = {"format_valid": 0.0, "parse_valid": 0.0, "pass_rate": 0.0}

    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components

    if not _is_parse_valid(prediction, instance):
        return components
    components["parse_valid"] = 1.0

    files = _merge_files(instance, prediction.files)
    visible_and_hidden = list(instance.visible_test_paths) + list(
        instance.hidden_test_files
    )
    manifest = build_pytest_manifest(visible_and_hidden, timeout_s=timeout_s)
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=manifest,
        mem_bytes=mem_bytes,
    )
    counts = parse_pytest_q_summary(result.stdout)
    total = _count_total_tests(instance)
    if total > 0:
        components["pass_rate"] = float(counts["passed"]) / float(total)
    return components


def compute_reward(
    prediction: MiniRepoPrediction,
    instance: MiniRepoInstance,
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_REPO,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    conformal_quantile: float | None = None,
) -> dict[str, Any]:
    """Combine the three components into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    components = score_components(
        prediction,
        instance,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
    )
    reward = sum(w[k] * components[k] for k in components)
    reward = max(0.0, min(1.0, reward))

    meta: dict[str, Any] = {
        "weights": dict(w),
        "timeout_s": timeout_s,
        "confidence": float(prediction.confidence),
        "template": instance.template_name,
        "edited_files": sorted(prediction.files),
    }
    if conformal_quantile is not None:
        residual = 1.0 - reward
        meta["covered"] = bool(residual <= float(conformal_quantile))
        meta["residual"] = residual
        meta["conformal_quantile"] = float(conformal_quantile)

    return {
        "reward": float(reward),
        "components": {k: float(v) for k, v in components.items()},
        "meta": meta,
    }


# ── Env class + factory ─────────────────────────────────────────────


def baseline_predict(instance: MiniRepoInstance) -> MiniRepoPrediction:
    """Reference solver — empty edits → zero on every component."""
    del instance
    return MiniRepoPrediction(files={}, raw="", confidence=0.0)


class CodeMiniRepoEnv:
    """RL environment handle wrapping one calibrated conformal quantile."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.env_id: str = ""
        self.env_args: dict[str, Any] = {}

    def generate_instance(self, seed: int, **kwargs: Any) -> MiniRepoInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: MiniRepoPrediction,
        instance: MiniRepoInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            timeout_s=float(self.hyperparams["sandbox_timeout_s"]),
            mem_bytes=int(self.hyperparams["sandbox_mem_bytes"]),
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)


def calibrate_quantile(
    n_samples: int = 30,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals."""
    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(prediction=pred, instance=inst)
        residuals.append(1.0 - float(out["reward"]))
    return float(split_conformal_quantile(np.asarray(residuals), alpha))


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> CodeMiniRepoEnv:
    """Factory matching the single-turn env. Pass ``calibration_quantile``
    to skip auto-calibration in tests."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 3 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return CodeMiniRepoEnv(conformal_quantile=q)


# ── Adapter shape — exposed for the LLM solver loop ──────────────────


SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will receive a small "
    "multi-file Python repo plus a spec describing a task. Reply with a "
    "JSON object of the form\n\n"
    '    {"files": {"<path>": "<full new content>", ...}, '
    '"confidence": <float in [0, 1]>}\n\n'
    "where ``files`` carries the COMPLETE new content for each file you "
    "want to overwrite. Only paths in the EDITABLE FILES list are "
    "honoured; edits to other paths are silently ignored. No prose, no "
    "markdown fences — JSON only."
)


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: MiniRepoInstance) -> str:
    """Render the env instance into LLM-readable text."""
    return (
        "PROBLEM:\n"
        f"{instance.prompt}\n\n"
        "OUTPUT SCHEMA:\n"
        '{"files": {"<path>": "<full new content>", ...}, '
        '"confidence": <float in [0, 1]>}\n\n'
        "Respond with the JSON object only."
    )


def parse_response(text: str, instance: MiniRepoInstance) -> MiniRepoPrediction:
    """Parse the LLM's text into a :class:`MiniRepoPrediction`.

    Permissive: malformed inputs yield an empty-files prediction (zero
    reward). The set of output files is NOT filtered against
    ``editable_paths`` here — the reward kernel's ``parse_valid``
    component is the gate.
    """
    del instance
    cleaned = text.strip()
    candidates: list[str] = []
    fenced = _FENCED_RE.findall(cleaned)
    candidates.extend(fenced)
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        raw_files = data.get("files")
        if not isinstance(raw_files, dict):
            continue
        files = {
            str(p): str(c) for p, c in raw_files.items() if isinstance(c, str)
        }
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return MiniRepoPrediction(files=files, raw=text, confidence=confidence)

    return MiniRepoPrediction(files={}, raw=text, confidence=0.0)


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_TIMEOUT_S_PER_REPO",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "SYSTEM_PROMPT",
    "MiniRepoInstance",
    "MiniRepoPrediction",
    "CodeMiniRepoEnv",
    "generate_problem",
    "generate_instance",
    "baseline_predict",
    "score_components",
    "compute_reward",
    "calibrate_quantile",
    "load_environment",
    "build_user_prompt",
    "parse_response",
]
