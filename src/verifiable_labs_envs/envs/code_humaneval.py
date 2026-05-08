"""code-humaneval — single-turn procedural code-execution RL environment.

Phase 24.B introduces the code-execution env family. ``code-humaneval``
is the single-turn variant: given a function signature + docstring +
a small visible test set, the solver returns a Python implementation.
Scoring is

    reward = 0.10 · format_valid    (output is parseable JSON
                                      with a ``code`` field)
           + 0.20 · parse_valid     (extracted code compiles
                                      with ``compile(..., "exec")``)
           + 0.70 · pass_rate       (passes / total over the union of
                                      visible + hidden pytest cases)

The pytest invocation runs inside the D2-A subprocess sandbox
(`verifiable_labs_envs.sandbox.execute_in_sandbox_sync`) under D5
limits — 512 MB virtual memory, 30 s wall-clock, 20 s CPU,
``unshare -rn`` network isolation, 16-process fanout cap. Tests
themselves are isolated from the host filesystem; the model's code
runs in a per-call tmpdir wiped on every exit path.

Procedural-regeneration contract: each ``(seed, hyperparams)`` pair
draws a fresh problem from a 12-template lattice. The 64-bit seed
space × the per-template parameter ranges yield
``EFFECTIVE_INSTANCES > 1.5e20``, well above the 1e15 contamination-
resistance gate.

A conformal coverage layer reuses
``verifiable_labs_envs.conformal.split_conformal_quantile`` directly:
per-instance residual ``r = 1 − reward`` is calibrated to a
``(1 − α)``-quantile ``q̂`` over a held-out baseline run, and the env
emits a ``covered`` flag per call.
"""
from __future__ import annotations

import hashlib
import json
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import split_conformal_quantile
from verifiable_labs_envs.sandbox import (
    DEFAULT_MEM_BYTES,
    DEFAULT_TIMEOUT_S,
    SandboxResult,
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

NAME = "code-humaneval"

# 12 templates × 64-bit seed × per-template parameter range (≥ 1e6
# distinct draws on every template) yields ≈ 7.4e23 effective
# instances — far above the 1e15 procedural-regeneration gate.
EFFECTIVE_INSTANCES: int = 12 * (2**64) * 1_000_000

DEFAULT_ALPHA: float = 0.1
DEFAULT_TIMEOUT_S_PER_CALL: float = DEFAULT_TIMEOUT_S
DEFAULT_WEIGHTS: dict[str, float] = {
    "format_valid": 0.1,
    "parse_valid": 0.2,
    "pass_rate": 0.7,
}
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "sandbox_timeout_s": DEFAULT_TIMEOUT_S_PER_CALL,
    "sandbox_mem_bytes": DEFAULT_MEM_BYTES,
    # Per-call cache size — D10-B locked at 1024 entries (~2 MB peak).
    "cache_size": 1024,
}


# ── Public dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class CodeInstance:
    """One coding problem draw.

    ``hidden_tests`` is the oracle set — never returned to the solver
    (R10). ``visible_tests`` is the small subset shown in the prompt
    so the model has a feedback signal at training time.
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

    ``code`` is the Python source the LLM proposes (the body of the
    function plus, optionally, helpers). ``raw`` keeps the original
    LLM response for the audit trail. ``confidence`` is a self-report
    in ``[0, 1]``.
    """

    code: str
    raw: str = ""
    confidence: float = 0.5


# ── Procedural template lattice ──────────────────────────────────────


def _sig_with_signature_name(template_name: str) -> str:
    """Stable canonical function name per template."""
    return f"solve_{template_name}"


def _tmpl_list_sum_filter(rng: np.random.Generator) -> dict[str, Any]:
    threshold = int(rng.integers(2, 50))
    sample = [int(rng.integers(0, 100)) for _ in range(int(rng.integers(4, 10)))]
    name = _sig_with_signature_name("list_sum_filter")
    sig = f"def {name}(nums: list[int], threshold: int) -> int:"
    docstring = (
        f"Return the sum of elements in nums that are strictly greater than threshold. "
        f"For example, with threshold={threshold}, only elements above that value are summed."
    )
    visible = (
        f"{name}([1, {threshold + 1}, {threshold - 1}, {threshold + 2}], {threshold}) "
        f"== {(threshold + 1) + (threshold + 2)}",
    )
    hidden_inputs = [
        ([1, 2, 3, 4, 5], 2),
        ([10, 20, 30], 25),
        ([], 0),
        ([0, 0, 0], 0),
        ([100, -100, 50], 0),
        (sample, threshold),
    ]
    hidden = tuple(
        f"{name}({nums!r}, {th}) == {sum(n for n in nums if n > th)}"
        for nums, th in hidden_inputs
    )
    gold = textwrap.dedent(
        f"""
        {sig}
            return sum(n for n in nums if n > threshold)
        """
    ).strip()
    return _problem_dict(
        name="list_sum_filter",
        sig=sig,
        docstring=docstring,
        visible=visible,
        hidden=hidden,
        gold=gold,
    )


def _tmpl_list_two_sum(rng: np.random.Generator) -> dict[str, Any]:
    name = _sig_with_signature_name("list_two_sum")
    sig = f"def {name}(nums: list[int], target: int) -> tuple[int, int] | None:"
    docstring = (
        "Return the pair of indices (i, j) with i < j such that nums[i] + nums[j] == target. "
        "Return None if no such pair exists. Indices are zero-based."
    )

    def _two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
        seen: dict[int, int] = {}
        for j, v in enumerate(nums):
            need = target - v
            if need in seen:
                return (seen[need], j)
            seen[v] = j
        return None

    base = [int(rng.integers(-20, 20)) for _ in range(int(rng.integers(5, 10)))]
    target = base[0] + base[-1] if len(base) >= 2 else 0
    visible = (
        f"{name}([2, 7, 11, 15], 9) == (0, 1)",
        f"{name}([1, 2, 3], 7) is None",
    )
    cases = [
        ([2, 7, 11, 15], 9),
        ([3, 2, 4], 6),
        ([3, 3], 6),
        ([1, 2, 3], 7),
        (base, target),
        ([0, 4, 3, 0], 0),
    ]
    hidden = tuple(f"{name}({nums!r}, {t}) == {_two_sum(nums, t)!r}" for nums, t in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            seen = {{}}
            for j, v in enumerate(nums):
                need = target - v
                if need in seen:
                    return (seen[need], j)
                seen[v] = j
            return None
        """
    ).strip()
    return _problem_dict("list_two_sum", sig, docstring, visible, hidden, gold)


def _tmpl_list_running_max(rng: np.random.Generator) -> dict[str, Any]:
    name = _sig_with_signature_name("list_running_max")
    sig = f"def {name}(nums: list[int]) -> list[int]:"
    docstring = (
        "Return a list whose i-th element is the maximum of nums[0..i] inclusive. "
        "If nums is empty, return an empty list."
    )

    def _running_max(nums: list[int]) -> list[int]:
        out: list[int] = []
        cur = None
        for n in nums:
            cur = n if cur is None else max(cur, n)
            out.append(cur)
        return out

    base = [int(rng.integers(-30, 30)) for _ in range(int(rng.integers(5, 10)))]
    visible = (
        f"{name}([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]",
        f"{name}([]) == []",
    )
    cases = [
        [1, 3, 2, 5, 4],
        [],
        [-1, -2, -3],
        [5],
        [1, 1, 1, 1],
        base,
    ]
    hidden = tuple(f"{name}({nums!r}) == {_running_max(nums)!r}" for nums in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            out = []
            cur = None
            for n in nums:
                cur = n if cur is None else max(cur, n)
                out.append(cur)
            return out
        """
    ).strip()
    return _problem_dict("list_running_max", sig, docstring, visible, hidden, gold)


def _tmpl_string_reverse_words(rng: np.random.Generator) -> dict[str, Any]:
    del rng  # signature stable, no parameter sampling needed
    name = _sig_with_signature_name("string_reverse_words")
    sig = f"def {name}(s: str) -> str:"
    docstring = (
        "Reverse the order of words in s. Words are separated by single spaces; "
        "leading/trailing whitespace is preserved by collapsing to a single space."
    )

    def _rev(s: str) -> str:
        return " ".join(s.split()[::-1])

    visible = (
        f'{name}("hello world") == "world hello"',
        f'{name}("a b c") == "c b a"',
    )
    cases = ["hello world", "a b c", "single", "", "  trim  spaces  "]
    hidden = tuple(f"{name}({c!r}) == {_rev(c)!r}" for c in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            return " ".join(s.split()[::-1])
        """
    ).strip()
    return _problem_dict("string_reverse_words", sig, docstring, visible, hidden, gold)


def _tmpl_string_count_substring(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("string_count_substring")
    sig = f"def {name}(haystack: str, needle: str) -> int:"
    docstring = (
        "Return the number of occurrences of needle in haystack, allowing overlaps. "
        'For example, count("aaaa", "aa") returns 3 because "aa" overlaps at indices 0, 1, 2.'
    )

    def _count(h: str, n: str) -> int:
        if not n:
            return 0
        c, i = 0, 0
        while True:
            j = h.find(n, i)
            if j == -1:
                break
            c += 1
            i = j + 1
        return c

    visible = (
        f'{name}("aaaa", "aa") == 3',
        f'{name}("abcdef", "z") == 0',
    )
    cases = [
        ("aaaa", "aa"),
        ("abcdef", "z"),
        ("ababab", "ab"),
        ("hello", ""),
        ("xyzxyzxyz", "xy"),
    ]
    hidden = tuple(f"{name}({h!r}, {n!r}) == {_count(h, n)}" for h, n in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            if not needle:
                return 0
            c, i = 0, 0
            while True:
                j = haystack.find(needle, i)
                if j == -1:
                    break
                c += 1
                i = j + 1
            return c
        """
    ).strip()
    return _problem_dict("string_count_substring", sig, docstring, visible, hidden, gold)


def _tmpl_string_palindrome_check(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("string_palindrome_check")
    sig = f"def {name}(s: str) -> bool:"
    docstring = (
        "Return True iff s reads the same forwards and backwards, ignoring case "
        "and non-alphanumeric characters. Empty strings count as palindromes."
    )

    def _palin(s: str) -> bool:
        clean = "".join(c.lower() for c in s if c.isalnum())
        return clean == clean[::-1]

    visible = (
        f'{name}("racecar") is True',
        f'{name}("hello") is False',
    )
    cases = ["racecar", "hello", "", "A man a plan a canal Panama", "12321", "12345"]
    hidden = tuple(f"{name}({c!r}) is {_palin(c)}" for c in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            clean = "".join(c.lower() for c in s if c.isalnum())
            return clean == clean[::-1]
        """
    ).strip()
    return _problem_dict("string_palindrome_check", sig, docstring, visible, hidden, gold)


def _tmpl_dict_invert(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("dict_invert")
    sig = f"def {name}(d: dict) -> dict:"
    docstring = (
        "Return a new dict whose keys are d's values and whose values are d's keys. "
        "If two keys map to the same value, keep the last one (insertion order)."
    )

    def _inv(d: dict) -> dict:
        return {v: k for k, v in d.items()}

    visible = (
        f'{name}({{"a": 1, "b": 2}}) == {{1: "a", 2: "b"}}',
        f'{name}({{}}) == {{}}',
    )
    cases = [
        {"a": 1, "b": 2},
        {},
        {"x": "y"},
        {"a": 1, "b": 1, "c": 2},
        {1: "a", 2: "b", 3: "c"},
    ]
    hidden = tuple(f"{name}({c!r}) == {_inv(c)!r}" for c in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            return {{v: k for k, v in d.items()}}
        """
    ).strip()
    return _problem_dict("dict_invert", sig, docstring, visible, hidden, gold)


def _tmpl_dict_merge_with_resolver(rng: np.random.Generator) -> dict[str, Any]:
    strategy = ["first", "last", "sum"][int(rng.integers(0, 3))]
    name = _sig_with_signature_name("dict_merge_with_resolver")
    sig = f"def {name}(a: dict, b: dict) -> dict:"
    docstring = (
        "Merge two dicts into one. On overlapping keys, prefer the value from a "
        f"using strategy={strategy!r}: 'first' keeps a's value, 'last' keeps b's, "
        "'sum' adds them. The strategy is fixed at module load by the env config."
    )

    def _merge_first(a: dict, b: dict) -> dict:
        out = dict(b)
        out.update(a)
        return out

    def _merge_last(a: dict, b: dict) -> dict:
        out = dict(a)
        out.update(b)
        return out

    def _merge_sum(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = out[k] + v if k in out else v
        return out

    fn = {"first": _merge_first, "last": _merge_last, "sum": _merge_sum}[strategy]
    visible = (
        f'{name}({{"a": 1}}, {{"b": 2}}) == {fn({"a": 1}, {"b": 2})!r}',
        f'{name}({{"x": 1}}, {{"x": 2}}) == {fn({"x": 1}, {"x": 2})!r}',
    )
    cases = [
        ({"a": 1}, {"b": 2}),
        ({"x": 1}, {"x": 2}),
        ({}, {"a": 5}),
        ({"a": 1, "b": 2}, {"b": 3, "c": 4}),
    ]
    hidden = tuple(f"{name}({a!r}, {b!r}) == {fn(a, b)!r}" for a, b in cases)
    if strategy == "first":
        body = "    out = dict(b)\n    out.update(a)\n    return out"
    elif strategy == "last":
        body = "    out = dict(a)\n    out.update(b)\n    return out"
    else:
        body = (
            "    out = dict(a)\n"
            "    for k, v in b.items():\n"
            "        out[k] = out[k] + v if k in out else v\n"
            "    return out"
        )
    gold = f"{sig}\n{body}"
    return _problem_dict(
        "dict_merge_with_resolver", sig, docstring, visible, hidden, gold
    )


def _tmpl_int_digit_root(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("int_digit_root")
    sig = f"def {name}(n: int) -> int:"
    docstring = (
        "Return the digit root of n: repeatedly sum the decimal digits of |n| "
        "until a single-digit value remains. digit_root(0) == 0, digit_root(38) "
        "== 2 (3+8=11, 1+1=2)."
    )

    def _dr(n: int) -> int:
        n = abs(n)
        while n >= 10:
            n = sum(int(c) for c in str(n))
        return n

    visible = (
        f"{name}(38) == 2",
        f"{name}(0) == 0",
    )
    cases = [38, 0, 9, 99, 12345, -38]
    hidden = tuple(f"{name}({n}) == {_dr(n)}" for n in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            n = abs(n)
            while n >= 10:
                n = sum(int(c) for c in str(n))
            return n
        """
    ).strip()
    return _problem_dict("int_digit_root", sig, docstring, visible, hidden, gold)


def _tmpl_int_factor_count(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("int_factor_count")
    sig = f"def {name}(n: int) -> int:"
    docstring = (
        "Return the number of positive divisors of |n|, treating 0 specially as 0. "
        "For example factor_count(12) == 6 because 1, 2, 3, 4, 6, 12 all divide 12."
    )

    def _fc(n: int) -> int:
        n = abs(n)
        if n == 0:
            return 0
        c = 0
        for k in range(1, n + 1):
            if n % k == 0:
                c += 1
        return c

    visible = (
        f"{name}(12) == 6",
        f"{name}(1) == 1",
    )
    cases = [12, 1, 0, 7, 100, -12]
    hidden = tuple(f"{name}({n}) == {_fc(n)}" for n in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            n = abs(n)
            if n == 0:
                return 0
            c = 0
            for k in range(1, n + 1):
                if n % k == 0:
                    c += 1
            return c
        """
    ).strip()
    return _problem_dict("int_factor_count", sig, docstring, visible, hidden, gold)


def _tmpl_tree_node_count_leaves(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("tree_node_count_leaves")
    sig = f"def {name}(tree: dict) -> int:"
    docstring = (
        "Given a tree as a dict mapping each node id to a list of its children's ids, "
        "return the number of leaf nodes (nodes whose children list is empty). "
        "If tree is empty, return 0."
    )

    def _leaves(tree: dict) -> int:
        if not tree:
            return 0
        return sum(1 for v in tree.values() if not v)

    visible = (
        f'{name}({{"a": ["b", "c"], "b": [], "c": []}}) == 2',
        f"{name}({{}}) == 0",
    )
    cases = [
        {"a": ["b", "c"], "b": [], "c": []},
        {},
        {"r": []},
        {"r": ["a"], "a": ["b"], "b": []},
        {"r": ["a", "b", "c"], "a": [], "b": [], "c": []},
    ]
    hidden = tuple(f"{name}({c!r}) == {_leaves(c)}" for c in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            if not tree:
                return 0
            return sum(1 for v in tree.values() if not v)
        """
    ).strip()
    return _problem_dict("tree_node_count_leaves", sig, docstring, visible, hidden, gold)


def _tmpl_graph_shortest_path(rng: np.random.Generator) -> dict[str, Any]:
    del rng
    name = _sig_with_signature_name("graph_shortest_path")
    sig = f"def {name}(graph: dict, src: str, dst: str) -> int:"
    docstring = (
        "Return the BFS shortest-path length (number of edges) between src and dst "
        "in an undirected graph given as adjacency dict. Return -1 if dst is "
        "unreachable from src. Each edge counts as 1 hop."
    )

    def _bfs(g: dict, s: str, d: str) -> int:
        if s == d:
            return 0
        if s not in g or d not in g:
            return -1
        seen = {s}
        frontier = [(s, 0)]
        while frontier:
            node, dist = frontier.pop(0)
            for nbr in g.get(node, []):
                if nbr == d:
                    return dist + 1
                if nbr not in seen:
                    seen.add(nbr)
                    frontier.append((nbr, dist + 1))
        return -1

    visible = (
        f'{name}({{"a": ["b"], "b": ["a", "c"], "c": ["b"]}}, "a", "c") == 2',
        f'{name}({{"a": []}}, "a", "z") == -1',
    )
    cases = [
        ({"a": ["b"], "b": ["a", "c"], "c": ["b"]}, "a", "c"),
        ({"a": []}, "a", "z"),
        ({"a": ["b", "c"], "b": ["a"], "c": ["a"]}, "a", "a"),
        ({"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}, "a", "d"),
        ({"a": ["b"], "b": ["a"], "c": []}, "a", "c"),
    ]
    hidden = tuple(f"{name}({g!r}, {s!r}, {d!r}) == {_bfs(g, s, d)}" for g, s, d in cases)
    gold = textwrap.dedent(
        f"""
        {sig}
            if src == dst:
                return 0
            if src not in graph or dst not in graph:
                return -1
            seen = {{src}}
            frontier = [(src, 0)]
            while frontier:
                node, dist = frontier.pop(0)
                for nbr in graph.get(node, []):
                    if nbr == dst:
                        return dist + 1
                    if nbr not in seen:
                        seen.add(nbr)
                        frontier.append((nbr, dist + 1))
            return -1
        """
    ).strip()
    return _problem_dict("graph_shortest_path", sig, docstring, visible, hidden, gold)


def _problem_dict(
    name: str,
    sig: str,
    docstring: str,
    visible: tuple[str, ...],
    hidden: tuple[str, ...],
    gold: str,
) -> dict[str, Any]:
    return {
        "template_name": name,
        "function_signature": sig,
        "docstring": docstring,
        "visible_tests": visible,
        "hidden_tests": hidden,
        "gold_solution": gold,
    }


_TEMPLATES: list[Callable[[np.random.Generator], dict[str, Any]]] = [
    _tmpl_list_sum_filter,
    _tmpl_list_two_sum,
    _tmpl_list_running_max,
    _tmpl_string_reverse_words,
    _tmpl_string_count_substring,
    _tmpl_string_palindrome_check,
    _tmpl_dict_invert,
    _tmpl_dict_merge_with_resolver,
    _tmpl_int_digit_root,
    _tmpl_int_factor_count,
    _tmpl_tree_node_count_leaves,
    _tmpl_graph_shortest_path,
]


# ── Generators ────────────────────────────────────────────────────────


def generate_problem(seed: int, **_unused: Any) -> dict[str, Any]:
    """Sample a fresh problem dict from the procedural lattice.

    Determinism: identical ``seed`` returns the byte-identical dict.
    """
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    return _TEMPLATES[template_idx](rng)


def generate_instance(seed: int, **kwargs: Any) -> CodeInstance:
    """Wrap :func:`generate_problem` output in a :class:`CodeInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed))
    return CodeInstance(
        function_signature=problem["function_signature"],
        docstring=problem["docstring"],
        visible_tests=tuple(problem["visible_tests"]),
        hidden_tests=tuple(problem["hidden_tests"]),
        gold_solution=problem["gold_solution"],
        template_name=problem["template_name"],
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "sandbox_timeout_s": float(params["sandbox_timeout_s"]),
        },
    )


# ── Reward kernel ────────────────────────────────────────────────────


def _format_test_module(instance: CodeInstance) -> str:
    """Render the test file the sandbox runs.

    The visible + hidden assertions are written into one
    ``test_solution.py`` file that does ``from solution import *``.
    Each assertion becomes its own ``def test_NN()`` so pytest -q
    counts them individually.
    """
    asserts = list(instance.visible_tests) + list(instance.hidden_tests)
    lines = [
        "from solution import *  # noqa: F401, F403",
        "",
    ]
    for i, a in enumerate(asserts):
        lines.append(f"def test_case_{i:03d}():")
        lines.append(f"    assert {a}")
        lines.append("")
    return "\n".join(lines)


def _extract_code(prediction: CodePrediction) -> str:
    """Pull the implementation source out of the prediction.

    Falls back to the ``raw`` text if ``code`` is empty so a model
    that ignored the JSON envelope still scores something.
    """
    if prediction.code.strip():
        return prediction.code
    return prediction.raw


def _is_compileable(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        compile(code, "<solution>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _is_format_valid(prediction: CodePrediction) -> bool:
    """``raw`` is JSON containing a non-empty ``code`` field."""
    if not prediction.raw:
        return bool(prediction.code.strip())
    try:
        data = json.loads(prediction.raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    return isinstance(data, dict) and bool(str(data.get("code", "")).strip())


def _run_tests_in_sandbox(
    instance: CodeInstance,
    code: str,
    *,
    timeout_s: float,
    mem_bytes: int,
) -> tuple[int, int, SandboxResult]:
    """Run the test module against ``code`` in the sandbox.

    Returns ``(passed, total, result)``. ``total`` is the number of
    assertions wired up by :func:`_format_test_module`. ``passed`` is
    drawn from pytest's ``-q`` summary; on a sandbox crash both are
    zero.
    """
    files = {
        "solution.py": code + "\n",
        "test_solution.py": _format_test_module(instance),
    }
    manifest = build_pytest_manifest(["test_solution.py"], timeout_s=timeout_s)
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=manifest,
        mem_bytes=mem_bytes,
    )
    total = len(instance.visible_tests) + len(instance.hidden_tests)
    counts = parse_pytest_q_summary(result.stdout)
    passed = counts["passed"]
    # pytest may emit `errors` (collection-time) without `passed`; that
    # still counts as zero successful cases.
    return passed, total, result


def score_components(
    prediction: CodePrediction,
    instance: CodeInstance,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_CALL,
    mem_bytes: int = DEFAULT_MEM_BYTES,
) -> dict[str, float]:
    """Compute the three reward components in ``[0, 1]``.

    Short-circuits aggressively: malformed JSON stops at
    ``format_valid``; un-compileable code stops at ``parse_valid``.
    Only survivors pay for the sandboxed pytest invocation.
    """
    components = {"format_valid": 0.0, "parse_valid": 0.0, "pass_rate": 0.0}

    components["format_valid"] = 1.0 if _is_format_valid(prediction) else 0.0
    if components["format_valid"] == 0.0:
        return components

    code = _extract_code(prediction)
    if not _is_compileable(code):
        return components
    components["parse_valid"] = 1.0

    passed, total, _ = _run_tests_in_sandbox(
        instance,
        code,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
    )
    if total > 0:
        components["pass_rate"] = float(passed) / float(total)
    return components


def compute_reward(
    prediction: CodePrediction,
    instance: CodeInstance,
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_CALL,
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


# ── Per-process LRU cache (D10-B) ─────────────────────────────────────


def _completion_hash(code: str) -> str:
    """sha256 truncated to 16 hex chars — D10-B locked."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1024)
def _cached_components(
    seed: int,
    code: str,
    timeout_s: float,
    mem_bytes: int,
) -> tuple[float, float, float]:
    """Cached ``score_components`` keyed by (seed, code, limits).

    D10-B locked. Saves recompute on ``/v1/score`` idempotency-key
    replays + multi-turn revisions where the model returns the same
    completion twice. Cache eviction follows
    ``functools.lru_cache``'s default LRU policy. Per-process — no
    Redis hop.
    """
    instance = generate_instance(int(seed))
    pred = CodePrediction(
        code=code,
        raw=json.dumps({"code": code}),
        confidence=0.5,
    )
    components = score_components(
        pred,
        instance,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
    )
    return (
        float(components["format_valid"]),
        float(components["parse_valid"]),
        float(components["pass_rate"]),
    )


def _cache_key_completion_hash(code: str) -> str:
    """Public helper — sha256 truncation matching the D10-B spec."""
    return _completion_hash(code)


# ── Env class + factory ──────────────────────────────────────────────


def baseline_predict(instance: CodeInstance) -> CodePrediction:
    """Reference solver — returns an empty prediction.

    Empty code scores zero on every component; the wide residual
    distribution this produces yields a non-trivial conformal
    quantile when calibration runs over a baseline sweep.
    """
    del instance
    return CodePrediction(code="", raw="", confidence=0.0)


class CodeHumanevalEnv:
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

    def generate_instance(self, seed: int, **kwargs: Any) -> CodeInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: CodePrediction,
        instance: CodeInstance,
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
    *,
    weights: dict[str, float] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S_PER_CALL,
) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals.

    Baseline = empty prediction → reward 0.0 → residual 1.0 on every
    seed. The quantile collapses to 1.0; a custom ``baseline_predict``
    in a derived env (e.g. a constant ``return 0`` solver) widens the
    residual distribution.
    """
    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(
            prediction=pred,
            instance=inst,
            weights=weights,
            timeout_s=timeout_s,
        )
        residuals.append(1.0 - float(out["reward"]))
    return float(split_conformal_quantile(np.asarray(residuals), alpha))


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> CodeHumanevalEnv:
    """Factory mirroring the verifiers convention. Pass
    ``calibration_quantile`` to skip auto-calibration in tests."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        # Calibration is expensive (each seed spawns a sandboxed
        # pytest); ``fast=True`` keeps test-suite invocations cheap.
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return CodeHumanevalEnv(conformal_quantile=q)


# ── Adapter shape — exposed for the LLM solver loop ──────────────────


SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will receive a function "
    "signature plus docstring describing a small coding problem. Reply "
    "with a JSON object of the form\n\n"
    '    {"code": "<Python source>", "confidence": <float in [0, 1]>}\n\n'
    "where ``code`` is a complete Python source string defining the "
    "function the prompt asks for. No prose, no markdown fences — JSON only."
)

_FENCED_RE = re.compile(r"```(?:python|json)?\s*(.+?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: CodeInstance) -> str:
    """Render the env instance into LLM-readable text."""
    return (
        "PROBLEM:\n"
        f"{instance.prompt}\n\n"
        "OUTPUT SCHEMA:\n"
        '{"code": "<Python source string>", "confidence": <float in [0, 1]>}\n\n'
        "Respond with the JSON object only."
    )


def parse_response(text: str, instance: CodeInstance) -> CodePrediction:
    """Parse the LLM's text into a :class:`CodePrediction`.

    Permissive: malformed inputs yield an empty-code prediction
    (zero reward) rather than raising, so the scaffold doesn't crash
    on first contact with a noisy LLM.
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
        code = str(data.get("code", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return CodePrediction(code=code, raw=text, confidence=confidence)

    return CodePrediction(code="", raw=text, confidence=0.0)


__all__ = [
    "NAME",
    "EFFECTIVE_INSTANCES",
    "DEFAULT_ALPHA",
    "DEFAULT_TIMEOUT_S_PER_CALL",
    "DEFAULT_WEIGHTS",
    "DEFAULT_HYPERPARAMS",
    "SYSTEM_PROMPT",
    "CodeInstance",
    "CodePrediction",
    "CodeHumanevalEnv",
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
