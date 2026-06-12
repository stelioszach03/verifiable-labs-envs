"""Shared primitives for long-context envs (Phase 27.B).

PHASE_27_PLAN.md locks four pieces of machinery the three
long-context envs (`long-context-needle`,
`long-context-synthesis`, `long-context-reasoning`) all consume:

1. **Procedural corpus generator** (D2-A): :func:`generate_corpus`
   builds a multi-document blob targeting a token budget, sampled
   from one of 10 topic templates × 64-bit seed.
2. **Needle injection** (D4-D): :func:`inject_needle` (single-needle
   position-varied), :func:`inject_multiple_needles` (count-varied),
   :func:`inject_distractors` (true + decoy needles).
3. **Multi-hop chain corpus** (D9-A): :func:`build_chain_corpus`
   embeds a fact chain across documents and returns the gold chain
   doc-id list for the reasoning env's verifier.
4. **Verification helpers** (D3-D): :func:`exact_match`,
   :func:`numeric_match`, :func:`token_f1` for the per-env reward
   shapes.

All primitives are pure-Python (standard library + numpy +
``tiktoken`` for token counting). No subprocess sandbox — long-
context envs score in-process via string match / token-F1.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# ── D5 limits (locked) ────────────────────────────────────────────────


DEFAULT_TEST_TOKENS: int = 4_000
SANDBOX_TEST_TOKENS: int = 16_000
DEFAULT_MAX_TOKENS: int = 128_000
DEFAULT_DOCUMENT_COUNT: int = 8
DEFAULT_MAX_CORPUS_BYTES: int = 64 * 1024 * 1024  # 64 MB


# ── Tokeniser wrapper ────────────────────────────────────────────────


def _get_encoder():
    """Lazy-load ``tiktoken cl100k_base``.

    Cached on the function so subsequent calls reuse the same encoder
    instance — encoder construction is non-trivial (BPE table load).
    """
    if _get_encoder._cache is None:
        import tiktoken  # noqa: PLC0415

        _get_encoder._cache = tiktoken.get_encoding("cl100k_base")
    return _get_encoder._cache


_get_encoder._cache = None  # type: ignore[attr-defined]


def count_tokens(text: str) -> int:
    """Return the ``cl100k_base`` token count of ``text``.

    Locked tokeniser per D5. Provider-agnostic — non-OpenAI callers
    can substitute their own tokeniser without affecting the env's
    reward shape (rewards are measured on text answers, not token
    counts).
    """
    if not text:
        return 0
    enc = _get_encoder()
    return len(enc.encode(text, disallowed_special=()))


# ── Document + Corpus dataclasses ────────────────────────────────────


@dataclass(frozen=True)
class Document:
    """One document in a procedurally generated corpus."""

    id: int
    title: str
    body: str

    def render(self, separator_template: str = "---DOCUMENT {id}: {title}---") -> str:
        return f"{separator_template.format(id=self.id, title=self.title)}\n\n{self.body}"


@dataclass(frozen=True)
class Corpus:
    """A frozen collection of procedurally generated documents."""

    documents: tuple[Document, ...]
    seed: int

    def render_prompt(
        self,
        question: str,
        *,
        separator_template: str = "---DOCUMENT {id}: {title}---",
        question_prefix: str = "QUESTION:\n",
    ) -> str:
        """Compose the full prompt blob (documents + question)."""
        parts = [doc.render(separator_template=separator_template) for doc in self.documents]
        parts.append(f"{question_prefix}{question}")
        return "\n\n".join(parts)

    def total_tokens(self) -> int:
        """Token count of the joined corpus body (no question, no separators)."""
        joined = "\n\n".join(doc.body for doc in self.documents)
        return count_tokens(joined)

    def with_documents(self, documents: tuple[Document, ...]) -> Corpus:
        """Return a new Corpus with the given documents."""
        return Corpus(documents=documents, seed=self.seed)


# ── Procedural topic templates (D2-A) ────────────────────────────────


# Reusable token pools. Kept small so the lattice generates visibly
# distinct prose across seeds without combinatorial explosion of memory.
_PROPER_NAMES: tuple[str, ...] = (
    "Adelard", "Bertrand", "Camille", "Dahlia", "Elena", "Fernando",
    "Genevieve", "Harold", "Iris", "Jasper", "Klara", "Leander",
    "Mireille", "Nikola", "Otto", "Petra", "Quirin", "Rosalind",
    "Soren", "Thalia", "Ursula", "Vincenzo", "Wilhelmina", "Xavier",
    "Yelena", "Zachariah",
)
_CITIES: tuple[str, ...] = (
    "Aldermont", "Brookhaven", "Cinderwell", "Drakemoor", "Elvenhall",
    "Foxglove", "Greyhaven", "Hollowford", "Ironreach", "Junipersend",
    "Kelmoor", "Lyondale", "Marrowdeep", "Northgate", "Oakhaven",
    "Pelican", "Quintus", "Ravenshold", "Sevenstone", "Thornwick",
)
_INSTITUTIONS: tuple[str, ...] = (
    "Caldwell Foundation", "Drumlin Institute", "Eastvale Society",
    "Fenwick Trust", "Glenwood Academy", "Helleberg Bureau",
    "Iverson College", "Jovian Council", "Kettering Press",
    "Larkspur Authority", "Marquand Centre", "Nethersen Society",
)
_FILLER_VERBS: tuple[str, ...] = (
    "established", "examined", "documented", "observed", "outlined",
    "described", "evaluated", "summarised", "highlighted", "presented",
)
_TRANSITIONS: tuple[str, ...] = (
    "Furthermore,", "However,", "In addition,", "Notably,",
    "Subsequently,", "Conversely,", "Specifically,", "Meanwhile,",
)


def _sample(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    return pool[int(rng.integers(0, len(pool)))]


def _bio_sentence(rng: np.random.Generator) -> str:
    name = _sample(rng, _PROPER_NAMES)
    city = _sample(rng, _CITIES)
    inst = _sample(rng, _INSTITUTIONS)
    verb = _sample(rng, _FILLER_VERBS)
    year = int(rng.integers(1880, 1980))
    return f"{name}, born in {city}, {verb} the {inst} in {year}."


def _science_sentence(rng: np.random.Generator) -> str:
    name = _sample(rng, _PROPER_NAMES)
    inst = _sample(rng, _INSTITUTIONS)
    metric = ("yield", "throughput", "absorption", "scattering", "deflection")[
        int(rng.integers(0, 5))
    ]
    return (
        f"{name} of the {inst} reported a measured {metric} of "
        f"{int(rng.integers(10, 9000))} units in the experiment."
    )


def _news_sentence(rng: np.random.Generator) -> str:
    city = _sample(rng, _CITIES)
    name = _sample(rng, _PROPER_NAMES)
    return (
        f"In {city}, regional officials announced that {name} would "
        f"oversee the new initiative beginning next month."
    )


def _product_sentence(rng: np.random.Generator) -> str:
    name = _sample(rng, _PROPER_NAMES)
    obj = ("device", "module", "instrument", "kit", "appliance", "platform")[
        int(rng.integers(0, 6))
    ]
    return (
        f"The {obj}, used by {name}, performed reliably under "
        f"{int(rng.integers(50, 500))} hours of continuous testing."
    )


def _technical_sentence(rng: np.random.Generator) -> str:
    return (
        f"Section {int(rng.integers(1, 30))}.{int(rng.integers(1, 20))} "
        f"specifies the {_sample(rng, _FILLER_VERBS)} parameters for the "
        f"{_sample(rng, _INSTITUTIONS).lower()} compliance regime."
    )


def _legal_sentence(rng: np.random.Generator) -> str:
    name = _sample(rng, _PROPER_NAMES)
    return (
        f"The party of the first part, hereinafter referred to as "
        f"{name}, agrees to {_sample(rng, _FILLER_VERBS)} the terms "
        f"set forth in clause {int(rng.integers(1, 50))}."
    )


def _historical_sentence(rng: np.random.Generator) -> str:
    name = _sample(rng, _PROPER_NAMES)
    city = _sample(rng, _CITIES)
    return (
        f"During the autumn of {int(rng.integers(1500, 1900))}, "
        f"{name} departed from {city} on the expedition that would "
        f"later define the era."
    )


def _recipe_sentence(rng: np.random.Generator) -> str:
    return (
        f"Combine {int(rng.integers(2, 12))} measures of the primary "
        f"ingredient with {int(rng.integers(1, 5))} measures of the "
        f"secondary; rest for {int(rng.integers(5, 60))} minutes."
    )


def _travel_sentence(rng: np.random.Generator) -> str:
    a = _sample(rng, _CITIES)
    b = _sample(rng, _CITIES)
    while b == a:
        b = _sample(rng, _CITIES)
    return (
        f"The journey from {a} to {b} crosses three valleys and a "
        f"narrow ridge that locals call the {_sample(rng, _PROPER_NAMES)} Pass."
    )


def _interview_sentence(rng: np.random.Generator) -> str:
    name = _sample(rng, _PROPER_NAMES)
    return (
        f'"{_sample(rng, _TRANSITIONS).rstrip(",")} we never expected '
        f'such results," said {name} during the recorded interview.'
    )


_TEMPLATES: dict[str, Any] = {
    "bio_article": {
        "title_pattern": "Biographical entry: {name}",
        "sentence_fn": _bio_sentence,
    },
    "science_abstract": {
        "title_pattern": "Abstract: research at {inst}",
        "sentence_fn": _science_sentence,
    },
    "news_report": {
        "title_pattern": "Regional report: {city}",
        "sentence_fn": _news_sentence,
    },
    "product_review": {
        "title_pattern": "Field test review",
        "sentence_fn": _product_sentence,
    },
    "technical_manual": {
        "title_pattern": "Technical manual section",
        "sentence_fn": _technical_sentence,
    },
    "legal_document": {
        "title_pattern": "Legal agreement clause set",
        "sentence_fn": _legal_sentence,
    },
    "historical_summary": {
        "title_pattern": "Historical summary",
        "sentence_fn": _historical_sentence,
    },
    "recipe_collection": {
        "title_pattern": "Procedural recipe entry",
        "sentence_fn": _recipe_sentence,
    },
    "travel_log": {
        "title_pattern": "Travel log entry",
        "sentence_fn": _travel_sentence,
    },
    "interview_transcript": {
        "title_pattern": "Interview transcript excerpt",
        "sentence_fn": _interview_sentence,
    },
}

TOPIC_TEMPLATE_NAMES: tuple[str, ...] = tuple(_TEMPLATES.keys())


def _render_title(template_name: str, rng: np.random.Generator) -> str:
    pattern = _TEMPLATES[template_name]["title_pattern"]
    return pattern.format(
        name=_sample(rng, _PROPER_NAMES),
        inst=_sample(rng, _INSTITUTIONS),
        city=_sample(rng, _CITIES),
    )


def _generate_document_body(
    template_name: str,
    target_tokens: int,
    rng: np.random.Generator,
) -> str:
    """Generate filler text targeting ``target_tokens``.

    Two-phase generation:

    1. **Coarse phase** — emit batches of 8 sentences until the
       running total reaches 80% of target. Cheap on `count_tokens`.
    2. **Fine phase** — emit one sentence at a time and recheck after
       each. Stops as soon as the total reaches ``target_tokens``.

    This bounds the actual-vs-target overshoot to ~5% across the
    full 4 K – 128 K range without exploding sentence-generation
    cost (the coarse phase amortises the bulk of the work).
    """
    sentence_fn = _TEMPLATES[template_name]["sentence_fn"]
    sentences: list[str] = []
    safety = max(100, target_tokens)
    iterations = 0
    coarse_target = int(target_tokens * 0.80)

    # Phase 1 — coarse batches until we cross the 80% threshold.
    while iterations < safety:
        for _ in range(8):
            sentences.append(sentence_fn(rng))
            if int(rng.integers(0, 6)) == 0:
                sentences.append(_sample(rng, _TRANSITIONS))
        if count_tokens(" ".join(sentences)) >= coarse_target:
            break
        iterations += 8

    # Phase 2 — fine-grained, one sentence at a time.
    while iterations < safety:
        sentences.append(sentence_fn(rng))
        if int(rng.integers(0, 6)) == 0:
            sentences.append(_sample(rng, _TRANSITIONS))
        if count_tokens(" ".join(sentences)) >= target_tokens:
            break
        iterations += 1
    return " ".join(sentences)


def generate_corpus(
    seed: int,
    *,
    target_tokens: int = DEFAULT_TEST_TOKENS,
    document_count: int = DEFAULT_DOCUMENT_COUNT,
    template: str | None = None,
) -> Corpus:
    """Build a procedural corpus of ``document_count`` documents whose
    joined body lands in the ``[target_tokens × 0.85, target_tokens]``
    range.

    If ``template`` is None, each document samples its own template
    independently (heterogeneous corpus). If given, all documents use
    the same template.
    """
    if document_count < 1:
        raise ValueError(f"document_count must be >= 1; got {document_count}")
    if target_tokens < 64:
        raise ValueError(f"target_tokens must be >= 64; got {target_tokens}")

    rng = np.random.default_rng(int(seed))
    per_doc_tokens = max(64, target_tokens // document_count)
    documents: list[Document] = []
    for i in range(document_count):
        if template is None:
            tmpl = TOPIC_TEMPLATE_NAMES[int(rng.integers(0, len(TOPIC_TEMPLATE_NAMES)))]
        else:
            tmpl = template
        title = _render_title(tmpl, rng)
        body = _generate_document_body(tmpl, per_doc_tokens, rng)
        documents.append(Document(id=i, title=title, body=body))
    return Corpus(documents=tuple(documents), seed=int(seed))


# ── Needle injection (D4-D) ─────────────────────────────────────────


@dataclass(frozen=True)
class NeedleAnchor:
    """Records where a needle was placed in a corpus."""

    document_id: int
    char_offset: int
    needle_text: str
    is_distractor: bool = False


PositionMode = Literal["start", "middle", "end", "random"]


def _resolve_offset(
    body_len: int, position: PositionMode, rng: np.random.Generator
) -> int:
    """Return a character offset within ``body_len`` for the given
    position semantics."""
    if body_len <= 0:
        return 0
    if position == "start":
        return int(body_len * 0.05)
    if position == "middle":
        return int(body_len * 0.5)
    if position == "end":
        return max(0, body_len - max(1, int(body_len * 0.05)))
    if position == "random":
        return int(rng.integers(0, body_len))
    raise ValueError(f"unknown position: {position!r}")


def _inject_at_offset(body: str, offset: int, needle: str) -> str:
    """Insert ``needle`` at ``offset`` in ``body``. Pads with a sentence
    boundary so the needle reads as its own sentence."""
    offset = max(0, min(len(body), offset))
    # Snap to the nearest sentence boundary (period/space) so the
    # needle doesn't split a word.
    while offset < len(body) and body[offset] not in (" ", "."):
        offset += 1
    while offset > 0 and body[offset - 1] not in (" ", "."):
        offset -= 1
    needle_sentence = needle.strip()
    if not needle_sentence.endswith("."):
        needle_sentence = needle_sentence + "."
    if offset == 0:
        return f"{needle_sentence} {body}".strip()
    if offset >= len(body):
        prefix = body.rstrip()
        if prefix and not prefix.endswith("."):
            prefix = prefix + "."
        return f"{prefix} {needle_sentence}".strip()
    prefix = body[:offset].rstrip()
    suffix = body[offset:].lstrip()
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."
    return f"{prefix} {needle_sentence} {suffix}".strip()


def inject_needle(
    corpus: Corpus,
    *,
    needle_text: str,
    position: PositionMode = "random",
    rng: np.random.Generator,
    document_id: int | None = None,
) -> tuple[Corpus, NeedleAnchor]:
    """Insert ``needle_text`` into one document.

    Returns the updated corpus + a :class:`NeedleAnchor` recording the
    target document id, the character offset of the insertion, and
    the needle text. ``document_id`` defaults to a uniformly sampled
    document id; pass an explicit id to pin the target.
    """
    if not needle_text or not needle_text.strip():
        raise ValueError("needle_text must be non-empty")
    if document_id is None:
        document_id = int(rng.integers(0, len(corpus.documents)))
    if not (0 <= document_id < len(corpus.documents)):
        raise ValueError(
            f"document_id {document_id} out of range [0, {len(corpus.documents)})"
        )

    new_documents: list[Document] = list(corpus.documents)
    target = new_documents[document_id]
    offset = _resolve_offset(len(target.body), position, rng)
    new_body = _inject_at_offset(target.body, offset, needle_text)
    new_documents[document_id] = Document(
        id=target.id, title=target.title, body=new_body,
    )
    new_corpus = corpus.with_documents(tuple(new_documents))
    anchor = NeedleAnchor(
        document_id=document_id,
        char_offset=offset,
        needle_text=needle_text.strip(),
        is_distractor=False,
    )
    return new_corpus, anchor


def inject_multiple_needles(
    corpus: Corpus,
    *,
    needles: list[str],
    rng: np.random.Generator,
    position: PositionMode = "random",
) -> tuple[Corpus, tuple[NeedleAnchor, ...]]:
    """Insert N needles, one per distinct document at sampled positions.

    Requires ``len(needles) <= len(corpus.documents)``.
    """
    if len(needles) > len(corpus.documents):
        raise ValueError(
            f"too many needles ({len(needles)}) for "
            f"{len(corpus.documents)} documents"
        )
    if len(needles) == 0:
        return corpus, ()
    doc_ids = list(rng.permutation(len(corpus.documents))[: len(needles)])
    cur = corpus
    anchors: list[NeedleAnchor] = []
    for needle, doc_id in zip(needles, doc_ids, strict=True):
        cur, anchor = inject_needle(
            cur,
            needle_text=needle,
            position=position,
            rng=rng,
            document_id=int(doc_id),
        )
        anchors.append(anchor)
    return cur, tuple(anchors)


def inject_distractors(
    corpus: Corpus,
    *,
    true_needles: list[str],
    distractor_needles: list[str],
    rng: np.random.Generator,
    position: PositionMode = "random",
) -> tuple[Corpus, tuple[NeedleAnchor, ...]]:
    """Insert true needles + distractor needles into different documents.

    Returns combined anchors; the ``is_distractor`` flag separates the
    two classes. The reasoning env's verifier uses this to confirm
    the model picked the true needle, not a decoy.
    """
    total = len(true_needles) + len(distractor_needles)
    if total > len(corpus.documents):
        raise ValueError(
            f"too many needles ({total}) for {len(corpus.documents)} documents"
        )
    if total == 0:
        return corpus, ()
    doc_ids = list(rng.permutation(len(corpus.documents))[:total])
    cur = corpus
    anchors: list[NeedleAnchor] = []
    idx = 0
    for needle in true_needles:
        cur, anchor = inject_needle(
            cur, needle_text=needle, position=position, rng=rng,
            document_id=int(doc_ids[idx]),
        )
        anchors.append(
            NeedleAnchor(
                document_id=anchor.document_id,
                char_offset=anchor.char_offset,
                needle_text=anchor.needle_text,
                is_distractor=False,
            )
        )
        idx += 1
    for needle in distractor_needles:
        cur, anchor = inject_needle(
            cur, needle_text=needle, position=position, rng=rng,
            document_id=int(doc_ids[idx]),
        )
        anchors.append(
            NeedleAnchor(
                document_id=anchor.document_id,
                char_offset=anchor.char_offset,
                needle_text=anchor.needle_text,
                is_distractor=True,
            )
        )
        idx += 1
    return cur, tuple(anchors)


# ── Multi-hop chain corpus (D9-A) ───────────────────────────────────


@dataclass(frozen=True)
class ChainFact:
    """One fact in a multi-hop chain.

    ``text`` is the surface form embedded into a document.
    ``document_id`` records which document the fact landed in (set
    after :func:`build_chain_corpus` placement).
    ``is_distractor`` flags decoy facts placed alongside the true
    chain.
    """

    text: str
    document_id: int
    is_distractor: bool = False


@dataclass(frozen=True)
class ChainCorpus:
    """Corpus + chain-fact metadata."""

    corpus: Corpus
    facts: tuple[ChainFact, ...] = field(default_factory=tuple)

    def gold_chain_doc_ids(self) -> tuple[int, ...]:
        """Return the document IDs that carry the gold chain
        (excluding distractors), in chain order."""
        return tuple(f.document_id for f in self.facts if not f.is_distractor)


def build_chain_corpus(
    seed: int,
    *,
    chain_facts: list[str],
    distractor_facts: list[str] = (),
    document_count: int = 8,
    target_tokens: int = DEFAULT_TEST_TOKENS,
) -> ChainCorpus:
    """Build a corpus with the chain facts embedded across documents.

    Chain facts are placed in the order given (so the gold chain doc
    ids are recoverable by index). Distractor facts are placed in
    different documents than the chain facts.
    """
    if not chain_facts:
        raise ValueError("chain_facts must be non-empty")
    total = len(chain_facts) + len(list(distractor_facts))
    if total > document_count:
        raise ValueError(
            f"too many facts ({total}) for {document_count} documents"
        )
    base = generate_corpus(
        seed,
        target_tokens=target_tokens,
        document_count=document_count,
    )
    rng = np.random.default_rng(int(seed) + 1)
    cur, anchors = inject_distractors(
        base,
        true_needles=chain_facts,
        distractor_needles=list(distractor_facts),
        rng=rng,
        position="random",
    )
    facts: list[ChainFact] = []
    # Anchors come back with chain facts first (in order), then distractors.
    for anchor in anchors:
        facts.append(
            ChainFact(
                text=anchor.needle_text,
                document_id=anchor.document_id,
                is_distractor=anchor.is_distractor,
            )
        )
    return ChainCorpus(corpus=cur, facts=tuple(facts))


# ── Verification helpers (D3-D) ─────────────────────────────────────


def exact_match(predicted: str, gold: str, *, case_sensitive: bool = False) -> bool:
    """Substring match — D3-A.

    Whitespace-stripped comparison; case-insensitive by default.
    """
    if not predicted or not gold:
        return False
    p = predicted.strip()
    g = gold.strip()
    if not case_sensitive:
        p = p.lower()
        g = g.lower()
    return g in p


def numeric_match(predicted: str, gold: float, *, tol: float = 1e-6) -> bool:
    """Numeric-answer match with tolerance — D9 reasoning env."""
    if predicted is None:
        return False
    text = str(predicted).strip()
    if not text:
        return False
    # Pull the first numeric token (handles "the answer is 42" responses).
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return False
    try:
        return abs(float(match.group(0)) - float(gold)) <= tol
    except (TypeError, ValueError):
        return False


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def _normalise_for_f1(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace."""
    cleaned = _PUNCT_RE.sub(" ", text or "")
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned.lower()


def token_f1(predicted: str, gold: str) -> float:
    """SQuAD-style token-F1 — D3-C, synthesis env."""
    pred_tokens = _normalise_for_f1(predicted).split()
    gold_tokens = _normalise_for_f1(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    if not common:
        return 0.0
    common_count = sum(common.values())
    precision = common_count / len(pred_tokens)
    recall = common_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


__all__ = [
    "DEFAULT_DOCUMENT_COUNT",
    "DEFAULT_MAX_CORPUS_BYTES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEST_TOKENS",
    "SANDBOX_TEST_TOKENS",
    "TOPIC_TEMPLATE_NAMES",
    "ChainCorpus",
    "ChainFact",
    "Corpus",
    "Document",
    "NeedleAnchor",
    "PositionMode",
    "build_chain_corpus",
    "count_tokens",
    "exact_match",
    "generate_corpus",
    "inject_distractors",
    "inject_multiple_needles",
    "inject_needle",
    "numeric_match",
    "token_f1",
]
