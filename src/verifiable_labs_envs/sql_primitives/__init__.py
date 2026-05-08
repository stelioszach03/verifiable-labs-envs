"""Shared SQL primitives for the SQL env family (Phase 26.B).

PHASE_26_PLAN.md locks four pieces of machinery the two SQL envs
(`sql-single-turn` and `sql-multiturn`) consume:

1. **Sandbox** (D2-A): :func:`execute_query_sync` runs a model
   query against an ephemeral ``sqlite3.connect(":memory:")``
   connection seeded with the instance's CREATE/INSERT statements.
   D5 limits — 10 s wall-clock + 10 000-row cap + 32 KB query +
   64-deep CTE nesting — are enforced inline.
2. **Schema generator** (D3-A): :class:`Schema` + 8 procedural
   templates (`single_table_filter`, `single_table_aggregate`,
   `two_table_join`, `three_table_join`, `groupby_having`,
   `subquery_filter`, `cte_aggregate`, `date_arithmetic`).
3. **Read-only gate** (D8-C): the first non-comment / non-paren
   token of the submitted query must be ``SELECT`` / ``WITH`` /
   ``EXPLAIN``. DML and DDL are rejected at parse time before the
   sandbox runs the query.
4. **Result-set comparator** (D4-A + D9): :func:`compare_result_sets`
   matches gold and predicted rows under the locked element-wise
   rules — 1 × 10⁻⁶ float tolerance, ``NULL == NULL``, ordered
   comparison only when the gold query has an explicit ``ORDER BY``.

These primitives are pure-Python (standard library + numpy) — no
subprocess, no external deps. The module is the single source of
truth for the SQL-env reward kernel and the multi-turn rollout's
verifier feedback.
"""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── D5 limits (locked) ────────────────────────────────────────────────


DEFAULT_MAX_ROWS: int = 10_000
DEFAULT_TIMEOUT_S: float = 10.0
DEFAULT_MAX_QUERY_BYTES: int = 32 * 1024
DEFAULT_CTE_DEPTH_CAP: int = 64
DEFAULT_FLOAT_TOL: float = 1e-6


# ── D8-C read-only gate ──────────────────────────────────────────────


_ALLOWED_FIRST_TOKENS: frozenset[str] = frozenset({"SELECT", "WITH", "EXPLAIN"})

# Tokens that explicitly identify DML/DDL operations. The first-token
# gate already rejects anything outside the allow-list, but we keep
# this set so the rejection error messages can be specific.
_DML_DDL_TOKENS: frozenset[str] = frozenset({
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "PRAGMA",
    "ATTACH",
    "DETACH",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
})


# ── D9 forbidden constructs ──────────────────────────────────────────


# Any non-deterministic SQL function name (case-insensitive). The
# tokeniser strips word-boundary tokens and refuses if any of these
# appear.
_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset({
    "RANDOM",
    "RANDOMBLOB",
    "CURRENT_TIMESTAMP",
    "CURRENT_TIME",
    "CURRENT_DATE",
})


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class SqlExecution:
    """Outcome of one sandboxed query.

    ``rows`` is a tuple of value tuples (the natural sqlite3 fetch
    shape); ``column_names`` mirrors the cursor description's first
    fields. ``truncated`` indicates the row cap was reached. On
    error, ``rows`` is empty and ``error`` carries the message.
    """

    success: bool
    rows: tuple[tuple[Any, ...], ...]
    column_names: tuple[str, ...]
    rowcount: int
    truncated: bool
    error: str | None
    elapsed_ms: int


@dataclass(frozen=True)
class Schema:
    """Procedural schema spec for one instance.

    ``create_statements`` is the canonical CREATE TABLE list (sorted
    by table name for hash stability). ``seed_statements`` is the
    INSERT batch the sandbox replays. ``column_names_by_table`` is
    a frozen mapping for prompt rendering.
    """

    create_statements: tuple[str, ...]
    seed_statements: tuple[str, ...]
    table_names: tuple[str, ...]
    column_names_by_table: dict[str, tuple[str, ...]] = field(default_factory=dict)
    seed: int = 0

    def schema_hash(self) -> str:
        """sha256-truncated-to-16 of the canonical schema description."""
        canonical = "\n".join(sorted(self.create_statements))
        canonical += "\n--rows--\n" + "\n".join(self.seed_statements)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def render_markdown(self) -> str:
        """Schema description shown in the user prompt.

        Lists CREATE TABLE statements + per-table column rosters.
        Never includes the seeded INSERT data — the model must write
        queries that work on arbitrary instances of the schema.
        """
        lines: list[str] = []
        for table in self.table_names:
            lines.append(f"### `{table}`")
            cols = self.column_names_by_table.get(table, ())
            for col in cols:
                lines.append(f"- `{col}`")
            lines.append("")
        return "\n".join(lines).rstrip()


# ── Tokenisation + first-token gate ──────────────────────────────────


_LINE_COMMENT_RE = re.compile(r"--[^\n]*", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LEADING_PARENS_RE = re.compile(r"^\(+")


def _strip_comments(query: str) -> str:
    """Remove SQL comments (line and block) from ``query``."""
    cleaned = _BLOCK_COMMENT_RE.sub(" ", query)
    cleaned = _LINE_COMMENT_RE.sub("", cleaned)
    return cleaned


def _first_token(query: str) -> str | None:
    """First non-whitespace, non-comment, non-paren token (uppercased)."""
    cleaned = _strip_comments(query).strip()
    cleaned = _LEADING_PARENS_RE.sub("", cleaned).strip()
    if not cleaned:
        return None
    match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    if match is None:
        return None
    return match.group(0).upper()


def is_read_only_query(query: str) -> tuple[bool, str | None]:
    """Validate the D8-C read-only gate.

    Returns ``(True, None)`` if the first token is ``SELECT`` /
    ``WITH`` / ``EXPLAIN``. Otherwise returns ``(False, reason)``
    where ``reason`` names the offending token category.
    """
    token = _first_token(query)
    if token is None:
        return False, "empty query"
    if token in _ALLOWED_FIRST_TOKENS:
        return True, None
    if token in _DML_DDL_TOKENS:
        return False, f"unsupported operation; SELECT-only (rejected {token})"
    return False, f"unsupported leading token: {token!r}"


def _query_words(query: str) -> list[str]:
    """Uppercase word list (excluding comments + string literals)."""
    cleaned = _strip_comments(query)
    # Strip single-quoted string literals so a literal like
    # 'something with RANDOM in it' doesn't trip the forbidden-function
    # detector. Double-quoted identifiers (SQLite/Postgres style) are
    # also stripped.
    cleaned = re.sub(r"'(?:[^']|'')*'", " ", cleaned)
    cleaned = re.sub(r'"(?:[^"]|"")*"', " ", cleaned)
    return [t.upper() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)]


def has_forbidden_function(query: str) -> str | None:
    """Return the offending function name if any, else None."""
    words = set(_query_words(query))
    hits = words & _FORBIDDEN_FUNCTIONS
    if hits:
        return sorted(hits)[0]
    return None


def query_has_order_by(query: str) -> bool:
    """True iff the (top-level) query carries an ``ORDER BY`` clause.

    SQLite's grammar puts ``ORDER BY`` after the final compound
    SELECT, so a top-level word search is sufficient — subqueries
    inside ``IN (...)`` or scalar contexts don't affect the outer
    result's ordering.
    """
    words = _query_words(query)
    return any(
        words[i] == "ORDER" and words[i + 1] == "BY"
        for i in range(len(words) - 1)
    )


def query_has_limit_without_order_by(query: str) -> bool:
    """True iff the query has ``LIMIT`` but no ``ORDER BY`` (D9)."""
    words = _query_words(query)
    has_limit = "LIMIT" in words
    return has_limit and not query_has_order_by(query)


def cte_nesting_depth(query: str) -> int:
    """Count nested ``WITH`` clauses (top-level + recursive parts).

    ``WITH RECURSIVE x AS (... WITH y AS (...) ...) SELECT ...``
    nests 2 deep. We count occurrences of the keyword ``WITH`` in
    the comment-stripped, literal-stripped token stream.
    """
    return _query_words(query).count("WITH")


# ── Query canonicalisation + cache key ───────────────────────────────


_WS_RE = re.compile(r"\s+")


def canonicalise_query(query: str) -> str:
    """Lowercase + whitespace-normalise the query.

    Cleaned of comments first; then collapses every run of whitespace
    to a single space and lowercases everything outside string
    literals. Preserves single-quoted string literals byte-for-byte
    (case-sensitive comparisons are common, e.g. names).
    """
    cleaned = _strip_comments(query).strip()
    # Pull out string literals, lowercase the rest, then splice back.
    out: list[str] = []
    cursor = 0
    for m in re.finditer(r"'(?:[^']|'')*'", cleaned):
        prefix = cleaned[cursor:m.start()]
        out.append(_WS_RE.sub(" ", prefix.lower()))
        out.append(m.group(0))
        cursor = m.end()
    tail = cleaned[cursor:]
    out.append(_WS_RE.sub(" ", tail.lower()))
    return "".join(out).strip()


def query_canonical_hash(query: str) -> str:
    """sha256-truncated-to-16 of the canonical query form (D10-B)."""
    canon = canonicalise_query(query)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# ── Sandbox primitive ───────────────────────────────────────────────


def _make_progress_handler(start: float, timeout_s: float):
    """Closure that aborts via non-zero return after ``timeout_s``."""

    def _handler() -> int:
        return 1 if (time.monotonic() - start) > timeout_s else 0

    return _handler


def execute_query_sync(
    *,
    schema_sql: list[str],
    query: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
    cte_depth_cap: int = DEFAULT_CTE_DEPTH_CAP,
) -> SqlExecution:
    """Run ``query`` against an ephemeral SQLite DB seeded with
    ``schema_sql``.

    Pre-flight gates (in order):

    1. Empty query → ``error="empty query"``.
    2. Query bytes > ``max_query_bytes`` → reject.
    3. CTE nesting > ``cte_depth_cap`` → reject.
    4. First-token gate (D8-C) → reject DML/DDL.
    5. Forbidden-function check (D9) → reject ``RANDOM`` etc.
    6. ``LIMIT`` without ``ORDER BY`` (D9) → reject.

    On execution: progress-handler-based timeout (via
    :meth:`sqlite3.Connection.set_progress_handler`), eager fetch
    capped at ``max_rows`` (one extra row pulled to detect the cap),
    explicit ``connection.close()`` in ``finally``.
    """
    start = time.monotonic()

    if not query or not query.strip():
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False, error="empty query",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )
    if len(query.encode("utf-8")) > max_query_bytes:
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False,
            error=f"query exceeds {max_query_bytes}-byte cap",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    depth = cte_nesting_depth(query)
    if depth > cte_depth_cap:
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False,
            error=f"CTE nesting depth {depth} exceeds cap {cte_depth_cap}",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    ok, reason = is_read_only_query(query)
    if not ok:
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False, error=reason or "rejected by read-only gate",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    forbidden = has_forbidden_function(query)
    if forbidden is not None:
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False,
            error=f"non-deterministic function rejected: {forbidden}",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    if query_has_limit_without_order_by(query):
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False,
            error="LIMIT without ORDER BY is non-deterministic",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )

    conn = sqlite3.connect(":memory:")
    conn.set_progress_handler(_make_progress_handler(start, timeout_s), 1_000)
    try:
        for stmt in schema_sql:
            stmt = stmt.strip()
            if stmt:
                conn.executescript(stmt) if stmt.count(";") > 1 else conn.execute(stmt)
        cursor = conn.execute(query)
        column_names = (
            tuple(d[0] for d in cursor.description) if cursor.description else ()
        )
        # Pull one extra row to detect overflow.
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = tuple(tuple(r) for r in fetched[:max_rows])
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return SqlExecution(
            success=True,
            rows=rows,
            column_names=column_names,
            rowcount=len(rows),
            truncated=truncated,
            error=None,
            elapsed_ms=elapsed_ms,
        )
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            return SqlExecution(
                success=False, rows=(), column_names=(), rowcount=0,
                truncated=False, error=f"query timed out (>{timeout_s}s)",
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False, error=f"sqlite3 error: {msg}",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )
    except (sqlite3.DatabaseError, sqlite3.Warning, MemoryError) as exc:
        return SqlExecution(
            success=False, rows=(), column_names=(), rowcount=0,
            truncated=False, error=f"sqlite3 error: {exc}",
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )
    finally:
        conn.close()


# ── Result-set comparator ───────────────────────────────────────────


def _values_equal(a: Any, b: Any, *, float_tol: float) -> bool:
    """Element-wise comparator per D4-A + D9 rules."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b  # bool is int-subclass; force exact match.
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # NaN handling: match NaN with NaN so the comparator is
        # reflexive. Two NaNs in the same column position are treated
        # as equal — matches NULL-semantics for "missing data".
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            return True
        if isinstance(a, float) and math.isnan(a):
            return False
        if isinstance(b, float) and math.isnan(b):
            return False
        return abs(float(a) - float(b)) <= float_tol
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, (bytes, bytearray)) and isinstance(b, (bytes, bytearray)):
        return bytes(a) == bytes(b)
    return False


def _row_equal(
    gold_row: tuple[Any, ...],
    pred_row: tuple[Any, ...],
    *,
    float_tol: float,
) -> bool:
    if len(gold_row) != len(pred_row):
        return False
    return all(
        _values_equal(g, p, float_tol=float_tol)
        for g, p in zip(gold_row, pred_row, strict=True)
    )


def _canon_value(value: Any, *, float_tol: float) -> Any:
    """Stable, hashable key for unordered comparison.

    Floats are quantised to multiples of ``float_tol`` so two rows
    that differ only by floating-point rounding land on the same key.
    """
    if value is None:
        return ("__NULL__",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        # Quantise via division + rounding.
        q = round(value / float_tol)
        return ("float", q)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    return ("repr", repr(value))


def _canon_row(row: tuple[Any, ...], *, float_tol: float) -> tuple[Any, ...]:
    return tuple(_canon_value(v, float_tol=float_tol) for v in row)


def compare_result_sets(
    gold_rows: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    predicted_rows: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    *,
    ordered: bool,
    float_tol: float = DEFAULT_FLOAT_TOL,
) -> bool:
    """D4-A result-set equality with the D9 element-wise rules.

    Ordered comparison: pairwise zip of rows. Unordered comparison:
    sorted multisets of canonicalised rows. Empty result-sets compare
    equal regardless of ``ordered``.
    """
    g = list(gold_rows)
    p = list(predicted_rows)
    if len(g) != len(p):
        return False
    if not g:
        return True
    if ordered:
        return all(
            _row_equal(a, b, float_tol=float_tol)
            for a, b in zip(g, p, strict=True)
        )
    g_canon = sorted(_canon_row(r, float_tol=float_tol) for r in g)
    p_canon = sorted(_canon_row(r, float_tol=float_tol) for r in p)
    return g_canon == p_canon


# ── Procedural schema templates (D3-A) ──────────────────────────────


def _quote_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _tmpl_single_table_filter(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Products under a sampled price threshold."""
    n_rows = int(rng.integers(8, 20))
    threshold = int(rng.integers(5, 30))
    create = (
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER);"
    )
    inserts: list[str] = []
    rows: list[tuple[Any, ...]] = []
    for i in range(n_rows):
        price = int(rng.integers(1, 50))
        name = f"product_{i:03d}"
        inserts.append(
            f"INSERT INTO products (id, name, price) VALUES ({i + 1}, {_quote_str(name)}, {price});"
        )
        rows.append((i + 1, name, price))
    expected = sorted(
        [(rid, n, p) for rid, n, p in rows if p < threshold],
        key=lambda r: r[2],
    )
    gold_query = (
        f"SELECT id, name, price FROM products WHERE price < {threshold} "
        "ORDER BY price ASC, id ASC"
    )
    prompt = (
        f"List the products with price strictly less than {threshold}, "
        "ordered by price ascending (with id as a stable tiebreaker)."
    )
    return _problem_dict(
        template_name="single_table_filter",
        prompt=prompt,
        create_statements=(create,),
        seed_statements=tuple(inserts),
        table_names=("products",),
        column_names={"products": ("id", "name", "price")},
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=tuple(expected),
        seed=seed,
    )


def _tmpl_single_table_aggregate(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Total revenue per category (sales)."""
    categories = ("books", "tools", "games", "kitchen", "outdoor")
    n_rows = int(rng.integers(15, 30))
    create = (
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER);"
    )
    inserts: list[str] = []
    totals: dict[str, int] = {}
    for i in range(n_rows):
        cat = categories[int(rng.integers(0, len(categories)))]
        amt = int(rng.integers(5, 200))
        inserts.append(
            f"INSERT INTO sales (id, category, amount) VALUES ({i + 1}, {_quote_str(cat)}, {amt});"
        )
        totals[cat] = totals.get(cat, 0) + amt
    expected = sorted(totals.items())
    expected_rows = tuple((cat, total) for cat, total in expected)
    gold_query = (
        "SELECT category, SUM(amount) AS total FROM sales "
        "GROUP BY category ORDER BY category ASC"
    )
    prompt = (
        "Compute the total amount per category in the `sales` table. "
        "Return rows of `(category, total)` ordered by category ascending."
    )
    return _problem_dict(
        template_name="single_table_aggregate",
        prompt=prompt,
        create_statements=(create,),
        seed_statements=tuple(inserts),
        table_names=("sales",),
        column_names={"sales": ("id", "category", "amount")},
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=expected_rows,
        seed=seed,
    )


def _tmpl_two_table_join(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Names of customers with > N orders."""
    n_customers = int(rng.integers(5, 12))
    n_orders = int(rng.integers(20, 40))
    threshold = int(rng.integers(2, 5))

    create_customers = (
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
    )
    create_orders = (
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);"
    )
    customer_inserts: list[str] = []
    order_inserts: list[str] = []
    customers_by_id: dict[int, str] = {}

    for i in range(n_customers):
        name = f"customer_{i:03d}"
        cid = i + 1
        customers_by_id[cid] = name
        customer_inserts.append(
            f"INSERT INTO customers (id, name) VALUES ({cid}, {_quote_str(name)});"
        )

    counts: dict[int, int] = {}
    for j in range(n_orders):
        cid = int(rng.integers(1, n_customers + 1))
        counts[cid] = counts.get(cid, 0) + 1
        order_inserts.append(
            f"INSERT INTO orders (id, customer_id) VALUES ({j + 1}, {cid});"
        )

    expected_names = sorted(
        customers_by_id[cid] for cid, c in counts.items() if c > threshold
    )
    gold_query = (
        "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
        f"GROUP BY c.id, c.name HAVING COUNT(o.id) > {threshold} ORDER BY c.name ASC"
    )
    prompt = (
        f"Return the names of customers who have more than {threshold} orders. "
        "Order the results by name ascending."
    )
    return _problem_dict(
        template_name="two_table_join",
        prompt=prompt,
        create_statements=(create_customers, create_orders),
        seed_statements=tuple(customer_inserts + order_inserts),
        table_names=("customers", "orders"),
        column_names={
            "customers": ("id", "name"),
            "orders": ("id", "customer_id"),
        },
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=tuple((n,) for n in expected_names),
        seed=seed,
    )


def _tmpl_three_table_join(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Top-K products by units sold (orders + items + products)."""
    n_products = int(rng.integers(5, 10))
    n_orders = int(rng.integers(15, 25))
    create_products = (
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT);"
    )
    create_orders = (
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, day INTEGER);"
    )
    create_items = (
        "CREATE TABLE items (order_id INTEGER, product_id INTEGER, qty INTEGER);"
    )
    product_inserts: list[str] = []
    products_by_id: dict[int, str] = {}
    for i in range(n_products):
        pid = i + 1
        name = f"product_{i:03d}"
        products_by_id[pid] = name
        product_inserts.append(
            f"INSERT INTO products (id, name) VALUES ({pid}, {_quote_str(name)});"
        )

    order_inserts: list[str] = []
    item_inserts: list[str] = []
    qty_by_product: dict[int, int] = {}
    for j in range(n_orders):
        order_inserts.append(
            f"INSERT INTO orders (id, day) VALUES ({j + 1}, {int(rng.integers(1, 30))});"
        )
        # Each order has 1-3 items.
        for _ in range(int(rng.integers(1, 4))):
            pid = int(rng.integers(1, n_products + 1))
            qty = int(rng.integers(1, 6))
            qty_by_product[pid] = qty_by_product.get(pid, 0) + qty
            item_inserts.append(
                f"INSERT INTO items (order_id, product_id, qty) "
                f"VALUES ({j + 1}, {pid}, {qty});"
            )

    expected_rows = sorted(
        ((products_by_id[pid], total) for pid, total in qty_by_product.items()),
        key=lambda r: (-r[1], r[0]),
    )[:5]
    gold_query = (
        "SELECT p.name, SUM(i.qty) AS units FROM items i "
        "JOIN orders o ON o.id = i.order_id "
        "JOIN products p ON p.id = i.product_id "
        "GROUP BY p.id, p.name "
        "ORDER BY units DESC, p.name ASC LIMIT 5"
    )
    prompt = (
        "Return the top 5 products by total units sold across all "
        "orders. Each row is `(product_name, total_units)`, ordered "
        "by units descending (name ascending as tiebreaker)."
    )
    return _problem_dict(
        template_name="three_table_join",
        prompt=prompt,
        create_statements=(create_products, create_orders, create_items),
        seed_statements=tuple(product_inserts + order_inserts + item_inserts),
        table_names=("products", "orders", "items"),
        column_names={
            "products": ("id", "name"),
            "orders": ("id", "day"),
            "items": ("order_id", "product_id", "qty"),
        },
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=tuple(expected_rows),
        seed=seed,
    )


def _tmpl_groupby_having(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Event types with > N occurrences."""
    types = ("login", "logout", "click", "purchase", "view", "scroll")
    n_rows = int(rng.integers(30, 60))
    threshold = int(rng.integers(3, 8))
    create = (
        "CREATE TABLE events (id INTEGER PRIMARY KEY, type TEXT);"
    )
    inserts: list[str] = []
    counts: dict[str, int] = {}
    for i in range(n_rows):
        t = types[int(rng.integers(0, len(types)))]
        counts[t] = counts.get(t, 0) + 1
        inserts.append(
            f"INSERT INTO events (id, type) VALUES ({i + 1}, {_quote_str(t)});"
        )
    expected = sorted(
        ((t, c) for t, c in counts.items() if c > threshold),
        key=lambda r: r[0],
    )
    gold_query = (
        "SELECT type, COUNT(*) AS n FROM events GROUP BY type "
        f"HAVING COUNT(*) > {threshold} ORDER BY type ASC"
    )
    prompt = (
        f"Find the event types that occur strictly more than {threshold} "
        "times in the `events` table. Each row is `(type, count)`, "
        "ordered by type ascending."
    )
    return _problem_dict(
        template_name="groupby_having",
        prompt=prompt,
        create_statements=(create,),
        seed_statements=tuple(inserts),
        table_names=("events",),
        column_names={"events": ("id", "type")},
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=tuple(expected),
        seed=seed,
    )


def _tmpl_subquery_filter(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Employees earning above their department average."""
    departments = ("eng", "sales", "ops", "design")
    n_rows = int(rng.integers(10, 18))
    create = (
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, "
        "department TEXT, salary INTEGER);"
    )
    inserts: list[str] = []
    rows: list[tuple[int, str, str, int]] = []
    for i in range(n_rows):
        name = f"emp_{i:03d}"
        dept = departments[int(rng.integers(0, len(departments)))]
        salary = int(rng.integers(50, 200)) * 1000
        rows.append((i + 1, name, dept, salary))
        inserts.append(
            f"INSERT INTO employees (id, name, department, salary) "
            f"VALUES ({i + 1}, {_quote_str(name)}, {_quote_str(dept)}, {salary});"
        )
    avg_by_dept: dict[str, float] = {}
    counts: dict[str, int] = {}
    sums: dict[str, int] = {}
    for _, _, dept, salary in rows:
        sums[dept] = sums.get(dept, 0) + salary
        counts[dept] = counts.get(dept, 0) + 1
    for d in counts:
        avg_by_dept[d] = sums[d] / counts[d]
    expected_names = sorted(
        n for _, n, dept, salary in rows if salary > avg_by_dept[dept]
    )
    gold_query = (
        "SELECT name FROM employees e WHERE salary > "
        "(SELECT AVG(salary) FROM employees WHERE department = e.department) "
        "ORDER BY name ASC"
    )
    prompt = (
        "Return the names of employees whose salary is strictly greater "
        "than the average salary in their own department. Order by name ascending."
    )
    return _problem_dict(
        template_name="subquery_filter",
        prompt=prompt,
        create_statements=(create,),
        seed_statements=tuple(inserts),
        table_names=("employees",),
        column_names={
            "employees": ("id", "name", "department", "salary"),
        },
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=tuple((n,) for n in expected_names),
        seed=seed,
    )


def _tmpl_cte_aggregate(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Per-day order totals computed via a CTE."""
    n_rows = int(rng.integers(10, 18))
    create = (
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, day INTEGER, total INTEGER);"
    )
    inserts: list[str] = []
    totals_by_day: dict[int, int] = {}
    for i in range(n_rows):
        d = int(rng.integers(1, 8))
        t = int(rng.integers(10, 200))
        totals_by_day[d] = totals_by_day.get(d, 0) + t
        inserts.append(
            f"INSERT INTO orders (id, day, total) VALUES ({i + 1}, {d}, {t});"
        )
    expected = sorted(totals_by_day.items())
    expected_rows = tuple((day, total) for day, total in expected)
    gold_query = (
        "WITH per_day AS (SELECT day, SUM(total) AS day_total FROM orders GROUP BY day) "
        "SELECT day, day_total FROM per_day ORDER BY day ASC"
    )
    prompt = (
        "Compute the total order amount per day in the `orders` table "
        "using a CTE. Return rows of `(day, day_total)` ordered by day ascending."
    )
    return _problem_dict(
        template_name="cte_aggregate",
        prompt=prompt,
        create_statements=(create,),
        seed_statements=tuple(inserts),
        table_names=("orders",),
        column_names={"orders": ("id", "day", "total")},
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=expected_rows,
        seed=seed,
    )


def _tmpl_date_arithmetic(rng: np.random.Generator, seed: int) -> dict[str, Any]:
    """Sessions per hour-of-day across a sampled time window."""
    n_rows = int(rng.integers(20, 35))
    create = (
        "CREATE TABLE sessions (id INTEGER PRIMARY KEY, ts TEXT);"
    )
    inserts: list[str] = []
    counts_by_hour: dict[str, int] = {}
    base_year = 2024
    base_month = int(rng.integers(1, 12))
    base_day = int(rng.integers(1, 27))
    for i in range(n_rows):
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        ts = f"{base_year:04d}-{base_month:02d}-{base_day:02d} {hour:02d}:{minute:02d}:00"
        hour_key = f"{hour:02d}"
        counts_by_hour[hour_key] = counts_by_hour.get(hour_key, 0) + 1
        inserts.append(
            f"INSERT INTO sessions (id, ts) VALUES ({i + 1}, {_quote_str(ts)});"
        )
    expected = sorted(counts_by_hour.items())
    expected_rows = tuple((hour, count) for hour, count in expected)
    start_ts = f"{base_year:04d}-{base_month:02d}-{base_day:02d} 00:00:00"
    end_ts = f"{base_year:04d}-{base_month:02d}-{base_day:02d} 23:59:59"
    gold_query = (
        "SELECT STRFTIME('%H', ts) AS hour, COUNT(*) AS n FROM sessions "
        f"WHERE ts BETWEEN {_quote_str(start_ts)} AND {_quote_str(end_ts)} "
        "GROUP BY STRFTIME('%H', ts) ORDER BY hour ASC"
    )
    prompt = (
        "Group session rows by hour-of-day for the sampled date "
        f"({base_year}-{base_month:02d}-{base_day:02d}) and return "
        "`(hour, count)` ordered by hour ascending. Use SQLite's "
        "`STRFTIME('%H', ts)` to extract the hour."
    )
    return _problem_dict(
        template_name="date_arithmetic",
        prompt=prompt,
        create_statements=(create,),
        seed_statements=tuple(inserts),
        table_names=("sessions",),
        column_names={"sessions": ("id", "ts")},
        gold_query=gold_query,
        gold_query_is_ordered=True,
        gold_result_rows=expected_rows,
        seed=seed,
    )


def _problem_dict(
    *,
    template_name: str,
    prompt: str,
    create_statements: tuple[str, ...],
    seed_statements: tuple[str, ...],
    table_names: tuple[str, ...],
    column_names: dict[str, tuple[str, ...]],
    gold_query: str,
    gold_query_is_ordered: bool,
    gold_result_rows: tuple[tuple[Any, ...], ...],
    seed: int,
) -> dict[str, Any]:
    return {
        "template_name": template_name,
        "prompt": prompt,
        "create_statements": create_statements,
        "seed_statements": seed_statements,
        "table_names": table_names,
        "column_names": column_names,
        "gold_query": gold_query,
        "gold_query_is_ordered": gold_query_is_ordered,
        "gold_result_rows": gold_result_rows,
        "seed": seed,
    }


# Locked roster — 8 templates per PHASE_26_PLAN.md §8.2.
_TEMPLATES: tuple[Any, ...] = (
    _tmpl_single_table_filter,
    _tmpl_single_table_aggregate,
    _tmpl_two_table_join,
    _tmpl_three_table_join,
    _tmpl_groupby_having,
    _tmpl_subquery_filter,
    _tmpl_cte_aggregate,
    _tmpl_date_arithmetic,
)


def generate_problem(seed: int, **_unused: Any) -> dict[str, Any]:
    """Sample a fresh SQL problem from the procedural lattice."""
    rng = np.random.default_rng(int(seed))
    template_idx = int(rng.integers(0, len(_TEMPLATES)))
    return _TEMPLATES[template_idx](rng, int(seed))


def build_schema(problem: dict[str, Any]) -> Schema:
    """Wrap a problem dict's schema fields in a :class:`Schema`."""
    return Schema(
        create_statements=tuple(problem["create_statements"]),
        seed_statements=tuple(problem["seed_statements"]),
        table_names=tuple(problem["table_names"]),
        column_names_by_table=dict(problem["column_names"]),
        seed=int(problem["seed"]),
    )


__all__ = [
    "DEFAULT_CTE_DEPTH_CAP",
    "DEFAULT_FLOAT_TOL",
    "DEFAULT_MAX_QUERY_BYTES",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TIMEOUT_S",
    "Schema",
    "SqlExecution",
    "build_schema",
    "canonicalise_query",
    "compare_result_sets",
    "cte_nesting_depth",
    "execute_query_sync",
    "generate_problem",
    "has_forbidden_function",
    "is_read_only_query",
    "query_canonical_hash",
    "query_has_limit_without_order_by",
    "query_has_order_by",
]
