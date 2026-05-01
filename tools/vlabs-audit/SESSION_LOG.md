# Session log — vlabs-audit

Append-only timestamped log. Newest entry on top.

---

## 2026-05-02 — Phase 17, Sub-stage 17.E.5 (cost wiring + single-env fallbacks + hyphenation)

**Goal**: customer-facing-quality polish before the 17.F sample run —
populate the cost figure with real numbers, replace the broken-looking
single-bar charts with text cards on single-env audits, and fix awkward
LaTeX hyphenation.

### Done
- `src/vlabs_audit/runner.py`: every successful `verifiable run`
  invocation now harvests its cost from the SDK's
  ``Time: ... · Cost: $X.XXXX`` stdout summary line and back-fills it
  into the per-episode trace JSONL as ``estimated_cost_usd`` before
  ``stats.py`` reads it. New helpers ``_parse_cost_usd`` and
  ``_augment_trace_with_cost`` are unit-tested independently.
  Investigation: the SDK Trace dataclass has the field but
  ``cmd_run`` never populates it; tokens are not in the JSONL or
  stdout, only the aggregated cost line. Parsing that line is the
  cleanest path that doesn't require modifying restricted SDK code.
- `src/vlabs_audit/figures.py`:
  - ``_render_text_summary(ax, [(label, value), ...])`` helper —
    centred two-column text card for figures that would otherwise
    show a single-bar chart.
  - ``coverage_calibration``: when ``len(per_env) == 1`` switches to
    a card showing env name, empirical coverage, and "within target"
    or "below target" verdict.
  - ``score_breakdown``: when ``len(per_env) == 1`` switches to a card
    listing format-pass / format-fail / parse-error percentages for
    the one env. Multi-env path unchanged.
  - ``cost_per_correct`` already has a graceful "Cost data not
    available" plate; with the runner backfill above it now actually
    plots real bars on real audits.
- `templates/style/vlabs_report.cls`: added
  ``\RequirePackage{microtype}``, ``\hyphenpenalty=5000``,
  ``\tolerance=1000`` for cleaner line breaks (mid-word hyphens
  removed in customer-facing copy).
- `tests/test_runner.py`: 2 new tests — ``_parse_cost_usd`` parses
  4-decimal, 2-decimal, and em-dash forms and ignores stray earlier
  ``Cost:`` strings; ``_augment_trace_with_cost`` round-trips every
  pre-existing field. Existing
  ``test_default_episode_run_parses_subprocess_output`` extended to
  include a ``Cost: $0.0024`` line in the fake stdout and assert the
  resulting JSONL carries ``estimated_cost_usd``.
- `tests/test_figures.py`: 3 new tests using ``pdftotext`` to extract
  text from rendered PDFs (matplotlib subsets fonts so raw-byte
  search misses ASCII strings) — single-env coverage + breakdown
  cards render the env name and the right labels; below-target
  coverage labels the verdict accordingly.

### Tests
- `pytest tools/vlabs-audit/tests/` → **77 / 77 green** in 17 s
  (was 72; +5 = 2 runner cost + 3 figure single-env).
- `ruff check tools/vlabs-audit/src tools/vlabs-audit/tests` clean.

### Real PDF re-rendered
```
vlabs-audit audit --model anthropic/claude-haiku-4.5
  --envs sparse-fourier-recovery --episodes 5 --parallel 4
  --seed-start 9900 --output /tmp/vlabs-17e5-smoke/report.pdf
  --print-stats
```
- 5 / 5 success, mean 0.363, 95% CI [0.344, 0.379], cov-holdout 0.967.
- Per-trace cost: every JSONL has
  ``"estimated_cost_usd": 0.0014``; aggregate ``$0.0070`` over 5
  episodes; cost-per-correct ``$0.0014``.
- ``pdfinfo``: 11 pages, 133 686 bytes (was 124 327 — slightly
  larger because microtype embeds protrusion glyph data).
- Cost figure now plots a real bar at ``$0.0014 USD per correct
  answer`` instead of the placeholder.
- Coverage figure renders the text card:
  ``environment: sparse-fourier-recovery``,
  ``empirical coverage: 0.967``, ``target (0.90): within target``.
- Quality-breakdown figure renders the text card:
  ``environment: sparse-fourier-recovery``, ``format pass: 100.0%``,
  ``format fail: 0.0%``, ``parse error: 0.0%``.

### Open
- 17.F — sample run with anonymisation, ~$4 OpenRouter spend
  (3+ envs). With this polish the multi-env figures drop back to
  bar charts naturally — no extra code needed.
- Restricted-path diff outside `tools/vlabs-audit/` unchanged from
  17.A–E. No commits made.

---

## 2026-05-02 — Phase 17, Sub-stage 17.E (LaTeX templates + tectonic)

**Goal**: render an `AuditStats` and the four figures from 17.D into a
professional LaTeX/PDF capability report via `tectonic`. Final
customer-facing artifact for cold outreach + funding applications.

### Done
- `templates/style/vlabs_report.cls` — custom LaTeX class on top of
  `article` (11 pt, letter, 1 in margins). Loads `lato` (sans for
  headers via `\sffamily`), `lmodern` (serif body), `amsmath` (for
  `\text{}` inside `$...$`), `booktabs`, `graphicx`, `float` (for `[H]`
  figure floats), `hyperref` (dark-blue links), `fancyhdr` (audit-id in
  footer, page number right). Cover macro
  `\makevlabscover{title}{model}{date}` per spec; audit id is set
  separately via `\setvlabsaudit{...}` so the footer macro can pick it
  up on every page.
- `templates/main.tex.j2` — 28 lines. Just the document skeleton; the
  five section bodies are rendered into strings by Python and inlined
  via Jinja's `{{ exec_summary_tex }}` etc.
- `templates/sections/exec_summary.tex.j2` — top-line numbers, headline
  finding paragraph, coverage-compliance booktabs table, recommendation
  summary.
- `templates/sections/methodology.tex.j2` — one-paragraph conformal
  primer, audit-config table, env list, "what this audit doesn't do",
  Lei et al. 2018 bibitem.
- `templates/sections/results.tex.j2` — per-env subsection
  (description + booktabs metrics table + reward-distribution figure)
  followed by the three audit-level figures
  (`coverage_calibration`, `score_breakdown`, `cost_per_correct`).
  Figure paths are wrapped in `\detokenize{...}` so `_` in
  `reward_distribution_<env>.pdf` doesn't break LaTeX math-mode parsing.
- `templates/sections/recommendations.tex.j2` — bullet list seeded from
  `_recommendations(stats)`: low-mean / under-coverage / high
  parse-fail rules; conservative default ("no remedial action") when
  every env passes.
- `templates/sections/appendix.tex.j2` — reproducibility-metadata
  table, `verbatim` JSON dump of `stats.model_dump()`, BibTeX-style
  citation block, license note.
- `src/vlabs_audit/latex.py` (~520 LOC):
  - `latex_escape(value)` — handles every one of the 11 special chars
    listed in the plan; `\` is processed first via a sentinel so the
    multi-char `\textbackslash{}` replacement doesn't get re-escaped.
    Registered as `e` and `latex` Jinja filters (StrictUndefined).
  - `_build_context(stats)` — produces the dict the templates consume.
    Suffix conventions: `_tex` for already-escaped text, `_str` for
    pre-formatted numeric (safe to splice), `_raw` for intentionally
    unescaped (e.g. JSON for `verbatim`).
  - `render_tex(stats)` — pure-string output, no I/O, no subprocess.
    Used by every test that doesn't need real LaTeX compilation.
  - `render_report(stats, figures_dir, output_path)` — full pipeline.
    Detects `tectonic` on PATH and raises with the upstream install
    URL if missing. Copies `vlabs_report.cls` + every `*.pdf` from
    `figures_dir` alongside the rendered `main.tex` into a tempdir,
    runs `tectonic -X compile main.tex`, surfaces the last 2 KB of
    the log on compile failure, moves the resulting `main.pdf` to
    `output_path`.
  - Template-discovery is editable-install-aware AND wheel-aware: tries
    `tools/vlabs-audit/templates/` first, falls back to
    `vlabs_audit/templates/` (force-included into the wheel by a new
    rule in `pyproject.toml`).
- `src/vlabs_audit/cli.py` — passing `--output` + not `--dry-run` now
  auto-implies `--print-stats` and a tmp `--figures-dir`, runs the
  full pipeline, and prints `Report rendered to <path>`. LaTeX-render
  failures (missing tectonic, compile error) downgrade to a stderr
  warning so the audit + stats + figures still complete.
- `tests/test_latex.py` (11 tests): every special character in the
  escape table, safe-char passthrough, full-document section-headers
  presence, fixture-driven snapshot substring check, cover-macro
  receives 3 escaped args, figure paths are relative basenames,
  missing-tectonic raises with install URL, compile-failure surfaces
  log tail, full pipeline with subprocess stubbed (asserts cwd
  contents + final PDF bytes), tectonic timeout propagates,
  per-env table rows + parse-rate percentage rendered.
- `tests/fixtures/expected_sections.txt` — 18 substring needles
  used by the snapshot test.
- `pyproject.toml` — `[tool.hatch.build.targets.wheel.force-include]`
  ships `templates/` into the wheel under `vlabs_audit/templates/`.

### Tests
- `pytest tools/vlabs-audit/tests/` → **72 / 72 green** in 15 s
  (was 61; +11 new latex tests).
- `ruff check tools/vlabs-audit/src tools/vlabs-audit/tests` clean.

### Real PDF generated end-to-end (5-episode CLI smoke)
```
vlabs-audit audit --model anthropic/claude-haiku-4.5
  --envs sparse-fourier-recovery --episodes 5 --parallel 4
  --seed-start 9800 --output /tmp/vlabs-17e-cli/report.pdf --print-stats
```
- 5 / 5 success, mean 0.329, 95% CI [0.316, 0.345], cov-holdout 0.933.
- `Report rendered to /tmp/vlabs-17e-cli/report.pdf`
- `pdfinfo`: **11 pages**, letter (612 × 792 pt), 124 327 bytes,
  `Creator: LaTeX with hyperref`, `Producer: xdvipdfmx`.
- `pdftotext` confirms: cover page (title + model + audit id), full
  TOC, all five `\section{}` headers, every figure caption
  (`Reward Distribution`, `Empirical Coverage vs Target (90%)`,
  `Quality Breakdown by Environment`, `Cost per Correct Answer`),
  auto-generated recommendation referencing the actual mean
  (`Mean reward on sparse-fourier-recovery is 0.329, at or below
  the 0.4 threshold...`), JSON stats dump in the appendix, audit id
  on every footer.
- Within the 8–15 page target band.

### Bug caught + fixed during smoke
- First compile attempt failed with `Undefined control sequence` at
  `\text{resamples}` inside `$n_{\text{resamples}} = 1000$`. Fix: add
  `\RequirePackage{amsmath}` to the class. Caught only by real
  tectonic; unit tests are pure-string.

### Known follow-ups Stelios flagged for tomorrow
- `cost_per_correct.pdf` is still the placeholder (SDK doesn't surface
  `estimated_cost_usd`). The LaTeX caption for that figure already
  describes this honestly, but pre-17.F we may want to wire per-trace
  cost so the customer-facing artifact shows real numbers.
- `coverage_calibration.pdf` will look better with 3+ envs in 17.F.

### Open
- 17.F — sample run with anonymisation, ~$4 OpenRouter spend.
- 17.G — tests + CI integration.
- 17.H — full docs.
- Restricted-path diff outside `tools/vlabs-audit/` is unchanged from
  17.A–D: only the pre-existing M3/M4 modification of
  `src/verifiable_labs_envs/cli.py`. No new touches from 17.E.
- No commits made; working tree builds toward the final 2-commit
  structure at end of Phase 17.

---

## 2026-05-01 — Phase 17, Sub-stage 17.D (matplotlib PDF figures)

**Goal**: produce four publication-quality vector PDFs from a populated
`AuditStats` so the LaTeX template (17.E) can `\includegraphics` them
directly.

### Done
- `src/vlabs_audit/figures.py` (~280 LOC). Uses matplotlib's `Agg`
  backend (set before `pyplot` import — works headless / over SSH /
  in CI). Four figure types:
  1. `reward_distribution_<env>.pdf` — histogram + 95 % CI band +
     mean line, one per env. Falls back to a "no successful
     episodes" plate when the env is empty.
  2. `coverage_calibration.pdf` — bar chart of empirical (held-out)
     coverage vs target = 1 − α. Bars coloured green when at-or-above
     target, red below, grey when coverage data is missing/NaN.
  3. `score_breakdown.pdf` — horizontal stacked bar (format pass /
     format fail / parse error) per env; segments always sum to 1.0.
  4. `cost_per_correct.pdf` — USD per successful episode, log-scaled
     when the per-env range exceeds 100×. When *no* env has cost data
     (current SDK behaviour — traces don't carry
     `estimated_cost_usd`) renders a "Cost data not available" plate
     so the LaTeX `\includegraphics` still has a real figure to
     embed.
  - Common styling: 6.5 in width = LaTeX `\textwidth`, serif font @
    10 pt, no top/right spines, vector PDF, `bbox_inches="tight"`,
    metadata embedded (`/Title`, `/Author=vlabs-audit`,
    `/Subject=audit_id:<id>`).
  - `_coverage_color(coverage, target)` extracted as a public helper
    so calibration colour logic is unit-testable in isolation.
- `src/vlabs_audit/stats.py`: `EnvStats` gained two fields needed by
  the figures layer — `rewards: list[float]` (raw per-episode rewards
  for the histogram) and `total_cost_usd: float` (sum of
  `estimated_cost_usd` from the env's traces; `0.0` when missing).
  Default values keep the existing JSON round-trip test green; the
  LaTeX context just sees additional keys.
- `src/vlabs_audit/cli.py`: new `--figures-dir <path>` flag. Auto-
  implies `--print-stats`. Dry-run prints stub messages for both
  flags. Real run computes stats, prints the table, calls
  `render_all_figures`, then prints `Wrote N figures to <path>`.
  Stats failures (empty audit) and figure failures (no per-env
  data) downgrade to stderr warnings; the run itself still exits 0.
- `tests/test_figures.py` (8 functions, 14 sub-tests):
  end-to-end `render_all_figures` produces 5 PDFs with valid
  `%PDF-` magic and `>1 KB` size; PDF metadata contains
  `audit_id:<id>`; empty-`per_env` raises ValueError; missing-cost
  data renders a placeholder; non-zero-cost env gets a real bar;
  `_coverage_color` parametrised over 7 cases (above/at/below
  target + None/NaN); 0-success env still produces a valid
  placeholder; stacked bar with both parse-fail and format-fail
  works.
- `tests/test_cli.py`: 2 new tests for `--figures-dir` — dry-run
  stub (no files written) and real-run that asserts 4 PDFs land
  on disk with valid magic.

### Tests
- `pytest tools/vlabs-audit/tests/` → **61 / 61 green** in 5.3 s
  (was 45; +16 = 14 figures sub-tests + 2 CLI).
- `ruff check tools/vlabs-audit/src tools/vlabs-audit/tests` clean.
- Real smoke (5 episodes, `anthropic/claude-haiku-4.5` via
  OpenRouter, `sparse-fourier-recovery`, parallel = 4,
  `--figures-dir /tmp/vlabs-figs-demo`):
  ```
  status: {'success': 5}
  mean reward 0.328, 95 % CI [0.286, 0.364], cov-holdout 0.833
  Wrote 4 figures to /tmp/vlabs-figs-demo
  ```
  All four PDFs verified: magic `%PDF-`, sizes
  10 / 14 / 14 / 17 KB, both `audit_id` and `vlabs-audit` strings
  present in metadata.

### Known limitations
- `cost_per_correct` shows the placeholder plate on real runs because
  the SDK's `verifiable_labs_envs.traces.Trace` does not currently
  populate `estimated_cost_usd`. Wiring per-trace cost (either by
  parsing the SDK's stdout or persisting `token_input` /
  `token_output` and applying a replicated price table) is a small
  follow-up — out of scope for 17.D, which is about the figure
  pipeline, not new SDK plumbing.

### Open
- 17.E — Jinja2 LaTeX template + tectonic invocation, embedding the
  four figures + stats table.
- Restricted-path diff outside `tools/vlabs-audit/` is unchanged from
  17.A/B/C: only the pre-existing M3/M4 modification of
  `src/verifiable_labs_envs/cli.py`. No new touches from 17.D.

---

## 2026-05-01 — Phase 17, Sub-stage 17.C (stats aggregation)

**Goal**: read JSONL traces from audit storage, compute the metrics the
LaTeX report (17.E) needs — mean reward + 95 % bootstrap CI, parse
failure rate, format validity, held-out coverage — and surface them
through a `--print-stats` CLI flag.

### Done
- `src/vlabs_audit/stats.py` (~270 LOC):
  - `bootstrap_ci(values, alpha, n_resamples=1000, seed=42)` — percentile
    bootstrap CI over the mean. Deterministic (RNG-seeded). Empty input
    raises; singleton collapses to `(v, v)`.
  - `EnvStats` and `AuditStats` pydantic models (`extra="forbid"`).
  - `compute_audit_stats(store, audit_id, alpha=0.1)` — joins
    `audit_runs` rows with on-disk JSONL trace files, returns the full
    aggregate. Per-env block = mean reward + CI + parse-fail rate +
    format validity + held-out coverage (mean of seed-sorted second
    half). Cross-env aggregate is the unweighted mean over every
    successful episode (handles uneven episode counts).
  - Coverage extraction prefers top-level `coverage`, falls back to
    `reward_components.conformal` (the per-episode conformal hit the
    SDK already records for envs with conformal intervals — covers the
    sparse-fourier / phase-retrieval / super-resolution shapes
    out-of-the-box).
  - `format_stats_table(stats)` renders a fixed-width table mirroring
    `verifiable compare` for `--print-stats`.
- `src/vlabs_audit/cli.py`: new `--print-stats` flag on the `audit`
  command. Dry-run prints a stub line; real run computes
  `compute_audit_stats` after the runner finishes and pretty-prints
  the table. Stats failures (e.g. no successful episodes) downgrade
  to a stderr warning; the run itself still exits 0.
- `tests/test_stats.py` (11 tests): synthetic-distribution mean lies
  inside bootstrap CI, all-zeros collapse, all-failed → parse-fail
  rate = 1.0, bootstrap reproducibility (same seed deterministic,
  different seed differs), empty/singleton edge cases, manual coverage
  check, conformal fallback path, `model_dump_json` round-trips,
  unknown / no-rows audits raise clear errors, cross-env uneven
  counts, format-table renders header + rows + AGGREGATE.
- `tests/test_cli.py`: `--print-stats --dry-run` shows a stub message;
  end-to-end real run with a fake `runner_fn` writes Trace-shaped
  JSONL and asserts the rendered AGGREGATE row.

### Bug caught + fixed during smoke
- The SDK's `verifiable run` writes traces under
  `Path.home() / ".verifiable" / "runs"` with a *second-precision*
  timestamp filename. With `--parallel ≥ 2`, two workers running in
  the same wall-clock second collide on that filename and one
  `shutil.move` finds the file already gone. Fixed by giving each
  subprocess its own ephemeral `HOME` (`tempfile.TemporaryDirectory`)
  so the runs dir is unique per call. `PYTHONUSERBASE` is preserved
  so the sandboxed process can still import the `--user`-installed
  SDK. Asserted in `test_default_episode_run_parses_subprocess_output`
  (verifies the captured `env["HOME"]` is a fresh tmp dir, not the
  parent's).

### Tests
- `pytest tools/vlabs-audit/tests/` → **45 / 45 green** in 3.5 s.
- `ruff check tools/vlabs-audit/src tools/vlabs-audit/tests` clean.
- Real smoke (5 episodes, `anthropic/claude-haiku-4.5` via OpenRouter,
  parallel = 4):
  - 5 / 5 success
  - mean reward 0.354, 95 % CI [0.337, 0.371]
  - parse-fail 0 %, fmt-ok 100 %, cov-holdout 1.000
  (matches sparse-fourier-recovery's known zero-baseline regime —
  small models produce ~0.336 mean reward on this env.)

### Open
- 17.D — matplotlib figures (reward bar chart, coverage hit-rate plot).
- 17.E — Jinja2 LaTeX template + tectonic invocation.
- Working tree only modifies `tools/vlabs-audit/`. Restricted-path
  diff is empty (the unrelated M3/M4 baseline modification of
  `src/verifiable_labs_envs/cli.py` predates 17.B and is untouched).

---

## 2026-05-01 — Phase 17, Sub-stage 17.B (episode runner + SQLite storage)

**Goal**: drive a parallel batch of episodes against a local SQLite store,
shelling out to the SDK's `verifiable run` for each (env, model, seed).
Resume support for crashed mid-flight runs. CLI wired so `vlabs-audit
audit --config FILE` actually executes episodes (no longer "use
--dry-run").

### Done
- `src/vlabs_audit/storage.py` (~250 LOC): thread-safe `AuditStore` over
  a single SQLite connection (`check_same_thread=False`,
  `isolation_level=None`, internal `threading.Lock`). Schema:
  - `audits(id, model, config_json, started_at, finished_at)` — one row
    per `vlabs-audit audit` invocation.
  - `audit_runs(id, audit_id FK, env, episode_idx, seed, status,
    jsonl_path, reward, error, started_at, finished_at,
    UNIQUE(audit_id, env, episode_idx))` — state machine `pending →
    running → success | failed`.
  - `INSERT OR IGNORE` makes `schedule_episodes` idempotent.
  - `reset_stale_running` recovers crashed `running` rows back to
    `pending` so resume drains them.
  - DB path: `$VLABS_AUDIT_HOME/audits.db`, defaults `~/.vlabs-audit`.
- `src/vlabs_audit/runner.py` (~250 LOC): `EpisodeRunner` with
  `ThreadPoolExecutor(parallel)` workers, dependency-injectable
  `runner_fn` for testing. `default_episode_run` invokes the SDK CLI
  (`verifiable run --env X --model Y --episodes 1 --seed N`), parses
  the `Trace saved to <path>` line from stdout, and relocates the
  trace under the audit's `traces_dir/<env>__seed<n>.jsonl`. 600 s
  subprocess timeout. Falls back gracefully via stdlib `logging`
  (no `structlog` dep).
- `src/vlabs_audit/cli.py`: `audit` command now invokes the runner;
  `--resume <audit_id>` flag drains pending+stale rows of an existing
  audit. Reports `{audit_id, status counts}` on success.
- `tests/test_storage.py` (5 tests): create/get/finish, idempotent
  scheduling, success+failure transitions, stale-running reset,
  `default_home` env override.
- `tests/test_runner.py` (10 tests): batch execution + reward
  recording, failure isolation, real concurrency check (`parallel=4`
  must overlap workers), resume-after-crash drains pending + stale,
  unknown-audit raises, `_parse_trace_path` regex, full
  `default_episode_run` with subprocess + filesystem stubbed,
  PATH-missing + non-zero exit + timeout error paths.
- `tests/test_cli.py`: replaced stale "not implemented" assertion with
  an end-to-end fake-runner CLI test + an unknown-`--resume` failure
  test.

### Tests
- `pytest tools/vlabs-audit/tests/` → 32/32 green in 1.4 s.
- `ruff check tools/vlabs-audit/src tools/vlabs-audit/tests` clean.

### Deferred
- 5 real episodes via `verifiable run`: this WSL has no
  ANTHROPIC_API_KEY/OPENROUTER_API_KEY. The subprocess parsing path is
  fully exercised by `test_default_episode_run_parses_subprocess_output`
  (mocks `subprocess.run` with the SDK's actual stdout layout, asserts
  exact CLI argv). Real-API validation should run on Colab where keys
  are set, before the final 17 commit.

### Open
- 17.C — stats aggregation (mean / 95 % CI / per-env breakdown +
  `vlabs-calibrate` integration).
- Working tree only modifies `tools/vlabs-audit/` plus the M3/M4
  pre-existing baseline; nothing is committed yet (per the 2-commit
  plan at end of Phase 17).

---

## 2026-05-01 — Phase 17, Sub-stage 17.A (skeleton + CLI + YAML config)

**Goal**: scaffold `tools/vlabs-audit/` with a Typer CLI that parses YAML
configs, supports CLI overrides, and exits cleanly on `--dry-run` without
touching network or filesystem outside the test sandbox.

### Done
- `pyproject.toml`: `vlabs-audit` package, deps include `typer`, `pyyaml`,
  `pydantic`, `jinja2`, `matplotlib`, `httpx`, and the pinned
  `vlabs-calibrate==0.1.0a1`. Console script `vlabs-audit`.
- `PHASE_17_PLAN.md`: mirror of the chat-approved plan, eight sub-stages.
- `README.md`: provisional, points at the plan doc.
- `.gitignore`: local audit DB cache, build artefacts.
- `configs/sample.yaml`: Claude Haiku 4.5 audit example (3 envs × 30 ep).
- `src/vlabs_audit/__init__.py`: `__version__ = "0.0.1"`.
- `src/vlabs_audit/config.py`: pydantic `AuditConfig` (forbids extras,
  validates ranges) + `load_config()` that merges YAML and CLI overrides.
- `src/vlabs_audit/cli.py`: Typer app with `audit` and `version` commands;
  `--dry-run` prints the parsed config and exits 0.
- `tests/conftest.py`: `tmp_yaml_config` fixture.
- `tests/test_config.py`: 7 tests covering YAML loading, CLI overrides,
  validation bounds, extra-field rejection, None-override semantics.
- `tests/test_cli.py`: 7 tests covering Typer wiring, `--help`, `version`,
  `--dry-run`, override flags, error paths.

### Tests
- `pytest tools/vlabs-audit/tests/` → 14/14 green.

### Open
- 17.B is the next sub-stage: episode runner + SQLite storage.
- Stelios review of 17.A before proceeding.
