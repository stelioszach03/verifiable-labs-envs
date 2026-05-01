# Phase 17 — `vlabs-audit` capability report CLI

> Mirror of the planning doc shared in chat. Sub-stages 17.A → 17.H, eight in
> total. Each ends with green tests + a visible artefact. Two commits at the
> end (`feat(audit):` for code, `docs(audit):` for sample PDF + README).

## 0. Scope notes from quick exploration

| Existing | Status | Reuse posture |
|---|---|---|
| `leaderboard/app.py` + `data/` | Streamlit dashboard, no episode runner library | **Low reuse** — only conceptual borrowing |
| `src/verifiable_labs_envs/cli.py` (dev CLI) | Episode runner with provider routing | **High reuse** — call via subprocess |
| `packages/verifiable-labs/src/verifiable_labs/cli.py` (user-facing) | OpenRouter routing | **High reuse** — already pip-installed |
| `vlabs-calibrate==0.1.0a1` (PyPI) | `vc.calibrate(reward_fn, traces, alpha)` | **Direct dep** |

Confirmed: **additive only** — no existing path needs editing.

## 1. Architecture

```
$ vlabs-audit --config configs/sonnet46.yaml --output reports/sonnet46_audit.pdf

┌────────────────────────────────────────────────────────┐
│ tools/vlabs-audit/                                     │
│ src/vlabs_audit/                                       │
│   cli.py        ── Typer app, parses CLI/YAML config   │
│   config.py     ── pydantic AuditConfig                │
│   runner.py     ── shells out to `verifiable run` per  │
│                    env × episode, captures JSONL       │
│   storage.py    ── local SQLite                        │
│                    ~/.vlabs-audit/audits.db            │
│                    (audit_id, model, env, traces)      │
│   stats.py      ── reads JSONL → vc.calibrate()        │
│                    → empirical coverage, deltas        │
│   figures.py    ── matplotlib → PDF figures            │
│   latex.py      ── Jinja2 → .tex → tectonic → PDF      │
│   anonymize.py  ── model-name redaction for sample     │
│   report.py     ── orchestration                       │
│ templates/                                             │
│   main.tex.j2                                          │
│   sections/{exec_summary,methodology,results,          │
│             recommendations}.tex.j2                    │
│   style/vlabs_report.cls                               │
│ tests/                                                 │
│ pyproject.toml, README.md, SESSION_LOG.md              │
└────────────────────────────────────────────────────────┘
```

## 2. Sub-stages

| Stage | Deliverable | Days |
|---|---|---:|
| 17.A | Skeleton + Typer CLI + YAML config + 14 tests | 2 |
| 17.B | Episode runner + SQLite audit storage + 8 tests | 3 |
| 17.C | Stats aggregation via `vlabs_calibrate` + 6 tests | 2 |
| 17.D | Matplotlib figures (4 types) + 4 tests | 2 |
| 17.E | LaTeX templates + tectonic compile + 8 tests | 4 |
| 17.F | Sample run (Haiku 4.5 + GPT-4o-mini, 3 envs × 30 ep) + anonymize | 1 |
| 17.G | Test polish + CI integration | 2 |
| 17.H | Documentation | 1 |
| **Total** | | **17** |
| Slack | | 3 |
| **Realistic** | | **20 days = 4 weeks solo** |

## 3. Approved decisions (Stelios, 2026-05-01)

| # | Decision |
|---|---|
| 1 | Audit DB: **local SQLite** at `~/.vlabs-audit/audits.db` |
| 2 | CLI library: **Typer** |
| 3 | Sample report models: **Claude Haiku 4.5 + GPT-4o-mini** (no Gemini) |
| 4 | `reports/sample_anonymized.pdf`: **commit it** to the repo |
| 5 | Phase 13.x GRPO results: **skip** for v0.0.1 |
| 6 | Output format: **PDF only** for v0.0.1 |
| 7 | Supabase Storage upload: **defer** to v0.0.2 |
| 8 | Tectonic install fallback: **hard error** with install URL |

Plus: pin `vlabs-calibrate==0.1.0a1`; sample PDF requires Stelios review before commit; no AI references in commits; OpenRouter API key from env vars only.

## 4. Cost estimate (sample run)

3 envs × 30 episodes × 2 models = 180 episodes.

| Model | $/M input | $/M output | Cost / 12K episode | × 90 episodes |
|---|---:|---:|---:|---:|
| Claude Haiku 4.5 | 1.00 | 5.00 | $0.028 | $2.52 |
| GPT-4o-mini | 0.15 | 0.60 | $0.0036 | $0.32 |
| OpenRouter markup ~5% | | | | $0.14 |
| Buffer for retries (×1.3) | | | | +$0.92 |
| **Sample total** | | | | **~$4** |

User's $5–10 estimate is comfortable. Re-confirmation will be requested
before 17.F per the execution order.

## 5. Risk register (top 5)

| # | Risk | Mitigation |
|---|---|---|
| R1 | LaTeX compile fails on edge cases (escape, unicode, long cells) | Aggressive escape filter, snapshot tests, `tectonic` returns 0 only on success |
| R2 | OpenRouter rate-limits during sample run | Sequential by default + `--parallel`, exp backoff on 429, resume support per 17.B |
| R3 | tectonic install in CI is slow / breaks | Cache the binary in GitHub Actions; fallback `pdflatex` documented |
| R4 | Sample report leaks model identity | Anonymize at render + figure axes; recommendations stay generic |
| R5 | `traces.py` schema evolves and breaks parsing | Pin `vlabs-calibrate==0.1.0a1`; raw JSONL kept in audit storage so reparse is local fix |

## 6. Validation criteria — overview

| Sub-stage | Criterion |
|---|---|
| 17.A | `vlabs-audit --dry-run` prints parsed config; 14 unit tests green |
| 17.B | 5 real episodes complete + persisted; resume works; 8 tests green |
| 17.C | Numbers match `verifiable compare` baseline; 6 tests green |
| 17.D | 4–7 PDF figures, vector, embed cleanly; 4 tests green |
| 17.E | Full PDF compiles, 8–15 pages, all sections render; 8 tests green |
| 17.F | `reports/sample_anonymized.pdf` exists, no model names visible |
| 17.G | Repo-root pytest 609+ green, CI green |
| 17.H | README walks fresh dev to working sample in 15 min |

## 7. Execution order (per Stelios's spec)

1. Create this `PHASE_17_PLAN.md` file.
2. Implement **17.A** (skeleton + CLI + YAML config).
3. **STOP**, await review.
4. Implement 17.B → 17.E sequentially, brief status updates between.
5. Before 17.F: re-confirm cost estimate, await go-ahead for OpenRouter spend.
6. Implement 17.F.
7. **STOP**, show sample PDF, await review.
8. After approval, 17.G + 17.H, then **two commits**:
   - `feat(audit): Phase 17 — vlabs-audit CLI + LaTeX templates`
   - `docs(audit): Phase 17 sample report + README`
