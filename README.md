# Verifiable Labs

[![CI](https://github.com/stelioszach03/verifiable-labs-envs/actions/workflows/ci.yml/badge.svg)](https://github.com/stelioszach03/verifiable-labs-envs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/verifiable-labs?label=pypi%3Averifiable-labs&color=4c1)](https://pypi.org/project/verifiable-labs/)

> Verifiable Labs is the contamination-proof RL evaluation substrate for AI labs training scientific agents. Procedurally generated environments, classical-solver ground truth, conformal-calibrated rewards.

**Verifiable Labs is the API, SDK, and CLI layer for evaluating and training scientific AI agents on verifiable RL environments.**

Most AI eval tools test chatbots and apps. Verifiable Labs generates scientific environments with **objective rewards**, **calibrated uncertainty**, **procedural regeneration**, **classical baselines**, and **training-signal potential** — tasks that are continuous, uncertainty-sensitive, and impossible to solve by memorising static benchmark answers.

> **Status:** v0.1.0-alpha (developer preview). 10 live environments across compressed sensing, super-resolution, medical CT/MRI, and phase retrieval. Hosted REST API + Python SDK + `verifiable` CLI shipped. The platform is open and rate-limited; treat the public endpoint as a developer playground until v0.2 (auth + Redis sessions). Full roadmap: [`docs/company/roadmap.md`](docs/company/roadmap.md).

- 🔗 Hugging Face leaderboard — https://huggingface.co/spaces/stelioszach03/scientific-rl-benchmark
- 🔗 Prime Intellect Hub envs — [`sparse-fourier-recovery`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery), [`-multiturn`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery-multiturn), [`-tools`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery-tools), [`mri-knee-reconstruction`](https://app.primeintellect.ai/dashboard/environments/stelioszach/mri-knee-reconstruction), [`-multiturn`](https://app.primeintellect.ai/dashboard/environments/stelioszach/mri-knee-reconstruction-multiturn), [`phase-retrieval`](https://app.primeintellect.ai/dashboard/environments/stelioszach/phase-retrieval), [`-multiturn`](https://app.primeintellect.ai/dashboard/environments/stelioszach/phase-retrieval-multiturn), [`super-resolution-div2k-x4`](https://app.primeintellect.ai/dashboard/environments/stelioszach/super-resolution-div2k-x4), [`lodopab-ct-simplified`](https://app.primeintellect.ai/dashboard/environments/stelioszach/lodopab-ct-simplified), [`-multiturn`](https://app.primeintellect.ai/dashboard/environments/stelioszach/lodopab-ct-simplified-multiturn).
- 🔗 Paper (preprint, OpenReview submission pending) — [`paper/main.pdf`](paper/main.pdf)

## Install

```bash
pip install verifiable-labs

# Verify
verifiable --version          # → verifiable-labs 0.1.0a4
verifiable list               # → 10 environments

# Run a benchmark
export OPENROUTER_API_KEY=sk-or-...
verifiable run --env sparse-fourier-recovery \
    --model openai/gpt-4o-mini --episodes 3 --seed 42
```

The `verifiable-labs` PyPI package pulls in the heavy `verifiable-labs-envs` automatically, so the CLI runs every environment locally — no hosted API needed for `pip install verifiable-labs`. For the lightweight HTTP-client surface only, see the [Python SDK](#sdk-quickstart) section below.

## 90-second quickstart

The full developer loop, from clone to a Markdown report a YC reviewer can read.

```bash
git clone https://github.com/stelioszach03/verifiable-labs-envs.git
cd verifiable-labs-envs
pip install -e ".[dev]"

# 1. List the 10 envs.
verifiable envs

# 2. Run a zero-amplitude agent on sparse-Fourier (3 episodes, no API key needed).
verifiable run \
    --env sparse-fourier-recovery \
    --agent examples/agents/zero_agent.py \
    --n 3 --out runs/demo.jsonl

# 3. Render a Markdown evaluation report.
verifiable report --run runs/demo.jsonl --out reports/demo.md

# 4. Compare two agents side-by-side.
verifiable run --env sparse-fourier-recovery --agent examples/agents/random_agent.py \
    --n 3 --out runs/random.jsonl
verifiable compare --runs runs/demo.jsonl runs/random.jsonl
```

The JSONL written by `verifiable run` is a stable schema (`Trace` in [`src/verifiable_labs_envs/traces.py`](src/verifiable_labs_envs/traces.py)) so a CI workflow or a downstream tool can read it without manual parsing. A pre-rendered example lives at [`reports/zero_smoke.md`](reports/zero_smoke.md).

## Three on-ramps

| surface | use when | install |
|---|---|---|
| [`verifiable` CLI](docs/api-reference/cli.md) | local evaluation, CI gates, shipping a reproducible run | `pip install -e ".[dev]"` (this repo) |
| [Python SDK](docs/api-reference/python-sdk.md) | scripted access to the hosted API; sync + async | `pip install verifiable-labs` |
| [Hosted REST API](docs/api-reference/rest-api.md) | language-agnostic eval, no Python needed | `curl https://api.verifiable-labs.com/v1/health` |

### SDK quickstart

```python
from verifiable_labs import Client

with Client(base_url="https://api.verifiable-labs.com") as c:
    env = c.env("stelioszach/sparse-fourier-recovery")
    result = env.evaluate(seed=42, answer=my_model_output)
    print(result.reward, result.components, result.coverage)
```

`AsyncClient` mirrors the sync API one-to-one. The SDK is on PyPI as [`verifiable-labs`](https://pypi.org/project/verifiable-labs/) (current: `0.1.0a4`). The same package also re-exports `load_environment` and `list_environments` so you can drive envs locally without going through the hosted API:

```python
from verifiable_labs import load_environment, list_environments

print(list_environments())              # ['lodopab-ct-simplified', ...]
env = load_environment("sparse-fourier-recovery")
result = env.run_baseline(seed=42)
print(result["reward"])
```

### Hosted API quickstart

```bash
curl https://api.verifiable-labs.com/v1/health
# {"status": "ok", "version": "0.1.0-alpha", ...}
curl https://api.verifiable-labs.com/v1/environments | jq '.environments[].id'
```

OpenAPI UI at `/docs`. v0.1 is unauthenticated and rate-limited (30 req/min/IP); v0.2 adds per-user keys. See [`deploy/api/README.md`](deploy/api/README.md) for self-hosting (Render / Fly.io / Docker).

## Onboarding your own agent

The CLI's `--agent` flag accepts three forms:

```bash
# 1. Python file with a top-level `solve(observation: dict) -> dict`.
verifiable run --env <id> --agent path/to/my_agent.py --n 5 --out runs/me.jsonl

# 2. Subprocess (any language). Reads JSON on stdin, writes JSON on stdout.
verifiable run --env <id> --agent "cmd:./my_solver --quiet" --n 5 --out runs/me.jsonl

# 3. OpenAI-compatible HTTP endpoint (OpenAI / OpenRouter / local vLLM / etc.).
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    verifiable run --env <id> --agent "openai:anthropic/claude-haiku-4.5" \
    --n 5 --out runs/llm.jsonl
```

Five-minute onboarding guide: [`docs/ONBOARD_AGENT_5_MIN.md`](docs/ONBOARD_AGENT_5_MIN.md).
Examples: [`examples/agents/`](examples/agents).

## What exists today vs what's next

| | shipped | planned |
|---|---|---|
| **environments** | 10 envs across 5 domains | 5 new envs (holographic 3D, EM tomography, seismic FWI, inverse rendering, protein distogram) — v0.2 |
| **API** | `/v1/{health, environments, sessions, leaderboard}` (open, rate-limited) | per-user auth, Redis-backed sessions — v0.2 |
| **SDK** | sync + async clients on PyPI as `verifiable-labs` (re-exports `load_environment` for local mode) | optional slim install — Tier-1 polish |
| **CLI** | `envs · run · compare · report · init-env · validate-env` | static viewer / dashboard — v0.3 stretch |
| **training signal** | prompt-search proof in [`notebooks/training_proof.ipynb`](notebooks/training_proof.ipynb), heuristic search in [`examples/training_signal_demo.py`](examples/training_signal_demo.py) | TRL / vLLM bindings — v0.2 |
| **compliance** | aggregate report template + PDF generator | real attestation system — v0.3 speculative |

Full roadmap: [`docs/company/roadmap.md`](docs/company/roadmap.md).

---

## Research findings (meta-benchmark v3, 2026-04-24)

Three honest takeaways from the v0.1 benchmark sweep:

1. **Classical baselines still beat every tested LLM on every env** — the battery is not saturated.
2. **Sparse compressed-sensing outputs are the hardest for LLMs** (sparse-F and phase-retrieval cluster at ~0.35 mean across 3 cheap models). 2D-image envs (MRI / super-res / CT) are 2× easier for LLMs because they can parrot a provided classical baseline.
3. **Claude Haiku 4.5 is the best cheap model for scientific reasoning**, with a 0.604 cross-env mean — consistently ahead of GPT-5.4-mini (0.465) and GPT-5.4-nano (0.458). Full table in [`results/meta_benchmark_v3_summary.md`](results/meta_benchmark_v3_summary.md).

## Why "verifiable"

Frontier reasoning models are trained with verifiable rewards (RLVR). Today's RL environments are mostly text-only, saturate quickly, and miss the continuous, ill-posed reasoning that real science requires. This package provides environments where:

1. The **forward operator** is exact and JIT-compiled (JAX), so a model must actually invert physics.
2. The **reward** is a weighted sum of reconstruction quality (PSNR, SSIM, or task-appropriate metric) and **conformal-prediction coverage** — models are rewarded for honest posterior width, not overconfident point estimates.
3. Measurements are **procedurally regenerated per evaluation call**, so fixed-string memorization is structurally impossible.

## Environments (10 live on Prime Intellect Hub)

| # | Environment | Domain | Forward operator | Classical baseline |
|---|---|---|---|---|
| 1 | `sparse-fourier-recovery` | compressed sensing | subsampled orthonormal 1D DFT | OMP with LS-covariance σ̂ |
| 2 | `sparse-fourier-recovery-multiturn` | compressed sensing | same, 3-turn dialogue | residual-feedback refinement |
| 3 | `sparse-fourier-recovery-tools` | compressed sensing | same, primitive-composition tool-use | fft/ifft/soft-threshold/residual/norm primitives |
| 4 | `super-resolution-div2k-x4` | image | Gaussian blur + 4× decimation | bicubic with edge-weighted σ̂ |
| 5 | `lodopab-ct-simplified` | medical imaging (CT) | 2D parallel-beam Radon | FBP with edge-weighted σ̂ (phantom default; real-patient LoDoPaB-CT via `use_real_data=True`) |
| 6 | `lodopab-ct-simplified-multiturn` | medical imaging (CT) | same, 3-turn dialogue | FBP-residual feedback |
| 7 | **`phase-retrieval`** (new sprint-giga) | crystallography / CDI | magnitude-only subsampled DFT | Gerchberg-Saxton (alternating projection) |
| 8 | **`phase-retrieval-multiturn`** (new) | crystallography / CDI | same, 3-turn dialogue | magnitude-residual feedback |
| 9 | **`mri-knee-reconstruction`** (new sprint-giga) | medical imaging (MRI) | 2D DFT + 4× Cartesian undersampling | zero-filled inverse FFT |
|10 | **`mri-knee-reconstruction-multiturn`** (new) | medical imaging (MRI) | same, 3-turn dialogue | k-space residual feedback |

## Classical-baseline benchmark (5 seeds each, default hyperparameters)

| environment | reference reward | zero reward | gap | conformal q |
|---|---:|---:|---:|---:|
| `lodopab-ct-simplified` | 0.712 | 0.151 | +0.561 | 0.241 |
| `sparse-fourier-recovery` | 0.869 | 0.336 | +0.533 | 1.587 |
| `super-resolution-div2k-x4` | 0.629 | 0.425 | +0.203 | 2.167 |

Reproduce with `python benchmarks/run_all.py --seeds 5`.

## Multi-turn rollouts (`sparse-fourier-recovery-multiturn`)

Ships a 3-turn conversation variant of `sparse-fourier-recovery`: turn 1 is the full problem, turns 2–3 show the Fourier-domain residual `r = y - A(x_hat)` of the previous answer and ask for a correction.

Async benchmark (3 models × 3 instances × 3 turns = 27 calls, $0.09 total, 33.6 s wall-clock with `Semaphore(10)`):

| Model | Turn 0 → Turn 1 → Turn 2 | Final | Episodes failed |
|---|---|---:|---|
| Claude Haiku 4.5 | 0.371 → 0.380 → 0.363 | 0.363 | 0/3 |
| Claude Sonnet 4.6 | 0.348 → 0.348 → 0.347 | 0.347 | 2/3 (turn-1 parse) |
| GPT-5.4 mini | 0.353 → 0.331 → 0.331 | 0.331 | 0/3 |

Headline finding: **frontier LLMs do not yet know how to use residual feedback constructively on sparse-Fourier recovery.** Scores plateau or regress at turns 2–3. This is itself the most actionable signal in the entire benchmark — it's exactly the surface RLVR post-training on these environments would be expected to improve.

Raw data: [`results/multiturn_sparse_fourier_recovery_multiturn.csv`](results/multiturn_sparse_fourier_recovery_multiturn.csv). Plot: [`results/multiturn_sparse_fourier_recovery_multiturn_curves.png`](results/multiturn_sparse_fourier_recovery_multiturn_curves.png).

## Tool-use rollouts (`sparse-fourier-recovery-tools`)

Same underlying problem as `sparse-fourier-recovery`, but the LLM is given **5 Python primitive tools** it must compose itself over ISTA-like iterations before committing to a final answer. No tool returns a full reconstruction on its own — the model has to iterate `forward → residual → adjoint → threshold` to converge.

- `fft_tool(signal_x1000)` → apply ``A = S·F`` to a length-n dense candidate.
- `ifft_tool(spectrum_re_x1000, spectrum_im_x1000)` → adjoint of A (zero-fill at mask + inverse DFT).
- `threshold_tool(signal_x1000, tau_x1000)` → elementwise soft-threshold (the ISTA proximal step).
- `compute_residual_tool(signal_x1000)` → returns `r = y − A(x)` + L2 / max-abs.
- `sparsity_norm_tool(signal_x1000)` → returns ‖x‖₁, ‖x‖₂, nonzero count.

Cap: 30 tool calls per episode (rebench used 5–15). Tools reference instance-bound state so call payloads stay small.

> **History — v0.1 was an oracle-delegation artifact.** The original
> tool-use env exposed an `ista_tool()` that returned the OMP oracle's
> answer. In the Task-4.1 benchmark all three tested models called it
> once and scored a byte-identical **0.858** per seed — the fingerprint
> of oracle adoption, not reasoning. v0.3 (2026-04-24 polish) removes
> `ista_tool` and replaces it with the five primitives above. A
> regression test (`test_no_single_tool_call_leaks_the_answer`) verifies
> no primitive transmits the target to the model.

v0.3 rebench (3 cheap models × 3 seeds, **$0.64 total** under $1 cap):

| Model | Mean reward (parsed) | Parse fails | Best episode |
|---|---:|---|---:|
| Claude Haiku 4.5 | 0.404 (n=1) | 1/2 seeds | 0.404 |
| GPT-5.4 mini | 0.403 (n=3) | 0/3 | 0.408 |
| GPT-5.4 nano | — | 3/3 | FAIL |

Empty-answer floor ≈ **0.354**, classical OMP baseline ≈ **0.931**.
All parsed rewards cluster just above the empty-answer floor — the
primitive tool set is genuinely hard, cheap LLMs cannot yet compose
ISTA from primitives. Tool sequences differ across models (no
byte-identical v0.1-style pattern). Full analysis:
[`results/sparse_fourier_reconciliation.md`](results/sparse_fourier_reconciliation.md)
("v0.3 follow-up"). Raw data:
[`results/llm_benchmark_tools_v2.csv`](results/llm_benchmark_tools_v2.csv). Reproduce:

```bash
python benchmarks/run_tools_v2_rebench.py \
  --models anthropic/claude-haiku-4.5,openai/gpt-5.4-mini,openai/gpt-5.4-nano \
  --n-instances 3 --max-tool-calls 5 --max-cost 0.30 --conformal-quantile 1.587
```

### Multi-turn CT (`lodopab-ct-simplified-multiturn`, phantom mode, 3 models × 3 instances × 3 turns, $0.56, 141 s)

Same 3-turn design: turn 1 takes a 32×32 FBP, turn 2–3 take the sinogram-residual back-projection (downsampled to 32×32, encoded as signed int8 + scale factor).

| Model | Turn 0 → Turn 1 → Turn 2 | Final mean | Episodes failed |
|---|---|---:|---|
| Claude Sonnet 4.6 | 0.618 → 0.645 → **0.657** | 0.657 | 1/3 (turn-1 parse) |
| GPT-5.4 mini | 0.622 → 0.642 → 0.641 | 0.622 | 1/3 (turn-2 parse) |
| Claude Haiku 4.5 | 0.626 → 0.488 → **0.344** | 0.550 | 2/3 (turn-2 parse) |

**Key finding (different from sparse-F!)**: Sonnet 4.6 improves **monotonically** across turns (+3.9 pp), GPT-5.4 mini plateaus after the first-turn bump, and Claude Haiku 4.5 **regresses severely** (−28.2 pp turn 2 → turn 3) — the residual-image feedback actively confuses it. Multi-turn rollouts surface a differential capability that single-turn scores completely mask.

Raw data: [`results/multiturn_lodopab_ct_simplified_multiturn.csv`](results/multiturn_lodopab_ct_simplified_multiturn.csv). Plot: [`results/multiturn_lodopab_ct_simplified_multiturn_curves.png`](results/multiturn_lodopab_ct_simplified_multiturn_curves.png).

Reproduce with:

```bash
python benchmarks/run_multiturn_benchmark.py \
  --env sparse-fourier-recovery-multiturn \
  --models anthropic/claude-haiku-4.5,anthropic/claude-sonnet-4.6,openai/gpt-5.4-mini \
  --n 3 --max-turns 3 --max-cost 2.0 --conformal-quantile 1.587
```

## Real-data CT (LoDoPaB-CT validation, opt-in via `use_real_data=True`)

Phase 2 adds a real-patient-geometry path on `lodopab-ct-simplified`: 3552 validation slices from the LoDoPaB-CT dataset (Leuschner et al. 2021, Nature Scientific Data) drawn from the LIDC-IDRI clinical chest-CT cohort. CI defaults stay on the phantom rotation so no download is required. One-shot activation:

```bash
bash scripts/download_lodopab_validation.sh      # ~1.5 GB zip, 28 HDF5 chunks
python -c "from verifiable_labs_envs.envs import lodopab_ct as ct; print(ct.load_environment(use_real_data=True).run_baseline(seed=0))"
```

Spot-check numbers (this repo, Apr 2026):

| Solver | Mode | Mean reward | Notes |
|---|---|---:|---|
| Classical FBP | phantom (5 seeds) | 0.712 | Sprint 0 baseline |
| Classical FBP | **real (10 seeds)** | **0.731** | mean PSNR 0.62, SSIM 0.64 — real CT is structurally cleaner than the synthetic phantoms |
| Claude Haiku 4.5 | phantom (5 seeds) | 0.615 | Sprint 0 0/5 parse-fail |
| Claude Haiku 4.5 | **real (3 seeds)** | 0.694 on 1/3 success | 2/3 parse-fails — "expected 32 entries, got 31" on seeds 0 and 2. Real CT grids are harder for the model to transcribe without losing count than the phantom pattern. |

Raw data: [`results/ct_real_spotcheck.csv`](results/ct_real_spotcheck.csv).

## v2 benchmark — 4 models × 6 envs (Sprint 1)

Full 6-environment sweep including multi-turn and tool-use variants. Opus 4.7 dropped from this sweep because Sonnet ≈ Opus within noise in Sprint 0 and keeping it would have blown the $3 cap.

| model | SparseF | SparseF-MT | SparseF-Tools | SuperRes | CT | CT-MT | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-4.5 | 0.364 | 0.351 | 0.334 | 0.726 | 0.640 | 0.527 | **0.490** |
| claude-sonnet-4.6 | 0.305 | 0.328 | 0.337 | 0.726 | 0.580 | 0.640 | **0.486** |
| gpt-5.4 | 0.293 | 0.365 | 0.306 | 0.721 | 0.601 | 0.654 | **0.490** |
| gpt-5.4-mini | 0.338 | 0.363 | 0.354 | 0.534 | 0.505 | 0.371 | **0.411** |
| **env mean** | 0.325 | 0.352 | 0.333 | 0.677 | 0.581 | 0.548 | |

Three new findings the v2 sweep surfaces:

- **Multi-turn helps *frontier* models on CT, hurts *small* models** — GPT-5.4 CT 0.60 → CT-MT 0.65, Sonnet 0.58 → 0.64; Haiku CT 0.64 → CT-MT 0.53, mini 0.51 → 0.37. Budget models can't maintain coherence across the residual-feedback protocol; frontier models use the extra turns productively.
- **Sparse-Fourier stays flat across single-turn / multi-turn / tool-use** (all 0.29–0.37). No rollout format unlocks compressed sensing for any tested model. The `SparseF-Tools` column in the v2 table above was a v0.1 run where the tool-use env still shipped the `ista_tool` oracle; after the v0.3 rebench with primitive-only tools (see the tool-use section above), cheap LLMs cluster right at the empty-answer floor — reinforcing this finding, not contradicting it.
- **SuperRes saturates for the Claude-Sonnet / Claude-Haiku / GPT-5.4 cluster** at ~0.72–0.73, with GPT-5.4-mini trailing at 0.53. Compression-style image denoising is the easiest task in the battery; all frontier models converge.

Heatmap: [`results/benchmark_v2_heatmap.png`](results/benchmark_v2_heatmap.png). Raw data: [`results/llm_benchmark_v2.csv`](results/llm_benchmark_v2.csv). Full summary with caveats: [`results/benchmark_v2_summary.md`](results/benchmark_v2_summary.md).

## LLM benchmark v1 (OpenRouter, 5 seeds each, total spend $1.89)

| Model | SparseFourier | SuperRes | LoDoPaB-CT | Mean (3 envs) |
|---|---:|---:|---:|---:|
| **Reference baseline (OMP / bicubic / FBP)** | **0.869** | **0.629** | **0.712** | **0.737** |
| Claude Opus 4.7 | 0.300 | 0.628 | 0.625 | 0.518 |
| Claude Sonnet 4.6 | 0.316 | 0.629 | 0.595 | 0.513 |
| **Claude Haiku 4.5** | 0.361 | 0.625 | 0.615 | **0.534** |
| GPT-5.4 | 0.311 | 0.601 | 0.571 | 0.494 |
| GPT-5.4 mini | 0.340 | 0.464 *(1/5 fail)* | 0.578 *(1/5 fail)* | 0.460 |
| GPT-5.4 nano | 0.350 | 0.528 *(2/6 fail)* | 0.197 *(4/6 fail)* | 0.358 |
| Zero baseline | 0.336 | 0.425 | 0.151 | 0.304 |

Clean discrimination across model tiers and clean rank-ordering against the expert classical baselines. The environments measure capability, not chance:

- **Classical expert algorithms (mean 0.737) beat every general-purpose LLM** on these inverse problems.
- **Sparse-Fourier is a weak LLM discriminator** (all models 0.30–0.36, barely above zero baseline 0.336) — compressed sensing is not yet a text-completion task.
- **Super-resolution and CT produce a useful ranking** (Haiku / Sonnet / Opus / GPT-5.4 cluster at ~0.60, small models drop off).
- **JSON-count parse-failure rate scales inversely with model size**: `gpt-5.4-nano` fails 33% of grid outputs, `gpt-5.4-mini` 11%, everything Haiku-and-above 0% — a legitimate discrimination axis on its own.
- **Cross-env correlation matrix** (Spearman, n=6 models): SuperRes ↔ CT = +0.66 (same structural task); SparseF ↔ image envs = −0.26 to −0.37 (different capabilities). The three envs measure different things. Full methodology in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md); heatmap in [`results/env_correlation_heatmap.png`](results/env_correlation_heatmap.png).

![Cross-env Spearman correlation heatmap](results/env_correlation_heatmap.png)

Reproduce with `python benchmarks/run_llm_benchmark.py --preset paid-full`. See [`results/llm_benchmark.md`](results/llm_benchmark.md) for the full analysis and [`results/llm_benchmark.csv`](results/llm_benchmark.csv) for per-call raw data.

## Install

### Full monorepo (developers + research use)

```bash
git clone https://github.com/stelioszach03/verifiable-labs-envs
cd verifiable-labs-envs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",api" to also install the FastAPI server
pytest                            # 254+ tests green
```

> **macOS + iCloud Drive — known venv gotcha (read this if you live in `~/Documents/`).**
> If your repo clone is inside an iCloud-synced folder (the default `~/Documents/`
> is synced if "Desktop & Documents Folders" is on in System Settings → Apple ID →
> iCloud), iCloud Drive will sporadically corrupt the editable install's `.pth`
> file by hardlinking it to a stale cached copy. The symptom is
> `ModuleNotFoundError: No module named 'verifiable_labs_envs'` (or
> `verifiable_labs_api`) right after a successful `pip install -e .`. The smoking
> gun is link count `2` on `.venv/lib/python3.11/site-packages/_editable_impl_verifiable_labs_envs.pth`.
>
> **Fix (recommended): create the venv outside iCloud-synced storage and symlink
> it back in:**
>
> ```bash
> deactivate 2>/dev/null
> mv .venv .venv.broken-icloud           # keep for forensics; delete later
> mkdir -p ~/.venvs
> python3.11 -m venv --copies ~/.venvs/verifiable-labs
> ln -s ~/.venvs/verifiable-labs .venv   # all existing scripts still work
> source .venv/bin/activate
> pip install -e ".[dev,api]"
> python -c "import verifiable_labs_envs, verifiable_labs_api; print('OK')"
> ```
>
> **Alternative (in-place):** Apple respects the `.nosync` suffix on directories
> as an "exclude from iCloud sync" flag. Rename `.venv` → `.venv.nosync` and
> symlink:
>
> ```bash
> mv .venv .venv.nosync
> ln -s .venv.nosync .venv
> pip install -e ".[dev,api]" --force-reinstall
> ```
>
> Linux and Windows installers are unaffected.

### Single environment via Prime Intellect Hub (now live)

All six envs are published on the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments):

```bash
pip install prime
prime login
prime env install stelioszach/sparse-fourier-recovery
# or any of: sparse-fourier-recovery-multiturn, sparse-fourier-recovery-tools,
#            super-resolution-div2k-x4, lodopab-ct-simplified, lodopab-ct-simplified-multiturn
```

### Single environment via GitHub subdirectory

```bash
pip install "git+https://github.com/stelioszach03/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-sparse-fourier"
# or: verifiable-labs-sparse-fourier-multiturn, -tools, super-resolution,
#     lodopab-ct, lodopab-ct-multiturn, envs-core
```

Then:

```python
from verifiable_labs_sparse_fourier import load_environment
env = load_environment()
out = env.run_baseline(seed=0)
```

## Quickstart

```python
from verifiable_labs_envs import load_environment

env = load_environment("sparse-fourier-recovery")
result = env.run_baseline(seed=0)
print(result["reward"])            # e.g. 0.931
print(result["components"])        # {"nmse": 0.977, "support": 0.900, "conformal": 0.900}
print(result["meta"]["coverage"])  # 0.80 — fraction of support entries inside the conformal interval
```

Any custom solver can be scored by returning a `Prediction(x_hat, sigma_hat, support_hat=...)`
and passing it to `env.score(prediction, instance)`.

Walkthrough across all three environments:

```bash
python examples/quickstart.py
```

## Contamination resistance

Every environment in this repo is structurally resistant to the three attacks that have hollowed out static text benchmarks: train-set leakage, answer-string matching, and distribution creep. Full analysis in [`docs/CONTAMINATION.md`](docs/CONTAMINATION.md). Headline numbers:

- `sparse-fourier-recovery` — the per-instance state space is continuous (10 real-valued amplitudes + 128 real-valued complex-noise coordinates), on top of `C(256, 10) × C(256, 64) ≈ 10⁷³` combinatorial arrangements of support and mask.
- `super-resolution-div2k-x4` and `lodopab-ct-simplified` — the discrete image / phantom set is small (6 and 5 respectively, a known v0.0.1 weakness flagged in the doc), but measurement noise is regenerated per call so memorizing the HR image doesn't reproduce the measurement.
- An empirical memorization probe at [`scripts/memorization_probe.py`](scripts/memorization_probe.py) confirms: across Haiku 4.5, GPT-5.4 mini, and GPT-5.4 nano on `sparse-fourier-recovery`, all three models show cross-seed reward std ≥ 0.02 (no constant-output signatures). Raw data: [`results/memorization_probe.csv`](results/memorization_probe.csv).

## Documentation

- [`docs/conformal.md`](docs/conformal.md) — the conformal-coverage reward term: why it's there, how it's calibrated, what it rewards.
- [`docs/CONTAMINATION.md`](docs/CONTAMINATION.md) — contamination resistance analysis, per-env effective instance count, empirical probe methodology.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — benchmark aggregation, cross-env correlation interpretation, failure taxonomy.
- [`docs/LEADERBOARD.md`](docs/LEADERBOARD.md) — Gradio leaderboard + HF Spaces deploy.

## Leaderboard

Static [Gradio leaderboard](leaderboard/app.py) backed by the v2 benchmark CSV — three tabs (Overview, Methodology, Submit). Run locally:

```bash
cd leaderboard && pip install -r requirements.txt && python app.py
```

HF Spaces deploy pending `HF_TOKEN` setup; see [`docs/LEADERBOARD.md`](docs/LEADERBOARD.md) for the exact deploy command.

## Author

Stelios Zacharioudakis — finishing BSc CS at the University of Athens (NKUA). Research on calibrated astronomical inverse imaging.

## License

Apache 2.0. See `LICENSE`.
