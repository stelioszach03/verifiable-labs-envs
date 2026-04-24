"""Verifiable Labs — Scientific RL Environments leaderboard.

Static Gradio app backed by ``data/llm_benchmark_v2.csv``. Designed for
HuggingFace Spaces deployment. Zero live API calls — the data is
pre-computed in this repo's v2 benchmark.

Tabs:
1. Overview — filterable reward table + heatmap.
2. Methodology — conformal calibration, procedural regeneration,
   multi-turn, and tool-use explainers.
3. Submit — email-capture form (appends to local TSV).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "data" / "llm_benchmark_v2.csv"
HEATMAP_PNG = HERE / "data" / "benchmark_v2_heatmap.png"
SUBMISSIONS_TSV = HERE / "submissions.tsv"

ENV_ORDER = [
    "sparse-fourier-recovery",
    "sparse-fourier-recovery-multiturn",
    "sparse-fourier-recovery-tools",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
    "lodopab-ct-simplified-multiturn",
]
ENV_LABEL = {
    "sparse-fourier-recovery": "SparseF",
    "sparse-fourier-recovery-multiturn": "SparseF-MT",
    "sparse-fourier-recovery-tools": "SparseF-Tools",
    "super-resolution-div2k-x4": "SuperRes",
    "lodopab-ct-simplified": "CT",
    "lodopab-ct-simplified-multiturn": "CT-MT",
}


def _load_final_turn() -> pd.DataFrame:
    """Reward of highest-turn successful row per (model, env, seed)."""
    best: dict[tuple[str, str, int], tuple[int, float, bool]] = {}
    fail_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_counts: dict[tuple[str, str], int] = defaultdict(int)
    with DATA_CSV.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["env"], row["model"], int(row["seed"]))
            total_counts[(row["env"], row["model"])] += 1
            try:
                reward = float(row["reward"])
                turn = int(row["turn"])
                parse_ok = row.get("parse_ok") == "True"
            except (ValueError, TypeError):
                fail_counts[(row["env"], row["model"])] += 1
                continue
            if not parse_ok:
                fail_counts[(row["env"], row["model"])] += 1
                continue
            if key not in best or turn > best[key][0]:
                best[key] = (turn, reward, parse_ok)

    pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (env_name, model, _seed), (_, reward, _ok) in best.items():
        pair[(env_name, model)].append(reward)

    models = sorted({m for (_, m) in pair})
    rows = []
    for model in models:
        row = {"model": model.split("/")[-1]}
        for env_name in ENV_ORDER:
            vals = pair.get((env_name, model), [])
            row[ENV_LABEL[env_name]] = round(fmean(vals), 3) if vals else None
        valid = [v for v in row.values() if isinstance(v, float)]
        row["mean"] = round(sum(valid) / len(valid), 3) if valid else None
        rows.append(row)
    # Append env-mean row
    env_mean = {"model": "** env mean **"}
    for env_name in ENV_ORDER:
        col = [pair[(env_name, m)] for m in models if (env_name, m) in pair]
        flat = [v for sub in col for v in sub]
        env_mean[ENV_LABEL[env_name]] = round(fmean(flat), 3) if flat else None
    env_mean["mean"] = None
    rows.append(env_mean)
    return pd.DataFrame(rows)


def render_overview_table(env_filter: str, model_filter: str) -> pd.DataFrame:
    df = _load_final_turn()
    if env_filter != "all":
        keep = [c for c in df.columns if c in ("model", env_filter, "mean")]
        df = df[keep]
    if model_filter != "all":
        df = df[df["model"].str.contains(model_filter, case=False, na=False) |
                df["model"].str.startswith("**")]
    return df


def submit(name: str, model_id: str, email: str, notes: str) -> str:
    if not model_id.strip():
        return "Please provide a model id (e.g., anthropic/claude-haiku-4.5)."
    if "@" not in (email or ""):
        return "Please provide a valid email so we can reach you."
    SUBMISSIONS_TSV.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not SUBMISSIONS_TSV.exists()
    with SUBMISSIONS_TSV.open("a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if header_needed:
            w.writerow(["timestamp_utc", "name", "model_id", "email", "notes"])
        w.writerow([
            datetime.now(UTC).isoformat(),
            name.strip(), model_id.strip(), email.strip(), notes.strip(),
        ])
    return (
        f"Thanks — we'll benchmark **{model_id.strip()}** in the next batch "
        "and email you at the address provided. Typical batch cadence: every two weeks."
    )


METHODOLOGY_MD = """
## How rewards are computed

Every environment scores with a **weighted rubric** that combines
point-estimate quality (e.g. PSNR / SSIM for images, NMSE + support-F1
for sparse Fourier) with a **conformal-coverage** term that rewards
honest per-entry uncertainty estimates.

- **Split-conformal calibration** (Lei et al. 2018): for each baseline,
  collect residuals on a held-out calibration set, take the
  `(1-alpha)` quantile, and use that as `q_alpha`. At score time the
  interval is `[x_hat - q_alpha * sigma_hat, x_hat + q_alpha * sigma_hat]`.
- The conformal reward peaks when empirical coverage matches the target
  `1-alpha` exactly, and penalises both over-confident (too narrow) and
  over-conservative (too wide) uncertainty predictions.

## Why these envs are contamination-resistant

Every call regenerates the measurement from the seed. There's no fixed
`(y, x*)` pair an adversary can memorise. Details + per-env effective
instance counts in
[CONTAMINATION.md](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/docs/CONTAMINATION.md).

## Multi-turn envs

Three-turn rollouts where the server sends back the forward-model
residual (Fourier for sparse-F, FBP-space for CT) as the user message
between turns. The LLM is expected to propose a correction each turn.
Reward comes from the final turn's prediction; `meta.turn_rewards`
exposes the trajectory.

## Tool-use env

Four explicit tools (`fft`, `ifft`, `ista`, `check_residual`) that the
LLM can call up to five times before emitting its final answer. We
score the final answer; meta records the tool-call count and sequence.

## References

- [conformal.md](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/docs/conformal.md)
- [METHODOLOGY.md](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/docs/METHODOLOGY.md)
- RLVR miscalibration: arXiv:2509.21882, arXiv:2510.00915.
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Verifiable Labs — Scientific RL benchmark") as app:
        gr.Markdown(
            "# Verifiable Labs — Scientific RL Environments leaderboard\n\n"
            "Physics-grounded RL environments for inverse problems, with "
            "conformal-calibrated rewards and procedural measurement "
            "regeneration. [GitHub](https://github.com/stelioszach03/verifiable-labs-envs)."
        )
        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.Markdown(
                    "Mean reward per (model, env) on the final turn. "
                    "Dashes mean no successful row for that pair in this sweep."
                )
                df_initial = _load_final_turn()
                env_choices = ["all"] + [ENV_LABEL[e] for e in ENV_ORDER]
                env_filter = gr.Dropdown(choices=env_choices, value="all",
                                         label="Environment filter")
                model_filter = gr.Textbox(value="all",
                                          label="Model filter (substring, 'all' = no filter)")
                table = gr.Dataframe(value=df_initial, wrap=True)
                refresh = gr.Button("Refresh")
                refresh.click(render_overview_table, inputs=[env_filter, model_filter],
                              outputs=table)
                if HEATMAP_PNG.exists():
                    gr.Image(str(HEATMAP_PNG), label="v2 heatmap",
                             show_label=True, interactive=False)

            with gr.Tab("Methodology"):
                gr.Markdown(METHODOLOGY_MD)

            with gr.Tab("Submit a model"):
                gr.Markdown(
                    "Request a benchmark run of your model. We run a batch "
                    "every ~2 weeks; we'll email you when yours is included."
                )
                name = gr.Textbox(label="Your name (optional)")
                model_id = gr.Textbox(label="OpenRouter model id",
                                      placeholder="anthropic/claude-haiku-4.5")
                email = gr.Textbox(label="Contact email")
                notes = gr.Textbox(label="Notes (optional)", lines=3)
                out = gr.Markdown()
                gr.Button("Submit").click(submit,
                                          inputs=[name, model_id, email, notes],
                                          outputs=out)
    return app


if __name__ == "__main__":
    build_app().launch()
