# Leaderboard — live on HuggingFace Spaces

Sprint 1 Phase 7 deliverable. The Gradio leaderboard is live:

**https://huggingface.co/spaces/stelioszach03/scientific-rl-benchmark**

Three tabs, all backed by the v2 benchmark CSV:

1. **Overview** — filterable reward table per (model, env), with the
   pre-rendered heatmap and download link for the raw CSV.
2. **Methodology** — conformal calibration explainer, procedural-
   regeneration story, multi-turn + tool-use protocol descriptions,
   links to `docs/conformal.md` / `CONTAMINATION.md`.
3. **Submit** — email-capture form for model-benchmark requests. Writes
   to a per-Space `submissions.tsv` on HF's persistent storage.

## Deploy workflow used

```bash
# One-time HF auth (token with `write` scope)
export HF_TOKEN=hf_xxx...

# From the repo root
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = "stelioszach03/scientific-rl-benchmark"
create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio",
            token=os.environ['HF_TOKEN'], exist_ok=True, private=False)
api.upload_folder(
    repo_id=repo_id, repo_type="space",
    folder_path="leaderboard",
    commit_message="Initial leaderboard deploy",
    token=os.environ['HF_TOKEN'],
)
PY
```

## Local run (no HF auth needed)

```bash
cd leaderboard
pip install -r requirements.txt
python app.py
# http://127.0.0.1:7860/
```

## Refreshing the data

The Space bundles `leaderboard/data/llm_benchmark_v2.csv` +
`benchmark_v2_heatmap.png` as the static data source. To refresh:

1. Run `benchmarks/run_v2_benchmark.py` with the desired scope.
2. `python analysis/plot_v2_heatmap.py` to regenerate the heatmap.
3. `cp results/llm_benchmark_v2.csv leaderboard/data/` and repeat for the PNG.
4. Re-upload: `huggingface-cli upload stelioszach03/scientific-rl-benchmark leaderboard --repo-type=space`.

## Security note

The deploy token used in Sprint 1's Task D was shared in chat during the
session. Rotate it at https://huggingface.co/settings/tokens (delete the
`verifiable-labs-deploy` token and mint a fresh one) before any further
automated deploys, since chat logs may be retained.
