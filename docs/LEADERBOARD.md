# Leaderboard — HuggingFace Spaces

Phase 7 of Sprint 1 ships a Gradio leaderboard under `leaderboard/`, ready
to deploy to HuggingFace Spaces. The app runs locally today; the Space
publish is paused pending a `HF_TOKEN` that was not configured during
this session.

## What exists

- `leaderboard/app.py` — 3-tab Gradio app (Overview / Methodology / Submit).
- `leaderboard/requirements.txt` — gradio + pandas + plotly.
- `leaderboard/data/llm_benchmark_v2.csv` — the 78-row v2 snapshot.
- `leaderboard/data/benchmark_v2_heatmap.png` — pre-rendered heatmap.
- `leaderboard/README.md` — the HF Spaces metadata header
  (`sdk: gradio`, `sdk_version: 4.36.0`, `app_file: app.py`) so the
  moment a `HF_TOKEN` is available the folder uploads as a working Space.

## Local run (always works)

```bash
cd leaderboard
pip install -r requirements.txt
python app.py
# opens http://127.0.0.1:7860/
```

## HuggingFace Spaces deploy (pending auth)

1. Obtain an HF token with `write` scope at
   https://huggingface.co/settings/tokens and export it:

   ```bash
   export HF_TOKEN=hf_xxx...
   ```

2. Create the Space + upload:

   ```bash
   pip install huggingface_hub
   python - <<'PY'
   import os
   from huggingface_hub import HfApi
   api = HfApi()
   repo_id = "verifiable-labs/scientific-rl-benchmark"
   api.create_repo(repo_id=repo_id, repo_type="space",
                   space_sdk="gradio", token=os.environ["HF_TOKEN"],
                   exist_ok=True)
   api.upload_folder(repo_id=repo_id, repo_type="space",
                     folder_path="leaderboard",
                     token=os.environ["HF_TOKEN"])
   PY
   ```

3. Visit
   https://huggingface.co/spaces/verifiable-labs/scientific-rl-benchmark
   to confirm; link it from the main README's "Leaderboard" section once
   live.

## Why deploy isn't completed in this sprint

`HF_TOKEN` is not present in the project's `.env`. The deploy requires
a one-time browser flow at huggingface.co to mint a token, which we
can't complete headlessly. The entire `leaderboard/` folder is self-
contained and deploy-ready — the only missing ingredient is auth.
