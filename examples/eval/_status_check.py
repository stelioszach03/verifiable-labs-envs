"""Quick combined status check for Phase E + Phase 14."""
import json
from collections import defaultdict
from pathlib import Path

# Phase E
LB = Path("/content/drive/MyDrive/verifiable-labs/leaderboard")
results_path = LB / "results.jsonl"
costs_path = LB / "costs.jsonl"

print("=== PHASE E (leaderboard) ===")
if costs_path.exists():
    last_cost = costs_path.read_text().splitlines()[-1]
    d = json.loads(last_cost)
    print(f"  n_done: {d['n_done']} / {d['n_total']}  ({100*d['n_done']/d['n_total']:.1f}%)")
    print(f"  cumulative_usd: ${d['cumulative_usd']:.3f}")
    print(f"  elapsed_sec: {d['elapsed_sec']:.0f}s ({d['elapsed_sec']/60:.0f} min)")
    print(f"  eta_sec: {d['eta_sec']:.0f}s ({d['eta_sec']/3600:.1f}h remaining)")
    print(f"  timestamp: {d['timestamp']}")

print("\n--- Per-model format validity ---")
counts = defaultdict(lambda: [0, 0])
with results_path.open() as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        m = d["model"]
        counts[m][0] += 1
        if d.get("format_valid"):
            counts[m][1] += 1
for m, (n, ok) in counts.items():
    rate = 100 * ok / n if n else 0
    bar = "█" * int(rate / 5) + "·" * (20 - int(rate / 5))
    print(f"  {m:<35} {ok:>3}/{n:<3} ({rate:>5.1f}%)  {bar}")

# Phase 14
print("\n=== PHASE 14 (multi-env GRPO) ===")
P14 = Path("/content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_multi_v1")
log_path = P14 / "training_log.jsonl"
summary_path = P14 / "training_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print(f"  ✅ COMPLETED: max_steps_reached={summary['max_steps_reached']}")
    print(f"     wall_time_sec={summary['wall_time_sec']:.0f}s")
    print(f"     peak_vram_gb={summary['peak_vram_gb']:.1f}")
elif log_path.exists():
    lines = log_path.read_text().splitlines()
    if lines:
        last = json.loads(lines[-1])
        print(f"  IN PROGRESS: step={last.get('step')} reward={last.get('reward', 'N/A')}")
        print(f"     n_log_events={len(lines)}")
    else:
        print("  IN PROGRESS: no log events yet")
else:
    print("  no log file yet")

# Checkpoints
ckpts = sorted(P14.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if ckpts:
    print(f"\n  checkpoints saved: {[c.name for c in ckpts]}")

# Stdout tail
stdout_path = Path("/content/drive/MyDrive/verifiable-labs/training_outputs/grpo_multi_v1_stdout.log")
if stdout_path.exists():
    text = stdout_path.read_text()
    # Find the latest step progress in tqdm output
    last_lines = text.splitlines()[-3:]
    print("\n  recent stdout:")
    for line in last_lines:
        # tqdm uses \r — show just the last segment
        last_seg = line.split("\r")[-1]
        if last_seg.strip():
            print(f"    {last_seg[:120]}")
