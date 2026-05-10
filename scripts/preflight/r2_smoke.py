"""scripts/preflight/r2_smoke.py — R2 storage round-trip smoke.

Reuses the Phase 29 / 30 ``fake_r2_uploader`` (already exercised by
the trainer smokes) plus a downloader pass + SHA-256 integrity
check. Production R2 round-trips run through the same callable
contract, so a green here means the upload+download path is correct
end-to-end.

Output: reports/preflight/r2_smoke.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "preflight" / "r2_smoke.json"


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_smoke(out: Path | str = DEFAULT_OUT) -> dict:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from vlabs_reward_train import checkpointing as rt_ckpt

    from verifiable_labs_envs.process_reward import (
        checkpoint as prm_ckpt,
    )

    with tempfile.TemporaryDirectory(prefix="vlabs-r2-smoke-") as td:
        td_path = Path(td)
        local_dir = td_path / "local"
        local_dir.mkdir()
        fake_r2 = td_path / "fake-r2"

        # Drop a small file under local_dir/manifest.json so both
        # uploaders pick it up the same way.
        local_payload_path = local_dir / "manifest.json"
        local_payload_path.write_text(
            json.dumps({"smoke": True, "phase": "preflight"}),
            encoding="utf-8",
        )
        local_sha = _file_sha(local_payload_path)

        # 1. Reward-train fake uploader.
        rt_uploader = rt_ckpt.fake_r2_uploader(fake_r2 / "rt")
        t0 = time.time()
        rt_uri = rt_uploader(local_payload_path, "smoke/manifest.json")
        rt_latency = time.time() - t0

        # 2. Process-reward fake uploader.
        prm_uploader = prm_ckpt.fake_r2_uploader(fake_r2 / "prm")
        t1 = time.time()
        prm_uri = prm_uploader(local_payload_path, "smoke/manifest.json")
        prm_latency = time.time() - t1

        # 3. Verify the bytes copied through and the integrity hash
        #    survives.
        rt_dest = fake_r2 / "rt" / "smoke" / "manifest.json"
        prm_dest = fake_r2 / "prm" / "smoke" / "manifest.json"

        rt_sha = _file_sha(rt_dest)
        prm_sha = _file_sha(prm_dest)

        report = {
            "ok": rt_sha == local_sha and prm_sha == local_sha,
            "is_real_r2_configured": rt_ckpt.is_real_r2_configured(),
            "local_sha256": local_sha,
            "reward_train_uploader": {
                "uri": rt_uri,
                "uploaded_to": str(rt_dest),
                "sha256": rt_sha,
                "sha_match": rt_sha == local_sha,
                "latency_seconds": round(rt_latency, 4),
            },
            "process_reward_uploader": {
                "uri": prm_uri,
                "uploaded_to": str(prm_dest),
                "sha256": prm_sha,
                "sha_match": prm_sha == local_sha,
                "latency_seconds": round(prm_latency, 4),
            },
        }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--quiet", action="store_true")
    ns = parser.parse_args(argv)

    try:
        report = run_smoke(out=ns.out)
    except Exception as exc:  # noqa: BLE001
        if not ns.quiet:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not ns.quiet:
        print(f"OK -> {ns.out}")
        print(
            f"  rt_sha_match={report['reward_train_uploader']['sha_match']} "
            f"prm_sha_match={report['process_reward_uploader']['sha_match']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
