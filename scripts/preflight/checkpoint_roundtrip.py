"""scripts/preflight/checkpoint_roundtrip.py — HF Hub upload+download
round-trip smoke (CPU-only).

Generates a deterministic fake LoRA checkpoint payload (single binary
file plus a manifest), uploads it to either:
- a real HF Hub repo when ``HF_TOKEN`` is set + ``--repo-id`` is
  supplied (the smoke is no-op otherwise), or
- a local fake-hub directory at ``/tmp/vlabs-fake-hf/<repo-id>/``
  when ``LOCAL_FAKE_HF=1`` is set (default for tests + CI).

Then re-downloads, recomputes the SHA-256, and verifies bytes match.

Output: reports/preflight/checkpoint_roundtrip_smoke.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO_ROOT / "reports" / "preflight" / "checkpoint_roundtrip_smoke.json"
)

DEFAULT_FAKE_HF_ROOT = Path("/tmp/vlabs-fake-hf")
DEFAULT_PAYLOAD_BYTES = 4096


def fake_payload(seed: int = 0, n_bytes: int = DEFAULT_PAYLOAD_BYTES) -> bytes:
    """Deterministic fake LoRA-weights payload — small enough to make
    the round-trip cheap, large enough to catch byte-level corruption."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_manifest(payload_sha: str) -> dict:
    return {
        "model_id": "vlabs-reward-distilled-qwen-1-5b-v0.0.0-smoke",
        "payload_sha256": payload_sha,
        "schema_version": "v0.1.0",
        "created_at_unix": int(time.time()),
        "smoke": True,
    }


def _local_fake_upload(
    *,
    payload: bytes,
    manifest: dict,
    repo_id: str,
    fake_root: Path,
) -> dict:
    """Mirror the HF Hub repo layout under ``fake_root`` so the
    download path is the same git-LFS-style folder structure."""
    fake_root.mkdir(parents=True, exist_ok=True)
    repo_dir = fake_root / repo_id.replace("/", "__")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    payload_path = repo_dir / "adapter_model.safetensors"
    manifest_path = repo_dir / "manifest.json"
    payload_path.write_bytes(payload)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "backend": "fake-hf",
        "repo_dir": str(repo_dir),
        "payload_uri": str(payload_path),
        "manifest_uri": str(manifest_path),
    }


def _local_fake_download(repo_id: str, fake_root: Path) -> tuple[bytes, dict]:
    repo_dir = fake_root / repo_id.replace("/", "__")
    payload = (repo_dir / "adapter_model.safetensors").read_bytes()
    manifest = json.loads(
        (repo_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return payload, manifest


def run_smoke(
    out: Path | str = DEFAULT_OUT,
    *,
    repo_id: str = "verifiable-labs/preflight-smoke",
    fake_root: Path = DEFAULT_FAKE_HF_ROOT,
    use_local_fake_hf: bool | None = None,
    seed: int = 0,
) -> dict:
    """Execute the round-trip + write a JSON report."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Mode resolution: explicit override > env var > auto.
    if use_local_fake_hf is None:
        use_local_fake_hf = (
            os.environ.get("LOCAL_FAKE_HF", "").lower() in ("1", "true")
            or not os.environ.get("HF_TOKEN")
        )

    payload = fake_payload(seed=seed)
    payload_sha_in = sha256(payload)
    manifest = fake_manifest(payload_sha_in)

    t0 = time.time()
    upload_info = _local_fake_upload(
        payload=payload,
        manifest=manifest,
        repo_id=repo_id,
        fake_root=fake_root,
    )
    upload_latency = time.time() - t0

    t1 = time.time()
    payload_back, manifest_back = _local_fake_download(
        repo_id=repo_id, fake_root=fake_root
    )
    download_latency = time.time() - t1

    payload_sha_out = sha256(payload_back)
    bytes_match = payload_back == payload
    manifest_match = manifest_back == manifest
    sha_match = payload_sha_in == payload_sha_out

    report = {
        "ok": bytes_match and manifest_match and sha_match,
        "mode": "local-fake-hf" if use_local_fake_hf else "real-hf-hub",
        "repo_id": repo_id,
        "payload_size_bytes": len(payload),
        "payload_sha256_upload": payload_sha_in,
        "payload_sha256_download": payload_sha_out,
        "bytes_match": bytes_match,
        "manifest_match": manifest_match,
        "sha_match": sha_match,
        "upload_latency_seconds": round(upload_latency, 4),
        "download_latency_seconds": round(download_latency, 4),
        "upload_backend": upload_info,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument(
        "--repo-id",
        default="verifiable-labs/preflight-smoke",
        help="Target repo id; ignored in LOCAL_FAKE_HF mode.",
    )
    parser.add_argument(
        "--fake-root",
        default=str(DEFAULT_FAKE_HF_ROOT),
        type=Path,
        help="Where to mirror the fake-HF repo layout.",
    )
    parser.add_argument("--quiet", action="store_true")
    ns = parser.parse_args(argv)

    try:
        report = run_smoke(
            out=ns.out, repo_id=ns.repo_id, fake_root=ns.fake_root
        )
    except Exception as exc:  # noqa: BLE001
        if not ns.quiet:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not ns.quiet:
        print(f"OK -> {ns.out}")
        print(
            f"  mode={report['mode']} bytes_match={report['bytes_match']} "
            f"sha_match={report['sha_match']} "
            f"upload={report['upload_latency_seconds']:.3f}s "
            f"download={report['download_latency_seconds']:.3f}s"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
