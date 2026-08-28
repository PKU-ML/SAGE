"""Download released SAGE checkpoints and verify their provenance hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sage.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--registry", default="configs/checkpoints.json")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install the release dependencies first: pip install -e ."
        ) from exc

    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in registry.items():
        path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=entry["filename"],
                revision=args.revision,
                local_dir=out_dir,
            )
        )
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise RuntimeError(
                f"{name}: hash mismatch, expected {entry['sha256']}, got {observed}"
            )
        print(f"verified {name}: {path}")


if __name__ == "__main__":
    main()
