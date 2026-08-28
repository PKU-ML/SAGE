#!/usr/bin/env python3
"""Remove machine-local paths while preserving every evaluation record."""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage.provenance import semantic_manifest_hash  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize_pusht(path: Path, payload: dict) -> dict:
    records = []
    for row in payload["records"]:
        records.append(
            {
                "episode_id": int(row["episode_id"]),
                "local_episode": int(row.get("local_episode", row["episode_id"])),
                "start_frame": int(row["start_frame"]),
                "goal_frame": int(row["goal_frame"]),
                "goal_offset_steps": int(row["goal_offset_steps"]),
                "record_id": str(row["record_id"]),
                "split": str(row.get("split", "test")),
            }
        )
    output = {
        "protocol_id": "sage-paper-v1",
        "benchmark": "pusht",
        "seed": int(payload.get("seed", path.parent.name.removeprefix("seed"))),
        "dataset": "pusht_expert_train.lance",
        "eval_split": str(payload.get("eval_split", "test")),
        "goal_offset_steps": int(payload["goal_offset_steps"]),
        "image_size": int(payload.get("image_size", 224)),
        "num_eval": int(payload["num_eval"]),
        "num_valid_starts": int(payload["num_valid_starts"]),
        "records": records,
        "source_file_sha256": file_sha256(path),
        "source_manifest_sha256": payload.get("manifest_sha256"),
    }
    output["semantic_manifest_sha256"] = semantic_manifest_hash(output)
    return output


def sanitize_cube(path: Path, payload: dict) -> dict:
    output = {
        "protocol_id": "sage-paper-v1",
        "benchmark": "cube",
        "dataset": "ogbench/cube_single_expert.h5",
        "split": "data/splits/ogbench_cube_single_split_seed42.json",
        "split_name": str(payload.get("split_name", "test")),
        "goal_offset_steps": int(payload["goal_offset_steps"]),
        "num_eval": int(payload["num_eval"]),
        "num_valid_starts": int(payload["num_valid_starts"]),
        "seed": int(payload["seed"]),
        "episodes_idx": [int(value) for value in payload["episodes_idx"]],
        "start_steps": [int(value) for value in payload["start_steps"]],
        "source_file_sha256": file_sha256(path),
    }
    output["semantic_manifest_sha256"] = semantic_manifest_hash(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    count = 0
    for path in sorted(source.glob("pusht/seed*/h*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(sanitize_pusht(path, payload), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        count += 1
    for path in sorted(source.glob("cube/seed*/h*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(sanitize_cube(path, payload), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        count += 1
    print(f"sanitized {count} manifests under {destination}")


if __name__ == "__main__":
    main()
