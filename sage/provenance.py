"""Protocol and checkpoint provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "checkpoints.json"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


sha256_file = sha256


def load_checkpoint_registry() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def verify_checkpoint(name: str, path: str | Path) -> None:
    registry = load_checkpoint_registry()
    if name not in registry:
        raise KeyError(f"Unknown checkpoint key: {name}")
    expected = registry[name]["sha256"]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint hash mismatch for {name}:\n"
            f"  expected={expected}\n"
            f"  actual={actual}\n"
            f"  path={Path(path)}"
        )


def semantic_manifest_hash(payload: dict) -> str:
    records = []
    if "records" in payload:
        for row in payload["records"]:
            records.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "start_frame": int(row["start_frame"]),
                    "goal_frame": int(row["goal_frame"]),
                    "goal_offset_steps": int(row["goal_offset_steps"]),
                    "record_id": str(row["record_id"]),
                    "split": str(row.get("split", "test")),
                }
            )
    else:
        records = [
            {
                "episode_id": int(episode),
                "start_frame": int(start),
                "goal_frame": int(start) + int(payload["goal_offset_steps"]),
                "goal_offset_steps": int(payload["goal_offset_steps"]),
                "record_id": (
                    f"test_ep{int(episode)}_s{int(start):04d}_"
                    f"g{int(start) + int(payload['goal_offset_steps']):04d}"
                ),
                "split": str(payload.get("split_name", "test")),
            }
            for episode, start in zip(
                payload["episodes_idx"], payload["start_steps"], strict=True
            )
        ]
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(
    payload: dict,
    *,
    benchmark: str,
    seed: int,
    protocol_id: str,
) -> None:
    """Validate the immutable identity fields used by a paper evaluation."""
    if payload.get("protocol_id") != protocol_id:
        raise RuntimeError(
            f"Manifest protocol mismatch: expected {protocol_id!r}, "
            f"got {payload.get('protocol_id')!r}"
        )
    if payload.get("benchmark") != benchmark:
        raise RuntimeError(
            f"Manifest benchmark mismatch: expected {benchmark!r}, "
            f"got {payload.get('benchmark')!r}"
        )
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"Manifest seed mismatch: expected {seed}, got {payload.get('seed')}"
        )
    expected_hash = payload.get("semantic_manifest_sha256")
    actual_hash = semantic_manifest_hash(payload)
    if expected_hash != actual_hash:
        raise RuntimeError(
            "Manifest semantic hash mismatch:\n"
            f"  expected={expected_hash}\n"
            f"  actual={actual_hash}"
        )
