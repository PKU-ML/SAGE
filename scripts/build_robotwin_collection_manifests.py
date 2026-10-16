#!/usr/bin/env python3
"""Build split-preserving manifests for place_a2b_left stride-1 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


VIEWS = (
    "vision/cam_head/colors",
    "vision/cam_left_wrist/colors",
    "vision/cam_right_wrist/colors",
)


def episode_files(root: Path, split: str) -> list[Path]:
    return sorted(root.glob(f"{split}/shard_*/**/data/episode_*.hdf5"))


def inspect_episode(path: Path) -> int:
    with h5py.File(path, "r") as source:
        if int(source.attrs.get("lewm_visual_stride", -1)) != 1:
            raise ValueError(f"{path}: not a stride-1 episode")
        action_blocks = source["lewm/action_blocks"]
        if action_blocks.ndim != 3 or action_blocks.shape[1:] != (1, 14):
            raise ValueError(
                f"{path}: expected action blocks (T, 1, 14), got {action_blocks.shape}"
            )
        frames = int(action_blocks.shape[0])
        for key in VIEWS:
            if len(source[key]) != frames:
                raise ValueError(
                    f"{path}: {key} has {len(source[key])} frames, expected {frames}"
                )
    return frames


def write_manifest(path: Path, root: Path, files: list[Path], split: str) -> None:
    episodes = []
    for index, source in enumerate(files):
        episodes.append(
            {
                "index": index,
                "source": source.relative_to(root).as_posix(),
                "frames": inspect_episode(source),
                "task_id": 0,
                "action_mode": "ordinary",
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == len(files):
            print(f"{split}: verified {index + 1}/{len(files)}", flush=True)
    path.write_text(
        json.dumps(
            {
                "format": "robotwin_compressed_v1",
                "source_root": str(root.resolve()),
                "episodes": episodes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=8000)
    parser.add_argument("--validation-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    args = parser.parse_args()

    splits = {
        "train": episode_files(args.root, "train"),
        "validation": episode_files(args.root, "validation"),
        "test": episode_files(args.root, "test"),
    }
    expected = {"train": args.train_episodes, "validation": args.validation_episodes,
                "test": args.test_episodes}
    if any(n <= 0 for n in expected.values()):
        parser.error("All split counts must be positive")
    actual = {name: len(files) for name, files in splits.items()}
    if actual != expected:
        raise RuntimeError(f"split mismatch: expected {expected}, found {actual}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, files in splits.items():
        write_manifest(args.out_dir / f"{split}.json", args.root, files, split)
    (args.out_dir / "dataset_contract.json").write_text(
        json.dumps(
            {
                "protocol": "robotwin_place_a2b_left_stride1_v1",
                "splits": expected,
                "visual_stride": 1,
                "action_block": [1, 14],
                "views": list(VIEWS),
                "split_policy": "native train/shard_00-07, validation/shard_00, test/shard_00; no reshuffle",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
