#!/usr/bin/env python3
"""Cache one frozen dual-view LeWM embedding for every Scene2 frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from sage.native_models import load_world_model
from stable_pretraining import data as dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lewm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument(
        "--finalize-workers",
        nargs="*",
        help="Verify these worker markers cover every frame, then mark cache complete.",
    )
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-name", default="worker")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trainer_preprocessor(key: str, image_size: int):
    stats = dt.dataset_stats.ImageNet
    return dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source=key, target=key),
        dt.transforms.Resize(image_size, source=key, target=key),
    )


def preprocess_frames(
    value: np.ndarray,
    preprocess,
    key: str,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.uint8)).permute(
        0, 3, 1, 2
    )
    sample = {key: tensor}
    tensor = preprocess(sample)[key].to(device, non_blocking=True)
    return tensor[:, None]


def initialize(args: argparse.Namespace, frame_count: int) -> None:
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(f"Choose a new cache output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.float16,
        shape=(frame_count, args.latent_dim),
    )
    cache.flush()
    metadata = {
        "dataset": str(args.dataset.resolve()),
        "dataset_size": args.dataset.stat().st_size,
        "dataset_mtime_ns": args.dataset.stat().st_mtime_ns,
        "lewm_checkpoint": str(args.lewm_checkpoint.resolve()),
        "lewm_checkpoint_sha256": sha256(args.lewm_checkpoint),
        "frame_count": frame_count,
        "latent_dim": args.latent_dim,
        "dtype": "float16",
        "preprocessing": "stablewm-trainer-native-imagenet-resize",
        "image_size": int(args.image_size),
        "complete": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


def finalize(args: argparse.Namespace, frame_count: int) -> None:
    if not args.finalize_workers:
        raise ValueError("--finalize-workers requires at least one worker name")
    ranges = []
    records = []
    for worker in args.finalize_workers:
        marker = args.output.parent / f"{args.output.stem}.{worker}.done.json"
        if not marker.exists():
            raise FileNotFoundError(f"Missing cache worker marker: {marker}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        ranges.append((int(record["start"]), int(record["end"])))
        records.append(record)
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError(f"Cache ranges are incomplete or overlapping: {ranges}")
        cursor = end
    if cursor != frame_count:
        raise ValueError(f"Cache ranges end at {cursor}, expected {frame_count}")
    metadata_path = args.output.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["frame_count"]) != frame_count:
        raise ValueError("Cache metadata frame count changed")
    if metadata["lewm_checkpoint_sha256"] != sha256(args.lewm_checkpoint):
        raise ValueError("LeWM checkpoint does not match initialized cache")
    metadata["complete"] = True
    metadata["workers"] = records
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"finalized {args.output} with ranges={ranges}", flush=True)


def main() -> None:
    args = parse_args()
    with h5py.File(args.dataset, "r", swmr=True) as handle:
        frame_count = int(handle["pixels"].shape[0])
        if int(handle["wrist_pixels"].shape[0]) != frame_count:
            raise ValueError("Agent and wrist frame counts differ")
    if args.initialize:
        initialize(args, frame_count)
        return
    if args.finalize_workers is not None:
        finalize(args, frame_count)
        return
    if not args.output.exists():
        raise FileNotFoundError(f"Initialize cache first: {args.output}")
    metadata = json.loads(args.output.with_suffix('.json').read_text())
    if metadata['lewm_checkpoint_sha256'] != sha256(args.lewm_checkpoint):
        raise ValueError('Worker world model differs from the initialized cache')
    cache = np.load(args.output, mmap_mode="r+")
    if cache.shape != (frame_count, args.latent_dim) or cache.dtype != np.float16:
        raise ValueError(f"Unexpected cache layout: {cache.shape}, {cache.dtype}")
    start = max(0, int(args.start))
    end = frame_count if args.end is None else min(frame_count, int(args.end))
    if not 0 <= start < end <= frame_count:
        raise ValueError(f"Invalid frame range [{start}, {end}) for {frame_count}")

    device = torch.device(args.device)
    lewm = load_world_model(args.lewm_checkpoint, device)
    lewm.requires_grad_(False)
    preprocess = {
        key: trainer_preprocessor(key, args.image_size)
        for key in ("pixels", "wrist_pixels")
    }
    started = time.monotonic()
    with (
        h5py.File(args.dataset, "r", swmr=True) as handle,
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'),
    ):
        for batch_start in range(start, end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end)
            info = {
                "pixels": preprocess_frames(
                    handle["pixels"][batch_start:batch_end],
                    preprocess["pixels"],
                    "pixels",
                    device,
                ),
                "wrist_pixels": preprocess_frames(
                    handle["wrist_pixels"][batch_start:batch_end],
                    preprocess["wrist_pixels"],
                    "wrist_pixels",
                    device,
                ),
            }
            embedding = lewm.encode(info)["emb"]
            if embedding.ndim != 3 or embedding.shape[1] != 1:
                raise ValueError(f"Unexpected LeWM embedding shape: {embedding.shape}")
            values = embedding[:, 0].float().cpu().numpy().astype(np.float16)
            if values.shape != (batch_end - batch_start, args.latent_dim):
                raise ValueError(f"Unexpected cached value shape: {values.shape}")
            cache[batch_start:batch_end] = values
            if (batch_start - start) // args.batch_size % 50 == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                done = batch_end - start
                print(
                    f"{args.worker_name} frames={done}/{end-start} "
                    f"fps={done/elapsed:.1f}",
                    flush=True,
                )
    cache.flush()
    marker = args.output.parent / f"{args.output.stem}.{args.worker_name}.done.json"
    marker.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "start": start,
                "end": end,
                "frames": end - start,
                "seconds": time.monotonic() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done {marker}", flush=True)


if __name__ == "__main__":
    main()
