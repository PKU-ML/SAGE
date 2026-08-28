"""Small training helpers shared by the paper models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def subset_specs(specs, max_count: int, seed: int):
    if max_count <= 0 or len(specs) <= max_count:
        return list(specs)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(specs), size=max_count, replace=False)
    return [specs[int(index)] for index in sorted(indices.tolist())]


def update(target: dict, values: dict, count: int) -> None:
    target["_count"] = target.get("_count", 0) + int(count)
    for key, value in values.items():
        target[key] = target.get(key, 0.0) + float(value) * int(count)


def finalize(values: dict) -> dict:
    count = max(int(values.get("_count", 0)), 1)
    return {
        key: float(value) / count
        for key, value in values.items()
        if key != "_count"
    }


def move_stats(stats: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in stats.items()
    }


def finite_training_mask(*tensors: torch.Tensor) -> torch.Tensor:
    mask = None
    for tensor in tensors:
        row_mask = torch.isfinite(tensor.flatten(1)).all(dim=1)
        mask = row_mask if mask is None else mask & row_mask
    if mask is None:
        raise ValueError("finite_training_mask requires at least one tensor")
    return mask


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _split_episodes(split: dict, name: str) -> set[int]:
    values = split.get(name)
    if values is None:
        values = split.get("episodes", {}).get(name)
    if values is None:
        raise KeyError(f"Split has no {name!r} episode list")
    return {int(value) for value in values}


def compute_stats(
    dataset,
    split: dict,
    lowdim_keys,
    train_specs=None,
    args=None,
) -> dict:
    """Compute action and low-dimensional normalization on train episodes only."""

    del train_specs
    columns = set(dataset.column_names)
    episode_key = "episode_idx" if "episode_idx" in columns else "episode"
    episodes = np.asarray(dataset.get_col_data(episode_key), dtype=np.int64)
    mask = np.isin(episodes, np.asarray(sorted(_split_episodes(split, "train"))))

    action = np.asarray(dataset.get_col_data("action")[mask], dtype=np.float32)
    action = action[np.isfinite(action).all(axis=1)]
    if not action.size:
        raise ValueError("The train split contains no finite actions")

    action_mean = torch.from_numpy(action.mean(axis=0).astype(np.float32))
    action_std = torch.from_numpy(
        np.maximum(action.std(axis=0), 1.0e-6).astype(np.float32)
    )

    all_lowdim_keys = list(lowdim_keys)
    if args is not None:
        all_lowdim_keys += list(getattr(args, "goal_lowdim_keys", []) or [])
    if all_lowdim_keys:
        missing = [key for key in all_lowdim_keys if key not in columns]
        if missing:
            raise KeyError(f"Missing low-dimensional columns: {missing}")
        lowdim = np.concatenate(
            [
                np.asarray(dataset.get_col_data(key)[mask], dtype=np.float32)
                for key in all_lowdim_keys
            ],
            axis=-1,
        )
        lowdim = lowdim[np.isfinite(lowdim).all(axis=1)]
        if not lowdim.size:
            raise ValueError("The train split contains no finite state rows")
        lowdim_mean = torch.from_numpy(lowdim.mean(axis=0).astype(np.float32))
        lowdim_std = torch.from_numpy(
            np.maximum(lowdim.std(axis=0), 1.0e-6).astype(np.float32)
        )
    else:
        lowdim_mean = torch.zeros(0, dtype=torch.float32)
        lowdim_std = torch.zeros(0, dtype=torch.float32)

    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "lowdim_mean": lowdim_mean,
        "lowdim_std": lowdim_std,
        "lowdim_keys": list(lowdim_keys),
        "goal_lowdim_keys": (
            list(getattr(args, "goal_lowdim_keys", []) or [])
            if args is not None
            else []
        ),
    }
