"""Frame-level frozen LeWM latent cache utilities.

The cache is intentionally keyed by frames, not by training windows.  A single
cache built with ``num_steps=1`` can serve fixed-horizon priors, variable-time
priors, and subgoal generators as long as they use the same dataset, frameskip,
image transform, and frozen LeWM checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def build_episode_base(clip_indices, num_episodes: int) -> np.ndarray:
    """Return base index for mapping ``(episode, start)`` to cache row.

    For datasets loaded with ``num_steps=1``, clip indices are contiguous inside
    each episode.  The resulting row index is therefore
    ``episode_base[episode] + start``.  We validate this instead of assuming it,
    because a wrong frame lookup would silently corrupt every cached experiment.
    """

    episode_base = np.full(int(num_episodes), -1, dtype=np.int64)
    for row, (episode, start) in enumerate(clip_indices):
        episode = int(episode)
        start = int(start)
        base = int(row) - start
        if episode_base[episode] < 0:
            episode_base[episode] = base
        elif int(episode_base[episode]) != base:
            raise ValueError(
                "num_steps=1 clip_indices are not contiguous: "
                f"row={row} episode={episode} start={start} "
                f"base={base} expected={int(episode_base[episode])}"
            )
    if bool((episode_base < 0).any()):
        missing = np.flatnonzero(episode_base < 0)[:10].tolist()
        raise ValueError(f"Missing episodes in frame cache lookup: {missing}")
    return episode_base


class FrameLatentCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.meta_path = self.cache_dir / "meta.json"
        self.latents_path = self.cache_dir / "latents.npy"
        self.episode_base_path = self.cache_dir / "episode_base.npy"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing frame cache metadata: {self.meta_path}")
        if not self.latents_path.exists():
            raise FileNotFoundError(f"Missing frame cache latents: {self.latents_path}")
        if not self.episode_base_path.exists():
            raise FileNotFoundError(f"Missing frame cache episode bases: {self.episode_base_path}")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.latents = np.load(self.latents_path, mmap_mode="r")
        self.episode_base = np.load(self.episode_base_path, mmap_mode="r")
        expected = tuple(int(x) for x in self.meta["latent_array_shape"])
        if tuple(self.latents.shape) != expected:
            raise ValueError(
                f"Latent cache shape mismatch: file={self.latents.shape} meta={expected}"
            )

    def validate(self, *, dataset: str, policy: str, frameskip: int, image_size: int) -> None:
        checks = {
            "dataset": dataset,
            "policy": policy,
            "frameskip": int(frameskip),
            "image_size": int(image_size),
        }
        for key, expected in checks.items():
            actual = self.meta.get(key)
            if actual != expected:
                raise ValueError(
                    f"Frame cache mismatch for {key}: cache={actual!r} requested={expected!r}"
                )

    def frame_index(self, local_episode: int, start: int) -> int:
        local_episode = int(local_episode)
        start = int(start)
        if local_episode < 0 or local_episode >= int(self.episode_base.shape[0]):
            raise IndexError(f"local_episode out of cache range: {local_episode}")
        index = int(self.episode_base[local_episode]) + start
        if index < 0 or index >= int(self.latents.shape[0]):
            raise IndexError(
                f"frame cache index out of range: episode={local_episode} start={start} index={index}"
            )
        return index

    def batch_indices(
        self,
        local_episodes,
        starts,
        offsets,
    ) -> np.ndarray:
        episodes = np.asarray(local_episodes, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        return np.asarray(self.episode_base[episodes], dtype=np.int64) + starts + offsets

    def get(self, indices, *, device: torch.device | str | None = None, dtype=None) -> torch.Tensor:
        if torch.is_tensor(indices):
            index_np = indices.detach().cpu().numpy()
        else:
            index_np = np.asarray(indices)
        values = np.asarray(self.latents[index_np], dtype=np.float32)
        tensor = torch.from_numpy(values)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor
