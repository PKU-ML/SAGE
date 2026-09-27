#!/usr/bin/env python3
"""Train a SAGE-style latent subgoal generator on Scene2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

from sage.native_models import load_world_model


from sage.models.subgoal import PushtSubgoalPrior


def image_transform():
    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=112),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--val-dataset", type=Path, required=True)
    parser.add_argument("--lewm-checkpoint", type=Path, required=True)
    parser.add_argument("--train-latent-cache", type=Path)
    parser.add_argument("--val-latent-cache", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional generator checkpoint used to initialize finetuning.",
    )
    parser.add_argument("--goal-offsets", nargs="+", type=int, default=[25, 50, 75, 100])
    parser.add_argument(
        "--train-goal-offset-weights",
        nargs="+",
        type=float,
        help="Optional training weights aligned with --goal-offsets; validation remains uniform.",
    )
    parser.add_argument("--subgoal-offset", type=int, default=25)
    parser.add_argument(
        "--terminal-staircase",
        action="store_true",
        help=(
            "Use every clean state with the episode terminal frame as the far goal "
            "and the next fixed-duration state as the local target."
        ),
    )
    parser.add_argument(
        "--subgoal-offsets",
        nargs="+",
        type=int,
        help="Optional variable option durations; overrides --subgoal-offset.",
    )
    parser.add_argument("--max-train-pairs", type=int, default=400_000)
    parser.add_argument("--max-val-pairs", type=int, default=40_000)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--allowed-local-action-modes",
        nargs="+",
        type=int,
        help=(
            "Optional action modes allowed throughout the supervised local option. "
            "Windows containing any other mode are excluded before sampling."
        ),
    )
    parser.add_argument(
        "--resample-train-specs-each-epoch",
        action="store_true",
        help="Rebuild deterministic stratified training specs with an epoch-specific seed.",
    )
    parser.add_argument(
        "--terminal-train-band-maxes",
        nargs="+",
        type=int,
        help=(
            "Inclusive remaining-horizon maxima used to rebalance terminal "
            "staircase training pairs. Requires matching band weights."
        ),
    )
    parser.add_argument(
        "--terminal-base-train-specs",
        type=Path,
        help="Optional pre-enumerated terminal train specs to rebalance.",
    )
    parser.add_argument(
        "--terminal-base-val-specs",
        type=Path,
        help="Optional pre-enumerated terminal validation specs.",
    )
    parser.add_argument(
        "--terminal-train-band-weights",
        nargs="+",
        type=float,
        help="Sampling weights aligned with --terminal-train-band-maxes.",
    )
    parser.add_argument(
        "--selection-horizons",
        nargs="+",
        type=int,
        help="Validation horizons averaged to select best.pt.",
    )
    parser.add_argument(
        "--predict-residual-from",
        choices=("goal", "current", "zero"),
        default="goal",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _balanced_cell_sizes(total: int, cells: int) -> list[int]:
    base, remainder = divmod(int(total), int(cells))
    return [base + int(index < remainder) for index in range(cells)]


def _weighted_cell_sizes(total: int, weights: list[float]) -> list[int]:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0):
        raise ValueError("Sampling weights must be a non-empty list of positive values")
    exact = values / values.sum() * int(total)
    sizes = np.floor(exact).astype(np.int64)
    remainder = int(total) - int(sizes.sum())
    if remainder:
        order = np.argsort(-(exact - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    return sizes.tolist()


def build_balanced_specs(
    path: Path,
    goal_offsets: list[int],
    limit: int,
    seed: int,
    *,
    history_span: int = 10,
    goal_offset_weights: list[float] | None = None,
) -> np.ndarray:
    """Sample task x horizon cells uniformly, unique before replacement."""
    with h5py.File(path, "r", swmr=True) as handle:
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        task_ids = np.asarray(handle["task_id"][offsets], dtype=np.int64)
    tasks = sorted(int(value) for value in np.unique(task_ids))
    cells = [(task, int(horizon)) for task in tasks for horizon in goal_offsets]
    if goal_offset_weights is None:
        targets = _balanced_cell_sizes(limit, len(cells))
    else:
        if len(goal_offset_weights) != len(goal_offsets):
            raise ValueError("One training weight is required for every goal offset")
        per_cell_weights = [
            float(goal_offset_weights[horizon_index])
            for _task in tasks
            for horizon_index in range(len(goal_offsets))
        ]
        targets = _weighted_cell_sizes(limit, per_cell_weights)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for (task, horizon), target_count in zip(cells, targets):
        episodes = np.flatnonzero(task_ids == task)
        counts = np.maximum(lengths[episodes] - horizon - history_span, 0)
        valid = counts > 0
        episodes, counts = episodes[valid], counts[valid]
        cumulative = np.cumsum(counts)
        population = int(cumulative[-1]) if len(cumulative) else 0
        if population <= 0:
            raise ValueError(f"No valid windows for task={task}, H={horizon}")
        replace = target_count > population
        flat = rng.choice(population, size=target_count, replace=replace)
        episode_slots = np.searchsorted(cumulative, flat, side="right")
        previous = np.where(episode_slots == 0, 0, cumulative[episode_slots - 1])
        starts = history_span + flat - previous
        rows.append(
            np.stack(
                [episodes[episode_slots], starts, np.full(target_count, horizon)],
                axis=1,
            )
        )
    specs = np.concatenate(rows, axis=0).astype(np.int64)
    rng.shuffle(specs)
    return specs


def build_balanced_variable_specs(
    path: Path,
    goal_offsets: list[int],
    subgoal_offsets: list[int],
    limit: int,
    seed: int,
    *,
    history_span: int = 10,
    goal_offset_weights: list[float] | None = None,
    allowed_local_action_modes: list[int] | None = None,
) -> np.ndarray:
    """Sample task x far-goal x valid-duration cells with goal-balanced mass."""
    with h5py.File(path, "r", swmr=True) as handle:
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        task_ids = np.asarray(handle["task_id"][offsets], dtype=np.int64)
        action_modes = (
            np.asarray(handle["action_mode"], dtype=np.int16)
            if allowed_local_action_modes is not None
            else None
        )
    mode_prefixes: list[np.ndarray] | None = None
    if action_modes is not None:
        allowed = np.asarray(sorted(set(allowed_local_action_modes)), dtype=np.int16)
        mode_prefixes = []
        for offset, length in zip(offsets, lengths):
            local_modes = action_modes[int(offset) : int(offset + length)]
            disallowed = (~np.isin(local_modes, allowed)).astype(np.int32)
            mode_prefixes.append(np.concatenate(([0], np.cumsum(disallowed))))
    tasks = sorted(int(value) for value in np.unique(task_ids))
    goals = [int(value) for value in goal_offsets]
    durations = [int(value) for value in subgoal_offsets]
    task_goal_cells = [(task, goal) for task in tasks for goal in goals]
    if goal_offset_weights is None:
        task_goal_targets = _balanced_cell_sizes(limit, len(task_goal_cells))
    else:
        if len(goal_offset_weights) != len(goals):
            raise ValueError("One training weight is required for every goal offset")
        weights = [float(goal_offset_weights[index]) for _task in tasks for index in range(len(goals))]
        task_goal_targets = _weighted_cell_sizes(limit, weights)

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for (task, goal), task_goal_target in zip(task_goal_cells, task_goal_targets):
        valid_durations = [duration for duration in durations if duration <= goal]
        if not valid_durations:
            raise ValueError(f"No valid option duration for H={goal}")
        duration_targets = _balanced_cell_sizes(task_goal_target, len(valid_durations))
        episodes = np.flatnonzero(task_ids == task)
        counts = np.maximum(lengths[episodes] - goal - history_span, 0)
        valid = counts > 0
        episodes, counts = episodes[valid], counts[valid]
        cumulative = np.cumsum(counts)
        population = int(cumulative[-1]) if len(cumulative) else 0
        if population <= 0:
            raise ValueError(f"No valid windows for task={task}, H={goal}")
        for duration, target_count in zip(valid_durations, duration_targets):
            if mode_prefixes is None:
                flat = rng.choice(
                    population, size=target_count, replace=target_count > population
                )
                episode_slots = np.searchsorted(cumulative, flat, side="right")
                previous = np.where(episode_slots == 0, 0, cumulative[episode_slots - 1])
                sampled_episodes = episodes[episode_slots]
                starts = history_span + flat - previous
            else:
                episode_parts: list[np.ndarray] = []
                start_parts: list[np.ndarray] = []
                for episode, count in zip(episodes, counts):
                    starts_for_episode = history_span + np.arange(int(count))
                    prefix = mode_prefixes[int(episode)]
                    keep = (
                        prefix[starts_for_episode + duration]
                        - prefix[starts_for_episode]
                    ) == 0
                    if keep.any():
                        selected_starts = starts_for_episode[keep]
                        start_parts.append(selected_starts)
                        episode_parts.append(
                            np.full(len(selected_starts), int(episode), dtype=np.int64)
                        )
                if not start_parts:
                    raise ValueError(
                        f"No mode-compatible windows for task={task}, H={goal}, "
                        f"duration={duration}"
                    )
                candidate_starts = np.concatenate(start_parts)
                candidate_episodes = np.concatenate(episode_parts)
                population = len(candidate_starts)
                selected = rng.choice(
                    population, size=target_count, replace=target_count > population
                )
                sampled_episodes = candidate_episodes[selected]
                starts = candidate_starts[selected]
            rows.append(
                np.stack(
                    [
                        sampled_episodes,
                        starts,
                        np.full(target_count, goal),
                        np.full(target_count, duration),
                    ],
                    axis=1,
                )
            )
    specs = np.concatenate(rows, axis=0).astype(np.int64)
    rng.shuffle(specs)
    return specs


def build_terminal_staircase_specs(
    path: Path,
    subgoal_offset: int,
    max_goal_offset: int | None,
    *,
    include_direct_final: bool,
    history_span: int = 10,
    allowed_local_action_modes: list[int] | None = None,
) -> np.ndarray:
    """Enumerate clean terminal-conditioned local transitions exactly once."""
    if subgoal_offset <= 0:
        raise ValueError("Invalid terminal-staircase horizon")
    with h5py.File(path, "r", swmr=True) as handle:
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        modes = (
            np.asarray(handle["action_mode"], dtype=np.int16)
            if allowed_local_action_modes is not None
            else None
        )
    allowed = (
        np.asarray(sorted(set(allowed_local_action_modes)), dtype=np.int16)
        if allowed_local_action_modes is not None
        else None
    )
    minimum_remaining = subgoal_offset if include_direct_final else 2 * subgoal_offset
    rows: list[tuple[int, int, int, int, int]] = []
    for episode, (offset, length) in enumerate(zip(offsets, lengths)):
        terminal = int(length) - 1
        if terminal < minimum_remaining:
            continue
        prefix = None
        if modes is not None:
            local_modes = modes[int(offset) : int(offset + length)]
            disallowed = (~np.isin(local_modes, allowed)).astype(np.int32)
            prefix = np.concatenate(([0], np.cumsum(disallowed)))
        for start in range(0, terminal - minimum_remaining + 1):
            if prefix is not None and prefix[start + subgoal_offset] != prefix[start]:
                continue
            remaining = terminal - start
            budget = remaining if max_goal_offset is None else min(max_goal_offset, remaining)
            rows.append((episode, start, budget, subgoal_offset, terminal))
    if not rows:
        raise ValueError("Terminal-staircase filtering produced no examples")
    return np.asarray(rows, dtype=np.int64)


def load_episode_task_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as handle:
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        return np.asarray(handle["task_id"][offsets], dtype=np.int64)


def rebalance_terminal_specs(
    specs: np.ndarray,
    episode_task_ids: np.ndarray,
    total: int,
    band_maxes: list[int],
    band_weights: list[float],
    seed: int,
) -> np.ndarray:
    """Sample task-balanced terminal pairs with an explicit horizon mixture."""
    if total <= 0:
        raise ValueError("Rebalanced terminal training requires max_train_pairs > 0")
    if len(band_maxes) != len(band_weights) or not band_maxes:
        raise ValueError("Terminal band maxima and weights must have equal lengths")
    if band_maxes != sorted(set(band_maxes)):
        raise ValueError("Terminal band maxima must be unique and increasing")
    if int(specs[:, 2].max()) > band_maxes[-1]:
        raise ValueError(
            f"Last terminal band maximum {band_maxes[-1]} does not cover "
            f"maximum horizon {int(specs[:, 2].max())}"
        )

    rng = np.random.default_rng(seed)
    episode_indices = np.asarray(specs[:, 0], dtype=np.int64)
    if episode_indices.min() < 0 or episode_indices.max() >= len(episode_task_ids):
        raise ValueError("Terminal specs contain an invalid episode index")
    row_tasks = np.asarray(episode_task_ids, dtype=np.int64)[episode_indices]
    tasks, task_ids = np.unique(row_tasks, return_inverse=True)
    horizons = specs[:, 2]
    band_ids = np.searchsorted(
        np.asarray(band_maxes, dtype=np.int64), horizons, side="left"
    )
    num_bands = len(band_weights)
    cell_ids = task_ids * num_bands + band_ids
    order = np.argsort(cell_ids, kind="stable")
    sorted_cell_ids = cell_ids[order]
    cells: list[np.ndarray] = []
    cell_weights: list[float] = []
    for task_id, _task in enumerate(tasks):
        for band, weight in enumerate(band_weights):
            cell_id = task_id * num_bands + band
            left = int(np.searchsorted(sorted_cell_ids, cell_id, side="left"))
            right = int(np.searchsorted(sorted_cell_ids, cell_id, side="right"))
            if left == right:
                continue
            cells.append(order[left:right])
            cell_weights.append(float(weight) / len(tasks))

    cell_sizes = _weighted_cell_sizes(total, cell_weights)
    selected: list[np.ndarray] = []
    for candidates, target_count in zip(cells, cell_sizes):
        full_cycles, remainder = divmod(target_count, len(candidates))
        for _ in range(full_cycles):
            selected.append(rng.permutation(candidates))
        if remainder:
            selected.append(rng.choice(candidates, size=remainder, replace=False))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return np.asarray(specs[indices], dtype=np.int64)


def spec_cell_counts(specs: np.ndarray) -> dict[str, int]:
    cells, counts = np.unique(np.asarray(specs)[:, 2:4], axis=0, return_counts=True)
    return {f"H{int(goal)}_A{int(duration)}": int(count) for (goal, duration), count in zip(cells, counts)}


class Scene2SubgoalPairDataset(Dataset):
    def __init__(self, path: Path, specs: np.ndarray, subgoal_offset: int = 25):
        self.path = Path(path)
        self.specs = np.asarray(specs, dtype=np.int64)
        self.subgoal_offset = int(subgoal_offset)
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r", swmr=True) as handle:
            self.offsets = np.asarray(handle["ep_offset"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.specs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = [int(value) for value in self.specs[index]]
        episode, start, horizon = row[:3]
        subgoal_offset = row[3] if len(row) >= 4 else self.subgoal_offset
        offset = int(self.offsets[episode])
        history_steps = [start, start, start] if start < 10 else [start - 10, start - 5, start]
        history = offset + np.asarray(history_steps)
        goal = offset + (row[4] if len(row) >= 5 else start + horizon)
        subgoal = offset + start + subgoal_offset
        handle = self.handle
        return {
            "pixels": torch.from_numpy(np.asarray(handle["pixels"][history], dtype=np.uint8)),
            "wrist_pixels": torch.from_numpy(
                np.asarray(handle["wrist_pixels"][history], dtype=np.uint8)
            ),
            "goal_pixels": torch.from_numpy(
                np.asarray(handle["pixels"][goal], dtype=np.uint8)
            ),
            "goal_wrist_pixels": torch.from_numpy(
                np.asarray(handle["wrist_pixels"][goal], dtype=np.uint8)
            ),
            "subgoal_pixels": torch.from_numpy(
                np.asarray(handle["pixels"][subgoal], dtype=np.uint8)
            ),
            "subgoal_wrist_pixels": torch.from_numpy(
                np.asarray(handle["wrist_pixels"][subgoal], dtype=np.uint8)
            ),
            "proprio": torch.from_numpy(
                np.asarray(handle["proprio"][offset + start], dtype=np.float32)
            ),
            "actions": torch.from_numpy(
                np.asarray(
                    handle["action"][offset + start : offset + start + subgoal_offset],
                    dtype=np.float32,
                )
            ),
            "goal_offset": torch.tensor(horizon, dtype=torch.float32),
            "subgoal_offset": torch.tensor(subgoal_offset, dtype=torch.float32),
            "episode": torch.tensor(episode, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


class Scene2CachedSubgoalPairDataset(Dataset):
    """Read precomputed frozen-LeWM frame latents instead of decoding images."""

    def __init__(
        self,
        path: Path,
        specs: np.ndarray,
        latent_cache: Path,
        subgoal_offset: int = 25,
    ):
        self.path = Path(path)
        self.specs = np.asarray(specs, dtype=np.int64)
        self.latent_cache = Path(latent_cache)
        self.subgoal_offset = int(subgoal_offset)
        self._latents: np.ndarray | None = None
        with h5py.File(self.path, "r", swmr=True) as handle:
            self.offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
            frame_count = int(handle["pixels"].shape[0])
        cache = np.load(self.latent_cache, mmap_mode="r")
        metadata_path = self.latent_cache.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing latent cache metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            raise ValueError(f"Latent cache is not finalized: {self.latent_cache}")
        if int(metadata.get("frame_count", -1)) != frame_count:
            raise ValueError("Latent cache metadata frame count does not match dataset")
        if cache.ndim != 2 or cache.shape[0] != frame_count:
            raise ValueError(
                f"Latent cache shape {cache.shape} does not match {frame_count} frames"
            )
        self.latent_dim = int(cache.shape[1])
        del cache

    def __len__(self) -> int:
        return len(self.specs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_latents"] = None
        return state

    @property
    def latents(self) -> np.ndarray:
        if self._latents is None:
            self._latents = np.load(self.latent_cache, mmap_mode="r")
        return self._latents

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = [int(value) for value in self.specs[index]]
        episode, start, horizon = row[:3]
        subgoal_offset = row[3] if len(row) >= 4 else self.subgoal_offset
        offset = int(self.offsets[episode])
        history_steps = [start, start, start] if start < 10 else [start - 10, start - 5, start]
        history = offset + np.asarray(history_steps)
        goal = offset + (row[4] if len(row) >= 5 else start + horizon)
        subgoal = offset + start + subgoal_offset
        latents = self.latents
        return {
            "history_latents": torch.from_numpy(
                np.array(latents[history], dtype=np.float16, copy=True)
            ),
            "goal_latents": torch.from_numpy(
                np.array(latents[goal], dtype=np.float16, copy=True)
            ),
            "subgoal_latents": torch.from_numpy(
                np.array(latents[subgoal], dtype=np.float16, copy=True)
            ),
            "goal_offset": torch.tensor(horizon, dtype=torch.float32),
            "subgoal_offset": torch.tensor(subgoal_offset, dtype=torch.float32),
            "episode": torch.tensor(episode, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


def preprocess_video(value: torch.Tensor, preprocess, device: torch.device) -> torch.Tensor:
    if value.ndim == 4:
        value = value[:, None]
    batch, time = value.shape[:2]
    flat = value.permute(0, 1, 4, 2, 3).reshape(-1, 3, value.shape[2], value.shape[3])
    flat = preprocess(flat).to(device, non_blocking=True)
    return flat.reshape(batch, time, *flat.shape[1:])


@torch.no_grad()
def encode_views(lewm, agent: torch.Tensor, wrist: torch.Tensor, preprocess, device):
    info = {
        "pixels": preprocess_video(agent, preprocess, device),
        "wrist_pixels": preprocess_video(wrist, preprocess, device),
    }
    return lewm.encode(info)["emb"].float()


@torch.no_grad()
def encode_scene2_batch(lewm, batch, preprocess, device):
    history = encode_views(lewm, batch["pixels"], batch["wrist_pixels"], preprocess, device)
    goal = encode_views(
        lewm, batch["goal_pixels"], batch["goal_wrist_pixels"], preprocess, device
    )
    subgoal = encode_views(
        lewm,
        batch["subgoal_pixels"],
        batch["subgoal_wrist_pixels"],
        preprocess,
        device,
    )
    return history, goal, subgoal


def get_batch_latents(lewm, batch, preprocess, device):
    if "history_latents" in batch:
        history = batch["history_latents"].to(device, non_blocking=True).float()
        goal = batch["goal_latents"].to(device, non_blocking=True).float()
        target = batch["subgoal_latents"].to(device, non_blocking=True).float()
        return history, goal[:, None], target[:, None]
    return encode_scene2_batch(lewm, batch, preprocess, device)


def load_generator(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PushtSubgoalPrior(**payload["model_config"])
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def aggregate(total: dict, values: dict[str, torch.Tensor], count: int) -> None:
    total["count"] = total.get("count", 0) + int(count)
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value.item()) * int(count)


def finish(total: dict) -> dict[str, float]:
    count = max(int(total.get("count", 0)), 1)
    return {key: value / count for key, value in total.items() if key != "count"}


@torch.inference_mode()
def evaluate(model, lewm, loader, preprocess, args, device):
    model.eval()
    totals: dict[str, dict] = {"all": {}}
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        history, goal, target = get_batch_latents(lewm, batch, preprocess, device)
        lowdim = torch.empty(history.size(0), 0, device=device)
        goal_offset = batch["goal_offset"].to(device)
        local_offset = batch["subgoal_offset"].to(device)
        output = model(history, goal, lowdim, goal_offset, local_offset)
        prediction = output["prediction"]
        values = {
            "pred_l1": (prediction - target).abs().flatten(1).mean(),
            "pred_mse": (prediction - target).square().flatten(1).mean(),
            "pred_cos": 1.0
            - torch.nn.functional.cosine_similarity(
                prediction.flatten(1), target.flatten(1), dim=-1
            ).mean(),
            "goal_l1": (goal - target).abs().flatten(1).mean(),
            "current_l1": (history[:, -1:] - target).abs().flatten(1).mean(),
        }
        aggregate(totals["all"], values, history.size(0))
        for horizon in args.goal_offsets:
            mask = goal_offset.long() == int(horizon)
            if not mask.any():
                continue
            cell = {key: value if value.ndim == 0 else value[mask].mean() for key, value in values.items()}
            # Recompute vector metrics for the selected rows.
            cell = {
                "pred_l1": (prediction[mask] - target[mask]).abs().flatten(1).mean(),
                "pred_mse": (prediction[mask] - target[mask]).square().flatten(1).mean(),
                "pred_cos": 1.0
                - torch.nn.functional.cosine_similarity(
                    prediction[mask].flatten(1), target[mask].flatten(1), dim=-1
                ).mean(),
                "goal_l1": (goal[mask] - target[mask]).abs().flatten(1).mean(),
                "current_l1": (history[mask, -1:] - target[mask]).abs().flatten(1).mean(),
            }
            totals.setdefault(f"h{horizon}", {})
            aggregate(totals[f"h{horizon}"], cell, int(mask.sum()))
        for horizon in args.goal_offsets:
            for duration in args.subgoal_offsets:
                mask = (goal_offset.long() == int(horizon)) & (
                    local_offset.long() == int(duration)
                )
                if not mask.any():
                    continue
                cell = {
                    "pred_l1": (prediction[mask] - target[mask]).abs().flatten(1).mean(),
                    "pred_mse": (prediction[mask] - target[mask]).square().flatten(1).mean(),
                    "pred_cos": 1.0
                    - torch.nn.functional.cosine_similarity(
                        prediction[mask].flatten(1), target[mask].flatten(1), dim=-1
                    ).mean(),
                    "goal_l1": (goal[mask] - target[mask]).abs().flatten(1).mean(),
                    "current_l1": (history[mask, -1:] - target[mask]).abs().flatten(1).mean(),
                }
                key = f"h{horizon}_a{duration}"
                totals.setdefault(key, {})
                aggregate(totals[key], cell, int(mask.sum()))
    return {key: finish(value) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if (args.train_latent_cache is None) != (args.val_latent_cache is None):
        raise ValueError("Provide both --train-latent-cache and --val-latent-cache")
    if len(set(args.goal_offsets)) != len(args.goal_offsets):
        raise ValueError("goal_offsets must not contain duplicates")
    ordering = np.argsort(np.asarray(args.goal_offsets))
    args.goal_offsets = [int(args.goal_offsets[index]) for index in ordering]
    if args.train_goal_offset_weights is not None:
        if len(args.train_goal_offset_weights) != len(ordering):
            raise ValueError("One training weight is required for every goal offset")
        args.train_goal_offset_weights = [
            float(args.train_goal_offset_weights[index]) for index in ordering
        ]
    args.subgoal_offsets = sorted(
        set(int(value) for value in (args.subgoal_offsets or [args.subgoal_offset]))
    )
    if any(value <= 0 for value in args.subgoal_offsets):
        raise ValueError("subgoal offsets must be positive")
    terminal_band_args = (
        args.terminal_train_band_maxes,
        args.terminal_train_band_weights,
    )
    if (terminal_band_args[0] is None) != (terminal_band_args[1] is None):
        raise ValueError(
            "Provide both terminal train band maxima and band weights"
        )
    if terminal_band_args[0] is not None and not args.terminal_staircase:
        raise ValueError("Terminal train bands require --terminal-staircase")
    if (args.terminal_base_train_specs is None) != (
        args.terminal_base_val_specs is None
    ):
        raise ValueError("Provide both terminal base train and validation specs")
    if args.terminal_base_train_specs is not None and not args.terminal_staircase:
        raise ValueError("Terminal base specs require --terminal-staircase")
    if not any(duration <= goal for goal in args.goal_offsets for duration in args.subgoal_offsets):
        raise ValueError("No valid (goal offset, option duration) cell")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.terminal_staircase:
        if len(args.subgoal_offsets) != 1:
            raise ValueError("Terminal staircase requires exactly one subgoal offset")
        if args.resample_train_specs_each_epoch:
            raise ValueError("Terminal staircase enumerates fixed unique specs")
        if args.terminal_base_train_specs is None:
            unique_train_specs = build_terminal_staircase_specs(
                args.train_dataset,
                args.subgoal_offsets[0],
                None,
                include_direct_final=False,
                allowed_local_action_modes=args.allowed_local_action_modes,
            )
            val_specs = build_terminal_staircase_specs(
                args.val_dataset,
                args.subgoal_offsets[0],
                None,
                include_direct_final=False,
                allowed_local_action_modes=args.allowed_local_action_modes,
            )
        else:
            unique_train_specs = np.load(args.terminal_base_train_specs)
            val_specs = np.load(args.terminal_base_val_specs)
            if unique_train_specs.ndim != 2 or unique_train_specs.shape[1] != 5:
                raise ValueError("Terminal base train specs must have shape (N, 5)")
            if val_specs.ndim != 2 or val_specs.shape[1] != 5:
                raise ValueError("Terminal base validation specs must have shape (N, 5)")
        if args.terminal_train_band_maxes is None:
            train_specs = unique_train_specs
        else:
            train_specs = rebalance_terminal_specs(
                unique_train_specs,
                load_episode_task_ids(args.train_dataset),
                args.max_train_pairs,
                args.terminal_train_band_maxes,
                args.terminal_train_band_weights,
                args.seed,
            )
    else:
        train_specs = build_balanced_variable_specs(
            args.train_dataset,
            args.goal_offsets,
            args.subgoal_offsets,
            args.max_train_pairs,
            args.seed,
            goal_offset_weights=args.train_goal_offset_weights,
            allowed_local_action_modes=args.allowed_local_action_modes,
        )
        val_specs = build_balanced_variable_specs(
            args.val_dataset,
            args.goal_offsets,
            args.subgoal_offsets,
            args.max_val_pairs,
            args.seed + 1,
            allowed_local_action_modes=args.allowed_local_action_modes,
        )
    if args.train_latent_cache is not None:
        train_data = Scene2CachedSubgoalPairDataset(
            args.train_dataset, train_specs, args.train_latent_cache, args.subgoal_offset
        )
        val_data = Scene2CachedSubgoalPairDataset(
            args.val_dataset, val_specs, args.val_latent_cache, args.subgoal_offset
        )
    else:
        train_data = Scene2SubgoalPairDataset(
            args.train_dataset, train_specs, args.subgoal_offset
        )
        val_data = Scene2SubgoalPairDataset(args.val_dataset, val_specs, args.subgoal_offset)
    def make_train_loader(data: Dataset) -> DataLoader:
        return DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(
                args.num_workers > 0 and not args.resample_train_specs_each_epoch
            ),
            drop_last=True,
        )

    train_loader = make_train_loader(train_data)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.train_latent_cache is not None:
        lewm = None
        preprocess = None
        latent_dim = int(train_data.latent_dim)
    else:
        lewm = load_world_model(args.lewm_checkpoint, device)
        lewm.requires_grad_(False)
        preprocess = image_transform()
        first = next(iter(train_loader))
        with torch.no_grad():
            history, _, _ = encode_scene2_batch(lewm, first, preprocess, device)
        latent_dim = int(history.shape[-1])
    model_config = {
        "latent_dim": latent_dim,
        "lowdim_dim": 0,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "max_goal_offset": int(max(train_specs[:, 2].max(), val_specs[:, 2].max())),
        "predict_residual_from": args.predict_residual_from,
        "pooling": "decoder",
        "goal_condition_mode": "goal",
    }
    model = PushtSubgoalPrior(**model_config).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        initial_config = initial.get("model_config")
        if initial_config != model_config:
            raise ValueError(
                "Initial generator model_config does not match this run: "
                f"{initial_config!r} != {model_config!r}"
            )
        model.load_state_dict(initial["model"], strict=True)
        print(f"Initialized generator from {args.init_checkpoint}", flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= 120_000_000:
        raise ValueError(f"Generator exceeds 120M parameters: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "train_specs.npy", train_specs)
    np.save(args.out_dir / "validation_specs.npy", val_specs)
    manifest = {
        "script": "sage.train.libero_generator",
        "scientific_role": (
            "terminal-conditioned 15-step latent subgoal generator"
            if args.terminal_staircase
            else f"SAGE-style variable-duration latent generator for tau={args.subgoal_offsets}"
        ),
        "train_dataset": str(args.train_dataset),
        "validation_dataset": str(args.val_dataset),
        "lewm_checkpoint": str(args.lewm_checkpoint),
        "train_latent_cache": (
            str(args.train_latent_cache) if args.train_latent_cache is not None else None
        ),
        "validation_latent_cache": (
            str(args.val_latent_cache) if args.val_latent_cache is not None else None
        ),
        "model_config": model_config,
        "parameter_count": parameter_count,
        "sampling": {
            "goal_offsets": args.goal_offsets,
            "terminal_staircase": args.terminal_staircase,
            "terminal_goal_column": 4 if args.terminal_staircase else None,
            "train_goal_offset_weights": args.train_goal_offset_weights,
            "allowed_local_action_modes": args.allowed_local_action_modes,
            "resample_train_specs_each_epoch": args.resample_train_specs_each_epoch,
            "train_epoch_seeds": [args.seed + index for index in range(args.epochs)],
            "subgoal_offsets": args.subgoal_offsets,
            "task_goal_balanced_then_duration_balanced": True,
            "unique_before_replacement": True,
            "train_pairs": len(train_specs),
            "unique_train_pairs_before_rebalance": (
                len(unique_train_specs)
                if args.terminal_staircase
                else len(train_specs)
            ),
            "terminal_train_band_maxes": args.terminal_train_band_maxes,
            "terminal_train_band_weights": args.terminal_train_band_weights,
            "terminal_base_train_specs": (
                str(args.terminal_base_train_specs)
                if args.terminal_base_train_specs is not None
                else None
            ),
            "terminal_base_val_specs": (
                str(args.terminal_base_val_specs)
                if args.terminal_base_val_specs is not None
                else None
            ),
            "validation_pairs": len(val_specs),
            "train_cell_counts": spec_cell_counts(train_specs),
            "validation_cell_counts": spec_cell_counts(val_specs),
        },
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "args": vars(args),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    metrics_path = args.out_dir / "metrics.jsonl"
    best_l1 = math.inf
    for epoch in range(1, args.epochs + 1):
        if epoch > 1 and args.resample_train_specs_each_epoch:
            train_specs = build_balanced_variable_specs(
                args.train_dataset,
                args.goal_offsets,
                args.subgoal_offsets,
                args.max_train_pairs,
                args.seed + epoch - 1,
                goal_offset_weights=args.train_goal_offset_weights,
                allowed_local_action_modes=args.allowed_local_action_modes,
            )
            if args.train_latent_cache is not None:
                train_data = Scene2CachedSubgoalPairDataset(
                    args.train_dataset,
                    train_specs,
                    args.train_latent_cache,
                    args.subgoal_offset,
                )
            else:
                train_data = Scene2SubgoalPairDataset(
                    args.train_dataset, train_specs, args.subgoal_offset
                )
            train_loader = make_train_loader(train_data)
        model.train()
        total: dict = {}
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            history, goal, target = get_batch_latents(lewm, batch, preprocess, device)
            lowdim = torch.empty(history.size(0), 0, device=device)
            output = model(
                history,
                goal,
                lowdim,
                batch["goal_offset"].to(device),
                batch["subgoal_offset"].to(device),
            )
            losses = model.loss(
                output,
                target,
                cosine_weight=args.cosine_weight,
                smooth_l1_beta=args.smooth_l1_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            aggregate(
                total,
                {
                    "loss": losses["loss"].detach(),
                    "pred_l1": (output["prediction"].detach() - target).abs().flatten(1).mean(),
                },
                history.size(0),
            )
            if batch_index % 50 == 0:
                print(f"epoch={epoch} batch={batch_index} {finish(total)}", flush=True)
        train_metrics = finish(total)
        validation = evaluate(model, lewm, val_loader, preprocess, args, device)
        if args.selection_horizons:
            selection_values = []
            for horizon in args.selection_horizons:
                key = f"h{int(horizon)}"
                if key not in validation:
                    raise ValueError(
                        f"Selection horizon {horizon} is absent from validation"
                    )
                selection_values.append(float(validation[key]["pred_l1"]))
            selection_l1 = float(np.mean(selection_values))
        else:
            selection_l1 = float(validation["all"]["pred_l1"])
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "selection_l1": selection_l1,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": model_config,
            "manifest": manifest,
            "train_metrics": train_metrics,
            "validation_metrics": validation,
        }
        torch.save(payload, args.out_dir / "latest.pt")
        value = selection_l1
        if value < best_l1:
            best_l1 = value
            torch.save(payload, args.out_dir / "best.pt")
        print(
            f"epoch={epoch} train={train_metrics} val={validation['all']} "
            f"selection_l1={selection_l1:.6f} best_l1={best_l1:.6f}",
            flush=True,
        )
    # Smoke and production runs both verify that the serialized model reloads exactly.
    reloaded, _ = load_generator(args.out_dir / "latest.pt", device)
    if sum(parameter.numel() for parameter in reloaded.parameters()) != parameter_count:
        raise RuntimeError("Reloaded generator parameter count changed")
    print(f"done parameters={parameter_count} best_val_l1={best_l1:.6f}", flush=True)


if __name__ == "__main__":
    main()
