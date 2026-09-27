#!/usr/bin/env python3
"""Train a visual-only SAGE action prior on generated Scene2 subgoals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.train.libero_generator import (
    build_balanced_variable_specs,
    build_terminal_staircase_specs,
    load_generator,
    spec_cell_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--val-dataset", type=Path, required=True)
    parser.add_argument("--train-latent-cache", type=Path, required=True)
    parser.add_argument("--val-latent-cache", type=Path, required=True)
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--train-specs", type=Path)
    parser.add_argument("--val-specs", type=Path)
    parser.add_argument("--train-local-goal-cache", type=Path)
    parser.add_argument("--val-local-goal-cache", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--goal-offsets", nargs="+", type=int, required=True)
    parser.add_argument("--train-goal-offset-weights", nargs="+", type=float)
    parser.add_argument("--subgoal-offset", type=int, default=15)
    parser.add_argument(
        "--terminal-staircase",
        action="store_true",
        help="Train on every clean state with the episode terminal frame as far goal.",
    )
    parser.add_argument(
        "--subgoal-offsets",
        nargs="+",
        type=int,
        help="Optional variable option durations; overrides --subgoal-offset.",
    )
    parser.add_argument(
        "--allowed-action-modes",
        nargs="+",
        type=int,
        default=[0, 1],
        help=(
            "Action modes allowed anywhere in a supervised action chunk. "
            "Scene2 uses 0=oracle, 1=noisy oracle, and 2=random burst."
        ),
    )
    parser.add_argument("--max-train-pairs", type=int, default=800_000)
    parser.add_argument("--max-val-pairs", type=int, default=25_500)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--save-epochs", nargs="+", type=int, default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mode-l1-weight", type=float, default=0.05)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_cache(dataset: Path, cache_path: Path) -> tuple[int, int]:
    with h5py.File(dataset, "r", swmr=True) as handle:
        frame_count = int(handle["pixels"].shape[0])
    metadata_path = cache_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete", False):
        raise ValueError(f"Latent cache is not finalized: {cache_path}")
    cache = np.load(cache_path, mmap_mode="r")
    if cache.ndim != 2 or cache.shape[0] != frame_count:
        raise ValueError(f"Cache {cache.shape} does not match {frame_count} frames")
    if int(metadata["latent_dim"]) != int(cache.shape[1]):
        raise ValueError("Cache metadata latent dimension differs")
    return frame_count, int(cache.shape[1])


class VisualActionPairDataset(Dataset):
    def __init__(
        self,
        dataset: Path,
        latent_cache: Path,
        specs: np.ndarray,
        max_subgoal_offset: int,
        action_mean: np.ndarray,
        action_scale: np.ndarray,
        local_goal_cache: Path | None = None,
    ) -> None:
        self.dataset = Path(dataset)
        self.latent_cache = Path(latent_cache)
        self.specs = np.asarray(specs, dtype=np.int64)
        self.max_subgoal_offset = int(max_subgoal_offset)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.local_goal_cache = Path(local_goal_cache) if local_goal_cache else None
        self._latents: np.ndarray | None = None
        self._local_goals: np.ndarray | None = None
        with h5py.File(self.dataset, "r", swmr=True) as handle:
            self.offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
            # Only ~53 MB for the 10k train set; forked workers share these pages.
            self.actions = np.asarray(handle["action"][:], dtype=np.float32)
        if self.local_goal_cache is not None:
            cached_goals = np.load(self.local_goal_cache, mmap_mode="r")
            if cached_goals.ndim != 2 or cached_goals.shape[0] != len(self.specs):
                raise ValueError(
                    f"Local-goal cache {cached_goals.shape} does not match "
                    f"{len(self.specs)} specs"
                )
            frame_latents = np.load(self.latent_cache, mmap_mode="r")
            if cached_goals.shape[1] != frame_latents.shape[1]:
                raise ValueError("Local-goal cache and frame cache latent dimensions differ")

    def __len__(self) -> int:
        return len(self.specs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_latents"] = None
        state["_local_goals"] = None
        return state

    @property
    def latents(self) -> np.ndarray:
        if self._latents is None:
            self._latents = np.load(self.latent_cache, mmap_mode="r")
        return self._latents

    @property
    def local_goals(self) -> np.ndarray:
        if self.local_goal_cache is None:
            raise RuntimeError("No local-goal cache was configured")
        if self._local_goals is None:
            self._local_goals = np.load(self.local_goal_cache, mmap_mode="r")
        return self._local_goals

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = [int(value) for value in self.specs[index]]
        episode, start, horizon = row[:3]
        subgoal_offset = row[3] if len(row) >= 4 else self.max_subgoal_offset
        base = int(self.offsets[episode])
        # Match online full-episode rollout: unavailable initial history is
        # padded with the episode's first observation, never the previous episode.
        history_indices = base + np.asarray(
            [max(0, start - 10), max(0, start - 5), start]
        )
        far_index = base + (row[4] if len(row) >= 5 else start + horizon)
        raw = self.actions[base + start : base + start + subgoal_offset]
        normalized = (raw - self.action_mean) / self.action_scale
        if subgoal_offset % 5:
            raise ValueError("subgoal_offset must be divisible by the LeWM frameskip 5")
        action_tokens = subgoal_offset // 5
        padded = np.zeros((self.max_subgoal_offset // 5, 35), dtype=np.float32)
        padded[:action_tokens] = normalized.reshape(action_tokens, 35)
        result = {
            "history_latents": torch.from_numpy(
                np.array(self.latents[history_indices], dtype=np.float16, copy=True)
            ),
            "far_goal_latents": torch.from_numpy(
                np.array(self.latents[far_index], dtype=np.float16, copy=True)
            ),
            "actions": torch.from_numpy(
                padded
            ),
            "action_tokens": torch.tensor(action_tokens, dtype=torch.long),
            "goal_offset": torch.tensor(horizon, dtype=torch.float32),
            "subgoal_offset": torch.tensor(subgoal_offset, dtype=torch.float32),
        }
        if self.local_goal_cache is not None:
            result["local_goal_latents"] = torch.from_numpy(
                np.array(self.local_goals[index], dtype=np.float16, copy=True)
            )
        return result


def filter_specs_by_action_mode(
    dataset: Path,
    specs: np.ndarray,
    allowed_modes: list[int],
) -> np.ndarray:
    """Keep only dynamically coherent chunks without random interventions."""
    with h5py.File(dataset, "r", swmr=True) as handle:
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        modes = np.asarray(handle["action_mode"], dtype=np.int16)
    disallowed = ~np.isin(modes, np.asarray(allowed_modes, dtype=np.int16))
    prefix = np.concatenate(([0], np.cumsum(disallowed, dtype=np.int64)))
    starts = offsets[specs[:, 0]] + specs[:, 1]
    action_horizons = specs[:, 3]
    clean = prefix[starts + action_horizons] == prefix[starts]
    return np.asarray(specs[clean], dtype=np.int64)


def resample_clean_specs_by_cell(
    dataset: Path,
    target_specs: np.ndarray,
    clean_specs: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Restore task/horizon cell counts after filtering, using clean rows only."""
    with h5py.File(dataset, "r", swmr=True) as handle:
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        episode_tasks = np.asarray(handle["task_id"][offsets], dtype=np.int64)
    rng = np.random.default_rng(seed)
    target_cells = np.stack(
        [episode_tasks[target_specs[:, 0]], target_specs[:, 2], target_specs[:, 3]], axis=1
    )
    clean_cells = np.stack(
        [episode_tasks[clean_specs[:, 0]], clean_specs[:, 2], clean_specs[:, 3]], axis=1
    )
    rows: list[np.ndarray] = []
    for task, horizon, duration in np.unique(target_cells, axis=0):
        target_count = int(
            np.all(target_cells == np.asarray([task, horizon, duration]), axis=1).sum()
        )
        candidates = clean_specs[
            np.all(clean_cells == np.asarray([task, horizon, duration]), axis=1)
        ]
        if not len(candidates):
            raise ValueError(f"No clean windows for task={task}, H={horizon}, tau={duration}")
        choice = rng.choice(
            len(candidates), size=target_count, replace=len(candidates) < target_count
        )
        rows.append(candidates[choice])
    result = np.concatenate(rows, axis=0).astype(np.int64)
    rng.shuffle(result)
    return result


def aggregate(total: dict, values: dict[str, torch.Tensor], count: int) -> None:
    total["count"] = total.get("count", 0) + int(count)
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value.item()) * int(count)


def finish(total: dict) -> dict[str, float]:
    count = max(int(total.get("count", 0)), 1)
    return {key: value / count for key, value in total.items() if key != "count"}


def prepare(batch, generator, device, *, use_bf16: bool):
    history = batch["history_latents"].to(device, non_blocking=True).float()
    far_goal = batch["far_goal_latents"].to(device, non_blocking=True).float()[:, None]
    goal_offset = batch["goal_offset"].to(device, non_blocking=True)
    subgoal_offset = batch["subgoal_offset"].to(device, non_blocking=True)
    if "local_goal_latents" in batch:
        inference_goal = batch["local_goal_latents"].to(
            device, non_blocking=True
        ).float()[:, None]
    else:
        if generator is None:
            raise RuntimeError("A batch without cached local goals requires the generator")
        # Match online routing: when the final goal is already within one option,
        # use it directly and never query the generator out of distribution.
        inference_goal = far_goal.clone()
        needs_generation = goal_offset > subgoal_offset
        if needs_generation.any():
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                inference_goal[needs_generation] = generator(
                    history[needs_generation],
                    far_goal[needs_generation],
                    torch.empty(int(needs_generation.sum().item()), 0, device=device),
                    goal_offset[needs_generation],
                    subgoal_offset[needs_generation],
                )["prediction"]
    # Inference tensors cannot be saved by the prior's LayerNorm backward.
    local_goal = inference_goal.detach().clone().float()
    return (
        history,
        local_goal.float(),
        far_goal,
        batch["actions"].to(device, non_blocking=True),
        goal_offset,
        subgoal_offset,
        batch["action_tokens"].to(device, non_blocking=True),
    )


def select_prepared(prepared, mask, action_tokens: int):
    history, local_goal, far_goal, actions, goal_offset, subgoal_offset, _ = prepared
    return (
        history[mask],
        local_goal[mask],
        far_goal[mask],
        actions[mask, :action_tokens],
        goal_offset[mask],
        subgoal_offset[mask],
    )


def forward_prior(model, prepared, *, use_bf16: bool):
    history, local_goal, far_goal, actions, goal_offset, subgoal_offset = prepared
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        outputs = model(
            history,
            local_goal,
            torch.empty(history.size(0), 0, device=history.device),
            action_horizon=actions.size(1),
            far_goal_latents=far_goal,
            goal_offset_steps=goal_offset,
            subgoal_offset_steps=subgoal_offset,
        )
    # Keep the likelihood reduction in FP32 even when the Transformer uses BF16.
    outputs = {key: value.float() for key, value in outputs.items()}
    return outputs


@torch.inference_mode()
def evaluate(model, generator, loader, args, device):
    model.eval()
    total: dict = {}
    rng = torch.Generator(device=device).manual_seed(args.seed + 991)
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        prepared = prepare(batch, generator, device, use_bf16=args.bf16)
        action_token_values = prepared[-1]
        for action_tokens in torch.unique(action_token_values).tolist():
            mask = action_token_values == int(action_tokens)
            group = select_prepared(prepared, mask, int(action_tokens))
            history, local_goal, far_goal, actions, goal_offset, subgoal_offset = group
            outputs = forward_prior(model, group, use_bf16=args.bf16)
            top = model.top_mode(
                history, local_goal, torch.empty(history.size(0), 0, device=device),
                action_horizon=int(action_tokens), far_goal_latents=far_goal,
                goal_offset_steps=goal_offset, subgoal_offset_steps=subgoal_offset,
            )
            samples = model.sample(
                history, local_goal, torch.empty(history.size(0), 0, device=device),
                args.eval_samples, generator=rng, action_horizon=int(action_tokens),
                far_goal_latents=far_goal, goal_offset_steps=goal_offset,
                subgoal_offset_steps=subgoal_offset,
            )
            sample_errors = (samples - actions[:, None]).abs().mean(dim=(-1, -2))
            best_sample = sample_errors.min(1).values
            metrics = {
                "nll": model.nll(outputs, actions),
                "top_l1_normalized": (top - actions).abs().mean(),
                "best_mode_l1_normalized": model.best_mode_l1(outputs, actions),
                f"best{args.eval_samples}_l1_normalized": best_sample.mean(),
            }
            aggregate(total, metrics, history.size(0))
    return finish(total)


def main() -> None:
    args = parse_args()
    spec_paths = (args.train_specs, args.val_specs)
    local_goal_caches = (args.train_local_goal_cache, args.val_local_goal_cache)
    if any(path is not None for path in spec_paths) != all(
        path is not None for path in spec_paths
    ):
        raise ValueError("Provide both --train-specs and --val-specs")
    if any(path is not None for path in local_goal_caches) != all(
        path is not None for path in local_goal_caches
    ):
        raise ValueError(
            "Provide both --train-local-goal-cache and --val-local-goal-cache"
        )
    if all(path is not None for path in local_goal_caches) and not all(
        path is not None for path in spec_paths
    ):
        raise ValueError("Cached local goals require explicit aligned specs")
    order = np.argsort(np.asarray(args.goal_offsets))
    args.goal_offsets = [int(args.goal_offsets[i]) for i in order]
    if args.train_goal_offset_weights is not None:
        if len(args.train_goal_offset_weights) != len(order):
            raise ValueError("One weight is required for every goal offset")
        args.train_goal_offset_weights = [float(args.train_goal_offset_weights[i]) for i in order]
    args.subgoal_offsets = sorted(
        set(int(value) for value in (args.subgoal_offsets or [args.subgoal_offset]))
    )
    if any(value <= 0 or value % 5 for value in args.subgoal_offsets):
        raise ValueError("subgoal offsets must be positive and divisible by 5")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    _, train_latent_dim = validate_cache(args.train_dataset, args.train_latent_cache)
    _, val_latent_dim = validate_cache(args.val_dataset, args.val_latent_cache)
    if train_latent_dim != val_latent_dim:
        raise ValueError("Train and validation latent dimensions differ")
    if args.terminal_staircase and args.train_specs is None:
        if len(args.subgoal_offsets) != 1:
            raise ValueError("Terminal staircase requires exactly one subgoal offset")
        train_specs = build_terminal_staircase_specs(
            args.train_dataset,
            args.subgoal_offsets[0],
            None,
            include_direct_final=True,
            allowed_local_action_modes=args.allowed_action_modes,
        )
        val_specs = build_terminal_staircase_specs(
            args.val_dataset,
            args.subgoal_offsets[0],
            None,
            include_direct_final=True,
            allowed_local_action_modes=args.allowed_action_modes,
        )
        if args.max_train_pairs and len(train_specs) > args.max_train_pairs:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(
                len(train_specs), size=args.max_train_pairs, replace=False
            )
            train_specs = train_specs[indices]
        if args.max_val_pairs and len(val_specs) > args.max_val_pairs:
            rng = np.random.default_rng(args.seed + 1)
            indices = rng.choice(
                len(val_specs), size=args.max_val_pairs, replace=False
            )
            val_specs = val_specs[indices]
        train_specs_before_filter = len(train_specs)
        val_specs_before_filter = len(val_specs)
        unique_clean_train_pairs = len(train_specs)
    elif args.train_specs is not None:
        train_specs = np.asarray(np.load(args.train_specs), dtype=np.int64)
        val_specs = np.asarray(np.load(args.val_specs), dtype=np.int64)
        if train_specs.ndim != 2 or train_specs.shape[1] not in (4, 5):
            raise ValueError("Training specs must have shape [N, 4] or [N, 5]")
        if val_specs.ndim != 2 or val_specs.shape[1] not in (4, 5):
            raise ValueError("Validation specs must have shape [N, 4] or [N, 5]")
        if args.terminal_staircase and (
            train_specs.shape[1] != 5 or val_specs.shape[1] != 5
        ):
            raise ValueError(
                "Explicit terminal-staircase specs must include the terminal-frame column"
            )
        if len(filter_specs_by_action_mode(
            args.train_dataset, train_specs, args.allowed_action_modes
        )) != len(train_specs):
            raise ValueError("Explicit training specs contain a disallowed action mode")
        if len(filter_specs_by_action_mode(
            args.val_dataset, val_specs, args.allowed_action_modes
        )) != len(val_specs):
            raise ValueError("Explicit validation specs contain a disallowed action mode")
        train_specs_before_filter = len(train_specs)
        val_specs_before_filter = len(val_specs)
        unique_clean_train_pairs = len(np.unique(train_specs, axis=0))
    else:
        train_specs = build_balanced_variable_specs(
            args.train_dataset, args.goal_offsets, args.subgoal_offsets,
            args.max_train_pairs, args.seed,
            goal_offset_weights=args.train_goal_offset_weights,
        )
        val_specs = build_balanced_variable_specs(
            args.val_dataset, args.goal_offsets, args.subgoal_offsets,
            args.max_val_pairs, args.seed + 1,
        )
        train_specs_before_filter = len(train_specs)
        val_specs_before_filter = len(val_specs)
        clean_train_specs = filter_specs_by_action_mode(
            args.train_dataset, train_specs, args.allowed_action_modes
        )
        clean_val_specs = filter_specs_by_action_mode(
            args.val_dataset, val_specs, args.allowed_action_modes
        )
        if not len(clean_train_specs) or not len(clean_val_specs):
            raise ValueError("Action-mode filtering removed every training or validation pair")
        unique_clean_train_pairs = len(clean_train_specs)
        train_specs = resample_clean_specs_by_cell(
            args.train_dataset, train_specs, clean_train_specs, args.seed + 2
        )
        val_specs = resample_clean_specs_by_cell(
            args.val_dataset, val_specs, clean_val_specs, args.seed + 3
        )
    with h5py.File(args.train_dataset, "r", swmr=True) as handle:
        train_modes = np.asarray(handle["action_mode"][:], dtype=np.int16)
        allowed = np.isin(train_modes, np.asarray(args.allowed_action_modes, dtype=np.int16))
        train_actions = np.asarray(handle["action"][:], dtype=np.float64)[allowed]
    action_mean = train_actions.mean(axis=0).astype(np.float32)
    action_scale = train_actions.std(axis=0).clip(1e-6).astype(np.float32)
    del train_actions
    train_data = VisualActionPairDataset(
        args.train_dataset, args.train_latent_cache, train_specs, max(args.subgoal_offsets),
        action_mean, action_scale, args.train_local_goal_cache,
    )
    val_data = VisualActionPairDataset(
        args.val_dataset, args.val_latent_cache, val_specs, max(args.subgoal_offsets),
        action_mean, action_scale, args.val_local_goal_cache,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    if args.train_local_goal_cache is not None:
        generator = None
        generator_payload = torch.load(
            args.generator_checkpoint, map_location="cpu", weights_only=False
        )
    else:
        generator, generator_payload = load_generator(args.generator_checkpoint, device)
        generator.requires_grad_(False)
    if int(generator_payload["model_config"]["latent_dim"]) != train_latent_dim:
        raise ValueError("Generator and frame-cache latent dimensions differ")
    plan_tokens = max(args.subgoal_offsets) // 5
    model_config = {
        "latent_dim": train_latent_dim,
        "lowdim_dim": 0,
        "action_dim": 35,
        "max_plan_horizon": plan_tokens,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "num_modes": args.num_modes,
        "dropout": 0.0,
        "max_goal_offset": int(max(train_specs[:, 2].max(), val_specs[:, 2].max())),
    }
    model = PushtVariableTransformerGoalPrior(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_payload = None
    start_epoch = 1
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_epoch = int(resume_payload["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but --epochs is {args.epochs}"
            )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "train_specs.npy", train_specs)
    np.save(args.out_dir / "validation_specs.npy", val_specs)
    manifest = {
        "script": "sage.train.libero_prior",
        "scientific_role": (
            "terminal-conditioned visual trajectory-GMM prior"
            if args.terminal_staircase
            else f"visual-only variable-duration SAGE trajectory-GMM prior for tau={args.subgoal_offsets}"
        ),
        "train_dataset": str(args.train_dataset), "validation_dataset": str(args.val_dataset),
        "train_latent_cache": str(args.train_latent_cache), "val_latent_cache": str(args.val_latent_cache),
        "train_specs": str(args.train_specs) if args.train_specs else None,
        "validation_specs": str(args.val_specs) if args.val_specs else None,
        "train_local_goal_cache": (
            str(args.train_local_goal_cache) if args.train_local_goal_cache else None
        ),
        "validation_local_goal_cache": (
            str(args.val_local_goal_cache) if args.val_local_goal_cache else None
        ),
        "generator_checkpoint": str(args.generator_checkpoint),
        "resume_checkpoint": (
            str(args.resume_checkpoint) if args.resume_checkpoint else None
        ),
        "generator_runtime_loaded": generator is not None,
        "generated_subgoal_ratio": float(
            np.mean(train_specs[:, 2] > train_specs[:, 3])
        ),
        "direct_final_goal_when_within_option": True,
        "visual_only": True,
        "history_padding": "repeat_episode_initial_observation",
        "action_mean": action_mean.tolist(), "action_scale": action_scale.tolist(),
        "model_config": model_config, "parameter_count": parameter_count,
        "checkpoint_selection": f"best{args.eval_samples}_l1_normalized",
        "sampling": {"goal_offsets": args.goal_offsets, "weights": args.train_goal_offset_weights,
                     "terminal_staircase": args.terminal_staircase,
                     "terminal_goal_column": 4 if args.terminal_staircase else None,
                     "subgoal_offsets": args.subgoal_offsets, "train_pairs": len(train_specs),
                     "validation_pairs": len(val_specs),
                     "train_cell_counts": spec_cell_counts(train_specs),
                     "validation_cell_counts": spec_cell_counts(val_specs),
                     "allowed_action_modes": args.allowed_action_modes,
                     "train_pairs_before_mode_filter": train_specs_before_filter,
                     "unique_clean_train_pairs": unique_clean_train_pairs,
                     "validation_pairs_before_mode_filter": val_specs_before_filter},
        "args": vars(args),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    metrics_path = args.out_dir / "metrics.jsonl"
    coverage_key = f"best{args.eval_samples}_l1_normalized"
    best_coverage = math.inf
    best_nll = math.inf
    if resume_payload is not None:
        previous_validation = resume_payload.get("validation_metrics", {})
        best_coverage = float(previous_validation.get(coverage_key, math.inf))
        best_nll = float(previous_validation.get("nll", math.inf))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total: dict = {}
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            prepared = prepare(batch, generator, device, use_bf16=args.bf16)
            optimizer.zero_grad(set_to_none=True)
            batch_size = prepared[0].size(0)
            loss = torch.zeros((), device=device)
            batch_nll = torch.zeros((), device=device)
            action_token_values = prepared[-1]
            for action_tokens in torch.unique(action_token_values).tolist():
                mask = action_token_values == int(action_tokens)
                group = select_prepared(prepared, mask, int(action_tokens))
                actions = group[3]
                outputs = forward_prior(model, group, use_bf16=args.bf16)
                weight = float(mask.sum()) / float(batch_size)
                nll = model.nll(outputs, actions)
                best_mode = model.best_mode_l1(outputs, actions)
                loss = loss + weight * (nll + args.mode_l1_weight * best_mode)
                batch_nll = batch_nll + weight * nll.detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            aggregate(total, {"loss": loss.detach(), "nll": batch_nll}, batch_size)
            if batch_index % 50 == 0:
                print(f"epoch={epoch} batch={batch_index} {finish(total)}", flush=True)
        train_metrics = finish(total)
        validation = evaluate(model, generator, val_loader, args, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "model_config": model_config, "manifest": manifest, "train_metrics": train_metrics,
            "validation_metrics": validation,
        }
        torch.save(payload, args.out_dir / "latest.pt")
        if epoch in set(args.save_epochs):
            torch.save(payload, args.out_dir / f"epoch_{epoch:03d}.pt")
        if validation[coverage_key] < best_coverage:
            best_coverage = validation[coverage_key]
            torch.save(payload, args.out_dir / "best.pt")
        selection_loss = validation["nll"]
        if selection_loss < best_nll:
            best_nll = selection_loss
            torch.save(payload, args.out_dir / "best_nll.pt")
        print(f"epoch={epoch} train={train_metrics} val={validation} best_coverage={best_coverage:.6f}", flush=True)
    payload = torch.load(args.out_dir / "latest.pt", map_location="cpu", weights_only=False)
    reload_config = dict(payload["model_config"])
    reloaded = PushtVariableTransformerGoalPrior(**reload_config)
    reloaded.load_state_dict(payload["model"])
    print(f"done parameters={parameter_count} best_coverage={best_coverage:.6f} best_nll={best_nll:.6f}", flush=True)


if __name__ == "__main__":
    main()
