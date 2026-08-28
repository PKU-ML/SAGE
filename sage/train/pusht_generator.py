"""Train a self-supervised Push-T latent subgoal prior.

The target is provided by the expert trajectory itself: from a planning state t
and a farther goal t+H, predict the next local subgoal latent t+25. This is the
first diagnostic stage for a hierarchical prior: if this predictor is accurate,
we can later ask a local action prior / CEM solver to reach the predicted
subgoal instead of trying to bridge the whole horizon in one action proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sage.runtime.lewm import (
    build_window_specs,
    encode_lewm_context,
    image_batch_to_lewm,
    load_json,
    load_lewm,
    load_swm_dataset,
    lowdim_from_batch,
    normalize_lowdim,
)
from sage.models.subgoal import PushtSubgoalPrior, save_subgoal_prior
from sage.sampling import sample_dense_unique_pairs
from sage.training import (
    compute_stats,
    finalize,
    move_stats,
    subset_specs,
    update,
)


class SubgoalPairDataset(Dataset):
    def __init__(
        self,
        dataset,
        specs,
        pair_spec_indices,
        pair_goal_frames,
        pair_subgoal_frames,
        pair_goal_offsets,
        pair_subgoal_offsets,
    ):
        self.dataset = dataset
        self.specs = list(specs)
        self.pair_spec_indices = np.asarray(pair_spec_indices, dtype=np.int64)
        self.pair_goal_frames = np.asarray(pair_goal_frames, dtype=np.int64)
        self.pair_subgoal_frames = np.asarray(pair_subgoal_frames, dtype=np.int64)
        self.pair_goal_offsets = np.asarray(pair_goal_offsets, dtype=np.int64)
        self.pair_subgoal_offsets = np.asarray(pair_subgoal_offsets, dtype=np.int64)
    def __len__(self):
        return int(len(self.pair_spec_indices))
    def __getitem__(self, index):
        spec = self.specs[int(self.pair_spec_indices[index])]
        item = self.dataset[spec.dataset_index]
        goal_frame = int(self.pair_goal_frames[index])
        subgoal_frame = int(self.pair_subgoal_frames[index])
        goal = self.dataset._load_slice(
            spec.local_episode,
            goal_frame,
            goal_frame + 1,
        )
        subgoal = self.dataset._load_slice(
            spec.local_episode,
            subgoal_frame,
            subgoal_frame + 1,
        )
        item["goal_pixels"] = torch.as_tensor(goal["pixels"])
        item["subgoal_pixels"] = torch.as_tensor(subgoal["pixels"])
        item["episode_id"] = torch.tensor(spec.episode_id, dtype=torch.long)
        item["start"] = torch.tensor(spec.start, dtype=torch.long)
        item["goal_frame"] = torch.tensor(goal_frame, dtype=torch.long)
        item["subgoal_frame"] = torch.tensor(subgoal_frame, dtype=torch.long)
        item["goal_offset"] = torch.tensor(
            int(self.pair_goal_offsets[index]), dtype=torch.long
        )
        item["subgoal_offset"] = torch.tensor(
            int(self.pair_subgoal_offsets[index]), dtype=torch.long
        )
        return item
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pusht_expert_train.lance")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--pooling",
        choices=["attention", "mean", "decoder"],
        default="attention",
    )
    parser.add_argument(
        "--predict-residual-from",
        choices=["goal", "current", "zero"],
        default="goal",
    )
    parser.add_argument(
        "--goal-condition-mode",
        choices=["goal", "zero", "current"],
        default="goal",
        help=(
            "Which latent is exposed as the far-goal condition."
        ),
    )
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--subgoal-offset", type=int, default=25)
    parser.add_argument(
        "--subgoal-offsets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional set of supervised local subgoal offsets in env steps. "
            "For hierarchical generators, use e.g. 5 10 25 or 5 10 15 20 25. "
            "If omitted, --subgoal-offset is used."
        ),
    )
    parser.add_argument("--goal-offsets", nargs="+", type=int, default=[50, 100, 150])
    parser.add_argument(
        "--subgoal-schedule",
        choices=["all", "midpoint", "exponential"],
        default="all",
        help=(
            "How to pair far goals with supervised subgoals. 'all' keeps every "
            "requested --subgoal-offsets <= H. 'midpoint' uses round(H * "
            "--midpoint-fraction). 'exponential' uses the largest "
            "--exponential-base-offset * 2^k strictly below H."
        ),
    )
    parser.add_argument("--midpoint-fraction", type=float, default=0.5)
    parser.add_argument("--exponential-base-offset", type=int, default=25)
    parser.add_argument("--include-h25-anchor", action="store_true")
    parser.add_argument("--include-final-goal", action="store_true")
    parser.add_argument("--lowdim-keys", nargs="+", default=["state", "proprio"])
    parser.add_argument("--no-lowdim", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=200000)
    parser.add_argument("--max-val-windows", type=int, default=20000)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument(
        "--balance-goal-offsets",
        action="store_true",
        help=(
            "When pair limits are active, sample approximately equal numbers "
            "for each requested far-goal offset. This changes only the sampled "
            "training distribution; model architecture and targets stay fixed."
        ),
    )
    parser.add_argument(
        "--dense-joint-sampling",
        action="store_true",
        help=(
            "Sample a joint-balanced (goal_offset, subgoal_offset) dataset "
            "directly from the full spec pool without reusing windows."
        ),
    )
    parser.add_argument(
        "--dense-balance-goals",
        action="store_true",
        help="With --dense-joint-sampling, give each far-goal offset equal total mass, then divide it among valid local subgoal offsets.",
    )
    parser.add_argument(
        "--dense-allow-repeats",
        action="store_true",
        help="After broad unique coverage, fill an oversized dense budget with balanced repeats.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory; useful after CUDA/NVML driver errors.",
    )
    args = parser.parse_args()
    if args.no_lowdim:
        args.lowdim_keys = []
    if args.include_h25_anchor and 25 not in args.goal_offsets:
        args.goal_offsets = [25] + list(args.goal_offsets)
    args.goal_offsets = sorted(set(int(x) for x in args.goal_offsets))
    if args.subgoal_offsets is None:
        args.subgoal_offsets = [int(args.subgoal_offset)]
    args.subgoal_offsets = sorted(set(int(x) for x in args.subgoal_offsets))
    if any(x <= 0 for x in args.subgoal_offsets):
        raise ValueError(f"subgoal offsets must be positive, got {args.subgoal_offsets}")
    if args.exponential_base_offset <= 0:
        raise ValueError("--exponential-base-offset must be positive")
    return args


def scheduled_subgoal_offsets(goal_offset: int, args) -> list[int]:
    goal_offset = int(goal_offset)
    if args.subgoal_schedule == "all":
        return [int(x) for x in args.subgoal_offsets if int(x) <= goal_offset]
    if args.subgoal_schedule == "midpoint":
        local = int(round(float(goal_offset) * float(args.midpoint_fraction)))
        local = max(1, min(goal_offset, local))
        return [local]
    if args.subgoal_schedule == "exponential":
        base = int(args.exponential_base_offset)
        if goal_offset <= base:
            return [goal_offset]
        local = base
        while local * 2 < goal_offset:
            local *= 2
        return [local]
    raise ValueError(f"Unsupported subgoal schedule: {args.subgoal_schedule}")


def subset_pair_indices(
    labels: np.ndarray,
    *,
    limit: int,
    seed: int,
    balanced: bool,
) -> np.ndarray:
    total = int(len(labels))
    if limit <= 0 or limit >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if not balanced:
        return np.sort(rng.choice(total, size=int(limit), replace=False))

    groups = [np.flatnonzero(labels == value) for value in sorted(np.unique(labels))]
    base, remainder = divmod(int(limit), len(groups))
    selected: list[np.ndarray] = []
    for group_index, group in enumerate(groups):
        requested = base + int(group_index < remainder)
        take = min(int(len(group)), requested)
        if take:
            selected.append(rng.choice(group, size=take, replace=False))
    keep = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)

    # Redistribute any quota left by a short group without duplicating pairs.
    if len(keep) < int(limit):
        available = np.setdiff1d(
            np.arange(total, dtype=np.int64), keep, assume_unique=False
        )
        extra = rng.choice(available, size=int(limit) - len(keep), replace=False)
        keep = np.concatenate([keep, extra])
    return np.sort(keep.astype(np.int64, copy=False))


def build_subgoal_pairs(dataset, specs, args, *, split_name: str):
    if args.dense_joint_sampling:
        limit = int(args.max_train_pairs if split_name == "train" else args.max_val_pairs)
        pair_specs, pair_goals, pair_subgoals, diagnostics = sample_dense_unique_pairs(
            dataset,
            specs,
            history_len=args.history_len,
            frameskip=args.frameskip,
            goal_offsets=list(args.goal_offsets),
            action_offsets=list(args.subgoal_offsets),
            limit=limit,
            seed=args.seed + (8101 if split_name == "train" else 9101),
            allow_repeats=args.dense_allow_repeats,
            balance_by_goal=args.dense_balance_goals,
        )
        current = np.asarray(
            [
                int(spec.start) + (int(args.history_len) - 1) * int(args.frameskip)
                for spec in specs
            ],
            dtype=np.int64,
        )
        print(
            f"dense-{split_name} sampler {json.dumps(diagnostics, sort_keys=True)}",
            flush=True,
        )
        return (
            pair_specs,
            current[pair_specs] + pair_goals,
            current[pair_specs] + pair_subgoals,
            pair_goals,
            pair_subgoals,
            int(len(pair_specs)),
        )
    spec_indices = []
    goal_frames = []
    subgoal_frames = []
    goal_offsets = []
    subgoal_offsets = []
    regular_offsets = [int(x) for x in args.goal_offsets]
    for spec_index, spec in enumerate(specs):
        current = int(spec.start) + (int(args.history_len) - 1) * int(args.frameskip)
        final = int(dataset.lengths[spec.local_episode]) - 1
        for offset in regular_offsets:
            if offset <= 0:
                continue
            goal_frame = current + offset
            for requested_subgoal_offset in scheduled_subgoal_offsets(offset, args):
                local_offset = int(requested_subgoal_offset)
                subgoal_frame = current + local_offset
                if goal_frame <= final and subgoal_frame <= final:
                    spec_indices.append(spec_index)
                    goal_frames.append(goal_frame)
                    subgoal_frames.append(subgoal_frame)
                    goal_offsets.append(offset)
                    subgoal_offsets.append(local_offset)
        if args.include_final_goal and final > current:
            offset = final - current
            for requested_subgoal_offset in scheduled_subgoal_offsets(offset, args):
                local_offset = min(int(requested_subgoal_offset), offset)
                if local_offset <= 0:
                    continue
                spec_indices.append(spec_index)
                goal_frames.append(final)
                subgoal_frames.append(current + local_offset)
                goal_offsets.append(offset)
                subgoal_offsets.append(local_offset)
    total = len(spec_indices)
    if total <= 0:
        raise ValueError(f"No valid {split_name} subgoal pairs")
    limit = int(args.max_train_pairs if split_name == "train" else args.max_val_pairs)
    if limit and limit < total:
        keep = subset_pair_indices(
            np.asarray(goal_offsets, dtype=np.int64),
            limit=limit,
            seed=args.seed + (8101 if split_name == "train" else 9101),
            balanced=bool(args.balance_goal_offsets),
        )
        spec_indices = np.asarray(spec_indices, dtype=np.int64)[keep]
        goal_frames = np.asarray(goal_frames, dtype=np.int64)[keep]
        subgoal_frames = np.asarray(subgoal_frames, dtype=np.int64)[keep]
        goal_offsets = np.asarray(goal_offsets, dtype=np.int64)[keep]
        subgoal_offsets = np.asarray(subgoal_offsets, dtype=np.int64)[keep]
    return (
        np.asarray(spec_indices, dtype=np.int64),
        np.asarray(goal_frames, dtype=np.int64),
        np.asarray(subgoal_frames, dtype=np.int64),
        np.asarray(goal_offsets, dtype=np.int64),
        np.asarray(subgoal_offsets, dtype=np.int64),
        total,
    )
def prepare_batch(batch, lewm, stats, args, device):
    dtype = next(lewm.parameters()).dtype
    history_pixels = image_batch_to_lewm(
        batch["pixels"][:, : args.history_len].to(device),
        args.image_size,
    ).to(dtype)
    goal_pixels = image_batch_to_lewm(
        batch["goal_pixels"].to(device),
        args.image_size,
    ).to(dtype)
    subgoal_pixels = image_batch_to_lewm(
        batch["subgoal_pixels"].to(device),
        args.image_size,
    ).to(dtype)
    with torch.no_grad():
        history_latents = encode_lewm_context(lewm, history_pixels)
        goal_latents = encode_lewm_context(lewm, goal_pixels)
        subgoal_latents = encode_lewm_context(lewm, subgoal_pixels)
    lowdim = lowdim_from_batch(
        batch,
        args.history_len,
        args.lowdim_keys,
    ).to(device)
    lowdim_n = normalize_lowdim(lowdim, stats)
    goal_offset = batch["goal_offset"].to(device).float()
    subgoal_offset = batch["subgoal_offset"].to(device).float()
    return history_latents, goal_latents, subgoal_latents, lowdim_n, goal_offset, subgoal_offset
def per_sample_scores(prediction, target, base, history_latents, goal_latents):
    target_tokens = target.size(1)
    current = history_latents[:, -target_tokens:]
    flat_pred = prediction.flatten(1)
    flat_target = target.flatten(1)
    return {
        "pred_l1": torch.abs(prediction - target).flatten(1).mean(dim=1),
        "pred_mse": (prediction - target).pow(2).flatten(1).mean(dim=1),
        "pred_cos": 1.0 - torch.nn.functional.cosine_similarity(
            flat_pred,
            flat_target,
            dim=-1,
        ),
        "base_l1": torch.abs(base - target).flatten(1).mean(dim=1),
        "goal_l1": torch.abs(goal_latents - target).flatten(1).mean(dim=1),
        "current_l1": torch.abs(current - target).flatten(1).mean(dim=1),
    }
def update_vector(total, values, mask=None):
    if mask is None:
        n = next(iter(values.values())).numel()
        selected = values
    else:
        n = int(mask.sum().item())
        if n <= 0:
            return
        selected = {key: value[mask] for key, value in values.items()}
    total["n"] = total.get("n", 0) + int(n)
    for key, value in selected.items():
        total[key] = total.get(key, 0.0) + float(value.sum().item())
def finalize_vector(total):
    n = max(int(total.get("n", 0)), 1)
    return {key: value / n for key, value in total.items() if key != "n"}
@torch.no_grad()
def evaluate(model, lewm, loader, stats, args, device):
    model.eval()
    totals = {"all": {}}
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        history, goal, target, lowdim, goal_offset, subgoal_offset = prepare_batch(
            batch,
            lewm,
            stats,
            args,
            device,
        )
        outputs = model(history, goal, lowdim, goal_offset, subgoal_offset)
        scores = per_sample_scores(
            outputs["prediction"],
            target,
            outputs["base"],
            history,
            goal,
        )
        update_vector(totals["all"], scores)
        for offset in torch.unique(goal_offset.long()).tolist():
            key = f"h{int(offset)}"
            totals.setdefault(key, {})
            update_vector(totals[key], scores, goal_offset.long() == int(offset))
    return {key: finalize_vector(value) for key, value in totals.items()}
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split = load_json(args.split)
    dataset = load_swm_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        frameskip=args.frameskip,
        num_steps=args.history_len,
        keys_to_load=["pixels", "action", "proprio", "state"],
    )
    train_pool = build_window_specs(
        dataset,
        split,
        "train",
        context_len=args.history_len,
        plan_horizon=1,
    )
    val_pool = build_window_specs(
        dataset,
        split,
        "val",
        context_len=args.history_len,
        plan_horizon=1,
    )
    train_specs = subset_specs(train_pool, args.max_train_windows, args.seed)
    val_specs = subset_specs(val_pool, args.max_val_windows, args.seed + 1)
    if not train_specs or not val_specs:
        raise ValueError("Empty train or val window set")
    train_pairs = build_subgoal_pairs(dataset, train_specs, args, split_name="train")
    val_pairs = build_subgoal_pairs(dataset, val_specs, args, split_name="val")
    print(
        f"windows train_pool={len(train_pool)} val_pool={len(val_pool)} "
        f"selected_train_windows={len(train_specs)} selected_val_windows={len(val_specs)} "
        f"train_pairs={len(train_pairs[0])}/{train_pairs[-1]} "
        f"val_pairs={len(val_pairs[0])}/{val_pairs[-1]} "
        f"goal_offsets={args.goal_offsets} subgoal_offsets={args.subgoal_offsets}",
        flush=True,
    )
    stats_cpu = compute_stats(dataset, split, args.lowdim_keys)
    stats = move_stats(stats_cpu, device)
    train_data = SubgoalPairDataset(dataset, train_specs, *train_pairs[:-1])
    val_data = SubgoalPairDataset(dataset, val_specs, *val_pairs[:-1])
    loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": not args.no_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    lewm = load_lewm(args.policy, device=device, bf16=args.bf16)
    first = next(iter(DataLoader(train_data, batch_size=1)))
    history, goal, target, lowdim, goal_offset, subgoal_offset = prepare_batch(
        first,
        lewm,
        stats,
        args,
        device,
    )
    model_config = {
        "latent_dim": int(history.size(-1)),
        "lowdim_dim": int(stats_cpu["lowdim_mean"].numel()),
        "hidden_dim": int(args.hidden_dim),
        "num_heads": int(args.num_heads),
        "depth": int(args.depth),
        "max_goal_offset": int(max(max(args.goal_offsets), max(args.subgoal_offsets))),
        "predict_residual_from": str(args.predict_residual_from),
        "pooling": str(args.pooling),
        "goal_condition_mode": str(args.goal_condition_mode),
    }
    model = PushtSubgoalPrior(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"
    start_epoch = 0
    best_val = float("inf")
    if latest_path.exists() and not args.no_resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_config") != model_config:
            raise ValueError("Checkpoint model_config mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val_pred_l1", best_val))
        print(f"resuming {latest_path} epoch={start_epoch}", flush=True)
    manifest = {
        "script": "scripts/pusht/train_pusht_subgoal_prior.py",
        "prior_type": "pusht_subgoal_prior_v1",
        "scientific_hypothesis": (
            "Expert trajectories provide self-supervised intermediate latent "
            "subgoals. Learning p(z_{t+25}|z_t,z_{t+H},H) can factor long "
            "planning into reachable local goals plus a local action prior."
        ),
        "dataset": args.dataset,
        "split": args.split,
        "args": vars(args),
        "model_config": model_config,
        "train_pool_windows": len(train_pool),
        "val_pool_windows": len(val_pool),
        "selected_train_windows": len(train_specs),
        "selected_val_windows": len(val_specs),
        "selected_train_pairs": len(train_pairs[0]),
        "available_train_pairs": int(train_pairs[-1]),
        "selected_val_pairs": len(val_pairs[0]),
        "available_val_pairs": int(val_pairs[-1]),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = {}
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            history, goal, target, lowdim, goal_offset, subgoal_offset = prepare_batch(
                batch,
                lewm,
                stats,
                args,
                device,
            )
            outputs = model(history, goal, lowdim, goal_offset, subgoal_offset)
            losses = model.loss(
                outputs,
                target,
                cosine_weight=args.cosine_weight,
                smooth_l1_beta=args.smooth_l1_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scores = per_sample_scores(
                outputs["prediction"].detach(),
                target,
                outputs["base"].detach(),
                history,
                goal,
            )
            update(
                total,
                {
                    "loss": losses["loss"].item(),
                    "smooth_l1": losses["smooth_l1"].item(),
                    "cosine": losses["cosine"].item(),
                    "pred_l1": scores["pred_l1"].mean().item(),
                    "goal_l1": scores["goal_l1"].mean().item(),
                    "current_l1": scores["current_l1"].mean().item(),
                },
                target.size(0),
            )
            if batch_index % 20 == 0:
                row = finalize(total)
                print(
                    f"epoch={epoch + 1} batch={batch_index} "
                    f"loss={row['loss']:.4f} pred_l1={row['pred_l1']:.4f} "
                    f"goal_l1={row['goal_l1']:.4f} current_l1={row['current_l1']:.4f}",
                    flush=True,
                )
        train_metrics = finalize(total)
        val_metrics = evaluate(model, lewm, val_loader, stats, args, device)
        selection = float(val_metrics["all"]["pred_l1"])
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train": train_metrics,
                        "val": val_metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"val_pred_l1={val_metrics['all']['pred_l1']:.4f} "
            f"val_goal_l1={val_metrics['all']['goal_l1']:.4f} "
            f"val_current_l1={val_metrics['all']['current_l1']:.4f} "
            f"val_cos={val_metrics['all']['pred_cos']:.4f}",
            flush=True,
        )
        payload = {
            "prior_type": "pusht_subgoal_prior_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": stats_cpu,
            "model_config": model_config,
            "epoch": epoch + 1,
            "best_val_pred_l1": min(best_val, selection),
            "last_train_metrics": train_metrics,
            "last_val_metrics": val_metrics,
            "run_manifest": manifest,
        }
        save_subgoal_prior(latest_path, payload)
        if selection < best_val:
            best_val = selection
            save_subgoal_prior(best_path, payload)
            print(f"wrote new best checkpoint: {best_path}", flush=True)
        print(f"wrote latest checkpoint: {latest_path}", flush=True)
if __name__ == "__main__":
    main()
