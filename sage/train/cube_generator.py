"""Train a latent subgoal generator for LeWM dataset-planning tasks.

The supervised target is an intermediate replay latent.  Given the current
history and a farther replay goal at t+H, the model predicts the latent at
t+S, where S is usually one local control chunk (25 env steps for Cube/PushT
with horizon=5 and action_block=5).  Online evaluation can then replace the
world-model goal embedding by this predicted local subgoal.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings(
    "ignore",
    message="lancedb fork support is experimental.*",
    category=RuntimeWarning,
)

from sage.runtime.lewm import (
    build_window_specs,
    encode_lewm_context,
    image_batch_to_lewm,
    load_json,
    load_lewm,
    load_swm_dataset,
    lowdim_from_batch,
    normalize_lowdim,
    split_episode_sets,
)
from sage.models.subgoal import PushtSubgoalPrior, save_subgoal_prior
from sage.sampling import sample_dense_unique_pairs
from sage.training import (
    compute_stats,
    move_stats,
    subset_specs,
)
from sage.runtime.frame_latent_cache import FrameLatentCache


class SubgoalPairDataset(Dataset):
    def __init__(
        self,
        dataset,
        specs,
        pair_spec_indices,
        pair_goal_offsets,
        pair_subgoal_offsets,
        *,
        context_len: int,
        frameskip: int,
        episode_base=None,
    ) -> None:
        self.dataset = dataset
        self.specs = list(specs)
        self.pair_spec_indices = np.asarray(pair_spec_indices, dtype=np.int64)
        self.pair_goal_offsets = np.asarray(pair_goal_offsets, dtype=np.int64)
        self.pair_subgoal_offsets = np.asarray(pair_subgoal_offsets, dtype=np.int64)
        self.context_len = int(context_len)
        self.frameskip = int(frameskip)
        self.episode_base = None if episode_base is None else np.asarray(episode_base, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.pair_spec_indices))

    def __getitem__(self, index: int) -> dict:
        spec = self.specs[int(self.pair_spec_indices[index])]
        item = self.dataset[spec.dataset_index]
        cur = (self.context_len - 1) * self.frameskip
        goal_step = int(self.pair_goal_offsets[index])
        subgoal_step = int(self.pair_subgoal_offsets[index])
        if goal_step % self.frameskip or subgoal_step % self.frameskip:
            raise ValueError("goal/subgoal offsets must be divisible by frameskip")
        goal_index = cur // self.frameskip + goal_step // self.frameskip
        subgoal_index = cur // self.frameskip + subgoal_step // self.frameskip
        item["goal_pixels"] = item["pixels"][goal_index : goal_index + 1]
        item["subgoal_pixels"] = item["pixels"][subgoal_index : subgoal_index + 1]
        item["episode_id"] = torch.tensor(spec.episode_id, dtype=torch.long)
        item["start"] = torch.tensor(spec.start, dtype=torch.long)
        item["goal_offset"] = torch.tensor(goal_step, dtype=torch.long)
        item["subgoal_offset"] = torch.tensor(subgoal_step, dtype=torch.long)
        if self.episode_base is not None:
            base = int(self.episode_base[int(spec.local_episode)]) + int(spec.start)
            item["history_frame_indices"] = torch.arange(
                base,
                base + int(self.context_len),
                dtype=torch.long,
            )
            item["goal_frame_index"] = torch.tensor(base + int(goal_index), dtype=torch.long)
            item["subgoal_frame_index"] = torch.tensor(base + int(subgoal_index), dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy", default="quentinll/lewm-cube")
    parser.add_argument(
        "--frame-latent-cache",
        default=None,
        help="Optional frame-level LeWM encode cache built by build_frame_latent_cache.py.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--pooling", choices=["attention", "mean", "decoder"], default="attention")
    parser.add_argument(
        "--predict-residual-from",
        choices=["goal", "current", "zero"],
        default="zero",
    )
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--subgoal-offset", type=int, default=25)
    parser.add_argument(
        "--subgoal-offsets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional variable local subgoal offsets in env steps. If omitted, "
            "--subgoal-offset is used. Offsets larger than the far goal are skipped."
        ),
    )
    parser.add_argument("--goal-offsets", nargs="+", type=int, default=[50, 75, 100])
    parser.add_argument("--include-h25-anchor", action="store_true")
    parser.add_argument("--lowdim-keys", nargs="+", default=["observation"])
    parser.add_argument("--no-lowdim", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=200000)
    parser.add_argument("--max-val-windows", type=int, default=20000)
    parser.add_argument("--max-train-pairs", type=int, default=200000)
    parser.add_argument("--max-val-pairs", type=int, default=20000)
    parser.add_argument(
        "--dense-joint-sampling",
        action="store_true",
        help="Joint-balance (goal_offset, subgoal_offset) cells with globally unique windows.",
    )
    parser.add_argument(
        "--dense-balance-goals",
        action="store_true",
        help="Balance the dense pair budget over goal offsets before subgoal offsets.",
    )
    parser.add_argument(
        "--dense-allow-repeats",
        action="store_true",
        help="After broad unique coverage, fill an oversized dense budget with balanced repeats.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory for CUDA driver configurations that reject it.",
    )
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.no_lowdim:
        args.lowdim_keys = []
    if args.include_h25_anchor and 25 not in args.goal_offsets:
        args.goal_offsets = [25, *args.goal_offsets]
    args.goal_offsets = sorted(set(int(x) for x in args.goal_offsets))
    if args.subgoal_offsets is None:
        args.subgoal_offsets = [int(args.subgoal_offset)]
    args.subgoal_offsets = sorted(set(int(x) for x in args.subgoal_offsets))
    if any(int(x) <= 0 for x in args.subgoal_offsets):
        raise ValueError(f"subgoal offsets must be positive, got {args.subgoal_offsets}")
    return args


def build_pairs(dataset, specs, args, *, split_name: str):
    if args.dense_joint_sampling:
        limit = int(args.max_train_pairs if split_name == "train" else args.max_val_pairs)
        spec_indices, goal_offsets, subgoal_offsets, diagnostics = sample_dense_unique_pairs(
            dataset,
            specs,
            history_len=args.context_len,
            frameskip=args.frameskip,
            goal_offsets=list(args.goal_offsets),
            action_offsets=list(args.subgoal_offsets),
            limit=limit,
            seed=args.seed + (811 if split_name == "train" else 919),
            allow_repeats=args.dense_allow_repeats,
            balance_by_goal=args.dense_balance_goals,
        )
        print(f"dense-{split_name} sampler {json.dumps(diagnostics, sort_keys=True)}", flush=True)
        return spec_indices, goal_offsets, subgoal_offsets, int(len(spec_indices))
    spec_indices: list[int] = []
    goal_offsets: list[int] = []
    subgoal_offsets: list[int] = []
    for spec_index, _spec in enumerate(specs):
        for offset in args.goal_offsets:
            if offset <= 0:
                continue
            if offset % args.frameskip:
                raise ValueError(f"goal offset {offset} must be divisible by frameskip={args.frameskip}")
            for local in args.subgoal_offsets:
                local = int(local)
                if local > int(offset):
                    continue
                if local % args.frameskip:
                    raise ValueError(f"subgoal offset {local} must be divisible by frameskip={args.frameskip}")
                spec_indices.append(spec_index)
                goal_offsets.append(int(offset))
                subgoal_offsets.append(int(local))
    total = len(spec_indices)
    limit = int(args.max_train_pairs if split_name == "train" else args.max_val_pairs)
    if limit and limit < total:
        rng = np.random.default_rng(args.seed + (811 if split_name == "train" else 919))
        keep = np.sort(rng.choice(total, size=limit, replace=False))
        spec_indices = np.asarray(spec_indices, dtype=np.int64)[keep].tolist()
        goal_offsets = np.asarray(goal_offsets, dtype=np.int64)[keep].tolist()
        subgoal_offsets = np.asarray(subgoal_offsets, dtype=np.int64)[keep].tolist()
    return (
        np.asarray(spec_indices, dtype=np.int64),
        np.asarray(goal_offsets, dtype=np.int64),
        np.asarray(subgoal_offsets, dtype=np.int64),
        total,
    )


def _cached_latents(frame_cache: FrameLatentCache, indices, device):
    latents = frame_cache.get(indices, device=device, dtype=torch.float32)
    if latents.ndim == 4 and latents.size(-2) == 1:
        latents = latents.squeeze(-2)
    return latents


def prepare_batch(batch, lewm, stats, args, device, *, frame_cache: FrameLatentCache | None = None):
    if frame_cache is None:
        dtype = next(lewm.parameters()).dtype
        history_pixels = image_batch_to_lewm(
            batch["pixels"][:, : args.context_len].to(device, non_blocking=True),
            args.image_size,
        ).to(dtype=dtype)
        goal_pixels = image_batch_to_lewm(
            batch["goal_pixels"].to(device, non_blocking=True),
            args.image_size,
        ).to(dtype=dtype)
        subgoal_pixels = image_batch_to_lewm(
            batch["subgoal_pixels"].to(device, non_blocking=True),
            args.image_size,
        ).to(dtype=dtype)
        with torch.no_grad():
            history_latents = encode_lewm_context(lewm, history_pixels)
            goal_latents = encode_lewm_context(lewm, goal_pixels)
            subgoal_latents = encode_lewm_context(lewm, subgoal_pixels)
    else:
        history_latents = _cached_latents(frame_cache, batch["history_frame_indices"], device)
        goal_latents = _cached_latents(frame_cache, batch["goal_frame_index"], device)
        subgoal_latents = _cached_latents(frame_cache, batch["subgoal_frame_index"], device)
    if args.lowdim_keys:
        lowdim = lowdim_from_batch(batch, args.context_len, args.lowdim_keys).to(device, non_blocking=True)
        lowdim_n = normalize_lowdim(lowdim, stats)
    else:
        lowdim_n = torch.empty(history_latents.size(0), 0, device=device)
    return (
        history_latents,
        goal_latents,
        subgoal_latents,
        lowdim_n,
        batch["goal_offset"].to(device).float(),
        batch["subgoal_offset"].to(device).float(),
    )


def per_sample_scores(prediction, target, base, history_latents, goal_latents):
    target_tokens = target.size(1)
    current = history_latents[:, -target_tokens:]
    return {
        "pred_l1": torch.abs(prediction - target).flatten(1).mean(dim=1),
        "pred_mse": (prediction - target).pow(2).flatten(1).mean(dim=1),
        "pred_cos": 1.0 - torch.nn.functional.cosine_similarity(
            prediction.flatten(1),
            target.flatten(1),
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
def evaluate(model, lewm, loader, stats, args, device, *, frame_cache: FrameLatentCache | None = None):
    model.eval()
    totals: dict[str, dict] = {"all": {}}
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        history, goal, target, lowdim, goal_offset, subgoal_offset = prepare_batch(
            batch, lewm, stats, args, device, frame_cache=frame_cache
        )
        outputs = model(history, goal, lowdim, goal_offset, subgoal_offset)
        scores = per_sample_scores(outputs["prediction"], target, outputs["base"], history, goal)
        update_vector(totals["all"], scores)
        for offset in args.goal_offsets:
            mask = goal_offset.long() == int(offset)
            totals.setdefault(f"h{offset}", {})
            update_vector(totals[f"h{offset}"], scores, mask)
    return {key: finalize_vector(value) for key, value in totals.items()}


def scalar_loss(model, target, outputs, args):
    loss = model.loss(
        outputs,
        target,
        cosine_weight=args.cosine_weight,
        smooth_l1_beta=args.smooth_l1_beta,
    )
    return loss


def main() -> None:
    args = parse_args()
    dataset_name = str(args.dataset).lower()
    if str(args.policy).lower().startswith("pusht/") or "pusht" in dataset_name:
        raise ValueError(
            "PushT subgoal generators must use scripts/pusht/train_pusht_subgoal_prior.py. "
            "The generic Cube/OGBench data path has different window and episode semantics."
        )
    # Reuse the action-prior stats helper, which expects these optional flags.
    args.tworoom_local_goal = False
    args.cube_local_goal = False
    args.goal_lowdim_keys = []
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split = load_json(args.split)
    max_goal_frames = max(args.goal_offsets) // args.frameskip
    max_subgoal_frames = max(int(x) for x in args.subgoal_offsets) // int(args.frameskip)
    num_steps = int(args.context_len) + max(max_goal_frames, max_subgoal_frames)
    keys_to_load = ["pixels", *args.lowdim_keys]
    dataset = load_swm_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        frameskip=args.frameskip,
        num_steps=num_steps,
        keys_to_load=list(dict.fromkeys(keys_to_load)),
    )
    planning_horizon = max_subgoal_frames if args.dense_joint_sampling else max_goal_frames
    train_specs = build_window_specs(
        dataset,
        split,
        "train",
        context_len=args.context_len,
        plan_horizon=planning_horizon,
    )
    val_specs = build_window_specs(
        dataset,
        split,
        "val",
        context_len=args.context_len,
        plan_horizon=planning_horizon,
    )
    train_specs = subset_specs(train_specs, args.max_train_windows, args.seed)
    val_specs = subset_specs(val_specs, args.max_val_windows, args.seed + 1)
    if not train_specs or not val_specs:
        raise ValueError(f"Need non-empty train/val specs, got {len(train_specs)}/{len(val_specs)}")

    if args.lowdim_keys:
        stats_cpu = compute_stats(dataset, split, args.lowdim_keys, train_specs=train_specs, args=args)
    else:
        stats_cpu = {
            "lowdim_mean": torch.empty(0),
            "lowdim_std": torch.empty(0),
            "lowdim_keys": [],
            "goal_lowdim_keys": [],
        }
    stats = move_stats(stats_cpu, device)

    train_idx, train_goals, train_subgoals, train_total = build_pairs(dataset, train_specs, args, split_name="train")
    val_idx, val_goals, val_subgoals, val_total = build_pairs(dataset, val_specs, args, split_name="val")
    print(
        "windows "
        f"train_pool={len(train_specs)} val_pool={len(val_specs)} "
        f"train_pairs={len(train_idx)}/{train_total} val_pairs={len(val_idx)}/{val_total} "
        f"goal_offsets={args.goal_offsets} subgoal_offsets={args.subgoal_offsets}",
        flush=True,
    )

    frame_cache = None
    episode_base = None
    lewm = None
    if args.frame_latent_cache:
        frame_cache = FrameLatentCache(args.frame_latent_cache)
        frame_cache.validate(
            dataset=args.dataset,
            policy=args.policy,
            frameskip=args.frameskip,
            image_size=args.image_size,
        )
        episode_base = np.asarray(frame_cache.episode_base, dtype=np.int64)
        print(f"using frame latent cache: {args.frame_latent_cache}", flush=True)
    else:
        print("loading frozen LeWM...", flush=True)
        lewm = load_lewm(args.policy, device=device, bf16=args.bf16)
    train_loader = DataLoader(
        SubgoalPairDataset(
            dataset,
            train_specs,
            train_idx,
            train_goals,
            train_subgoals,
            context_len=args.context_len,
            frameskip=args.frameskip,
            episode_base=episode_base,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        SubgoalPairDataset(
            dataset,
            val_specs,
            val_idx,
            val_goals,
            val_subgoals,
            context_len=args.context_len,
            frameskip=args.frameskip,
            episode_base=episode_base,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    first = next(iter(DataLoader(
        SubgoalPairDataset(
            dataset,
            train_specs[:1],
            np.asarray([0]),
            np.asarray([args.goal_offsets[0]]),
            np.asarray([min(args.subgoal_offset, args.goal_offsets[0])]),
            context_len=args.context_len,
            frameskip=args.frameskip,
            episode_base=episode_base,
        ),
        batch_size=1,
    )))
    with torch.no_grad():
        if frame_cache is None:
            pixels = image_batch_to_lewm(
                first["pixels"][:, : args.context_len].to(device), args.image_size
            ).to(dtype=next(lewm.parameters()).dtype)
            latent_dim = int(encode_lewm_context(lewm, pixels).size(-1))
        else:
            latent_dim = int(
                _cached_latents(frame_cache, first["history_frame_indices"], device).size(-1)
            )
    lowdim_dim = int(stats_cpu["lowdim_mean"].numel())
    model_config = {
        "latent_dim": latent_dim,
        "lowdim_dim": lowdim_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "max_goal_offset": max(args.goal_offsets),
        "predict_residual_from": args.predict_residual_from,
        "pooling": args.pooling,
    }
    model = PushtSubgoalPrior(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"
    start_epoch = 0
    best_val = float("inf")
    if latest_path.exists() and not args.no_resume:
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val_pred_l1", best_val))
        print(f"resumed epoch={start_epoch} best_val={best_val:.4f}", flush=True)

    manifest = {
        "script": "scripts/lewm_prior/train_subgoal_prior.py",
        "prior_type": "pusht_subgoal_prior_v1",
        "dataset": args.dataset,
        "split": args.split,
        "args": vars(args),
        "model_config": model_config,
        "selected_train_windows": len(train_specs),
        "selected_val_windows": len(val_specs),
        "selected_train_pairs": int(len(train_idx)),
        "selected_val_pairs": int(len(val_idx)),
        "train_pool_windows": int(len(split_episode_sets(split)["train"])),
        "stats_lowdim_keys": list(args.lowdim_keys),
        "scientific_hypothesis": (
            "Expert replay trajectories supervise p(z_{t+S}|history,z_{t+H},H,S). "
            "Online planning can replace a far latent goal with a generated local subgoal."
        ),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        model.train()
        total = {"n": 0, "loss": 0.0, "pred_l1": 0.0}
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            history, goal, target, lowdim, goal_offset, subgoal_offset = prepare_batch(
                batch, lewm, stats, args, device, frame_cache=frame_cache
            )
            outputs = model(history, goal, lowdim, goal_offset, subgoal_offset)
            losses = scalar_loss(model, target, outputs, args)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n = int(target.size(0))
            pred_l1 = torch.abs(outputs["prediction"].detach() - target).flatten(1).mean().item()
            total["n"] += n
            total["loss"] += float(losses["loss"].item()) * n
            total["pred_l1"] += pred_l1 * n
            if batch_index % 50 == 0:
                denom = max(total["n"], 1)
                print(
                    f"epoch={epoch} batch={batch_index} "
                    f"loss={total['loss']/denom:.4f} pred_l1={total['pred_l1']/denom:.4f}",
                    flush=True,
                )

        val = evaluate(model, lewm, val_loader, stats, args, device, frame_cache=frame_cache)
        train_loss = total["loss"] / max(total["n"], 1)
        train_l1 = total["pred_l1"] / max(total["n"], 1)
        all_val = val.get("all", {})
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} train_pred_l1={train_l1:.4f} "
            f"val_pred_l1={all_val.get('pred_l1', float('nan')):.4f} "
            f"val_current_l1={all_val.get('current_l1', float('nan')):.4f} "
            f"val_goal_l1={all_val.get('goal_l1', float('nan')):.4f}",
            flush=True,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pred_l1": train_l1,
            "val": val,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        val_pred_l1 = float(all_val.get("pred_l1", float("inf")))
        best_val = min(best_val, val_pred_l1)
        payload = {
            "prior_type": "pusht_subgoal_prior_v1",
            "epoch": epoch,
            "best_val_pred_l1": best_val,
            "model_config": model_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": stats_cpu,
            "run_manifest": manifest,
        }
        save_subgoal_prior(latest_path, payload)
        print(f"wrote latest checkpoint: {latest_path}", flush=True)
        if val_pred_l1 <= best_val + 1.0e-12:
            save_subgoal_prior(best_path, payload)
            print(f"wrote new best checkpoint: {best_path}", flush=True)
    print(f"done best_val_pred_l1={best_val:.4f} wrote={latest_path}", flush=True)


if __name__ == "__main__":
    main()
