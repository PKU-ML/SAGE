"""Train a variable-time LeWM action prior for OGBench/Cube-style datasets.

This mirrors the PushT variable-time prior path, but keeps the data plumbing
generic for HDF5/Lance datasets used by ``scripts/lewm_prior``.  Each row
chooses a far replay goal at ``t + Delta`` and a local subgoal / action chunk
length ``tau``.  The action prior predicts a tau-step action chunk conditioned
on history latents, a proposal-goal latent, the far goal latent,
low-dimensional state, and the pair (Delta, tau).  ``--prior-goal-source``
selects either the local ``t + tau`` latent used by the hierarchy or the direct
far-goal ``t + Delta`` latent used by the matched flat-prior ablation.
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
    denormalize_action_blocks,
    encode_lewm_context,
    image_batch_to_lewm,
    load_json,
    load_lewm,
    load_swm_dataset,
    lowdim_from_batch,
    normalize_action_blocks,
    normalize_lowdim,
    save_bc_checkpoint,
    target_action_chunk,
)
from sage.models.subgoal import load_subgoal_prior
from sage.sampling import sample_dense_unique_pairs
from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.training import (
    compute_stats,
    finalize,
    finite_training_mask,
    move_stats,
    subset_specs,
    update,
)
from sage.runtime.frame_latent_cache import FrameLatentCache


class VariableActionDataset(Dataset):
    def __init__(self, dataset, specs, rows, *, context_len: int, frameskip: int, episode_base=None):
        self.dataset = dataset
        self.specs = list(specs)
        self.rows = list(rows)
        self.context_len = int(context_len)
        self.frameskip = int(frameskip)
        self.episode_base = None if episode_base is None else np.asarray(episode_base, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[int(index)]
        spec = self.specs[int(row["spec_index"])]
        item = self.dataset[spec.dataset_index]
        cur = (self.context_len - 1) * self.frameskip
        tau = int(row["action_offset"])
        delta = int(row["goal_offset"])
        if tau % self.frameskip or delta % self.frameskip:
            raise ValueError("action/goal offsets must be divisible by frameskip")
        local_index = cur // self.frameskip + tau // self.frameskip
        far_index = cur // self.frameskip + delta // self.frameskip
        item["goal_pixels"] = item["pixels"][local_index : local_index + 1]
        item["far_goal_pixels"] = item["pixels"][far_index : far_index + 1]
        item["action_offset"] = torch.tensor(tau, dtype=torch.long)
        item["goal_offset"] = torch.tensor(delta, dtype=torch.long)
        item["action_tokens"] = torch.tensor(tau // self.frameskip, dtype=torch.long)
        item["episode_id"] = torch.tensor(spec.episode_id, dtype=torch.long)
        item["start"] = torch.tensor(spec.start, dtype=torch.long)
        if self.episode_base is not None:
            base = int(self.episode_base[int(spec.local_episode)]) + int(spec.start)
            item["history_frame_indices"] = torch.arange(
                base,
                base + int(self.context_len),
                dtype=torch.long,
            )
            item["goal_frame_index"] = torch.tensor(base + int(local_index), dtype=torch.long)
            item["far_goal_frame_index"] = torch.tensor(base + int(far_index), dtype=torch.long)
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
    parser.add_argument("--mode-l1-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-offsets", nargs="+", type=int, default=[10, 15, 20, 25, 30, 35])
    parser.add_argument("--goal-offsets", nargs="+", type=int, default=[25, 50, 75, 100])
    parser.add_argument("--lowdim-keys", nargs="+", default=["observation"])
    parser.add_argument("--no-lowdim", action="store_true")
    parser.add_argument("--subgoal-generator-checkpoint", default=None)
    parser.add_argument("--generated-subgoal-ratio", type=float, default=0.0)
    parser.add_argument("--eval-use-generated-subgoal", action="store_true")
    parser.add_argument(
        "--prior-goal-source",
        choices=["local", "far"],
        default="local",
        help=(
            "Latent supplied through the option prior's proposal-goal tokens. "
            "local uses the expert/generated t+tau subgoal; far duplicates "
            "the t+Delta query goal for the matched flat-prior ablation."
        ),
    )
    parser.add_argument("--max-train-windows", type=int, default=300000)
    parser.add_argument("--max-val-windows", type=int, default=30000)
    parser.add_argument("--max-train-examples", type=int, default=300000)
    parser.add_argument("--max-val-examples", type=int, default=30000)
    parser.add_argument(
        "--dense-joint-sampling",
        action="store_true",
        help="Joint-balance (goal_offset, action_offset) cells with globally unique windows.",
    )
    parser.add_argument(
        "--dense-balance-goals",
        action="store_true",
        help="Balance the dense example budget over goal offsets before action offsets.",
    )
    parser.add_argument(
        "--dense-allow-repeats",
        action="store_true",
        help="After broad unique coverage, fill an oversized dense budget with balanced repeats.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--coverage-threshold", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--save-epochs",
        nargs="*",
        type=int,
        default=[],
        help="Also save the specified epoch_XXX.pt snapshots.",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory; useful after CUDA/NVML driver errors.",
    )
    args = parser.parse_args()
    if args.no_lowdim:
        args.lowdim_keys = []
    args.action_offsets = sorted(set(int(x) for x in args.action_offsets))
    args.goal_offsets = sorted(set(int(x) for x in args.goal_offsets))
    if any(x <= 0 or x % int(args.frameskip) for x in args.action_offsets):
        raise ValueError("--action-offsets must be positive multiples of frameskip")
    if any(x <= 0 or x % int(args.frameskip) for x in args.goal_offsets):
        raise ValueError("--goal-offsets must be positive multiples of frameskip")
    if not (0.0 <= float(args.generated_subgoal_ratio) <= 1.0):
        raise ValueError("--generated-subgoal-ratio must be in [0,1]")
    if args.generated_subgoal_ratio > 0 and not args.subgoal_generator_checkpoint:
        raise ValueError("--generated-subgoal-ratio requires --subgoal-generator-checkpoint")
    if args.generated_subgoal_ratio > 0:
        args.eval_use_generated_subgoal = True
    if args.prior_goal_source == "far" and args.subgoal_generator_checkpoint:
        raise ValueError(
            "--prior-goal-source far cannot be combined with "
            "--subgoal-generator-checkpoint"
        )
    if args.prior_goal_source == "far" and args.eval_use_generated_subgoal:
        raise ValueError(
            "--prior-goal-source far cannot be combined with "
            "--eval-use-generated-subgoal"
        )
    return args


def build_rows(dataset, specs, args, *, split_name: str):
    if args.dense_joint_sampling:
        limit = int(args.max_train_examples if split_name == "train" else args.max_val_examples)
        spec_indices, goal_offsets, action_offsets, diagnostics = sample_dense_unique_pairs(
            dataset,
            specs,
            history_len=args.context_len,
            frameskip=args.frameskip,
            goal_offsets=list(args.goal_offsets),
            action_offsets=list(args.action_offsets),
            limit=limit,
            seed=args.seed + (3101 if split_name == "train" else 4101),
            allow_repeats=args.dense_allow_repeats,
            balance_by_goal=args.dense_balance_goals,
        )
        print(f"dense-{split_name} sampler {json.dumps(diagnostics, sort_keys=True)}", flush=True)
        rows = [
            {
                "spec_index": int(spec_index),
                "goal_offset": int(goal_offset),
                "action_offset": int(action_offset),
                "action_tokens": int(action_offset) // int(args.frameskip),
            }
            for spec_index, goal_offset, action_offset in zip(spec_indices, goal_offsets, action_offsets)
        ]
        return rows, len(rows)
    rows = []
    max_delta = max(args.goal_offsets)
    max_tau = max(args.action_offsets)
    for spec_index, spec in enumerate(specs):
        final = int(dataset.lengths[int(spec.local_episode)]) - 1
        current = int(spec.start) + (int(args.context_len) - 1) * int(args.frameskip)
        if current + max(max_delta, max_tau) > final:
            continue
        for delta in args.goal_offsets:
            if current + int(delta) > final:
                continue
            for tau in args.action_offsets:
                if int(tau) > int(delta):
                    continue
                if current + int(tau) <= final:
                    rows.append(
                        {
                            "spec_index": spec_index,
                            "goal_offset": int(delta),
                            "action_offset": int(tau),
                            "action_tokens": int(tau) // int(args.frameskip),
                        }
                    )
    total = len(rows)
    if total <= 0:
        raise ValueError(f"No valid {split_name} variable-action rows")
    limit = int(args.max_train_examples if split_name == "train" else args.max_val_examples)
    if limit and limit < total:
        rng = np.random.default_rng(int(args.seed) + (3101 if split_name == "train" else 4101))
        keep = np.sort(rng.choice(total, size=limit, replace=False))
        rows = [rows[int(i)] for i in keep]
    return rows, total


def _cached_latents(frame_cache: FrameLatentCache, indices, device):
    latents = frame_cache.get(indices, device=device, dtype=torch.float32)
    if latents.ndim == 4 and latents.size(-2) == 1:
        latents = latents.squeeze(-2)
    return latents


def prepare_batch(
    batch,
    lewm,
    stats,
    args,
    device,
    *,
    subgoal_generator=None,
    subgoal_stats=None,
    generated_ratio: float = 0.0,
    frame_cache: FrameLatentCache | None = None,
):
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
        far_goal_pixels = image_batch_to_lewm(
            batch["far_goal_pixels"].to(device, non_blocking=True),
            args.image_size,
        ).to(dtype=dtype)
        with torch.no_grad():
            history_latents = encode_lewm_context(lewm, history_pixels)
            goal_latents = encode_lewm_context(lewm, goal_pixels)
            far_goal_latents = encode_lewm_context(lewm, far_goal_pixels)
    else:
        history_latents = _cached_latents(frame_cache, batch["history_frame_indices"], device)
        goal_latents = _cached_latents(frame_cache, batch["goal_frame_index"], device)
        far_goal_latents = _cached_latents(frame_cache, batch["far_goal_frame_index"], device)
    if args.prior_goal_source == "far":
        # Keep the architecture and row distribution identical to the
        # hierarchical prior while removing its local-subgoal interface.
        goal_latents = far_goal_latents
    if args.lowdim_keys:
        lowdim = lowdim_from_batch(batch, args.context_len, args.lowdim_keys).to(device, non_blocking=True)
        lowdim_n = normalize_lowdim(lowdim, stats)
    else:
        lowdim = torch.empty(history_latents.size(0), 0, device=device)
        lowdim_n = lowdim
    goal_offsets = batch["goal_offset"].to(device).float()
    action_offsets = batch["action_offset"].to(device).float()
    if subgoal_generator is not None and float(generated_ratio) > 0.0:
        gen_stats = subgoal_stats if subgoal_stats is not None else stats
        lowdim_gen = normalize_lowdim(lowdim, gen_stats)
        with torch.no_grad():
            generated = subgoal_generator(
                history_latents,
                far_goal_latents,
                lowdim_gen,
                goal_offsets,
                action_offsets,
            )["prediction"].float()
        if float(generated_ratio) >= 1.0:
            goal_latents = generated
        else:
            mask = (torch.rand(goal_latents.size(0), device=device) < float(generated_ratio)).view(-1, 1, 1)
            goal_latents = torch.where(mask, generated, goal_latents)
    target_full = target_action_chunk(
        batch,
        args.context_len,
        max(args.action_offsets) // int(args.frameskip),
    ).to(device, non_blocking=True)
    target_n = normalize_action_blocks(target_full, stats)
    tokens = batch["action_tokens"].to(device).long()
    return history_latents, goal_latents, far_goal_latents, lowdim_n, target_full, target_n, tokens, goal_offsets, action_offsets


def grouped_loss(model, hist, goal, far_goal, lowdim, target_n, tokens, goal_offsets, action_offsets, mode_l1_weight):
    total_loss = target_n.new_tensor(0.0)
    total_count = 0
    metrics = {}
    for length in torch.unique(tokens).tolist():
        mask = tokens == int(length)
        count = int(mask.sum().item())
        if count <= 0:
            continue
        out = model(
            hist[mask],
            goal[mask],
            lowdim[mask],
            action_horizon=int(length),
            far_goal_latents=far_goal[mask],
            goal_offset_steps=goal_offsets[mask],
            subgoal_offset_steps=action_offsets[mask],
        )
        target = target_n[mask, : int(length)]
        nll = model.nll(out, target)
        best_l1 = model.best_mode_l1(out, target)
        loss = nll + float(mode_l1_weight) * best_l1
        total_loss = total_loss + loss * count
        total_count += count
        update(metrics, {"loss": loss.item(), "nll": nll.item(), "best_mode_l1_norm": best_l1.item()}, count)
    return total_loss / max(total_count, 1), finalize(metrics)


@torch.no_grad()
def prior_eval(
    model,
    lewm,
    loader,
    stats,
    args,
    device,
    *,
    subgoal_generator=None,
    subgoal_stats=None,
    frame_cache: FrameLatentCache | None = None,
):
    model.eval()
    total = {}
    by_tau = {}
    rng = torch.Generator(device=device).manual_seed(int(args.seed) + 99)
    eval_ratio = 1.0 if (args.eval_use_generated_subgoal and subgoal_generator is not None) else 0.0
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        hist, goal, far_goal, lowdim, target, target_n, tokens, goal_offsets, action_offsets = prepare_batch(
            batch,
            lewm,
            stats,
            args,
            device,
            subgoal_generator=subgoal_generator,
            subgoal_stats=subgoal_stats,
            generated_ratio=eval_ratio,
            frame_cache=frame_cache,
        )
        valid = finite_training_mask(lowdim, target)
        if not bool(valid.any()):
            continue
        for length in torch.unique(tokens[valid]).tolist():
            mask = (tokens == int(length)) & valid
            count = int(mask.sum().item())
            if count <= 0:
                continue
            out = model(
                hist[mask],
                goal[mask],
                lowdim[mask],
                action_horizon=int(length),
                far_goal_latents=far_goal[mask],
                goal_offset_steps=goal_offsets[mask],
                subgoal_offset_steps=action_offsets[mask],
            )
            tgt_n = target_n[mask, : int(length)]
            tgt = target[mask, : int(length)]
            nll = model.nll(out, tgt_n)
            top = out["logits"].argmax(dim=-1)
            gather = top[:, None, None, None].expand(-1, 1, int(length), model.action_dim)
            top_n = out["means"][:, :, : int(length)].gather(1, gather).squeeze(1)
            top_raw = denormalize_action_blocks(top_n, stats)
            top_l1 = torch.abs(top_raw - tgt).mean(dim=(-1, -2))
            samples_n = model.sample(
                hist[mask],
                goal[mask],
                lowdim[mask],
                args.eval_samples,
                generator=rng,
                action_horizon=int(length),
                far_goal_latents=far_goal[mask],
                goal_offset_steps=goal_offsets[mask],
                subgoal_offset_steps=action_offsets[mask],
            )
            samples = denormalize_action_blocks(samples_n, stats)
            sample_l1 = torch.abs(samples - tgt.unsqueeze(1)).mean(dim=(-1, -2))
            best = sample_l1.min(dim=1).values
            values = {
                "nll": nll.item(),
                "top_l1": top_l1.mean().item(),
                f"best{args.eval_samples}_l1": best.mean().item(),
                "coverage": (best <= args.coverage_threshold).float().mean().item(),
            }
            update(total, values, count)
            update(by_tau.setdefault(f"tau{int(length) * int(args.frameskip)}", {}), values, count)
    out = {"all": finalize(total)}
    out.update({key: finalize(value) for key, value in by_tau.items()})
    return out


def main() -> None:
    args = parse_args()
    dataset_name = str(args.dataset).lower()
    if str(args.policy).lower().startswith("pusht/") or "pusht" in dataset_name:
        raise ValueError(
            "PushT variable priors must use scripts/pusht/train_pusht_variable_action_prior.py. "
            "This generic Cube/OGBench trainer filters windows by the maximum far-goal offset "
            "and therefore changes the PushT state distribution."
        )
    # Reuse stats helper flags from train_action_prior.
    args.tworoom_local_goal = False
    args.cube_local_goal = False
    args.goal_lowdim_keys = []
    args.use_goal_latent = False
    args.current_latent_only = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split = load_json(args.split)
    max_tokens = max(args.action_offsets) // int(args.frameskip)
    max_delta_frames = max(args.goal_offsets) // int(args.frameskip)
    num_steps = int(args.context_len) + max(max_tokens, max_delta_frames)
    keys_to_load = ["pixels", "action", *args.lowdim_keys]
    dataset = load_swm_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        frameskip=args.frameskip,
        num_steps=num_steps,
        keys_to_load=list(dict.fromkeys(keys_to_load)),
    )
    planning_horizon = max_tokens if args.dense_joint_sampling else max(max_tokens, max_delta_frames)
    train_pool = build_window_specs(dataset, split, "train", context_len=args.context_len, plan_horizon=planning_horizon)
    val_pool = build_window_specs(dataset, split, "val", context_len=args.context_len, plan_horizon=planning_horizon)
    train_specs = subset_specs(train_pool, args.max_train_windows, args.seed)
    val_specs = subset_specs(val_pool, args.max_val_windows, args.seed + 1)
    train_rows, train_total = build_rows(dataset, train_specs, args, split_name="train")
    val_rows, val_total = build_rows(dataset, val_specs, args, split_name="val")
    print(
        f"variable-action rows train={len(train_rows)}/{train_total} "
        f"val={len(val_rows)}/{val_total} action_offsets={args.action_offsets} "
        f"goal_offsets={args.goal_offsets} max_tokens={max_tokens}",
        flush=True,
    )

    if args.lowdim_keys:
        stats_cpu = compute_stats(dataset, split, args.lowdim_keys, train_specs=train_specs, args=args)
    else:
        stats_cpu = {
            "action_mean": torch.from_numpy(np.asarray(dataset.get_col_data("action"), dtype=np.float32).mean(axis=0).astype(np.float32)),
            "action_std": torch.from_numpy(np.maximum(np.asarray(dataset.get_col_data("action"), dtype=np.float32).std(axis=0), 1e-6).astype(np.float32)),
            "lowdim_mean": torch.empty(0),
            "lowdim_std": torch.empty(0),
            "lowdim_keys": [],
            "goal_lowdim_keys": [],
        }
    stats = move_stats(stats_cpu, device)

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
    subgoal_generator = None
    subgoal_stats = None
    subgoal_checkpoint = None
    if args.subgoal_generator_checkpoint:
        subgoal_generator, subgoal_stats, subgoal_checkpoint = load_subgoal_prior(args.subgoal_generator_checkpoint, device)
        subgoal_generator.requires_grad_(False)
        subgoal_generator.eval()
        print(
            f"loaded subgoal generator={args.subgoal_generator_checkpoint} "
            f"generated_ratio={args.generated_subgoal_ratio} eval_generated={args.eval_use_generated_subgoal}",
            flush=True,
        )

    train_loader = DataLoader(
        VariableActionDataset(
            dataset,
            train_specs,
            train_rows,
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
        VariableActionDataset(
            dataset,
            val_specs,
            val_rows,
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
    # Keep the full spec table here: rows store their original spec_index.
    # Slicing specs without remapping row indices makes this single-batch
    # shape probe fail for sampled rows whose spec_index is not zero.
    first = next(iter(DataLoader(
        VariableActionDataset(
            dataset,
            train_specs,
            train_rows[:1],
            context_len=args.context_len,
            frameskip=args.frameskip,
            episode_base=episode_base,
        ),
        batch_size=1,
    )))
    hist, goal, far_goal, lowdim, target, target_n, tokens, goal_offsets, action_offsets = prepare_batch(
        first,
        lewm,
        stats,
        args,
        device,
        generated_ratio=0.0,
        frame_cache=frame_cache,
    )
    model_config = {
        "architecture": "variable_transformer",
        "latent_dim": int(hist.size(-1)),
        "lowdim_dim": int(stats_cpu["lowdim_mean"].numel()),
        "action_dim": int(stats_cpu["action_mean"].numel() * int(args.frameskip)),
        "max_plan_horizon": int(max_tokens),
        "hidden_dim": int(args.hidden_dim),
        "num_heads": int(args.num_heads),
        "depth": int(args.depth),
        "num_modes": int(args.num_modes),
        "max_goal_offset": int(max(args.goal_offsets)),
    }
    model_kwargs = dict(model_config)
    model_kwargs.pop("architecture")
    model = PushtVariableTransformerGoalPrior(**model_kwargs).to(device)
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
        best_val = float(ckpt.get("best_val_nll", best_val))
        print(f"resuming epoch={start_epoch} best_val={best_val:.4f}", flush=True)

    manifest = {
        "script": "scripts/lewm_prior/train_variable_action_prior.py",
        "dataset": args.dataset,
        "split": args.split,
        "policy": args.policy,
        "model_config": model_config,
        "args": vars(args),
        "subgoal_generator_checkpoint": args.subgoal_generator_checkpoint,
        "subgoal_generator_model_config": None if subgoal_checkpoint is None else subgoal_checkpoint.get("model_config"),
        "selected_train_windows": len(train_specs),
        "selected_val_windows": len(val_specs),
        "selected_train_rows": len(train_rows),
        "selected_val_rows": len(val_rows),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        model.train()
        total = {}
        skipped = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            hist, goal, far_goal, lowdim, target, target_n, tokens, goal_offsets, action_offsets = prepare_batch(
                batch,
                lewm,
                stats,
                args,
                device,
                subgoal_generator=subgoal_generator,
                subgoal_stats=subgoal_stats,
                generated_ratio=args.generated_subgoal_ratio,
                frame_cache=frame_cache,
            )
            valid = finite_training_mask(lowdim, target)
            if not bool(valid.any()):
                skipped += int(target.size(0))
                continue
            if not bool(valid.all()):
                skipped += int((~valid).sum().item())
                hist, goal, far_goal = hist[valid], goal[valid], far_goal[valid]
                lowdim, target_n = lowdim[valid], target_n[valid]
                tokens, goal_offsets, action_offsets = tokens[valid], goal_offsets[valid], action_offsets[valid]
            loss, parts = grouped_loss(
                model,
                hist,
                goal,
                far_goal,
                lowdim,
                target_n,
                tokens,
                goal_offsets,
                action_offsets,
                args.mode_l1_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            update(total, parts, int(tokens.numel()))
            if batch_index % 20 == 0:
                row = finalize(total)
                print(
                    f"epoch={epoch} batch={batch_index} loss={row.get('loss', 0):.4f} "
                    f"nll={row.get('nll', 0):.4f} best_mode_l1_norm={row.get('best_mode_l1_norm', 0):.4f}",
                    flush=True,
                )
        train_metrics = finalize(total)
        train_metrics["skipped_nonfinite"] = skipped
        val_metrics = prior_eval(
            model,
            lewm,
            val_loader,
            stats,
            args,
            device,
            subgoal_generator=subgoal_generator,
            subgoal_stats=subgoal_stats,
            frame_cache=frame_cache,
        )
        val_nll = float(val_metrics.get("all", {}).get("nll", float("inf")))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, sort_keys=True) + "\n")
        all_val = val_metrics.get("all", {})
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_metrics.get('loss', float('nan')):.4f} "
            f"val_nll={all_val.get('nll', float('nan')):.4f} "
            f"val_top_l1={all_val.get('top_l1', float('nan')):.4f} "
            f"val_best{args.eval_samples}_l1={all_val.get(f'best{args.eval_samples}_l1', float('nan')):.4f} "
            f"val_cov={all_val.get('coverage', float('nan')):.4f}",
            flush=True,
        )
        best_val = min(best_val, val_nll)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": stats_cpu,
            "model_config": model_config,
            "epoch": epoch,
            "best_val_nll": best_val,
            "last_train_metrics": train_metrics,
            "last_val_metrics": val_metrics,
            "run_manifest": manifest,
        }
        save_bc_checkpoint(latest_path, payload)
        print(f"wrote latest checkpoint: {latest_path}", flush=True)
        if val_nll <= best_val + 1.0e-12:
            save_bc_checkpoint(best_path, payload)
            print(f"wrote new best checkpoint: {best_path}", flush=True)
        if int(epoch) in set(args.save_epochs):
            snapshot_path = out_dir / f"epoch_{epoch:03d}.pt"
            save_bc_checkpoint(snapshot_path, payload)
            print(f"wrote epoch snapshot: {snapshot_path}", flush=True)
    print(f"done best_val_nll={best_val:.4f} wrote={latest_path}", flush=True)


if __name__ == "__main__":
    main()
