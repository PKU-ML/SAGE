"""Dataset, normalization, and LeWM helpers used by SAGE."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def phase_id_from_fraction(frac: float, num_phases: int = 5) -> int:
    frac = min(max(float(frac), 0.0), 0.999999)
    return int(frac * int(num_phases))


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_swm_dataset(
    dataset_name: str,
    *,
    frameskip: int,
    num_steps: int,
    cache_dir: str | None = None,
    keys_to_load: list[str] | None = None,
):
    import stable_worldmodel as swm

    return swm.data.load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        frameskip=int(frameskip),
        num_steps=int(num_steps),
        keys_to_load=keys_to_load or ["pixels", "action", "proprio", "state"],
    )


def dataset_schema_names(dataset) -> set[str]:
    names = set(getattr(dataset, "_schema_names", getattr(dataset, "column_names", [])))
    h5_path = getattr(dataset, "h5_path", None)
    if h5_path is not None:
        try:
            import h5py

            with h5py.File(str(h5_path), "r") as f:
                names.update(map(str, f.keys()))
        except Exception:
            pass
    return names


def resolve_dataset_col(dataset, candidates: Iterable[str]) -> str:
    names = dataset_schema_names(dataset)
    for col in candidates:
        if col in names:
            return col
    raise KeyError(f"None of columns {list(candidates)} found in dataset schema: {sorted(names)}")


def episode_column(dataset) -> str:
    return resolve_dataset_col(dataset, ["episode_idx", "ep_idx"])


def step_column(dataset) -> str:
    return resolve_dataset_col(dataset, ["step_idx", "step"])


def episode_ids(dataset) -> np.ndarray:
    ep_col = dataset.get_col_data(episode_column(dataset))
    ids = []
    for off in dataset.offsets:
        ids.append(int(ep_col[int(off)]))
    return np.asarray(ids, dtype=np.int64)


def split_episode_sets(split: dict) -> dict[str, set[int]]:
    """Return train/val/test episode sets from either legacy or compact split JSON.

    PushT split files use train_episode_idx / val_episode_idx / test_episode_idx.
    Newer OGBench-style split files use train / val / test.  Supporting both keeps
    the no-overlap checks and downstream window builders shared across datasets.
    """

    def values(name: str):
        legacy_key = f"{name}_episode_idx"
        if legacy_key in split:
            return split[legacy_key]
        if name in split:
            return split[name]
        raise KeyError(f"split must contain '{legacy_key}' or '{name}'")

    return {
        "train": set(map(int, values("train"))),
        "val": set(map(int, values("val"))),
        "test": set(map(int, values("test"))),
    }


def split_for_episode(ep_id: int, split: dict) -> str | None:
    ep_id = int(ep_id)
    for name, values in split_episode_sets(split).items():
        if ep_id in values:
            return name
    return None


def assert_no_split_overlap(split: dict) -> None:
    sets = split_episode_sets(split)
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"Episode split overlap between {a} and {b}: {sorted(overlap)[:10]}")


def image_batch_to_lewm(pixels: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Convert raw Lance pixels [B,T,C,H,W] uint8 to LeWM normalized float."""

    if pixels.ndim != 5:
        raise ValueError(f"Expected pixels [B,T,C,H,W], got {tuple(pixels.shape)}")
    x = pixels.float()
    if x.max() > 2.0:
        x = x / 255.0
    b, t, c, h, w = x.shape
    x = F.interpolate(
        x.reshape(b * t, c, h, w),
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    ).reshape(b, t, c, int(image_size), int(image_size))
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def load_lewm(policy: str, device: torch.device, bf16: bool = False):
    import stable_worldmodel as swm

    policy_path = Path(policy).expanduser()
    try:
        if policy_path.is_file() and policy_path.name.endswith("_object.ckpt"):
            run_prefix = str(policy_path)[: -len("_object.ckpt")]
            model = swm.policy.AutoCostModel(run_prefix)
        else:
            model = swm.policy.AutoCostModel(policy)
    except (AssertionError, FileNotFoundError, RuntimeError):
        model = swm.wm.utils.load_pretrained(policy)
    model = model.to(device)
    if bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_lewm_context(lewm: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    info = {"pixels": pixels}
    try:
        # PreJEPA / DINO-WM checkpoints may have action or proprio encoders in
        # ``extra_encoders``. For action-prior inputs we want the visual latent
        # only; low-dimensional state is provided through the prior's own MLP.
        out = lewm.encode(info, emb_keys=[])
    except TypeError:
        out = lewm.encode(info)
    emb = out["emb"].float()
    if emb.ndim == 4:
        emb = emb.mean(dim=2)
    return emb


@torch.no_grad()
def rollout_lewm(
    lewm: nn.Module,
    context_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    action_candidates: torch.Tensor,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return LeWM final goal cost and predicted embeddings for candidates.

    Args:
        context_pixels: [B,H,C,224,224] normalized.
        goal_pixels: [B,1,C,224,224] normalized.
        action_candidates: [B,K,T,A] normalized action blocks.
    """

    b, k, _, _ = action_candidates.shape
    info = {
        "pixels": context_pixels[:, None].expand(b, k, *context_pixels.shape[1:]).contiguous(),
        "goal": goal_pixels[:, None].expand(b, k, *goal_pixels.shape[1:]).contiguous(),
        "action": torch.zeros(
            b,
            k,
            context_pixels.size(1),
            action_dim,
            device=action_candidates.device,
            dtype=action_candidates.dtype,
        ),
    }
    costs = lewm.get_cost(info, action_candidates)
    pred = info["predicted_emb"].float()
    return costs.float(), pred


def tensor_stats_from_rows(values: np.ndarray, row_mask: np.ndarray) -> dict[str, torch.Tensor]:
    selected = np.asarray(values[row_mask], dtype=np.float32)
    mean = torch.from_numpy(selected.mean(axis=0).astype(np.float32))
    std = torch.from_numpy(selected.std(axis=0).astype(np.float32))
    std = torch.clamp(std, min=1.0e-6)
    return {"mean": mean, "std": std}


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.to(x.device, x.dtype) + mean.to(x.device, x.dtype)


def normalize_action_blocks(actions: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = stats["action_mean"].to(actions.device, actions.dtype)
    std = stats["action_std"].to(actions.device, actions.dtype)
    return normalize(actions.reshape(*actions.shape[:-1], -1, mean.numel()), mean, std).reshape_as(actions)


def denormalize_action_blocks(actions: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = stats["action_mean"].to(actions.device, actions.dtype)
    std = stats["action_std"].to(actions.device, actions.dtype)
    return denormalize(actions.reshape(*actions.shape[:-1], -1, mean.numel()), mean, std).reshape_as(actions)


def normalize_lowdim(lowdim: torch.Tensor, stats: dict) -> torch.Tensor:
    if lowdim.numel() == 0:
        return lowdim
    return normalize(lowdim, stats["lowdim_mean"], stats["lowdim_std"])


class PushtBCPrior(nn.Module):
    """Fixed-duration trajectory GMM used by compatibility baselines."""

    def __init__(
        self,
        latent_dim: int,
        lowdim_dim: int,
        action_dim: int = 10,
        plan_horizon: int = 5,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_modes: int = 8,
        trunk_depth: int = 2,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.lowdim_dim = int(lowdim_dim)
        self.action_dim = int(action_dim)
        self.plan_horizon = int(plan_horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_modes = int(num_modes)
        self.trunk_depth = int(trunk_depth)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pool = nn.MultiheadAttention(self.hidden_dim, num_heads=num_heads, batch_first=True)
        self.lowdim_encoder = nn.Sequential(
            nn.LayerNorm(self.lowdim_dim),
            nn.Linear(self.lowdim_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        trunk_layers: list[nn.Module] = [
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(max(self.trunk_depth - 1, 0)):
            trunk_layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                ]
            )
        self.trunk = nn.Sequential(*trunk_layers)
        out_dim = self.num_modes * self.plan_horizon * self.action_dim
        self.logit_head = nn.Linear(self.hidden_dim, self.num_modes)
        self.mean_head = nn.Linear(self.hidden_dim, out_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, out_dim)

        nn.init.trunc_normal_(self.pool_query, std=0.02)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def forward(self, latents: torch.Tensor, lowdim: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.latent_proj(self.latent_norm(latents))
        query = self.pool_query.expand(tokens.size(0), -1, -1)
        pooled, _ = self.pool(query, tokens, tokens, need_weights=False)
        pooled = pooled.squeeze(1)
        lowdim_feat = self.lowdim_encoder(lowdim)
        hidden = self.trunk(torch.cat([pooled, lowdim_feat], dim=-1))
        logits = self.logit_head(hidden)
        means = self.mean_head(hidden).view(-1, self.num_modes, self.plan_horizon, self.action_dim)
        log_stds = self.log_std_head(hidden).view_as(means).clamp(self.min_log_std, self.max_log_std)
        return {"logits": logits, "means": means, "log_stds": log_stds}

    def component_log_probs(self, outputs: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        means = outputs["means"]
        log_stds = outputs["log_stds"]
        target = actions.unsqueeze(1)
        inv_std = torch.exp(-log_stds)
        log_prob = -0.5 * ((target - means) * inv_std).pow(2) - log_stds
        log_prob = log_prob - 0.5 * math.log(2.0 * math.pi)
        return log_prob.sum(dim=(-1, -2))

    def nll(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        comp = self.component_log_probs(outputs, target)
        log_mix = F.log_softmax(outputs["logits"], dim=-1)
        return -torch.logsumexp(log_mix + comp, dim=-1).mean()

    def log_prob(self, latents: torch.Tensor, lowdim: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        outputs = self(latents, lowdim)
        b, n = actions.shape[:2]
        means = outputs["means"][:, None]
        log_stds = outputs["log_stds"][:, None]
        target = actions[:, :, None]
        inv_std = torch.exp(-log_stds)
        comp = -0.5 * ((target - means) * inv_std).pow(2) - log_stds
        comp = comp - 0.5 * math.log(2.0 * math.pi)
        comp = comp.sum(dim=(-1, -2))
        log_mix = F.log_softmax(outputs["logits"], dim=-1)[:, None]
        return torch.logsumexp(log_mix + comp, dim=-1).reshape(b, n)

    def best_mode_l1(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        per_mode = torch.abs(outputs["means"] - target.unsqueeze(1)).mean(dim=(-1, -2))
        return per_mode.min(dim=1).values.mean()

    @torch.no_grad()
    def sample(
        self,
        latents: torch.Tensor,
        lowdim: torch.Tensor,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        outputs = self(latents, lowdim)
        probs = F.softmax(outputs["logits"], dim=-1)
        mode_ids = torch.multinomial(
            probs,
            num_samples=int(num_samples),
            replacement=True,
            generator=generator,
        )
        gather_index = mode_ids[:, :, None, None].expand(-1, -1, self.plan_horizon, self.action_dim)
        means = outputs["means"].gather(1, gather_index)
        log_stds = outputs["log_stds"].gather(1, gather_index)
        noise = torch.randn(
            means.shape,
            dtype=means.dtype,
            device=means.device,
            generator=generator,
        )
        return means + noise * torch.exp(log_stds)

    @torch.no_grad()
    def top_mode(self, latents: torch.Tensor, lowdim: torch.Tensor) -> torch.Tensor:
        outputs = self(latents, lowdim)
        top = outputs["logits"].argmax(dim=-1)
        gather_index = top[:, None, None, None].expand(-1, 1, self.plan_horizon, self.action_dim)
        return outputs["means"].gather(1, gather_index).squeeze(1)


def save_bc_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_bc_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_config = dict(ckpt["model_config"])
    architecture = model_config.pop("architecture", "bc_prior")
    if architecture == "variable_transformer":
        from sage.models.action_prior import PushtVariableTransformerGoalPrior

        model_config.pop("pooling", None)
        model = PushtVariableTransformerGoalPrior(**model_config)
    else:
        model = PushtBCPrior(**model_config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    stats = {}
    for key, value in ckpt["stats"].items():
        stats[key] = value.to(device) if torch.is_tensor(value) else value
    return model, stats, ckpt


@dataclass(frozen=True)
class WindowSpec:
    dataset_index: int
    local_episode: int
    episode_id: int
    start: int
    phase_id: int


def build_window_specs(
    dataset,
    split: dict,
    split_name: str,
    *,
    context_len: int,
    plan_horizon: int,
    num_phases: int = 5,
) -> list[WindowSpec]:
    assert_no_split_overlap(split)
    ep_ids = episode_ids(dataset)
    allowed = split_episode_sets(split)[split_name]
    specs: list[WindowSpec] = []
    for idx, (local_ep, start) in enumerate(dataset.clip_indices):
        ep_id = int(ep_ids[int(local_ep)])
        if ep_id not in allowed:
            continue
        length = int(dataset.lengths[int(local_ep)])
        # The target reaches roughly start + (context_len - 1 + plan_horizon) * frameskip.
        denom = max(length - dataset.span, 1)
        frac = float(start) / float(denom)
        specs.append(
            WindowSpec(
                dataset_index=idx,
                local_episode=int(local_ep),
                episode_id=ep_id,
                start=int(start),
                phase_id=phase_id_from_fraction(frac, num_phases=num_phases),
            )
        )
    return specs


def lowdim_from_batch(batch: dict, context_len: int, keys: Iterable[str]) -> torch.Tensor:
    batch_size = None
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            batch_size = int(value.shape[0])
            break
    if batch_size is None:
        raise KeyError("Could not infer batch size for lowdim extraction")
    if not list(keys):
        return torch.empty(batch_size, 0, dtype=torch.float32)
    parts = []
    for key in keys:
        if key not in batch:
            continue
        value = batch[key]
        if value.ndim < 3:
            raise ValueError(f"Expected {key} [B,T,D], got {tuple(value.shape)}")
        parts.append(value[:, int(context_len) - 1].float())
    if not parts:
        raise KeyError(f"None of lowdim keys {list(keys)} found in batch")
    return torch.cat(parts, dim=-1)


def target_action_chunk(batch: dict, context_len: int, plan_horizon: int) -> torch.Tensor:
    actions = batch["action"].float()
    start = int(context_len) - 1
    end = start + int(plan_horizon)
    if actions.size(1) < end:
        raise ValueError(f"Need action length >= {end}, got {actions.size(1)}")
    return actions[:, start:end]
