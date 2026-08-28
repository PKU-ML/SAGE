"""Canonical PushT component evaluation for the SAGE paper."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from torchvision.transforms import v2 as transforms

from sage.models.action_prior import load_action_prior
from sage.models.subgoal import load_subgoal_prior
from sage.provenance import sha256_file, verify_manifest
from sage.runtime.lewm import (
    image_batch_to_lewm,
    load_json,
    load_lewm,
    normalize_lowdim,
)
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy


METHODS = (
    "base_cem",
    "far_goal_prior_cem",
    "lewm_generator",
    "generator_prior_top",
    "sage",
)

METHOD_DESCRIPTIONS = {
    "base_cem": "zero-mean Gaussian CEM scored against the final goal",
    "far_goal_prior_cem": "far-goal action-prior proposals refined by LeWM CEM",
    "lewm_generator": "zero-mean Gaussian CEM scored against generated subgoals",
    "generator_prior_top": "generated subgoals with the prior top mode; no LeWM ranking",
    "sage": "generated subgoals and action-prior proposals refined by LeWM CEM",
}


class ArrayNormalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1.0e-6)

    def transform(self, value):
        return (value - self.mean) / self.std

    def inverse_transform(self, value):
        return value * self.std + self.mean


def image_transform(image_size: int, dtype: torch.dtype):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=int(image_size)),
        ]
    )


def tensor_last(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value[:, -1] if value.ndim >= 3 else value


def expand_for_candidates(info: dict, count: int, device, dtype) -> dict:
    expanded = {}
    for key, value in info.items():
        if key.startswith("_proposal_"):
            continue
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            value = value.to(device=device, dtype=target_dtype)
            expanded[key] = value[:, None].expand(
                value.size(0), int(count), *value.shape[1:]
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:, None], int(count), axis=1)
        else:
            expanded[key] = value
    return expanded


class SAGECostModel(torch.nn.Module):
    """Connect the two SAGE proposal networks to a frozen LeWM cost model."""

    def __init__(
        self,
        lewm,
        generator,
        generator_stats,
        action_prior,
        action_stats,
        *,
        goal_offset_steps: int,
        action_block: int,
        image_size: int,
    ):
        super().__init__()
        self.lewm = lewm
        self.generator = generator
        self.generator_stats = generator_stats
        self.action_prior = action_prior
        self.action_stats = action_stats
        self.goal_offset_steps = int(goal_offset_steps)
        self.action_block = int(action_block)
        self.image_size = int(image_size)
        self.option_duration_steps = 25
        self._subgoal_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._generated = 0
        self._used_final_goal = 0

    @property
    def action_dim(self) -> int:
        if self.action_prior is None:
            raise RuntimeError("This method does not use an action prior")
        return int(self.action_prior.action_dim)

    @staticmethod
    def _batch_ids(info: dict, key: str) -> list[int]:
        if key not in info:
            raise KeyError(f"Missing planner metadata {key}")
        value = info[key]
        if torch.is_tensor(value):
            while value.ndim > 1:
                value = value[:, 0]
            return value.detach().cpu().long().tolist()
        value = np.asarray(value)
        while value.ndim > 1:
            value = value[:, 0]
        return value.astype(np.int64).tolist()

    def _normalized_lowdim(self, info: dict, stats: dict, batch: int, device):
        parts = []
        for key in stats.get("lowdim_keys", ["state", "proprio"]):
            if key in info:
                value = info[key]
                if value.ndim >= 4:
                    value = value[:, 0]
                parts.append(tensor_last(value).to(device=device, dtype=torch.float32))
        if parts:
            lowdim = torch.cat(parts, dim=-1)
        else:
            lowdim = torch.zeros(
                batch,
                int(stats["lowdim_mean"].numel()),
                device=device,
                dtype=torch.float32,
            )
        return normalize_lowdim(lowdim, stats)

    def _final_goal_latents(self, info: dict) -> torch.Tensor:
        goal = info["goal"]
        if goal.ndim == 6:
            goal = goal[:, 0]
        goal = goal.to(
            device=next(self.lewm.parameters()).device,
            dtype=next(self.lewm.parameters()).dtype,
        )
        return self.lewm.encode({"pixels": goal})["emb"].float()

    def _history_latents(self, info: dict) -> torch.Tensor:
        pixels = info.get("_proposal_pixels_raw")
        if pixels is None:
            raise KeyError("SAGE requires the raw three-frame proposal history")
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        if pixels.ndim == 6:
            pixels = pixels[:, 0]
        if pixels.shape[-1] in {1, 3, 4}:
            pixels = pixels.permute(0, 1, 4, 2, 3)
        pixels = image_batch_to_lewm(pixels, self.image_size).to(
            device=next(self.lewm.parameters()).device,
            dtype=next(self.lewm.parameters()).dtype
        )
        return self.lewm.encode({"pixels": pixels})["emb"].float()

    @staticmethod
    def _step_vector(value, default: int, batch: int, device):
        if value is None:
            return torch.full(
                (batch,), float(default), device=device, dtype=torch.float32
            )
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=torch.float32)
            while tensor.ndim > 1:
                tensor = tensor[:, 0]
            tensor = tensor.reshape(-1)
        else:
            array = np.asarray(value, dtype=np.float32)
            while array.ndim > 1:
                array = array[:, 0]
            tensor = torch.as_tensor(array, device=device).reshape(-1)
        return tensor.expand(batch) if tensor.numel() == 1 else tensor

    def _offsets(self, info: dict, batch: int, device, action_horizon: int):
        duration = int(action_horizon) * self.action_block
        remaining = self._step_vector(
            info.get("_remaining_steps"),
            self.goal_offset_steps,
            batch,
            device,
        )
        option = self._step_vector(
            info.get("_option_duration_steps"),
            duration,
            batch,
            device,
        )
        return torch.maximum(remaining, option), option

    @torch.no_grad()
    def _local_goal_latents(self, info: dict) -> torch.Tensor:
        if self.generator is None:
            return self._final_goal_latents(info)
        device = next(self.lewm.parameters()).device
        env_ids = self._batch_ids(info, "_env_id")
        call_ids = self._batch_ids(info, "_plan_call")
        remaining = self._step_vector(
            info.get("_remaining_steps"),
            self.goal_offset_steps,
            len(env_ids),
            device,
        ).long()
        duration = self._step_vector(
            info.get("_option_duration_steps"),
            self.option_duration_steps,
            len(env_ids),
            device,
        ).long()

        keys = [
            (int(env_id), int(call_id), int(remaining[row]), int(duration[row]))
            for row, (env_id, call_id) in enumerate(zip(env_ids, call_ids))
        ]
        missing = [row for row, key in enumerate(keys) if key not in self._subgoal_cache]
        if missing:
            final_goal = self._final_goal_latents(info)
            history = self._history_latents(info)
            lowdim = self._normalized_lowdim(
                info, self.generator_stats, history.size(0), device
            )
            for row in missing:
                key = keys[row]
                rem = int(remaining[row])
                tau = int(duration[row])
                if rem <= tau:
                    prediction = final_goal[row : row + 1]
                    self._used_final_goal += 1
                else:
                    prediction = self.generator(
                        history[row : row + 1],
                        final_goal[row : row + 1],
                        lowdim[row : row + 1],
                        torch.tensor([rem], device=device, dtype=torch.float32),
                        torch.tensor([tau], device=device, dtype=torch.float32),
                    )["prediction"]
                    self._generated += 1
                self._subgoal_cache[key] = prediction.detach()
        outputs = []
        for key in keys:
            outputs.append(self._subgoal_cache[key])
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def sample_candidates(
        self,
        info: dict,
        *,
        num_samples: int,
        action_horizon: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if self.action_prior is None:
            raise RuntimeError("Prior candidate sampling requested without a prior")
        device = next(self.lewm.parameters()).device
        history = self._history_latents(info)
        local_goal = self._local_goal_latents(info)
        far_goal = self._final_goal_latents(info)
        lowdim = self._normalized_lowdim(
            info, self.action_stats, history.size(0), device
        )
        goal_steps, option_steps = self._offsets(
            info, history.size(0), device, action_horizon
        )
        return self.action_prior.sample(
            history,
            local_goal,
            lowdim,
            int(num_samples),
            generator=generator,
            action_horizon=int(action_horizon),
            far_goal_latents=far_goal,
            goal_offset_steps=goal_steps,
            subgoal_offset_steps=option_steps,
        )

    @torch.no_grad()
    def top_candidate(self, info: dict, *, action_horizon: int) -> torch.Tensor:
        if self.action_prior is None:
            raise RuntimeError("Prior top mode requested without a prior")
        device = next(self.lewm.parameters()).device
        history = self._history_latents(info)
        local_goal = self._local_goal_latents(info)
        far_goal = self._final_goal_latents(info)
        lowdim = self._normalized_lowdim(
            info, self.action_stats, history.size(0), device
        )
        goal_steps, option_steps = self._offsets(
            info, history.size(0), device, action_horizon
        )
        return self.action_prior.top_mode(
            history,
            local_goal,
            lowdim,
            action_horizon=int(action_horizon),
            far_goal_latents=far_goal,
            goal_offset_steps=goal_steps,
            subgoal_offset_steps=option_steps,
        )

    @torch.no_grad()
    def get_cost(self, info: dict, actions: torch.Tensor) -> torch.Tensor:
        lewm_info = {
            key: value for key, value in info.items() if not key.startswith("_")
        }
        lewm_info["goal_emb"] = self._local_goal_latents(info)
        return self.lewm.get_cost(lewm_info, actions)

    def diagnostics(self) -> dict:
        return {
            "generated_subgoals": self._generated,
            "used_final_goal": self._used_final_goal,
        }


class PriorInitializedCEM:
    """Initialize CEM from SAGE proposals, then apply LeWM-ranked updates."""

    def __init__(
        self,
        model: SAGECostModel,
        *,
        candidates: int,
        rounds: int,
        elites: int,
        seed: int,
        device: torch.device,
    ):
        self.model = model
        self.candidates = int(candidates)
        self.rounds = int(rounds)
        self.elites = int(elites)
        self.seed = int(seed)
        self.device = device
        self._dtype = next(model.parameters()).dtype
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.rounds < 1:
            raise ValueError("CEM rounds must be positive")
        if not 2 <= self.elites <= self.candidates:
            raise ValueError("Require 2 <= elites <= candidates")

    def configure(self, *, action_space, n_envs, config):
        if not isinstance(action_space, Box):
            raise TypeError("SAGE requires a continuous Box action space")
        env_action_dim = int(np.prod(action_space.shape[1:]))
        expected = env_action_dim * int(config.action_block)
        if expected != self.model.action_dim:
            raise ValueError(
                f"Action prior dim {self.model.action_dim} != environment block dim {expected}"
            )
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = expected

    def __call__(self, info, init_action=None):
        return self.solve(info, init_action=init_action)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def horizon(self):
        return int(self._config.horizon)

    @staticmethod
    def _fit(candidates, costs, elites):
        values, indices = torch.topk(costs, k=elites, dim=1, largest=False)
        rows = torch.arange(candidates.size(0), device=candidates.device)[:, None]
        selected = candidates[rows, indices]
        return selected.mean(1), selected.std(1), values

    @torch.inference_mode()
    def solve(self, info: dict, init_action=None):
        del init_action
        horizon = self.horizon
        candidates = self.model.sample_candidates(
            info,
            num_samples=self.candidates,
            action_horizon=horizon,
            generator=self.generator,
        ).to(device=self.device, dtype=self._dtype)
        expanded = expand_for_candidates(
            info, self.candidates, self.device, self._dtype
        )
        costs = self.model.get_cost(expanded, candidates)
        mean, std, elite_costs = self._fit(candidates, costs, self.elites)

        for _ in range(1, self.rounds):
            candidates = torch.randn(
                mean.size(0),
                self.candidates,
                horizon,
                self._action_dim,
                generator=self.generator,
                device=self.device,
                dtype=self._dtype,
            )
            candidates = candidates * std[:, None] + mean[:, None]
            candidates[:, 0] = mean
            costs = self.model.get_cost(expanded, candidates)
            mean, std, elite_costs = self._fit(
                candidates, costs, self.elites
            )
        return {
            "actions": mean.detach().cpu(),
            "costs": elite_costs.mean(1).detach().cpu().tolist(),
        }


class GaussianCEM(PriorInitializedCEM):
    """True zero-mean Gaussian CEM; no action prior enters the solver."""

    def configure(self, *, action_space, n_envs, config):
        if not isinstance(action_space, Box):
            raise TypeError("Gaussian CEM requires a continuous Box action space")
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:])) * int(
            config.action_block
        )

    @torch.inference_mode()
    def solve(self, info: dict, init_action=None):
        if init_action is not None:
            raise ValueError("base_cem and lewm_generator forbid warm starts")
        horizon = self.horizon
        batch = len(next(iter(info.values())))
        if getattr(self.model, "generator", None) is not None:
            # Candidate expansion intentionally drops raw proposal history. Cache
            # the generated local goal once at the unexpanded planner query.
            self.model._local_goal_latents(info)
        mean = torch.zeros(
            batch,
            horizon,
            self._action_dim,
            device=self.device,
            dtype=self._dtype,
        )
        std = torch.ones_like(mean)
        expanded = expand_for_candidates(
            info, self.candidates, self.device, self._dtype
        )
        elite_costs = None
        for _ in range(self.rounds):
            candidates = torch.randn(
                batch,
                self.candidates,
                horizon,
                self._action_dim,
                generator=self.generator,
                device=self.device,
                dtype=self._dtype,
            )
            candidates = candidates * std[:, None] + mean[:, None]
            candidates[:, 0] = mean
            costs = self.model.get_cost(expanded, candidates)
            mean, std, elite_costs = self._fit(candidates, costs, self.elites)
            std = std.clamp_min(1.0e-6)
        return {
            "actions": mean.detach().cpu(),
            "costs": elite_costs.mean(1).detach().cpu().tolist(),
        }


class PriorTopMode:
    """Execute the highest-weight prior component without LeWM scoring."""

    def __init__(self, model: SAGECostModel):
        self.model = model

    def configure(self, *, action_space, n_envs, config):
        del action_space
        self._n_envs = int(n_envs)
        self._config = config

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def horizon(self):
        return int(self._config.horizon)

    @property
    def action_dim(self):
        return int(self.model.action_dim)

    def __call__(self, info, init_action=None):
        return self.solve(info, init_action=init_action)

    @torch.inference_mode()
    def solve(self, info, init_action=None):
        if init_action is not None:
            raise ValueError("generator_prior_top forbids warm starts")
        actions = self.model.top_candidate(
            info, action_horizon=self.horizon
        )
        return {"actions": actions.detach().cpu(), "costs": [float("nan")] * len(actions)}


class ScheduledPolicy(WorldModelPolicy):
    """Execute the paper schedule while preserving the trained history cadence."""

    def __init__(
        self,
        *args,
        schedule_steps: list[int],
        goal_offset_steps: int,
        history_length: int,
        frameskip: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedule = [int(value) for value in schedule_steps]
        self.goal_offset = int(goal_offset_steps)
        self.history_length = int(history_length)
        self.frameskip = int(frameskip)
        self._frames = None
        self._steps = None
        self._stage = None
        self._elapsed = None
        self._plan_call = 0

    def set_env(self, env):
        super().set_env(env)
        count = int(env.num_envs)
        self._frames = [deque(maxlen=self.history_length) for _ in range(count)]
        self._steps = np.zeros(count, dtype=np.int64)
        self._stage = np.zeros(count, dtype=np.int64)
        self._elapsed = np.zeros(count, dtype=np.int64)
        self._action_buffer = [
            deque(maxlen=max(self.schedule)) for _ in range(count)
        ]
        self._next_init = None

    def _duration(self, env_id: int) -> int:
        stage = min(int(self._stage[env_id]), len(self.schedule) - 1)
        return self.schedule[stage]

    def get_action(self, info_dict: dict, **kwargs):
        del kwargs
        payload = dict(info_dict)
        count = int(self.env.num_envs)
        raw_pixels = np.asarray(info_dict["pixels"])
        current = raw_pixels[:, -1] if raw_pixels.ndim >= 5 else raw_pixels
        flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(count, dtype=bool)),
            dtype=bool,
        )
        for env_id in range(count):
            if flush[env_id]:
                self._frames[env_id].clear()
                self._steps[env_id] = 0
                self._stage[env_id] = 0
                self._elapsed[env_id] = 0
                self._action_buffer[env_id].clear()
            if not self._frames[env_id] or self._steps[env_id] % self.frameskip == 0:
                self._frames[env_id].append(current[env_id].copy())
            self._steps[env_id] += 1

        histories = []
        for frames in self._frames:
            rows = list(frames)
            rows = [rows[0]] * (self.history_length - len(rows)) + rows
            histories.append(np.stack(rows))
        payload["_proposal_pixels_raw"] = np.stack(histories)
        payload["_env_id"] = np.arange(count, dtype=np.int64)
        payload["_plan_call"] = np.full(count, self._plan_call, dtype=np.int64)
        self._plan_call += 1

        info = self._prepare_info(payload)
        needs_flush = info.pop("_needs_flush", None)
        if needs_flush is not None:
            for env_id in range(count):
                if needs_flush[env_id]:
                    self._action_buffer[env_id].clear()
                    self._stage[env_id] = 0
                    self._elapsed[env_id] = 0
        terminated = info.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(count, dtype=bool)
        )
        replans = [
            env_id
            for env_id in range(count)
            if not dead[env_id] and not self._action_buffer[env_id]
        ]
        groups: dict[int, list[int]] = {}
        for env_id in replans:
            groups.setdefault(self._duration(env_id), []).append(env_id)

        old_horizon = int(self.cfg.horizon)
        old_receding = int(self.cfg.receding_horizon)
        try:
            for duration, env_ids in groups.items():
                if duration % int(self.cfg.action_block):
                    raise ValueError(
                        f"Schedule duration {duration} is not divisible by "
                        f"action_block={self.cfg.action_block}"
                    )
                tokens = duration // int(self.cfg.action_block)
                object.__setattr__(self.cfg, "horizon", tokens)
                object.__setattr__(self.cfg, "receding_horizon", tokens)
                index = torch.as_tensor(env_ids, dtype=torch.long)
                sliced = {}
                for key, value in info.items():
                    if torch.is_tensor(value):
                        sliced[key] = value[index]
                    elif isinstance(value, np.ndarray):
                        sliced[key] = value[env_ids]
                    elif isinstance(value, list):
                        sliced[key] = [value[i] for i in env_ids]
                    else:
                        sliced[key] = value
                elapsed = np.asarray(
                    [self._elapsed[i] for i in env_ids], dtype=np.int64
                )
                sliced["_remaining_steps"] = np.maximum(
                    self.goal_offset - elapsed, duration
                )
                sliced["_option_duration_steps"] = np.full(
                    len(env_ids), duration, dtype=np.int64
                )
                outputs = self.solver(sliced, init_action=None)
                plan = outputs["actions"][:, :tokens].reshape(
                    len(env_ids), duration, -1
                )
                for row, env_id in enumerate(env_ids):
                    self._action_buffer[env_id].extend(plan[row])
                    self._elapsed[env_id] += duration
                    self._stage[env_id] += 1
        finally:
            object.__setattr__(self.cfg, "horizon", old_horizon)
            object.__setattr__(self.cfg, "receding_horizon", old_receding)

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((count, action_dim), float("nan"))
        for env_id in range(count):
            if not dead[env_id]:
                action[env_id] = self._action_buffer[env_id].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


def set_determinism(seed: int):
    os.environ["PUSHT_CPU_MULTINOMIAL"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--policy", required=True, help="Frozen LeWM checkpoint")
    parser.add_argument("--generator")
    parser.add_argument("--action-prior", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--paper-config", default="configs/paper.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_determinism(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = load_json(args.manifest)
    paper = load_json(args.paper_config)
    verify_manifest(
        manifest,
        benchmark="pusht",
        seed=args.seed,
        protocol_id=paper["protocol_id"],
    )
    horizon = int(manifest["goal_offset_steps"])
    if int(manifest["num_eval"]) != len(manifest["records"]):
        raise ValueError("Manifest num_eval does not match record count")
    if args.seed not in paper["sample_seeds"]:
        raise ValueError(f"Seed {args.seed} is not a paper sample seed")
    schedule = list(paper["schedule"][str(horizon)])
    if sum(schedule) != horizon:
        raise ValueError(f"Schedule {schedule} does not sum to H={horizon}")

    planner = paper["planner"]
    dataset = swm.data.load_dataset(args.dataset, cache_dir=args.cache_dir)
    records = manifest["records"]
    episodes = [int(row["episode_id"]) for row in records]
    starts = [int(row["start_frame"]) for row in records]

    uses_generator = args.method in {"lewm_generator", "generator_prior_top", "sage"}
    uses_prior = args.method in {
        "far_goal_prior_cem",
        "generator_prior_top",
        "sage",
    }
    if uses_generator != bool(args.generator):
        requirement = "requires" if uses_generator else "forbids"
        raise ValueError(f"{args.method} {requirement} --generator")

    if uses_generator:
        generator, generator_stats, generator_ckpt = load_subgoal_prior(
            args.generator, device
        )
    else:
        generator, generator_stats, generator_ckpt = None, None, None

    loaded_prior, prior_stats, prior_ckpt = load_action_prior(
        args.action_prior, device
    )
    prior = loaded_prior if uses_prior else None
    lewm = load_lewm(args.policy, device=device, bf16=args.bf16)
    model = SAGECostModel(
        lewm,
        generator,
        generator_stats or prior_stats,
        prior,
        prior_stats,
        goal_offset_steps=horizon,
        action_block=int(planner["action_block"]),
        image_size=args.image_size,
    ).to(device)
    model.eval().requires_grad_(False)

    if args.method == "generator_prior_top":
        solver = PriorTopMode(model)
    else:
        solver_type = (
            PriorInitializedCEM
            if args.method in {"far_goal_prior_cem", "sage"}
            else GaussianCEM
        )
        solver = solver_type(
            model,
            candidates=int(planner["candidates"]),
            rounds=int(planner["cem_rounds"]),
            elites=int(planner["elites"]),
            seed=args.seed,
            device=device,
        )
    process = {
        "action": ArrayNormalizer(
            prior_stats["action_mean"].detach().cpu().numpy(),
            prior_stats["action_std"].detach().cpu().numpy(),
        )
    }
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform = {
        "pixels": image_transform(args.image_size, dtype),
        "goal": image_transform(args.image_size, dtype),
    }
    initial_tokens = schedule[0] // int(planner["action_block"])
    policy = ScheduledPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=initial_tokens,
            receding_horizon=initial_tokens,
            action_block=int(planner["action_block"]),
            warm_start=False,
        ),
        process=process,
        transform=transform,
        schedule_steps=schedule,
        goal_offset_steps=horizon,
        history_length=int(planner["history_length"]),
        frameskip=int(planner["frameskip"]),
    )
    budget = int(paper["environment_budget_multiplier"]["pusht"]) * horizon
    world = swm.World(
        "swm/PushT-v1",
        num_envs=len(records),
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=2 * budget,
    )
    world.set_policy(policy)
    video_dir = Path(args.out_dir) / "videos" if args.video else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        seed=args.seed,
        start_steps=starts,
        goal_offset=horizon,
        eval_budget=budget,
        episodes_idx=episodes,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ],
        video=video_dir,
    )
    result = {
        "protocol_id": paper["protocol_id"],
        "protocol_kind": "paper",
        "benchmark": "pusht",
        "method": args.method,
        "method_description": METHOD_DESCRIPTIONS[args.method],
        "seed": args.seed,
        "horizon": horizon,
        "schedule": schedule,
        "num_eval": len(records),
        "metrics": {
            "success_rate": float(metrics["success_rate"]),
            "episode_successes": np.asarray(
                metrics["episode_successes"]
            ).astype(bool).tolist(),
        },
        "record_ids": [row["record_id"] for row in records],
        "planner": {
            **planner,
            "effective_cem_rounds": (
                0 if args.method == "generator_prior_top" else planner["cem_rounds"]
            ),
            "warm_start": False,
        },
        "environment_budget": budget,
        "checkpoints": {
            "lewm": args.policy,
            "generator": (
                {
                    "path": args.generator,
                    "sha256": sha256_file(args.generator),
                    "epoch": generator_ckpt.get("epoch"),
                    "role": "local_goal_generation",
                }
                if uses_generator
                else None
            ),
            "action_prior": {
                "path": args.action_prior,
                "sha256": sha256_file(args.action_prior),
                "epoch": prior_ckpt.get("epoch"),
                "role": "proposal_generation" if uses_prior else "normalization_only",
            },
        },
        "subgoal_diagnostics": model.diagnostics(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["metrics"], sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
