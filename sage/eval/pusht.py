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
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms

from sage.models.action_prior import load_action_prior


DINOWM_ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
DINOWM_ACTION_STD = torch.tensor([0.2019, 0.2002])
DINOWM_PROPRIO_MEAN = np.array(
    [236.6155, 264.5674, -2.93032027, 2.54307914], dtype=np.float64
)
DINOWM_PROPRIO_STD = np.array(
    [101.1202, 87.0112, 74.84556075, 74.14009094], dtype=np.float64
)
from sage.models.subgoal import load_subgoal_prior
from sage.provenance import sha256_file, verify_manifest
from sage.runtime.lewm import (
    encode_lewm_context,
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
    "final_goal_scoring",
    "sage",
)

METHOD_DESCRIPTIONS = {
    "base_cem": "zero-mean Gaussian CEM scored against the final goal",
    "far_goal_prior_cem": "far-goal action-prior proposals refined by LeWM CEM",
    "lewm_generator": "zero-mean Gaussian CEM scored against generated subgoals",
    "generator_prior_top": "generated subgoals with the prior top mode; no LeWM ranking",
    "final_goal_scoring": "subgoal-conditioned proposals scored against the final goal",
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


def image_transform(image_size: int, dtype: torch.dtype, *, dinowm=False):
    normalization = (
        {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        if dinowm
        else spt.data.dataset_stats.ImageNet
    )
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Resize(size=int(image_size)),
            transforms.CenterCrop(size=int(image_size)),
            transforms.Normalize(**normalization),
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
        proposal_image_normalization: str = "imagenet",
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
        self.proposal_image_normalization = str(proposal_image_normalization)
        self.option_duration_steps = 25
        self._subgoal_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._generated = 0
        self._used_final_goal = 0
        self.register_buffer(
            "prior_action_mean",
            action_stats["action_mean"].detach().float().reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "prior_action_std",
            action_stats["action_std"].detach().float().reshape(-1),
            persistent=False,
        )

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
        # Proposal networks were trained on the SAGE ImageNet-normalized latent
        # cache. Keep that contract separate from DINO-WM's native [-1, 1]
        # visual inputs used for dynamics scoring.
        raw_goal = "_proposal_goal_raw" in info
        goal = info["_proposal_goal_raw"] if raw_goal else info["goal"]
        if not torch.is_tensor(goal):
            goal = torch.as_tensor(goal)
        if goal.ndim == 6:
            goal = goal[:, 0]
        if goal.shape[-1] in {1, 3, 4}:
            goal = goal.permute(0, 1, 4, 2, 3)
        # Planner goal tensors are already normalized; ImageNet values can
        # legitimately exceed 2. Only raw observations need preprocessing.
        if raw_goal:
            goal = image_batch_to_lewm(
                goal, self.image_size, self.proposal_image_normalization
            )
        goal = goal.to(
            device=next(self.lewm.parameters()).device,
            dtype=next(self.lewm.parameters()).dtype,
        )
        return encode_lewm_context(self.lewm, goal)

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
        pixels = image_batch_to_lewm(
            pixels, self.image_size, self.proposal_image_normalization
        ).to(
            device=next(self.lewm.parameters()).device,
            dtype=next(self.lewm.parameters()).dtype
        )
        return encode_lewm_context(self.lewm, pixels)

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
        # Chunk evaluation, not sampling: the memory budget must not change
        # the proposal RNG stream or the CEM iteration order.
        env_batch = max(1, int(os.environ.get("SAGE_ENV_BATCH", "1")))
        if actions.size(0) > env_batch:
            costs = []
            for start in range(0, actions.size(0), env_batch):
                stop = start + env_batch
                sliced = {
                    key: value[start:stop]
                    if torch.is_tensor(value) or isinstance(value, (np.ndarray, list))
                    else value
                    for key, value in info.items()
                }
                costs.append(self.get_cost(sliced, actions[start:stop]))
            return torch.cat(costs)
        lewm_info = {
            key: value for key, value in info.items() if not key.startswith("_")
        }
        if getattr(self, "score_final_goal", False):
            if "proprio" in getattr(self.lewm, "extra_encoders", {}):
                raise ValueError("Final-goal Scoring is the LeWM component ablation")
            lewm_info["goal_emb"] = self._final_goal_latents(info)
            return self.lewm.get_cost(lewm_info, actions)
        extra_encoders = getattr(self.lewm, "extra_encoders", {})
        if "proprio" in extra_encoders:
            # The proposal prior and released DINO-WM checkpoint were trained
            # with different action statistics. Convert prior-normalized raw
            # action blocks into DINO-WM coordinates before prediction.
            raw_action_dim = int(self.prior_action_mean.numel())
            action_shape = actions.shape
            raw_actions = actions.reshape(*action_shape[:-1], -1, raw_action_dim)
            prior_mean = self.prior_action_mean.to(actions)
            prior_std = self.prior_action_std.to(actions)
            raw_actions = raw_actions * prior_std + prior_mean
            dino_mean = DINOWM_ACTION_MEAN.to(actions)
            dino_std = DINOWM_ACTION_STD.to(actions)
            actions = ((raw_actions - dino_mean) / dino_std).reshape(action_shape)
            remaining = self._step_vector(
                info.get("_remaining_steps"),
                self.goal_offset_steps,
                actions.size(0),
                actions.device,
            )
            duration = self._step_vector(
                info.get("_option_duration_steps"),
                actions.size(2) * self.action_block,
                actions.size(0),
                actions.device,
            )
            if bool(torch.all(remaining <= duration)):
                chunk_size = int(os.environ.get("DINO_COST_CHUNK", "32"))
                return torch.cat(
                    [
                        self.lewm.get_cost(
                            dict(lewm_info),
                            actions[:, start : start + chunk_size],
                        )
                        for start in range(0, actions.size(1), chunk_size)
                    ],
                    dim=1,
                )

            # DINO-WM / PreJEPA uses the environment's real proprioception in
            # its action-conditioned rollout. SAGE supplies a generated visual
            # subgoal, so score the predicted visual endpoint against that
            # subgoal instead of asking PreJEPA to re-encode the far-goal image.
            local_goal = self._local_goal_latents(info)
            target = local_goal[:, -1] if local_goal.ndim >= 3 else local_goal
            if target.ndim == 3:
                target = target.mean(dim=-2)
            chunk_size = int(os.environ.get("DINO_COST_CHUNK", "16"))
            costs = []
            for start in range(0, actions.size(1), chunk_size):
                action_chunk = actions[:, start : start + chunk_size]
                if action_chunk.size(1) != chunk_size and hasattr(
                    self.lewm, "_init_cached_info"
                ):
                    del self.lewm._init_cached_info
                rollout = self.lewm.rollout(dict(lewm_info), action_chunk)
                predicted = rollout["predicted_pixels_emb"][:, :, -1]
                if predicted.ndim == 4:
                    predicted = predicted.mean(dim=-2)
                costs.append(
                    torch.nn.functional.mse_loss(
                        predicted,
                        target[:, None].expand_as(predicted),
                        reduction="none",
                    ).mean(dim=-1)
                )
            return torch.cat(costs, dim=1)
        local_goal = self._local_goal_latents(info)
        lewm_info["goal_emb"] = local_goal
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
        return self._solve_batch(info)

    def _solve_batch(self, info: dict):
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
        # Historical Gaussian CEM completes all rounds for one environment
        # before drawing the next environment's samples (batch_size=1).
        if batch > 1:
            outputs = []
            for row in range(batch):
                sliced = {
                    key: value[row:row + 1]
                    if torch.is_tensor(value) or isinstance(value, (np.ndarray, list))
                    else value
                    for key, value in info.items()
                }
                outputs.append(self.solve(sliced))
            return {
                "actions": torch.cat([item["actions"] for item in outputs]),
                "costs": [cost for item in outputs for cost in item["costs"]],
            }
        if getattr(self.model, "generator", None) is not None:
            # Candidate expansion intentionally drops raw proposal history. Cache
            # the generated local goal once at the unexpanded planner query.
            self.model._local_goal_latents(info)
        mean = torch.zeros(
            batch,
            horizon,
            self._action_dim,
            device=self.device,
            # Historical zero initialization is FP32 even with a BF16 model.
            # The first noise draw uses model precision; addition promotes the
            # candidates and subsequent elite statistics to FP32.
            dtype=torch.float32,
        )
        std = torch.ones_like(mean, dtype=self._dtype)
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
        payload["_proposal_goal_raw"] = np.asarray(info_dict["goal"]).copy()
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
    parser.add_argument("--action-prior")
    parser.add_argument(
        "--action-stats",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data/stats/pusht_train_seed42.json"
        ),
        help="Train-split action normalization, independent of the action prior.",
    )
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

    uses_generator = args.method in {"lewm_generator", "generator_prior_top", "final_goal_scoring", "sage"}
    uses_prior = args.method in {
        "far_goal_prior_cem",
        "generator_prior_top",
        "final_goal_scoring",
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

    action_stats_payload = load_json(args.action_stats)
    dataset_action_stats = {
        "action_mean": torch.as_tensor(
            action_stats_payload["action_mean"], device=device, dtype=torch.float32
        ),
        "action_std": torch.as_tensor(
            action_stats_payload["action_std"], device=device, dtype=torch.float32
        ),
    }
    if uses_prior:
        if not args.action_prior:
            raise ValueError(f"{args.method} requires --action-prior")
        loaded_prior, prior_stats, prior_ckpt = load_action_prior(
            args.action_prior, device
        )
        for key in ("action_mean", "action_std"):
            if not torch.allclose(
                prior_stats[key].float(),
                dataset_action_stats[key],
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError(
                    f"Action-prior {key} does not match independent train-split stats"
                )
        prior = loaded_prior
        component_stats = prior_stats
    else:
        prior = None
        prior_ckpt = None
        component_stats = generator_stats or {}
    lewm = load_lewm(args.policy, device=device, bf16=args.bf16)
    is_dinowm = "proprio" in getattr(lewm, "extra_encoders", {})
    if is_dinowm and not uses_prior:
        planner_action_stats = {
            "action_mean": DINOWM_ACTION_MEAN.to(device=device, dtype=torch.float32),
            "action_std": DINOWM_ACTION_STD.to(device=device, dtype=torch.float32),
        }
        action_coordinate = "dinowm_native_standardized_action"
    else:
        planner_action_stats = dataset_action_stats
        action_coordinate = action_stats_payload["coordinate_system"]
    runtime_stats = {**component_stats, **planner_action_stats}
    proposal_manifest = (
        generator_ckpt.get("run_manifest", {})
        if generator_ckpt is not None
        else (prior_ckpt or {}).get("run_manifest", {})
    )
    proposal_normalization = proposal_manifest.get("args", {}).get(
        "image_normalization", "imagenet"
    )
    model = SAGECostModel(
        lewm,
        generator,
        generator_stats or runtime_stats,
        prior,
        runtime_stats,
        goal_offset_steps=horizon,
        action_block=int(planner["action_block"]),
        image_size=args.image_size,
        proposal_image_normalization=proposal_normalization,
    ).to(device)
    model.eval().requires_grad_(False)

    model.score_final_goal = args.method == "final_goal_scoring"
    if args.method == "generator_prior_top":
        solver = PriorTopMode(model)
    else:
        solver_type = (
            PriorInitializedCEM
            if args.method in {"far_goal_prior_cem", "final_goal_scoring", "sage"}
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
            planner_action_stats["action_mean"].detach().cpu().numpy(),
            planner_action_stats["action_std"].detach().cpu().numpy(),
        )
    }
    if "proprio" in getattr(lewm, "extra_encoders", {}):
        proprio_process = StandardScaler()
        proprio_process.mean_ = DINOWM_PROPRIO_MEAN.copy()
        proprio_process.scale_ = DINOWM_PROPRIO_STD.copy()
        proprio_process.var_ = proprio_process.scale_**2
        proprio_process.n_features_in_ = proprio_process.mean_.shape[0]
        proprio_process.n_samples_seen_ = 1
        process["proprio"] = proprio_process
        process["goal_proprio"] = proprio_process
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform = {
        "pixels": image_transform(args.image_size, dtype, dinowm=is_dinowm),
        "goal": image_transform(args.image_size, dtype, dinowm=is_dinowm),
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
        "protocol_kind": "paper" if len(records) == int(paper["num_eval"]) else "subset",
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
            "action_prior": (
                {
                    "path": args.action_prior,
                    "sha256": sha256_file(args.action_prior),
                    "epoch": prior_ckpt.get("epoch"),
                    "role": "proposal_generation",
                }
                if uses_prior
                else None
            ),
            "action_stats": {
                "path": args.action_stats if uses_prior or not is_dinowm else None,
                "sha256": (
                    sha256_file(args.action_stats)
                    if uses_prior or not is_dinowm
                    else None
                ),
                "coordinate_system": action_coordinate,
                "mean": planner_action_stats["action_mean"].detach().cpu().tolist(),
                "std": planner_action_stats["action_std"].detach().cpu().tolist(),
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
