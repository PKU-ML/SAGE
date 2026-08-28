"""Canonical OGBench Cube component evaluation for the SAGE paper."""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

from sage.models.action_prior import load_action_prior
from sage.models.subgoal import load_subgoal_prior
from sage.provenance import sha256_file, verify_manifest
from sage.runtime.lewm import encode_lewm_context, load_json, load_lewm, normalize_lowdim


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


class FixedScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1.0e-6)

    def transform(self, value):
        return ((np.asarray(value, dtype=np.float32) - self.mean) / self.std).astype(
            np.float32
        )

    def inverse_transform(self, value):
        return (np.asarray(value, dtype=np.float32) * self.std + self.mean).astype(
            np.float32
        )


def image_transform(image_size: int, dtype: torch.dtype):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=int(image_size)),
        ]
    )


class CubeSAGEModel(torch.nn.Module):
    def __init__(
        self,
        *,
        lewm,
        generator,
        generator_stats,
        prior,
        prior_stats,
        lowdim_keys: list[str],
        context_length: int,
        goal_offset_steps: int,
        action_block: int,
        device: torch.device,
    ):
        super().__init__()
        self.lewm = lewm
        self.generator = generator
        self.generator_stats = generator_stats
        self.prior = prior
        self.prior_stats = prior_stats
        self.lowdim_keys = list(lowdim_keys)
        self.context_length = int(context_length)
        self.goal_offset_steps = int(goal_offset_steps)
        self.action_block = int(action_block)
        self.device = device
        self.dtype = next(lewm.parameters()).dtype
        self._cache = {}

    @property
    def action_dim(self):
        if self.prior is None:
            raise RuntimeError("This method does not use an action prior")
        return int(self.prior.action_dim)

    def _history(self, info: dict):
        pixels = info.get("prior_pixels", info["pixels"])
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        pixels = pixels.to(self.device, dtype=self.dtype)
        if pixels.ndim == 6:
            pixels = pixels[:, 0]
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        if pixels.size(1) < self.context_length:
            pad = pixels[:, :1].expand(
                -1, self.context_length - pixels.size(1), -1, -1, -1
            )
            pixels = torch.cat([pad, pixels], dim=1)
        return encode_lewm_context(self.lewm, pixels[:, -self.context_length :])

    def _goal(self, info: dict):
        goal = info["goal"]
        if not torch.is_tensor(goal):
            goal = torch.as_tensor(goal)
        goal = goal.to(self.device, dtype=self.dtype)
        if goal.ndim == 6:
            goal = goal[:, 0]
        if goal.ndim == 4:
            goal = goal[:, None]
        return encode_lewm_context(self.lewm, goal[:, -1:])

    def _lowdim(self, info: dict, stats: dict):
        parts = []
        for key in self.lowdim_keys:
            if key not in info:
                raise KeyError(f"Missing action-prior input {key!r}")
            value = info[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(self.device, dtype=torch.float32)
            if value.ndim == 3:
                value = value[:, -1]
            parts.append(value)
        return normalize_lowdim(torch.cat(parts, dim=-1), stats)

    @staticmethod
    def _ids(info: dict, key: str, batch: int):
        value = info.get(key, np.arange(batch))
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        while value.ndim > 1:
            value = value[:, 0]
        return value.astype(np.int64).tolist()

    @staticmethod
    def _steps(info: dict, key: str, default: int, batch: int, device):
        value = info.get(key)
        if value is None:
            return torch.full(
                (batch,), float(default), device=device, dtype=torch.float32
            )
        if torch.is_tensor(value):
            value = value.to(device=device, dtype=torch.float32)
        else:
            value = torch.as_tensor(value, device=device, dtype=torch.float32)
        while value.ndim > 1:
            value = value[:, 0]
        return value.reshape(-1)

    @torch.no_grad()
    def local_goal(self, info: dict):
        if self.generator is None:
            return self._goal(info)
        batch = len(info["pixels"])
        env_ids = self._ids(info, "_env_id", batch)
        call_ids = self._ids(info, "_plan_call", batch)
        remaining = self._steps(
            info, "_remaining_steps", self.goal_offset_steps, batch, self.device
        )
        duration = self._steps(
            info, "_option_duration_steps", 25, batch, self.device
        )
        keys = [
            (int(env_id), int(call_id), int(remaining[row]), int(duration[row]))
            for row, (env_id, call_id) in enumerate(zip(env_ids, call_ids))
        ]
        missing = [row for row, key in enumerate(keys) if key not in self._cache]
        if missing:
            final = self._goal(info)
            history = self._history(info)
            lowdim = self._lowdim(info, self.generator_stats)
            for row in missing:
                key = keys[row]
                rem, tau = int(remaining[row]), int(duration[row])
                if rem <= tau:
                    prediction = final[row : row + 1]
                else:
                    prediction = self.generator(
                        history[row : row + 1],
                        final[row : row + 1],
                        lowdim[row : row + 1],
                        torch.tensor([rem], device=self.device, dtype=torch.float32),
                        torch.tensor([tau], device=self.device, dtype=torch.float32),
                    )["prediction"]
                self._cache[key] = prediction.detach()
        return torch.cat([self._cache[key] for key in keys])

    @torch.no_grad()
    def sample_prior(self, info: dict, count: int, horizon: int, generator):
        if self.prior is None:
            raise RuntimeError("Prior candidate sampling requested without a prior")
        history = self._history(info)
        local = self.local_goal(info)
        final = self._goal(info)
        lowdim = self._lowdim(info, self.prior_stats)
        batch = history.size(0)
        duration = self._steps(
            info,
            "_option_duration_steps",
            int(horizon) * self.action_block,
            batch,
            self.device,
        )
        remaining = self._steps(
            info, "_remaining_steps", self.goal_offset_steps, batch, self.device
        )
        remaining = torch.maximum(remaining, duration)
        top = self.prior.top_mode(
            history,
            local,
            lowdim,
            action_horizon=int(horizon),
            far_goal_latents=final,
            goal_offset_steps=remaining,
            subgoal_offset_steps=duration,
        )[:, None]
        if int(count) == 1:
            return top
        samples = self.prior.sample(
            history,
            local,
            lowdim,
            int(count) - 1,
            generator=generator,
            action_horizon=int(horizon),
            far_goal_latents=final,
            goal_offset_steps=remaining,
            subgoal_offset_steps=duration,
        )
        return torch.cat([top, samples], dim=1)

    @torch.no_grad()
    def top_prior(self, info: dict, horizon: int):
        if self.prior is None:
            raise RuntimeError("Prior top mode requested without a prior")
        history = self._history(info)
        local = self.local_goal(info)
        final = self._goal(info)
        lowdim = self._lowdim(info, self.prior_stats)
        batch = history.size(0)
        duration = self._steps(
            info,
            "_option_duration_steps",
            int(horizon) * self.action_block,
            batch,
            self.device,
        )
        remaining = self._steps(
            info, "_remaining_steps", self.goal_offset_steps, batch, self.device
        )
        return self.prior.top_mode(
            history,
            local,
            lowdim,
            action_horizon=int(horizon),
            far_goal_latents=final,
            goal_offset_steps=torch.maximum(remaining, duration),
            subgoal_offset_steps=duration,
        )

    @torch.no_grad()
    def get_cost(self, info: dict, actions: torch.Tensor):
        cost_info = {
            key: value
            for key, value in info.items()
            if key != "prior_pixels" and not key.startswith("_")
        }
        cost_info["goal_emb"] = self.local_goal(info)
        return self.lewm.get_cost(cost_info, actions)


class PriorInitializedCEM:
    def __init__(self, model, *, candidates, rounds, elites, seed, score_batch_size):
        self.model = model
        self.candidates = int(candidates)
        self.rounds = int(rounds)
        self.elites = int(elites)
        self.generator = torch.Generator(device=model.device).manual_seed(int(seed))
        self.score_batch_size = int(score_batch_size)

    def configure(self, *, action_space, n_envs, config):
        self._config = config
        self._n_envs = int(n_envs)
        env_dim = int(np.prod(action_space.shape[1:]))
        self._action_dim = env_dim * int(config.action_block)
        if self._action_dim != self.model.action_dim:
            raise ValueError(
                f"Prior action dim {self.model.action_dim} != env block dim {self._action_dim}"
            )

    def __call__(self, info, init_action=None):
        return self.solve(info, init_action)

    @property
    def horizon(self):
        return int(self._config.horizon)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def n_envs(self):
        return self._n_envs

    def _expand(self, info, count):
        expanded = {}
        for key, value in info.items():
            if key == "prior_pixels":
                continue
            if torch.is_tensor(value):
                value = value.to(self.model.device)
                if value.is_floating_point():
                    value = value.to(self.model.dtype)
                expanded[key] = value[:, None].expand(
                    value.size(0), count, *value.shape[1:]
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None], count, axis=1)
            else:
                expanded[key] = value
        return expanded

    def _score(self, info, candidates):
        count = int(candidates.size(1))
        chunk = min(self.score_batch_size, count)
        outputs = []
        for start in range(0, count, chunk):
            end = min(start + chunk, count)
            expanded = self._expand(info, end - start)
            outputs.append(
                self.model.get_cost(
                    expanded,
                    candidates[:, start:end].to(
                        self.model.device, dtype=self.model.dtype
                    ),
                )
                .float()
                .cpu()
            )
        return torch.cat(outputs, dim=1).to(candidates.device)

    @staticmethod
    def _fit(candidates, costs, topk):
        values, indices = torch.topk(costs, k=topk, dim=1, largest=False)
        rows = torch.arange(candidates.size(0), device=candidates.device)[:, None]
        selected = candidates[rows, indices]
        return selected.mean(1), selected.std(1).clamp_min(1.0e-6), values

    @torch.no_grad()
    def solve(self, info, init_action=None):
        del init_action
        candidates = self.model.sample_prior(
            info, self.candidates, self.horizon, self.generator
        )
        costs = self._score(info, candidates)
        mean, std, values = self._fit(candidates, costs, self.elites)
        for _ in range(1, self.rounds):
            candidates = torch.randn(
                mean.size(0),
                self.candidates,
                self.horizon,
                self._action_dim,
                generator=self.generator,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            candidates = candidates * std[:, None] + mean[:, None]
            candidates[:, 0] = mean
            costs = self._score(info, candidates)
            mean, std, values = self._fit(candidates, costs, self.elites)
        return {
            "actions": mean.detach().cpu(),
            "costs": values.mean(1).detach().cpu().tolist(),
        }


class GaussianCEM(PriorInitializedCEM):
    """True zero-mean Gaussian CEM; no action prior enters the solver."""

    def configure(self, *, action_space, n_envs, config):
        self._config = config
        self._n_envs = int(n_envs)
        env_dim = int(np.prod(action_space.shape[1:]))
        self._action_dim = env_dim * int(config.action_block)

    @torch.no_grad()
    def solve(self, info, init_action=None):
        if init_action is not None:
            raise ValueError("base_cem and lewm_generator forbid warm starts")
        batch = len(next(iter(info.values())))
        mean = torch.zeros(
            batch,
            self.horizon,
            self._action_dim,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        std = torch.ones_like(mean)
        values = None
        for _ in range(self.rounds):
            candidates = torch.randn(
                batch,
                self.candidates,
                self.horizon,
                self._action_dim,
                generator=self.generator,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            candidates = candidates * std[:, None] + mean[:, None]
            candidates[:, 0] = mean
            costs = self._score(info, candidates)
            mean, std, values = self._fit(candidates, costs, self.elites)
        return {
            "actions": mean.detach().cpu(),
            "costs": values.mean(1).detach().cpu().tolist(),
        }


class PriorTopMode:
    """Execute the highest-weight prior component without LeWM scoring."""

    def __init__(self, model):
        self.model = model

    def configure(self, *, action_space, n_envs, config):
        del action_space
        self._n_envs = int(n_envs)
        self._config = config

    @property
    def horizon(self):
        return int(self._config.horizon)

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return int(self.model.action_dim)

    def __call__(self, info, init_action=None):
        return self.solve(info, init_action=init_action)

    @torch.no_grad()
    def solve(self, info, init_action=None):
        if init_action is not None:
            raise ValueError("generator_prior_top forbids warm starts")
        actions = self.model.top_prior(info, self.horizon)
        return {"actions": actions.detach().cpu(), "costs": [float("nan")] * len(actions)}


class CubeScheduledPolicy(swm.policy.WorldModelPolicy):
    def __init__(
        self,
        *args,
        schedule_steps,
        goal_offset_steps,
        history_length,
        history_stride,
        lowdim_keys,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedule = [int(value) for value in schedule_steps]
        self.goal_offset = int(goal_offset_steps)
        self.history_length = int(history_length)
        self.history_stride = int(history_stride)
        self.lowdim_keys = list(lowdim_keys)
        self._history = None
        self._step = None
        self._stage = None
        self._elapsed = None
        self._plan_call = 0

    def set_env(self, env):
        super().set_env(env)
        count = int(env.num_envs)
        self._history = [deque(maxlen=self.history_length) for _ in range(count)]
        self._step = np.zeros(count, dtype=np.int64)
        self._stage = np.zeros(count, dtype=np.int64)
        self._elapsed = np.zeros(count, dtype=np.int64)
        self._action_buffer = [
            deque(maxlen=max(self.schedule)) for _ in range(count)
        ]
        self._next_init = None

    def _envs(self, count):
        envs = getattr(self.env, "envs", None)
        if envs is None and hasattr(getattr(self.env, "unwrapped", None), "envs"):
            envs = self.env.unwrapped.envs
        if envs is None:
            envs = [getattr(self.env, "unwrapped", self.env)] * count
        return [getattr(env, "unwrapped", env) for env in envs]

    def _inject_lowdim(self, info: dict, count: int):
        envs = self._envs(count)
        if "observation" in self.lowdim_keys and "observation" not in info:
            rows = []
            for env in envs:
                old_type = getattr(env, "_ob_type", None)
                if old_type == "pixels":
                    env._ob_type = "states"
                try:
                    rows.append(np.asarray(env.compute_observation(), dtype=np.float32))
                finally:
                    if old_type == "pixels":
                        env._ob_type = old_type
            info["observation"] = np.stack(rows)
        if "privileged_target_block_pos" in self.lowdim_keys:
            values = []
            for env in envs:
                target = int(getattr(env, "_target_block", 0))
                ids = getattr(env, "_cube_target_mocap_ids")
                values.append(np.asarray(env._data.mocap_pos[ids[target]], dtype=np.float32))
            info["privileged_target_block_pos"] = np.stack(values)
        source = "privileged/target_block_yaw"
        if (
            "privileged_target_block_yaw" in self.lowdim_keys
            and "privileged_target_block_yaw" not in info
            and source in info
        ):
            info["privileged_target_block_yaw"] = info[source]

    def _inject_history(self, info, flush, count):
        pixels = info["pixels"]
        histories = []
        for env_id in range(count):
            current = pixels[env_id, -1].detach()
            history = self._history[env_id]
            if flush[env_id] or not history:
                history.clear()
                for _ in range(self.history_length):
                    history.append(current)
                self._step[env_id] = 0
            elif self._step[env_id] % self.history_stride == 0:
                history.append(current)
            self._step[env_id] += 1
            histories.append(torch.stack(list(history)))
        info["prior_pixels"] = torch.stack(histories)

    def get_action(self, info_dict, **kwargs):
        del kwargs
        info = self._prepare_info(info_dict)
        count = int(self.env.num_envs)
        needs_flush = info.pop("_needs_flush", None)
        flush = (
            np.asarray(needs_flush, dtype=bool)
            if needs_flush is not None
            else np.zeros(count, dtype=bool)
        )
        for env_id in range(count):
            if flush[env_id]:
                self._action_buffer[env_id].clear()
                self._stage[env_id] = 0
                self._elapsed[env_id] = 0
        self._inject_lowdim(info, count)
        self._inject_history(info, flush, count)
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
        info["_env_id"] = np.arange(count, dtype=np.int64)
        info["_plan_call"] = np.full(count, self._plan_call, dtype=np.int64)
        self._plan_call += 1
        groups = {}
        for env_id in replans:
            stage = min(int(self._stage[env_id]), len(self.schedule) - 1)
            groups.setdefault(self.schedule[stage], []).append(env_id)
        old_horizon = int(self.cfg.horizon)
        old_receding = int(self.cfg.receding_horizon)
        try:
            for duration, env_ids in groups.items():
                tokens = duration // int(self.cfg.action_block)
                object.__setattr__(self.cfg, "horizon", tokens)
                object.__setattr__(self.cfg, "receding_horizon", tokens)
                indices = torch.as_tensor(env_ids, dtype=torch.long)
                sliced = {}
                for key, value in info.items():
                    if torch.is_tensor(value):
                        sliced[key] = value[indices]
                    elif isinstance(value, np.ndarray):
                        sliced[key] = value[env_ids]
                    elif isinstance(value, list):
                        sliced[key] = [value[i] for i in env_ids]
                    else:
                        sliced[key] = value
                elapsed = np.asarray([self._elapsed[i] for i in env_ids])
                sliced["_remaining_steps"] = np.maximum(
                    self.goal_offset - elapsed, duration
                )
                sliced["_option_duration_steps"] = np.full(
                    len(env_ids), duration, dtype=np.int64
                )
                actions = self.solver(sliced, init_action=None)["actions"]
                plan = actions[:, :tokens].reshape(len(env_ids), duration, -1)
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
        return self.process["action"].inverse_transform(action)


def cube_callables():
    return [
        {
            "method": "set_state",
            "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
        },
        {
            "method": "set_target_pos",
            "args": {
                "cube_id": {"in_dataset": False, "value": 0},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            },
        },
        {"method": "pre_step", "args": {}},
        {"method": "post_step", "args": {}},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy", default="quentinll/lewm-cube")
    parser.add_argument("--generator")
    parser.add_argument("--action-prior", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--paper-config", default="configs/paper.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--env-name", default="swm/OGBCube-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paper = load_json(args.paper_config)
    manifest = load_json(args.manifest)
    verify_manifest(
        manifest,
        benchmark="cube",
        seed=args.seed,
        protocol_id=paper["protocol_id"],
    )
    horizon = int(manifest["goal_offset_steps"])
    schedule = list(paper["schedule"][str(horizon)])
    if sum(schedule) != horizon:
        raise ValueError("Paper schedule does not sum to its horizon")
    if len(manifest["episodes_idx"]) != int(manifest["num_eval"]):
        raise ValueError("Manifest episode count mismatch")
    if args.seed not in paper["sample_seeds"]:
        raise ValueError(f"Seed {args.seed} is not a paper sample seed")

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
    loaded_prior, prior_stats, prior_ckpt = load_action_prior(args.action_prior, device)
    prior = loaded_prior if uses_prior else None
    run_args = prior_ckpt.get("run_manifest", {}).get("args", {})
    lowdim_keys = [
        *prior_stats.get("lowdim_keys", run_args.get("lowdim_keys", ["observation"])),
        *prior_stats.get("goal_lowdim_keys", run_args.get("goal_lowdim_keys", [])),
    ]
    context = int(run_args.get("context_len", run_args.get("history_len", 3)))
    lewm = load_lewm(args.policy, device=device, bf16=args.bf16)
    model = CubeSAGEModel(
        lewm=lewm,
        generator=generator,
        generator_stats=generator_stats or prior_stats,
        prior=prior,
        prior_stats=prior_stats,
        lowdim_keys=lowdim_keys,
        context_length=context,
        goal_offset_steps=horizon,
        action_block=int(paper["planner"]["action_block"]),
        device=device,
    ).to(device)
    model.eval().requires_grad_(False)
    planner = paper["planner"]
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
            candidates=planner["candidates"],
            rounds=planner["cem_rounds"],
            elites=planner["elites"],
            seed=args.seed,
            score_batch_size=args.score_batch_size,
        )
    process = {
        "action": FixedScaler(
            prior_stats["action_mean"].detach().cpu().numpy(),
            prior_stats["action_std"].detach().cpu().numpy(),
        )
    }
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform = {
        "pixels": image_transform(args.image_size, dtype),
        "goal": image_transform(args.image_size, dtype),
    }
    first_tokens = schedule[0] // int(planner["action_block"])
    policy = CubeScheduledPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=first_tokens,
            receding_horizon=first_tokens,
            history_len=1,
            action_block=int(planner["action_block"]),
            warm_start=False,
        ),
        process=process,
        transform=transform,
        schedule_steps=schedule,
        goal_offset_steps=horizon,
        history_length=context,
        history_stride=int(planner["frameskip"]),
        lowdim_keys=lowdim_keys,
    )
    world = swm.World(
        args.env_name,
        num_envs=int(manifest["num_eval"]),
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=2 * horizon,
    )
    world.set_policy(policy)
    video_dir = Path(args.out_dir) / "videos" if args.video else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    metrics = world.evaluate(
        dataset=swm.data.load_dataset(args.dataset),
        episodes_idx=[int(v) for v in manifest["episodes_idx"]],
        start_steps=[int(v) for v in manifest["start_steps"]],
        goal_offset=horizon,
        eval_budget=horizon,
        callables=cube_callables(),
        video=video_dir,
    )
    result = {
        "protocol_id": paper["protocol_id"],
        "protocol_kind": "paper",
        "benchmark": "cube",
        "method": args.method,
        "method_description": METHOD_DESCRIPTIONS[args.method],
        "seed": args.seed,
        "horizon": horizon,
        "schedule": schedule,
        "num_eval": int(manifest["num_eval"]),
        "metrics": {
            "success_rate": float(metrics["success_rate"]),
            "episode_successes": np.asarray(
                metrics["episode_successes"]
            ).astype(bool).tolist(),
        },
        "planner": {
            **planner,
            "effective_cem_rounds": (
                0 if args.method == "generator_prior_top" else planner["cem_rounds"]
            ),
            "warm_start": False,
        },
        "environment_budget": horizon,
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
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["metrics"], sort_keys=True))
    print(f"Wrote {path}")
    world.close()


if __name__ == "__main__":
    main()
