#!/usr/bin/env python3
"""Locked tail-query evaluation for stride-1 RoboTwin action proposals."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
# RoboTwin requires SAPIEN to be imported before Torch initializes CUDA.
_ROBOTWIN_SITE = os.environ.get("ROBOTWIN_SITE_PACKAGES")
if _ROBOTWIN_SITE and _ROBOTWIN_SITE not in sys.path:
    sys.path.append(_ROBOTWIN_SITE)
import sapien.core as sapien  # noqa: F401,E402
import torch
import yaml

from transformers import ViTConfig, ViTModel
from stable_worldmodel.wm.lewm import MultiViewLeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
from sage.models.subgoal import PushtSubgoalPrior
from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.native_data import materialize, validate_queries
from sage.provenance import sha256_file
from sage.runtime.eval_resume import validate_resume, retry_failed_tail


CAMERAS = {
    "pixels": "head_camera",
    "left_wrist_pixels": "left_camera",
    "right_wrist_pixels": "right_camera",
}

H5_CAMERAS = {
    "pixels": "vision/cam_head/colors",
    "left_wrist_pixels": "vision/cam_left_wrist/colors",
    "right_wrist_pixels": "vision/cam_right_wrist/colors",
}

_SAPIEN_RENDER_PROBE = None


class HandoffReached(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true', help='Retry a trailing infrastructure error using its exact pre-query RNG; preserve the failed attempt.')
    parser.add_argument('--max-new-queries', type=int, default=0,
                        help='Save and exit after this many new queries; 0 runs the entire slice.')
    parser.add_argument(
        "--robotwin-site-packages",
        type=Path,
        default=None,
    )
    parser.add_argument("--query-bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lewm-checkpoint", type=Path, required=True)
    parser.add_argument("--generator-checkpoint", type=Path)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument(
        "--controller",
        choices=("gaussian_lewm", "prior_top", "sage"),
        required=True,
    )
    parser.add_argument(
        "--gaussian-mean-mode",
        choices=("state_residual",),
        default="state_residual",
        help="Paper Base: cumulative Gaussian residuals from the observed joint state.",
    )
    parser.add_argument(
        "--gaussian-fixed-range-fraction",
        type=float,
        default=0.05,
        help=(
            "Per-step standard deviation for fixed_residual as a fraction of "
            "the environment action range."
        ),
    )
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--chunk-steps", type=int, default=15)
    parser.add_argument(
        "--recovery-raw-actions",
        type=int,
        default=120,
        help="Extra closed-loop action budget after the nominal tail horizon.",
    )
    parser.add_argument("--horizon", type=int, required=True)
    parser.set_defaults(full_episode=False)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=("place_a2b_left",), default="place_a2b_left")
    parser.add_argument("--task-config", default="lewm_place_a2b_left_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def load_module(path: Path, cls, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    proposal_type = config.pop("proposal_type", None)
    if cls is PushtVariableTransformerGoalPrior:
        if proposal_type not in (None, "gmm") or config.pop("uniform_mixture", False):
            raise ValueError("The paper release uses a learned-weight GMM")
    model = cls(**config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def load_lewm(path: Path, device: torch.device):
    """Construct the published checkpoint without importing the training stack."""
    encoder = ViTModel(
        ViTConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=224,
            patch_size=14,
        ),
        add_pooling_layer=False,
        use_mask_token=False,
    )
    encoder.config.interpolate_pos_encoding = True
    model = MultiViewLeWM(
        encoder=encoder,
        predictor=Predictor(
            num_frames=3, input_dim=1152, hidden_dim=608, output_dim=1152,
            depth=8, heads=10, mlp_dim=2432, dim_head=64,
            dropout=0.1, emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=14, emb_dim=1152),
        projector=MLP(
            input_dim=384, output_dim=384, hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=1152, output_dim=1152, hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        view_keys=("pixels", "left_wrist_pixels", "right_wrist_pixels"),
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


class OnlineEncoder:
    def __init__(self, model, device: torch.device, size: int = 224):
        self.model = model
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.size = int(size)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def tensorize(self, views: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        result = {}
        for key in CAMERAS:
            item = torch.from_numpy(np.ascontiguousarray(views[key])).permute(2, 0, 1).float() / 255.0
            item = (item - self.mean) / self.std
            item = torch.nn.functional.interpolate(
                item[None], size=(self.size, self.size), mode="bilinear", align_corners=False
            )[0]
            result[key] = item[None, None].to(self.device, dtype=self.dtype)
        return result

    @torch.inference_mode()
    def encode(self, views: dict[str, np.ndarray]) -> torch.Tensor:
        return self.model.encode(self.tensorize(views))["emb"][:, 0].float()


def observation_views(observation: dict) -> dict[str, np.ndarray]:
    source = observation["observation"]
    return {
        target: cv2.cvtColor(
            np.asarray(source[name]["rgb"], dtype=np.uint8), cv2.COLOR_BGR2RGB
        )
        for target, name in CAMERAS.items()
    }


def load_goal_views(query: dict) -> dict[str, np.ndarray]:
    mapping = {
        "pixels": "cam_head",
        "left_wrist_pixels": "cam_left_wrist",
        "right_wrist_pixels": "cam_right_wrist",
    }
    views = {}
    for target, source in mapping.items():
        image = cv2.imread(query["terminal_goal"][source]["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(query["terminal_goal"][source]["path"])
        views[target] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return views


def prepare_runtime(args: argparse.Namespace):
    global _SAPIEN_RENDER_PROBE
    root = args.robotwin_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    # Keep the LeWM Python environment authoritative and append only the
    # RoboTwin runtime. Prepending it replaces Torch/Gym binary dependencies.
    if args.robotwin_site_packages is not None:
        sys.path.append(str(args.robotwin_site_packages))
    from sage.runtime.robotwin import Sapien_TEST, embodiment_args

    # SAPIEN 3.0 beta can crash while destroying a ray-tracing renderer.
    # Retain the preflight probe for the evaluator lifetime instead of
    # immediately releasing it after construction.
    _SAPIEN_RENDER_PROBE = Sapien_TEST()
    config_path = root / "env_cfg" / "task_config" / f"{args.task_config}.yml"
    config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    config.update(task_name=args.task, task_config=args.task_config)
    embodiment_args(config)
    config.update(need_plan=False, render_freq=0, save_data=False, save_video=False)
    module = __import__(f"envs.{args.task}", fromlist=[args.task])
    return getattr(module, args.task), config


def stride1_paths(manifest_path: Path) -> dict[tuple[str, int], Path]:
    payload = json.loads(manifest_path.read_text())
    root = Path(payload["source_root"])
    result = {}
    for row in payload["episodes"]:
        source = Path(row["source"])
        shard = source.parts[1]
        episode = int(source.stem.split("_")[-1])
        result[(shard, episode)] = root / source
    return result


def replay_to_handoff(env, query: dict) -> dict:
    capture_index = 0
    observation = None

    def tap():
        nonlocal capture_index, observation
        if capture_index == int(query["handoff_capture_index"]):
            observation = env.get_obs()
            raise HandoffReached
        capture_index += 1

    env._take_picture = tap
    try:
        env.play_once()
        raise RuntimeError("oracle ended before handoff")
    except HandoffReached:
        pass
    if observation is None:
        raise RuntimeError("handoff observation missing")
    return observation


def decode_h5_jpeg(value) -> np.ndarray:
    raw = value.tobytes() if isinstance(value, np.ndarray) else bytes(value)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("invalid HDF5 JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def load_handoff_context(path: Path, handoff: int, encoder: OnlineEncoder, device):
    with h5py.File(path, "r") as handle:
        if handoff < 2 or handoff >= len(handle[H5_CAMERAS["pixels"]]):
            raise IndexError((path, handoff))
        latents = []
        for index in range(handoff - 2, handoff + 1):
            views = {
                key: decode_h5_jpeg(handle[source][index])
                for key, source in H5_CAMERAS.items()
            }
            latents.append(encoder.encode(views)[0])
        actions = np.asarray(handle["lewm/action_blocks"][:, 0], dtype=np.float32)
        past = torch.as_tensor(actions[handoff - 2 : handoff], device=device)
    return torch.stack(latents, dim=0)[None], past


@torch.inference_mode()
def local_goal(generator, history, far_goal, remaining: int, chunk_steps: int):
    if remaining <= chunk_steps:
        return far_goal[:, None]
    offsets = torch.tensor([remaining], device=history.device, dtype=torch.float32)
    subgoal = torch.full_like(offsets, float(chunk_steps))
    return generator(
        history, far_goal[:, None], torch.empty(1, 0, device=history.device), offsets, subgoal
    )["prediction"]


@torch.inference_mode()
def prior_candidates(prior, history, local, far_goal, remaining, chunk_steps, count, rng):
    empty = torch.empty(1, 0, device=history.device)
    goal_steps = torch.tensor([remaining], device=history.device, dtype=torch.float32)
    subgoal_steps = torch.full_like(goal_steps, float(chunk_steps))
    top = prior.top_mode(
        history, local, empty, chunk_steps, far_goal[:, None], goal_steps, subgoal_steps
    )[:, None]
    if count <= 1:
        return top
    sampled = prior.sample(
        history, local, empty, count - 1, generator=rng,
        action_horizon=chunk_steps, far_goal_latents=far_goal[:, None],
        goal_offset_steps=goal_steps, subgoal_offset_steps=subgoal_steps,
    )
    return torch.cat([top, sampled], dim=1)


@torch.inference_mode()
def rank_with_lewm(model, history, past_actions, future_actions, target):
    """Match recursive B training: two past actions plus 15 proposed actions."""
    candidates = future_actions[0]
    count = candidates.size(0)
    prefix = (
        candidates[:, :1].expand(-1, 2, -1)
        if past_actions is None
        else past_actions[None].expand(count, -1, -1)
    )
    actions = torch.cat([prefix, candidates], dim=1)
    act_emb = model.action_encoder(actions.to(next(model.parameters()).dtype))
    embeddings = [item.expand(count, -1) for item in history[0].unbind(dim=0)]
    for step in range(candidates.size(1)):
        hist = torch.stack(embeddings[-3:], dim=1)
        embeddings.append(model.predict(hist, act_emb[:, step : step + 3])[:, -1])
    endpoint = embeddings[-1].float()
    costs = (endpoint - target.float()).square().sum(dim=-1)
    best = int(costs.argmin())
    return future_actions[:, best], costs, endpoint, best


def create_env(query: dict, env_class, base_config: dict):
    config = dict(base_config)
    config["save_path"] = query["source_dir"]
    env = env_class()
    env.setup_demo(now_ep_num=query["episode_index"], seed=query["seed"], **config)
    with Path(query["trajectory"]).open("rb") as handle:
        trajectory = pickle.load(handle)
    config["left_joint_path"] = trajectory["left_joint_path"]
    config["right_joint_path"] = trajectory["right_joint_path"]
    env.set_path_lst(config)
    return env


def current_joint_vector(env) -> np.ndarray:
    return np.asarray(
        env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
        dtype=np.float32,
    )


def robot_action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    """Return native qpos-command bounds without consulting demonstration data."""
    lower: list[float] = []
    upper: list[float] = []
    for joints in (env.robot.left_arm_joints, env.robot.right_arm_joints):
        for joint in joints:
            limits = np.asarray(joint.get_limits(), dtype=np.float32).reshape(-1, 2)[0]
            if not np.isfinite(limits).all() or limits[1] <= limits[0]:
                limits = np.asarray([-np.pi, np.pi], dtype=np.float32)
            lower.append(float(limits[0]))
            upper.append(float(limits[1]))
        lower.append(0.0)
        upper.append(1.0)
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def atomic_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.recovery_raw_actions < 0:
        raise ValueError("recovery-raw-actions must be nonnegative")
    if args.recovery_raw_actions % args.chunk_steps:
        raise ValueError("recovery-raw-actions must be divisible by chunk-steps")
    if not 0.0 < args.gaussian_fixed_range_fraction <= 0.5:
        raise ValueError("gaussian-fixed-range-fraction must be in (0, 0.5]")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    bank = json.loads(args.query_bank.read_text())
    queries = [
        query for query in bank["queries"]
        if int(query["horizon_raw_actions"]) == args.horizon
    ]
    selected = queries[args.query_start : args.query_start + args.num_queries]
    if len(selected) != args.num_queries:
        raise ValueError(f"found {len(selected)}/{args.num_queries} H{args.horizon} queries")
    # A worker needs only its assigned dependencies, not every other shard.
    validate_queries({"queries": selected}, args.data_root)
    selected = materialize({"queries": selected}, args.data_root)["queries"]
    paths = {(q["shard"], int(q["episode_index"])): Path(q["episode_hdf5"])
             for q in selected}

    # Initialize SAPIEN/Vulkan before Torch creates a CUDA context.
    env_class, base_config = prepare_runtime(args)
    lewm = load_lewm(args.lewm_checkpoint, device)
    generator = prior = None
    generator_payload = prior_payload = {}
    if args.controller != "gaussian_lewm":
        if args.generator_checkpoint is None or args.prior_checkpoint is None:
            raise ValueError(f"{args.controller} requires generator and prior checkpoints")
        generator, generator_payload = load_module(
            args.generator_checkpoint, PushtSubgoalPrior, device
        )
        prior, prior_payload = load_module(
            args.prior_checkpoint, PushtVariableTransformerGoalPrior, device
        )
        if prior.action_dim != 14 or prior.max_plan_horizon < args.chunk_steps:
            raise ValueError((prior.action_dim, prior.max_plan_horizon, args.chunk_steps))
    stats = np.load(args.action_stats)
    action_mean = torch.as_tensor(stats["mean"], device=device, dtype=torch.float32)
    action_std = torch.as_tensor(stats["std"], device=device, dtype=torch.float32)
    encoder = OnlineEncoder(lewm, device)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    proposal_type = (
        "current_state_anchored_gaussian_random_walk"
        if args.controller == "gaussian_lewm"
        else getattr(prior, "proposal_type", "gmm")
    )
    payload = {
        "protocol": {
            "name": "robotwin_stride1_terminal_gmm_tail_v1",
            "query_manifest_sha256": sha256_file(args.query_bank),
            "query_start": args.query_start,
            "num_queries": args.num_queries,
            "input_sha256": {
                name: sha256_file(path) for name, path in (
                    ("world_model", args.lewm_checkpoint),
                    ("generator", args.generator_checkpoint),
                    ("prior", args.prior_checkpoint),
                    ("action_stats", args.action_stats),
                ) if path is not None
            },
            "controller": args.controller,
            "gaussian_mean_mode": (
                args.gaussian_mean_mode if args.controller == "gaussian_lewm" else None
            ),
            "gaussian_fixed_range_fraction": (
                args.gaussian_fixed_range_fraction
                if args.controller == "gaussian_lewm"
                and args.gaussian_mean_mode
                in {"fixed_residual", "fixed_zero", "state_residual"}
                else None
            ),
            "gaussian_proposal_uses_expert_statistics": not (
                args.controller == "gaussian_lewm"
                and args.gaussian_mean_mode
                in {"fixed_residual", "fixed_zero", "state_residual"}
            ),
            "gaussian_anchor": (
                "current_observed_robot_state"
                if args.controller == "gaussian_lewm"
                and args.gaussian_mean_mode == "state_residual"
                else (
                    "last_prefix_or_executed_action"
                    if args.controller == "gaussian_lewm"
                    and args.gaussian_mean_mode in {"residual", "fixed_residual"}
                    else "zero_or_distribution_mean"
                )
            ),
            "gaussian_temporal_sampling": (
                "iid_zero_centered"
                if args.controller == "gaussian_lewm"
                and args.gaussian_mean_mode == "fixed_zero"
                else (
                    "cumulative_random_walk"
                    if args.controller == "gaussian_lewm"
                    and args.gaussian_mean_mode
                    in {"residual", "fixed_residual", "state_residual"}
                    else "iid"
                )
            ),
            "proposal_type": proposal_type,
            "chunk_raw_actions": args.chunk_steps, "replan_every_raw_actions": args.chunk_steps,
            "num_candidates": (
                args.num_candidates if args.controller in {"gaussian_lewm", "sage"} else 1
            ),
            "cem_rounds": 1, "iterative_cem": False, "history_frames": 3,
            "standalone_candidate": (
                "none; all Gaussian candidates are ranked by LeWM"
                if args.controller == "gaussian_lewm" else "highest_weight_gmm_component_mean"
            ),
            "initial_history": "three stored stride1 expert frames at handoff",
            "initial_action_history": "two stored stride1 expert actions at handoff",
            "replay_check": "exact_joint_commands_and_reported_image_differences_v1",
            "goal": "sha_verified_terminal_multiview_image",
            "horizon": "full" if args.full_episode else args.horizon,
            "recovery_raw_actions": args.recovery_raw_actions,
            "max_steps": (
                "per_query_terminal_token_x5_plus_recovery"
                if args.full_episode
                else args.horizon + args.recovery_raw_actions
            ),
            "official_success_early_stop": True,
            "lewm_checkpoint": str(args.lewm_checkpoint),
            "generator_checkpoint": (
                str(args.generator_checkpoint) if args.generator_checkpoint else None
            ),
            "prior_checkpoint": (
                str(args.prior_checkpoint) if args.prior_checkpoint else None
            ),
            "generator_epoch": generator_payload.get("epoch"),
            "prior_epoch": prior_payload.get("epoch"), "seed": args.seed,
        },
        "results": [],
    }
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if args.retry_failed:
            existing = retry_failed_tail(existing, payload['protocol'],
                [q['query_id'] for q in selected], rng.get_state().cpu().tolist())
        state = validate_resume(existing, payload["protocol"], [q["query_id"] for q in selected])
        if state is not None:
            rng.set_state(torch.tensor(state, dtype=torch.uint8))
        payload = existing
    completed = {row["query_id"] for row in payload["results"]}
    new_queries = 0

    for query in selected:
        key = (query["shard"], int(query["episode_index"]))
        query_id = query["query_id"]
        if args.full_episode:
            query_id = query_id.rsplit("-H", 1)[0] + "-full"
        if query_id in completed:
            continue
        started, env = time.time(), None
        rng_before = rng.get_state().cpu().tolist()
        try:
            env = create_env(query, env_class, base_config)
            fixed_lower = fixed_upper = fixed_scale = None
            if (
                args.controller == "gaussian_lewm"
                and args.gaussian_mean_mode
                in {"fixed_residual", "fixed_zero", "state_residual"}
            ):
                lower_np, upper_np = robot_action_bounds(env)
                if lower_np.shape != tuple(action_mean.shape):
                    raise ValueError((lower_np.shape, tuple(action_mean.shape)))
                fixed_lower = torch.as_tensor(lower_np, device=device)
                fixed_upper = torch.as_tensor(upper_np, device=device)
                fixed_scale = (
                    args.gaussian_fixed_range_fraction * (fixed_upper - fixed_lower)
                )
            if args.full_episode:
                query = dict(query)
                query["handoff_capture_index"] = 0
            handoff_observation = replay_to_handoff(env, query)
            from sage.runtime.robotwin_replay import inspect_handoff
            replay_diagnostics = inspect_handoff(
                paths[key], int(query["handoff_capture_index"]), current_joint_vector(env),
                observation_views(handoff_observation), H5_CAMERAS,
            )
            if args.full_episode:
                current = encoder.encode(observation_views(handoff_observation))[0]
                history = current[None, None].expand(1, 3, -1).clone()
                past_actions = None
                token_span = int(query["goal_token"]) - int(query["handoff_token"])
                if token_span <= 0 or int(query["horizon_raw_actions"]) % token_span:
                    raise ValueError(
                        "cannot infer raw-action stride from query: "
                        f"horizon={query['horizon_raw_actions']} token_span={token_span}"
                    )
                raw_actions_per_token = int(query["horizon_raw_actions"]) // token_span
                nominal_steps = int(query["terminal_token"]) * raw_actions_per_token
            else:
                history, past_actions = load_handoff_context(
                    paths[key], int(query["handoff_capture_index"]), encoder, device
                )
                nominal_steps = args.horizon
            far_goal = encoder.encode(load_goal_views(query))
            max_steps = nominal_steps + args.recovery_raw_actions
            executed = 0
            success = bool(env.check_success())
            stages = []
            query_rng = None
            if args.controller == "gaussian_lewm":
                query_seed = (
                    args.seed * 1_000_003
                    + int(query["seed"]) * 9_176
                    + args.horizon * 101
                )
                query_rng = torch.Generator(device=device).manual_seed(query_seed)
            while executed < max_steps and not success:
                total_remaining = max_steps - executed
                nominal_remaining = nominal_steps - executed
                if nominal_remaining > 0:
                    current_chunk = min(args.chunk_steps, nominal_remaining)
                    goal_remaining = nominal_remaining
                else:
                    current_chunk = min(args.chunk_steps, total_remaining)
                    goal_remaining = current_chunk
                if args.controller == "gaussian_lewm":
                    local = far_goal[:, None]
                    normalized = torch.randn(
                        (1, args.num_candidates, current_chunk, action_mean.numel()),
                        device=device,
                        generator=query_rng,
                    ).clamp_(-5.0, 5.0)
                else:
                    assert generator is not None and prior is not None
                    local = local_goal(
                        generator, history, far_goal, goal_remaining, current_chunk
                    )
                if args.controller != "gaussian_lewm":
                    normalized = prior_candidates(
                        prior, history, local, far_goal, goal_remaining, current_chunk,
                        args.num_candidates if args.controller == "sage" else 1, rng,
                    )
                if args.controller == "gaussian_lewm":
                    assert fixed_scale is not None
                    residuals = normalized.float() * fixed_scale
                    start_action = torch.as_tensor(current_joint_vector(env), device=device, dtype=torch.float32)
                    raw = start_action[None, None, None] + residuals.cumsum(dim=2)
                    assert fixed_lower is not None and fixed_upper is not None
                    raw = torch.maximum(
                        torch.minimum(raw, fixed_upper[None, None, None]),
                        fixed_lower[None, None, None],
                    )
                else:
                    raw = normalized.float() * action_std + action_mean
                if args.controller in {"gaussian_lewm", "sage"}:
                    chosen, costs, predicted_endpoints, chosen_index = rank_with_lewm(
                        lewm, history, past_actions, raw, local[:, 0]
                    )
                    selected_cost, top_cost = float(costs[chosen_index]), float(costs[0])
                    selected_prediction = predicted_endpoints[chosen_index]
                else:
                    chosen, chosen_index, selected_cost, top_cost = raw[:, 0], 0, None, None
                    costs = predicted_endpoints = selected_prediction = None

                current_latent = history[0, -1].float()
                local_target = local[0, 0].float()
                final_target = far_goal[0].float()
                current_to_local = float((current_latent - local_target).square().sum())
                current_to_final = float((current_latent - final_target).square().sum())

                stage_actions, new_latents = [], []
                execute_count = min(int(chosen.size(1)), max_steps - executed)
                for action_index, action in enumerate(chosen[0, :execute_count]):
                    env.take_action(action.detach().cpu().numpy(), action_type="qpos")
                    stage_actions.append(action.detach()); executed += 1
                    success = bool(env.check_success())
                    # Only the final three observations become the next LeWM
                    # history. Official success is still checked every step.
                    if action_index >= execute_count - 3 or success:
                        new_latents.append(encoder.encode(observation_views(env.get_obs()))[0])
                    if success or executed >= max_steps:
                        break
                merged = torch.cat([history[0], torch.stack(new_latents)], dim=0)
                history = merged[-3:][None]
                actual_endpoint = history[0, -1].float()
                action_tensor = torch.stack(stage_actions)
                if past_actions is None:
                    past_actions = action_tensor[-2:]
                    if past_actions.size(0) == 1:
                        past_actions = past_actions.expand(2, -1).clone()
                else:
                    past_actions = torch.cat([past_actions, action_tensor], dim=0)[-2:]
                stage_record = {
                    "stage": len(stages),
                    "remaining_expert_budget": max(nominal_remaining, 0),
                    "remaining_total_budget": total_remaining,
                    "executed_total": executed, "chosen_index": chosen_index,
                    "selected_pred_cost": selected_cost, "prior_top_pred_cost": top_cost,
                    "current_to_local_cost": current_to_local,
                    "current_to_final_cost": current_to_final,
                    "actual_endpoint_to_local_cost": float(
                        (actual_endpoint - local_target).square().sum()
                    ),
                    "actual_endpoint_to_final_cost": float(
                        (actual_endpoint - final_target).square().sum()
                    ),
                    "success": success,
                }
                if costs is not None:
                    sorted_costs = costs.float().sort().values
                    stage_record.update({
                        "candidate_pred_cost_min": float(sorted_costs[0]),
                        "candidate_pred_cost_median": float(sorted_costs[len(sorted_costs) // 2]),
                        "candidate_pred_cost_max": float(sorted_costs[-1]),
                        "selected_prediction_error": float(
                            (selected_prediction - actual_endpoint).square().sum()
                        ),
                    })
                stages.append(stage_record)
            row = {
                "query_id": query_id, "episode_index": int(query["episode_index"]),
                "seed": int(query["seed"]),
                "horizon": "full" if args.full_episode else args.horizon,
                "nominal_raw_steps": nominal_steps,
                "recovery_raw_actions": args.recovery_raw_actions,
                "max_raw_steps": max_steps, "executed_raw_steps": executed,
                "official_success": success, "elapsed_seconds": time.time() - started,
                "replay_diagnostics": replay_diagnostics,
                "stages": stages,
            }
        except Exception as exc:
            row = {
                "query_id": query_id, "episode_index": int(query["episode_index"]),
                "seed": int(query["seed"]), "official_success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "proposal_rng_state_before": rng_before,
                "elapsed_seconds": time.time() - started,
            }
        finally:
            if env is not None:
                try: env.close_env(clear_cache=True)
                except Exception: pass
        payload["results"].append(row)
        payload["proposal_rng_state"] = rng.get_state().cpu().tolist()
        successes = sum(bool(item.get("official_success")) for item in payload["results"])
        payload["summary"] = {
            "completed": len(payload["results"]), "successes": successes,
            "success_rate": successes / len(payload["results"]),
            "errors": sum("error" in item for item in payload["results"]),
        }
        atomic_write(args.out, payload)
        print(json.dumps({**payload["summary"], "last": row}), flush=True)
        if "error" in row:
            raise RuntimeError(f"Evaluation failed for {query_id}: {row['error']}")
        new_queries += 1
        if args.max_new_queries > 0 and new_queries >= args.max_new_queries:
            break

if __name__ == "__main__":
    main()
