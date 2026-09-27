"""Terminal-goal LIBERO evaluation with portable data and inference bundles.

The controller follows the original terminal GMM evaluator: history stride 5,
15 raw actions per decision, top plus 63 GMM samples, and one LeWM ranking.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

from sage.native_data import resolve_path, validate_queries
from sage.native_models import load_components


class Controller:
    def __init__(self, bundle, method, device, seed=42):
        self.models, metadata = load_components(bundle, device)
        self.wm, self.gen, self.prior = (self.models[k] for k in ("world_model", "generator", "prior"))
        self.method, self.device, self.seed = method, device, seed
        self.rng = torch.Generator(device=device)
        stats = json.loads((Path(bundle) / "action_stats.json").read_text())
        self.mean, self.std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        prior_metadata = metadata["prior"]["manifest"]
        self.prior_mean = np.asarray(prior_metadata["action_mean"], np.float32)
        self.prior_scale = np.asarray(prior_metadata["action_scale"], np.float32)
        self.max_horizon = min(self.gen.max_goal_offset, self.prior.max_goal_offset)
        self.preprocess = transforms.Compose([
            transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(size=112),
        ])

    def images(self, array):
        return self.preprocess(torch.from_numpy(np.asarray(array)).permute(0, 3, 1, 2))[None].to(self.device)

    def encode(self, agent, wrist):
        return self.wm.encode({"pixels": self.images(agent), "wrist_pixels": self.images(wrist)})["emb"].float()

    @torch.inference_mode()
    def actions(self, agent, wrist, final_agent, final_wrist, recent, remaining, query_index, stage_start):
        history = self.encode(agent, wrist)
        final = self.encode(final_agent[None], final_wrist[None])
        remaining = min(max(15, remaining), self.max_horizon)
        lowdim = torch.empty(1, 0, device=self.device)
        delta = torch.full((1,), float(remaining), device=self.device)
        tau = torch.full((1,), 15., device=self.device)
        local = final
        if self.method != "base" and remaining > 15:
            local = self.gen(history, final, lowdim, delta, tau)["prediction"]
        if self.method == "base":
            # This is the historical fixed_residual Base, not IID Gaussian.
            rng = np.random.default_rng(self.seed + 1009 * query_index + 17 * stage_start)
            increments = rng.normal(0., 0.1, size=(64, 15, 7))
            candidates = recent[-1][None, None] + np.cumsum(increments, axis=1)
        else:
            kwargs = dict(action_horizon=3, far_goal_latents=final,
                          goal_offset_steps=delta, subgoal_offset_steps=tau)
            top = self.prior.top_mode(history, local, lowdim, **kwargs)
            if self.method == "prior_top":
                normalized = top[:, None]
            else:
                self.rng.manual_seed(self.seed + 1009 * query_index + 17 * stage_start)
                sampled = self.prior.sample(history, local, lowdim, 63, generator=self.rng, **kwargs)
                normalized = torch.cat([top[:, None], sampled], dim=1)
            candidates = normalized.float().cpu().numpy().reshape(-1, 15, 7)
            candidates = candidates * self.prior_scale + self.prior_mean
        candidates = np.clip(candidates, -1., 1.).astype(np.float32)
        candidates[:, :, 6] = np.where(candidates[:, :, 6] >= 0, 1., -1.)
        if self.method == "prior_top":
            return candidates[0], {"selected_index": 0, "candidate_count": 1}
        future = ((candidates - self.mean) / self.std).reshape(len(candidates), 3, 35)
        known = ((recent - self.mean) / self.std).reshape(2, 35)
        plans = np.concatenate([np.broadcast_to(known[None], (len(candidates), 2, 35)), future], axis=1)
        count = len(candidates)
        info = {
            "pixels": self.images(agent)[:, None].expand(-1, count, -1, -1, -1, -1),
            "wrist_pixels": self.images(wrist)[:, None].expand(-1, count, -1, -1, -1, -1),
            "goal": self.images(final_agent[None])[:, None].expand(-1, count, -1, -1, -1, -1),
            "goal_wrist_pixels": self.images(final_wrist[None])[:, None].expand(-1, count, -1, -1, -1, -1),
            "goal_emb": local, "action": torch.zeros(1, count, 3, 35, device=self.device),
        }
        costs = self.wm.get_cost(info, torch.from_numpy(plans.astype(np.float32))[None].to(self.device))[0]
        selected = int(costs.argmin())
        return candidates[selected], {"selected_index": selected, "candidate_count": count,
                                      "selected_pred_cost": float(costs[selected]),
                                      "candidate0_pred_cost": float(costs[0])}


def make_env(suite, task, render=True):
    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/native/libero_environment.json").read_text())
    spec = config["suites"][suite]
    bddl = next(Path(get_libero_path("bddl_files")).rglob(spec["tasks"][task]))
    kwargs = dict(config["env_kwargs"])
    if not render:
        kwargs.update(has_offscreen_renderer=False, use_camera_obs=False)
    kwargs["bddl_file_name"] = str(bddl)
    return TASK_MAPPING[spec["problem_name"]](**kwargs)


def proprio(obs):
    return np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("libero_scene2", "libero_caddy"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--handoff-anchors", type=Path, help="Optional SHA-locked cross-CPU prefix-anchor index")
    parser.add_argument("--headless-reset-history", action="store_true",
                        help="Enable cameras only at the final reset; retain the same handoff checks")
    parser.add_argument("--method", choices=("base", "prior_top", "sage"), required=True)
    parser.add_argument("--horizon", type=int, choices=(30, 60, 90, 120))
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    from sage.runtime.libero import configure
    configure(args.libero_root, args.out.parent / "libero_config", args.data_root)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest = json.loads(args.queries.read_text())
    if manifest["suite"] != args.suite:
        raise ValueError("Query suite and model suite differ")
    rows = [(i, q) for i, q in enumerate(manifest["queries"])
            if args.horizon is None or q["horizon"] == args.horizon]
    rows = rows[args.query_start:args.query_start + args.num_queries]
    if len(rows) != args.num_queries:
        raise ValueError("Insufficient queries for the requested slice")
    validate_queries({'queries': [q for _, q in rows]}, args.data_root)
    controller = Controller(args.bundle, args.method, args.device, args.seed)
    from sage.provenance import sha256_file
    identities = {name: sha256_file(args.bundle / name)
                  for name in ("world_model/weights.pt", "generator.pt", "prior.pt", "action_stats.json")}
    manifest_sha = sha256_file(args.queries)
    results = []
    for index, q in rows:
        env = make_env(args.suite, q["task"], render=not args.headless_reset_history)
        try:
            with h5py.File(resolve_path(args.data_root, q["source_shard"]), "r") as f:
                episode = int(q["source_episode"])
                # Reproduce the constructor/reset history of the collection shard.
                for previous in range(episode):
                    env.seed(int(f["episode_seed"][int(f["ep_offset"][previous])]))
                    env.reset()
                if args.headless_reset_history:
                    if not env.hard_reset:
                        raise ValueError("Headless reset history requires hard_reset")
                    env.has_offscreen_renderer = True
                    env.use_camera_obs = True
                    from robosuite.utils.binding_utils import MjRenderContextOffscreen
                    if env.sim._render_context_offscreen is None:
                        MjRenderContextOffscreen(env.sim, device_id=env.render_gpu_device_id)
                    # robosuite hard_reset updates existing sensors rather than
                    # adding observables that were absent at construction.
                    for name, observable in env._setup_observables().items():
                        if name not in env._observables:
                            env.add_observable(observable)
                env.seed(int(q["seed"]))
                obs = env.reset()
                for _ in range(int(q.get("settle_steps", 0))):
                    obs, _, _, _ = env.step(np.zeros(7, np.float32))
                offset = int(f["ep_offset"][episode]); length = int(f["ep_len"][episode])
                start, target, horizon = (int(q[k]) for k in ("start", "target", "horizon"))
                if not (0 <= start <= target < length and 0 <= target - start - horizon < 15):
                    raise ValueError("Invalid terminal query indices")
                actions = np.asarray(f["action"][offset:offset + length], np.float32)
                for action in actions[:start]:
                    obs, _, _, _ = env.step(action)
                if args.handoff_anchors:
                    from sage.runtime.libero_anchor import apply_indexed_anchor
                    obs = apply_indexed_anchor(env, args.handoff_anchors, args.suite, q)
                certificate = {}
                for key, obs_key in (("pixels", "agentview_image"), ("wrist_pixels", "robot0_eye_in_hand_image")):
                    diff = np.abs(np.asarray(obs[obs_key], np.int16) - f[key][offset + start].astype(np.int16))
                    certificate[key] = {"max": int(diff.max()), "mae": float(diff.mean())}
                certificate["proprio_max"] = float(np.max(np.abs(proprio(obs) - f["proprio"][offset + start])))
                if any(certificate[k]["max"] > 1 or certificate[k]["mae"] > 0.01
                       for k in ("pixels", "wrist_pixels")) or certificate["proprio_max"] > 2e-5:
                    raise RuntimeError(f"Replay certificate failed: {certificate}")
                if env._check_success():
                    raise RuntimeError("Query is already successful at handoff")
                indices = [max(0, t) for t in range(start - 10, start + 1)]
                agent = [np.asarray(f["pixels"][offset + t]) for t in indices]
                wrist = [np.asarray(f["wrist_pixels"][offset + t]) for t in indices]
                final_agent = np.asarray(f["pixels"][offset + target])
                final_wrist = np.asarray(f["wrist_pixels"][offset + target])
                recent = np.zeros((10, 7), np.float32)
                available = actions[max(0, start - 10):start]
                if len(available): recent[-len(available):] = available
            success = False; stages = []; executed = 0
            for stage_start in range(0, horizon + 120, 15):
                chunk, diagnostics = controller.actions(
                    np.stack([agent[k] for k in (-11, -6, -1)]),
                    np.stack([wrist[k] for k in (-11, -6, -1)]), final_agent, final_wrist,
                    recent, target - start - stage_start, int(q.get("query_index", index)), stage_start)
                for action in chunk:
                    obs, _, _, _ = env.step(action)
                    success = bool(success or env._check_success())
                    agent.append(np.asarray(obs["agentview_image"]))
                    wrist.append(np.asarray(obs["robot0_eye_in_hand_image"]))
                    recent = np.concatenate([recent, action[None]])[-10:]
                    executed += 1
                stages.append(diagnostics)
                if success: break
            results.append({"query_index": int(q.get("query_index", index)), "task": q["task"],
                            "episode_seed": q["seed"], "horizon": horizon, "official_success_ever": success,
                            "official_success_at_endpoint": bool(env._check_success()),
                            "executed_steps": executed, "handoff": certificate, "stages": stages})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            payload = {"checkpoints_sha256": identities, "query_manifest_sha256": manifest_sha,
                       "handoff_anchor_index_sha256": sha256_file(args.handoff_anchors) if args.handoff_anchors else None,
                       "run_scope": "paper_cell" if args.num_queries == 150 else "subset",
                       "protocol": {"suite": args.suite, "method": args.method, "K": 64,
                       "seed": args.seed, "query_start": args.query_start, "num_queries": args.num_queries,
                       "headless_reset_history": args.headless_reset_history,
                       "rank_rounds": 1, "commitment": 15, "recovery": 120,
                       "success_check": "every action; stop after current chunk"}, "results": results,
                       "summary": {"completed": len(results), "successes": sum(r["official_success_ever"] for r in results)}}
            temporary = args.out.with_suffix(".partial")
            temporary.write_text(json.dumps(payload, indent=2) + "\n"); temporary.replace(args.out)
            print(json.dumps(payload["summary"]), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
