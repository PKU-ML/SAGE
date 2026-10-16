#!/usr/bin/env python3
"""Collect RoboTwin demonstrations on a LEWM-aligned visual time grid.

The stock collector serializes every rendered observation to a temporary pickle,
then reads all pickles back to encode the final HDF5.  This collector keeps the
same simulator rollout and oracle joint paths, but buffers observations in
memory, renders every ``visual_stride`` capture, and writes the episode once.
It also stores every intermediate joint state in each visual interval under
``lewm/action_blocks`` so a loader can construct five-step action tokens.
"""

import argparse
import json
import os
import pickle
import re
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import sapien.core as sapien  # noqa: F401  Required before env imports.
import yaml
from sapien.render import clear_cache

sys.path.append("./")

from envs import *  # noqa: F401,F403
from description.utils.generate_episode_instructions import generate_episode_descriptions
from envs.utils.pkl2hdf5 import (
    append_data_to_structure,
    create_xpolicylab_hdf5,
    parse_dict_structure,
)
from test_render import Sapien_TEST


def _embodiment_args(args):
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as handle:
        embodiments = yaml.load(handle, Loader=yaml.FullLoader)

    def load_config(kind):
        path = embodiments[kind]["file_path"]
        with open(os.path.join(path, "config.yml"), encoding="utf-8") as handle:
            return path, yaml.load(handle, Loader=yaml.FullLoader)

    kind = args["embodiment"]
    if len(kind) == 1:
        left_path, left_cfg = load_config(kind[0])
        args.update(
            left_robot_file=left_path,
            right_robot_file=left_path,
            left_embodiment_config=left_cfg,
            right_embodiment_config=left_cfg,
            dual_arm_embodied=True,
            embodiment_name=str(kind[0]),
        )
    elif len(kind) == 3:
        left_path, left_cfg = load_config(kind[0])
        right_path, right_cfg = load_config(kind[1])
        args.update(
            left_robot_file=left_path,
            right_robot_file=right_path,
            left_embodiment_config=left_cfg,
            right_embodiment_config=right_cfg,
            embodiment_dis=kind[2],
            dual_arm_embodied=False,
            embodiment_name=f"{kind[0]}+{kind[1]}",
        )
    else:
        raise ValueError("embodiment must contain one or three entries")


def _instruction(task_name, episode_info, episode_idx, out_dir, language_num):
    fallback = task_name.replace("_", " ")
    generated = generate_episode_descriptions(
        task_name, [episode_info.get("info", {})], int(language_num or 100)
    )
    descriptions = generated[0] if generated else {"seen": [fallback], "unseen": [fallback]}
    if not descriptions.get("seen"):
        descriptions["seen"] = [fallback]
    if not descriptions.get("unseen"):
        descriptions["unseen"] = [fallback]
    instruction_dir = Path(out_dir) / "instruction"
    instruction_dir.mkdir(parents=True, exist_ok=True)
    with (instruction_dir / f"episode_{episode_idx:07d}.json").open("w", encoding="utf-8") as handle:
        json.dump({"seen": descriptions["seen"], "unseen": descriptions["unseen"]}, handle)
    return descriptions["seen"]


def _state_only(env):
    """Collect control state without synchronizing the renderer."""
    left = env.robot.get_left_arm_jointState()
    right = env.robot.get_right_arm_jointState()
    return np.asarray(left + right, dtype=np.float32)


class FastEpisodeBuffer:
    def __init__(self, env, visual_stride):
        self.env = env
        self.visual_stride = visual_stride
        self.capture_index = 0
        self.raw_state = []
        self.visual_raw_index = []
        self.visual_frames = []

    def take_picture(self):
        raw_index = len(self.raw_state)
        self.raw_state.append(_state_only(self.env))
        if self.capture_index % self.visual_stride == 0:
            # get_obs performs the sole render for this visual token.
            self.visual_frames.append(self.env.get_obs())
            self.visual_raw_index.append(raw_index)
        self.capture_index += 1

    def write(self, path, instructions):
        if len(self.visual_frames) < 2:
            raise RuntimeError("fast collector captured fewer than two visual observations")

        # Retain only visual pairs separated by exactly visual_stride raw states.
        valid = [
            idx
            for idx in range(len(self.visual_raw_index) - 1)
            if self.visual_raw_index[idx + 1] - self.visual_raw_index[idx] == self.visual_stride
        ]
        if not valid:
            raise RuntimeError("no complete visual/action intervals were captured")
        last = valid[-1] + 1
        frames = self.visual_frames[: last + 1]
        raw_indices = self.visual_raw_index[: last + 1]

        packed = parse_dict_structure(frames[0])
        for frame in frames:
            append_data_to_structure(packed, frame)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        create_xpolicylab_hdf5(packed, path, instructions, self.visual_stride)

        raw = np.asarray(self.raw_state, dtype=np.float32)
        blocks = np.stack(
            [raw[start + 1 : start + self.visual_stride + 1] for start in raw_indices[:-1]],
            axis=0,
        )
        with h5py.File(path, "a") as handle:
            handle.attrs["lewm_visual_stride"] = int(self.visual_stride)
            group = handle.create_group("lewm")
            group.create_dataset("visual_raw_indices", data=np.asarray(raw_indices, dtype=np.int32))
            group.create_dataset("action_blocks", data=blocks, compression="gzip")


def _load_oracle_path(source_dir, episode_idx):
    path = Path(source_dir) / "_traj_data" / f"episode{episode_idx}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def run(args):
    config_path = os.path.join(CONFIGS_PATH, f"{args.task_config}.yml")
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    cfg.update(task_name=args.task_name, task_config=args.task_config, save_path=args.output_path)
    _embodiment_args(cfg)
    embodiment_dir = re.sub(r"[^A-Za-z0-9_]+", "_", cfg["embodiment_name"]).lower()
    cfg["save_path"] = os.path.join(cfg["save_path"], cfg["task_config"], cfg["task_name"], embodiment_dir)
    cfg.update(need_plan=False, render_freq=0, save_data=True)

    seed_file = Path(args.seed_source_path) / "seed.txt"
    seeds = [int(item) for item in seed_file.read_text().split()]
    if len(seeds) < args.episode_num:
        raise ValueError(f"only {len(seeds)} saved seeds, need {args.episode_num}")

    env_module = __import__(f"envs.{args.task_name}", fromlist=[args.task_name])
    env = getattr(env_module, args.task_name)()
    output_data = Path(cfg["save_path"]) / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    for episode_idx in range(args.episode_num):
        target = output_data / f"episode_{episode_idx:07d}.hdf5"
        if target.exists():
            continue
        print(f"fast collecting episode {episode_idx}/{args.episode_num}", flush=True)
        env.setup_demo(now_ep_num=episode_idx, seed=seeds[episode_idx], **cfg)
        trajectory = _load_oracle_path(args.seed_source_path, episode_idx)
        cfg["left_joint_path"] = trajectory["left_joint_path"]
        cfg["right_joint_path"] = trajectory["right_joint_path"]
        env.set_path_lst(cfg)
        buffer = FastEpisodeBuffer(env, args.visual_stride)
        env._take_picture = buffer.take_picture
        info = env.play_once()
        assert env.check_success(), f"oracle replay failed at episode {episode_idx}"
        instructions = _instruction(args.task_name, info, episode_idx, cfg["save_path"], cfg.get("language_num"))
        buffer.write(str(target), instructions)
        env.close_env(clear_cache=((episode_idx + 1) % int(cfg["clear_cache_freq"]) == 0))
        clear_cache()


if __name__ == "__main__":
    # Match RoboTwin's native collector: this installs/validates the renderer
    # configuration before any task constructs a SAPIEN scene.
    Sapien_TEST()
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed-source-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--episode-num", type=int, required=True)
    parser.add_argument("--visual-stride", type=int, default=5)
    run(parser.parse_args())
