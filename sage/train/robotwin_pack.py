"""Pack ordered stride-1 frame latents and raw RoboTwin joint actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from sage.provenance import sha256_file

ACTION_PARTS = ("left_arm_joint_states", "left_ee_joint_states",
                "right_arm_joint_states", "right_ee_joint_states")


def source_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Episode source must be inside --data-root")
    return path


def raw_actions(path):
    with h5py.File(path, "r") as source:
        if "action/joint_states" in source:
            action = np.asarray(source["action/joint_states"], dtype=np.float32)
        else:
            action = np.concatenate([np.asarray(source[f"action/{part}"],
                                    dtype=np.float32) for part in ACTION_PARTS], axis=1)
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"Expected [T,14] actions, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Non-finite joint actions")
    return action


def fit_stats(root, episodes, output):
    total = np.zeros(14, dtype=np.float64)
    squared = np.zeros(14, dtype=np.float64)
    count = 0
    for episode in episodes:
        action = raw_actions(source_path(root, episode["source"])).astype(np.float64)
        total += action.sum(axis=0)
        squared += np.square(action).sum(axis=0)
        count += len(action)
    if count == 0:
        raise ValueError("Cannot fit action statistics on an empty split")
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-6))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=mean.astype(np.float32), std=std.astype(np.float32), count=count)
    return mean.astype(np.float32), std.astype(np.float32)


def pack(manifest_path, flat_path, root, out, stats_path, fit=False):
    manifest_path, flat_path, out = map(Path, (manifest_path, flat_path, out))
    episodes = json.loads(manifest_path.read_text())["episodes"]
    lengths = np.asarray([int(row["frames"]) for row in episodes], dtype=np.int64)
    if not len(lengths) or (lengths <= 0).any():
        raise ValueError("Every episode needs at least one frame")
    flat = np.load(flat_path, mmap_mode="r")
    if flat.ndim != 2 or int(lengths.sum()) != len(flat):
        raise ValueError("Flat latents must contain one row per manifest frame")
    # Check all sources before creating potentially large output arrays.
    for episode, length in zip(episodes, lengths):
        if len(raw_actions(source_path(root, episode["source"]))) != length:
            raise ValueError(f"Action/frame mismatch for {episode['source']}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("latents.npy", "actions.npy", "metadata.json"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    if fit:
        if Path(stats_path).exists():
            raise FileExistsError(stats_path)
        mean, std = fit_stats(root, episodes, stats_path)
    else:
        with np.load(stats_path) as stats:
            mean, std = stats["mean"], stats["std"]
    if mean.shape != (14,) or std.shape != (14,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid training action statistics")
    max_length, dim = int(lengths.max()), int(flat.shape[1])
    latents = np.lib.format.open_memmap(out / "latents.npy", mode="w+",
        dtype=np.float16, shape=(len(episodes), max_length, dim))
    actions = np.lib.format.open_memmap(out / "actions.npy", mode="w+",
        dtype=np.float16, shape=(len(episodes), max_length, 14))
    latents[:] = 0
    actions[:] = 0
    offset = 0
    for index, (episode, length) in enumerate(zip(episodes, lengths.tolist())):
        values = flat[offset:offset + length]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite latents in episode {index}")
        latents[index, :length] = values
        action = raw_actions(source_path(root, episode["source"]))
        actions[index, :length] = ((action - mean) / std).astype(np.float16)
        offset += length
    latents.flush()
    actions.flush()
    metadata = dict(episodes=len(episodes), frames=int(lengths.sum()), max_length=max_length,
        latent_dim=dim, action_dim=14, cache_stride=1, lengths=lengths.tolist(),
        manifest_sha256=sha256_file(manifest_path), flat_latents_sha256=sha256_file(flat_path),
        action_stats_sha256=sha256_file(stats_path), complete=True)
    provenance = flat_path.with_suffix('.json')
    if provenance.exists():
        encoding = json.loads(provenance.read_text())
        if encoding.get('manifest_sha256') != metadata['manifest_sha256'] or encoding.get('cache_sha256') != metadata['flat_latents_sha256']:
            raise ValueError('Frame cache provenance does not match its manifest or contents')
        metadata['encoder_checkpoint_sha256'] = encoding['checkpoint_sha256']
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--flat-latents", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--fit-action-stats", action="store_true",
                        help="Use only for the training split; validation reuses these statistics")
    args = parser.parse_args()
    print(json.dumps(pack(args.manifest, args.flat_latents, args.data_root,
        args.out_dir, args.action_stats, args.fit_action_stats)))


if __name__ == "__main__":
    main()
