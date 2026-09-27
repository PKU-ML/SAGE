"""Export trusted training checkpoints as path-free inference-only weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clean_metadata(manifest):
    keys = ("action_mean", "action_scale", "visual_only", "parameter_count",
            "goal_offsets", "subgoal_offset", "cache_stride")
    result = {key: manifest[key] for key in keys if key in manifest}
    sampling_keys = ("subgoal_offsets", "terminal_staircase", "allowed_local_action_modes",
                     "train_pairs", "validation_pairs")
    if "sampling" in manifest:
        result["sampling"] = {k: manifest["sampling"][k] for k in sampling_keys
                              if k in manifest["sampling"]}
    return result


def export(source, destination, role):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if role == "world_model":
        state = checkpoint
        config = json.loads(source.with_name("config.json").read_text())
        result = state
    else:
        state = checkpoint["model"]
        config = checkpoint["model_config"]
        result = {"model": state, "model_config": config,
                  "epoch": checkpoint.get("epoch"),
                  "manifest": clean_metadata(checkpoint.get("manifest", {}))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(result, destination)
    if role == "world_model":
        destination.with_name("config.json").write_text(json.dumps(config, indent=2) + "\n")
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    restored_state = restored if role == "world_model" else restored["model"]
    original_digest = state_hash(state)
    if state_hash(restored_state) != original_digest:
        raise RuntimeError("Export changed model tensors")
    return {"filename": destination.name, "role": role, "sha256": sha256(destination),
            "bytes": destination.stat().st_size, "source_sha256": sha256(source),
            "state_sha256": original_digest, "tensor_identity_verified": True,
            "selected_epoch": checkpoint.get("epoch") if role != "world_model" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for role, source, filename in (
        ("world_model", args.world_model, "world_model/weights.pt"),
        ("generator", args.generator, "generator.pt"),
        ("prior", args.prior, "prior.pt"),
    ):
        rows[filename] = export(source, args.out_dir / filename, role)
    config = args.out_dir / "world_model/config.json"
    rows["world_model/config.json"] = {"sha256": sha256(config), "bytes": config.stat().st_size}
    (args.out_dir / "export.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
