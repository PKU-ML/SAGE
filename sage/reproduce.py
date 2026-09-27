"""Launch the locked PushT/Cube paper protocols without machine-specific paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("base_cem", "far_goal_prior_cem", "lewm_generator",
           "generator_prior_top", "final_goal_scoring", "sage")


def verify_asset(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")


def build_command(suite, method, seed, horizon, *, dataset, checkpoints, output, device):
    benchmark = suite["benchmark"]
    command = [sys.executable, "-m", f"sage.eval.{benchmark}",
               "--method", method, "--dataset", str(dataset),
               "--policy", str(checkpoints / suite["world_model"]),
               "--manifest", str(ROOT / f"data/manifests/{benchmark}/seed{seed}/h{horizon}.json"),
               "--paper-config", str(ROOT / suite["config"]),
               "--action-stats", str(ROOT / f"data/stats/{benchmark}_train_seed42.json"),
               "--seed", str(seed), "--device", device,
               "--out-dir", str(output / method / f"seed{seed}" / f"h{horizon}")]
    if method in {"lewm_generator", "generator_prior_top", "final_goal_scoring", "sage"}:
        command += ["--generator", str(checkpoints / suite["generator"])]
    if method in {"generator_prior_top", "final_goal_scoring", "sage"}:
        command += ["--action-prior", str(checkpoints / suite["prior"])]
    elif method == "far_goal_prior_cem":
        command += ["--action-prior", str(checkpoints / suite["far_prior"])]
    return command


def main():
    suites = json.loads((ROOT / "configs/suites.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=sorted(suites))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[32, 42, 52], choices=[32, 42, 52])
    parser.add_argument("--horizons", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150],
                        choices=[25, 50, 75, 100, 125, 150])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = suites[args.suite]
    if suite.get("engine") == "native":
        parser.error("Use python -m sage.reproduce_native for LIBERO/RoboTwin suites")
    methods = args.methods or suite["methods"]
    if any(method not in suite["methods"] for method in methods):
        parser.error("The requested method is not part of this paper suite")
    assets = json.loads((ROOT / "configs/assets.json").read_text())
    required = {suite["world_model"]}
    required.update(suite.get("companion_assets", []))
    for method in methods:
        if method in {"lewm_generator", "generator_prior_top", "final_goal_scoring", "sage"}:
            required.add(suite["generator"])
        if method in {"generator_prior_top", "final_goal_scoring", "sage"}:
            required.add(suite["prior"])
        if method == "far_goal_prior_cem":
            required.add(suite["far_prior"])
    if not args.dry_run:
        for filename in sorted(required):
            bundled = assets[filename].get("bundled_file")
            destination = args.checkpoints / filename
            if bundled and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / bundled, destination)
            verify_asset(args.checkpoints / filename, assets[filename]["sha256"])
    # Never silently reuse an output from a different checkpoint or protocol.
    for method in methods:
        for seed in args.seeds:
            for horizon in args.horizons:
                destination = args.out / method / f"seed{seed}" / f"h{horizon}"
                if not args.dry_run and destination.exists() and any(destination.iterdir()):
                    raise FileExistsError(f"Choose a new output directory: {destination}")
                command = build_command(suite, method, seed, horizon,
                                        dataset=args.dataset, checkpoints=args.checkpoints.resolve(),
                                        output=args.out.resolve(), device=args.device)
                print(json.dumps(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
