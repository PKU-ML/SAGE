"""Run the locked native manipulation protocols with local dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from sage.assets import install


def main():
    root = Path(__file__).resolve().parents[1]
    suites = json.loads((root / "configs/suites.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("libero_scene2", "libero_caddy", "robotwin_a2b"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--methods", nargs="+", choices=("base", "prior_top", "sage"), default=["base", "prior_top", "sage"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=(30, 60, 90, 120), default=[30, 60, 90, 120])
    parser.add_argument("--full-episode", action="store_true")
    parser.add_argument("--groups", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--robotwin-root", type=Path)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--handoff-anchors", type=Path, help="LIBERO prefix-anchor index for the selected manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = suites[args.suite]
    robotwin = args.suite == "robotwin_a2b"
    if robotwin and (args.full_episode or args.robotwin_root is None):
        parser.error("RoboTwin A2B is a tail-only suite and requires --robotwin-root")
    if robotwin and args.handoff_anchors:
        parser.error("Handoff anchors are specific to LIBERO")
    if not robotwin and args.libero_root is None:
        parser.error("LIBERO requires --libero-root pointing to its pinned checkout")
    registry = json.loads((root / "configs/assets.json").read_text())
    names = [suite[k] for k in ("world_model", "generator", "prior")] + suite.get("companion_assets", [])
    if not args.dry_run:
        data_registry = json.loads((root / 'data/manifests' / args.suite / 'assets.json').read_text())
        install(data_registry, args.data_root, verify_only=True, project_root=root)
        install({n: registry[n] for n in names}, args.checkpoints, verify_only=True, project_root=root)
    bundle = (args.checkpoints / args.suite).resolve()
    queries = root / "data/manifests" / args.suite / ("full.json" if args.full_episode else "tail.json")
    for method in args.methods:
        for horizon in ([None] if args.full_episode else args.horizons):
            for group in args.groups:
                name = "full" if horizon is None else f"h{horizon}"
                output = (args.out / method / name / f"group{group}.json").resolve()
                if output.exists() and not args.dry_run and not robotwin:
                    raise FileExistsError(f"Use a fresh output directory: {output}")
                if robotwin:
                    command = [sys.executable, "-m", "sage.eval.robotwin",
                        "--robotwin-root", str(args.robotwin_root.resolve()),
                        "--query-bank", str(queries), "--data-root", str(args.data_root.resolve()),
                        "--lewm-checkpoint", str(bundle / "world_model/weights.pt"),
                        "--generator-checkpoint", str(bundle / "generator.pt"),
                        "--prior-checkpoint", str(bundle / "prior.pt"),
                        "--action-stats", str(root / "data/stats/native/robotwin_a2b.npz"),
                        "--controller", "gaussian_lewm" if method == "base" else method,
                        "--max-new-queries", "8",
                        "--horizon", str(horizon), "--seed", str(42 + group)]
                else:
                    command = [sys.executable, "-m", "sage.eval.libero", "--suite", args.suite,
                        "--libero-root", str(args.libero_root.resolve()),
                        "--queries", str(queries), "--data-root", str(args.data_root.resolve()),
                        "--bundle", str(bundle), "--method", method,
                        "--seed", str(20260829 if args.suite == "libero_scene2" else 42)]
                    if horizon is not None:
                        command += ["--horizon", str(horizon)]
                    if args.handoff_anchors:
                        command += ["--handoff-anchors", str(args.handoff_anchors.resolve())]
                command += ["--query-start", str(group * 50), "--num-queries", "50",
                            "--device", args.device, "--out", str(output)]
                print(json.dumps(command), flush=True)
                if not args.dry_run:
                    previous_count = -1
                    while True:
                        subprocess.run(command, cwd=root, check=True)
                        if not robotwin:
                            break
                        completed = len(json.loads(output.read_text())['results'])
                        if completed == 50:
                            break
                        if completed <= previous_count or completed > 50:
                            raise RuntimeError(f'Evaluation did not make valid progress: {output}')
                        previous_count = completed


if __name__ == "__main__":
    main()
