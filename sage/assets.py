"""SHA-verified installation from Hugging Face or a local release mirror."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

from sage.provenance import sha256_file


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe asset name: {name}")
    return Path(*path.parts)


def install(registry, out_dir, *, repo_id=None, revision=None, source_dir=None,
            verify_only=False, project_root=None):
    out_dir = Path(out_dir).resolve()
    for name, entry in registry.items():
        relative = safe_relative(entry.get("filename", name))
        target = out_dir / relative
        if not target.resolve().is_relative_to(out_dir):
            raise ValueError(f"Asset escapes destination: {name}")
        if target.is_file():
            if sha256_file(target) != entry["sha256"]:
                raise RuntimeError(f"{relative}: existing file has the wrong SHA256")
            print(f"verified {relative}")
            continue
        if verify_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        if temporary.is_symlink():
            raise ValueError(f"Refusing symlink temporary file: {temporary}")
        if "bundled_file" in entry:
            source = Path(project_root) / safe_relative(entry["bundled_file"])
            shutil.copyfile(source, temporary)
        elif source_dir is not None:
            shutil.copyfile(Path(source_dir) / relative, temporary)
        else:
            if entry.get("publication_status") == "local_only":
                raise RuntimeError(f"{relative} is staged locally, not yet published. "
                                   "Use --source-dir with the local release mirror.")
            from huggingface_hub import hf_hub_download
            source = hf_hub_download(repo_id=repo_id or entry["repo_id"],
                                     filename=relative.as_posix(),
                                     repo_type=entry.get("repo_type", "model"),
                                     revision=revision or entry.get("revision", "main"))
            shutil.copyfile(source, temporary)
        if sha256_file(temporary) != entry["sha256"]:
            raise RuntimeError(f"{relative}: downloaded/copied file has the wrong SHA256")
        temporary.replace(target)
        print(f"installed {relative}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--registry", type=Path, default=root / "configs/assets.json")
    parser.add_argument("--suite", help="Suite name from configs/suites.json")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if args.suite:
        suite = json.loads((root / "configs/suites.json").read_text())[args.suite]
        names = [suite[k] for k in ("world_model", "generator", "prior", "far_prior") if k in suite]
        names += suite.get("companion_assets", [])
        registry = {name: registry[name] for name in names}
    install(registry, args.out_dir, repo_id=args.repo_id, revision=args.revision,
            source_dir=args.source_dir, verify_only=args.verify_only, project_root=root)


if __name__ == "__main__":
    main()
