"""Validate or relocate the native benchmark's locked query dependencies.

Paths in a public manifest are relative to --data-root. Query order and
query_index are preserved because they are part of the planner RNG contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sage.assets import safe_relative
from sage.provenance import sha256_file


FILE_KEYS = {"source_shard", "episode_hdf5", "trajectory", "path"}
DIRECTORY_KEYS = {"source_dir"}


def resolve_path(root, value):
    root = Path(root).resolve()
    path = (root / safe_relative(value)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dependency escapes the data root: {value}")
    return path


def walk_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FILE_KEYS | DIRECTORY_KEYS and isinstance(item, str):
                yield key, item, value.get("sha256") if key == "path" else None
            else:
                yield from walk_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_paths(item)


def validate_queries(manifest, root):
    queries = manifest["queries"]
    checked = {}
    horizons = {}
    for query in queries:
        horizon = str(query.get("horizon_raw_actions", query.get("horizon", "full")))
        horizons[horizon] = horizons.get(horizon, 0) + 1
        for key, value, expected in walk_paths(query):
            path = resolve_path(root, value)
            if key in DIRECTORY_KEYS:
                if not path.is_dir():
                    raise FileNotFoundError(path)
                continue
            if not path.is_file():
                raise FileNotFoundError(path)
            if value not in checked:
                checked[value] = sha256_file(path) if expected else None
            if expected and checked[value] != expected:
                raise RuntimeError(f"Dependency SHA mismatch: {value}")
    return {"queries": len(queries), "horizon_counts": horizons,
            "unique_files": len(checked), "dependency_files_present": True,
            "replay_certified": False}


def inspect_hdf5(path):
    import h5py
    path = Path(path).resolve()
    dependencies = set()
    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.is_virtual:
                for source in obj.virtual_sources():
                    filename = source.file_name
                    if isinstance(filename, bytes):
                        filename = filename.decode()
                    p = Path(filename)
                    if p.is_absolute():
                        raise ValueError(f"Non-portable VDS source: {name}: {filename}")
                    target = (path.parent / p).resolve()
                    if not target.is_file():
                        raise FileNotFoundError(target)
                    with h5py.File(target, "r") as shard:
                        if source.dset_name not in shard:
                            raise KeyError(f"Missing VDS dataset {source.dset_name} in {target.name}")
                    dependencies.add(filename)
        handle.visititems(visit)
    return {"vds_dependencies": sorted(dependencies), "portable_sources": True}


def materialize(manifest, root):
    def transform(value):
        if isinstance(value, dict):
            return {k: str(resolve_path(root, v)) if k in FILE_KEYS | DIRECTORY_KEYS
                    and isinstance(v, str) else transform(v) for k, v in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        return value
    return transform(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--materialize", type=Path,
                        help="Write a private relocated copy for a native evaluator")
    args = parser.parse_args()
    report = {}
    if args.queries:
        manifest = json.loads(args.queries.read_text())
        report.update(validate_queries(manifest, args.data_root))
        if args.materialize:
            if args.materialize.exists():
                raise FileExistsError(args.materialize)
            args.materialize.parent.mkdir(parents=True, exist_ok=True)
            args.materialize.write_text(json.dumps(materialize(manifest, args.data_root), indent=2) + "\n")
    if args.index:
        report.update(inspect_hdf5(args.index))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
