"""Noninteractive LIBERO configuration isolated from the user's home config."""
import os
import sys
from pathlib import Path

import yaml


def configure(root, config_dir, datasets):
    root, config_dir = Path(root).resolve(), Path(config_dir).resolve()
    benchmark = root / "libero/libero"
    if not (benchmark / "bddl_files").is_dir() or not (benchmark / "assets").is_dir():
        raise FileNotFoundError(f"Incomplete LIBERO runtime: {root}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {"benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
              "init_states": str(benchmark / "init_files"), "assets": str(benchmark / "assets"),
              "datasets": str(Path(datasets).resolve())}
    destination = config_dir / "config.yaml"
    if destination.exists() and yaml.safe_load(destination.read_text()) != config:
        raise ValueError("Output directory already has a different LIBERO runtime configuration")
    destination.write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(root))
