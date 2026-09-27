"""Check native simulator ABI versions without loading a model or a dataset."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("libero", "robotwin"))
    parser.add_argument("--gpu", action="store_true", help="Bounded NVIDIA driver probe before simulator import")
    parser.add_argument("--numerics", action="store_true", help="Record NumPy BLAS dispatch; package versions alone do not certify replay")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "runtime/native_contracts.json").read_text())[args.suite]
    mismatches = []
    if not platform.python_version().startswith(contract["python"] + "."):
        mismatches.append(f"Python {platform.python_version()} != {contract['python']}")
    observed = {}
    for name, expected in contract["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        observed[name] = actual
        if actual.split("+")[0] != expected:
            mismatches.append(f"{name}: {actual} != {expected}")
    driver = None
    numerics = None
    if args.numerics:
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import numpy as np; np.show_runtime()"],
                capture_output=True, text=True, timeout=20, check=True,
            )
            numerics = probe.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            mismatches.append(f"Numerical runtime probe failed: {error}")
    if args.gpu:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10, check=True)
            driver = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            mismatches.append(f"NVIDIA driver probe failed: {error}. SAPIEN calls nvidia-smi during import; repair the driver before running.")
    print(json.dumps({"observed": observed, "mismatches": mismatches, "driver": driver,
                      "numerical_runtime": numerics,
                      "renderer_tested": False, "replay_certified": False}, indent=2))
    raise SystemExit(bool(mismatches))


if __name__ == "__main__":
    main()
