"""Aggregate and validate the fixed PushT/Cube component table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_result(path: Path, benchmark: str, method: str, seed: int, horizon: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_kind": "paper",
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "horizon": horizon,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")
    return float(payload["metrics"]["success_rate"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--paper-config", default="configs/paper.json")
    parser.add_argument("--out", default="results/component_table.json")
    parser.add_argument("--expected-tolerance", type=float, default=2.0)
    args = parser.parse_args()

    paper = json.loads(Path(args.paper_config).read_text(encoding="utf-8"))
    seeds = [int(value) for value in paper["sample_seeds"]]
    horizons = [int(value) for value in paper["horizons"]]
    methods = list(paper["methods"])
    rows = {}
    flat_rows = []
    for benchmark in ("pusht", "cube"):
        rows[benchmark] = {}
        for method in methods:
            rows[benchmark][method] = {}
            for horizon_index, horizon in enumerate(horizons):
                values = []
                for seed in seeds:
                    path = (
                        Path(args.root)
                        / benchmark
                        / method
                        / f"seed{seed}"
                        / f"h{horizon}"
                        / "results.json"
                    )
                    values.append(load_result(path, benchmark, method, seed, horizon))
                mean = float(np.mean(values))
                expected = float(
                    paper["expected_success_percent"][benchmark][method][horizon_index]
                )
                if abs(mean - expected) > args.expected_tolerance:
                    raise ValueError(
                        f"{benchmark}/{method}/H{horizon}: mean {mean:.3f} "
                        f"differs from recorded {expected:.3f}"
                    )
                summary = {
                    "per_seed": values,
                    "mean": mean,
                    "sample_std": float(np.std(values, ddof=1)),
                    "recorded_mean": expected,
                }
                rows[benchmark][method][str(horizon)] = summary
                flat_rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "horizon": horizon,
                        "mean": mean,
                        "sample_std": summary["sample_std"],
                        **{f"seed_{seed}": value for seed, value in zip(seeds, values)},
                    }
                )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    markdown = []
    for benchmark in ("pusht", "cube"):
        markdown.extend(
            [
                f"## {benchmark}",
                "",
                "| Method | " + " | ".join(f"H{h}" for h in horizons) + " |",
                "|:--|" + "--:|" * len(horizons),
            ]
        )
        for method in methods:
            values = [rows[benchmark][method][str(h)]["mean"] for h in horizons]
            markdown.append(
                f"| {method} | " + " | ".join(f"{v:.1f}" for v in values) + " |"
            )
        markdown.append("")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {output}, {csv_path}, and {markdown_path}")


if __name__ == "__main__":
    main()
