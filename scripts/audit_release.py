#!/usr/bin/env python3
"""Fail fast on protocol drift or private/development artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from sage.provenance import semantic_manifest_hash, sha256_file  # noqa: E402

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH_PARTS = {
    "_source",
    "__pycache__",
    ".pytest_cache",
    "backups",
}
FORBIDDEN_PATTERNS = {
    "private path": re.compile(r"/data[0-9]+/ltchen|C:\\\\Users\\\\ccclt", re.I),
    "private host": re.compile(r"222\\.29\\.(?:2\\.246|136\\.18)", re.I),
    "development naming": re.compile(
        r"cherry.?pick|best[_ -]?seed|INVALID_|bak_20\\d{6}|watch_20\\d{6}",
        re.I,
    ),
}


def audit_tree() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if any(part in FORBIDDEN_PATH_PARTS for part in path.parts):
            errors.append(f"forbidden path: {path.relative_to(ROOT)}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {path.relative_to(ROOT)}")
    return errors


def audit_protocol() -> list[str]:
    errors: list[str] = []
    config = json.loads(
        (ROOT / "configs" / "paper.json").read_text(encoding="utf-8")
    )
    if config["sample_seeds"] != [32, 42, 52]:
        errors.append("paper sample seeds must be exactly [32, 42, 52]")
    if config.get("methods") != [
        "base_cem",
        "far_goal_prior_cem",
        "lewm_generator",
        "generator_prior_top",
        "sage",
    ]:
        errors.append("paper method list changed")
    if config["planner"] != {
        "candidates": 300,
        "cem_rounds": 30,
        "elites": 30,
        "variance_scale": 1.0,
        "action_block": 5,
        "history_length": 3,
        "frameskip": 5,
        "precision": "bf16",
    }:
        errors.append("canonical planner configuration changed")

    for benchmark in ("pusht", "cube"):
        for seed in (32, 42, 52):
            for horizon in (25, 50, 75, 100, 125, 150):
                path = (
                    ROOT
                    / "data"
                    / "manifests"
                    / benchmark
                    / f"seed{seed}"
                    / f"h{horizon}.json"
                )
                if not path.is_file():
                    errors.append(f"missing manifest: {path.relative_to(ROOT)}")
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload.get("seed", -1)) != seed:
                    errors.append(f"manifest seed mismatch: {path.relative_to(ROOT)}")
                actual = semantic_manifest_hash(payload)
                expected = payload.get("semantic_manifest_sha256")
                if actual != expected:
                    errors.append(
                        f"manifest hash mismatch: {path.relative_to(ROOT)}"
                    )
                if int(payload["num_eval"]) != 50:
                    errors.append(f"manifest is not n=50: {path.relative_to(ROOT)}")

    checksum_path = ROOT / "data" / "manifests" / "SHA256SUMS"
    checksum_lines = checksum_path.read_text(encoding="ascii").splitlines()
    if len(checksum_lines) != 36:
        errors.append("manifest SHA256SUMS must contain exactly 36 entries")
    for line in checksum_lines:
        expected, relative = line.split("  ", maxsplit=1)
        path = ROOT / "data" / "manifests" / relative
        if not path.is_file():
            errors.append(f"checksum references missing manifest: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"manifest byte hash mismatch: {relative}")
    return errors


def audit_training() -> list[str]:
    errors: list[str] = []
    config = json.loads(
        (ROOT / "configs" / "training.json").read_text(encoding="utf-8")
    )
    expected_shared = {
        "batch_size": 128,
        "bf16": True,
        "frameskip": 5,
        "goal_offsets": [
            15, 20, 25, 30, 40, 45, 50, 60,
            65, 75, 90, 100, 115, 125, 140, 150,
        ],
        "hidden_dim": 512,
        "history_length": 3,
        "learning_rate": 0.0001,
        "num_heads": 8,
        "num_workers": 4,
        "pin_memory": False,
        "weight_decay": 0.0001,
    }
    if config.get("shared") != expected_shared:
        errors.append("canonical shared training configuration changed")

    expected_benchmark_fields = {
        "pusht": {
            "allow_repeats": False,
            "max_train_windows": 0,
            "max_val_windows": 0,
            "generator_seed": 148,
            "prior_seed": 149,
        },
        "cube": {
            "allow_repeats": True,
            "max_train_windows": 200000,
            "max_val_windows": 20000,
            "generator_seed": 422,
            "prior_seed": 423,
        },
    }
    for benchmark, expected in expected_benchmark_fields.items():
        generator = config.get(benchmark, {}).get("generator", {})
        prior = config.get(benchmark, {}).get("action_prior", {})
        far_prior = config.get(benchmark, {}).get("far_action_prior", {})
        checks = {
            "allow_repeats": (
                generator.get("allow_repeats"),
                prior.get("allow_repeats"),
                expected["allow_repeats"],
            ),
            "max_train_windows": (
                generator.get("max_train_windows"),
                prior.get("max_train_windows"),
                expected["max_train_windows"],
            ),
            "max_val_windows": (
                generator.get("max_val_windows"),
                prior.get("max_val_windows"),
                expected["max_val_windows"],
            ),
            "generator_seed": (
                generator.get("seed"),
                expected["generator_seed"],
                expected["generator_seed"],
            ),
            "prior_seed": (
                prior.get("seed"),
                expected["prior_seed"],
                expected["prior_seed"],
            ),
        }
        for field, values in checks.items():
            if not all(value == values[-1] for value in values):
                errors.append(f"{benchmark} training field changed: {field}")
        if generator.get("epochs") != 6:
            errors.append(f"{benchmark} generator epoch must be 6")
        if prior.get("epochs") != 5 or prior.get("selected_epoch") != 3:
            errors.append(
                f"{benchmark} prior must train for 5 epochs and select epoch 3"
            )
        if generator.get("option_durations") != [15, 20, 25]:
            errors.append(f"{benchmark} generator durations changed")
        if prior.get("option_durations") != [15, 20, 25]:
            errors.append(f"{benchmark} prior durations changed")
        if prior.get("generated_subgoal_ratio") != 1.0:
            errors.append(f"{benchmark} generated-subgoal ratio changed")
        if far_prior.get("conditioning_goal_source") != "far":
            errors.append(f"{benchmark} far prior must use the final goal")
        if far_prior.get("generated_subgoal_ratio") != 0.0:
            errors.append(f"{benchmark} far prior cannot use generated subgoals")
        if far_prior.get("epochs") != 3 or far_prior.get("selected_epoch") != 3:
            errors.append(f"{benchmark} far prior must select epoch 3")
    return errors


def audit_method_semantics() -> list[str]:
    errors: list[str] = []
    for benchmark in ("pusht", "cube"):
        path = ROOT / "sage" / "eval" / f"{benchmark}.py"
        text = path.read_text(encoding="utf-8")
        if "class GaussianCEM" not in text:
            errors.append(f"{benchmark} is missing true Gaussian CEM")
        if "forbid warm starts" not in text:
            errors.append(f"{benchmark} Gaussian CEM does not reject warm starts")
        if '"normalization_only"' not in text:
            errors.append(f"{benchmark} does not record normalization-only priors")
    return errors


def audit_checkpoint_registry() -> list[str]:
    errors: list[str] = []
    registry = json.loads(
        (ROOT / "configs" / "checkpoints.json").read_text(encoding="utf-8")
    )
    expected_epochs = {
        "pusht_generator": 6,
        "pusht_action_prior": 3,
        "pusht_far_action_prior": 3,
        "cube_generator": 6,
        "cube_action_prior": 3,
        "cube_far_action_prior": 3,
    }
    if set(registry) != set(expected_epochs):
        errors.append("checkpoint registry keys changed")
        return errors
    for name, epoch in expected_epochs.items():
        entry = registry[name]
        if entry.get("epoch") != epoch:
            errors.append(f"checkpoint epoch changed: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            errors.append(f"invalid checkpoint SHA-256: {name}")
        if int(entry.get("size_bytes", 0)) <= 0:
            errors.append(f"invalid checkpoint size: {name}")
    return errors


def main() -> None:
    errors = (
        audit_tree()
        + audit_protocol()
        + audit_training()
        + audit_method_semantics()
        + audit_checkpoint_registry()
    )
    if errors:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(errors))
    print("release audit: ok")


if __name__ == "__main__":
    main()
