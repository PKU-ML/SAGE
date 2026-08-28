import json
from pathlib import Path

from sage.provenance import semantic_manifest_hash


ROOT = Path(__file__).resolve().parents[1]


def test_paper_schedule():
    payload = json.loads((ROOT / "configs/paper.json").read_text())
    assert payload["sample_seeds"] == [32, 42, 52]
    assert payload["horizons"] == [25, 50, 75, 100, 125, 150]
    assert payload["methods"] == [
        "base_cem",
        "far_goal_prior_cem",
        "lewm_generator",
        "generator_prior_top",
        "sage",
    ]
    for horizon in payload["horizons"]:
        schedule = payload["schedule"][str(horizon)]
        assert sum(schedule) == horizon
        assert set(schedule) <= {15, 20, 25}


def test_manifests():
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
                payload = json.loads(path.read_text())
                assert payload["protocol_id"] == "sage-paper-v1"
                assert int(payload["seed"]) == seed
                assert int(payload["goal_offset_steps"]) == horizon
                assert int(payload["num_eval"]) == 50
                assert (
                    semantic_manifest_hash(payload)
                    == payload["semantic_manifest_sha256"]
                )


def test_checkpoint_registry():
    registry = json.loads((ROOT / "configs/checkpoints.json").read_text())
    assert set(registry) == {
        "pusht_generator",
        "pusht_action_prior",
        "pusht_far_action_prior",
        "cube_generator",
        "cube_action_prior",
        "cube_far_action_prior",
    }
    assert registry["pusht_generator"]["epoch"] == 6
    assert registry["cube_generator"]["epoch"] == 6
    assert registry["pusht_action_prior"]["epoch"] == 3
    assert registry["cube_action_prior"]["epoch"] == 3
    assert registry["pusht_far_action_prior"]["epoch"] == 3
    assert registry["cube_far_action_prior"]["epoch"] == 3
    for entry in registry.values():
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)
