import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sage.reproduce import build_command, verify_asset

ROOT = Path(__file__).resolve().parents[1]


def test_all_suites_have_locked_assets():
    suites = json.loads((ROOT / "configs/suites.json").read_text())
    assets = json.loads((ROOT / "configs/assets.json").read_text())
    assert len(suites) == 9
    for suite in suites.values():
        for key in ("world_model", "generator", "prior", "far_prior"):
            if key in suite:
                assert len(assets[suite[key]]["sha256"]) == 64
        if suite.get("engine") == "native":
            assert (ROOT / "data/manifests" / suite["benchmark"] / "tail.json").is_file()
        else:
            assert (ROOT / suite["config"]).is_file()
        if suite["world_model"].endswith(".ckpt"):
            assert suite["world_model"].endswith("_object.ckpt")
        for companion in suite.get("companion_assets", []):
            assert companion in assets
            verify_asset(ROOT / assets[companion]["bundled_file"], assets[companion]["sha256"])


def test_base_never_loads_a_prior():
    suites = json.loads((ROOT / "configs/suites.json").read_text())
    for suite in suites.values():
        if suite.get("engine") == "native":
            continue
        cmd = build_command(suite, "base_cem", 32, 50, dataset="dataset",
                            checkpoints=Path("checkpoints"), output=Path("results"), device="cpu")
        assert "--generator" not in cmd
        assert "--action-prior" not in cmd
        assert "--action-stats" in cmd


def test_asset_hash_is_checked():
    with TemporaryDirectory() as root:
        path = Path(root) / "weights.pt"
        path.write_bytes(b"wrong weights")
        try:
            verify_asset(path, "0" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupted checkpoints must not be accepted")
