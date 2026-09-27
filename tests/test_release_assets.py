import hashlib
import tempfile
import unittest
from pathlib import Path

from sage.assets import install, safe_relative
from sage.native_data import materialize, validate_queries


class ReleaseAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_asset_rejects_absolute_or_escaping_paths(self):
        for name in ["/tmp/a", "../a", "a/../../b", "C:/a", "a\\b", ""]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)

    def test_local_asset_install_and_hash_check(self):
        source = self.root / "source"
        source.mkdir()
        (source / "a.pt").write_bytes(b"checkpoint")
        registry = {"a.pt": {"sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                              "publication_status": "local_only"}}
        install(registry, self.root / "dest", source_dir=source)
        install(registry, self.root / "dest", verify_only=True)
        (self.root / "dest/a.pt").write_bytes(b"wrong")
        with self.assertRaises(RuntimeError):
            install(registry, self.root / "dest", source_dir=source)

    def test_unpublished_asset_cannot_silently_download(self):
        with self.assertRaisesRegex(RuntimeError, "not yet published"):
            install({"a.pt": {"sha256": "0" * 64, "publication_status": "local_only"}}, self.root)

    def test_query_relocation_preserves_order_indices_and_goal(self):
        (self.root / "episode.h5").write_bytes(b"fixture")
        manifest = {"queries": [{"source_shard": "episode.h5", "query_index": 17,
                                   "start": 50, "target": 80, "horizon": 30}]}
        report = validate_queries(manifest, self.root)
        self.assertEqual(report["horizon_counts"], {"30": 1})
        self.assertFalse(report["replay_certified"])
        relocated = materialize(manifest, self.root)
        self.assertEqual(relocated["queries"][0]["source_shard"], str(self.root / "episode.h5"))
        self.assertEqual(relocated["queries"][0]["query_index"], 17)
        self.assertEqual(manifest["queries"][0]["source_shard"], "episode.h5")

    def test_query_missing_dependency_is_not_ignored(self):
        with self.assertRaises(FileNotFoundError):
            validate_queries({"queries": [{"trajectory": "missing.pkl"}]}, self.root)

if __name__ == "__main__":
    unittest.main()
