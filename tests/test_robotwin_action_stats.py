import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sage.assets import install


ROOT = Path(__file__).resolve().parents[1]
SHA256 = 'b8921f1b24e2c808e7fdbb93823a90d379d80dc1e638940416dd50290ec1774d'


class RoboTwinActionStatsTests(unittest.TestCase):
    def test_paper_stats_are_bundled_and_registered(self):
        assets = json.loads((ROOT / 'configs/assets.json').read_text())
        suite = json.loads((ROOT / 'configs/suites.json').read_text())['robotwin_a2b']
        name = 'robotwin_a2b/action_stats.npz'
        self.assertIn(name, suite['companion_assets'])
        record = assets[name]
        source = ROOT / record['bundled_file']
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SHA256)
        self.assertEqual(record['sha256'], SHA256)
        self.assertEqual(source.stat().st_size, record['bytes'])
        self.assertEqual(record['publication_status'], 'published')
        self.assertEqual(record['filename'], name)
        with np.load(source, allow_pickle=False) as stats:
            self.assertEqual(set(stats.files), {'mean', 'std', 'count'})
            for key in ('mean', 'std'):
                self.assertEqual(stats[key].shape, (14,))
                self.assertEqual(stats[key].dtype, np.dtype('float32'))
                self.assertTrue(np.isfinite(stats[key]).all())
            self.assertTrue((stats['std'] > 0).all())
            self.assertEqual(stats['count'].dtype, np.dtype('int64'))
            self.assertEqual(stats['count'].shape, ())
            self.assertEqual(int(stats['count']), 1188823)

    def test_install_and_verify_without_a_private_mirror(self):
        name = 'robotwin_a2b/action_stats.npz'
        record = json.loads((ROOT / 'configs/assets.json').read_text())[name]
        with tempfile.TemporaryDirectory() as folder:
            install({name: record}, folder, project_root=ROOT)
            install({name: record}, folder, project_root=ROOT, verify_only=True)
            target = Path(folder) / name
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), SHA256)


if __name__ == '__main__':
    unittest.main()
