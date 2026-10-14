import json
from pathlib import Path
import unittest

from sage.assets import safe_relative


ROOT = Path(__file__).resolve().parents[1]
SUITES = ('libero_scene2', 'libero_caddy', 'robotwin_a2b')


class PublicNativeRegistryTests(unittest.TestCase):
    def test_all_native_splits_have_public_download_records(self):
        archives = json.loads((ROOT / 'data/archives.json').read_text())
        expected = {f'{suite}_{split}.tar' for suite in SUITES
                    for split in ('train', 'validation', 'evaluation')}
        self.assertEqual(set(archives), expected)
        for name, record in archives.items():
            self.assertEqual(record['filename'], name)
            self.assertEqual(record['repo_id'], 'CLTRAY/SAGE-data')
            self.assertEqual(record['repo_type'], 'dataset')
            self.assertEqual(len(record['revision']), 40)
            self.assertEqual(len(record['sha256']), 64)
            if name.endswith('_train.tar'):
                self.assertEqual(record['publication_status'], 'published_in_parts')
                self.assertEqual(sum(p['bytes'] for p in record['parts']), record['bytes'])
                self.assertEqual(len({p['hf_filename'] for p in record['parts']}), len(record['parts']))
                safe_relative(record['parts_manifest'])
                for part in record['parts']:
                    self.assertEqual(part['revision'], record['revision'])
                    self.assertEqual(part['publication_status'], 'published')
                    self.assertEqual(len(part['sha256']), 64)
                    safe_relative(part['hf_filename'])
            else:
                self.assertEqual(record['publication_status'], 'published')
                safe_relative(record['hf_filename'])

    def test_evaluation_files_reference_their_public_archive(self):
        for suite, count in zip(SUITES, (39, 6, 750)):
            inventory = json.loads((ROOT / 'data/manifests' / suite / 'assets.json').read_text())
            self.assertEqual(len(inventory), count)
            for name, record in inventory.items():
                self.assertEqual(record['filename'], name)
                self.assertEqual(record['publication_status'], 'published_in_archive')
                self.assertEqual(record['archive_registry'], 'data/archives.json')
                self.assertEqual(record['archive'], f'{suite}_evaluation.tar')
                self.assertEqual(len(record['sha256']), 64)
                safe_relative(name)


if __name__ == '__main__':
    unittest.main()
