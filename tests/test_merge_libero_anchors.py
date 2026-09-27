import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

source = Path(__file__).resolve().parents[1] / 'scripts/merge_libero_anchors.py'
spec = importlib.util.spec_from_file_location('merge_anchors', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MergeAnchorsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.queries = self.root / 'queries.json'
        self.rows = [{'query_index': 0}, {'query_index': 1}]
        self.queries.write_text(json.dumps({'suite': 'test', 'queries': self.rows}))
        self.indices = []
        for i, row in enumerate(self.rows):
            folder = self.root / f'shard{i}'
            folder.mkdir()
            anchor = folder / 'state.npz'
            anchor.write_bytes(bytes([i]))
            key = hashlib.sha256(json.dumps({'suite': 'test', 'query': row},
                                           sort_keys=True).encode()).hexdigest()
            index = folder / 'index.json'
            index.write_text(json.dumps({'schema': 1,
                'query_manifest_sha256': module.sha256_file(self.queries),
                'anchors': {key: {'path': anchor.name,
                    'sha256': module.sha256_file(anchor), 'source_prefix_proprio_exact': True}}}))
            self.indices.append(index)

    def test_complete(self):
        output = self.root / 'index.json'
        self.assertEqual(module.merge(self.queries, self.indices, output), 2)
        entries = json.loads(output.read_text())['anchors'].values()
        self.assertTrue(all(entry['path'].startswith('shard') for entry in entries))

    def test_missing(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            module.merge(self.queries, self.indices[:1], self.root / 'index.json')

    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            module.merge(self.queries, self.indices * 2, self.root / 'index.json')

    def test_corruption(self):
        (self.root / 'shard0/state.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            module.merge(self.queries, self.indices, self.root / 'index.json')


if __name__ == '__main__':
    unittest.main()
