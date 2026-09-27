import json
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from sage.build_native_index import build_index
from sage.provenance import sha256_file


class PortableIndexTest(unittest.TestCase):
    def test_relocation_preserves_order_and_episode_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'original'
            root.mkdir()
            records = []
            for i in range(2):
                path = root / f'shard{i}.h5'
                with h5py.File(path, 'w') as f:
                    f['pixels'] = np.full((5, 4, 4, 3), i + 1, dtype=np.uint8)
                    f['action'] = np.full((5, 7), i + 1, dtype=np.float32)
                    f['ep_len'] = [2, 3]
                    f['ep_offset'] = [0, 2]
                records.append({'path': path.name, 'sha256': sha256_file(path)})
            manifest = root / 'shards.json'
            manifest.write_text(json.dumps({'shards': records}))
            build_index(manifest, root, root / 'indices/train.h5')
            moved = Path(folder) / 'relocated'
            shutil.move(str(root), moved)
            with h5py.File(moved / 'indices/train.h5', 'r') as f:
                np.testing.assert_array_equal(f['ep_offset'][:], [0, 2, 5, 7])
                np.testing.assert_array_equal(f['action'][:, 0], [1]*5 + [2]*5)
                self.assertTrue(all(not Path(s.file_name).is_absolute() for s in f['pixels'].virtual_sources()))
            with self.assertRaises(FileExistsError):
                build_index(moved / 'shards.json', moved, moved / 'indices/train.h5')


if __name__ == '__main__':
    unittest.main()
