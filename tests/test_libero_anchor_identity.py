import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sage.runtime.libero_anchor import restore_anchor


class IdentityTests(unittest.TestCase):
    def attempt(self, metadata, context, digest=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'anchor.npz'
            np.savez(path, metadata=np.asarray(json.dumps(metadata)))
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict('sys.modules', {'mujoco': SimpleNamespace(__version__='2.3.7')}):
                restore_anchor(None, path, context, digest or checksum)

    def test_changed_archive_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.attempt({}, {}, '0' * 64)

    def test_different_query_rejected_before_restore(self):
        with self.assertRaisesRegex(ValueError, 'different query'):
            self.attempt({'schema': 1, 'context': {'query': 28}}, {'query': 29})

    def test_different_simulator_rejected_before_restore(self):
        with self.assertRaisesRegex(ValueError, 'simulator/model'):
            self.attempt({'schema': 1, 'context': {}, 'mujoco': 'other'}, {})


if __name__ == '__main__':
    unittest.main()
