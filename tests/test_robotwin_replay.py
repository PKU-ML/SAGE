import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from sage.runtime.robotwin_replay import inspect_handoff


class RobotwinReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "episode.hdf5"
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode('.jpg', self.image)
        self.assertTrue(ok)
        with h5py.File(self.path, 'w') as handle:
            handle.create_dataset('lewm/action_blocks', data=np.zeros((2, 1, 14), np.float32))
            frames = handle.create_dataset('camera', (3,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            for index in range(3):
                frames[index] = jpeg

    def inspect(self, command=None, image=None):
        return inspect_handoff(self.path, 1, np.zeros(14) if command is None else command,
                               {'pixels': self.image if image is None else image}, {'pixels': 'camera'})

    def test_exact_command_does_not_claim_render_certificate(self):
        report = self.inspect()
        self.assertTrue(report['joint_command_exact'])
        self.assertFalse(report['render_certificate_passed'])
        self.assertEqual(report['image_differences']['pixels']['mae_255'], 0)

    def test_command_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'joint-command mismatch'):
            self.inspect(command=np.ones(14))

    def test_image_difference_is_reported(self):
        report = self.inspect(image=np.full_like(self.image, 10))
        self.assertEqual(report['image_differences']['pixels']['mae_255'], 10)

    def test_wrong_image_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
            self.inspect(image=self.image[:4])


if __name__ == '__main__':
    unittest.main()
