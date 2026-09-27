import random
from types import SimpleNamespace
import unittest

import numpy as np
from sage.runtime.libero_anchor import observe_without_advancing_clock


class ClockTests(unittest.TestCase):
    def test_refresh_preserves_clock_and_rng(self):
        sensor = SimpleNamespace(_time_since_last_sample=0.008, _current_delay=0.0, _sampled=True)
        def refresh(force_update):
            self.assertTrue(force_update)
            sensor._time_since_last_sample += 0.002
            sensor._sampled = False
            np.random.random()
            random.random()
            return {'fresh_image': 1}
        env = SimpleNamespace(_observables={'camera': sensor}, _get_observations=refresh)
        np.random.seed(11); random.seed(11)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        self.assertEqual(observe_without_advancing_clock(env), {'fresh_image': 1})
        self.assertEqual(sensor._time_since_last_sample, 0.008)
        self.assertTrue(sensor._sampled)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        self.assertEqual(random.getstate(), python_state)

    def test_clock_restored_even_if_render_fails(self):
        sensor = SimpleNamespace(_time_since_last_sample=0.008, _current_delay=0.0, _sampled=True)
        def fail(force_update):
            sensor._time_since_last_sample = 1.0
            raise RuntimeError('renderer failed')
        env = SimpleNamespace(_observables={'camera': sensor}, _get_observations=fail)
        with self.assertRaisesRegex(RuntimeError, 'renderer failed'):
            observe_without_advancing_clock(env)
        self.assertEqual(sensor._time_since_last_sample, 0.008)


if __name__ == '__main__':
    unittest.main()
