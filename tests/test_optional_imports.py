import subprocess
import sys
import unittest
import importlib.util


class OptionalImportTests(unittest.TestCase):
    def test_package_import_does_not_load_simulators(self):
        code = '''import sys
import stable_worldmodel
import stable_worldmodel.wm
for name in ('torch', 'sapien', 'mujoco', 'lancedb', 'stable_pretraining', 'gymnasium'):
    assert name not in sys.modules, name
assert 'World' in stable_worldmodel.__all__
assert 'MultiViewLeWM' in stable_worldmodel.wm.__all__
try:
    stable_worldmodel.nonexistent
except AttributeError:
    pass
else:
    raise AssertionError('unknown attribute was not rejected')
'''
        subprocess.run([sys.executable, '-c', code], check=True)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
        ('gymnasium', 'imageio', 'pygame', 'pymunk', 'ogbench')),
        'PushT/Cube core optional dependencies required')
    def test_world_registers_environments_before_make(self):
        code = '''from unittest.mock import patch
import gymnasium as gym
from stable_worldmodel.world.world import _make_env
with patch.object(gym, 'make') as make:
    _make_env('swm/PushT-v1', 100, [], add_pixels=False)
    make.assert_called_once_with('swm/PushT-v1', max_episode_steps=100)
assert 'swm/PushT-v1' in gym.envs.registry
assert 'swm/OGBCube-v0' in gym.envs.registry
'''
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
