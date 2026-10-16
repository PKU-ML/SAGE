import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RoboTwinTrainingRecipeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'Requires HDF5')
    def test_collection_requires_complete_camera_action_records(self):
        import h5py
        from sage.collect_robotwin import CONFIG, TASK, verify_collection
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suffix = Path(CONFIG) / TASK / 'aloha_agilex'
            oracle = root / 'oracle_paths/train/shard_00' / suffix
            oracle.mkdir(parents=True)
            (oracle / 'seed.txt').write_text('10000000 ')
            data = root / 'data/episodes/train/shard_00' / suffix / 'data'
            data.mkdir(parents=True)
            path = data / 'episode_0000000.hdf5'
            with h5py.File(path, 'w') as episode:
                episode.attrs['lewm_visual_stride'] = 1
                episode.create_dataset('lewm/action_blocks', shape=(4, 1, 14), dtype='f4')
                for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                    episode.create_dataset(f'vision/{camera}/colors', shape=(4,), dtype='u1')
            args = SimpleNamespace(out=root, split='train', shard=0, episodes=1)
            verify_collection(args)
            with h5py.File(path, 'a') as episode:
                episode.attrs['lewm_visual_stride'] = 5
            with self.assertRaises(ValueError):
                verify_collection(args)

    @unittest.skipIf(os.name == 'nt', 'Collector resource links run on Linux')
    def test_curobo_asset_links_are_idempotent_and_do_not_rewrite_assets(self):
        from sage.collect_robotwin import link_curobo_assets
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            curobo, runtime = root / 'curobo', root / 'runtime'
            (runtime / 'assets').mkdir(parents=True)
            source = runtime / 'assets/geometry.txt'
            source.write_text('unchanged')
            link_curobo_assets(curobo, runtime)
            link_curobo_assets(curobo, runtime)
            for name in ('assets/assets', 'configs/robot/assets'):
                link = curobo / 'src/curobo/content' / name
                self.assertEqual(link.resolve(), (runtime / 'assets').resolve())
            self.assertEqual(source.read_text(), 'unchanged')
            (root / 'other/assets').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                link_curobo_assets(curobo, root / 'other')

    def test_saved_training_recipe_is_preserved(self):
        for stage, batch, steps in [('a2', 24, 4), ('b2', 72, 8)]:
            cfg = yaml.safe_load((ROOT / f'sage/train/config/robotwin_{stage}.yaml').read_text())
            self.assertEqual(cfg['trainer']['max_epochs'], 2)
            self.assertEqual(cfg['trainer']['devices'], [0, 1, 2, 3])
            self.assertEqual(cfg['loader']['batch_size'], batch)
            self.assertEqual(cfg['data']['dataset']['frameskip'], 1)
            self.assertEqual(cfg['data']['dataset']['num_steps'], steps)
            self.assertEqual(cfg['optimizer']['lr'], 5e-5)
        cfg = yaml.safe_load((ROOT / 'sage/train/config/robotwin_b2.yaml').read_text())
        self.assertEqual(cfg['freeze_modules'], ['encoder', 'projector'])
        self.assertEqual(cfg['loss']['recursive_rollout_steps'], 5)
        self.assertEqual(cfg['loss']['recursive_rollout_weight'], .1)

    def test_loss_function_matches_original_ast(self):
        source = ast.parse((ROOT / 'sage/train/robotwin_lewm.py').read_text())
        loss = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'lejepa_forward')
        # Ignore Python-version-specific empty type parameters and dump formatting.
        def canonical(node):
            if isinstance(node, ast.AST):
                return [type(node).__name__, {k: canonical(v) for k, v in ast.iter_fields(node)
                                             if k != 'type_params'}]
            if isinstance(node, list):
                return [canonical(v) for v in node]
            return node
        digest = hashlib.sha256(json.dumps(canonical(loss), sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, '6a2710719d16b40b9965e130249a64080ed2917a1e283f6aec1376245ae0b986')

    def test_launcher_keeps_global_batch_and_frozen_encoder_cache(self):
        output = subprocess.check_output([sys.executable, '-m', 'sage.train_robotwin',
            '--data-root', 'data-root', '--out', 'run-root', '--gpus', '1,2',
            '--stages', 'a', 'cache', 'b', '--dry-run'], cwd=ROOT, text=True)
        commands = [json.loads(line) for line in output.splitlines()]
        self.assertIn('loader.batch_size=48', commands[0])
        self.assertIn('loader.batch_size=144', commands[-1])
        for command in commands[1:-1]:
            if 'sage.train.robotwin_cache' in command:
                path = command[command.index('--world-model') + 1]
                self.assertIn('robotwin_a2', path)

    def test_collection_uses_split_seed_ranges_and_stride_one(self):
        for split, seed in [('train', 10200000), ('validation', 20200000), ('test', 30200000)]:
            output = subprocess.check_output([sys.executable, '-m', 'sage.collect_robotwin',
                '--robotwin-root', 'runtime-root', '--curobo-root', 'curobo-root',
                '--out', 'data-root', '--split', split, '--shard', '2', '--dry-run'], cwd=ROOT, text=True)
            commands = [json.loads(line) for line in output.splitlines()]
            self.assertEqual(commands[0][commands[0].index('--seed-start') + 1], str(seed))
            self.assertEqual(commands[1][commands[1].index('--visual-stride') + 1], '1')

    def test_collection_bundle_has_no_unregistered_or_corrupt_files(self):
        inventory = json.loads((ROOT / 'runtime/robotwin/collection_files.json').read_text())
        folder = ROOT / 'runtime/robotwin/collection'
        self.assertEqual(set(inventory), {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()})
        for record in inventory.values():
            content = (ROOT / record['bundled_file']).read_bytes()
            self.assertEqual(len(content), record['bytes'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), record['sha256'])


if __name__ == '__main__':
    unittest.main()
