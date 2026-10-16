"""Collect success-filtered A2B joint paths, then write raw-stride-1 episodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sage.assets import install, safe_relative
from sage.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
TASK = 'place_a2b_left'
CONFIG = 'lewm_place_a2b_left_v1'


def verify_curobo(path):
    record = json.loads((ROOT / 'runtime/robotwin/curobo.json').read_text())
    for name, expected in record['source_files'].items():
        source = path / safe_relative(name)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f'cuRobo source mismatch: {name}; install the pinned revision')


def link_curobo_assets(curobo_root, robotwin_root):
    """Expose the locked assets under cuRobo's two standard resource roots."""
    assets = (robotwin_root / 'assets').resolve()
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    content = curobo_root / 'src/curobo/content'
    for relative in ('assets/assets', 'configs/robot/assets'):
        link = content / relative
        if link.is_symlink() and link.resolve() == assets:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f'{link} already exists; use a separate cuRobo checkout')
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(assets, target_is_directory=True)


def commands(args):
    shard = f'shard_{args.shard:02d}'
    oracle = args.out / 'oracle_paths' / args.split / shard
    data = args.out / 'data/episodes' / args.split / shard
    seed = {'train': 10000000, 'validation': 20000000, 'test': 30000000}[args.split] + 100000 * args.shard
    if args.seed_start is not None:
        seed = args.seed_start
    source = oracle / CONFIG / TASK / 'aloha_agilex'
    return [
        [sys.executable, 'scripts/collect_data.py', TASK, CONFIG,
         '--seed-start', str(seed), '--episode-num', str(args.episodes), '--save-path', str(oracle)],
        [sys.executable, 'scripts/collect_data_fast_lewm.py', TASK, CONFIG,
         '--seed-source-path', str(source), '--output-path', str(data),
         '--episode-num', str(args.episodes), '--visual-stride', '1'],
    ]


def verify_collection(args):
    import h5py

    shard = f'shard_{args.shard:02d}'
    suffix = Path(CONFIG) / TASK / 'aloha_agilex'
    source = args.out / 'oracle_paths' / args.split / shard / suffix
    seeds = [int(s) for s in (source / 'seed.txt').read_text().split()]
    if len(seeds) < args.episodes:
        raise ValueError('Oracle did not produce the requested accepted seeds')
    data = args.out / 'data/episodes' / args.split / shard / suffix / 'data'
    for index in range(args.episodes):
        path = data / f'episode_{index:07d}.hdf5'
        with h5py.File(path, 'r') as episode:
            shape = episode['lewm/action_blocks'].shape
            if int(episode.attrs.get('lewm_visual_stride', -1)) != 1 or len(shape) != 3 or shape[1:] != (1, 14) or not shape[0]:
                raise ValueError(f'Incomplete stride-1 episode: {path}')
            for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                if len(episode[f'vision/{camera}/colors']) != shape[0]:
                    raise ValueError(f'Camera/action length mismatch: {path}')
    print(json.dumps({'complete': True, 'episodes': args.episodes, 'split': args.split,
                      'shard': args.shard, 'visual_stride': 1, 'data': str(data)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--curobo-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.shard < 0 or args.episodes <= 0:
        parser.error('Use a non-negative shard and positive episode count')
    args.robotwin_root, args.curobo_root, args.out = (p.resolve() for p in
                                                   (args.robotwin_root, args.curobo_root, args.out))
    if not args.dry_run:
        verify_curobo(args.curobo_root)
        registry = json.loads((ROOT / 'runtime/robotwin/collection_files.json').read_text())
        install(registry, args.robotwin_root, project_root=ROOT)
        link_curobo_assets(args.curobo_root, args.robotwin_root)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = os.pathsep.join([str(args.curobo_root / 'src'), str(args.robotwin_root),
                                       str(args.robotwin_root / 'scripts'), env.get('PYTHONPATH', '')])
    for command in commands(args):
        print(json.dumps(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=args.robotwin_root, env=env, check=True)
    if not args.dry_run:
        verify_collection(args)


if __name__ == '__main__':
    main()
