"""Run the released raw-stride-1 A2B training recipe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from sage.assets import safe_relative
from sage.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('prepare', 'a', 'cache', 'b', 'generator', 'prior', 'export')


def prepare(args):
    for split, path in (('train', args.train_manifest), ('validation', args.validation_manifest)):
        manifest = json.loads(path.read_text())
        rows = manifest['episodes'][:2] if args.smoke else manifest['episodes']
        if not rows:
            raise ValueError('Empty training split')
        for row in rows:
            source = (args.data_root / safe_relative(row['source'])).resolve()
            if not source.is_relative_to(args.data_root.resolve()) or not source.is_file():
                raise FileNotFoundError(source)
        local = dict(manifest, source_root=str(args.data_root.resolve()), episodes=rows)
        target = args.out / 'manifests' / f'{split}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and json.loads(target.read_text()) != local:
            raise ValueError(f'Existing manifest differs: {target}; use a fresh output directory')
        target.write_text(json.dumps(local, indent=2) + '\n')


def commands(args, stage):
    out = args.out.resolve()
    manifests = out / 'manifests'
    frames = out / 'frame_cache' / 'a2'
    cache = out / 'cache'
    epoch = 1 if args.smoke else 2
    a = out / 'checkpoints/robotwin_a2' / f'weights_epoch_{epoch}.pt'
    if stage in ('a', 'b'):
        batch = 2 if args.smoke else (96 if stage == 'a' else 288)
        if batch % len(args.gpus):
            raise ValueError('GPU count must divide the recipe global batch')
        command = [sys.executable, '-m', 'sage.train.robotwin_lewm',
                   '--config-name', 'robotwin_a2' if stage == 'a' else 'robotwin_b2',
                   f'trainer.devices={list(range(len(args.gpus)))}',
                   f'loader.batch_size={batch // len(args.gpus)}',
                   f'loader.num_workers={args.workers}',
                   f'loader.persistent_workers={str(args.workers > 0).lower()}',
                   'loader.prefetch_factor=2' if args.workers else 'loader.prefetch_factor=null']
        if stage == 'b':
            command.append(f'init_checkpoint={a.as_posix()}')
        if len(args.gpus) == 1:
            command.append('trainer.strategy=auto')
        if args.smoke:
            command += ['trainer.max_epochs=1', '+trainer.limit_train_batches=1',
                        '+trainer.limit_val_batches=1', 'trainer.precision=32-true']
        yield command
    elif stage == 'cache':
        for split in ('train', 'validation'):
            yield [sys.executable, '-m', 'sage.train.robotwin_cache',
                   '--manifest', str(manifests / f'{split}.json'), '--data-root', str(args.data_root.resolve()),
                   '--world-model', str(a), '--out', str(frames / f'{split}.npy'),
                   '--batch-size', '16' if args.smoke else '48', '--workers', str(args.workers)]
            command = [sys.executable, '-m', 'sage.train.robotwin_pack',
                       '--manifest', str(manifests / f'{split}.json'), '--data-root', str(args.data_root.resolve()),
                       '--flat-latents', str(frames / f'{split}.npy'), '--out-dir', str(cache / split),
                       '--action-stats', str(cache / 'action_stats.npz')]
            if split == 'train':
                command.append('--fit-action-stats')
            yield command
    elif stage in ('generator', 'prior'):
        command = [sys.executable, '-m', 'sage.train.robotwin_terminal', '--stage', stage,
                   '--train-cache', str(cache / 'train'), '--val-cache', str(cache / 'validation'),
                   '--out-dir', str(out / stage), '--workers', str(args.workers)]
        if stage == 'prior':
            command += ['--generator-checkpoint', str(out / 'generator/epoch1.pt')]
        if args.smoke:
            command += ['--epochs', '1', '--batch-size', '2', '--max-train-batches', '1',
                        '--max-val-batches', '1']
        elif not args.new_task:
            command += ['--recipe', str(ROOT / 'configs/robotwin_training.json')]
        yield command


def export_bundle(args):
    out = args.out.resolve()
    epoch = 1 if args.smoke else 2
    source = out / 'checkpoints/robotwin_b2'
    files = {'world_model/weights.pt': source / f'weights_epoch_{epoch}.pt',
             'world_model/config.json': source / 'config.json',
             'generator.pt': out / 'generator/epoch1.pt',
             'prior.pt': out / 'prior/latest.pt',
             'action_stats.npz': out / 'cache/action_stats.npz'}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        target = out / 'bundle' / name
        if target.exists() and sha256_file(target) != sha256_file(path):
            raise ValueError(f'Conflicting export: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-manifest', type=Path, default=ROOT / 'data/training/robotwin_a2b/train.json')
    parser.add_argument('--validation-manifest', type=Path, default=ROOT / 'data/training/robotwin_a2b/validation.json')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--new-task', action='store_true', help='Use supplied manifests without A2B window checks')
    parser.add_argument('--smoke', action='store_true', help='Two episodes/split and one batch/epoch; not the full recipe')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.gpus = [int(x) for x in args.gpus.split(',')]
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or args.workers < 0:
        parser.error('Provide distinct GPU indices and non-negative workers')
    if args.smoke and len(args.gpus) != 1:
        parser.error('Smoke mode uses one GPU')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)),
               ROBOTWIN_STRIDE1_LEWM_DATA=str((args.out / 'manifests').resolve()),
               ROBOTWIN_STRIDE1_LEWM_ROOT=str(args.out.resolve()),
               STABLEWM_HOME=str(args.out.resolve()))
    for stage in args.stages:
        if stage in ('prepare', 'export'):
            print(json.dumps({'stage': stage, 'out': str(args.out.resolve())}), flush=True)
            if not args.dry_run:
                prepare(args) if stage == 'prepare' else export_bundle(args)
            continue
        for command in commands(args, stage):
            print(json.dumps(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
