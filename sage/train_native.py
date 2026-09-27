"""Train the paper LIBERO components from frozen multi-view LeWM latents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    recipe = json.loads((ROOT / 'configs/native_training.json').read_text())[args.suite]
    cache = args.out / 'cache'
    common = ['--train-dataset', str(args.train), '--val-dataset', str(args.validation),
        '--train-latent-cache', str(cache / 'train.npy'),
        '--val-latent-cache', str(cache / 'validation.npy'), '--device', args.device]
    if args.stage == 'cache':
        for split, dataset in [('train', args.train), ('validation', args.validation)]:
            base = [sys.executable, '-m', 'sage.train.libero_cache',
                '--dataset', str(dataset), '--lewm-checkpoint', str(args.world_model),
                '--output', str(cache / f'{split}.npy'), '--device', args.device,
                '--latent-dim', '768', '--image-size', '112']
            yield base + ['--initialize']
            yield base + ['--worker-name', 'single']
            yield base + ['--finalize-workers', 'single']
        return
    spec = recipe[args.stage]
    from sage.train.libero_generator import build_terminal_staircase_specs
    for split, dataset in [('train', args.train), ('validation', args.validation)]:
        windows = build_terminal_staircase_specs(dataset, 15, None,
            include_direct_final=args.stage == 'prior', allowed_local_action_modes=[0, 1])
        if len(windows) != spec[f'{split}_windows']:
            raise ValueError(f'{split} has {len(windows)} windows; paper recipe expects {spec[f"{split}_windows"]}')
        actual = hashlib.sha256(windows.tobytes()).hexdigest()
        expected = recipe['window_sha256'][args.stage][split]
        if actual != expected:
            raise ValueError(f'{split} window identities/order differ from the paper recipe')
    command = [sys.executable, '-m', f'sage.train.libero_{args.stage}', *common,
        '--out-dir', str(args.out / args.stage), '--terminal-staircase',
        '--subgoal-offsets', '15', '--max-train-pairs', '0', '--max-val-pairs', '0',
        '--lr', '0.0001', '--weight-decay', '0.0001', '--grad-clip', '1.0',
        '--num-workers', str(args.workers)]
    for name in ['hidden_dim', 'num_heads', 'depth', 'epochs', 'batch_size', 'seed']:
        command += ['--' + name.replace('_', '-'), str(spec[name])]
    if args.stage == 'generator':
        command += ['--lewm-checkpoint', str(args.world_model),
            '--goal-offsets', *map(str, range(30, 256, 15)),
            '--predict-residual-from', 'current', '--cosine-weight', '0.1',
            '--smooth-l1-beta', '0.05', '--allowed-local-action-modes', '0', '1']
    else:
        generator = args.generator or args.out / 'generator/best.pt'
        command += ['--generator-checkpoint', str(generator),
            '--goal-offsets', *map(str, range(15, 256, 15)),
            '--allowed-action-modes', '0', '1', '--num-modes', '8',
            '--mode-l1-weight', '0.05', '--eval-samples', '128', '--bf16',
            '--save-epochs', '4']
    yield command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['libero_scene2', 'libero_caddy'], required=True)
    parser.add_argument('--stage', choices=['cache', 'generator', 'prior'], required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--world-model', type=Path, required=True)
    parser.add_argument('--generator', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for command in commands(args):
        print(json.dumps(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
